# -*- coding: utf-8 -*-
from __future__ import annotations
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
from itertools import combinations
import random
from torch.utils.data import DataLoader, Subset

from config import ProjectConfig
from data import make_loaders as make_loaders_data
from model import FNN_CMR_MoE
from losses import (
    smoe_balance_loss,
    similarity_loss_cka,
    pairwise_cka_matrix,
    expert_first_layer_orth_loss,
    expert_layer_orth_loss,
    expert_lora_A_orth_loss,
    expert_lora_B_orth_loss,
    shared_moe_layer_orth_loss,
    l1_sparsity_loss,
    group_lasso_loss,
)
from metrics import (
    cross_entropy, confusion_matrix, per_class_accuracy, macro_f1_from_cm,
    expert_usage_entropy, expert_load_gini, mutual_info_expert_regime,
    accumulate_label_expert_counts, label_expert_frequencies,
    accumulate_regime_expert_counts, regime_expert_frequencies,
)
from utils import now_str, set_seed, setup_cpu, save_json, ensure_dir, to_device
from plots import save_label_expert_heatmap, save_regime_expert_heatmap, save_cka_heatmap


def _make_adamw(cfg: ProjectConfig, params, lr: Optional[float] = None) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        params,
        lr=float(cfg.train.lr if lr is None else lr),
        weight_decay=float(cfg.train.weight_decay),
        betas=(float(cfg.train.adamw_beta1), float(cfg.train.adamw_beta2)),
        eps=float(cfg.train.adamw_eps),
    )


def _save_state_dict(model: torch.nn.Module, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state, out_path)


def _compute_total_loss(cfg: ProjectConfig, model: torch.nn.Module, logits: torch.Tensor, y: torch.Tensor, aux: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
    parts: Dict[str, torch.Tensor] = {}

    if cfg.data_variant in ("data2", "data3"):
        if y.ndim == 1:
            y = y.view(-1, 1)
        parts["task"] = F.mse_loss(logits, y)
    else:
        parts["task"] = F.cross_entropy(logits, y)

    # CMR budget loss removed
    parts["cmr"] = torch.tensor(0.0, device=logits.device)

    moe_aux = aux.get("moe_aux") or {}
    has_moe = "dispatch_mask" in moe_aux

    # SMoE balance loss
    if cfg.model.lb_enabled and has_moe:
        parts["lb"] = smoe_balance_loss(
            router_scores=moe_aux["router_scores"],
            dispatch_mask=moe_aux["dispatch_mask"].detach(),
            tau0=cfg.model.lb_tau0,
        ) * cfg.model.lb_lambda
    else:
        parts["lb"] = torch.tensor(0.0, device=logits.device)

    # similarity loss (CKA)
    if cfg.model.sim_enabled and has_moe and cfg.model.topk >= 2 and cfg.model.routing_mode == "row_topk":
        parts["sim"] = similarity_loss_cka(
            moe_aux["expert_feats"],
            dispatch_mask=moe_aux.get("dispatch_mask"),
            trunk_hidden=aux.get("trunk_hidden"),
            expert_modules=getattr(model, "moe", None).experts if getattr(model, "moe", None) is not None else None,
            router_scores=moe_aux.get("router_scores"),
            topk=cfg.model.topk,
            proj_module=getattr(model, "moe", None).sim_proj if getattr(model, "moe", None) is not None else None,
            expert_feat_indices=moe_aux.get("expert_feat_indices"),
            f_star=int(cfg.model.sim_f_star),
            t_star=float(cfg.model.sim_t_star),
            kernel=str(cfg.model.sim_kernel),
            sigma=float(cfg.model.sim_sigma),
        ) * cfg.model.sim_lambda
    else:
        parts["sim"] = torch.tensor(0.0, device=logits.device)

    if cfg.model.expert_orth_lambda > 0 and getattr(model, "moe", None) is not None:
        if cfg.model.expert_type == "lora":
            parts["orth"] = expert_lora_A_orth_loss(model.moe.experts) * cfg.model.expert_orth_lambda
        else:
            parts["orth"] = expert_first_layer_orth_loss(model.moe.experts) * cfg.model.expert_orth_lambda
    else:
        parts["orth"] = torch.tensor(0.0, device=logits.device)
    if cfg.model.expert_second_layer_orth_lambda > 0 and getattr(model, "moe", None) is not None:
        if cfg.model.expert_type == "lora":
            parts["orth2"] = expert_lora_B_orth_loss(model.moe.experts) * cfg.model.expert_second_layer_orth_lambda
        else:
            parts["orth2"] = expert_layer_orth_loss(model.moe.experts, linear_index=1) * cfg.model.expert_second_layer_orth_lambda
    else:
        parts["orth2"] = torch.tensor(0.0, device=logits.device)

    # sparsity (experts vs shared separate)
    if cfg.model.expert_l1_lambda > 0 and getattr(model, "moe", None) is not None:
        parts["expert_l1"] = l1_sparsity_loss(model.moe.experts) * cfg.model.expert_l1_lambda
    else:
        parts["expert_l1"] = torch.tensor(0.0, device=logits.device)
    if cfg.model.expert_group_lasso_lambda > 0 and getattr(model, "moe", None) is not None:
        parts["expert_group"] = group_lasso_loss(model.moe.experts) * cfg.model.expert_group_lasso_lambda
    else:
        parts["expert_group"] = torch.tensor(0.0, device=logits.device)

    shared_mods = []
    if getattr(model, "shared2", None) is not None:
        shared_mods = list(model.shared2)
    elif getattr(model, "shared2_big", None) is not None:
        shared_mods = [model.shared2_big]
    if cfg.model.shared_l1_lambda > 0 and shared_mods:
        parts["shared_l1"] = l1_sparsity_loss(shared_mods) * cfg.model.shared_l1_lambda
    else:
        parts["shared_l1"] = torch.tensor(0.0, device=logits.device)
    if cfg.model.shared_group_lasso_lambda > 0 and shared_mods:
        parts["shared_group"] = group_lasso_loss(shared_mods) * cfg.model.shared_group_lasso_lambda
    else:
        parts["shared_group"] = torch.tensor(0.0, device=logits.device)
    if cfg.model.shared_moe_first_layer_orth_lambda > 0 and shared_mods and getattr(model, "moe", None) is not None:
        parts["shared_moe_orth1"] = (
            shared_moe_layer_orth_loss(shared_mods, model.moe.experts, linear_index=0)
            * cfg.model.shared_moe_first_layer_orth_lambda
        )
    else:
        parts["shared_moe_orth1"] = torch.tensor(0.0, device=logits.device)
    if cfg.model.shared_moe_second_layer_orth_lambda > 0 and shared_mods and getattr(model, "moe", None) is not None:
        parts["shared_moe_orth2"] = (
            shared_moe_layer_orth_loss(shared_mods, model.moe.experts, linear_index=1)
            * cfg.model.shared_moe_second_layer_orth_lambda
        )
    else:
        parts["shared_moe_orth2"] = torch.tensor(0.0, device=logits.device)

    total = sum(parts.values())
    parts_f = {k: float(v.detach().cpu().item()) for k, v in parts.items()}
    parts_f["total"] = float(total.detach().cpu().item())
    return total, parts_f



@torch.no_grad()
def _topk_pairs_from_counts(counts: torch.Tensor, topk: int) -> List[Tuple[int, int]]:
    pairs = []
    E = counts.shape[0]
    for i in range(E):
        for j in range(i + 1, E):
            pairs.append((counts[i, j].item(), i, j))
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))
    return [(i, j) for _, i, j in pairs[:topk]]


@torch.no_grad()
def _count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _expert_param_count(model: torch.nn.Module) -> int:
    if getattr(model, "moe", None) is None:
        return 0
    return sum(p.numel() for p in model.moe.experts.parameters())


def _effective_param_count(total_params: int, expert_params: int, active_ratio: float) -> int:
    base = total_params - expert_params
    return int(round(base + expert_params * active_ratio))


def _linear_flops(batch: int, in_dim: int, out_dim: int) -> int:
    if batch <= 0:
        return 0
    return int(2 * batch * in_dim * out_dim)


def _progressive_hidden_dims(base_hidden: int, depth: int) -> List[int]:
    d = int(depth)
    if d <= 1:
        return []
    h0 = max(1, int(base_hidden))
    return [h0 * (2 ** i) for i in range(d - 1)]


def _mlp_flops_by_dims(batch: int, dims: List[int]) -> int:
    if batch <= 0:
        return 0
    if len(dims) < 2:
        return 0
    flops = 0
    for i in range(len(dims) - 1):
        flops += _linear_flops(batch, int(dims[i]), int(dims[i + 1]))
    return flops


def _low_rank_linear_flops(batch: int, in_dim: int, out_dim: int, rank: int) -> int:
    if batch <= 0:
        return 0
    r = max(1, int(rank))
    return int(2 * batch * (in_dim * r + r * out_dim))


def _expert_mlp_flops(cfg: ProjectConfig, batch: int) -> int:
    depth = int(cfg.model.expert_mlp_depth)
    in_dim = int(cfg.model.input_dim)
    out_dim = int(cfg.model.expert_feature_dim)
    hidden_dims = _progressive_hidden_dims(cfg.model.expert_hidden_dim, depth)
    layer_dims = [in_dim] + hidden_dims + [out_dim]

    if cfg.model.expert_type == "lora":
        rank = int(cfg.model.expert_lora_rank)
        flops = 0
        for i in range(len(layer_dims) - 1):
            flops += _low_rank_linear_flops(batch, layer_dims[i], layer_dims[i + 1], rank)
        return flops

    return _mlp_flops_by_dims(batch, layer_dims)


def _moe_gate_flops(cfg: ProjectConfig, batch: int) -> int:
    in_dim = cfg.model.input_dim
    E = cfg.model.num_experts
    if cfg.model.gate_type == "linear":
        return _linear_flops(batch, in_dim, E)
    if cfg.model.gate_type == "nonlinear":
        h = int(cfg.model.gate_hidden)
        return _linear_flops(batch, in_dim, h) + _linear_flops(batch, h, E)
    return 0


def _shared2_flops(cfg: ProjectConfig, batch: int) -> int:
    hidden_dims = _progressive_hidden_dims(cfg.model.shared2_hidden_dim, cfg.model.shared2_mlp_depth)
    layer_dims = [int(cfg.model.input_dim)] + hidden_dims + [int(cfg.model.shared2_out_dim)]
    base = _mlp_flops_by_dims(batch, layer_dims)
    num_shared = max(1, int(cfg.model.num_shared_experts))
    return base * num_shared


def _forward_flops_batch(cfg: ProjectConfig, model: torch.nn.Module, aux: Dict, batch: int) -> int:
    if batch <= 0:
        return 0

    in_dim = cfg.model.input_dim
    D = cfg.model.expert_feature_dim
    C = cfg.model.num_classes

    # big-MLP baseline (no MoE)
    if cfg.model.gate1_mode == "constant_bigmlp":
        H = int(cfg.model.bigmlp_hidden)
        flops = _linear_flops(batch, in_dim, H)
        flops += _linear_flops(batch, H, D)
        flops += _linear_flops(batch, D, C)
        return flops

    flops = 0

    # gate1 (CMR) linear gate
    if cfg.model.gate1_fuse_mode == "gate":
        flops += _linear_flops(batch, in_dim, 1)

    # shared experts
    if cfg.model.gate1_constant > 0.0 and getattr(model, "shared2", None) is not None:
        flops += _shared2_flops(cfg, batch)

    # MoE routing + experts
    moe_aux = aux.get("moe_aux") or {}
    dm = moe_aux.get("dispatch_mask")
    if getattr(model, "moe", None) is not None and dm is not None:
        if not bool(moe_aux.get("forced_expert", False)):
            flops += _moe_gate_flops(cfg, batch)
        counts = dm.sum(dim=0)
        for e in range(int(cfg.model.num_experts)):
            n_e = int(counts[e].item())
            flops += _expert_mlp_flops(cfg, n_e)

    # classifier head
    flops += _linear_flops(batch, D, C)
    return flops


@torch.no_grad()
def _eval_loss_only(cfg: ProjectConfig, model: torch.nn.Module, loader, device: str) -> float:
    model.eval()
    losses = []
    for x, y, _ in loader:
        x, y = to_device(x, device), to_device(y, device)
        logits, _ = model(x)
        if cfg.data_variant in ("data2", "data3"):
            y_f = y.float()
            if y_f.ndim == 1:
                y_f = y_f.view(-1, 1)
            diff = logits - y_f
            losses.append(float((diff * diff).mean().item()))
        else:
            losses.append(F.cross_entropy(logits, y).item())
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def _predict_probs(model: torch.nn.Module, loader, device: str, cfg: ProjectConfig | None = None, split: str = "test") -> np.ndarray:
    model.eval()
    probs = []
    for x, _, regime in loader:
        x = to_device(x, device)
        regime = to_device(regime, device)
        if cfg is None:
            logits, _ = model(x)
        else:
            forced_expert = None
            if split == "test" and cfg.train.test_routing == "fixed":
                forced_expert = regime
            shared_only = split == "test" and cfg.train.test_expert_mode == "only_share"
            moe_only = split == "test" and cfg.train.test_expert_mode in ("only_moe", "only_moe_top1")
            force_top1 = split == "test" and cfg.train.test_expert_mode == "only_moe_top1"
            logits, _ = model(
                x,
                forced_expert=forced_expert,
                shared_only=shared_only,
                moe_only=moe_only,
                force_top1=force_top1,
                force_avg_fuse=(split == "test" and cfg.train.test_routing == "fixed"),
            )
        p = F.softmax(logits, dim=-1)
        probs.append(p.detach().cpu().numpy())
    if not probs:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(probs, axis=0)


@torch.no_grad()
def _predict_values(model: torch.nn.Module, loader, device: str, cfg: ProjectConfig | None = None, split: str = "test") -> np.ndarray:
    model.eval()
    preds = []
    for x, _, regime in loader:
        x = to_device(x, device)
        regime = to_device(regime, device)
        if cfg is None:
            logits, _ = model(x)
        else:
            forced_expert = None
            if split == "test" and cfg.train.test_routing == "fixed":
                forced_expert = regime
            shared_only = split == "test" and cfg.train.test_expert_mode == "only_share"
            moe_only = split == "test" and cfg.train.test_expert_mode in ("only_moe", "only_moe_top1")
            force_top1 = split == "test" and cfg.train.test_expert_mode == "only_moe_top1"
            logits, _ = model(
                x,
                forced_expert=forced_expert,
                shared_only=shared_only,
                moe_only=moe_only,
                force_top1=force_top1,
                force_avg_fuse=(split == "test" and cfg.train.test_routing == "fixed"),
            )
        preds.append(logits.detach().cpu().numpy())
    if not preds:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(preds, axis=0)


def _gate_cosine(model: torch.nn.Module) -> float:
    if getattr(model, "moe", None) is None:
        return 0.0
    gate = model.moe.gate
    if hasattr(gate, "weight"):
        W = gate.weight.detach().cpu().numpy()
    else:
        W = None
        for mod in reversed(list(gate.modules())):
            if isinstance(mod, torch.nn.Linear) and mod.out_features == model.moe.E:
                W = mod.weight.detach().cpu().numpy()
                break
        if W is None:
            return 0.0
    E = W.shape[0]
    if E <= 1:
        return 0.0
    Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
    sim = Wn @ Wn.T
    offdiag = sim[~np.eye(E, dtype=bool)]
    return float(offdiag.mean()) if offdiag.size else 0.0


def _expert_weight_cosine(model: torch.nn.Module) -> float:
    def _kth_linear_like_weight(mod: torch.nn.Module, k: int):
        idx = 0
        for m in mod.modules():
            if isinstance(m, torch.nn.Linear):
                w = m.weight
            elif hasattr(m, "effective_weight") and callable(getattr(m, "effective_weight")):
                w = m.effective_weight()
                if not torch.is_tensor(w):
                    w = None
            else:
                w = None
            if w is not None:
                if idx == k:
                    return w
                idx += 1
        return None

    if getattr(model, "moe", None) is None:
        return 0.0
    experts = model.moe.experts
    weights = []
    for exp in experts:
        W = _kth_linear_like_weight(exp, 0)
        if W is not None:
            weights.append(W.detach().cpu().flatten().numpy())
    E = len(weights)
    if E < 2:
        return 0.0
    W = np.stack(weights, axis=0)
    W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
    sim = W @ W.T
    offdiag = sim[~np.eye(E, dtype=bool)]
    return float(offdiag.mean()) if offdiag.size else 0.0


@torch.no_grad()
def _prune_moe_experts_by_norm(cfg: ProjectConfig, model: torch.nn.Module) -> Dict[str, float]:
    if not cfg.model.prune_experts_enabled or getattr(model, "moe", None) is None:
        return {"prune_enabled": 0.0, "pruned_experts": 0.0, "kept_experts": 0.0}
    E = model.moe.E
    if E <= 0:
        return {"prune_enabled": 1.0, "pruned_experts": 0.0, "kept_experts": 0.0}
    norms = []
    for exp in model.moe.experts:
        total = 0.0
        for p in exp.parameters():
            total += float((p.detach() ** 2).sum().item())
        norms.append(total ** 0.5)
    norms_t = torch.tensor(norms)
    keep = norms_t >= float(cfg.model.prune_experts_norm_threshold)
    min_keep = max(1, int(cfg.model.prune_experts_min_keep))
    if int(keep.sum().item()) < min_keep:
        topk = torch.topk(norms_t, k=min_keep).indices
        keep = torch.zeros_like(keep)
        keep[topk] = True
    model.moe.pruned_mask = keep.to(torch.bool)
    return {
        "prune_enabled": 1.0,
        "pruned_experts": float((~keep).sum().item()),
        "kept_experts": float(keep.sum().item()),
    }


def _finetune_after_prune(
    cfg: ProjectConfig,
    model: torch.nn.Module,
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
) -> Dict[str, List[float]]:
    curves = {"train_losses": [], "val_losses": [], "test_losses": []}
    if not cfg.model.prune_finetune_enabled:
        return curves
    epochs = max(1, int(cfg.model.prune_finetune_epochs))
    lr = float(cfg.train.lr) * float(cfg.model.prune_finetune_lr_scale)
    opt = _make_adamw(cfg, model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        epoch_losses = []
        for x, y, _ in train_loader:
            x, y = to_device(x, device), to_device(y, device)
            opt.zero_grad(set_to_none=True)
            logits, aux = model(x)
            total, parts = _compute_total_loss(cfg, model, logits, y, aux)
            total.backward()
            if cfg.train.grad_clip and cfg.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            epoch_losses.append(parts["total"])
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        curves["train_losses"].append(train_loss)

        valm = _evaluate_split(cfg, model, val_loader, device, split="val", compute_sim_metrics=False)
        testm = _evaluate_split(cfg, model, test_loader, device, split="test", compute_sim_metrics=False)
        val_loss = float(valm["loss"])
        test_loss = float(testm["loss"])
        curves["val_losses"].append(val_loss)
        curves["test_losses"].append(test_loss)
        print(
            f"[{now_str()}] finetune epoch={ep} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} test_loss={test_loss:.6f}"
        )
    return curves


def _pes_from_expert_outputs(expert_outs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """expert_outs: [B,E,D], returns per-sample PES [B]."""
    if expert_outs.dim() != 3:
        return torch.zeros((expert_outs.shape[0],), device=expert_outs.device)
    B, E, _ = expert_outs.shape
    if E < 2 or B == 0:
        return torch.zeros((B,), device=expert_outs.device)
    v = expert_outs / (expert_outs.norm(dim=-1, keepdim=True) + eps)
    cos = torch.matmul(v, v.transpose(1, 2))  # [B,E,E]
    diag = cos.diagonal(dim1=1, dim2=2).sum(dim=1)
    off_sum = cos.sum(dim=(1, 2)) - diag
    return off_sum / (E * (E - 1))


def _choose_subsets(E: int, k: int, max_subsets: int, seed: int) -> List[Tuple[int, ...]]:
    if k <= 0 or E <= k:
        return [tuple(range(E))]
    all_subsets = list(combinations(range(E), k))
    if max_subsets and max_subsets > 0 and len(all_subsets) > max_subsets:
        rng = random.Random(seed)
        return rng.sample(all_subsets, k=max_subsets)
    return all_subsets


def _cka_offdiag_mean_over_subsets(cka_mat: torch.Tensor, subsets: List[Tuple[int, ...]]) -> float:
    if cka_mat.numel() == 0:
        return 0.0
    E = int(cka_mat.shape[0])
    if E <= 1:
        return 0.0
    vals = []
    for sub in subsets:
        if len(sub) <= 1:
            continue
        idx = torch.tensor(sub, device=cka_mat.device, dtype=torch.long)
        M = cka_mat.index_select(0, idx).index_select(1, idx)
        k = int(M.shape[0])
        off = M[~torch.eye(k, dtype=torch.bool, device=cka_mat.device)]
        off = off[torch.isfinite(off)]
        if off.numel() > 0:
            vals.append(off.mean())
    if not vals:
        return 0.0
    return float(torch.stack(vals).mean().detach().cpu().item())


def _subset_pes_from_expert_outputs(
    expert_outs: torch.Tensor,
    subsets: List[Tuple[int, ...]],
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return per-sample PES averaged over the given expert subsets."""
    if expert_outs.dim() != 3:
        return torch.zeros((expert_outs.shape[0],), device=expert_outs.device)
    B, E, _ = expert_outs.shape
    if B == 0 or E < 2:
        return torch.zeros((B,), device=expert_outs.device)

    v = expert_outs / (expert_outs.norm(dim=-1, keepdim=True) + eps)
    cos = torch.matmul(v, v.transpose(1, 2))  # [B,E,E]
    per_subset = []
    for sub in subsets:
        k = len(sub)
        if k < 2:
            continue
        idx = torch.tensor(sub, device=expert_outs.device, dtype=torch.long)
        C = cos.index_select(1, idx).index_select(2, idx)  # [B,k,k]
        diag = C.diagonal(dim1=1, dim2=2).sum(dim=1)
        off_sum = C.sum(dim=(1, 2)) - diag
        per_subset.append(off_sum / (k * (k - 1)))
    if not per_subset:
        return torch.zeros((B,), device=expert_outs.device)
    return torch.stack(per_subset, dim=0).mean(dim=0)


def _evaluate_split(
    cfg: ProjectConfig,
    model: torch.nn.Module,
    loader,
    device: str,
    split: str,
    out_dir: Optional[Path] = None,
    file_prefix: str = "",
    compute_flops: bool = False,
    compute_sim_metrics: bool = False,
) -> Dict:
    model.eval()
    if cfg.train.test_routing not in ("score", "fixed"):
        raise ValueError(f"Unsupported test_routing={cfg.train.test_routing!r}; expected 'score' or 'fixed'.")

    C = cfg.model.num_classes
    E = cfg.model.num_experts
    R = cfg.model.num_regimes
    is_regression = cfg.data_variant in ("data2", "data3")

    losses = []
    ents = []
    ginis = []
    mis = []
    active_counts = []
    total_samples = 0
    total_steps = 0
    t0 = time.time()
    flops_total = 0.0
    pes_sum = 0.0
    pes_count = 0
    sse = 0.0
    sae = 0.0
    sum_y = 0.0
    sum_y2 = 0.0
    count_y = 0

    # routing freq
    label_counts = torch.zeros((C, E), device=device)
    label_totals = torch.zeros((C,), device=device)
    regime_counts = torch.zeros((R, E), device=device)
    regime_totals = torch.zeros((R,), device=device)

    # gather for confusion matrix (classification only)
    all_pred = []
    all_true = []

    # collect features for CKA (bounded)
    feat_bank: Dict[int, torch.Tensor] = {}
    max_tokens_per_expert = 2048
    subsets = _choose_subsets(
        E=E,
        k=int(cfg.model.sim_metric_subset_k),
        max_subsets=int(cfg.model.sim_metric_subset_max_subsets),
        seed=int(cfg.model.sim_metric_subset_seed),
    )

    if split == "test" and cfg.train.test_routing == "fixed" and E < R:
        raise ValueError("test_routing=fixed requires num_experts >= num_regimes")

    for x, y, regime in loader:
        total_steps += 1
        total_samples += int(x.shape[0])
        x, y, regime = to_device(x, device), to_device(y, device), to_device(regime, device)

        forced_expert = None
        if split == "test" and cfg.train.test_routing == "fixed":
            forced_expert = regime
        shared_only = split == "test" and cfg.train.test_expert_mode == "only_share"
        moe_only = split == "test" and cfg.train.test_expert_mode in ("only_moe", "only_moe_top1")
        force_top1 = split == "test" and cfg.train.test_expert_mode == "only_moe_top1"
        logits, aux = model(
            x,
            forced_expert=forced_expert,
            shared_only=shared_only,
            moe_only=moe_only,
            force_top1=force_top1,
            force_avg_fuse=(split == "test" and cfg.train.test_routing == "fixed"),
        )
        if compute_flops:
            flops_total += float(_forward_flops_batch(cfg, model, aux, int(x.shape[0])))
        if is_regression:
            y_f = y.float()
            if y_f.ndim == 1:
                y_f = y_f.view(-1, 1)
            diff = logits - y_f
            sse += float((diff * diff).sum().item())
            sae += float(diff.abs().sum().item())
            sum_y += float(y_f.sum().item())
            sum_y2 += float((y_f * y_f).sum().item())
            count_y += int(y_f.numel())
            losses.append(float((diff * diff).mean().item()))
        else:
            losses.append(F.cross_entropy(logits, y).item())

        moe_aux = aux.get("moe_aux") or {}
        dm = moe_aux.get("dispatch_mask")
        if dm is not None:
            ents.append(expert_usage_entropy(dm))
            ginis.append(expert_load_gini(dm))
            mis.append(mutual_info_expert_regime(dm, regime))
            active_counts.append(dm.float().sum(dim=1).mean().item())

            if not is_regression:
                label_counts, label_totals = accumulate_label_expert_counts(label_counts, label_totals, y, dm, num_labels=C)
            regime_counts, regime_totals = accumulate_regime_expert_counts(regime_counts, regime_totals, regime, dm, num_regimes=R)

        if not is_regression:
            pred = logits.argmax(dim=-1)
            all_pred.append(pred.detach().cpu())
            all_true.append(y.detach().cpu())

        if compute_sim_metrics:
            trunk_hidden = aux.get("trunk_hidden")
            if trunk_hidden is not None and getattr(model, "moe", None) is not None:
                expert_outs = torch.stack([model.moe.experts[e](trunk_hidden) for e in range(E)], dim=1)
                pes_vals = _subset_pes_from_expert_outputs(expert_outs, subsets=subsets)
                pes_sum += float(pes_vals.sum().detach().cpu().item())
                pes_count += int(trunk_hidden.shape[0])

                for e in range(E):
                    fe = expert_outs[:, e].detach()
                    if e not in feat_bank:
                        feat_bank[e] = fe[:max_tokens_per_expert]
                    else:
                        cur = feat_bank[e]
                        if cur.shape[0] < max_tokens_per_expert:
                            need = max_tokens_per_expert - cur.shape[0]
                            feat_bank[e] = torch.cat([cur, fe[:need]], dim=0)

    if not is_regression:
        all_pred = torch.cat(all_pred, dim=0)
        all_true = torch.cat(all_true, dim=0)

        cm = confusion_matrix(all_pred, all_true, num_classes=C)
        pacc = per_class_accuracy(cm)
        mf1 = macro_f1_from_cm(cm)
        micro_acc = float(cm.diag().sum().item() / max(cm.sum().item(), 1))
    else:
        cm = torch.zeros((C, C), dtype=torch.long)
        pacc = torch.zeros((C,), dtype=torch.float32)
        mf1 = 0.0
        micro_acc = 0.0

    label_freq = torch.zeros((C, E), device=device)
    if not is_regression and label_totals.sum() > 0:
        label_freq = label_expert_frequencies(label_counts, label_totals)

    if regime_totals.sum() > 0:
        regime_freq = regime_expert_frequencies(regime_counts, regime_totals)
    else:
        regime_freq = torch.zeros((R, E), device=device)

    if compute_sim_metrics and feat_bank:
        cka_mat = pairwise_cka_matrix(feat_bank, num_experts=E)
    else:
        cka_mat = torch.zeros((E, E), device=device)

    dt = time.time() - t0
    samples_per_sec = float(total_samples / dt) if dt > 0 else 0.0
    avg_step_time = float(dt / max(total_steps, 1))
    active_ratio = float(np.mean(active_counts) / E) if active_counts else 0.0
    pes = float(pes_sum / pes_count) if pes_count > 0 else 0.0
    cka_offdiag = _cka_offdiag_mean_over_subsets(cka_mat, subsets=subsets) if compute_sim_metrics else 0.0

    metrics = {
        "split": split,
        "loss": float(np.mean(losses)),
        "acc": micro_acc,
        "macro_f1": float(mf1),
        "expert_cosine": _expert_weight_cosine(model),
        "per_class_acc": pacc.tolist(),
        "confusion_matrix": cm.tolist(),
        "expert_entropy": float(np.mean(ents)),
        "expert_gini": float(np.mean(ginis)),
        "mi_expert_regime": float(np.mean(mis)) if mis else 0.0,
        "cka_offdiag_mean": cka_offdiag,
        "expert_pes": pes,
        "samples_per_sec": samples_per_sec,
        "avg_step_time": avg_step_time,
        "active_expert_ratio": active_ratio,
        "label_expert_freq": label_freq.detach().cpu().tolist(),
        "regime_expert_freq": regime_freq.detach().cpu().tolist(),
        "cka_matrix": cka_mat.detach().cpu().tolist(),
    }
    if is_regression:
        mse = float(sse / max(count_y, 1))
        mae = float(sae / max(count_y, 1))
        mean_y = float(sum_y / max(count_y, 1))
        sst = float(sum_y2 - max(count_y, 1) * mean_y * mean_y)
        r2 = 1.0 - (sse / sst) if sst > 1e-12 else 0.0
        metrics["loss"] = mse
        metrics["mse"] = mse
        metrics["mae"] = mae
        metrics["r2"] = float(r2)
    if compute_flops:
        metrics["flops_total"] = float(flops_total)

    if out_dir is not None:
        ensure_dir(out_dir)
        save_label_expert_heatmap(label_freq, out_dir / f"{file_prefix}{split}_label_expert_heatmap.png", title_prefix=split)
        save_regime_expert_heatmap(regime_freq, out_dir / f"{file_prefix}{split}_regime_expert_heatmap.png", title_prefix=split)
        if compute_sim_metrics:
            save_cka_heatmap(cka_mat, out_dir / f"{file_prefix}{split}_cka_heatmap.png", title_prefix=split)

    return metrics


def train_one_rep(cfg: ProjectConfig, rep_idx: int, run_dir: Path, save_plots: bool = True) -> Dict:
    """Train/eval one repetition. Returns dict with:
      - train_losses (per epoch)
      - val_losses   (per epoch)
      - test_losses  (per epoch)
      - best_val_loss
      - test_metrics (dict)
      - val_metrics  (dict)
      - train_routing_epoch_stats (optional)
    """
    # ---- CPU setup & seeds ----
    if cfg.train.test_routing not in ("score", "fixed"):
        raise ValueError(f"Unsupported test_routing={cfg.train.test_routing!r}; expected 'score' or 'fixed'.")
    setup_cpu(cfg.train.num_threads)

    rep_seed = cfg.train.seed + rep_idx * cfg.train.rep_seed_offset
    set_seed(rep_seed)

    # data regeneration happens here (per rep)
    data_cfg = cfg.active_data()
    train_loader, val_loader, test_loader = make_loaders_data(
        data_cfg,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        seed=rep_seed,
        data_variant=cfg.data_variant,
    )

    device = "cpu"
    model = FNN_CMR_MoE(cfg.model).to(device)
    opt = _make_adamw(cfg, model.parameters())

    rep_prefix = f"rep_{rep_idx:03d}_"
    plots_dir = ensure_dir(run_dir / "plots") if save_plots else None
    rep_model_dir = ensure_dir(run_dir / "model" / f"rep{rep_idx + 1}")

    train_init_metrics = {"cka_offdiag_mean": 0.0, "expert_pes": 0.0}
    if cfg.train.compute_sim_metrics:
        initm = _evaluate_split(
            cfg,
            model,
            train_loader,
            device,
            split="train",
            compute_sim_metrics=True,
        )
        train_init_metrics = {
            "cka_offdiag_mean": float(initm.get("cka_offdiag_mean", 0.0)),
            "expert_pes": float(initm.get("expert_pes", 0.0)),
            "cka_matrix": initm.get("cka_matrix", []),
        }

    train_losses = []
    val_losses = []
    test_losses = []

    best_val = float("inf")
    best_epoch = -1
    best_state = None
    no_improve = 0

    step = 0
    total_train_samples = 0
    total_train_steps = 0
    total_train_time = 0.0
    train_active_counts = []
    train_flops_total = 0.0
    last_epoch_num = 0
    for epoch in range(cfg.train.epochs):
        t0 = time.time()
        model.train()

        epoch_losses = []
        epoch_part_sums: Dict[str, float] = {}
        epoch_part_count = 0
        t_train = time.time()
        for x, y, regime in train_loader:
            total_train_steps += 1
            total_train_samples += int(x.shape[0])
            x, y = to_device(x, device), to_device(y, device)

            opt.zero_grad(set_to_none=True)
            logits, aux = model(x)
            dm = (aux.get("moe_aux") or {}).get("dispatch_mask")
            if dm is not None:
                train_active_counts.append(dm.float().sum(dim=1).mean().item())
            fwd_flops = _forward_flops_batch(cfg, model, aux, int(x.shape[0]))
            # forward + backward ≈ forward * 3 for linear layers (dX + dW)
            train_flops_total += float(fwd_flops) * 3.0

            total, parts = _compute_total_loss(cfg, model, logits, y, aux)
            total.backward()
            if cfg.train.grad_clip and cfg.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()

            epoch_losses.append(parts["total"])
            for k, v in parts.items():
                epoch_part_sums[k] = epoch_part_sums.get(k, 0.0) + float(v)
            epoch_part_count += 1
            step += 1
        total_train_time += time.time() - t_train

        train_loss = float(np.mean(epoch_losses))
        epoch_part_means = {
            k: (v / max(epoch_part_count, 1))
            for k, v in epoch_part_sums.items()
        }
        train_losses.append(train_loss)

        # epoch-end val/test
        valm = _evaluate_split(cfg, model, val_loader, device, split="val", out_dir=None, file_prefix=rep_prefix, compute_sim_metrics=False)
        val_loss = float(valm["loss"])
        val_losses.append(val_loss)

        testm_epoch = _evaluate_split(cfg, model, test_loader, device, split="test", out_dir=None, file_prefix=rep_prefix, compute_sim_metrics=False)
        test_loss = float(testm_epoch["loss"])
        test_losses.append(test_loss)
        epoch_num = epoch + 1
        last_epoch_num = epoch_num

        if epoch_num == 1 or (epoch_num % 10 == 0):
            _save_state_dict(model, rep_model_dir / f"epoch_{epoch_num:03d}.pth")

        orth_total = (
            float(epoch_part_means.get("orth", 0.0))
            + float(epoch_part_means.get("orth2", 0.0))
            + float(epoch_part_means.get("shared_moe_orth1", 0.0))
            + float(epoch_part_means.get("shared_moe_orth2", 0.0))
        )
        sparsity_total = (
            float(epoch_part_means.get("expert_l1", 0.0))
            + float(epoch_part_means.get("shared_l1", 0.0))
            + float(epoch_part_means.get("expert_group", 0.0))
            + float(epoch_part_means.get("shared_group", 0.0))
        )
        task_name = "mse" if cfg.data_variant in ("data2", "data3") else "ce"
        part_items = [
            f"{task_name}={float(epoch_part_means.get('task', 0.0)):.6f}",
            f"lb={float(epoch_part_means.get('lb', 0.0)):.6f}",
            f"sim={float(epoch_part_means.get('sim', 0.0)):.6f}",
            f"orth_total={orth_total:.6f}",
            f"sparsity_total={sparsity_total:.6f}",
        ]
        part_str = ", ".join(part_items)

        dt = time.time() - t0
        print(
            f"[{now_str()}] rep={rep_idx} epoch={epoch} time={dt:.1f}s "
            f"train_loss={train_loss:.6f} ({part_str}) val_loss={val_loss:.6f} test_loss={test_loss:.6f}"
        )

        # early stop / best model
        improved = (best_val - val_loss) > cfg.train.early_stop_min_delta
        if improved:
            best_val = val_loss
            best_epoch = epoch
            no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            _save_state_dict(model, rep_model_dir / "best.pth")
        else:
            no_improve += 1

        if cfg.train.early_stop_enabled and no_improve >= cfg.train.early_stop_patience:
            print(f"[{now_str()}] rep={rep_idx} early-stop at epoch={epoch} best_val={best_val:.6f}")
            break

    if last_epoch_num > 0:
        _save_state_dict(model, rep_model_dir / "last.pth")

    # load best and test (inference forced to row_topk)
    if best_state is not None:
        model.load_state_dict(best_state)

    prune_stats = _prune_moe_experts_by_norm(cfg, model)
    if prune_stats["prune_enabled"] > 0:
        print(
            f"[{now_str()}] Prune experts: kept={int(prune_stats['kept_experts'])} "
            f"pruned={int(prune_stats['pruned_experts'])} "
            f"threshold={cfg.model.prune_experts_norm_threshold}"
        )
    finetune_curves = _finetune_after_prune(
        cfg,
        model,
        train_loader,
        val_loader,
        test_loader,
        device,
    )
    pretrain_train_losses = list(train_losses)
    pretrain_val_losses = list(val_losses)
    pretrain_test_losses = list(test_losses)
    if finetune_curves["train_losses"]:
        train_losses.extend(finetune_curves["train_losses"])
        val_losses.extend(finetune_curves["val_losses"])
        test_losses.extend(finetune_curves["test_losses"])
        if val_losses:
            best_epoch = int(np.argmin(val_losses))
            best_val = float(val_losses[best_epoch])
        _save_state_dict(model, rep_model_dir / "last.pth")

    # force inference routing mode
    prev_mode = cfg.model.routing_mode
    cfg.model.routing_mode = "row_topk"
    model.cfg.routing_mode = "row_topk"
    if model.moe is not None:
        model.moe.cfg.routing_mode = "row_topk"

    trainm = _evaluate_split(cfg, model, train_loader, device, split="train", compute_sim_metrics=cfg.train.compute_sim_metrics)
    t_test = time.time()
    testm = _evaluate_split(
        cfg,
        model,
        test_loader,
        device,
        split="test",
        out_dir=None,
        file_prefix=rep_prefix,
        compute_flops=True,
        compute_sim_metrics=False,
    )
    testm["total_time_sec"] = float(time.time() - t_test)

    total_params = _count_params(model)
    expert_params = _expert_param_count(model)
    gate_cosine = _gate_cosine(model)
    train_active_ratio = float(np.mean(train_active_counts) / cfg.model.num_experts) if train_active_counts else 0.0
    trainm["samples_per_sec"] = float(total_train_samples / total_train_time) if total_train_time > 0 else 0.0
    trainm["avg_step_time"] = float(total_train_time / max(total_train_steps, 1))
    trainm["total_time_sec"] = float(total_train_time)
    trainm["active_expert_ratio"] = train_active_ratio
    trainm["total_params"] = total_params
    trainm["active_params"] = _effective_param_count(total_params, expert_params, train_active_ratio)
    trainm["flops_total"] = float(train_flops_total)
    trainm["expert_gate_cosine"] = gate_cosine
    trainm["prune_enabled"] = prune_stats["prune_enabled"]
    trainm["pruned_experts"] = prune_stats["pruned_experts"]
    trainm["kept_experts"] = prune_stats["kept_experts"]
    trainm["prune_norm_threshold"] = float(cfg.model.prune_experts_norm_threshold)

    testm["total_params"] = total_params
    testm["active_params"] = _effective_param_count(total_params, expert_params, testm.get("active_expert_ratio", 0.0))
    testm["expert_gate_cosine"] = gate_cosine
    testm["prune_enabled"] = prune_stats["prune_enabled"]
    testm["pruned_experts"] = prune_stats["pruned_experts"]
    testm["kept_experts"] = prune_stats["kept_experts"]
    testm["prune_norm_threshold"] = float(cfg.model.prune_experts_norm_threshold)

    test_pred = None
    test_pred_prob = None
    if cfg.data_variant in ("data2", "data3"):
        test_pred = _predict_values(model, test_loader, device, cfg=cfg, split="test")
    else:
        test_pred_prob = _predict_probs(model, test_loader, device, cfg=cfg, split="test")

    # restore
    cfg.model.routing_mode = prev_mode
    model.cfg.routing_mode = prev_mode
    if model.moe is not None:
        model.moe.cfg.routing_mode = prev_mode

    return {
        "rep_idx": rep_idx,
        "rep_seed": rep_seed,
        "pretrain_train_losses": pretrain_train_losses,
        "pretrain_val_losses": pretrain_val_losses,
        "pretrain_test_losses": pretrain_test_losses,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "test_losses": test_losses,
        "best_val": best_val,
        "best_epoch": best_epoch,
        "val_metrics_last": {"loss": val_losses[-1] if val_losses else None},
        "train_init_metrics": train_init_metrics,
        "train_metrics": trainm,
        "test_metrics": testm,
        "test_pred": test_pred,
        "test_pred_prob": test_pred_prob,
    }
