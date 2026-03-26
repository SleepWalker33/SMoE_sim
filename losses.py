# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn.functional as F


def gate1_budget_loss(g1: torch.Tensor, budget: float, kind: str = "l1") -> torch.Tensor:
    if g1.dim() > 1:
        g1 = g1.squeeze(-1)
    if kind == "l1":
        return (g1 - budget).abs().mean()
    if kind == "l2":
        return ((g1 - budget) ** 2).mean()
    raise ValueError(kind)


def smoe_balance_loss(router_scores: torch.Tensor, dispatch_mask: torch.Tensor, tau0: float = 1.0, eps: float = 1e-9) -> torch.Tensor:
    """SMoE/Fedus balance loss (Eq.9).
    router_scores: [B,E] raw scores s_i(t)
    dispatch_mask: [B,E] hard routing mask (no grad)
    """
    B, E = router_scores.shape
    counts = dispatch_mask.float().sum(dim=0)        # [E]
    t = (counts / (float(B) + eps)).detach()         # t_i

    p = F.softmax(router_scores / max(tau0, 1e-6), dim=-1)  # p_i(t) with tau0
    inner = (p * t.unsqueeze(0)).sum(dim=-1)               # per token
    return (E / (float(B) + eps)) * inner.sum()


def _center_rows(H: torch.Tensor) -> torch.Tensor:
    return H - H.mean(dim=0, keepdim=True)


def _center_kernel(K: torch.Tensor) -> torch.Tensor:
    """Double-center a kernel matrix (HSIC-style centering)."""
    mean_row = K.mean(dim=0, keepdim=True)
    mean_col = K.mean(dim=1, keepdim=True)
    mean_all = K.mean()
    return K - mean_row - mean_col + mean_all


def _rbf_kernel(H: torch.Tensor, sigma: float) -> torch.Tensor:
    """RBF (Gaussian) kernel matrix for rows of H."""
    norm = (H * H).sum(dim=1, keepdim=True)
    dist2 = (norm + norm.T - 2.0 * (H @ H.T)).clamp_min(0.0)
    return torch.exp(-dist2 / (2.0 * sigma * sigma))


def compute_cka(
    H1: torch.Tensor,
    H2: torch.Tensor,
    kernel: str = "linear",
    sigma: float = 0.85,
    eps: float = 1e-12,
) -> torch.Tensor:
    """CKA with selectable kernel (linear or rbf)."""
    if kernel == "rbf":
        K = _center_kernel(_rbf_kernel(H1, sigma))
        L = _center_kernel(_rbf_kernel(H2, sigma))
    else:  # linear
        H1c = _center_rows(H1)
        H2c = _center_rows(H2)
        K = H1c @ H1c.T
        L = H2c @ H2c.T
    hsic = (K * L).sum()
    norm1 = (K * K).sum().clamp_min(eps).sqrt()
    norm2 = (L * L).sum().clamp_min(eps).sqrt()
    return hsic / (norm1 * norm2 + eps)


def linear_cka(H1: torch.Tensor, H2: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Linear CKA via Gram matrices (kept for eval/pairwise_cka_matrix compatibility)."""
    return compute_cka(H1, H2, kernel="linear", eps=eps)


def similarity_loss_cka(
    expert_feats: Dict[int, torch.Tensor],
    dispatch_mask: Optional[torch.Tensor] = None,
    trunk_hidden: Optional[torch.Tensor] = None,
    expert_modules: Optional[torch.nn.ModuleList] = None,
    router_scores: Optional[torch.Tensor] = None,
    topk: int = 2,
    proj_module: Optional[torch.nn.Module] = None,
    expert_feat_indices: Optional[Dict[int, torch.Tensor]] = None,
    f_star: int = 2,
    t_star: float = 0.0,
    kernel: str = "linear",
    sigma: float = 0.85,
) -> torch.Tensor:
    # Online top-2: group tokens by their top-2 experts in this batch, then CKA per pair.
    # Prefer pre-computed expert_feat_indices to avoid re-running experts with a different dropout mask.
    if router_scores is not None and topk >= 2 and (expert_modules is not None or expert_feat_indices is not None):
        E = len(expert_modules) if expert_modules is not None else (max(expert_feat_indices.keys()) + 1 if expert_feat_indices else 0)
        ref_device = router_scores.device
        if E < 2:
            return torch.tensor(0.0, device=ref_device)
        top2 = torch.topk(router_scores, k=min(2, router_scores.shape[-1]), dim=-1).indices  # [B,2]
        a = torch.minimum(top2[:, 0], top2[:, 1])
        b = torch.maximum(top2[:, 0], top2[:, 1])
        pair_codes = a * E + b
        uniq = torch.unique(pair_codes)
        sims = []
        for code in uniq.tolist():
            i = int(code) // E
            j = int(code) % E
            sel = pair_codes == code
            # f_star: require at least f_star tokens sharing this pair
            if sel.float().sum().item() < max(2, f_star):
                continue

            if (expert_feat_indices is not None
                    and i in expert_feat_indices and j in expert_feat_indices
                    and i in expert_feats and j in expert_feats):
                # Strict token alignment on shared set S_ij:
                # S_ij = tokens with current top2-pair code (i,j) AND dispatched to both i and j.
                idx_i = expert_feat_indices[i]
                idx_j = expert_feat_indices[j]
                sel_idx = torch.where(sel)[0]
                if sel_idx.numel() < max(2, f_star):
                    continue

                if dispatch_mask is not None:
                    both = dispatch_mask[sel_idx, i] & dispatch_mask[sel_idx, j]
                    common_idx = sel_idx[both]
                else:
                    sel_i = idx_i[torch.isin(idx_i, sel_idx)]
                    sel_j = idx_j[torch.isin(idx_j, sel_idx)]
                    common_idx = sel_i[torch.isin(sel_i, sel_j)]

                if common_idx.numel() < max(2, f_star):
                    continue

                pos_i = torch.isin(idx_i, common_idx)
                pos_j = torch.isin(idx_j, common_idx)
                hi = expert_feats[i][pos_i]
                hj = expert_feats[j][pos_j]
                if hi.shape[0] < 2 or hj.shape[0] < 2:
                    continue
                if hi.shape[0] != hj.shape[0]:
                    # Extra safety when mask/index sources are not perfectly synchronized.
                    n = min(hi.shape[0], hj.shape[0])
                    if n < 2:
                        continue
                    hi = hi[:n]
                    hj = hj[:n]
            elif expert_modules is not None and trunk_hidden is not None:
                # Fallback: re-run experts (dropout mask differs from forward pass).
                if sel.float().sum().item() < max(2, f_star):
                    continue
                hi = expert_modules[i](trunk_hidden[sel])
                hj = expert_modules[j](trunk_hidden[sel])
            else:
                continue

            if proj_module is not None:
                hi = proj_module(hi)
                hj = proj_module(hj)
            cka_val = compute_cka(hi, hj, kernel=kernel, sigma=sigma)
            # t_star: skip pairs whose CKA is already below the threshold (not similar enough to penalize)
            if cka_val.item() < t_star:
                continue
            sims.append(cka_val)
        if not sims:
            return torch.tensor(0.0, device=ref_device)
        return torch.stack(sims).mean()

    # Fallback path: undefined sampling controls -> compute CKA on all pairs.
    ids = sorted(expert_feats.keys())
    if len(ids) < 2:
        device = next(iter(expert_feats.values())).device if ids else "cpu"
        return torch.tensor(0.0, device=device)

    min_n = min(expert_feats[i].shape[0] for i in ids)
    if min_n < max(2, f_star):
        return torch.tensor(0.0, device=expert_feats[ids[0]].device)

    Fmap = {i: expert_feats[i][:min_n] for i in ids}
    all_pairs = [(ids[a], ids[b]) for a in range(len(ids)) for b in range(a + 1, len(ids))]

    sims = []
    for i, j in all_pairs:
        cka_val = compute_cka(Fmap[i], Fmap[j], kernel=kernel, sigma=sigma)
        if cka_val.item() >= t_star:
            sims.append(cka_val)
    if not sims:
        return torch.tensor(0.0, device=Fmap[ids[0]].device)
    return torch.stack(sims).mean()


def _linear_weight_params(mod: torch.nn.Module):
    def _weight_of(m: torch.nn.Module):
        if isinstance(m, torch.nn.Linear):
            return m.weight
        if hasattr(m, "effective_weight") and callable(getattr(m, "effective_weight")):
            w = m.effective_weight()
            if torch.is_tensor(w):
                return w
        return None

    for m in mod.modules():
        w = _weight_of(m)
        if w is not None:
            yield w


def l1_sparsity_loss(mods) -> torch.Tensor:
    total = None
    for mod in mods:
        for W in _linear_weight_params(mod):
            val = W.abs().sum()
            total = val if total is None else total + val
    if total is None:
        return torch.tensor(0.0)
    return total


def group_lasso_loss(mods, eps: float = 1e-8) -> torch.Tensor:
    total = None
    for mod in mods:
        for W in _linear_weight_params(mod):
            # group by output unit (rows)
            row_norm = (W * W).sum(dim=1).clamp_min(eps).sqrt().sum()
            total = row_norm if total is None else total + row_norm
    if total is None:
        return torch.tensor(0.0)
    return total


def pairwise_cka_matrix(expert_feats: Dict[int, torch.Tensor], num_experts: int) -> torch.Tensor:
    """Return [E,E] CKA matrix; missing experts -> NaN."""
    device = next(iter(expert_feats.values())).device if expert_feats else "cpu"
    E = num_experts
    M = torch.full((E, E), float("nan"), device=device)
    for i in range(E):
        M[i, i] = 1.0

    ids = sorted(expert_feats.keys())
    if not ids:
        return M

    min_n = min(expert_feats[i].shape[0] for i in ids)
    if min_n < 2:
        return M

    Fmap = {i: expert_feats[i][:min_n] for i in ids}
    for i in ids:
        for j in ids:
            if j < i:
                continue
            if i == j:
                v = torch.tensor(1.0, device=device)
            else:
                v = linear_cka(Fmap[i], Fmap[j])
            M[i, j] = v
            M[j, i] = v
    return M


def expert_first_layer_orth_loss(experts: torch.nn.ModuleList) -> torch.Tensor:
    """Penalize non-orthogonality among experts' first linear layer weights."""
    if experts is None or len(experts) < 2:
        return torch.tensor(0.0)
    weights = []
    device = None
    for exp in experts:
        W = _kth_linear_weight(exp, 0)
        if W is None:
            continue
        device = W.device
        weights.append(W.reshape(W.shape[0], -1))
    if len(weights) < 2:
        return torch.tensor(0.0, device=device if device is not None else "cpu")
    mats = [w.reshape(w.shape[0], -1) for w in weights]
    loss = torch.tensor(0.0, device=device if device is not None else "cpu")
    count = 0
    for i in range(len(mats)):
        Wi = mats[i] / (mats[i].norm(dim=1, keepdim=True) + 1e-12)
        for j in range(i + 1, len(mats)):
            Wj = mats[j] / (mats[j].norm(dim=1, keepdim=True) + 1e-12)
            prod = Wi @ Wj.T
            loss = loss + (prod * prod).mean()
            count += 1
    return loss / max(count, 1)


def _kth_linear_weight(mod: torch.nn.Module, k: int):
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


def _orth_loss_from_weights(weights: List[torch.Tensor]) -> torch.Tensor:
    if len(weights) < 2:
        device = weights[0].device if weights else "cpu"
        return torch.tensor(0.0, device=device)
    mats = [w.reshape(w.shape[0], -1) for w in weights]
    device = mats[0].device
    loss = torch.tensor(0.0, device=device)
    count = 0
    for i in range(len(mats)):
        Wi = mats[i] / (mats[i].norm(dim=1, keepdim=True) + 1e-12)
        for j in range(i + 1, len(mats)):
            Wj = mats[j] / (mats[j].norm(dim=1, keepdim=True) + 1e-12)
            prod = Wi @ Wj.T
            loss = loss + (prod * prod).mean()
            count += 1
    return loss / max(count, 1)


def expert_layer_orth_loss(experts: torch.nn.ModuleList, linear_index: int) -> torch.Tensor:
    if experts is None or len(experts) < 2:
        return torch.tensor(0.0)
    weights = []
    for exp in experts:
        W = _kth_linear_weight(exp, linear_index)
        if W is not None:
            weights.append(W)
    return _orth_loss_from_weights(weights)


def _collect_lora_factors(mod: torch.nn.Module, factor: str) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    for m in mod.modules():
        if hasattr(m, factor):
            t = getattr(m, factor)
            if torch.is_tensor(t):
                out.append(t)
    return out


def _orth_loss_from_A_factors(mats: List[torch.Tensor]) -> torch.Tensor:
    # A_i^T A_j -> encourage LoRA input subspaces to be orthogonal.
    if len(mats) < 2:
        device = mats[0].device if mats else "cpu"
        return torch.tensor(0.0, device=device)
    device = mats[0].device
    loss = torch.tensor(0.0, device=device)
    count = 0
    for i in range(len(mats)):
        Ai = mats[i]
        Ai = Ai / (Ai.norm(dim=0, keepdim=True) + 1e-12)
        for j in range(i + 1, len(mats)):
            Aj = mats[j]
            Aj = Aj / (Aj.norm(dim=0, keepdim=True) + 1e-12)
            prod = Ai.T @ Aj
            loss = loss + (prod * prod).mean()
            count += 1
    return loss / max(count, 1)


def _orth_loss_from_B_factors(mats: List[torch.Tensor]) -> torch.Tensor:
    # B_i B_j^T -> encourage LoRA output subspaces to be orthogonal.
    if len(mats) < 2:
        device = mats[0].device if mats else "cpu"
        return torch.tensor(0.0, device=device)
    device = mats[0].device
    loss = torch.tensor(0.0, device=device)
    count = 0
    for i in range(len(mats)):
        Bi = mats[i]
        Bi = Bi / (Bi.norm(dim=1, keepdim=True) + 1e-12)
        for j in range(i + 1, len(mats)):
            Bj = mats[j]
            Bj = Bj / (Bj.norm(dim=1, keepdim=True) + 1e-12)
            prod = Bi @ Bj.T
            loss = loss + (prod * prod).mean()
            count += 1
    return loss / max(count, 1)


def expert_lora_A_orth_loss(experts: torch.nn.ModuleList) -> torch.Tensor:
    """LoRA-only: orthogonality across experts on all A factors (layer-wise average)."""
    if experts is None or len(experts) < 2:
        return torch.tensor(0.0)
    A_by_expert = [_collect_lora_factors(exp, "A") for exp in experts]
    min_layers = min((len(v) for v in A_by_expert), default=0)
    if min_layers <= 0:
        device = next(experts[0].parameters()).device if len(experts) > 0 else "cpu"
        return torch.tensor(0.0, device=device)
    vals = []
    for li in range(min_layers):
        mats = [v[li] for v in A_by_expert]
        vals.append(_orth_loss_from_A_factors(mats))
    return torch.stack(vals).mean() if vals else torch.tensor(0.0, device=next(experts[0].parameters()).device)


def expert_lora_B_orth_loss(experts: torch.nn.ModuleList) -> torch.Tensor:
    """LoRA-only: orthogonality across experts on all B factors (layer-wise average)."""
    if experts is None or len(experts) < 2:
        return torch.tensor(0.0)
    B_by_expert = [_collect_lora_factors(exp, "B") for exp in experts]
    min_layers = min((len(v) for v in B_by_expert), default=0)
    if min_layers <= 0:
        device = next(experts[0].parameters()).device if len(experts) > 0 else "cpu"
        return torch.tensor(0.0, device=device)
    vals = []
    for li in range(min_layers):
        mats = [v[li] for v in B_by_expert]
        vals.append(_orth_loss_from_B_factors(mats))
    return torch.stack(vals).mean() if vals else torch.tensor(0.0, device=next(experts[0].parameters()).device)


def shared_moe_layer_orth_loss(shared_mods: List[torch.nn.Module], experts: torch.nn.ModuleList, linear_index: int) -> torch.Tensor:
    if not shared_mods or experts is None:
        return torch.tensor(0.0)
    shared_weights = []
    for mod in shared_mods:
        W = _kth_linear_weight(mod, linear_index)
        if W is not None:
            shared_weights.append(W)
    expert_weights = []
    for exp in experts:
        W = _kth_linear_weight(exp, linear_index)
        if W is not None:
            expert_weights.append(W)
    if not shared_weights or not expert_weights:
        device = shared_weights[0].device if shared_weights else (expert_weights[0].device if expert_weights else "cpu")
        return torch.tensor(0.0, device=device)
    weights = shared_weights + expert_weights
    return _orth_loss_from_weights(weights)
