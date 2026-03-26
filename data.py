# -*- coding: utf-8 -*-
"""Synthetic tabular generator.
data1: classification with latent regimes.
data2: regression from regime-specific smooth generators.
data3: regression from continuous piecewise-linear generator.
Dataset returns (x, y, regime_id).
"""
from __future__ import annotations
from typing import Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from itertools import combinations
import random

from config import Data1Config, Data2Config, Data3Config
from losses import pairwise_cka_matrix


def _make_correlated_cov(d: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    A = rng.normal(size=(d, d))
    cov = (A @ A.T) / d
    cov = cov / np.trace(cov) * d
    return cov * scale


def _act_np(h: np.ndarray, act: str) -> np.ndarray:
    if act == "tanh":
        return np.tanh(h)
    if act == "relu":
        return np.maximum(h, 0.0)
    if act == "gelu":
        return 0.5 * h * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (h + 0.044715 * (h**3))))
    raise ValueError(act)


def _mlp_forward_np(x: np.ndarray, weights: list, act: str) -> np.ndarray:
    h = x
    for i, layer in enumerate(weights):
        if isinstance(layer, dict):
            lt = layer.get("type", "dense")
            if lt == "dense":
                W = layer["W"]
                b = layer["b"]
                h = h @ W + b
            elif lt == "lora":
                A = layer["A"]
                B = layer["B"]
                b = layer["b"]
                rank = int(layer.get("rank", A.shape[1]))
                alpha = float(layer.get("alpha", float(rank)))
                scale = alpha / max(1.0, float(rank))
                h = (h @ A) @ B
                h = h * scale + b
            else:
                raise ValueError(f"Unknown layer type: {lt}")
        else:
            W, b = layer
            h = h @ W + b
        if i < len(weights) - 1:
            h = _act_np(h, act)
    return h


def _regime_cos_target(cfg: Data1Config) -> float:
    level = cfg.regime_sim_level
    if level == "high":
        return cfg.regime_cos_target_high
    if level == "low":
        return cfg.regime_cos_target_low
    return cfg.regime_cos_target_mid


def _regime_func_cos_target(cfg: Data1Config) -> float:
    level = cfg.regime_func_sim_level
    if level == "high":
        return cfg.regime_func_cos_target_high
    if level == "low":
        return cfg.regime_func_cos_target_low
    return cfg.regime_func_cos_target_mid


def _make_regime_matrix(cfg: Data1Config, task_rng: np.random.Generator) -> np.ndarray:
    """Original phi-based regime matrix (phi_dim = 3*d)."""
    d = _effective_input_dim(cfg)
    phi_dim = d * 3
    R = cfg.num_regimes
    M = task_rng.normal(scale=1.0, size=(phi_dim, R + 1)).astype(np.float32)
    Q, _ = np.linalg.qr(M)
    v_shared = Q[:, 0]
    U = Q[:, 1:R + 1]
    cos_target = float(np.clip(_regime_cos_target(cfg), 0.0, 1.0))
    s = float(np.sqrt(cos_target))
    t = float(np.sqrt(max(0.0, 1.0 - s * s)))
    A = (s * v_shared[:, None] + t * U).T.astype(np.float32)
    return A


def _make_regime_matrix_linear(cfg: Data1Config, task_rng: np.random.Generator) -> np.ndarray:
    """Linear regime matrix operating directly on x (dim = d)."""
    d = _effective_input_dim(cfg)
    R = cfg.num_regimes
    M = task_rng.normal(scale=1.0, size=(d, R + 1)).astype(np.float32)
    Q, _ = np.linalg.qr(M)
    v_shared = Q[:, 0]
    U = Q[:, 1:R + 1]
    cos_target = float(np.clip(_regime_cos_target(cfg), 0.0, 1.0))
    s = float(np.sqrt(cos_target))
    t = float(np.sqrt(max(0.0, 1.0 - s * s)))
    A = (s * v_shared[:, None] + t * U).T.astype(np.float32)
    return A


def _make_regime_gate_nonlinear(cfg: Data1Config, task_rng: np.random.Generator):
    """Single hidden layer gate: x -> relu -> scores.  Returns (W1, b1, W2, b2)."""
    d = _effective_input_dim(cfg)
    R = cfg.num_regimes
    h = int(getattr(cfg, "regime_gate_hidden", 32))
    W1 = task_rng.normal(scale=1.0 / np.sqrt(d), size=(d, h)).astype(np.float32)
    b1 = task_rng.normal(scale=0.1, size=(h,)).astype(np.float32)
    W2 = task_rng.normal(scale=1.0 / np.sqrt(h), size=(h, R)).astype(np.float32)
    b2 = task_rng.normal(scale=0.1, size=(R,)).astype(np.float32)
    return W1, b1, W2, b2


def _compute_gate_scores(cfg: Data1Config, x: np.ndarray, task_rng: np.random.Generator):
    """Compute gate logits/scores for regime assignment (linear/nonlinear/softmax).

    Returns (scores, gate_params) where gate_params can be reused for MC estimation.
    NOTE: "phi" gate type is handled inline by callers, not through this function.
    """
    gate_type = getattr(cfg, "regime_gate_type", "phi")

    if gate_type == "linear":
        A = _make_regime_matrix_linear(cfg, task_rng)
        scores = x @ A.T
        return scores, {"type": "linear", "A": A}

    elif gate_type == "nonlinear":
        W1, b1, W2, b2 = _make_regime_gate_nonlinear(cfg, task_rng)
        h = np.maximum(x @ W1 + b1, 0.0)  # relu
        scores = h @ W2 + b2
        return scores, {"type": "nonlinear", "W1": W1, "b1": b1, "W2": W2, "b2": b2}

    elif gate_type == "softmax":
        A = _make_regime_matrix_linear(cfg, task_rng)
        T = float(getattr(cfg, "regime_gate_temperature", 1.0))
        scores = x @ A.T / T
        return scores, {"type": "softmax", "A": A, "T": T}

    else:
        raise ValueError(f"Unknown regime_gate_type: {gate_type}")


def _assign_regime(
    cfg: Data1Config,
    scores: np.ndarray,
    gate_params: dict,
    data_rng: np.random.Generator,
) -> np.ndarray:
    """Assign regime from gate scores, respecting gate type and noise."""
    gate_type = gate_params["type"]
    n, R = scores.shape

    if gate_type == "softmax":
        # softmax probability sampling
        exp_s = np.exp(scores - scores.max(axis=1, keepdims=True))
        probs = exp_s / exp_s.sum(axis=1, keepdims=True)
        regime = np.array([data_rng.choice(R, p=probs[i]) for i in range(n)], dtype=np.int64)
    else:
        # linear / nonlinear / phi: argmax with optional Gaussian noise
        if cfg.regime_noise_std > 0:
            scores = scores + data_rng.normal(scale=cfg.regime_noise_std, size=scores.shape)
        regime = np.argmax(scores, axis=1).astype(np.int64)

    return regime


def _recompute_gate_scores(
    gate_params: dict,
    x: np.ndarray,
) -> np.ndarray:
    """Recompute gate scores from gate_params (for MC estimation)."""
    gt = gate_params["type"]
    if gt == "linear":
        return x @ gate_params["A"].T
    elif gt == "nonlinear":
        h = np.maximum(x @ gate_params["W1"] + gate_params["b1"], 0.0)
        return h @ gate_params["W2"] + gate_params["b2"]
    elif gt == "softmax":
        return x @ gate_params["A"].T / gate_params["T"]
    else:
        raise ValueError(gt)


def _resolve_data_seed(split: str, seed: int) -> int:
    split_offset = {"train": 0, "val": 10_000, "test": 20_000}[split]
    return int(seed + 1 + split_offset)


def _resolve_task_data_norm_seeds(
    cfg: Data1Config,
    split: str,
    seed: int,
) -> Tuple[int, int, int]:
    """Resolve (task_seed, data_seed, norm_seed) for split generation."""
    seed = int(seed)
    if getattr(cfg, "fixed_test_set", False):
        task_seed = int(cfg.seed)
        data_seed = int(cfg.seed) if split == "test" else seed
        norm_seed = int(cfg.seed)
    else:
        task_seed = seed
        data_seed = seed
        norm_seed = seed
    return task_seed, data_seed, norm_seed


def _compute_train_norm_stats(
    cfg: Data1Config,
    seed: int,
    task_seed: int | None = None,
    data_seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    n_train = int(cfg.n_train)
    task_seed = int(seed if task_seed is None else task_seed)
    data_seed = int(seed if data_seed is None else data_seed)
    task_rng = np.random.default_rng(task_seed)
    params = _make_data1_task_params(cfg, task_rng)
    mean = params["mean"]
    cov = params["cov"]
    data_rng = np.random.default_rng(_resolve_data_seed("train", data_seed))
    x = data_rng.multivariate_normal(mean, cov, size=n_train).astype(np.float32)
    mu = x.mean(0, keepdims=True)
    sigma = x.std(0, keepdims=True)
    return mu, sigma


def _make_regime_mlp_weights(
    cfg: Data1Config,
    task_rng: np.random.Generator,
) -> list:
    d = _effective_input_dim(cfg)
    depth = cfg.regime_mlp_depth
    hid = cfg.regime_mlp_hidden
    feat_dim = max(1, cfg.regime_mlp_out)
    layer_shapes = []
    if depth <= 1:
        layer_shapes.append((d, feat_dim))
    else:
        in_dim = d
        for _ in range(depth - 1):
            layer_shapes.append((in_dim, hid))
            in_dim = hid
        layer_shapes.append((in_dim, feat_dim))

    cos_target = float(np.clip(_regime_func_cos_target(cfg), 0.0, 1.0))
    s = float(np.sqrt(cos_target))
    t = float(np.sqrt(max(0.0, 1.0 - s * s)))

    regime_type = getattr(cfg, "regime_nonshared_type", "mlp")
    if regime_type == "mlp":
        shared_layers = []
        for in_dim, out_dim in layer_shapes:
            W = task_rng.normal(scale=1.0 / np.sqrt(in_dim), size=(in_dim, out_dim)).astype(np.float32)
            b = task_rng.normal(scale=0.1, size=(out_dim,)).astype(np.float32)
            shared_layers.append((W, b))

        weights_by_r = []
        for _r in range(cfg.num_regimes):
            layers = []
            for (W_shared, b_shared), (in_dim, out_dim) in zip(shared_layers, layer_shapes):
                W_noise = task_rng.normal(scale=1.0 / np.sqrt(in_dim), size=(in_dim, out_dim)).astype(np.float32)
                b_noise = task_rng.normal(scale=0.1, size=(out_dim,)).astype(np.float32)
                W = (s * W_shared + t * W_noise).astype(np.float32)
                b = (s * b_shared + t * b_noise).astype(np.float32)
                layers.append((W, b))
            weights_by_r.append(layers)

        if cfg.regime_mlp_sparsity == "sparse" and cfg.regime_mlp_depth >= 2:
            if cfg.num_regimes < 4:
                raise ValueError("regime_mlp_sparsity='sparse' requires num_regimes >= 4")
            W0, _ = weights_by_r[0][0]
            W0[:, : W0.shape[1] // 2] = 0.0
            W1, _ = weights_by_r[1][0]
            W1[:, W1.shape[1] // 2 :] = 0.0
            W2, _ = weights_by_r[2][1]
            W2[:, : W2.shape[1] // 2] = 0.0
            W3, _ = weights_by_r[3][1]
            W3[:, W3.shape[1] // 2 :] = 0.0
        return weights_by_r

    if regime_type == "lora":
        rank_cfg = int(getattr(cfg, "regime_lora_rank", 16))
        alpha = float(getattr(cfg, "regime_lora_alpha", max(rank_cfg, 1)))
        shared_layers = []
        for in_dim, out_dim in layer_shapes:
            r = max(1, min(rank_cfg, int(in_dim), int(out_dim)))
            A = task_rng.normal(scale=1.0 / np.sqrt(in_dim), size=(in_dim, r)).astype(np.float32)
            B = task_rng.normal(scale=1.0 / np.sqrt(max(r, 1)), size=(r, out_dim)).astype(np.float32)
            b = task_rng.normal(scale=0.1, size=(out_dim,)).astype(np.float32)
            shared_layers.append((A, B, b, r))

        weights_by_r = []
        for _r in range(cfg.num_regimes):
            layers = []
            for (A_shared, B_shared, b_shared, r), (in_dim, out_dim) in zip(shared_layers, layer_shapes):
                A_noise = task_rng.normal(scale=1.0 / np.sqrt(in_dim), size=(in_dim, r)).astype(np.float32)
                B_noise = task_rng.normal(scale=1.0 / np.sqrt(max(r, 1)), size=(r, out_dim)).astype(np.float32)
                b_noise = task_rng.normal(scale=0.1, size=(out_dim,)).astype(np.float32)
                A = (s * A_shared + t * A_noise).astype(np.float32)
                B = (s * B_shared + t * B_noise).astype(np.float32)
                b = (s * b_shared + t * b_noise).astype(np.float32)
                layers.append({
                    "type": "lora",
                    "A": A,
                    "B": B,
                    "b": b,
                    "rank": int(r),
                    "alpha": float(alpha),
                })
            weights_by_r.append(layers)

        if cfg.regime_mlp_sparsity == "sparse" and cfg.regime_mlp_depth >= 2:
            if cfg.num_regimes < 4:
                raise ValueError("regime_mlp_sparsity='sparse' requires num_regimes >= 4")
            L0 = weights_by_r[0][0]
            L0["B"][:, : L0["B"].shape[1] // 2] = 0.0
            L1 = weights_by_r[1][0]
            L1["B"][:, L1["B"].shape[1] // 2 :] = 0.0
            L2 = weights_by_r[2][1]
            L2["B"][:, : L2["B"].shape[1] // 2] = 0.0
            L3 = weights_by_r[3][1]
            L3["B"][:, L3["B"].shape[1] // 2 :] = 0.0
        return weights_by_r

    raise ValueError(f"Unknown regime_nonshared_type: {regime_type}")


def _data1_logits_from_x_regime(
    cfg: Data1Config,
    x: np.ndarray,
    regime: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Compute logits from x/regime using data1 task params (no noise)."""
    task_seed = int(cfg.seed) if getattr(cfg, "fixed_test_set", False) else int(seed)
    task_rng = np.random.default_rng(task_seed)
    params = _make_data1_task_params(cfg, task_rng)
    shared_weights = params["shared_weights"]
    weights_by_r = params["weights_by_r"]
    head_W = params["head_W"]
    head_b = params["head_b"]

    d = _effective_input_dim(cfg)
    x_core = x[:, :d].astype(np.float32)
    C = cfg.num_classes
    R = cfg.num_regimes
    logit_scale = _data1_logit_scale(cfg) if C > 1 else 1.0

    logits = np.zeros((x_core.shape[0], C), dtype=np.float32)
    for r in range(R):
        idx = np.where(regime == r)[0]
        if idx.size == 0:
            continue
        feat = _mlp_forward_np(x_core[idx], weights_by_r[r], cfg.regime_activation)
        if shared_weights is not None:
            shared_feat = _mlp_forward_np(x_core[idx], shared_weights, cfg.regime_activation)
            feat = cfg.regime_specific_weight * feat + cfg.share_regime_weight * shared_feat
        logits[idx] = logit_scale * (feat @ head_W + head_b)
    return logits


def compute_regime_cosine(cfg: Data1Config, seed: int) -> float:
    task_seed = int(cfg.seed) if getattr(cfg, "fixed_test_set", False) else int(seed)
    task_rng = np.random.default_rng(task_seed)
    gate_type = getattr(cfg, "regime_gate_type", "phi")
    if gate_type == "phi":
        A = _make_regime_matrix(cfg, task_rng)
    elif gate_type in ("linear", "softmax"):
        A = _make_regime_matrix_linear(cfg, task_rng)
    elif gate_type == "nonlinear":
        # For nonlinear gate, cosine of weight rows is not meaningful.
        return 0.0
    else:
        raise ValueError(f"Unknown regime_gate_type: {gate_type}")
    R = A.shape[0]
    if R <= 1:
        return 0.0
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    sim = A_norm @ A_norm.T
    offdiag = sim[~np.eye(R, dtype=bool)]
    return float(offdiag.mean()) if offdiag.size else 0.0


def _entropy_from_counts(counts: np.ndarray) -> float:
    p = counts / (counts.sum() + 1e-9)
    return float(-(p * np.log(p + 1e-9)).sum())


def _gini_from_counts(counts: np.ndarray) -> float:
    x = np.sort(counts.astype(np.float64))
    n = x.size
    if n == 0 or x.sum() <= 0:
        return 0.0
    idx = np.arange(1, n + 1, dtype=np.float64)
    g = (2 * (idx * x).sum() / (n * x.sum())) - (n + 1) / n
    return float(g)


def _pes_from_expert_outputs_np(expert_outs: np.ndarray, eps: float = 1e-12) -> float:
    """expert_outs: [N,E,D], return mean pairwise cosine over experts per sample."""
    if expert_outs.ndim != 3:
        return 0.0
    n, e, _ = expert_outs.shape
    if n == 0 or e < 2:
        return 0.0
    norms = np.linalg.norm(expert_outs, axis=-1, keepdims=True)
    v = expert_outs / (norms + eps)
    cos = np.matmul(v, np.swapaxes(v, 1, 2))  # [N,E,E]
    diag = np.diagonal(cos, axis1=1, axis2=2).sum(axis=1)
    off_sum = cos.sum(axis=(1, 2)) - diag
    per = off_sum / (e * (e - 1))
    return float(per.mean())


def _choose_subsets_np(E: int, k: int, max_subsets: int, seed: int) -> list[tuple[int, ...]]:
    if k <= 0 or E <= k:
        return [tuple(range(E))]
    all_subsets = list(combinations(range(E), k))
    if max_subsets and max_subsets > 0 and len(all_subsets) > max_subsets:
        rng = random.Random(seed)
        return rng.sample(all_subsets, k=max_subsets)
    return all_subsets


def _cka_offdiag_mean_over_subsets_np(cka_mat: torch.Tensor, subsets: list[tuple[int, ...]]) -> float:
    E = int(cka_mat.shape[0])
    if E <= 1:
        return 0.0
    cka_np = cka_mat.detach().cpu().numpy()
    vals: list[float] = []
    for sub in subsets:
        if len(sub) <= 1:
            continue
        M = cka_np[np.ix_(sub, sub)]
        k = M.shape[0]
        off = M[~np.eye(k, dtype=bool)]
        off = off[np.isfinite(off)]
        if off.size:
            vals.append(float(off.mean()))
    return float(np.mean(vals)) if vals else 0.0


def _pes_over_subsets_np(expert_outs: np.ndarray, subsets: list[tuple[int, ...]], eps: float = 1e-12) -> float:
    if expert_outs.ndim != 3:
        return 0.0
    n, E, _ = expert_outs.shape
    if n == 0 or E < 2:
        return 0.0
    norms = np.linalg.norm(expert_outs, axis=-1, keepdims=True)
    v = expert_outs / (norms + eps)
    cos = np.matmul(v, np.swapaxes(v, 1, 2))  # [N,E,E]
    vals: list[float] = []
    for sub in subsets:
        k = len(sub)
        if k < 2:
            continue
        C = cos[np.ix_(np.arange(n), sub, sub)]  # [N,k,k]
        diag = np.diagonal(C, axis1=1, axis2=2).sum(axis=1)
        off_sum = C.sum(axis=(1, 2)) - diag
        per = off_sum / (k * (k - 1))
        vals.append(float(per.mean()))
    return float(np.mean(vals)) if vals else 0.0

def _make_data1_task_params(cfg: Data1Config, task_rng: np.random.Generator) -> Dict[str, object]:
    raw_d = cfg.input_dim
    d = _effective_input_dim(cfg)
    C = cfg.num_classes

    # --- single Gaussian parameters ---
    mean = task_rng.normal(scale=2.0, size=(raw_d,)).astype(np.float32)
    cov = _make_correlated_cov(raw_d, cfg.gmm_cov_scale, task_rng) if cfg.gmm_correlated else np.eye(raw_d) * cfg.gmm_cov_scale

    # --- regime gating parameters ---
    a = task_rng.uniform(0.5, 2.0, size=(raw_d,)).astype(np.float32)

    # --- regime classifier parameters (feature -> head) ---
    shared_weights = None
    feat_dim = max(1, cfg.regime_mlp_out)
    head_W = task_rng.normal(scale=1.0 / np.sqrt(feat_dim), size=(feat_dim, C)).astype(np.float32)
    head_b = task_rng.normal(scale=0.1, size=(C,)).astype(np.float32)
    if cfg.share_regime:
        shared_depth = cfg.shared_regime_mlp_depth if cfg.shared_regime_mlp_depth is not None else cfg.regime_mlp_depth
        shared_hid = cfg.shared_regime_mlp_hidden if cfg.shared_regime_mlp_hidden is not None else cfg.regime_mlp_hidden
        shared_out = cfg.shared_regime_mlp_out if cfg.shared_regime_mlp_out is not None else cfg.regime_mlp_out
        if int(shared_out) != int(cfg.regime_mlp_out):
            raise ValueError("shared_regime_mlp_out must match regime_mlp_out for feature fusion")
        depth = int(shared_depth)
        hid = int(shared_hid)
        layers = []
        if depth <= 1:
            W = task_rng.normal(scale=1.0 / np.sqrt(d), size=(d, feat_dim)).astype(np.float32)
            b = task_rng.normal(scale=0.1, size=(feat_dim,)).astype(np.float32)
            layers.append((W, b))
        else:
            in_dim = d
            for _ in range(depth - 1):
                W = task_rng.normal(scale=1.0 / np.sqrt(in_dim), size=(in_dim, hid)).astype(np.float32)
                b = task_rng.normal(scale=0.1, size=(hid,)).astype(np.float32)
                layers.append((W, b))
                in_dim = hid
            W = task_rng.normal(scale=1.0 / np.sqrt(in_dim), size=(in_dim, feat_dim)).astype(np.float32)
            b = task_rng.normal(scale=0.1, size=(feat_dim,)).astype(np.float32)
            layers.append((W, b))
        shared_weights = layers

    weights_by_r = _make_regime_mlp_weights(cfg, task_rng)

    return {
        "mean": mean,
        "cov": cov,
        "a": a,
        "shared_weights": shared_weights,
        "weights_by_r": weights_by_r,
        "head_W": head_W,
        "head_b": head_b,
        "feat_dim": feat_dim,
    }


def compute_true_regime_stats(
    cfg: Data1Config,
    seed: int,
    split: str = "train",
    max_tokens: int = 2048,
    subset_k: int = 4,
    subset_max_subsets: int = 0,
    subset_seed: int = 0,
) -> Dict[str, float]:
    x, _, regime = generate_data1(cfg, split, seed)
    task_seed, _data_seed, _norm_seed = _resolve_task_data_norm_seeds(cfg, split, seed)
    task_rng = np.random.default_rng(task_seed)
    params = _make_data1_task_params(cfg, task_rng)
    R = cfg.num_regimes
    if R <= 1:
        return {
            "entropy": 0.0,
            "gini": 0.0,
            "cka_offdiag_mean": 0.0,
            "pes": 0.0,
            "counts": [int(x.shape[0])],
            "x_has_nan": bool(np.isnan(x).any()),
            "x_near_constant_cols": 0,
        }

    counts = np.bincount(regime, minlength=R).astype(np.float64)
    entropy = _entropy_from_counts(counts)
    gini = _gini_from_counts(counts)
    x_has_nan = bool(np.isnan(x).any())
    col_std = np.std(x, axis=0)
    near_constant_cols = int(np.sum(col_std < 1e-6))

    # true PES and CKA using regime MLP outputs (full compute)
    shared_feat = None
    if params["shared_weights"] is not None:
        shared_feat = _mlp_forward_np(x, params["shared_weights"], cfg.regime_activation)
    feats_all = []
    for r in range(R):
        feat = _mlp_forward_np(x, params["weights_by_r"][r], cfg.regime_activation)
        if shared_feat is not None:
            feat = cfg.regime_specific_weight * feat + cfg.share_regime_weight * shared_feat
        feats_all.append(feat)
    expert_outs = np.stack(feats_all, axis=1)  # [N,R,D]

    feat_bank = {}
    for r in range(R):
        feats = expert_outs[:max_tokens, r, :]
        feat_bank[r] = torch.from_numpy(feats).to(torch.float32)

    cka_mat = pairwise_cka_matrix(feat_bank, num_experts=R)
    subsets = _choose_subsets_np(R, subset_k, subset_max_subsets, subset_seed)
    cka_offdiag = _cka_offdiag_mean_over_subsets_np(cka_mat, subsets=subsets)
    if not np.isfinite(cka_offdiag):
        cka_offdiag = 0.0

    pes = _pes_over_subsets_np(expert_outs, subsets=subsets)

    return {
        "entropy": entropy,
        "gini": gini,
        "cka_offdiag_mean": cka_offdiag,
        "pes": pes,
        "counts": counts.astype(int).tolist(),
        "x_has_nan": x_has_nan,
        "x_near_constant_cols": near_constant_cols,
    }


def compute_true_model_metrics(
    cfg: Data1Config,
    seed: int,
    split: str = "test",
) -> Dict[str, float]:
    """Evaluate the ground-truth generator on its own data."""
    x, y, regime = generate_data1(cfg, split, seed)
    task_seed, _data_seed, _norm_seed = _resolve_task_data_norm_seeds(cfg, split, seed)
    task_rng = np.random.default_rng(task_seed)
    params = _make_data1_task_params(cfg, task_rng)

    C = cfg.num_classes
    logit_scale = _data1_logit_scale(cfg) if C > 1 else 1.0
    logits = np.zeros((x.shape[0], C), dtype=np.float32)
    shared_feat = None
    if params["shared_weights"] is not None:
        shared_feat = _mlp_forward_np(x, params["shared_weights"], cfg.regime_activation)
    for r in range(cfg.num_regimes):
        idx = np.where(regime == r)[0]
        if idx.size == 0:
            continue
        feat = _mlp_forward_np(x[idx], params["weights_by_r"][r], cfg.regime_activation)
        if shared_feat is not None:
            feat = cfg.regime_specific_weight * feat + cfg.share_regime_weight * shared_feat[idx]
        logits[idx] = logit_scale * (feat @ params["head_W"] + params["head_b"])

    tlogits = torch.from_numpy(logits)
    ty = torch.from_numpy(y.astype(np.int64))
    loss = float(F.cross_entropy(tlogits, ty).item())
    pred = tlogits.argmax(dim=-1)
    acc = float((pred == ty).float().mean().item())

    cm = torch.zeros((C, C), dtype=torch.long)
    for t, p in zip(ty.view(-1), pred.view(-1)):
        cm[int(t), int(p)] += 1
    tp = cm.diag().float()
    fp = cm.sum(dim=0).float() - tp
    fn = cm.sum(dim=1).float() - tp
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-9)
    macro_f1 = float(f1.mean().item())

    return {
        "loss": loss,
        "micro_acc": acc,
        "macro_f1": macro_f1,
    }


def compute_true_model_probabilities(
    cfg: Data1Config,
    seed: int,
    split: str = "test",
) -> Dict[str, np.ndarray]:
    """Return true-generator class probabilities and true labels on a split."""
    x, y, regime = generate_data1(cfg, split, seed)
    task_seed, _data_seed, _norm_seed = _resolve_task_data_norm_seeds(cfg, split, seed)
    task_rng = np.random.default_rng(task_seed)
    params = _make_data1_task_params(cfg, task_rng)

    C = cfg.num_classes
    logit_scale = _data1_logit_scale(cfg) if C > 1 else 1.0
    logits = np.zeros((x.shape[0], C), dtype=np.float32)
    shared_feat = None
    if params["shared_weights"] is not None:
        shared_feat = _mlp_forward_np(x, params["shared_weights"], cfg.regime_activation)
    for r in range(cfg.num_regimes):
        idx = np.where(regime == r)[0]
        if idx.size == 0:
            continue
        feat = _mlp_forward_np(x[idx], params["weights_by_r"][r], cfg.regime_activation)
        if shared_feat is not None:
            feat = cfg.regime_specific_weight * feat + cfg.share_regime_weight * shared_feat[idx]
        logits[idx] = logit_scale * (feat @ params["head_W"] + params["head_b"])

    probs = torch.softmax(torch.from_numpy(logits), dim=-1).cpu().numpy().astype(np.float32)
    return {
        "probs": probs,
        "y_true": y.astype(np.int64),
    }


def compute_true_model_metrics_regression(
    cfg: Union[Data2Config, Data3Config],
    seed: int,
    split: str = "test",
) -> Dict[str, float]:
    """Evaluate the ground-truth generator on its own data (regression)."""
    if isinstance(cfg, Data3Config):
        x, y, _regime = generate_data3(cfg, split, seed)
        y_true = estimate_eta_data3_analytic_from_x(cfg, x, seed).astype(np.float32)
    else:
        x, y, regime = generate_data2(cfg, split, seed)
        y_true = _data1_logits_from_x_regime(cfg, x, regime, seed).astype(np.float32)
    diff = y_true - y.astype(np.float32)
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    mean_y = float(np.mean(y))
    sst = float(np.sum((y - mean_y) ** 2))
    sse = float(np.sum(diff * diff))
    r2 = 1.0 - (sse / sst) if sst > 1e-12 else 0.0
    return {"loss": mse, "mse": mse, "mae": mae, "r2": float(r2)}


def compute_true_regime_stats_data3(
    cfg: Data3Config,
    seed: int,
    split: str = "train",
    max_tokens: int = 2048,
    subset_k: int = 4,
    subset_max_subsets: int = 0,
    subset_seed: int = 0,
) -> Dict[str, float]:
    x, _y, regime = generate_data3(cfg, split, seed)
    R = cfg.num_regimes
    counts = np.bincount(regime, minlength=R).astype(np.float64)
    entropy = _entropy_from_counts(counts)
    gini = _gini_from_counts(counts)
    x_has_nan = bool(np.isnan(x).any())
    col_std = np.std(x, axis=0)
    near_constant_cols = int(np.sum(col_std < 1e-6))
    return {
        "entropy": entropy,
        "gini": gini,
        "cka_offdiag_mean": 0.0,
        "pes": 0.0,
        "counts": counts.astype(int).tolist(),
        "x_has_nan": x_has_nan,
        "x_near_constant_cols": near_constant_cols,
    }


def compute_label_regime_percent(
    cfg: Data1Config,
    seed: int,
    split: str = "train",
) -> np.ndarray:
    x, y, regime = generate_data1(cfg, split, seed)
    C = cfg.num_classes
    R = cfg.num_regimes
    counts = np.zeros((C, R), dtype=np.float64)
    for c in range(C):
        idx = (y == c)
        if idx.sum() == 0:
            continue
        counts[c] = np.bincount(regime[idx], minlength=R).astype(np.float64)
    totals = counts.sum(axis=1, keepdims=True)
    freq = counts / np.clip(totals, 1.0, None)
    return freq * 100.0


def generate_data1_with_seeds(
    cfg: Data1Config,
    split: str,
    task_seed: int,
    data_seed: int,
    norm_seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    IMPORTANT:
    - task_rng: controls the underlying task (functions, regimes)
    - data_rng: controls iid sampling (x, noise)
    - norm stats: computed from training split using norm_seed
    """
    assert split in ("train", "val", "test")
    n = {"train": cfg.n_train, "val": cfg.n_val, "test": cfg.n_test}[split]

    # ---- split RNGs ----
    task_rng = np.random.default_rng(int(task_seed))
    data_rng = np.random.default_rng(_resolve_data_seed(split, int(data_seed)))
    norm_mu, norm_sigma = _compute_train_norm_stats(
        cfg,
        int(norm_seed),
        task_seed=int(task_seed),
        data_seed=int(norm_seed),
    )

    raw_d = cfg.input_dim
    C = cfg.num_classes
    R = cfg.num_regimes

    # ===== Task parameters (task_rng only) =====

    params = _make_data1_task_params(cfg, task_rng)
    mean = params["mean"]
    cov = params["cov"]
    a = params["a"]
    shared_weights = params["shared_weights"]
    weights_by_r = params["weights_by_r"]
    head_W = params["head_W"]
    head_b = params["head_b"]
    logit_scale = _data1_logit_scale(cfg) if C > 1 else 1.0

    # ===== Sample data (data_rng only) =====

    # --- sample x from single Gaussian ---
    x = data_rng.multivariate_normal(mean, cov, size=n).astype(np.float32)

    # normalize
    x = (x - norm_mu) / (norm_sigma + 1e-6)
    if cfg.share_feature_dim and cfg.share_feature_dim > 0:
        sd = min(cfg.share_feature_dim, raw_d)
        idx = task_rng.choice(raw_d, size=sd, replace=False)
        shared_base = task_rng.normal(scale=1.0, size=(sd,)).astype(np.float32)
        noise = data_rng.normal(scale=cfg.share_feature_std, size=(n, sd)).astype(np.float32)
        x[:, idx] = shared_base + noise

    # --- latent regime assignment ---
    gate_type = getattr(cfg, "regime_gate_type", "phi")
    if cfg.input_mode == "x":
        if gate_type == "phi":
            # original: phi(x) = [x, x^2, sin(x*a)] @ A^T
            phi_parts = [x, x * x, np.sin(x * a)]
            phi = np.concatenate(phi_parts, axis=1)
            A = _make_regime_matrix(cfg, task_rng)
            gate_logits = phi @ A.T + data_rng.normal(scale=cfg.regime_noise_std, size=(n, R))
            regime = np.argmax(gate_logits, axis=1).astype(np.int64)
        else:
            # linear / nonlinear / softmax
            scores, gate_params = _compute_gate_scores(cfg, x, task_rng)
            regime = _assign_regime(cfg, scores, gate_params, data_rng)
        x_eff = x
    elif cfg.input_mode == "x_beta":
        if raw_d <= 1:
            xw = x
        else:
            cov = np.cov(x, rowvar=False, bias=True)
            eigvals, eigvecs = np.linalg.eigh(cov)
            inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals + 1e-6)) @ eigvecs.T
            xw = x @ inv_sqrt
        betas = task_rng.normal(size=(raw_d, R)).astype(np.float32)
        Q, _ = np.linalg.qr(betas)
        betas = Q[:, :R]
        scores = np.zeros((n, R), dtype=np.float32)
        for r in range(R):
            z = xw @ betas[:, r]
            a_r = float(np.dot(a, betas[:, r]))
            phi_r = np.stack([z, z * z, np.sin(z * a_r)], axis=1)
            w_r = task_rng.normal(scale=1.0, size=(3,)).astype(np.float32)
            scores[:, r] = phi_r @ w_r
        gate_logits = scores + data_rng.normal(scale=cfg.regime_noise_std, size=scores.shape)
        regime = np.argmax(gate_logits, axis=1).astype(np.int64)
        x_eff = np.zeros((n, 1), dtype=np.float32)
        for r in range(R):
            idx = np.where(regime == r)[0]
            if idx.size == 0:
                continue
            x_eff[idx, 0] = (xw[idx] @ betas[:, r]).astype(np.float32)
    elif cfg.input_mode == "x_B":
        m = int(cfg.proj_dim)
        if R * m > raw_d:
            raise ValueError("input_mode='x_B' requires num_regimes * proj_dim <= input_dim")
        Q, _ = np.linalg.qr(task_rng.normal(size=(raw_d, raw_d)))
        scores = np.zeros((n, R), dtype=np.float32)
        for r in range(R):
            B_r = Q[:, r * m:(r + 1) * m]
            z = x @ B_r
            a_r = B_r.T @ a
            phi_r = np.concatenate([z, z * z, np.sin(z * a_r)], axis=1)
            w_r = task_rng.normal(scale=1.0, size=(3 * m,)).astype(np.float32)
            scores[:, r] = phi_r @ w_r
        gate_logits = scores + data_rng.normal(scale=cfg.regime_noise_std, size=scores.shape)
        regime = np.argmax(gate_logits, axis=1).astype(np.int64)
        x_eff = np.zeros((n, m), dtype=np.float32)
        for r in range(R):
            idx = np.where(regime == r)[0]
            if idx.size == 0:
                continue
            B_r = Q[:, r * m:(r + 1) * m]
            x_eff[idx] = (x[idx] @ B_r).astype(np.float32)
    else:
        raise ValueError(cfg.input_mode)

    x = x_eff

    # --- compute logits (feature fusion then head) ---
    logits = np.zeros((n, C), dtype=np.float32)
    shared_feat = None
    if shared_weights is not None:
        shared_feat = _mlp_forward_np(x, shared_weights, cfg.regime_activation)
    for r in range(R):
        idx = np.where(regime == r)[0]
        if idx.size == 0:
            continue

        feat = _mlp_forward_np(x[idx], weights_by_r[r], cfg.regime_activation)
        if shared_feat is not None:
            feat = cfg.regime_specific_weight * feat + cfg.share_regime_weight * shared_feat[idx]
        logits[idx] = logit_scale * (feat @ head_W + head_b)

    logits += data_rng.normal(scale=cfg.logit_noise_std, size=logits.shape)
    y = np.argmax(logits, axis=1).astype(np.int64)

    # --- optional label flip ---
    if cfg.flip_prob > 0:
        flip = data_rng.random(size=n) < cfg.flip_prob
        y[flip] = data_rng.integers(0, C, size=int(flip.sum()))

    if cfg.regime_assign == "fixed":
        if cfg.num_regimes < cfg.num_classes:
            raise ValueError("regime_assign=fixed requires num_regimes >= num_classes")
        regime = y.copy()

    # --- spurious features ---
    if cfg.spurious_enabled and cfg.spurious_dim > 0:
        sdim = cfg.spurious_dim
        # Dedicated RNGs so spurious generation is independent of data_rng/task_rng state
        # and reproducible for any split without replaying the full pipeline.
        sp_noise_rng = np.random.default_rng(_resolve_data_seed(split, int(data_seed)) + 97531)
        sp_proj_rng = np.random.default_rng(int(data_seed) + 13579)
        sp = sp_noise_rng.normal(size=(n, sdim)).astype(np.float32)
        oh = np.eye(R, dtype=np.float32)[regime]
        W = sp_proj_rng.normal(scale=1.0, size=(R, sdim)).astype(np.float32)
        proj = oh @ W
        sp = (1.0 - cfg.spurious_corr) * sp + cfg.spurious_corr * proj
        # Normalize using training-split statistics to avoid split leakage.
        sp_mu, sp_sigma = _compute_train_spurious_norm_stats(
            cfg,
            int(norm_seed),
            task_seed=int(task_seed),
            data_seed=int(norm_seed),
        )
        sp = (sp - sp_mu) / (sp_sigma + 1e-6)
        x = np.concatenate([x, sp], axis=1).astype(np.float32)

    return x, y, regime


def generate_data1(
    cfg: Data1Config,
    split: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    IMPORTANT:
    - task_rng: controls the underlying task (functions, regimes)
    - data_rng: controls iid sampling (x, noise)
    Using the same seed guarantees the same task across splits.
    """
    task_seed, data_seed, norm_seed = _resolve_task_data_norm_seeds(cfg, split, seed)
    return generate_data1_with_seeds(cfg, split, task_seed, data_seed, norm_seed)


def generate_data1_raw_x(
    cfg: Data1Config,
    split: str,
    seed: int,
) -> np.ndarray:
    """Generate raw (pre-normalization) x for overlap checks."""
    assert split in ("train", "val", "test")
    n = {"train": cfg.n_train, "val": cfg.n_val, "test": cfg.n_test}[split]
    task_seed, data_seed, _norm_seed = _resolve_task_data_norm_seeds(cfg, split, seed)
    task_rng = np.random.default_rng(task_seed)
    data_rng = np.random.default_rng(_resolve_data_seed(split, data_seed))
    params = _make_data1_task_params(cfg, task_rng)
    mean = params["mean"]
    cov = params["cov"]
    x = data_rng.multivariate_normal(mean, cov, size=n).astype(np.float32)
    return x


def generate_data1_raw_xy(
    cfg: Data1Config,
    split: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate raw x plus (y, regime) produced by normalized x pipeline."""
    assert split in ("train", "val", "test")
    n = {"train": cfg.n_train, "val": cfg.n_val, "test": cfg.n_test}[split]

    # ---- split RNGs ----
    task_seed, data_seed, norm_seed = _resolve_task_data_norm_seeds(cfg, split, seed)
    task_rng = np.random.default_rng(task_seed)
    data_rng = np.random.default_rng(_resolve_data_seed(split, data_seed))
    norm_mu, norm_sigma = _compute_train_norm_stats(
        cfg,
        norm_seed,
        task_seed=task_seed,
        data_seed=norm_seed,
    )

    raw_d = cfg.input_dim
    C = cfg.num_classes
    R = cfg.num_regimes

    params = _make_data1_task_params(cfg, task_rng)
    mean = params["mean"]
    cov = params["cov"]
    a = params["a"]
    shared_weights = params["shared_weights"]
    weights_by_r = params["weights_by_r"]
    head_W = params["head_W"]
    head_b = params["head_b"]
    logit_scale = _data1_logit_scale(cfg) if C > 1 else 1.0

    # --- sample raw x from single Gaussian ---
    raw_x = data_rng.multivariate_normal(mean, cov, size=n).astype(np.float32)

    # normalize (same as generate_data1)
    x = (raw_x - norm_mu) / (norm_sigma + 1e-6)
    if cfg.share_feature_dim and cfg.share_feature_dim > 0:
        sd = min(cfg.share_feature_dim, raw_d)
        idx = task_rng.choice(raw_d, size=sd, replace=False)
        shared_base = task_rng.normal(scale=1.0, size=(sd,)).astype(np.float32)
        noise = data_rng.normal(scale=cfg.share_feature_std, size=(n, sd)).astype(np.float32)
        x[:, idx] = shared_base + noise

    # --- latent regime assignment ---
    gate_type = getattr(cfg, "regime_gate_type", "phi")
    if cfg.input_mode == "x":
        if gate_type == "phi":
            phi_parts = [x, x * x, np.sin(x * a)]
            phi = np.concatenate(phi_parts, axis=1)
            A = _make_regime_matrix(cfg, task_rng)
            gate_logits = phi @ A.T + data_rng.normal(scale=cfg.regime_noise_std, size=(n, R))
            regime = np.argmax(gate_logits, axis=1).astype(np.int64)
        else:
            scores, gate_params = _compute_gate_scores(cfg, x, task_rng)
            regime = _assign_regime(cfg, scores, gate_params, data_rng)
        x_eff = x
    elif cfg.input_mode == "x_beta":
        if raw_d <= 1:
            xw = x
        else:
            cov_x = np.cov(x, rowvar=False, bias=True)
            eigvals, eigvecs = np.linalg.eigh(cov_x)
            inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals + 1e-6)) @ eigvecs.T
            xw = x @ inv_sqrt
        betas = task_rng.normal(size=(raw_d, R)).astype(np.float32)
        Q, _ = np.linalg.qr(betas)
        betas = Q[:, :R]
        scores = np.zeros((n, R), dtype=np.float32)
        for r in range(R):
            z = xw @ betas[:, r]
            a_r = float(np.dot(a, betas[:, r]))
            phi_r = np.stack([z, z * z, np.sin(z * a_r)], axis=1)
            w_r = task_rng.normal(scale=1.0, size=(3,)).astype(np.float32)
            scores[:, r] = phi_r @ w_r
        gate_logits = scores + data_rng.normal(scale=cfg.regime_noise_std, size=scores.shape)
        regime = np.argmax(gate_logits, axis=1).astype(np.int64)
        x_eff = np.zeros((n, 1), dtype=np.float32)
        for r in range(R):
            idx = np.where(regime == r)[0]
            if idx.size == 0:
                continue
            x_eff[idx, 0] = (xw[idx] @ betas[:, r]).astype(np.float32)
    elif cfg.input_mode == "x_B":
        m = int(cfg.proj_dim)
        if R * m > raw_d:
            raise ValueError("input_mode='x_B' requires num_regimes * proj_dim <= input_dim")
        Q, _ = np.linalg.qr(task_rng.normal(size=(raw_d, raw_d)))
        scores = np.zeros((n, R), dtype=np.float32)
        for r in range(R):
            B_r = Q[:, r * m:(r + 1) * m]
            z = x @ B_r
            a_r = B_r.T @ a
            phi_r = np.concatenate([z, z * z, np.sin(z * a_r)], axis=1)
            w_r = task_rng.normal(scale=1.0, size=(3 * m,)).astype(np.float32)
            scores[:, r] = phi_r @ w_r
        gate_logits = scores + data_rng.normal(scale=cfg.regime_noise_std, size=scores.shape)
        regime = np.argmax(gate_logits, axis=1).astype(np.int64)
        x_eff = np.zeros((n, m), dtype=np.float32)
        for r in range(R):
            idx = np.where(regime == r)[0]
            if idx.size == 0:
                continue
            B_r = Q[:, r * m:(r + 1) * m]
            x_eff[idx] = (x[idx] @ B_r).astype(np.float32)
    else:
        raise ValueError(cfg.input_mode)

    x = x_eff

    # --- compute logits (feature fusion then head) ---
    logits = np.zeros((n, C), dtype=np.float32)
    shared_feat = None
    if shared_weights is not None:
        shared_feat = _mlp_forward_np(x, shared_weights, cfg.regime_activation)
    for r in range(R):
        idx = np.where(regime == r)[0]
        if idx.size == 0:
            continue

        feat = _mlp_forward_np(x[idx], weights_by_r[r], cfg.regime_activation)
        if shared_feat is not None:
            feat = cfg.regime_specific_weight * feat + cfg.share_regime_weight * shared_feat[idx]
        logits[idx] = logit_scale * (feat @ head_W + head_b)

    logits += data_rng.normal(scale=cfg.logit_noise_std, size=logits.shape)
    y = np.argmax(logits, axis=1).astype(np.int64)

    # --- optional label flip ---
    if cfg.flip_prob > 0:
        flip = data_rng.random(size=n) < cfg.flip_prob
        y[flip] = data_rng.integers(0, C, size=int(flip.sum()))

    if cfg.regime_assign == "fixed":
        if cfg.num_regimes < cfg.num_classes:
            raise ValueError("regime_assign=fixed requires num_regimes >= num_classes")
        regime = y.copy()

    # --- spurious features (skip: raw x is pre-normalization of base features only) ---
    return raw_x, y, regime


def _compute_train_spurious_norm_stats(
    cfg: "Data1Config",
    seed: int,
    task_seed: int | None = None,
    data_seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean/std of training-split spurious features for consistent cross-split normalization.

    Uses dedicated RNGs (independent of data_rng / task_rng) so the result can be reproduced
    without replaying the full generate_data1 RNG sequence.
    """
    from dataclasses import replace as _dc_replace
    task_seed = int(seed if task_seed is None else task_seed)
    data_seed = int(seed if data_seed is None else data_seed)
    # Generate training data WITHOUT spurious features to obtain regime assignments.
    cfg_no_sp = _dc_replace(cfg, spurious_enabled=False)
    _, _, regime_train = generate_data1_with_seeds(cfg_no_sp, "train", task_seed, data_seed, data_seed)

    n_train = int(cfg.n_train)
    sdim = int(cfg.spurious_dim)
    R = int(cfg.num_regimes)

    # Dedicated RNGs — identical to those used inside generate_data1's spurious block.
    sp_noise_rng = np.random.default_rng(_resolve_data_seed("train", data_seed) + 97531)
    sp_proj_rng = np.random.default_rng(int(data_seed) + 13579)

    sp_noise = sp_noise_rng.normal(size=(n_train, sdim)).astype(np.float32)
    W = sp_proj_rng.normal(scale=1.0, size=(R, sdim)).astype(np.float32)
    oh = np.eye(R, dtype=np.float32)[regime_train]
    proj = oh @ W
    sp_train = (1.0 - float(cfg.spurious_corr)) * sp_noise + float(cfg.spurious_corr) * proj
    return sp_train.mean(0, keepdims=True), sp_train.std(0, keepdims=True)


def estimate_eta_data1_mc(
    cfg: Data1Config,
    split: str,
    seed: int,
    mc_samples: int,
) -> np.ndarray:
    """Estimate eta(x)=P(y|x) via Monte-Carlo for data1 with fixed x per split."""
    assert split in ("train", "val", "test")
    if mc_samples <= 0:
        raise ValueError("mc_samples must be > 0")

    n = {"train": cfg.n_train, "val": cfg.n_val, "test": cfg.n_test}[split]

    task_seed, data_seed, norm_seed = _resolve_task_data_norm_seeds(cfg, split, seed)
    task_rng = np.random.default_rng(task_seed)
    data_rng = np.random.default_rng(_resolve_data_seed(split, data_seed))
    mc_rng = np.random.default_rng(task_seed + 2)
    norm_mu, norm_sigma = _compute_train_norm_stats(
        cfg,
        norm_seed,
        task_seed=task_seed,
        data_seed=norm_seed,
    )

    raw_d = cfg.input_dim
    C = cfg.num_classes
    R = cfg.num_regimes

    params = _make_data1_task_params(cfg, task_rng)
    mean = params["mean"]
    cov = params["cov"]
    a = params["a"]
    shared_weights = params["shared_weights"]
    weights_by_r = params["weights_by_r"]
    head_W = params["head_W"]
    head_b = params["head_b"]
    logit_scale = _data1_logit_scale(cfg) if C > 1 else 1.0

    # --- sample x (same as generate_data1) ---
    x = data_rng.multivariate_normal(mean, cov, size=n).astype(np.float32)
    x = (x - norm_mu) / (norm_sigma + 1e-6)
    if cfg.share_feature_dim and cfg.share_feature_dim > 0:
        sd = min(cfg.share_feature_dim, raw_d)
        idx = task_rng.choice(raw_d, size=sd, replace=False)
        shared_base = task_rng.normal(scale=1.0, size=(sd,)).astype(np.float32)
        noise = data_rng.normal(scale=cfg.share_feature_std, size=(n, sd)).astype(np.float32)
        x[:, idx] = shared_base + noise

    # --- precompute deterministic scores and x_eff by regime ---
    scores = None
    gate_params = None
    x_eff_by_r = []
    gate_type = getattr(cfg, "regime_gate_type", "phi")
    if cfg.input_mode == "x":
        if gate_type == "phi":
            phi_parts = [x, x * x, np.sin(x * a)]
            phi = np.concatenate(phi_parts, axis=1)
            A = _make_regime_matrix(cfg, task_rng)
            scores = phi @ A.T
            gate_params = {"type": "phi"}
        else:
            scores, gate_params = _compute_gate_scores(cfg, x, task_rng)
        x_eff_by_r = [x for _ in range(R)]
    elif cfg.input_mode == "x_beta":
        if raw_d <= 1:
            xw = x
        else:
            cov_x = np.cov(x, rowvar=False, bias=True)
            eigvals, eigvecs = np.linalg.eigh(cov_x)
            inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals + 1e-6)) @ eigvecs.T
            xw = x @ inv_sqrt
        betas = task_rng.normal(size=(raw_d, R)).astype(np.float32)
        Q, _ = np.linalg.qr(betas)
        betas = Q[:, :R]
        scores = np.zeros((n, R), dtype=np.float32)
        for r in range(R):
            z = xw @ betas[:, r]
            a_r = float(np.dot(a, betas[:, r]))
            phi_r = np.stack([z, z * z, np.sin(z * a_r)], axis=1)
            w_r = task_rng.normal(scale=1.0, size=(3,)).astype(np.float32)
            scores[:, r] = phi_r @ w_r
            x_eff_by_r.append(z[:, None].astype(np.float32))
        gate_params = {"type": "phi"}
    elif cfg.input_mode == "x_B":
        m = int(cfg.proj_dim)
        if R * m > raw_d:
            raise ValueError("input_mode='x_B' requires num_regimes * proj_dim <= input_dim")
        Q, _ = np.linalg.qr(task_rng.normal(size=(raw_d, raw_d)))
        scores = np.zeros((n, R), dtype=np.float32)
        for r in range(R):
            B_r = Q[:, r * m:(r + 1) * m]
            z = x @ B_r
            a_r = B_r.T @ a
            phi_r = np.concatenate([z, z * z, np.sin(z * a_r)], axis=1)
            w_r = task_rng.normal(scale=1.0, size=(3 * m,)).astype(np.float32)
            scores[:, r] = phi_r @ w_r
            x_eff_by_r.append(z.astype(np.float32))
        gate_params = {"type": "phi"}
    else:
        raise ValueError(cfg.input_mode)

    eta_counts = np.zeros((n, C), dtype=np.float64)
    for _ in range(int(mc_samples)):
        regime = _assign_regime(cfg, scores, gate_params, mc_rng)

        logits = np.zeros((n, C), dtype=np.float32)
        for r in range(R):
            idx = np.where(regime == r)[0]
            if idx.size == 0:
                continue
            xe = x_eff_by_r[r][idx]
            feat = _mlp_forward_np(xe, weights_by_r[r], cfg.regime_activation)
            if shared_weights is not None:
                shared_feat = _mlp_forward_np(xe, shared_weights, cfg.regime_activation)
                feat = cfg.regime_specific_weight * feat + cfg.share_regime_weight * shared_feat
            logits[idx] = logit_scale * (feat @ head_W + head_b)

        if cfg.logit_noise_std > 0:
            logits += mc_rng.normal(scale=cfg.logit_noise_std, size=logits.shape).astype(np.float32)
        y = np.argmax(logits, axis=1).astype(np.int64)

        if cfg.flip_prob > 0:
            flip = mc_rng.random(size=n) < cfg.flip_prob
            y[flip] = mc_rng.integers(0, C, size=int(flip.sum()))

        eta_counts[np.arange(n), y] += 1.0

    eta = (eta_counts / float(mc_samples)).astype(np.float32)
    return eta


def estimate_eta_data1_mc_from_x(
    cfg: Data1Config,
    x: np.ndarray,
    seed: int,
    mc_samples: int,
) -> np.ndarray:
    """Estimate eta(x)=P(y|x) via Monte-Carlo for given x (already normalized)."""
    if mc_samples <= 0:
        raise ValueError("mc_samples must be > 0")
    if getattr(cfg, "input_mode", "x") != "x":
        raise ValueError("estimate_eta_data1_mc_from_x only supports input_mode='x'")

    task_seed = int(cfg.seed) if getattr(cfg, "fixed_test_set", False) else int(seed)
    task_rng = np.random.default_rng(task_seed)
    mc_rng = np.random.default_rng(task_seed + 2)

    C = cfg.num_classes
    R = cfg.num_regimes

    params = _make_data1_task_params(cfg, task_rng)
    a = params["a"]
    shared_weights = params["shared_weights"]
    weights_by_r = params["weights_by_r"]
    head_W = params["head_W"]
    head_b = params["head_b"]
    logit_scale = _data1_logit_scale(cfg) if C > 1 else 1.0

    # ignore spurious features if present
    core_dim = _effective_input_dim(cfg)
    x_core = x[:, :core_dim].astype(np.float32)
    n = x_core.shape[0]

    gate_type = getattr(cfg, "regime_gate_type", "phi")
    if gate_type == "phi":
        phi_parts = [x_core, x_core * x_core, np.sin(x_core * a)]
        phi = np.concatenate(phi_parts, axis=1)
        A = _make_regime_matrix(cfg, task_rng)
        scores = phi @ A.T
        gate_params = {"type": "phi"}
    else:
        scores, gate_params = _compute_gate_scores(cfg, x_core, task_rng)

    eta_counts = np.zeros((n, C), dtype=np.float64)
    for _ in range(int(mc_samples)):
        regime = _assign_regime(cfg, scores, gate_params, mc_rng)

        logits = np.zeros((n, C), dtype=np.float32)
        for r in range(R):
            idx = np.where(regime == r)[0]
            if idx.size == 0:
                continue
            feat = _mlp_forward_np(x_core[idx], weights_by_r[r], cfg.regime_activation)
            if shared_weights is not None:
                shared_feat = _mlp_forward_np(x_core[idx], shared_weights, cfg.regime_activation)
                feat = cfg.regime_specific_weight * feat + cfg.share_regime_weight * shared_feat
            logits[idx] = logit_scale * (feat @ head_W + head_b)

        if cfg.logit_noise_std > 0:
            logits += mc_rng.normal(scale=cfg.logit_noise_std, size=logits.shape).astype(np.float32)
        y = np.argmax(logits, axis=1).astype(np.int64)

        if cfg.flip_prob > 0:
            flip = mc_rng.random(size=n) < cfg.flip_prob
            y[flip] = mc_rng.integers(0, C, size=int(flip.sum()))

        eta_counts[np.arange(n), y] += 1.0

    eta = (eta_counts / float(mc_samples)).astype(np.float32)
    return eta


def estimate_eta_data2_analytic_from_x(
    cfg: Data2Config,
    x: np.ndarray,
    regime: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Analytic eta(x) for data2 regression: noise-free logits given fixed regime."""
    return _data1_logits_from_x_regime(cfg, x, regime, seed).astype(np.float32)


def generate_data2(
    cfg: Data2Config,
    split: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert split in ("train", "val", "test")
    if cfg.regime_assign == "fixed":
        raise ValueError("data2 regression does not support regime_assign='fixed'")
    if cfg.flip_prob > 0:
        raise ValueError("data2 regression does not support flip_prob > 0")

    # For data2:
    # - When fixed_test_set=True, keep task fixed across reps and fix test raw x;
    #   normalization follows each rep's train stats.
    # - Otherwise, follow legacy behavior (task/data/norm all from rep seed).
    task_seed, data_seed, norm_seed = _resolve_task_data_norm_seeds(cfg, split, seed)
    # reuse data1 generator for x/regime (all params identical), then return continuous target
    x, _y_cls, regime = generate_data1_with_seeds(cfg, split, task_seed, data_seed, norm_seed)
    logits = _data1_logits_from_x_regime(cfg, x, regime, task_seed)
    if cfg.logit_noise_std > 0:
        data_rng = np.random.default_rng(_resolve_data_seed(split, data_seed) + 1)
        logits = logits + data_rng.normal(scale=cfg.logit_noise_std, size=logits.shape).astype(np.float32)
    y = logits.astype(np.float32)
    return x, y, regime


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    z = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def _data3_difficulty_coeffs(cfg: Data3Config) -> dict:
    diff = str(getattr(cfg, "data3_difficulty", "mid")).lower()
    if diff == "diffcult":
        diff = "difficult"
    if diff == "mid":
        return {
            "bias": float(cfg.d3_mid_bias),
            "u1_coef": float(cfg.d3_mid_u1_coef),
            "u2_coef": float(cfg.d3_mid_u2_coef),
            "hinge1_coef": float(cfg.d3_mid_hinge1_coef),
            "hinge2_coef": float(cfg.d3_mid_hinge2_coef),
            "hinge3_coef": float(cfg.d3_mid_hinge3_coef),
            "u1u2_coef": float(cfg.d3_mid_u1u2_coef),
            "u1_sq_coef": float(cfg.d3_mid_u1_sq_coef),
            "u2_sq_coef": float(cfg.d3_mid_u2_sq_coef),
            "sin_u1_coef": float(cfg.d3_mid_sin_u1_coef),
            "cos_u2_coef": float(cfg.d3_mid_cos_u2_coef),
            "cos_u1u2_coef": float(cfg.d3_mid_cos_u1u2_coef),
            "sin_freq": float(cfg.d3_mid_sin_freq),
            "cos_freq": float(cfg.d3_mid_cos_freq),
            "b1_u1_coef": float(cfg.d3_mid_b1_u1_coef),
            "b1_bias": float(cfg.d3_mid_b1_bias),
            "b2_u1_coef": float(cfg.d3_mid_b2_u1_coef),
            "b2_bias": float(cfg.d3_mid_b2_bias),
        }
    if diff == "difficult":
        return {
            "bias": float(cfg.d3_diff_bias),
            "u1_coef": float(cfg.d3_diff_u1_coef),
            "u2_coef": float(cfg.d3_diff_u2_coef),
            "hinge1_coef": float(cfg.d3_diff_hinge1_coef),
            "hinge2_coef": float(cfg.d3_diff_hinge2_coef),
            "hinge3_coef": float(cfg.d3_diff_hinge3_coef),
            "u1u2_coef": float(cfg.d3_diff_u1u2_coef),
            "u1_sq_coef": float(cfg.d3_diff_u1_sq_coef),
            "u2_sq_coef": float(cfg.d3_diff_u2_sq_coef),
            "sin_u1_coef": float(cfg.d3_diff_sin_u1_coef),
            "cos_u2_coef": float(cfg.d3_diff_cos_u2_coef),
            "cos_u1u2_coef": float(cfg.d3_diff_cos_u1u2_coef),
            "sin_freq": float(cfg.d3_diff_sin_freq),
            "cos_freq": float(cfg.d3_diff_cos_freq),
            "b1_u1_coef": float(cfg.d3_diff_b1_u1_coef),
            "b1_bias": float(cfg.d3_diff_b1_bias),
            "b2_u1_coef": float(cfg.d3_diff_b2_u1_coef),
            "b2_bias": float(cfg.d3_diff_b2_bias),
        }
    raise ValueError(f"Unsupported data3_difficulty={getattr(cfg, 'data3_difficulty', None)!r}; expected 'mid' or 'difficult'")


def _data3_piecewise_from_x(
    cfg: Data3Config,
    x: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (noise-free y, regime) from x for data3 nonlinear generator."""
    d = int(x.shape[1])
    task_seed = int(cfg.seed) if getattr(cfg, "fixed_test_set", False) else int(seed)
    task_rng = np.random.default_rng(task_seed + 2027)

    p1 = task_rng.normal(size=(d,)).astype(np.float32)
    p2 = task_rng.normal(size=(d,)).astype(np.float32)
    p1 /= float(np.linalg.norm(p1) + 1e-9)
    p2 /= float(np.linalg.norm(p2) + 1e-9)

    T = max(float(cfg.proj_temperature), 1e-6)
    u1 = _sigmoid_np((x @ p1) / T)
    u2 = _sigmoid_np((x @ p2) / T)
    coeffs = _data3_difficulty_coeffs(cfg)

    b1 = u2 + coeffs["b1_u1_coef"] * u1 + coeffs["b1_bias"]
    b2 = u2 + coeffs["b2_u1_coef"] * u1 + coeffs["b2_bias"]
    sin_term = np.sin(2.0 * np.pi * coeffs["sin_freq"] * u1)
    cos_term = np.cos(2.0 * np.pi * coeffs["cos_freq"] * u2)
    cos_mix = np.cos(2.0 * np.pi * coeffs["cos_freq"] * (u1 - u2))

    y_clean = (
        coeffs["bias"]
        + coeffs["u1_coef"] * u1
        + coeffs["u2_coef"] * u2
        + coeffs["hinge1_coef"] * np.maximum(0.0, -b1)
        + coeffs["hinge2_coef"] * np.maximum(0.0, -b2)
        + coeffs["hinge3_coef"] * np.maximum(0.0, u1 - 0.55)
        + coeffs["u1u2_coef"] * (u1 * u2)
        + coeffs["u1_sq_coef"] * (u1 * u1)
        + coeffs["u2_sq_coef"] * (u2 * u2)
        + coeffs["sin_u1_coef"] * sin_term
        + coeffs["cos_u2_coef"] * cos_term
        + coeffs["cos_u1u2_coef"] * cos_mix
    )
    y_clean = y_clean.astype(np.float32).reshape(-1, 1)

    region4 = (((b1 < 0).astype(np.int64) << 1) | (b2 < 0).astype(np.int64)).astype(np.int64)
    R = int(cfg.num_regimes)
    if R <= 0:
        raise ValueError("data3 requires num_regimes > 0")
    if R <= 4:
        regime = region4 % R
    else:
        fine = np.floor(np.clip(u1, 0.0, 0.999999) * R).astype(np.int64)
        regime = (region4 + fine) % R
    return y_clean, regime.astype(np.int64)


def _compute_train_spurious_norm_stats_data3(
    cfg: Data3Config,
    seed: int,
    task_seed: int | None = None,
    data_seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    from dataclasses import replace as _dc_replace

    task_seed = int(seed if task_seed is None else task_seed)
    data_seed = int(seed if data_seed is None else data_seed)
    cfg_no_sp = _dc_replace(cfg, spurious_enabled=False)
    _x_train, _y_train, regime_train = generate_data3_with_seeds(cfg_no_sp, "train", task_seed, data_seed, data_seed)

    n_train = int(cfg.n_train)
    sdim = int(cfg.spurious_dim)
    R = int(cfg.num_regimes)

    sp_noise_rng = np.random.default_rng(_resolve_data_seed("train", data_seed) + 97531)
    sp_proj_rng = np.random.default_rng(int(data_seed) + 13579)

    sp_noise = sp_noise_rng.normal(size=(n_train, sdim)).astype(np.float32)
    W = sp_proj_rng.normal(scale=1.0, size=(R, sdim)).astype(np.float32)
    oh = np.eye(R, dtype=np.float32)[regime_train]
    proj = oh @ W
    sp_train = (1.0 - float(cfg.spurious_corr)) * sp_noise + float(cfg.spurious_corr) * proj
    return sp_train.mean(0, keepdims=True), sp_train.std(0, keepdims=True)


def generate_data3_with_seeds(
    cfg: Data3Config,
    split: str,
    task_seed: int,
    data_seed: int,
    norm_seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert split in ("train", "val", "test")

    from dataclasses import replace as _dc_replace

    # Reuse data1's x-generation pipeline to keep split/seed/normalization behavior aligned.
    cfg_no_sp = _dc_replace(cfg, spurious_enabled=False)
    x, _y_tmp, _regime_tmp = generate_data1_with_seeds(
        cfg_no_sp,
        split,
        int(task_seed),
        int(data_seed),
        int(norm_seed),
    )
    y_clean, regime = _data3_piecewise_from_x(cfg, x, int(task_seed))

    if float(cfg.target_noise_std) > 0:
        data_rng = np.random.default_rng(_resolve_data_seed(split, int(data_seed)) + 1)
        y = y_clean + data_rng.normal(scale=float(cfg.target_noise_std), size=y_clean.shape).astype(np.float32)
    else:
        y = y_clean

    if cfg.spurious_enabled and cfg.spurious_dim > 0:
        sdim = int(cfg.spurious_dim)
        R = int(cfg.num_regimes)
        sp_noise_rng = np.random.default_rng(_resolve_data_seed(split, int(data_seed)) + 97531)
        sp_proj_rng = np.random.default_rng(int(data_seed) + 13579)
        sp = sp_noise_rng.normal(size=(x.shape[0], sdim)).astype(np.float32)
        W = sp_proj_rng.normal(scale=1.0, size=(R, sdim)).astype(np.float32)
        oh = np.eye(R, dtype=np.float32)[regime]
        proj = oh @ W
        sp = (1.0 - float(cfg.spurious_corr)) * sp + float(cfg.spurious_corr) * proj
        sp_mu, sp_sigma = _compute_train_spurious_norm_stats_data3(
            cfg,
            int(norm_seed),
            task_seed=int(task_seed),
            data_seed=int(norm_seed),
        )
        sp = (sp - sp_mu) / (sp_sigma + 1e-6)
        x = np.concatenate([x, sp], axis=1).astype(np.float32)

    return x.astype(np.float32), y.astype(np.float32), regime.astype(np.int64)


def generate_data3(
    cfg: Data3Config,
    split: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert split in ("train", "val", "test")
    task_seed, data_seed, norm_seed = _resolve_task_data_norm_seeds(cfg, split, seed)
    return generate_data3_with_seeds(cfg, split, task_seed, data_seed, norm_seed)


def estimate_eta_data3_analytic_from_x(
    cfg: Data3Config,
    x: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Analytic eta(x) for data3 regression: noise-free piecewise target."""
    y_clean, _regime = _data3_piecewise_from_x(cfg, x, seed)
    return y_clean.astype(np.float32)




class Data1Dataset(Dataset):
    def __init__(self, cfg: Data1Config, split: str, seed: int):
        self.x, self.y, self.regime = generate_data1(cfg, split, seed)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.x[idx]),
            torch.tensor(self.y[idx], dtype=torch.long),
            torch.tensor(self.regime[idx], dtype=torch.long),
        )


class Data2Dataset(Dataset):
    def __init__(self, cfg: Data2Config, split: str, seed: int):
        self.x, self.y, self.regime = generate_data2(cfg, split, seed)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.x[idx]),
            torch.from_numpy(self.y[idx]).to(torch.float32),
            torch.tensor(self.regime[idx], dtype=torch.long),
        )


class Data3Dataset(Dataset):
    def __init__(self, cfg: Data3Config, split: str, seed: int):
        self.x, self.y, self.regime = generate_data3(cfg, split, seed)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.x[idx]),
            torch.from_numpy(self.y[idx]).to(torch.float32),
            torch.tensor(self.regime[idx], dtype=torch.long),
        )


def make_loaders(
    cfg: Union[Data1Config, Data2Config, Data3Config],
    batch_size: int,
    num_workers: int,
    seed: int,
    data_variant: str,
):
    if data_variant == "data2":
        train_ds = Data2Dataset(cfg, "train", seed)
        val_ds = Data2Dataset(cfg, "val", seed)
        test_ds = Data2Dataset(cfg, "test", seed)
    elif data_variant == "data3":
        train_ds = Data3Dataset(cfg, "train", seed)
        val_ds = Data3Dataset(cfg, "val", seed)
        test_ds = Data3Dataset(cfg, "test", seed)
    else:
        train_ds = Data1Dataset(cfg, "train", seed)
        val_ds = Data1Dataset(cfg, "val", seed)
        test_ds = Data1Dataset(cfg, "test", seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )
    return train_loader, val_loader, test_loader


def _data1_logit_scale(cfg: Data1Config) -> float:
    return max(1e-6, float(getattr(cfg, "data1_logit_scale", 1.0)))


def _effective_input_dim(cfg: Data1Config) -> int:
    if getattr(cfg, "input_mode", "x") == "x_beta":
        return 1
    if getattr(cfg, "input_mode", "x") == "x_B":
        return int(cfg.proj_dim)
    return cfg.input_dim
