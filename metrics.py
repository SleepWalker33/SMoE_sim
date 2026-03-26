# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Tuple
import torch
import torch.nn.functional as F


@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return (pred == y).float().mean().item()


@torch.no_grad()
def cross_entropy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return F.cross_entropy(logits, y).item()


@torch.no_grad()
def confusion_matrix(pred: torch.Tensor, y: torch.Tensor, num_classes: int) -> torch.Tensor:
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for t, p in zip(y.view(-1), pred.view(-1)):
        cm[int(t), int(p)] += 1
    return cm


@torch.no_grad()
def per_class_accuracy(cm: torch.Tensor) -> torch.Tensor:
    # cm: [C,C], row=true, col=pred
    denom = cm.sum(dim=1).clamp_min(1)
    return cm.diag().float() / denom.float()


@torch.no_grad()
def macro_f1_from_cm(cm: torch.Tensor) -> float:
    # precision_i = tp/(tp+fp), recall_i=tp/(tp+fn)
    tp = cm.diag().float()
    fp = cm.sum(dim=0).float() - tp
    fn = cm.sum(dim=1).float() - tp

    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-9)
    return f1.mean().item()


@torch.no_grad()
def expert_usage_entropy(dispatch_mask: torch.Tensor) -> float:
    load = dispatch_mask.float().sum(dim=0)
    p = load / (load.sum() + 1e-9)
    ent = -(p * (p + 1e-9).log()).sum()
    return ent.item()


@torch.no_grad()
def expert_load_gini(dispatch_mask: torch.Tensor) -> float:
    load = dispatch_mask.float().sum(dim=0)
    sorted_load, _ = torch.sort(load)
    n = load.numel()
    idx = torch.arange(1, n + 1, device=load.device, dtype=load.dtype)
    g = (2 * (idx * sorted_load).sum() / (n * sorted_load.sum() + 1e-9)) - (n + 1) / n
    return g.item()


@torch.no_grad()
def mutual_info_expert_regime(dispatch_mask: torch.Tensor, regime: torch.Tensor) -> float:
    B, E = dispatch_mask.shape
    primary = dispatch_mask.float().argmax(dim=-1)
    r = regime
    R = int(r.max().item() + 1)
    joint = torch.zeros((E, R), device=dispatch_mask.device, dtype=torch.float32)
    for e in range(E):
        for rr in range(R):
            joint[e, rr] = ((primary == e) & (r == rr)).float().sum()
    joint = joint / (joint.sum() + 1e-9)
    pe = joint.sum(dim=1, keepdim=True)
    pr = joint.sum(dim=0, keepdim=True)
    mi = (joint * (joint / (pe @ pr + 1e-9) + 1e-9).log()).sum()
    return mi.item()


@torch.no_grad()
def accumulate_label_expert_counts(
    counts: torch.Tensor,
    totals: torch.Tensor,
    y: torch.Tensor,
    dispatch_mask: torch.Tensor,
    num_labels: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    for c in range(num_labels):
        idx = (y == c)
        n_c = idx.sum().item()
        if n_c == 0:
            continue
        totals[c] += float(n_c)
        counts[c] += dispatch_mask[idx].float().sum(dim=0)
    return counts, totals


@torch.no_grad()
def label_expert_frequencies(counts: torch.Tensor, totals: torch.Tensor) -> torch.Tensor:
    return counts / totals.unsqueeze(-1).clamp_min(1.0)


@torch.no_grad()
def accumulate_regime_expert_counts(
    counts: torch.Tensor,
    totals: torch.Tensor,
    regime: torch.Tensor,
    dispatch_mask: torch.Tensor,
    num_regimes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    for r in range(num_regimes):
        idx = (regime == r)
        n_r = idx.sum().item()
        if n_r == 0:
            continue
        totals[r] += float(n_r)
        counts[r] += dispatch_mask[idx].float().sum(dim=0)
    return counts, totals


@torch.no_grad()
def regime_expert_frequencies(counts: torch.Tensor, totals: torch.Tensor) -> torch.Tensor:
    return counts / totals.unsqueeze(-1).clamp_min(1.0)