# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import math
import torch


@dataclass
class RoutingResult:
    mask: torch.Tensor        # [B,E] bool
    weights: torch.Tensor     # [B,E] float (row-normalized on selected)
    capacity_per_expert: Optional[int] = None


def row_topk_routing(probs: torch.Tensor, k: int) -> RoutingResult:
    B, E = probs.shape
    k = min(int(k), E)
    topv, topi = torch.topk(probs, k=k, dim=-1)  # [B,k]
    w = topv / (topv.sum(dim=-1, keepdim=True) + 1e-9)

    mask = torch.zeros((B, E), device=probs.device, dtype=torch.bool)
    mask.scatter_(1, topi, True)

    weights = torch.zeros((B, E), device=probs.device, dtype=probs.dtype)
    weights.scatter_(1, topi, w)
    return RoutingResult(mask=mask, weights=weights, capacity_per_expert=None)


def expert_choice_routing(probs: torch.Tensor, capacity_factor: float) -> RoutingResult:
    """Each expert chooses top-m tokens, m = ceil(B*c/E)."""
    B, E = probs.shape
    m = int(math.ceil(B * float(capacity_factor) / E))
    m = max(1, min(m, B))

    mask = torch.zeros((B, E), device=probs.device, dtype=torch.bool)
    for e in range(E):
        col = probs[:, e]
        _, idx = torch.topk(col, k=m, dim=0)
        mask[idx, e] = True

    weight_unnorm = probs * mask.float()
    denom = weight_unnorm.sum(dim=-1, keepdim=True) + 1e-9
    weights = weight_unnorm / denom
    return RoutingResult(mask=mask, weights=weights, capacity_per_expert=m)