# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Tuple, Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig
from routing import row_topk_routing, expert_choice_routing


def activation_fn(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(name)


def _progressive_hidden_dims(base_hidden: int, depth: int) -> list[int]:
    """Depth-aware widths:
    depth=1 -> []
    depth=2 -> [h]
    depth=3 -> [h, 2h]
    depth=4 -> [h, 2h, 4h]
    """
    d = int(depth)
    if d <= 1:
        return []
    h0 = max(1, int(base_hidden))
    return [h0 * (2 ** i) for i in range(d - 1)]


class FFN(nn.Module):
    def __init__(self, in_dim: int, hidden_dims, out_dim: int, activation: str, dropout: float = 0.0, layernorm: bool = False):
        super().__init__()
        dims = [in_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if layernorm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(activation_fn(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Gate1CMR(nn.Module):
    def __init__(self, constant: float):
        super().__init__()
        self.constant = float(constant)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.full((h.shape[0], 1), self.constant, device=h.device, dtype=h.dtype)


class Gate1Linear(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.lin(h))

class RandomGate(nn.Module):
    def __init__(self, num_experts: int):
        super().__init__()
        self.E = int(num_experts)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.rand((h.shape[0], self.E), device=h.device, dtype=h.dtype)


class ConstantLearnedGate(nn.Module):
    def __init__(self, num_experts: int, init_logit: float = 0.0):
        super().__init__()
        self.logits = nn.Parameter(torch.full((int(num_experts),), float(init_logit)))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.logits.view(1, -1).expand(h.shape[0], -1)

class ExpertFeatureMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, depth: int, activation: str, dropout: float = 0.0):
        super().__init__()
        assert depth >= 1
        hidden_dims = _progressive_hidden_dims(hidden_dim, depth)
        layers = []
        if len(hidden_dims) == 0:
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            prev = in_dim
            for h in hidden_dims:
                layers.append(nn.Linear(prev, h))
                layers.append(activation_fn(activation))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                prev = h
            layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LowRankLinear(nn.Module):
    """Linear layer with low-rank factorization: W = (A @ B)^T."""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float = 1.0,
        bias: bool = True,
        lora_dropout: float = 0.0,
        init_mode: str = "kaiming_zero_b",
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = max(1, int(rank))
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.lora_dropout = float(lora_dropout)

        self.A = nn.Parameter(torch.empty(self.in_features, self.rank))
        self.B = nn.Parameter(torch.empty(self.rank, self.out_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features))
        else:
            self.register_parameter("bias", None)

        if init_mode == "kaiming_zero_b":
            nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
            nn.init.zeros_(self.B)
        elif init_mode == "normal":
            nn.init.normal_(self.A, mean=0.0, std=0.02)
            nn.init.normal_(self.B, mean=0.0, std=0.02)
        else:
            raise ValueError(init_mode)

    def effective_weight(self) -> torch.Tensor:
        # Return shape [out_features, in_features], same as nn.Linear.weight
        return (self.A @ self.B).T * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        if self.lora_dropout > 0:
            h = F.dropout(h, p=self.lora_dropout, training=self.training)
        y = (h @ self.A) @ self.B
        y = y * self.scaling
        if self.bias is not None:
            y = y + self.bias
        return y


class ExpertFeatureLoRA(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        depth: int,
        activation: str,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        lora_dropout: float = 0.0,
        init_mode: str = "kaiming_zero_b",
    ):
        super().__init__()
        assert depth >= 1
        hidden_dims = _progressive_hidden_dims(hidden_dim, depth)
        layers = []
        if len(hidden_dims) == 0:
            layers.append(
                LowRankLinear(in_dim, out_dim, rank=rank, alpha=alpha, lora_dropout=lora_dropout, init_mode=init_mode)
            )
        else:
            prev = in_dim
            for h in hidden_dims:
                layers.append(
                    LowRankLinear(prev, h, rank=rank, alpha=alpha, lora_dropout=lora_dropout, init_mode=init_mode)
                )
                layers.append(activation_fn(activation))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                prev = h
            layers.append(
                LowRankLinear(prev, out_dim, rank=rank, alpha=alpha, lora_dropout=lora_dropout, init_mode=init_mode)
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MoELayer(nn.Module):
    """MoE returns aggregated FEATURE (not logits)."""
    def __init__(self, in_dim: int, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.E = cfg.num_experts
        self.temperature = cfg.gate_temperature
        self.gumbel_noise = cfg.gumbel_noise
        self.pruned_mask = None
        if cfg.gate_type == "linear":
            self.gate = nn.Linear(in_dim, self.E, bias=False)
        elif cfg.gate_type == "nonlinear":
            self.gate = nn.Sequential(
                nn.Linear(in_dim, cfg.gate_hidden, bias=cfg.gate_bias),
                activation_fn(cfg.activation),
                nn.Linear(cfg.gate_hidden, self.E, bias=cfg.gate_bias),
            )
        elif cfg.gate_type == "random":
            self.gate = RandomGate(self.E)
        elif cfg.gate_type == "constant_learned":
            self.gate = ConstantLearnedGate(self.E)
        else:
            raise ValueError(cfg.gate_type)
        if cfg.expert_type == "mlp":
            self.experts = nn.ModuleList([
                ExpertFeatureMLP(
                    in_dim,
                    cfg.expert_hidden_dim,
                    cfg.expert_feature_dim,
                    cfg.expert_mlp_depth,
                    cfg.activation,
                    dropout=cfg.moe_dropout(),
                )
                for _ in range(self.E)
            ])
        elif cfg.expert_type == "lora":
            self.experts = nn.ModuleList([
                ExpertFeatureLoRA(
                    in_dim,
                    cfg.expert_hidden_dim,
                    cfg.expert_feature_dim,
                    cfg.expert_mlp_depth,
                    cfg.activation,
                    rank=cfg.expert_lora_rank,
                    alpha=cfg.expert_lora_alpha,
                    dropout=cfg.moe_dropout(),
                    lora_dropout=cfg.expert_lora_dropout,
                    init_mode=cfg.expert_lora_init,
                )
                for _ in range(self.E)
            ])
        else:
            raise ValueError(cfg.expert_type)
        if cfg.sim_proj_enabled:
            self.sim_proj = nn.Sequential(
                nn.Linear(cfg.expert_feature_dim, cfg.sim_proj_hidden),
                activation_fn(cfg.activation),
                nn.Linear(cfg.sim_proj_hidden, cfg.sim_proj_out),
            )
        else:
            self.sim_proj = None

    def _router_scores(self, h: torch.Tensor) -> torch.Tensor:
        scores = self.gate(h)
        if self.gumbel_noise and self.gumbel_noise > 0:
            g = -torch.log(-torch.log(torch.rand_like(scores).clamp(1e-6, 1 - 1e-6))).clamp(-5, 5)
            scores = scores + self.gumbel_noise * g
        return scores

    def forward(
        self,
        h: torch.Tensor,
        forced_expert: Optional[torch.Tensor] = None,
        force_top1: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        if forced_expert is None:
            scores = self._router_scores(h)  # [B,E]
            if self.pruned_mask is not None:
                mask = self.pruned_mask.to(device=scores.device)
                scores = scores.masked_fill(~mask.view(1, -1), -1e9)
            probs = F.softmax(scores / max(self.temperature, 1e-6), dim=-1)  # [B,E]

            if self.cfg.gate_type == "constant_learned" and (not force_top1):
                # For learned-constant gate, use all currently active experts (no top-k pruning).
                B, E = probs.shape
                if self.pruned_mask is None:
                    active = torch.ones((B, E), device=h.device, dtype=torch.bool)
                else:
                    active_1d = self.pruned_mask.to(device=h.device).view(1, -1)
                    active = active_1d.expand(B, -1)
                weights = probs * active.float()
                weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                mask = active
                capacity_per_expert = torch.full((self.E,), int(B), device=h.device, dtype=torch.long)
            elif self.cfg.routing_mode == "row_topk":
                rr = row_topk_routing(probs, 1 if force_top1 else self.cfg.topk)
                mask, weights = rr.mask, rr.weights
                capacity_per_expert = rr.capacity_per_expert
            elif self.cfg.routing_mode == "expert_choice":
                rr = expert_choice_routing(probs, self.cfg.capacity_factor)
                mask, weights = rr.mask, rr.weights
                capacity_per_expert = rr.capacity_per_expert
            else:
                raise ValueError(self.cfg.routing_mode)
        else:
            B = h.shape[0]
            scores = torch.zeros((B, self.E), device=h.device, dtype=h.dtype)
            probs = torch.zeros((B, self.E), device=h.device, dtype=h.dtype)
            idx = forced_expert.to(torch.long).view(-1)
            mask = torch.zeros((B, self.E), device=h.device, dtype=torch.bool)
            mask[torch.arange(B, device=h.device), idx] = True
            weights = mask.float()
            probs = weights
            capacity_per_expert = torch.full((self.E,), int(B), device=h.device, dtype=torch.long)
        B, E = probs.shape
        D = self.cfg.expert_feature_dim

        moe_feat = torch.zeros((B, D), device=h.device, dtype=h.dtype)
        expert_feats: Dict[int, torch.Tensor] = {}
        expert_feat_indices: Dict[int, torch.Tensor] = {}

        for e in range(E):
            idx = torch.where(mask[:, e])[0]
            if idx.numel() == 0:
                continue
            fe = self.experts[e](h[idx])      # [n_e, D]
            w = weights[idx, e]               # [n_e]
            moe_feat[idx] += fe * w.unsqueeze(-1)
            expert_feats[e] = fe
            expert_feat_indices[e] = idx      # token indices dispatched to expert e

        aux = {
            "router_scores": scores,
            "router_probs": probs,
            "dispatch_mask": mask,
            "dispatch_weights": weights,
            "expert_feats": expert_feats,
            "expert_feat_indices": expert_feat_indices,
            "capacity_per_expert": capacity_per_expert,
            "routing_mode": self.cfg.routing_mode,
            "forced_expert": forced_expert is not None,
        }
        return moe_feat, aux


class FNN_CMR_MoE(nn.Module):
    """Feature-level fusion:
      h_fuse = (1-g)*h_shared + g*h_moe
      logits = head(h_fuse)
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        trunk_out = cfg.input_dim
        self.trunk = nn.Identity()
        self.use_bigmlp = cfg.gate1_mode == "constant_bigmlp"
        self.use_shared = (not self.use_bigmlp) and (cfg.gate1_constant > 0.0)
        if self.use_bigmlp:
            self.gate1 = None
            self.gate1_learned = None
        elif cfg.gate1_fuse_mode == "gate":
            self.gate1 = Gate1Linear(trunk_out)
            self.gate1_learned = None
        elif cfg.gate1_fuse_mode == "learned":
            self.gate1 = None
            self.gate1_learned = nn.Parameter(torch.tensor(0.0))
        else:
            self.gate1 = Gate1CMR(0.5)
            self.gate1_learned = None

        if self.use_bigmlp:
            shared2_big_layers = [
                nn.Linear(trunk_out, cfg.bigmlp_hidden),
                activation_fn(cfg.activation),
            ]
            shared_drop = cfg.shared_dropout()
            if shared_drop and shared_drop > 0:
                shared2_big_layers.append(nn.Dropout(shared_drop))
            shared2_big_layers.extend([
                nn.Linear(cfg.bigmlp_hidden, cfg.expert_feature_dim),
                activation_fn(cfg.activation),
            ])
            if shared_drop and shared_drop > 0:
                shared2_big_layers.append(nn.Dropout(shared_drop))
            self.shared2_big = nn.Sequential(*shared2_big_layers)
            self.shared2 = None
            self.moe = None
        else:
            if self.use_shared:
                # shared experts -> [B, D]
                shared_modules = []
                shared_drop = cfg.shared_dropout()
                for _ in range(max(1, int(cfg.num_shared_experts))):
                    shared2_layers = []
                    shared_hidden_dims = _progressive_hidden_dims(cfg.shared2_hidden_dim, cfg.shared2_mlp_depth)
                    if len(shared_hidden_dims) == 0:
                        shared2_layers.append(nn.Linear(trunk_out, cfg.shared2_out_dim))
                    else:
                        prev = trunk_out
                        for h in shared_hidden_dims:
                            shared2_layers.append(nn.Linear(prev, h))
                            shared2_layers.append(activation_fn(cfg.activation))
                            if shared_drop and shared_drop > 0:
                                shared2_layers.append(nn.Dropout(shared_drop))
                            prev = h
                        shared2_layers.append(nn.Linear(prev, cfg.shared2_out_dim))
                    shared2_layers.append(activation_fn(cfg.activation))
                    if shared_drop and shared_drop > 0:
                        shared2_layers.append(nn.Dropout(shared_drop))
                    shared_modules.append(nn.Sequential(*shared2_layers))
                self.shared2 = nn.ModuleList(shared_modules)
            else:
                self.shared2 = None

            # MoE -> [B, D]
            self.moe = MoELayer(trunk_out, cfg)
            self.shared2_big = None

        # classifier after fusion
        self.head = nn.Linear(cfg.expert_feature_dim, cfg.num_classes)

    def forward(
        self,
        x: torch.Tensor,
        forced_expert: Optional[torch.Tensor] = None,
        shared_only: bool = False,
        moe_only: bool = False,
        force_top1: bool = False,
        force_avg_fuse: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        h = self.trunk(x)      # [B, trunk_out]
        if self.use_bigmlp:
            hs = self.shared2_big(h)   # [B,D]
            logits = self.head(hs)     # [B,C]
            g1 = torch.zeros((h.shape[0], 1), device=h.device, dtype=h.dtype)
            aux = {
                "g1": g1,
                "moe_aux": {},
                "shared_feats": hs,
                "moe_feats": None,
                "fused_feats": hs,
                "trunk_hidden": h,
            }
            return logits, aux

        if self.gate1 is not None:
            g1 = self.gate1(h)     # [B,1]
        else:
            g1 = torch.sigmoid(self.gate1_learned).view(1, 1).expand(h.shape[0], 1)
        if force_avg_fuse:
            g1 = torch.full_like(g1, 0.5)

        if self.training and self.cfg.gate1_dropout and self.cfg.gate1_dropout > 0:
            drop = (torch.rand_like(g1) < self.cfg.gate1_dropout).float()
            g1 = g1 * (1.0 - drop)

        if self.shared2 is None:
            hs = torch.zeros((h.shape[0], self.cfg.expert_feature_dim), device=h.device, dtype=h.dtype)
        else:
            hs_all = [m(h) for m in self.shared2]
            hs_stack = torch.stack(hs_all, dim=0)
            if self.cfg.shared2_merge == "sum":
                hs = hs_stack.sum(dim=0)
            else:
                hs = hs_stack.mean(dim=0)
        if hs.shape[1] != self.cfg.expert_feature_dim:
            raise ValueError("shared2_out_dim must equal expert_feature_dim for fusion")
        if shared_only and moe_only:
            raise ValueError("shared_only and moe_only cannot both be True")
        if shared_only:
            hm = torch.zeros_like(hs)
            moe_aux = {}
        else:
            hm, moe_aux = self.moe(h, forced_expert=forced_expert, force_top1=force_top1)      # [B,D]

        if shared_only:
            hf = hs
        elif moe_only:
            hf = hm
        elif self.cfg.gate1_constant <= 0.0:
            hf = hm
        elif force_avg_fuse or self.cfg.gate1_fuse_mode == "avg":
            hf = (1.0 - g1) * hs + g1 * hm
        else:
            hf = (1.0 - g1) * hs + g1 * hm
        logits = self.head(hf)         # [B,C]

        aux = {
            "g1": g1,
            "moe_aux": moe_aux,
            "shared_feats": hs,
            "moe_feats": hm,
            "fused_feats": hf,
            "trunk_hidden": h,
        }
        return logits, aux
