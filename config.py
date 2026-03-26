# -*- coding: utf-8 -*-
"""All hyperparameters in one place."""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, Union, List
try:
    from typing import Literal
except ImportError:  # Python < 3.8
    from typing_extensions import Literal



@dataclass
class Data1Config:
    seed: int = 42
    n_train: int = 50_000
    n_val: int = 10_000
    n_test: int = 10_000
    # split behavior
    fixed_test_set: bool = True  # True: task fixed to data.seed; test raw x fixed; norm fixed to data.seed

    input_dim: int = 16
    num_classes: int = 4
    num_regimes: int = 4
    input_mode: Literal["x", "x_beta", "x_B"] = "x"
    proj_dim: int = 8
    regime_assign: Literal["score", "fixed"] = "score"  # "score": gate-score-based assignment; "fixed": regime forced to equal true label

    # input distribution
    gmm_correlated: bool = True
    gmm_cov_scale: float = 1.0

    # regime gate type for data generation
    regime_gate_type: Literal["phi", "linear", "nonlinear", "softmax"] = "linear"
    regime_gate_hidden: int = 32  # hidden dim for nonlinear gate
    regime_gate_temperature: float = 1.0  # temperature for softmax gate

    # regime difficulty
    regime_noise_std: float = 0.0
    regime_sim_level: Literal["high", "mid", "low"] = "high"
    share_regime: bool = True
    regime_cos_target_low: float = 0.0
    regime_cos_target_mid: float = 0.4
    regime_cos_target_high: float = 0.9
    regime_func_sim_level: Literal["high", "mid", "low"] = "high"
    regime_func_cos_target_low: float = 0.0
    regime_func_cos_target_mid: float = 0.4
    regime_func_cos_target_high: float = 0.9
    share_regime_weight: float = 1.0
    regime_specific_weight: float = 1.0

    # regime label generator complexity (per-regime experts)
    regime_mlp_hidden: int = 16
    regime_mlp_out: int = 8
    regime_mlp_depth: int = 2
    regime_nonshared_type: Literal["mlp", "lora"] = "mlp"
    regime_lora_rank: int = 16
    regime_lora_alpha: float = 16.0
    # shared regime generator complexity (used when share_regime=True)
    shared_regime_mlp_hidden: Optional[int] = None
    shared_regime_mlp_out: Optional[int] = None
    shared_regime_mlp_depth: Optional[int] = None
    regime_mlp_sparsity: Literal["random", "sparse"] = "random"
    regime_activation: str = "gelu"  # "tanh"|"relu"|"gelu"

    # label noise
    logit_noise_std: float = 0.0
    flip_prob: float = 0.0  # continous only 0.0
    # scale factor for data1 class logits before noise/argmax/softmax
    data1_logit_scale: float = 1

    # spurious features
    spurious_enabled: bool = False
    spurious_dim: int = 0
    spurious_corr: float = 0.6
    share_feature_dim: int = 0
    share_feature_std: float = 0.02


@dataclass
class Data2Config(Data1Config):
    # regression output
    num_classes: int = 1


@dataclass
class Data3Config(Data1Config):
    # regression output
    num_classes: int = 1
    # additive noise for regression target
    target_noise_std: float = 0.0
    # soft projection temperature for u = sigmoid((p^T x)/T)
    proj_temperature: float = 1.0
    # nonlinear target difficulty: "mid" (harder) / "difficult" (hardest)
    # keep "diffcult" as accepted alias for convenience.
    data3_difficulty: Literal["mid", "difficult", "diffcult"] = "mid"

    # ---- MID difficulty coefficients ----
    d3_mid_bias: float = 0.20
    d3_mid_u1_coef: float = 0.95
    d3_mid_u2_coef: float = -0.35
    d3_mid_hinge1_coef: float = 0.90
    d3_mid_hinge2_coef: float = -0.70
    d3_mid_hinge3_coef: float = 0.35
    d3_mid_u1u2_coef: float = 0.55
    d3_mid_u1_sq_coef: float = 0.30
    d3_mid_u2_sq_coef: float = -0.22
    d3_mid_sin_u1_coef: float = 0.22
    d3_mid_cos_u2_coef: float = -0.18
    d3_mid_cos_u1u2_coef: float = 0.16
    d3_mid_sin_freq: float = 2.0
    d3_mid_cos_freq: float = 2.0
    d3_mid_b1_u1_coef: float = 0.60
    d3_mid_b1_bias: float = -0.75
    d3_mid_b2_u1_coef: float = -0.80
    d3_mid_b2_bias: float = -0.10

    # ---- DIFFICULT difficulty coefficients ----
    d3_diff_bias: float = 0.15
    d3_diff_u1_coef: float = 1.00
    d3_diff_u2_coef: float = -0.40
    d3_diff_hinge1_coef: float = 1.25
    d3_diff_hinge2_coef: float = -1.05
    d3_diff_hinge3_coef: float = 0.55
    d3_diff_u1u2_coef: float = 0.95
    d3_diff_u1_sq_coef: float = 0.55
    d3_diff_u2_sq_coef: float = -0.45
    d3_diff_sin_u1_coef: float = 0.42
    d3_diff_cos_u2_coef: float = -0.36
    d3_diff_cos_u1u2_coef: float = 0.32
    d3_diff_sin_freq: float = 4.0
    d3_diff_cos_freq: float = 5.0
    d3_diff_b1_u1_coef: float = 0.95
    d3_diff_b1_bias: float = -0.85
    d3_diff_b2_u1_coef: float = -1.15
    d3_diff_b2_bias: float = -0.08


@dataclass
class ModelConfig:
    seed: int = 42
    input_dim: int = 16
    num_classes: int = 4
    num_regimes: int = 4

    activation: str = "gelu"  # "relu"|"gelu"|"tanh"

    # Gate1 (CMR)
    gate1_mode: Literal["constant", "constant_bigmlp"] = "constant"
    gate1_constant: float = 1.0 #1有shared，0是纯moe
    gate1_dropout: float = 0.0
    gate1_fuse_mode: Literal["avg", "gate", "learned"] = "learned"

    # shared expert (projection to fusion dim)
    num_shared_experts: int = 1
    shared2_hidden_dim: int = 128  # base width; expands by depth: 256,512,1024,...
    shared2_out_dim: int = 8
    shared2_mlp_depth: int = 2
    shared2_merge: Literal["average", "sum"] = "sum"
    bigmlp_hidden: Optional[int] = None
    bigmlp_width_scale: Literal["params_match", "num_experts", "topk"] = "topk"
    bigmlp_width_multiplier: float = 1.0

    # MoE
    num_experts: int = 4
    topk: int = 2
    capacity_factor: float = 1.25
    expert_hidden_dim: int = 16  # base width; expands by depth: 128,256,512,...
    expert_feature_dim: int = 8
    expert_mlp_depth: int = 2
    expert_type: Literal["mlp", "lora"] = "mlp"
    expert_lora_rank: int = 16
    expert_lora_alpha: float = 16.0
    expert_lora_dropout: float = 0.0
    expert_lora_init: Literal["kaiming_zero_b", "normal"] = "normal"
    gate_type: Literal["linear", "nonlinear", "random", "constant_learned"] = "linear"
    gate_hidden: Optional[int] = None
    gate_bias: bool = False

    # router softmax temperature for actual routing: softmax(s/T)
    gate_temperature: float = 1.0
    gumbel_noise: float = 0.0
    routing_mode: Literal["row_topk", "expert_choice"] = "row_topk"

    # SMoE/Fedus balance loss (Eq.9)
    lb_enabled: bool = True
    lb_lambda: float = 0.001
    lb_tau0: float = 1.0

    # expert similarity regularization (CKA on expert features)
    sim_enabled: bool = False
    sim_lambda: float = 0.01
    sim_proj_enabled: bool = False
    sim_proj_hidden: Optional[int] = None
    sim_proj_out: Optional[int] = 16
    sim_f_star: int = 8          # min shared tokens for a pair to be penalized
    sim_t_star: float = 0.5      # min CKA value threshold; pairs below this are skipped
    sim_kernel: str = "linear"   # "linear" or "rbf"
    sim_sigma: float = 0.85      # RBF bandwidth (ignored when sim_kernel="linear")
    # For reporting CKA/PES, optionally normalize to fixed subset size K.
    # If num_experts > K, compute metric on all C(E,K) subsets (or a sampled subset) and average.
    sim_metric_subset_k: int = 4
    sim_metric_subset_max_subsets: int = 10  # 0 = use all subsets when E>K
    sim_metric_subset_seed: int = 42

    # expert first-layer orthogonality
    expert_orth_lambda: float = 0.0  # 1e-4----1e-2
    expert_second_layer_orth_lambda: float = 0.0
    shared_moe_first_layer_orth_lambda: float = 0.0
    shared_moe_second_layer_orth_lambda: float = 0.0
    # dropout modes
    # mode1: shared + moe use same dropout
    # mode2: shared and moe use separate dropout
    dropout_mode: Literal["mode1", "mode2"] = "mode2"
    dropout_mode1_rate: float = 0.0
    dropout_mode2_moe_rate: float = 0.0
    dropout_mode2_shared_rate: float = 0.0
    # legacy alias for mode1 rate (kept for backward compatibility)
    expert_dropout: Optional[float] = None
    
    # sparsity regularization (experts vs shared are separate)
    expert_l1_lambda: float = 0.0  #1e-5-5e-5-1e-4
    expert_group_lasso_lambda: float = 0.0
    shared_l1_lambda: float = 0.0
    shared_group_lasso_lambda: float = 0.0

    # expert pruning (MoE experts only)
    prune_experts_enabled: bool = False
    prune_experts_norm_threshold: float = 0.05
    prune_experts_min_keep: int = 2
    prune_finetune_enabled: bool = False
    prune_finetune_epochs: int = 5
    prune_finetune_lr_scale: float = 0.1

    def __post_init__(self):
        # backward compatibility for older config values
        if self.bigmlp_width_scale == "num_experts_plus_one":
            self.bigmlp_width_scale = "num_experts"
        elif self.bigmlp_width_scale == "topk_plus_one":
            self.bigmlp_width_scale = "topk"

        self.expert_lora_rank = max(1, int(self.expert_lora_rank))
        self.expert_lora_alpha = float(self.expert_lora_alpha)
        self.expert_lora_dropout = float(self.expert_lora_dropout)
        if self.expert_dropout is not None:
            self.dropout_mode1_rate = float(self.expert_dropout)
        if self.gate_hidden is None:
            in_dim = self.input_dim
            E = self.num_experts
            self.gate_hidden = max(1, int(round((in_dim * E) / (in_dim + E))))
        if self.bigmlp_hidden is None:
            trunk_out = self.input_dim
            D = self.expert_feature_dim
            H = self.expert_hidden_dim

            if self.bigmlp_width_scale == "num_experts":
                base_hidden = H * self.num_experts
            elif self.bigmlp_width_scale == "topk":
                base_hidden = H * self.topk
            else:
                # Match params of (shared2 + MoE) with a single big MLP.
                shared2_params = trunk_out * H + H + H * D + D
                if self.expert_mlp_depth <= 1:
                    expert_params = trunk_out * D + D
                else:
                    expert_params = trunk_out * D + D + (self.expert_mlp_depth - 1) * (D * D + D)
                moe_params = trunk_out * self.num_experts + self.num_experts * expert_params
                target_params = shared2_params + moe_params
                denom = trunk_out + D + 1
                base_hidden = (target_params - D) / denom

            self.bigmlp_hidden = max(1, int(round(base_hidden * self.bigmlp_width_multiplier)))
        if self.sim_proj_hidden is None:
            self.sim_proj_hidden = self.expert_feature_dim
        if self.sim_proj_out is None:
            self.sim_proj_out = self.expert_feature_dim

    def shared_dropout(self) -> float:
        if self.dropout_mode == "mode1":
            return float(self.dropout_mode1_rate)
        return float(self.dropout_mode2_shared_rate)

    def moe_dropout(self) -> float:
        if self.dropout_mode == "mode1":
            return float(self.dropout_mode1_rate)
        return float(self.dropout_mode2_moe_rate)


@dataclass
class TrainConfig:
    # global
    seed: int = 42
    device: str = "cpu"  # force CPU
    epochs: int = 150
    batch_size: int = 512#512
    lr: float = 5e-3#5e-3 
    weight_decay: float = 1e-3#1e-3 
    adamw_beta1: float = 0.9#0.9
    adamw_beta2: float = 0.999#0.999
    adamw_eps: float = 1e-8#1e-8
    grad_clip: float = 1.0

    # logging
    log_every: int = 100
    eval_every: int = 500
    test_routing: Literal["score", "fixed"] = "score"
    test_expert_mode: Literal["share_moe", "only_share", "only_moe", "only_moe_top1"] = "share_moe"

    # early stopping
    early_stop_enabled: bool = True
    early_stop_patience: int = 15
    early_stop_min_delta: float = 5e-4

    # CPU acceleration knobs
    num_threads: int = 32        # torch intra-op threads
    num_workers: int = 16        # DataLoader workers (0 is often safest on Windows)

    # similarity metrics (CKA / PES) — expensive; off by default
    compute_sim_metrics: bool = True

    # repetitions
    reps: int = 20
    rep_seed_offset: int = 1000  # each rep uses seed + rep*offset

    # optuna
    optuna_enabled: bool = False
    optuna_trials: int = 30
    # objective metric
    optuna_objective: Literal["best_val_loss"] = "best_val_loss"
    # search space: base training hyperparameters
    optuna_batch_size_choices: List[int] = field(default_factory=lambda: [128, 256, 512])
    optuna_lr_min: float = 2e-4
    optuna_lr_max: float = 5e-3
    optuna_weight_decay_min: float = 1e-4
    optuna_weight_decay_max: float = 5e-3
    # search space: AdamW key hyperparameters
    optuna_adamw_beta1_min: float = 0.85
    optuna_adamw_beta1_max: float = 0.95
    optuna_adamw_beta2_min: float = 0.95
    optuna_adamw_beta2_max: float = 0.9999
    optuna_adamw_eps_min: float = 1e-9
    optuna_adamw_eps_max: float = 1e-7

    # output
    output_dir: str = "runs"
    experiment_name: str = "exp_data1"


@dataclass
class ProjectConfig:
    data_variant: Literal["data1", "data2", "data3"] = "data1"
    data1: Data1Config = field(default_factory=Data1Config)
    data2: Data2Config = field(default_factory=Data2Config)
    data3: Data3Config = field(default_factory=Data3Config)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def active_data(self) -> Union[Data1Config, Data2Config, Data3Config]:
        if self.data_variant == "data2":
            return self.data2
        if self.data_variant == "data3":
            return self.data3
        return self.data1

    def sync_model_to_data(self) -> None:
        data_cfg = self.active_data()
        real_input_dim = data_cfg.input_dim
        if hasattr(data_cfg, "input_mode") and getattr(data_cfg, "input_mode") != "x":
            if data_cfg.input_mode == "x_beta":
                real_input_dim = 1
            elif data_cfg.input_mode == "x_B":
                real_input_dim = int(getattr(data_cfg, "proj_dim"))
        if data_cfg.spurious_enabled:
            real_input_dim += data_cfg.spurious_dim
        self.model.input_dim = real_input_dim
        self.model.num_classes = data_cfg.num_classes
        if hasattr(data_cfg, "num_regimes"):
            self.model.num_regimes = data_cfg.num_regimes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_variant": self.data_variant,
            "data1": asdict(self.data1),
            "data2": asdict(self.data2),
            "data3": asdict(self.data3),
            "model": asdict(self.model),
            "train": asdict(self.train),
        }


def make_default_config() -> ProjectConfig:
    cfg = ProjectConfig()
    cfg.model.seed = cfg.data1.seed = cfg.data2.seed = cfg.data3.seed = cfg.train.seed = 42
    cfg.sync_model_to_data()
    cfg.train.device = "cpu"
    return cfg
