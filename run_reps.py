# -*- coding: utf-8 -*-
from __future__ import annotations
import time
from pathlib import Path
import copy
from typing import Dict, List, Any
import numpy as np
import torch

from config import make_default_config, ProjectConfig
from utils import now_str, ensure_dir, save_json, setup_cpu, set_seed, hungarian_max_assignment
from trainer import train_one_rep
from data import (
    compute_regime_cosine,
    compute_true_regime_stats,
    compute_true_regime_stats_data3,
    compute_label_regime_percent,
    compute_true_model_metrics,
    compute_true_model_probabilities,
    compute_true_model_metrics_regression,
    generate_data1,
    generate_data2,
    generate_data3,
    estimate_eta_data1_mc,
    estimate_eta_data1_mc_from_x,
    estimate_eta_data2_analytic_from_x,
    estimate_eta_data3_analytic_from_x,
)
from plots import (
    save_loss_curves_with_sd,
    save_optuna_curve,
    save_histogram,
    save_biasvar_sample_grouped_plot,
    save_yhat_grouped_boxplot,
    save_prob_boxplot_grid_by_true_class,
)

ETA_MC_SAMPLES = 200
YHAT_HIST_MAX_PANELS = 12


def _stack_curves(curves: List[List[float]], max_len: int) -> np.ndarray:
    """Pad curves to [rep, max_len] with last value (or NaN if empty)."""
    R = len(curves)
    out = np.zeros((R, max_len), dtype=np.float64)
    for i, c in enumerate(curves):
        if len(c) == 0:
            out[i, :] = np.nan
            continue
        c = np.array(c, dtype=np.float64)
        if len(c) < max_len:
            pad = np.full((max_len - len(c),), c[-1], dtype=np.float64)
            c = np.concatenate([c, pad], axis=0)
        out[i, :] = c[:max_len]
    return out


def _mean_std_dict(xs: List[float]) -> Dict[str, float]:
    x = np.array(xs, dtype=np.float64)
    if len(x) == 1:
        return {"mean": float(x.mean()), "std": 0.0, "var": 0.0}
    return {"mean": float(x.mean()), "std": float(x.std(ddof=1)), "var": float(x.var(ddof=1))}


def _apply_optuna_params(cfg: ProjectConfig, params: Dict[str, float]) -> None:
    cfg.train.lr = float(params.get("lr", cfg.train.lr))
    cfg.train.weight_decay = float(params.get("weight_decay", cfg.train.weight_decay))
    cfg.train.batch_size = int(params.get("batch_size", cfg.train.batch_size))
    cfg.train.adamw_beta1 = float(params.get("adamw_beta1", cfg.train.adamw_beta1))
    cfg.train.adamw_beta2 = float(params.get("adamw_beta2", cfg.train.adamw_beta2))
    cfg.train.adamw_eps = float(params.get("adamw_eps", cfg.train.adamw_eps))


def _tune_hparams(cfg: ProjectConfig, run_dir: Path) -> None:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("optuna is not installed. Please install it before enabling optuna.") from exc

    trials = int(cfg.train.optuna_trials)

    def objective(trial: "optuna.Trial") -> float:
        batch_choices = sorted({int(v) for v in cfg.train.optuna_batch_size_choices if int(v) > 0})
        if not batch_choices:
            batch_choices = [int(cfg.train.batch_size)]

        lr_min = min(float(cfg.train.optuna_lr_min), float(cfg.train.optuna_lr_max))
        lr_max = max(float(cfg.train.optuna_lr_min), float(cfg.train.optuna_lr_max))
        wd_min = min(float(cfg.train.optuna_weight_decay_min), float(cfg.train.optuna_weight_decay_max))
        wd_max = max(float(cfg.train.optuna_weight_decay_min), float(cfg.train.optuna_weight_decay_max))
        b1_min = min(float(cfg.train.optuna_adamw_beta1_min), float(cfg.train.optuna_adamw_beta1_max))
        b1_max = max(float(cfg.train.optuna_adamw_beta1_min), float(cfg.train.optuna_adamw_beta1_max))
        b2_min = min(float(cfg.train.optuna_adamw_beta2_min), float(cfg.train.optuna_adamw_beta2_max))
        b2_max = max(float(cfg.train.optuna_adamw_beta2_min), float(cfg.train.optuna_adamw_beta2_max))
        eps_min = min(float(cfg.train.optuna_adamw_eps_min), float(cfg.train.optuna_adamw_eps_max))
        eps_max = max(float(cfg.train.optuna_adamw_eps_min), float(cfg.train.optuna_adamw_eps_max))

        params = {
            "batch_size": int(trial.suggest_categorical("batch_size", batch_choices)),
            "lr": float(lr_min if lr_min == lr_max else trial.suggest_float("lr", lr_min, lr_max, log=True)),
            "weight_decay": float(wd_min if wd_min == wd_max else trial.suggest_float("weight_decay", wd_min, wd_max, log=True)),
            "adamw_beta1": float(b1_min if b1_min == b1_max else trial.suggest_float("adamw_beta1", b1_min, b1_max)),
            "adamw_beta2": float(b2_min if b2_min == b2_max else trial.suggest_float("adamw_beta2", b2_min, b2_max)),
            "adamw_eps": float(eps_min if eps_min == eps_max else trial.suggest_float("adamw_eps", eps_min, eps_max, log=True)),
        }
        cfg_t = copy.deepcopy(cfg)
        _apply_optuna_params(cfg_t, params)
        res = train_one_rep(
            cfg_t,
            rep_idx=0,
            run_dir=run_dir / f"trial_{trial.number:03d}",
            save_plots=False,
        )
        if cfg.train.optuna_objective == "best_val_loss":
            return float(res["best_val"])
        raise ValueError(f"Unsupported optuna objective: {cfg.train.optuna_objective}")

    startup_trials = max(8, min(20, trials // 5))
    try:
        sampler = optuna.samplers.TPESampler(
            seed=int(cfg.train.seed),
            n_startup_trials=startup_trials,
            multivariate=True,
        )
    except TypeError:
        # Backward compatibility for older Optuna versions.
        sampler = optuna.samplers.TPESampler(
            seed=int(cfg.train.seed),
            n_startup_trials=startup_trials,
        )
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=trials)

    best_params = study.best_params
    best_val = float(study.best_value)
    _apply_optuna_params(cfg, best_params)
    print(f"[{now_str()}] optuna best: {best_params} best_val={best_val:.6f}")

    trial_records = [{"params": t.params, "best_val": t.value} for t in study.trials]
    save_json(
        run_dir / "optuna_results.json",
        {"trials": trial_records, "best": {"params": best_params, "best_val": best_val}},
    )
    trial_vals = [float(t.value) if t.value is not None else float("nan") for t in study.trials]
    save_optuna_curve(trial_vals, run_dir / "plots" / "optuna_val_loss.png")


def main():
    cfg = make_default_config()
    if cfg.train.test_routing not in ("score", "fixed"):
        raise ValueError(f"Unsupported test_routing={cfg.train.test_routing!r}; expected 'score' or 'fixed'.")

    # ===== CPU SETUP (ONCE, VERY EARLY) =====
    setup_cpu(cfg.train.num_threads)
    set_seed(cfg.train.seed)

    # ---- force CPU ----
    cfg.train.device = "cpu"
    set_seed(cfg.train.seed)

    run_id = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_dir = ensure_dir(Path(cfg.train.output_dir) / cfg.train.experiment_name / run_id)
    plots_dir = ensure_dir(run_dir / "plots")

    if cfg.train.optuna_enabled:
        _tune_hparams(cfg, run_dir)

    save_json(run_dir / "config.json", cfg.to_dict())
    print(f"[{now_str()}] run_dir = {run_dir}")

    rep_results: List[Dict[str, Any]] = []
    loss_log_dir = ensure_dir(run_dir / "loss_logs")
    for r in range(cfg.train.reps):
        print(f"\n[{now_str()}] ====== REP {r+1}/{cfg.train.reps} ======")
        rep_res = train_one_rep(cfg, rep_idx=r, run_dir=run_dir)
        rep_results.append(rep_res)
    save_json(
        loss_log_dir / "all_reps_losses.json",
        {
            "train_losses": [rr["train_losses"] for rr in rep_results],
            "val_losses": [rr["val_losses"] for rr in rep_results],
            "test_losses": [rr["test_losses"] for rr in rep_results],
        },
    )

    # ---- collect metrics ----
    def _collect(key: str, split: str):
        return [rr[f"{split}_metrics"][key] for rr in rep_results]

    is_regression = cfg.data_variant in ("data2", "data3")
    active_data_cfg = cfg.active_data()
    bias_var_payload: Dict[str, Any] = {
        "data_variant": cfg.data_variant,
        "fixed_test_set": bool(getattr(active_data_cfg, "fixed_test_set", False)),
        "enabled": False,
        "reason": "fixed_test_set_disabled_or_missing_predictions",
        "eta_mc_samples": int(ETA_MC_SAMPLES) if not is_regression else None,
        "test": {
            "seed": None,
            "num_reps": 0,
            "num_samples": 0,
            "rep_bias2": [],
            "rep_var": [],
            "sample_bias2": [],
            "sample_var": [],
            "y_hat_by_rep": [],
            "eta_test": [],
            "yhat_hist_sample_indices": [],
        },
    }
    if not is_regression:
        per_class_acc = np.array(_collect("per_class_acc", "test"), dtype=np.float64)
        per_class_acc_mean = per_class_acc.mean(axis=0)
        per_class_acc_sd = per_class_acc.std(axis=0, ddof=1) if cfg.train.reps > 1 else np.zeros_like(per_class_acc_mean)

        train_label_freq = np.array(_collect("label_expert_freq", "train"), dtype=np.float64)
        test_label_freq = np.array(_collect("label_expert_freq", "test"), dtype=np.float64)
        train_label_mean = np.nanmean(train_label_freq, axis=0) * 100.0
        test_label_mean = np.nanmean(test_label_freq, axis=0) * 100.0
        train_label_std = np.nanstd(train_label_freq, axis=0, ddof=1) * 100.0 if cfg.train.reps > 1 else np.zeros_like(train_label_mean)
        test_label_std = np.nanstd(test_label_freq, axis=0, ddof=1) * 100.0 if cfg.train.reps > 1 else np.zeros_like(test_label_mean)
    else:
        per_class_acc_mean = []
        per_class_acc_sd = []
        train_label_mean = []
        test_label_mean = []
        train_label_std = []
        test_label_std = []

    train_regime_freq = np.array(_collect("regime_expert_freq", "train"), dtype=np.float64)
    test_regime_freq = np.array(_collect("regime_expert_freq", "test"), dtype=np.float64)
    train_regime_mean = np.nanmean(train_regime_freq, axis=0) * 100.0
    test_regime_mean = np.nanmean(test_regime_freq, axis=0) * 100.0
    train_regime_std = np.nanstd(train_regime_freq, axis=0, ddof=1) * 100.0 if cfg.train.reps > 1 else np.zeros_like(train_regime_mean)
    test_regime_std = np.nanstd(test_regime_freq, axis=0, ddof=1) * 100.0 if cfg.train.reps > 1 else np.zeros_like(test_regime_mean)

    if cfg.data_variant in ("data1", "data2"):
        data_cfg = cfg.data1 if cfg.data_variant == "data1" else cfg.data2
        true_regime_cos = [compute_regime_cosine(data_cfg, cfg.train.seed + r * cfg.train.rep_seed_offset) for r in range(cfg.train.reps)]
        true_regime_stats = [
            compute_true_regime_stats(
                data_cfg,
                cfg.train.seed + r * cfg.train.rep_seed_offset,
                subset_k=cfg.model.sim_metric_subset_k,
                subset_max_subsets=cfg.model.sim_metric_subset_max_subsets,
                subset_seed=cfg.model.sim_metric_subset_seed,
            )
            for r in range(cfg.train.reps)
        ]
        true_regime_entropy = [s["entropy"] for s in true_regime_stats]
        true_regime_gini = [s["gini"] for s in true_regime_stats]
        true_regime_cka = [s["cka_offdiag_mean"] for s in true_regime_stats]
        true_regime_pes = [s["pes"] for s in true_regime_stats]
    elif cfg.data_variant == "data3":
        true_regime_cos = [0.0 for _ in range(cfg.train.reps)]
        true_regime_stats = [
            compute_true_regime_stats_data3(
                cfg.data3,
                cfg.train.seed + r * cfg.train.rep_seed_offset,
                subset_k=cfg.model.sim_metric_subset_k,
                subset_max_subsets=cfg.model.sim_metric_subset_max_subsets,
                subset_seed=cfg.model.sim_metric_subset_seed,
            )
            for r in range(cfg.train.reps)
        ]
        true_regime_entropy = [s["entropy"] for s in true_regime_stats]
        true_regime_gini = [s["gini"] for s in true_regime_stats]
        true_regime_cka = [s["cka_offdiag_mean"] for s in true_regime_stats]
        true_regime_pes = [s["pes"] for s in true_regime_stats]
    else:
        true_regime_cos = [0.0 for _ in range(cfg.train.reps)]
        true_regime_entropy = [0.0 for _ in range(cfg.train.reps)]
        true_regime_gini = [0.0 for _ in range(cfg.train.reps)]
        true_regime_cka = [0.0 for _ in range(cfg.train.reps)]
        true_regime_pes = [0.0 for _ in range(cfg.train.reps)]

    if cfg.data_variant == "data1":
        label_true = np.array(
            [compute_label_regime_percent(cfg.data1, cfg.train.seed + r * cfg.train.rep_seed_offset) for r in range(cfg.train.reps)],
            dtype=np.float64,
        )
        label_true_mean = np.nanmean(label_true, axis=0)
        label_true_std = np.nanstd(label_true, axis=0, ddof=1) if cfg.train.reps > 1 else np.zeros_like(label_true_mean)
        true_model = [
            compute_true_model_metrics(cfg.data1, cfg.train.seed + r * cfg.train.rep_seed_offset, split="test")
            for r in range(cfg.train.reps)
        ]
        true_model_loss = [m["loss"] for m in true_model]
        true_model_acc = [m["micro_acc"] for m in true_model]
        true_model_f1 = [m["macro_f1"] for m in true_model]
        true_model_mse = []
        true_model_mae = []
        true_model_r2 = []
    else:
        label_true_mean = []
        label_true_std = []
        reg_cfg = cfg.data2 if cfg.data_variant == "data2" else cfg.data3
        true_model = [
            compute_true_model_metrics_regression(reg_cfg, cfg.train.seed + r * cfg.train.rep_seed_offset, split="test")
            for r in range(cfg.train.reps)
        ]
        true_model_loss = [m["loss"] for m in true_model]
        true_model_mse = [m["mse"] for m in true_model]
        true_model_mae = [m["mae"] for m in true_model]
        true_model_r2 = [m["r2"] for m in true_model]
        true_model_acc = []
        true_model_f1 = []
    summary = {
        "train": {
            "loss": _mean_std_dict(_collect("loss", "train")),
            "expert_cosine": _mean_std_dict(_collect("expert_cosine", "train")),
            "expert_gate_cosine": _mean_std_dict(_collect("expert_gate_cosine", "train")),
            "expert_entropy": _mean_std_dict(_collect("expert_entropy", "train")),
            "expert_gini": _mean_std_dict(_collect("expert_gini", "train")),
            "cka_offdiag_mean": _mean_std_dict(_collect("cka_offdiag_mean", "train")),
            "expert_pes": _mean_std_dict(_collect("expert_pes", "train")),
            "samples_per_sec": _mean_std_dict(_collect("samples_per_sec", "train")),
            "avg_step_time": _mean_std_dict(_collect("avg_step_time", "train")),
            "total_train_time_sec": _mean_std_dict(_collect("total_time_sec", "train")),
            "flops_total": _mean_std_dict(_collect("flops_total", "train")),
            "active_expert_ratio": _mean_std_dict(_collect("active_expert_ratio", "train")),
            "total_params": _mean_std_dict(_collect("total_params", "train")),
            "active_params": _mean_std_dict(_collect("active_params", "train")),
        },
        "test": {
            "loss": _mean_std_dict(_collect("loss", "test")),
            "expert_cosine": _mean_std_dict(_collect("expert_cosine", "test")),
            "expert_gate_cosine": _mean_std_dict(_collect("expert_gate_cosine", "test")),
            "expert_entropy": _mean_std_dict(_collect("expert_entropy", "test")),
            "expert_gini": _mean_std_dict(_collect("expert_gini", "test")),
            "samples_per_sec": _mean_std_dict(_collect("samples_per_sec", "test")),
            "avg_step_time": _mean_std_dict(_collect("avg_step_time", "test")),
            "total_test_time_sec": _mean_std_dict(_collect("total_time_sec", "test")),
            "flops_total": _mean_std_dict(_collect("flops_total", "test")),
            "active_expert_ratio": _mean_std_dict(_collect("active_expert_ratio", "test")),
            "total_params": _mean_std_dict(_collect("total_params", "test")),
            "active_params": _mean_std_dict(_collect("active_params", "test")),
        },
    }
    summary["true_regime_cosine"] = _mean_std_dict(true_regime_cos)
    summary["true_regime_entropy"] = _mean_std_dict(true_regime_entropy)
    summary["true_regime_gini"] = _mean_std_dict(true_regime_gini)
    summary["true_regime_cka_offdiag_mean"] = _mean_std_dict(true_regime_cka)
    summary["true_regime_pes"] = _mean_std_dict(true_regime_pes)
    if bool(cfg.model.prune_experts_enabled):
        summary["prune"] = {
            "before_prune_expert_number": int(cfg.model.num_experts),
            "after_prune_expert_number": _mean_std_dict(_collect("kept_experts", "test")),
            "pruned_expert_number": _mean_std_dict(_collect("pruned_experts", "test")),
            "prune_norm_threshold": float(cfg.model.prune_experts_norm_threshold),
        }
    if all("train_init_metrics" in rr for rr in rep_results):
        init_cka = [float(rr["train_init_metrics"].get("cka_offdiag_mean", 0.0)) for rr in rep_results]
        init_pes = [float(rr["train_init_metrics"].get("expert_pes", 0.0)) for rr in rep_results]
        summary["train_init"] = {
            "cka_offdiag_mean": _mean_std_dict(init_cka),
            "expert_pes": _mean_std_dict(init_pes),
        }
    if not is_regression:
        summary["label_true_percent"] = label_true_mean.tolist() if isinstance(label_true_mean, np.ndarray) else label_true_mean
        summary["label_true_percent_std"] = label_true_std.tolist() if isinstance(label_true_std, np.ndarray) else label_true_std
        summary["true"] = {
            "loss": _mean_std_dict(true_model_loss),
            "micro_acc": _mean_std_dict(true_model_acc),
            "macro_f1": _mean_std_dict(true_model_f1),
        }
        summary["train"]["acc"] = _mean_std_dict(_collect("acc", "train"))
        summary["train"]["macro_f1"] = _mean_std_dict(_collect("macro_f1", "train"))
        summary["train"]["label_expert_percent"] = train_label_mean.tolist()
        summary["train"]["label_expert_percent_std"] = train_label_std.tolist()
        summary["test"]["acc"] = _mean_std_dict(_collect("acc", "test"))
        summary["test"]["macro_f1"] = _mean_std_dict(_collect("macro_f1", "test"))
        summary["test"]["per_class_acc_mean"] = per_class_acc_mean.tolist()
        summary["test"]["per_class_acc_std"] = per_class_acc_sd.tolist()
        summary["test"]["label_expert_percent"] = test_label_mean.tolist()
        summary["test"]["label_expert_percent_std"] = test_label_std.tolist()
        # ---- data1 bias/variance on fixed test set ----
        if getattr(cfg.data1, "fixed_test_set", False):
            fixed_seed = int(cfg.data1.seed)
            x_test, y_test, _regime_test = generate_data1(cfg.data1, "test", fixed_seed)
            if getattr(cfg.data1, "input_mode", "x") == "x":
                eta_test = estimate_eta_data1_mc_from_x(cfg.data1, x_test, fixed_seed, ETA_MC_SAMPLES)
            else:
                eta_test = estimate_eta_data1_mc(cfg.data1, "test", fixed_seed, ETA_MC_SAMPLES)
            test_preds = [rr.get("test_pred_prob") for rr in rep_results if rr.get("test_pred_prob") is not None]
            if test_preds:
                preds = np.stack(test_preds, axis=0)  # [R,N,C]
                mean_pred = preds.mean(axis=0)        # [N,C]
                var_pred = preds.var(axis=0)          # [N,C]

                # per-rep (standard) bias/var
                rep_bias2 = np.mean(np.sum((preds - eta_test[None, :, :]) ** 2, axis=2), axis=1)  # [R]
                rep_var = np.mean(np.sum((preds - mean_pred[None, :, :]) ** 2, axis=2), axis=1)    # [R]
                summary["test"]["bias2"] = _mean_std_dict(rep_bias2.tolist())
                summary["test"]["var"] = _mean_std_dict(rep_var.tolist())

                # per-sample bias/var
                sample_bias2 = np.sum((mean_pred - eta_test) ** 2, axis=1)  # [N]
                sample_var = np.sum(var_pred, axis=1)                       # [N]
                summary["test"]["bias2_sample"] = _mean_std_dict(sample_bias2.tolist())
                summary["test"]["var_sample"] = _mean_std_dict(sample_var.tolist())

                bias_var_payload["enabled"] = True
                bias_var_payload["reason"] = "ok"
                bias_var_payload["test"]["seed"] = fixed_seed
                bias_var_payload["test"]["num_reps"] = int(preds.shape[0])
                bias_var_payload["test"]["num_samples"] = int(preds.shape[1])
                bias_var_payload["test"]["rep_bias2"] = rep_bias2.astype(np.float64).tolist()
                bias_var_payload["test"]["rep_var"] = rep_var.astype(np.float64).tolist()
                bias_var_payload["test"]["sample_bias2"] = sample_bias2.astype(np.float64).tolist()
                bias_var_payload["test"]["sample_var"] = sample_var.astype(np.float64).tolist()
                y_true_arr = np.asarray(y_test, dtype=np.int64).ravel()
                bias_var_payload["test"]["y_true_test"] = y_true_arr.tolist()
        else:
            summary["test"]["bias2"] = {"mean": float("nan"), "std": float("nan"), "var": float("nan")}
            summary["test"]["var"] = {"mean": float("nan"), "std": float("nan"), "var": float("nan")}
            summary["test"]["bias2_sample"] = {"mean": float("nan"), "std": float("nan"), "var": float("nan")}
            summary["test"]["var_sample"] = {"mean": float("nan"), "std": float("nan"), "var": float("nan")}
    else:
        summary["true"] = {
            "loss": _mean_std_dict(true_model_loss),
            "mse": _mean_std_dict(true_model_mse),
            "mae": _mean_std_dict(true_model_mae),
            "r2": _mean_std_dict(true_model_r2),
        }
        summary["train"]["mse"] = _mean_std_dict(_collect("mse", "train"))
        summary["train"]["mae"] = _mean_std_dict(_collect("mae", "train"))
        summary["train"]["r2"] = _mean_std_dict(_collect("r2", "train"))
        summary["test"]["mse"] = _mean_std_dict(_collect("mse", "test"))
        summary["test"]["mae"] = _mean_std_dict(_collect("mae", "test"))
        summary["test"]["r2"] = _mean_std_dict(_collect("r2", "test"))
        # ---- regression bias/variance on fixed test set ----
        if cfg.data_variant == "data2":
            fixed_test_set = bool(getattr(cfg.data2, "fixed_test_set", False))
            fixed_seed = int(cfg.data2.seed)
            if fixed_test_set:
                x_test, _y_test, regime_test = generate_data2(cfg.data2, "test", fixed_seed)
                eta_test = estimate_eta_data2_analytic_from_x(cfg.data2, x_test, regime_test, fixed_seed)
            else:
                eta_test = None
        else:
            fixed_test_set = bool(getattr(cfg.data3, "fixed_test_set", False))
            fixed_seed = int(cfg.data3.seed)
            if fixed_test_set:
                x_test, _y_test, _regime_test = generate_data3(cfg.data3, "test", fixed_seed)
                eta_test = estimate_eta_data3_analytic_from_x(cfg.data3, x_test, fixed_seed)
            else:
                eta_test = None

        if fixed_test_set and eta_test is not None:
            test_preds = [rr.get("test_pred") for rr in rep_results if rr.get("test_pred") is not None]
            if test_preds:
                preds = np.stack(test_preds, axis=0)  # [R,N,1]
                mean_pred = preds.mean(axis=0)        # [N,1]
                var_pred = preds.var(axis=0)          # [N,1]

                # per-rep (standard) bias/var
                rep_bias2 = np.mean((preds - eta_test[None, :, :]) ** 2, axis=1).reshape(-1)  # [R]
                rep_var = np.mean((preds - mean_pred[None, :, :]) ** 2, axis=1).reshape(-1)    # [R]
                summary["test"]["bias2"] = _mean_std_dict(rep_bias2.tolist())
                summary["test"]["var"] = _mean_std_dict(rep_var.tolist())

                # per-sample bias/var
                sample_bias2 = ((mean_pred - eta_test) ** 2).reshape(-1)  # [N]
                sample_var = var_pred.reshape(-1)                         # [N]
                summary["test"]["bias2_sample"] = _mean_std_dict(sample_bias2.tolist())
                summary["test"]["var_sample"] = _mean_std_dict(sample_var.tolist())

                bias_var_payload["enabled"] = True
                bias_var_payload["reason"] = "ok"
                bias_var_payload["test"]["seed"] = fixed_seed
                bias_var_payload["test"]["num_reps"] = int(preds.shape[0])
                bias_var_payload["test"]["num_samples"] = int(preds.shape[1])
                bias_var_payload["test"]["rep_bias2"] = rep_bias2.astype(np.float64).tolist()
                bias_var_payload["test"]["rep_var"] = rep_var.astype(np.float64).tolist()
                bias_var_payload["test"]["sample_bias2"] = sample_bias2.astype(np.float64).tolist()
                bias_var_payload["test"]["sample_var"] = sample_var.astype(np.float64).tolist()
                yhat_by_rep = preds.reshape(preds.shape[0], preds.shape[1]).astype(np.float64)
                eta_flat = eta_test.reshape(-1).astype(np.float64)
                k = min(int(YHAT_HIST_MAX_PANELS), int(sample_bias2.shape[0]))
                if k > 0:
                    sorted_idx = np.argsort(sample_bias2)
                    pick_pos = np.linspace(0, sorted_idx.shape[0] - 1, k, dtype=np.int64)
                    hist_sample_idx = sorted_idx[pick_pos].astype(np.int64)
                else:
                    hist_sample_idx = np.array([], dtype=np.int64)
                bias_var_payload["test"]["y_hat_by_rep"] = yhat_by_rep.tolist()
                bias_var_payload["test"]["eta_test"] = eta_flat.tolist()
                bias_var_payload["test"]["yhat_hist_sample_indices"] = hist_sample_idx.tolist()
        else:
            summary["test"]["bias2"] = {"mean": float("nan"), "std": float("nan"), "var": float("nan")}
            summary["test"]["var"] = {"mean": float("nan"), "std": float("nan"), "var": float("nan")}
            summary["test"]["bias2_sample"] = {"mean": float("nan"), "std": float("nan"), "var": float("nan")}
            summary["test"]["var_sample"] = {"mean": float("nan"), "std": float("nan"), "var": float("nan")}
    summary["train"]["regime_expert_percent"] = train_regime_mean.tolist()
    summary["train"]["regime_expert_percent_std"] = train_regime_std.tolist()
    summary["test"]["regime_expert_percent"] = test_regime_mean.tolist()
    summary["test"]["regime_expert_percent_std"] = test_regime_std.tolist()

    # ---- mean±sd loss curves ----
    max_epoch = max(
        max(len(rr["train_losses"]) for rr in rep_results),
        max(len(rr["val_losses"]) for rr in rep_results),
        max(len(rr["test_losses"]) for rr in rep_results),
    )
    train_mat = _stack_curves([rr["train_losses"] for rr in rep_results], max_len=max_epoch)
    val_mat = _stack_curves([rr["val_losses"] for rr in rep_results], max_len=max_epoch)
    test_mat = _stack_curves([rr["test_losses"] for rr in rep_results], max_len=max_epoch)

    save_loss_curves_with_sd(train_mat, val_mat, plots_dir / "loss_curves_mean_sd.png", test_losses=test_mat)
    save_loss_curves_with_sd(train_mat, val_mat, plots_dir / "loss_curves_mean_sd_ylog.png", test_losses=test_mat, yscale="log")

    # ---- aggregate routing stats over reps ----

    regime_freqs = np.array([rr["test_metrics"]["regime_expert_freq"] for rr in rep_results], dtype=np.float64)
    regime_mean = torch.tensor(np.nanmean(regime_freqs, axis=0), dtype=torch.float32)

    if cfg.train.test_routing == "fixed":
        r2e_argmax = np.argmax(regime_mean.numpy(), axis=1)
        r2e_hungarian = hungarian_max_assignment(regime_mean.numpy())
        summary["test_1 to 1match"] = {
            "method": "hungarian_max_on_regime_expert_freq",
            "index_base": 1,
            "regime_to_expert": {f"regime{i+1}": f"expert{int(e)+1}" for i, e in enumerate(r2e_hungarian.tolist())},
        }
        summary["test_allow conflict"] = {
            "method": "per_regime_argmax_on_regime_expert_freq",
            "index_base": 1,
            "regime_to_expert": {f"regime{i+1}": f"expert{int(e)+1}" for i, e in enumerate(r2e_argmax.tolist())},
        }

    if is_regression:
        reg_cfg = cfg.data2 if cfg.data_variant == "data2" else cfg.data3
        fixed_seed = int(reg_cfg.seed)
        if cfg.data_variant == "data2":
            x_zx, _y_zx, regime_zx = generate_data2(cfg.data2, "train", fixed_seed)
            zx = estimate_eta_data2_analytic_from_x(cfg.data2, x_zx, regime_zx, fixed_seed)
        else:
            x_zx, _y_zx, _regime_zx = generate_data3(cfg.data3, "train", fixed_seed)
            zx = estimate_eta_data3_analytic_from_x(cfg.data3, x_zx, fixed_seed)
        zx_vals = np.asarray(zx, dtype=np.float64).ravel()
        summary[f"{cfg.data_variant}_zx"] = {
            "split": "train",
            "seed": fixed_seed,
            "mean": float(zx_vals.mean()) if zx_vals.size else float("nan"),
            "var": float(zx_vals.var()) if zx_vals.size else float("nan"),
        }
        save_histogram(
            zx_vals,
            plots_dir / f"{cfg.data_variant}_zx_hist.png",
            title=f"{cfg.data_variant} z(x) histogram (train split)",
            xlabel="z(x)",
            ylabel="count",
            bins=50,
        )

    # ---- reorder key display for train/test ----
    def _reorder_metrics(d: Dict[str, Any], prefix: List[str]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k in prefix:
            if k in d:
                out[k] = d[k]
        for k, v in d.items():
            if k not in out:
                out[k] = v
        return out

    if is_regression:
        order = ["loss", "mse", "bias2", "bias2_sample", "var", "var_sample", "mae", "r2"]
    else:
        order = ["loss", "bias2", "bias2_sample", "var", "var_sample", "acc", "macro_f1"]
    summary["train"] = _reorder_metrics(summary["train"], order)
    summary["test"] = _reorder_metrics(summary["test"], order)

    # ---- classification probability boxplots by true class (C x C) ----
    if cfg.data_variant == "data1":
        train_probs_all: List[np.ndarray] = []
        train_true_all: List[np.ndarray] = []
        test_probs_all: List[np.ndarray] = []
        test_true_all: List[np.ndarray] = []
        for r in range(cfg.train.reps):
            rep_seed = int(cfg.train.seed + r * cfg.train.rep_seed_offset)
            train_pack = compute_true_model_probabilities(cfg.data1, rep_seed, split="train")
            test_pack = compute_true_model_probabilities(cfg.data1, rep_seed, split="test")

            p_tr = np.asarray(train_pack["probs"], dtype=np.float64)
            y_tr = np.asarray(train_pack["y_true"], dtype=np.int64).reshape(-1)
            if p_tr.ndim == 2 and p_tr.shape[0] == y_tr.shape[0]:
                train_probs_all.append(p_tr)
                train_true_all.append(y_tr)

            p_te = np.asarray(test_pack["probs"], dtype=np.float64)
            y_te = np.asarray(test_pack["y_true"], dtype=np.int64).reshape(-1)
            if p_te.ndim == 2 and p_te.shape[0] == y_te.shape[0]:
                test_probs_all.append(p_te)
                test_true_all.append(y_te)

        if train_probs_all and train_true_all:
            train_probs_cat = np.concatenate(train_probs_all, axis=0)
            train_true_cat = np.concatenate(train_true_all, axis=0)
            save_prob_boxplot_grid_by_true_class(
                train_probs_cat,
                train_true_cat,
                plots_dir / "data1_generated_prob_boxplot_4x4.png",
                title="Data1 generated/train samples: p(class) grouped by true class (all reps)",
            )
        if test_probs_all and test_true_all:
            test_probs_cat = np.concatenate(test_probs_all, axis=0)
            test_true_cat = np.concatenate(test_true_all, axis=0)
            save_prob_boxplot_grid_by_true_class(
                test_probs_cat,
                test_true_cat,
                plots_dir / "data1_test_prob_boxplot_4x4.png",
                title="Data1 test samples: p(class) grouped by true class (all reps)",
            )

    # ---- save test y_hat and grouped y_hat boxplot ----
    test_yhat_payload: Dict[str, Any] = {
        "data_variant": cfg.data_variant,
        "is_regression": is_regression,
        "enabled": False,
        "reason": "missing_predictions",
        "test": {},
    }
    if not is_regression:
        test_pred_probs = [rr.get("test_pred_prob") for rr in rep_results if rr.get("test_pred_prob") is not None]
        if test_pred_probs:
            probs = np.stack(test_pred_probs, axis=0).astype(np.float64)  # [R,N,C]
            R, N, C = probs.shape
            mean_probs = probs.mean(axis=0)  # [N,C]
            pred_class = np.argmax(mean_probs, axis=1).astype(np.int64)  # [N]

            max_groups = int(min(4, C))
            values_by_group: List[np.ndarray] = []
            group_names: List[str] = []
            for c in range(max_groups):
                mask = pred_class == c
                vals = probs[:, mask, c].reshape(-1)
                values_by_group.append(vals)
                group_names.append(f"class {c} (n={int(mask.sum())})")

            save_yhat_grouped_boxplot(
                values_by_group,
                group_names,
                plots_dir / "yhat_grouped_boxplot.png",
                title="test y_hat grouped boxplot (classification)",
                ylabel="predicted probability",
            )

            test_yhat_payload["enabled"] = True
            test_yhat_payload["reason"] = "ok"
            test_yhat_payload["test"] = {
                "num_reps": int(R),
                "num_samples": int(N),
                "num_classes": int(C),
                "y_hat_prob_by_rep": probs.tolist(),
                "y_hat_class_by_rep": np.argmax(probs, axis=2).astype(np.int64).tolist(),
                "group_ref": "pred_class_from_mean_prob",
                "group_ids": pred_class.tolist(),
                "group_names": group_names,
            }
    else:
        test_preds = [rr.get("test_pred") for rr in rep_results if rr.get("test_pred") is not None]
        if test_preds:
            preds = np.stack(test_preds, axis=0).astype(np.float64)  # [R,N,1]
            yhat = preds.reshape(preds.shape[0], preds.shape[1])      # [R,N]
            R, N = yhat.shape
            mean_yhat = yhat.mean(axis=0)
            q25, q50, q75 = np.percentile(mean_yhat, [25, 50, 75])
            group_ids = np.digitize(mean_yhat, [q25, q50, q75]).astype(np.int64)  # 0,1,2,3

            values_by_group = []
            group_names = []
            for g in range(4):
                mask = group_ids == g
                vals = yhat[:, mask].reshape(-1)
                values_by_group.append(vals)
                if g == 0:
                    group_names.append(f"Q1 (n={int(mask.sum())})")
                elif g == 1:
                    group_names.append(f"Q2 (n={int(mask.sum())})")
                elif g == 2:
                    group_names.append(f"Q3 (n={int(mask.sum())})")
                else:
                    group_names.append(f"Q4 (n={int(mask.sum())})")

            save_yhat_grouped_boxplot(
                values_by_group,
                group_names,
                plots_dir / "yhat_grouped_boxplot.png",
                title="test y_hat grouped boxplot (regression quartiles)",
                ylabel="y_hat",
            )

            test_yhat_payload["enabled"] = True
            test_yhat_payload["reason"] = "ok"
            test_yhat_payload["test"] = {
                "num_reps": int(R),
                "num_samples": int(N),
                "y_hat_by_rep": yhat.tolist(),
                "y_hat_mean": mean_yhat.tolist(),
                "group_ref": "mean_y_hat_quartile",
                "group_ids": group_ids.tolist(),
                "group_names": group_names,
            }

    save_json(loss_log_dir / "test_yhat.json", test_yhat_payload)

    save_json(loss_log_dir / "bias_var.json", bias_var_payload)
    if bool(bias_var_payload.get("enabled", False)):
        bv_test = bias_var_payload.get("test", {})
        rep_bias2_plot = np.asarray(bv_test.get("rep_bias2", []), dtype=np.float64)
        rep_var_plot = np.asarray(bv_test.get("rep_var", []), dtype=np.float64)
        sample_bias2_plot = np.asarray(bv_test.get("sample_bias2", []), dtype=np.float64)
        sample_var_plot = np.asarray(bv_test.get("sample_var", []), dtype=np.float64)

        # Plot: sample-level grouped boxplot
        if not is_regression:
            y_true_test_plot = np.asarray(bv_test.get("y_true_test", []), dtype=np.int64)
            num_classes_plot = int(cfg.active_data().num_classes)
            if y_true_test_plot.size > 0 and sample_bias2_plot.size > 0:
                group_names_plot = [f"class {c}" for c in range(num_classes_plot)]
                save_biasvar_sample_grouped_plot(
                    sample_bias2_plot,
                    sample_var_plot,
                    y_true_test_plot,
                    group_names_plot,
                    plots_dir / "biasvar_sample_grouped.png",
                    title="Bias\u00b2 and Var by true class",
                )
        else:
            eta_test_plot = np.asarray(bv_test.get("eta_test", []), dtype=np.float64)
            if eta_test_plot.size > 0 and sample_bias2_plot.size > 0:
                q25, q50, q75 = np.percentile(eta_test_plot, [25, 50, 75])
                group_ids_plot = np.digitize(eta_test_plot, [q25, q50, q75])  # 0,1,2,3
                group_names_plot = [
                    f"Q1 (\u03b7\u2264{q25:.3f})",
                    f"Q2 ({q25:.3f}<\u03b7\u2264{q50:.3f})",
                    f"Q3 ({q50:.3f}<\u03b7\u2264{q75:.3f})",
                    f"Q4 (\u03b7>{q75:.3f})",
                ]
                save_biasvar_sample_grouped_plot(
                    sample_bias2_plot,
                    sample_var_plot,
                    group_ids_plot,
                    group_names_plot,
                    plots_dir / "biasvar_sample_grouped.png",
                    title="Bias\u00b2 and Var by \u03b7(x) quartile",
                )

    save_json(run_dir / "summary.json", summary)

    print(f"\n[{now_str()}] DONE. Summary saved to: {run_dir / 'summary.json'}")
    print(f"[{now_str()}] Plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
