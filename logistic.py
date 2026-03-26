# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import torch

from config import make_default_config, ProjectConfig
from data import generate_data1, generate_data2, generate_data3
from metrics import confusion_matrix, per_class_accuracy, macro_f1_from_cm
from utils import now_str, ensure_dir, save_json, setup_cpu, set_seed


def _mean_std_dict(xs: List[float]) -> Dict[str, float]:
    x = np.array(xs, dtype=np.float64)
    if len(x) == 1:
        return {"mean": float(x.mean()), "std": 0.0, "var": 0.0}
    return {"mean": float(x.mean()), "std": float(x.std(ddof=1)), "var": float(x.var(ddof=1))}


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    diff = y_pred - y_true
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - float(np.sum(diff * diff)) / denom if denom > 1e-12 else 0.0
    return {"mse": mse, "mae": mae, "r2": r2}


def _align_probs(
    probs: np.ndarray,
    classes: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Align predict_proba output to full class dimension."""
    if probs.shape[1] == num_classes:
        return probs
    out = np.zeros((probs.shape[0], num_classes), dtype=probs.dtype)
    for idx, c in enumerate(classes.tolist()):
        out[:, int(c)] = probs[:, idx]
    return out


def _cross_entropy_from_probs(probs: np.ndarray, y: np.ndarray) -> float:
    n = probs.shape[0]
    p = probs[np.arange(n), y]
    return float(-np.log(p + 1e-12).mean())


def _evaluate(
    cfg: ProjectConfig,
    probs: np.ndarray,
    y_true: np.ndarray,
) -> Dict[str, Any]:
    C = cfg.model.num_classes
    E = cfg.model.num_experts
    R = cfg.model.num_regimes

    pred = probs.argmax(axis=1)
    cm = confusion_matrix(torch.tensor(pred), torch.tensor(y_true), num_classes=C)
    pacc = per_class_accuracy(cm)
    mf1 = macro_f1_from_cm(cm)

    metrics = {
        "loss": _cross_entropy_from_probs(probs, y_true),
        "acc": float((pred == y_true).mean()),
        "macro_f1": float(mf1),
        "per_class_acc": pacc.tolist(),
        "confusion_matrix": cm.tolist(),
        # Logistic regression has no experts; keep shapes aligned with MoE metrics.
        "expert_entropy": 0.0,
        "expert_gini": 0.0,
        "mi_expert_regime": 0.0,
        "label_expert_freq": np.zeros((C, E), dtype=np.float32).tolist(),
        "regime_expert_freq": np.zeros((R, E), dtype=np.float32).tolist(),
        "cka_matrix": np.zeros((E, E), dtype=np.float32).tolist(),
    }
    return metrics


def _evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    m = _regression_metrics(y_true, y_pred)
    return {
        "loss": m["mse"],
        "mse": m["mse"],
        "mae": m["mae"],
        "r2": m["r2"],
    }




def _load_splits(cfg: ProjectConfig, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data_cfg = cfg.active_data()
    if cfg.data_variant in ("data2", "data3"):
        gen_fn = generate_data2 if cfg.data_variant == "data2" else generate_data3
        train = gen_fn(data_cfg, "train", seed)
        val = gen_fn(data_cfg, "val", seed)
        test = gen_fn(data_cfg, "test", seed)
    else:
        train = generate_data1(data_cfg, "train", seed)
        val = generate_data1(data_cfg, "val", seed)
        test = generate_data1(data_cfg, "test", seed)
    x_train, y_train, _ = train
    x_val, y_val, _ = val
    x_test, y_test, _ = test
    return x_train, y_train, x_val, y_val, x_test, y_test


def _make_model(params: Dict[str, Any], is_regression: bool):
    try:
        if is_regression:
            from sklearn.linear_model import Ridge
        else:
            from sklearn.linear_model import LogisticRegression
    except Exception as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("scikit-learn 未安装，请先安装后再使用 logistic.py。") from exc
    if is_regression:
        return Ridge(alpha=float(params["alpha"]))
    return LogisticRegression(
        penalty="l2",
        C=float(params["C"]),
        max_iter=int(params["max_iter"]),
        solver="lbfgs",
        multi_class="multinomial",
        n_jobs=1,
    )


def _tune_logistic(
    cfg: ProjectConfig,
    n_trials: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    try:
        import optuna
    except Exception as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("Optuna 未安装，请先安装后再使用 --tune。") from exc

    x_train, y_train, x_val, y_val, _, _ = _load_splits(cfg, seed=cfg.train.seed)
    is_regression = cfg.data_variant in ("data2", "data3")
    if is_regression:
        y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
        y_val = np.asarray(y_val, dtype=np.float64).reshape(-1)

    def objective(trial: optuna.Trial) -> float:
        if is_regression:
            params = {"alpha": trial.suggest_float("alpha", 1e-6, 10.0, log=True)}
            model = _make_model(params, is_regression=True)
            model.fit(x_train, y_train)
            pred = model.predict(x_val)
            return float(np.mean((pred - y_val) ** 2))
        params = {
            "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
            "max_iter": trial.suggest_int("max_iter", 200, 1000),
        }
        model = _make_model(params, is_regression=False)
        model.fit(x_train, y_train)
        probs = model.predict_proba(x_val)
        probs = _align_probs(probs, model.classes_, cfg.model.num_classes)
        return _cross_entropy_from_probs(probs, y_val)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_trial = study.best_trial
    worst_trial = max(study.trials, key=lambda t: t.value if t.value is not None else -1.0)

    tune_summary = {
        "direction": "minimize",
        "metric": "val_mse" if is_regression else "val_logloss",
        "n_trials": n_trials,
        "best": {
            "value": float(best_trial.value),
            "params": best_trial.params,
        },
        "worst": {
            "value": float(worst_trial.value),
            "params": worst_trial.params,
        },
    }
    if is_regression:
        best_params = {"alpha": best_trial.params["alpha"]}
    else:
        best_params = {"C": best_trial.params["C"], "max_iter": best_trial.params["max_iter"]}
    return best_params, tune_summary


def _train_one_rep(
    cfg: ProjectConfig,
    rep_idx: int,
    run_dir: Path,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    rep_seed = cfg.train.seed + rep_idx * cfg.train.rep_seed_offset
    set_seed(rep_seed)

    x_train, y_train, x_val, y_val, x_test, y_test = _load_splits(cfg, seed=rep_seed)

    is_regression = cfg.data_variant in ("data2", "data3")
    if is_regression:
        y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
        y_val = np.asarray(y_val, dtype=np.float64).reshape(-1)
        y_test = np.asarray(y_test, dtype=np.float64).reshape(-1)
        model = _make_model(params, is_regression=True)
        model.fit(x_train, y_train)
        train_pred = model.predict(x_train)
        val_pred = model.predict(x_val)
        test_pred = model.predict(x_test)
        train_loss = float(np.mean((train_pred - y_train) ** 2))
        val_loss = float(np.mean((val_pred - y_val) ** 2))
        test_metrics = _evaluate_regression(y_test, test_pred)
    else:
        model = _make_model(params, is_regression=False)
        model.fit(x_train, y_train)
        train_probs = _align_probs(model.predict_proba(x_train), model.classes_, cfg.model.num_classes)
        val_probs = _align_probs(model.predict_proba(x_val), model.classes_, cfg.model.num_classes)
        test_probs = _align_probs(model.predict_proba(x_test), model.classes_, cfg.model.num_classes)
        train_loss = _cross_entropy_from_probs(train_probs, y_train)
        val_loss = _cross_entropy_from_probs(val_probs, y_val)
        test_metrics = _evaluate(cfg, test_probs, y_test)

    return {
        "rep_idx": rep_idx,
        "rep_seed": rep_seed,
        "train_losses": [train_loss],
        "val_losses": [val_loss],
        "best_val": val_loss,
        "best_epoch": 0,
        "test_metrics": test_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-variant", choices=["data1", "data2", "data3"], default=None, help="选择 data1 / data2 / data3")
    parser.add_argument("--tune", action="store_true", help="使用 Optuna 自适应调参")
    parser.add_argument("--n-trials", type=int, default=30, help="Optuna 试验次数")
    args = parser.parse_args()

    cfg = make_default_config()
    if args.data_variant is not None:
        cfg.data_variant = args.data_variant
        cfg.sync_model_to_data()

    setup_cpu(cfg.train.num_threads)
    set_seed(cfg.train.seed)

    run_id = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_dir = ensure_dir(Path(cfg.train.output_dir) / f"{cfg.train.experiment_name}_logistic" / run_id)
    ensure_dir(run_dir / "plots")
    save_json(run_dir / "config.json", cfg.to_dict())

    is_regression = cfg.data_variant in ("data2", "data3")
    params = {"alpha": 1.0} if is_regression else {"C": 1.0, "max_iter": 500}
    if args.tune:
        params, tune_summary = _tune_logistic(cfg, n_trials=args.n_trials)
        print(f"[{now_str()}] Optuna 最优: {tune_summary['best']}")
        print(f"[{now_str()}] Optuna 最差: {tune_summary['worst']}")

    rep_results: List[Dict[str, Any]] = []
    for r in range(cfg.train.reps):
        print(f"\n[{now_str()}] ====== REP {r+1}/{cfg.train.reps} ======")
        rep_res = _train_one_rep(cfg=cfg, rep_idx=r, run_dir=run_dir, params=params)
        rep_results.append(rep_res)

    test_losses = [rr["test_metrics"]["loss"] for rr in rep_results]
    if is_regression:
        test_mse = [rr["test_metrics"]["mse"] for rr in rep_results]
        test_mae = [rr["test_metrics"]["mae"] for rr in rep_results]
        test_r2 = [rr["test_metrics"]["r2"] for rr in rep_results]
        summary = {
            "test_loss": _mean_std_dict(test_losses),
            "test_mse": _mean_std_dict(test_mse),
            "test_mae": _mean_std_dict(test_mae),
            "test_r2": _mean_std_dict(test_r2),
        }
    else:
        test_acc = [rr["test_metrics"]["acc"] for rr in rep_results]
        test_mf1 = [rr["test_metrics"]["macro_f1"] for rr in rep_results]
        test_entropy = [rr["test_metrics"]["expert_entropy"] for rr in rep_results]
        test_gini = [rr["test_metrics"]["expert_gini"] for rr in rep_results]
        test_mi = [rr["test_metrics"]["mi_expert_regime"] for rr in rep_results]

        per_class_acc = np.array([rr["test_metrics"]["per_class_acc"] for rr in rep_results], dtype=np.float64)
        per_class_acc_mean = per_class_acc.mean(axis=0)
        per_class_acc_sd = per_class_acc.std(axis=0, ddof=1) if cfg.train.reps > 1 else np.zeros_like(per_class_acc_mean)

        summary = {
            "test_loss": _mean_std_dict(test_losses),
            "test_acc": _mean_std_dict(test_acc),
            "test_macro_f1": _mean_std_dict(test_mf1),
            "test_expert_entropy": _mean_std_dict(test_entropy),
            "test_expert_gini": _mean_std_dict(test_gini),
            "test_mi_expert_regime": _mean_std_dict(test_mi),
            "per_class_acc_mean": per_class_acc_mean.tolist(),
            "per_class_acc_std": per_class_acc_sd.tolist(),
        }
    save_json(run_dir / "summary.json", summary)

    print(f"\n[{now_str()}] DONE. Summary saved to: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
