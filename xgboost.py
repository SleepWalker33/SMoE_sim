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


def _import_xgboost():
    """Avoid local file shadowing the xgboost package."""
    import importlib
    import sys

    removed = []
    for p in ["", str(Path(__file__).resolve().parent)]:
        if p in sys.path:
            sys.path.remove(p)
            removed.append(p)
    try:
        return importlib.import_module("xgboost")
    finally:
        for p in reversed(removed):
            sys.path.insert(0, p)


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
        # XGBoost has no experts; keep shapes aligned with MoE metrics.
        "expert_entropy": 0.0,
        "expert_gini": 0.0,
        "mi_expert_regime": 0.0,
        "label_expert_freq": np.zeros((C, E), dtype=np.float32).tolist(),
        "regime_expert_freq": np.zeros((R, E), dtype=np.float32).tolist(),
        "cka_matrix": np.zeros((E, E), dtype=np.float32).tolist(),
    }
    return metrics


def _evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
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


def _train_one_rep(
    cfg: ProjectConfig,
    rep_idx: int,
    run_dir: Path,
    xgboost_params: Dict[str, Any],
    num_boost_round: int,
    early_stopping_rounds: int,
) -> Dict[str, Any]:
    xgb = _import_xgboost()

    rep_seed = cfg.train.seed + rep_idx * cfg.train.rep_seed_offset
    set_seed(rep_seed)

    x_train, y_train, x_val, y_val, x_test, y_test = _load_splits(cfg, seed=rep_seed)
    is_regression = cfg.data_variant in ("data2", "data3")
    if is_regression:
        y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
        y_val = np.asarray(y_val, dtype=np.float64).reshape(-1)
        y_test = np.asarray(y_test, dtype=np.float64).reshape(-1)

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)
    dtest = xgb.DMatrix(x_test, label=y_test)

    evals_result: Dict[str, Dict[str, List[float]]] = {}
    bst = xgb.train(
        params=xgboost_params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=False,
    )

    if is_regression:
        train_losses = evals_result.get("train", {}).get("rmse", [])
        val_losses = evals_result.get("val", {}).get("rmse", [])
    else:
        train_losses = evals_result.get("train", {}).get("mlogloss", [])
        val_losses = evals_result.get("val", {}).get("mlogloss", [])
    best_val = float(min(val_losses)) if val_losses else float("nan")
    best_epoch = int(np.argmin(val_losses)) if val_losses else -1

    test_pred = bst.predict(dtest)
    if is_regression:
        test_metrics = _evaluate_regression(y_test, test_pred)
    else:
        if test_pred.ndim == 1:
            test_pred = _softmax_np(test_pred.reshape(-1, 1))
        test_metrics = _evaluate(cfg, test_pred, y_test)

    return {
        "rep_idx": rep_idx,
        "rep_seed": rep_seed,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val": best_val,
        "best_epoch": best_epoch,
        "test_metrics": test_metrics,
    }


def _tune_xgboost(
    cfg: ProjectConfig,
    n_trials: int,
    early_stopping_rounds: int,
) -> Tuple[Dict[str, Any], int, Dict[str, Any]]:
    try:
        import optuna
    except Exception as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("Optuna 未安装，请先安装后再使用 --tune。") from exc
    xgb = _import_xgboost()

    x_train, y_train, x_val, y_val, _, _ = _load_splits(cfg, seed=cfg.train.seed)
    is_regression = cfg.data_variant in ("data2", "data3")
    if is_regression:
        y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
        y_val = np.asarray(y_val, dtype=np.float64).reshape(-1)
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "seed": cfg.train.seed,
            "nthread": cfg.train.num_threads,
            "tree_method": "hist",
        }
        if is_regression:
            params["objective"] = "reg:squarederror"
            params["eval_metric"] = "rmse"
        else:
            params["objective"] = "multi:softprob"
            params["num_class"] = cfg.model.num_classes
            params["eval_metric"] = "mlogloss"
        num_boost_round = trial.suggest_int("num_boost_round", 100, 400)

        evals_result: Dict[str, Dict[str, List[float]]] = {}
        xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=[(dval, "val")],
            early_stopping_rounds=early_stopping_rounds,
            evals_result=evals_result,
            verbose_eval=False,
        )
        val_losses = evals_result.get("val", {}).get("rmse" if is_regression else "mlogloss", [])
        return float(min(val_losses)) if val_losses else float("inf")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_trial = study.best_trial
    worst_trial = max(study.trials, key=lambda t: t.value if t.value is not None else -1.0)

    best_params = {
        "objective": "reg:squarederror" if is_regression else "multi:softprob",
        "eval_metric": "rmse" if is_regression else "mlogloss",
        "seed": cfg.train.seed,
        "nthread": cfg.train.num_threads,
        "tree_method": "hist",
    }
    if not is_regression:
        best_params["num_class"] = cfg.model.num_classes
    best_params.update({k: v for k, v in best_trial.params.items() if k != "num_boost_round"})
    best_num_boost_round = int(best_trial.params.get("num_boost_round", 200))

    tune_summary = {
        "direction": "minimize",
        "metric": "val_rmse" if is_regression else "val_mlogloss",
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
    return best_params, best_num_boost_round, tune_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-variant", choices=["data1", "data2", "data3"], default=None, help="选择 data1 / data2 / data3")
    parser.add_argument("--num-boost-round", type=int, default=200)
    parser.add_argument("--early-stopping-rounds", type=int, default=20)
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
    run_dir = ensure_dir(Path(cfg.train.output_dir) / f"{cfg.train.experiment_name}_xgboost" / run_id)
    ensure_dir(run_dir / "plots")
    save_json(run_dir / "config.json", cfg.to_dict())

    is_regression = cfg.data_variant in ("data2", "data3")
    xgboost_params = {
        "objective": "reg:squarederror" if is_regression else "multi:softprob",
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "eval_metric": "rmse" if is_regression else "mlogloss",
        "seed": cfg.train.seed,
        "nthread": cfg.train.num_threads,
    }
    if not is_regression:
        xgboost_params["num_class"] = cfg.model.num_classes
    num_boost_round = args.num_boost_round

    if args.tune:
        best_params, best_num_boost_round, tune_summary = _tune_xgboost(
            cfg=cfg,
            n_trials=args.n_trials,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        xgboost_params = best_params
        num_boost_round = best_num_boost_round
        print(f"[{now_str()}] Optuna 最优: {tune_summary['best']}")
        print(f"[{now_str()}] Optuna 最差: {tune_summary['worst']}")

    rep_results: List[Dict[str, Any]] = []
    for r in range(cfg.train.reps):
        print(f"\n[{now_str()}] ====== REP {r+1}/{cfg.train.reps} ======")
        rep_res = _train_one_rep(
            cfg=cfg,
            rep_idx=r,
            run_dir=run_dir,
            xgboost_params=xgboost_params,
            num_boost_round=num_boost_round,
            early_stopping_rounds=args.early_stopping_rounds,
        )
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
