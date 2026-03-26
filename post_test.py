# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Literal

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn.functional as F

from config import ProjectConfig, make_default_config
from data import (
    estimate_eta_data1_mc,
    estimate_eta_data1_mc_from_x,
    estimate_eta_data2_analytic_from_x,
    estimate_eta_data3_analytic_from_x,
    generate_data1,
    generate_data2,
    generate_data3,
    make_loaders as make_loaders_data,
)
from metrics import (
    accumulate_label_expert_counts,
    accumulate_regime_expert_counts,
    confusion_matrix,
    expert_load_gini,
    expert_usage_entropy,
    label_expert_frequencies,
    macro_f1_from_cm,
    mutual_info_expert_regime,
    per_class_accuracy,
    regime_expert_frequencies,
)
from model import FNN_CMR_MoE, RandomGate
from trainer import (
    _count_params,
    _effective_param_count,
    _expert_param_count,
    _expert_weight_cosine,
    _forward_flops_batch,
    _gate_cosine,
)
from utils import ensure_dir, now_str, save_json, set_seed, setup_cpu, to_device


# ======= Post-Test Editable Defaults =======
# Default run model directory (contains rep1/rep2/.../best.pth)
DEFAULT_MODEL_DIR: str = "/userhome/home/yangdandan/ydd/MoE_sim/20260328/baseline2/runs/exp_data1/20260319_205640_lr0.01/model"
# Routing/eval mode choices: "trained" | "random_topk" | "random_top1" | "random_top0"
POST_TEST_GATE_MODE: Literal["trained", "random_topk", "random_top1", "random_top0"] = "trained"
# Whether to use shared expert branch at test time
POST_TEST_USE_SHARE_EXPERT: bool = True
# Optional overrides: <=0 means use value from config.json
POST_TEST_BATCH_SIZE: int = 0
POST_TEST_NUM_WORKERS: int = 0
# data1 bias-variance MC samples (used only for eta estimation in bias/var metrics)
ETA_MC_SAMPLES = 200


def _str2bool(v: str) -> bool:
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return True
    if s in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {v}")


def _mean_std_dict(xs: List[float]) -> Dict[str, float]:
    x = np.array(xs, dtype=np.float64)
    if x.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "var": float("nan")}
    if x.size == 1:
        return {"mean": float(x.mean()), "std": 0.0, "var": 0.0}
    return {"mean": float(x.mean()), "std": float(x.std(ddof=1)), "var": float(x.var(ddof=1))}


def _update_obj_fields(obj: Any, kv: Dict[str, Any]) -> None:
    for k, v in kv.items():
        if hasattr(obj, k):
            setattr(obj, k, v)


def _load_cfg_from_json(config_path: Path) -> ProjectConfig:
    cfg = make_default_config()
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    if "data_variant" in d:
        cfg.data_variant = d["data_variant"]
    _update_obj_fields(cfg.data1, d.get("data1", {}))
    _update_obj_fields(cfg.data2, d.get("data2", {}))
    _update_obj_fields(cfg.data3, d.get("data3", {}))
    _update_obj_fields(cfg.model, d.get("model", {}))
    _update_obj_fields(cfg.train, d.get("train", {}))
    cfg.sync_model_to_data()
    cfg.train.device = "cpu"
    return cfg


def _collect_rep_dirs(model_dir: Path) -> List[Tuple[int, Path]]:
    reps: List[Tuple[int, Path]] = []
    for p in model_dir.iterdir():
        if not p.is_dir():
            continue
        m = re.fullmatch(r"rep(\d+)", p.name)
        if m is None:
            continue
        rep_num = int(m.group(1))
        ckpt = p / "best.pth"
        if ckpt.exists():
            reps.append((rep_num, p))
    reps.sort(key=lambda x: x[0])
    return reps


@torch.no_grad()
def _evaluate_test_with_mode(
    cfg: ProjectConfig,
    model: torch.nn.Module,
    loader,
    device: str,
    *,
    shared_only: bool,
    moe_only: bool,
    force_top1: bool,
    compute_flops: bool = True,
) -> Dict[str, Any]:
    model.eval()

    C = int(cfg.model.num_classes)
    E = int(cfg.model.num_experts)
    R = int(cfg.model.num_regimes)
    is_regression = cfg.data_variant in ("data2", "data3")

    losses: List[float] = []
    ents: List[float] = []
    ginis: List[float] = []
    mis: List[float] = []
    active_counts: List[float] = []
    total_samples = 0
    total_steps = 0
    t0 = time.time()
    flops_total = 0.0
    sse = 0.0
    sae = 0.0
    sum_y = 0.0
    sum_y2 = 0.0
    count_y = 0

    label_counts = torch.zeros((C, E), device=device)
    label_totals = torch.zeros((C,), device=device)
    regime_counts = torch.zeros((R, E), device=device)
    regime_totals = torch.zeros((R,), device=device)

    all_pred = []
    all_true = []
    pred_probs = []
    pred_vals = []

    if cfg.train.test_routing == "fixed" and E < R:
        raise ValueError("test_routing=fixed requires num_experts >= num_regimes")

    for x, y, regime in loader:
        total_steps += 1
        total_samples += int(x.shape[0])
        x, y, regime = to_device(x, device), to_device(y, device), to_device(regime, device)

        forced_expert = regime if cfg.train.test_routing == "fixed" else None
        logits, aux = model(
            x,
            forced_expert=forced_expert,
            shared_only=shared_only,
            moe_only=moe_only,
            force_top1=force_top1,
            force_avg_fuse=(cfg.train.test_routing == "fixed"),
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
            pred_vals.append(logits.detach().cpu().numpy())
        else:
            losses.append(float(F.cross_entropy(logits, y).item()))
            p = F.softmax(logits, dim=-1)
            pred_probs.append(p.detach().cpu().numpy())
            pred = logits.argmax(dim=-1)
            all_pred.append(pred.detach().cpu())
            all_true.append(y.detach().cpu())

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
        all_pred_t = torch.cat(all_pred, dim=0) if all_pred else torch.zeros((0,), dtype=torch.long)
        all_true_t = torch.cat(all_true, dim=0) if all_true else torch.zeros((0,), dtype=torch.long)
        cm = confusion_matrix(all_pred_t, all_true_t, num_classes=C)
        pacc = per_class_accuracy(cm)
        mf1 = macro_f1_from_cm(cm)
        micro_acc = float(cm.diag().sum().item() / max(cm.sum().item(), 1))
    else:
        cm = torch.zeros((C, C), dtype=torch.long)
        pacc = torch.zeros((C,), dtype=torch.float32)
        mf1 = 0.0
        micro_acc = 0.0

    if not is_regression and label_totals.sum() > 0:
        label_freq = label_expert_frequencies(label_counts, label_totals)
    else:
        label_freq = torch.zeros((C, E), device=device)

    if regime_totals.sum() > 0:
        regime_freq = regime_expert_frequencies(regime_counts, regime_totals)
    else:
        regime_freq = torch.zeros((R, E), device=device)

    dt = time.time() - t0
    samples_per_sec = float(total_samples / dt) if dt > 0 else 0.0
    avg_step_time = float(dt / max(total_steps, 1))
    active_ratio = float(np.mean(active_counts) / max(E, 1)) if active_counts else 0.0
    expert_entropy = float(np.mean(ents)) if ents else 0.0
    expert_gini = float(np.mean(ginis)) if ginis else 0.0
    mi_val = float(np.mean(mis)) if mis else 0.0

    metrics: Dict[str, Any] = {
        "split": "test",
        "loss": float(np.mean(losses)) if losses else 0.0,
        "acc": micro_acc,
        "macro_f1": float(mf1),
        "expert_cosine": _expert_weight_cosine(model),
        "per_class_acc": pacc.tolist(),
        "confusion_matrix": cm.tolist(),
        "expert_entropy": expert_entropy,
        "expert_gini": expert_gini,
        "mi_expert_regime": mi_val,
        "samples_per_sec": samples_per_sec,
        "avg_step_time": avg_step_time,
        "total_time_sec": float(dt),
        "flops_total": float(flops_total),
        "active_expert_ratio": active_ratio,
        "label_expert_freq": label_freq.detach().cpu().tolist(),
        "regime_expert_freq": regime_freq.detach().cpu().tolist(),
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
        if pred_vals:
            metrics["pred_value"] = np.concatenate(pred_vals, axis=0).astype(np.float64).tolist()
    else:
        if pred_probs:
            metrics["pred_prob"] = np.concatenate(pred_probs, axis=0).astype(np.float64).tolist()
    return metrics


def _build_mode_flags(gate_mode: str, use_share_expert: bool) -> Tuple[bool, bool, bool]:
    if gate_mode == "random_top0":
        return True, False, False
    if gate_mode == "random_top1":
        return (False, not use_share_expert, True)
    if gate_mode == "random_topk":
        return (False, not use_share_expert, False)
    # trained
    return (False, not use_share_expert, False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc test for saved rep checkpoints.")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR, help="Path to .../model directory containing rep1/rep2/... folders.")
    parser.add_argument("--config_path", type=str, default="", help="Path to run config.json. Default: <model_dir>/../config.json")
    parser.add_argument("--gate_mode", type=str, default=POST_TEST_GATE_MODE, choices=["trained", "random_topk", "random_top1", "random_top0"])
    parser.add_argument("--use_share_expert", type=_str2bool, default=POST_TEST_USE_SHARE_EXPERT, help="Whether to use shared expert branch at test time.")
    parser.add_argument("--batch_size", type=int, default=POST_TEST_BATCH_SIZE, help="Override test batch size if > 0.")
    parser.add_argument("--num_workers", type=int, default=POST_TEST_NUM_WORKERS, help="Override dataloader workers if > 0.")
    parser.add_argument("--eta_mc_samples", type=int, default=ETA_MC_SAMPLES, help="MC samples for data1 eta estimation.")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"model_dir not found: {model_dir}")

    run_dir = model_dir.parent
    config_path = Path(args.config_path).expanduser().resolve() if args.config_path else (run_dir / "config.json").resolve()
    cfg = _load_cfg_from_json(config_path)
    if cfg.train.test_routing not in ("score", "fixed"):
        raise ValueError(f"Unsupported test_routing={cfg.train.test_routing!r}; expected 'score' or 'fixed'.")

    if args.batch_size > 0:
        cfg.train.batch_size = int(args.batch_size)
    if args.num_workers > 0:
        cfg.train.num_workers = int(args.num_workers)

    setup_cpu(cfg.train.num_threads)
    set_seed(cfg.train.seed)
    cfg.train.device = "cpu"
    device = "cpu"

    reps = _collect_rep_dirs(model_dir)
    if not reps:
        raise RuntimeError(f"No rep*/best.pth found under: {model_dir}")

    post_dir = ensure_dir(run_dir / "post_test")
    mode_tag = f"{args.gate_mode}_share{1 if args.use_share_expert else 0}"

    all_test_metrics: List[Dict[str, Any]] = []
    rep_meta: List[Dict[str, Any]] = []
    is_regression = cfg.data_variant in ("data2", "data3")

    shared_only, moe_only, force_top1 = _build_mode_flags(args.gate_mode, args.use_share_expert)

    for rep_num, rep_dir in reps:
        rep_idx = rep_num - 1
        rep_seed = int(cfg.train.seed + rep_idx * cfg.train.rep_seed_offset)
        data_cfg = cfg.active_data()
        _train_loader, _val_loader, test_loader = make_loaders_data(
            data_cfg,
            batch_size=int(cfg.train.batch_size),
            num_workers=int(cfg.train.num_workers),
            seed=rep_seed,
            data_variant=cfg.data_variant,
        )

        model = FNN_CMR_MoE(cfg.model).to(device)
        ckpt_path = rep_dir / "best.pth"
        state = torch.load(str(ckpt_path), map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        model.eval()

        cfg_eval = copy.deepcopy(cfg)
        if args.gate_mode in ("random_topk", "random_top1") and getattr(model, "moe", None) is not None:
            model.moe.gate = RandomGate(cfg.model.num_experts).to(device)
            cfg_eval.model.gate_type = "random"

        if args.gate_mode == "random_top0" and (not args.use_share_expert):
            model.shared2 = None
            cfg_eval.model.gate1_constant = 0.0

        testm = _evaluate_test_with_mode(
            cfg_eval,
            model,
            test_loader,
            device,
            shared_only=shared_only,
            moe_only=moe_only,
            force_top1=force_top1,
            compute_flops=True,
        )

        total_params = _count_params(model)
        expert_params = _expert_param_count(model)
        testm["total_params"] = total_params
        testm["active_params"] = _effective_param_count(total_params, expert_params, testm.get("active_expert_ratio", 0.0))
        testm["expert_gate_cosine"] = _gate_cosine(model)
        all_test_metrics.append(testm)
        rep_meta.append(
            {
                "rep": int(rep_num),
                "rep_seed": int(rep_seed),
                "checkpoint": str(ckpt_path),
                "missing_keys": list(missing),
                "unexpected_keys": list(unexpected),
            }
        )
        print(f"[{now_str()}] post_test rep={rep_num} loss={float(testm['loss']):.6f} acc={float(testm.get('acc', 0.0)):.6f}")

    def collect(k: str) -> List[Any]:
        return [m[k] for m in all_test_metrics]

    test_summary: Dict[str, Any] = {
        "loss": _mean_std_dict([float(v) for v in collect("loss")]),
        "expert_cosine": _mean_std_dict([float(v) for v in collect("expert_cosine")]),
        "expert_gate_cosine": _mean_std_dict([float(v) for v in collect("expert_gate_cosine")]),
        "expert_entropy": _mean_std_dict([float(v) for v in collect("expert_entropy")]),
        "expert_gini": _mean_std_dict([float(v) for v in collect("expert_gini")]),
        "samples_per_sec": _mean_std_dict([float(v) for v in collect("samples_per_sec")]),
        "avg_step_time": _mean_std_dict([float(v) for v in collect("avg_step_time")]),
        "total_test_time_sec": _mean_std_dict([float(v) for v in collect("total_time_sec")]),
        "flops_total": _mean_std_dict([float(v) for v in collect("flops_total")]),
        "active_expert_ratio": _mean_std_dict([float(v) for v in collect("active_expert_ratio")]),
        "total_params": _mean_std_dict([float(v) for v in collect("total_params")]),
        "active_params": _mean_std_dict([float(v) for v in collect("active_params")]),
        "regime_expert_percent": (np.nanmean(np.array(collect("regime_expert_freq"), dtype=np.float64), axis=0) * 100.0).tolist(),
        "regime_expert_percent_std": (
            np.nanstd(np.array(collect("regime_expert_freq"), dtype=np.float64), axis=0, ddof=1) * 100.0
            if len(all_test_metrics) > 1
            else np.zeros_like(np.nanmean(np.array(collect("regime_expert_freq"), dtype=np.float64), axis=0))
        ).tolist(),
    }

    if not is_regression:
        per_class_acc = np.array(collect("per_class_acc"), dtype=np.float64)
        label_freq = np.array(collect("label_expert_freq"), dtype=np.float64)
        test_summary["acc"] = _mean_std_dict([float(v) for v in collect("acc")])
        test_summary["macro_f1"] = _mean_std_dict([float(v) for v in collect("macro_f1")])
        test_summary["per_class_acc_mean"] = np.nanmean(per_class_acc, axis=0).tolist()
        test_summary["per_class_acc_std"] = (
            np.nanstd(per_class_acc, axis=0, ddof=1) if len(all_test_metrics) > 1 else np.zeros_like(np.nanmean(per_class_acc, axis=0))
        ).tolist()
        test_summary["label_expert_percent"] = (np.nanmean(label_freq, axis=0) * 100.0).tolist()
        test_summary["label_expert_percent_std"] = (
            np.nanstd(label_freq, axis=0, ddof=1) * 100.0 if len(all_test_metrics) > 1 else np.zeros_like(np.nanmean(label_freq, axis=0))
        ).tolist()

        if getattr(cfg.data1, "fixed_test_set", False):
            fixed_seed = int(cfg.data1.seed)
            x_test, _y_test, _regime_test = generate_data1(cfg.data1, "test", fixed_seed)
            if getattr(cfg.data1, "input_mode", "x") == "x":
                eta_test = estimate_eta_data1_mc_from_x(cfg.data1, x_test, fixed_seed, int(args.eta_mc_samples))
            else:
                eta_test = estimate_eta_data1_mc(cfg.data1, "test", fixed_seed, int(args.eta_mc_samples))
            preds = np.stack([np.asarray(m.get("pred_prob"), dtype=np.float64) for m in all_test_metrics], axis=0)
            mean_pred = preds.mean(axis=0)
            var_pred = preds.var(axis=0)
            rep_bias2 = np.mean(np.sum((preds - eta_test[None, :, :]) ** 2, axis=2), axis=1)
            rep_var = np.mean(np.sum((preds - mean_pred[None, :, :]) ** 2, axis=2), axis=1)
            sample_bias2 = np.sum((mean_pred - eta_test) ** 2, axis=1)
            sample_var = np.sum(var_pred, axis=1)
            test_summary["bias2"] = _mean_std_dict(rep_bias2.tolist())
            test_summary["var"] = _mean_std_dict(rep_var.tolist())
            test_summary["bias2_sample"] = _mean_std_dict(sample_bias2.tolist())
            test_summary["var_sample"] = _mean_std_dict(sample_var.tolist())
        else:
            test_summary["bias2"] = {"mean": float("nan"), "std": float("nan"), "var": float("nan")}
            test_summary["var"] = {"mean": float("nan"), "std": float("nan"), "var": float("nan")}
            test_summary["bias2_sample"] = {"mean": float("nan"), "std": float("nan"), "var": float("nan")}
            test_summary["var_sample"] = {"mean": float("nan"), "std": float("nan"), "var": float("nan")}
    else:
        test_summary["mse"] = _mean_std_dict([float(v) for v in collect("mse")])
        test_summary["mae"] = _mean_std_dict([float(v) for v in collect("mae")])
        test_summary["r2"] = _mean_std_dict([float(v) for v in collect("r2")])

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
            preds = np.stack([np.asarray(m.get("pred_value"), dtype=np.float64) for m in all_test_metrics], axis=0)
            mean_pred = preds.mean(axis=0)
            var_pred = preds.var(axis=0)
            rep_bias2 = np.mean((preds - eta_test[None, :, :]) ** 2, axis=1).reshape(-1)
            rep_var = np.mean((preds - mean_pred[None, :, :]) ** 2, axis=1).reshape(-1)
            sample_bias2 = ((mean_pred - eta_test) ** 2).reshape(-1)
            sample_var = var_pred.reshape(-1)
            test_summary["bias2"] = _mean_std_dict(rep_bias2.tolist())
            test_summary["var"] = _mean_std_dict(rep_var.tolist())
            test_summary["bias2_sample"] = _mean_std_dict(sample_bias2.tolist())
            test_summary["var_sample"] = _mean_std_dict(sample_var.tolist())
        else:
            test_summary["bias2"] = {"mean": float("nan"), "std": float("nan"), "var": float("nan")}
            test_summary["var"] = {"mean": float("nan"), "std": float("nan"), "var": float("nan")}
            test_summary["bias2_sample"] = {"mean": float("nan"), "std": float("nan"), "var": float("nan")}
            test_summary["var_sample"] = {"mean": float("nan"), "std": float("nan"), "var": float("nan")}

    out_obj = {
        "post_test": {
            "time": now_str(),
            "model_dir": str(model_dir),
            "config_path": str(config_path),
            "data_variant": cfg.data_variant,
            "gate_mode": args.gate_mode,
            "use_share_expert": bool(args.use_share_expert),
            "batch_size": int(cfg.train.batch_size),
            "num_workers": int(cfg.train.num_workers),
            "reps_evaluated": len(all_test_metrics),
        },
        "test": test_summary,
        "rep_meta": rep_meta,
    }

    out_path = post_dir / f"summary_{mode_tag}.json"
    save_json(out_path, out_obj)
    print(f"[{now_str()}] post_test summary saved: {out_path}")


if __name__ == "__main__":
    main()
