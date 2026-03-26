#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot mean losses for multiple runs.
1) train + val (same figure)
2) val + test (same figure)
3) test only
Use solid vs dashed to distinguish two losses; color = setting.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

BASE = Path("/Users/yangdandan/Desktop/document/vscode data/MoE_sim/20260328/baseline2")
EXP_ROOT = BASE / "runs" / "exp_data2"
OUT_DIR = BASE / "plots_data2"

# Only plot epochs that appear in at least MIN_REPS_PER_EPOCH reps.
MIN_REPS_PER_EPOCH = 2

# If you want to plot only selected runs, fill this dict:
# key = label shown in legend, value = run folder name under EXP_ROOT.
# Example:
# RUNS = {
#     "settingA": "20260301_012908",
#     "settingB": "20260301_081530",
# }

RUNS = {
        "e4k2share8":"20260320_134320_e4k2share8_hidden16_noise0*",
        # "e4k2share8_l1":"20260320_212248_e4k2share8_l1",
        # "e4k2share8_finetune": "20260320_213424_e4k2share8_finetune",
        # "e4k2share8_prune":"20260320_214406_e4k2share8_prune",
        "e4k2share8_l1_prune_finetune":"20260320_212332_e4k2share8_l1_prune_finetune",
        }


RUNS = {
        "e4k2share8":"20260320_134320_e4k2share8_hidden16_noise0*",
        "e4k2share8_l1_prune_finetune":"20260320_212332_e4k2share8_l1_prune_finetune",
        "e4k2share8_cka":"20260320_224145_e4k2share8_cka",
        "e4k2share8_orth":"20260321_032753_e4k2share8_orth",
        }

RUNS = {
        "e4k2share8":"20260322_082018_e4k2share8_share32_noise0.1",
        "e4k2share8_cka":"20260322_125803_e4k2share8_cka",
        "e4k2share8_orth":"20260322_143514_e4k2share8_orth",
        }

def _to_array(list_of_lists: List[List[float]]) -> np.ndarray:
    max_len = max(len(x) for x in list_of_lists)
    arr = np.full((len(list_of_lists), max_len), np.nan, dtype=float)
    for i, seq in enumerate(list_of_lists):
        arr[i, : len(seq)] = np.asarray(seq, dtype=float)
    return arr


def load_losses(
    path: Path,
    min_reps_per_epoch: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    train_key = None
    val_key = None
    test_key = None
    for k in data.keys():
        lk = k.lower()
        if "train" in lk and "loss" in lk:
            train_key = k
        if "val" in lk and "loss" in lk:
            val_key = k
        if "test" in lk and "loss" in lk:
            test_key = k
    if train_key is None or val_key is None or test_key is None:
        raise KeyError(f"Cannot find train/val/test loss keys in {path}. Keys={list(data.keys())}")

    train_reps = data[train_key]
    val_reps = data[val_key]
    test_reps = data[test_key]
    train_arr = _to_array(train_reps)
    val_arr = _to_array(val_reps)
    test_arr = _to_array(test_reps)

    train_mean = np.nanmean(train_arr, axis=0)
    val_mean = np.nanmean(val_arr, axis=0)
    test_mean = np.nanmean(test_arr, axis=0)

    min_reps = max(2, int(min_reps_per_epoch))
    train_counts = np.sum(np.isfinite(train_arr), axis=0)
    val_counts = np.sum(np.isfinite(val_arr), axis=0)
    test_counts = np.sum(np.isfinite(test_arr), axis=0)
    train_mean = np.where(train_counts >= min_reps, train_mean, np.nan)
    val_mean = np.where(val_counts >= min_reps, val_mean, np.nan)
    test_mean = np.where(test_counts >= min_reps, test_mean, np.nan)
    return train_mean, val_mean, test_mean


def _plot_train_val(series):
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10.colors
    for i, (label, (train_mean, val_mean, _test_mean)) in enumerate(series.items()):
        mask_t = np.isfinite(train_mean)
        mask_v = np.isfinite(val_mean)
        epochs_t = np.arange(1, len(train_mean) + 1)[mask_t]
        epochs_v = np.arange(1, len(val_mean) + 1)[mask_v]
        train_mean = train_mean[mask_t]
        val_mean = val_mean[mask_v]
        c = colors[i % len(colors)]
        if len(epochs_t) > 0:
            plt.plot(
                epochs_t,
                train_mean,
                color=c,
                linestyle="-",
                linewidth=1.5,
                label=f"{label} train",
            )
        if len(epochs_v) > 0:
            plt.plot(
                epochs_v,
                val_mean,
                color=c,
                linestyle="--",
                linewidth=1.5,
                label=f"{label} val",
            )
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("train + val loss (mean over reps)")
    plt.legend(ncol=2, fontsize=9)
    plt.grid(alpha=0.2)
    plt.tight_layout()

def _plot_val_test(series):
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10.colors
    for i, (label, (_train_mean, val_mean, test_mean)) in enumerate(series.items()):
        mask_v = np.isfinite(val_mean)
        mask_t = np.isfinite(test_mean)
        epochs_v = np.arange(1, len(val_mean) + 1)[mask_v]
        epochs_t = np.arange(1, len(test_mean) + 1)[mask_t]
        val_mean = val_mean[mask_v]
        test_mean = test_mean[mask_t]
        c = colors[i % len(colors)]
        if len(epochs_v) > 0:
            plt.plot(
                epochs_v,
                val_mean,
                color=c,
                linestyle="-",
                linewidth=1.5,
                label=f"{label} val",
            )
        if len(epochs_t) > 0:
            plt.plot(
                epochs_t,
                test_mean,
                color=c,
                linestyle="--",
                linewidth=1.5,
                label=f"{label} test",
            )
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("val + test loss (mean over reps)")
    plt.legend(ncol=2, fontsize=9)
    plt.grid(alpha=0.2)
    plt.tight_layout()

def _plot_test_only(series):
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10.colors
    for i, (label, (_train_mean, _val_mean, test_mean)) in enumerate(series.items()):
        mask_t = np.isfinite(test_mean)
        epochs_t = np.arange(1, len(test_mean) + 1)[mask_t]
        test_mean = test_mean[mask_t]
        c = colors[i % len(colors)]
        if len(epochs_t) > 0:
            plt.plot(
                epochs_t,
                test_mean,
                color=c,
                linestyle="-",
                linewidth=1.5,
                label=label,
            )
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("test loss (mean over reps)")
    plt.legend(ncol=2, fontsize=9)
    plt.grid(alpha=0.2)
    plt.tight_layout()

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    series = {}
    if RUNS:
        for label, run_name in RUNS.items():
            run_dir = EXP_ROOT / run_name
            loss_path = run_dir / "loss_logs" / "all_reps_losses.json"
            if not loss_path.exists():
                raise FileNotFoundError(loss_path)
            train_mean, val_mean, test_mean = load_losses(
                loss_path,
                min_reps_per_epoch=MIN_REPS_PER_EPOCH,
            )
            series[label] = (train_mean, val_mean, test_mean)
    else:
        for run_dir in sorted(EXP_ROOT.iterdir()):
            if not run_dir.is_dir():
                continue
            loss_path = run_dir / "loss_logs" / "all_reps_losses.json"
            if not loss_path.exists():
                continue
            label = run_dir.name
            train_mean, val_mean, test_mean = load_losses(
                loss_path,
                min_reps_per_epoch=MIN_REPS_PER_EPOCH,
            )
            series[label] = (train_mean, val_mean, test_mean)

    if not series:
        raise FileNotFoundError(f"No loss_logs/all_reps_losses.json found under {EXP_ROOT}")

    _plot_train_val(series)
    out_train_val = OUT_DIR / "train_val_mean.png"
    plt.savefig(out_train_val, dpi=200)

    _plot_val_test(series)
    out_val_test = OUT_DIR / "val_test_mean.png"
    plt.savefig(out_val_test, dpi=200)

    _plot_test_only(series)
    out_test = OUT_DIR / "test_mean.png"
    plt.savefig(out_test, dpi=200)

    print(f"Saved train+val plot to: {out_train_val}")
    print(f"Saved val+test plot to: {out_val_test}")
    print(f"Saved test plot to: {out_test}")


if __name__ == "__main__":
    main()
