# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt


def _ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_loss_curves_with_sd(
    train_losses: np.ndarray,  # [rep, epoch]
    val_losses: np.ndarray,    # [rep, epoch]
    out_path: str | Path,
    test_losses: Optional[np.ndarray] = None,  # [rep, epoch] or None
    title: str = "Loss curves (mean ± sd)",
    xscale: str = "linear",
    yscale: str = "linear",
):
    out_path = Path(out_path)
    _ensure_dir(out_path.parent)

    mean_train = train_losses.mean(axis=0)
    sd_train = train_losses.std(axis=0, ddof=1) if train_losses.shape[0] > 1 else np.zeros_like(mean_train)

    mean_val = val_losses.mean(axis=0)
    sd_val = val_losses.std(axis=0, ddof=1) if val_losses.shape[0] > 1 else np.zeros_like(mean_val)

    mean_test = None
    sd_test = None
    if test_losses is not None:
        mean_test = test_losses.mean(axis=0)
        sd_test = test_losses.std(axis=0, ddof=1) if test_losses.shape[0] > 1 else np.zeros_like(mean_test)

    x = np.arange(1, len(mean_train) + 1)

    plt.figure()
    plt.errorbar(
        x, mean_train, yerr=sd_train, fmt="o-", markersize=3, linewidth=1.5,
        capsize=2, elinewidth=1.0, label="train",
    )
    plt.errorbar(
        x, mean_val, yerr=sd_val, fmt="s-", markersize=3, linewidth=1.5,
        capsize=2, elinewidth=1.0, label="val",
    )
    if mean_test is not None and sd_test is not None:
        plt.errorbar(
            x, mean_test, yerr=sd_test, fmt="^-", markersize=3, linewidth=1.5,
            capsize=2, elinewidth=1.0, label="test", color="tab:green",
        )

    xlabel = "epoch"
    ylabel = "loss"
    if xscale == "log":
        xlabel = "epoch (log)"
    if yscale == "log":
        ylabel = "loss (log)"
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if xscale != "linear":
        plt.xscale(xscale)
    if yscale != "linear":
        plt.yscale(yscale)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _save_heatmap(mat: np.ndarray, title: str, xlabel: str, ylabel: str, out_path: str | Path):
    out_path = Path(out_path)
    _ensure_dir(out_path.parent)

    plt.figure()
    plt.imshow(mat, aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_label_expert_heatmap(freq: torch.Tensor, out_path: str | Path, title_prefix: str = ""):
    mat = freq.detach().cpu().numpy()
    title = (title_prefix + " " if title_prefix else "") + "label_id × expert_id routing frequency"
    _save_heatmap(mat, title, "expert_id", "label_id", out_path)


def save_regime_expert_heatmap(freq: torch.Tensor, out_path: str | Path, title_prefix: str = ""):
    mat = freq.detach().cpu().numpy()
    title = (title_prefix + " " if title_prefix else "") + "regime_id × expert_id routing frequency"
    _save_heatmap(mat, title, "expert_id", "regime_id", out_path)


def save_cka_heatmap(cka_mat: torch.Tensor, out_path: str | Path, title_prefix: str = ""):
    mat = cka_mat.detach().cpu().numpy()
    title = (title_prefix + " " if title_prefix else "") + "expert × expert CKA"
    _save_heatmap(mat, title, "expert_id", "expert_id", out_path)


def save_biasvar_heatmap(
    values: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    out_path: str | Path,
    title: str,
    x_label: str = "x[0]",
    y_label: str = "x[1]",
    cmap: str = "viridis",
):
    out_path = Path(out_path)
    _ensure_dir(out_path.parent)

    plt.figure()
    extent = (float(x_coords[0]), float(x_coords[-1]), float(y_coords[0]), float(y_coords[-1]))
    plt.imshow(values, origin="lower", extent=extent, aspect="auto", cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_optuna_curve(
    trial_vals: List[float],
    out_path: str | Path,
    title: str = "Optuna trials (val loss)",
    xlabel: str = "trial",
    ylabel: str = "val loss",
):
    out_path = Path(out_path)
    _ensure_dir(out_path.parent)

    if not trial_vals:
        return

    x = np.arange(1, len(trial_vals) + 1)
    y = np.array(trial_vals, dtype=np.float64)

    plt.figure()
    plt.plot(x, y, marker="o", markersize=3, linewidth=1.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_histogram(
    values: np.ndarray,
    out_path: str | Path,
    title: str = "Histogram",
    xlabel: str = "value",
    ylabel: str = "count",
    bins: int = 50,
):
    out_path = Path(out_path)
    _ensure_dir(out_path.parent)

    vals = np.asarray(values, dtype=np.float64).ravel()
    if vals.size == 0:
        return

    plt.figure()
    plt.hist(vals, bins=int(bins))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_biasvar_boxplot(
    bias_values: np.ndarray,
    var_values: np.ndarray,
    out_path: str | Path,
    title: str = "Bias/Variance boxplot",
    ylabel: str = "value",
) -> None:
    out_path = Path(out_path)
    _ensure_dir(out_path.parent)

    bias = np.asarray(bias_values, dtype=np.float64).ravel()
    var = np.asarray(var_values, dtype=np.float64).ravel()
    bias = bias[np.isfinite(bias)]
    var = var[np.isfinite(var)]

    data = []
    names = []
    if bias.size > 0:
        data.append(bias)
        names.append("bias2")
    if var.size > 0:
        data.append(var)
        names.append("var")
    if not data:
        return

    plt.figure()
    plt.boxplot(data, showmeans=True)
    x = np.arange(1, len(names) + 1)
    plt.xticks(x, names)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_yhat_histograms_per_sample(
    yhat_by_rep: np.ndarray,  # [R,N]
    eta_test: np.ndarray,     # [N]
    out_path: str | Path,
    sample_indices: Optional[np.ndarray] = None,
    max_panels: int = 12,
    bins: int = 20,
    title: str = "Per-sample y_hat histograms across reps",
) -> None:
    out_path = Path(out_path)
    _ensure_dir(out_path.parent)

    yhat = np.asarray(yhat_by_rep, dtype=np.float64)
    eta = np.asarray(eta_test, dtype=np.float64).reshape(-1)
    if yhat.ndim != 2 or yhat.size == 0:
        return
    if eta.size != yhat.shape[1]:
        return

    n_samples = int(yhat.shape[1])
    if n_samples <= 0:
        return

    if sample_indices is None:
        k = min(int(max_panels), n_samples)
        idx = np.linspace(0, n_samples - 1, k, dtype=np.int64)
    else:
        idx = np.asarray(sample_indices, dtype=np.int64).reshape(-1)
        idx = idx[(idx >= 0) & (idx < n_samples)]
        if idx.size == 0:
            return
        if idx.size > int(max_panels):
            idx = idx[: int(max_panels)]

    k = int(idx.size)
    cols = min(4, k)
    rows = int(np.ceil(k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 2.8 * rows), squeeze=False)

    for panel_i, sample_i in enumerate(idx.tolist()):
        r = panel_i // cols
        c = panel_i % cols
        ax = axes[r][c]
        vals = yhat[:, sample_i]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            ax.set_title(f"sample {sample_i}: no data")
            ax.grid(alpha=0.25)
            continue
        ax.hist(vals, bins=int(bins), alpha=0.75, color="tab:blue")
        eta_i = float(eta[sample_i])
        mu_i = float(vals.mean())
        ax.axvline(eta_i, color="tab:red", linestyle="--", linewidth=1.2, label="eta")
        ax.axvline(mu_i, color="tab:green", linestyle="-", linewidth=1.2, label="mean y_hat")
        ax.set_title(f"sample {sample_i}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="best")

    for panel_i in range(k, rows * cols):
        r = panel_i // cols
        c = panel_i % cols
        axes[r][c].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_biasvar_rep_plot(
    rep_bias2: np.ndarray,  # [R]
    rep_var: np.ndarray,    # [R]
    out_path: "str | Path",
    title: str = "Bias\u00b2 and Var per rep",
) -> None:
    out_path = Path(out_path)
    _ensure_dir(out_path.parent)

    rep_bias2 = np.asarray(rep_bias2, dtype=np.float64).ravel()
    rep_var = np.asarray(rep_var, dtype=np.float64).ravel()
    if rep_bias2.size == 0 or rep_var.size == 0:
        return

    R = rep_bias2.size
    means = [float(rep_bias2.mean()), float(rep_var.mean())]
    data = [rep_bias2, rep_var]
    x = np.array([1, 2])
    colors = ["tab:blue", "tab:orange"]
    labels = ["bias\u00b2", "var"]

    fig, ax = plt.subplots(figsize=(5, 4))
    for xi, m, vals, c, lbl in zip(x, means, data, colors, labels):
        ax.bar(xi, m, width=0.5, alpha=0.55, color=c, label=lbl)
        rng = np.random.default_rng(0)
        jitter = rng.uniform(-0.08, 0.08, size=R)
        ax.scatter(xi + jitter, vals, color="black", s=35, zorder=5, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("value")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_biasvar_sample_grouped_plot(
    sample_bias2: np.ndarray,  # [N]
    sample_var: np.ndarray,    # [N]
    group_ids: np.ndarray,     # [N] int
    group_names: List[str],
    out_path: "str | Path",
    title: str = "Bias\u00b2 and Var by group",
) -> None:
    """Grouped boxplot: for each group, two side-by-side boxes for bias2 and var."""
    from matplotlib.patches import Patch

    out_path = Path(out_path)
    _ensure_dir(out_path.parent)

    sample_bias2 = np.asarray(sample_bias2, dtype=np.float64).ravel()
    sample_var = np.asarray(sample_var, dtype=np.float64).ravel()
    group_ids = np.asarray(group_ids, dtype=np.int64).ravel()

    if sample_bias2.size == 0 or len(group_names) == 0:
        return

    G = len(group_names)
    bias2_positions: List[float] = []
    var_positions: List[float] = []
    bias2_data: List[np.ndarray] = []
    var_data: List[np.ndarray] = []
    tick_positions: List[float] = []

    for g in range(G):
        mask = group_ids == g
        b = sample_bias2[mask]
        v = sample_var[mask]
        b = b[np.isfinite(b)]
        v = v[np.isfinite(v)]
        pos_b = 3.0 * g + 1.0
        pos_v = 3.0 * g + 2.0
        bias2_positions.append(pos_b)
        var_positions.append(pos_v)
        bias2_data.append(b if b.size > 0 else np.array([0.0]))
        var_data.append(v if v.size > 0 else np.array([0.0]))
        tick_positions.append((pos_b + pos_v) / 2.0)

    fig_width = max(6.0, 1.8 * G)
    fig, ax = plt.subplots(figsize=(fig_width, 5))

    ax.boxplot(
        bias2_data, positions=bias2_positions, widths=0.55,
        patch_artist=True, showmeans=True,
        boxprops=dict(facecolor="tab:blue", alpha=0.6),
        medianprops=dict(color="navy", linewidth=1.5),
        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="navy", markersize=5),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
        flierprops=dict(marker=".", markersize=3, alpha=0.4),
    )
    ax.boxplot(
        var_data, positions=var_positions, widths=0.55,
        patch_artist=True, showmeans=True,
        boxprops=dict(facecolor="tab:orange", alpha=0.6),
        medianprops=dict(color="saddlebrown", linewidth=1.5),
        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="saddlebrown", markersize=5),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
        flierprops=dict(marker=".", markersize=3, alpha=0.4),
    )

    ax.legend(
        handles=[
            Patch(facecolor="tab:blue", alpha=0.6, label="bias\u00b2"),
            Patch(facecolor="tab:orange", alpha=0.6, label="var"),
        ],
        loc="upper right",
    )
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(group_names, rotation=15, ha="right")
    ax.set_ylabel("value")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_yhat_grouped_boxplot(
    values_by_group: List[np.ndarray],
    group_names: List[str],
    out_path: str | Path,
    title: str = "y_hat grouped boxplot",
    ylabel: str = "y_hat",
) -> None:
    out_path = Path(out_path)
    _ensure_dir(out_path.parent)

    if not values_by_group or not group_names:
        return
    if len(values_by_group) != len(group_names):
        return

    cleaned: List[np.ndarray] = []
    for vals in values_by_group:
        arr = np.asarray(vals, dtype=np.float64).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            arr = np.array([0.0], dtype=np.float64)
        cleaned.append(arr)

    G = len(cleaned)
    positions = np.arange(1, G + 1, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(max(7.0, 1.5 * G), 5.0))
    ax.boxplot(
        cleaned,
        positions=positions,
        widths=0.6,
        showmeans=True,
        patch_artist=True,
        boxprops=dict(facecolor="tab:blue", alpha=0.55),
        medianprops=dict(color="navy", linewidth=1.5),
        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="navy", markersize=5),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
        flierprops=dict(marker=".", markersize=3, alpha=0.35),
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(group_names, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_prob_boxplot_grid_by_true_class(
    probs: np.ndarray,          # [N,C]
    y_true: np.ndarray,         # [N]
    out_path: str | Path,
    title: str = "Probability boxplots by true class",
) -> None:
    out_path = Path(out_path)
    _ensure_dir(out_path.parent)

    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    if p.ndim != 2 or p.size == 0:
        return
    if y.size != p.shape[0]:
        return
    C = int(p.shape[1])
    if C <= 0:
        return

    fig, axes = plt.subplots(C, C, figsize=(3.0 * C, 2.4 * C), squeeze=False)
    for i in range(C):
        mask = y == i
        gi = p[mask] if np.any(mask) else np.zeros((0, C), dtype=np.float64)
        n_i = int(gi.shape[0])
        for j in range(C):
            ax = axes[i][j]
            vals = gi[:, j] if n_i > 0 else np.array([], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size > 0:
                ax.boxplot(
                    [vals],
                    widths=0.5,
                    showmeans=True,
                    patch_artist=True,
                    boxprops=dict(facecolor="tab:blue", alpha=0.55),
                    medianprops=dict(color="navy", linewidth=1.2),
                    meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="navy", markersize=4),
                    whiskerprops=dict(linewidth=0.9),
                    capprops=dict(linewidth=0.9),
                    flierprops=dict(marker=".", markersize=2, alpha=0.25),
                )
                ax.set_xticks([])
            else:
                ax.text(0.5, 0.5, "no samples", ha="center", va="center", fontsize=8, transform=ax.transAxes)
                ax.set_xticks([])
            ax.set_ylim(0.0, 1.0)
            if i == 0:
                ax.set_title(f"p(class {j + 1})", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"true class {i + 1}\n(n={n_i})", fontsize=9)
            ax.grid(axis="y", alpha=0.2)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_yhat_boxplot_with_truth(
    yhat_by_rep: np.ndarray,  # [R,N]
    eta_test: np.ndarray,     # [N]
    out_path: str | Path,
    sample_indices: Optional[np.ndarray] = None,
    max_boxes: int = 20,
    title: str = "Per-sample y_hat boxplot with truth",
    ylabel: str = "y_hat",
) -> None:
    out_path = Path(out_path)
    _ensure_dir(out_path.parent)

    yhat = np.asarray(yhat_by_rep, dtype=np.float64)
    eta = np.asarray(eta_test, dtype=np.float64).reshape(-1)
    if yhat.ndim != 2 or yhat.size == 0:
        return
    if eta.size != yhat.shape[1]:
        return

    n_samples = int(yhat.shape[1])
    if n_samples <= 0:
        return

    if sample_indices is None:
        k = min(int(max_boxes), n_samples)
        idx = np.linspace(0, n_samples - 1, k, dtype=np.int64)
    else:
        idx = np.asarray(sample_indices, dtype=np.int64).reshape(-1)
        idx = idx[(idx >= 0) & (idx < n_samples)]
        if idx.size == 0:
            return
        if idx.size > int(max_boxes):
            idx = idx[: int(max_boxes)]

    if idx.size == 0:
        return

    data = []
    truth = []
    labels = []
    for i in idx.tolist():
        vals = yhat[:, i]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        data.append(vals)
        truth.append(float(eta[i]))
        labels.append(str(int(i)))

    if not data:
        return

    x = np.arange(1, len(data) + 1)

    plt.figure(figsize=(max(10.0, 0.6 * len(data)), 5.0))
    plt.boxplot(data, positions=x, showmeans=True)
    # Truth values: one short horizontal line per sample box.
    half_width = 0.28
    for xi, yi in zip(x, truth):
        plt.hlines(yi, xi - half_width, xi + half_width, colors="tab:red", linewidth=2.0)
    plt.plot([], [], color="tab:red", linewidth=2.0, label="eta (truth)")
    plt.xticks(x, labels, rotation=0)
    plt.xlabel("sample index")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
