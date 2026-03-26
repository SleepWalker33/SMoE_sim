# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import json
import time
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def setup_cpu(num_threads: int) -> None:
    import os
    import torch

    num_threads = max(1, int(num_threads))

    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)

    # 只在“尚未设置过”的情况下调用
    try:
        torch.set_num_threads(num_threads)
    except RuntimeError:
        pass

    # interop 线程：**只允许设置一次**
    try:
        torch.set_num_interop_threads(min(4, num_threads))
    except RuntimeError:
        pass

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def to_device(batch, device: str):
    if isinstance(batch, (list, tuple)):
        return [to_device(x, device) for x in batch]
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if torch.is_tensor(batch):
        return batch.to(device)
    return batch


def hungarian_max_assignment(weight: np.ndarray) -> np.ndarray:
    """Return best 1-1 max assignment for rows -> cols. Returns col indices (0-based)."""
    w = np.asarray(weight, dtype=np.float64)
    n_rows, n_cols = w.shape
    n = max(n_rows, n_cols)
    maxv = float(np.max(w)) if w.size else 0.0
    cost = np.full((n, n), maxv, dtype=np.float64)
    cost[:n_rows, :n_cols] = maxv - w

    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(n + 1, dtype=np.float64)
    p = np.zeros(n + 1, dtype=np.int64)
    way = np.zeros(n + 1, dtype=np.int64)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf, dtype=np.float64)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(0, n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = np.zeros(n, dtype=np.int64)
    for j in range(1, n + 1):
        if p[j] > 0:
            assignment[p[j] - 1] = j - 1
    return assignment[:n_rows]
