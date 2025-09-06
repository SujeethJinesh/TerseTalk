from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def _rng(seed: int | None = None):
    return np.random.default_rng(seed)


def bootstrap_ci_diff(
    a: List[float],
    b: List[float],
    n_boot: int = 5000,
    conf: float = 0.95,
    paired: bool = True,
    seed: int | None = None,
) -> Tuple[float, float, float]:
    """Bootstrap CI for mean(a - b). Truncates to common length; paired by index by default."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    n = int(min(len(a_arr), len(b_arr)))
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    a_arr = a_arr[:n]
    b_arr = b_arr[:n]
    r = _rng(seed)

    if paired:
        idx = np.arange(n)
        diffs = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            s = r.choice(idx, size=n, replace=True)
            diffs[i] = float(np.mean(a_arr[s] - b_arr[s]))
    else:
        diffs = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            s1 = r.choice(a_arr, size=n, replace=True)
            s2 = r.choice(b_arr, size=n, replace=True)
            diffs[i] = float(np.mean(s1 - s2))

    mean = float(np.mean(a_arr - b_arr))
    alpha = (1.0 - conf) / 2.0
    lo = float(np.quantile(diffs, alpha))
    hi = float(np.quantile(diffs, 1 - alpha))
    return mean, lo, hi


def percent_reduction(before: List[float], after: List[float]) -> List[float]:
    """Per-item % reduction: (before - after) / max(before, eps)."""
    eps = 1e-9
    b = np.asarray(before, dtype=float)
    a = np.asarray(after, dtype=float)
    n = int(min(len(b), len(a)))
    if n == 0:
        return []
    b = b[:n]
    a = a[:n]
    return ((b - a) / np.maximum(b, eps)).tolist()


def bootstrap_mean_ci(
    x: List[float], n_boot: int = 5000, conf: float = 0.95, seed: int | None = None
) -> Tuple[float, float, float]:
    """Bootstrap CI for mean(x)."""
    arr = np.asarray(x, dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    r = _rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        s = r.choice(arr, size=len(arr), replace=True)
        means[i] = float(np.mean(s))
    alpha = (1.0 - conf) / 2.0
    return float(np.mean(arr)), float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def one_sided_p_gt_zero(x: List[float], n_boot: int = 10000, seed: int | None = None) -> float:
    """Approximate one-sided p-value that mean(x) <= 0 via bootstrap."""
    arr = np.asarray(x, dtype=float)
    if len(arr) == 0:
        return float("nan")
    r = _rng(seed)
    cnt = 0
    for _ in range(n_boot):
        s = r.choice(arr, size=len(arr), replace=True)
        if float(np.mean(s)) <= 0.0:
            cnt += 1
    return float(cnt / n_boot)


def noninferiority(
    treatment_acc: List[float],
    control_acc: List[float],
    delta: float = 0.02,
    conf: float = 0.95,
    seed: int | None = None,
) -> Dict[str, float | bool]:
    """
    H0: treat - control < -δ;  H1: treat - control >= -δ.
    PASS if CI_lower > -δ.
    """
    mean, lo, hi = bootstrap_ci_diff(treatment_acc, control_acc, conf=conf, paired=True, seed=seed)
    return {
        "mean_difference": float(mean),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "delta": float(delta),
        "is_noninferior": bool(lo > -delta),
    }

