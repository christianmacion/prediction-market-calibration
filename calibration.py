"""
calibration.py — calibration metrics for binary prediction markets.

Pure functions that take (predicted_prob, resolved_outcome) arrays and
return scalar metrics. Everything else is bookkeeping.

Metrics implemented:
    - brier_score        mean of (p - y)^2
    - log_score          mean of -[y log p + (1-y) log(1-p)]
    - expected_calibration_error (ECE)  — weighted avg of |acc - conf| over bins
    - maximum_calibration_error (MCE)  — max over bins of |acc - conf|
    - reliability_diagram data — per-bin (confidence, accuracy, count)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


# Minimum probability clip to avoid log(0). 1e-6 mirrors the typical choice
# in scoring rules (Brier/log) literature.
_EPS = 1e-6


@dataclass
class ReliabilityBin:
    """One bin of the reliability diagram."""
    bin_index: int
    lower: float
    upper: float
    confidence: float   # mean predicted prob in bin
    accuracy: float     # empirical fraction of 1s in bin
    count: int


@dataclass
class ReliabilityDiagram:
    """Binned reliability data. Empty bins have nan."""
    bins: list[ReliabilityBin]
    ece: float        # expected calibration error
    mce: float        # maximum calibration error
    n_points: int

    def nonempty(self) -> list[ReliabilityBin]:
        return [b for b in self.bins if b.count > 0]


def brier_score(p: Sequence[float], y: Sequence[int]) -> float:
    """Mean of (p - y)^2. Lower is better. Range [0, 1].

    Perfect calibration: mean of (p - y)^2 collapses to the irreducible
    variance of y, but Brier is typically reported as is.
    """
    p_arr = np.asarray(p, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if p_arr.shape != y_arr.shape:
        raise ValueError(f"shape mismatch p={p_arr.shape} y={y_arr.shape}")
    if p_arr.size == 0:
        return float("nan")
    return float(np.mean((p_arr - y_arr) ** 2))


def log_score(p: Sequence[float], y: Sequence[int]) -> float:
    """Mean of -[y log p + (1-y) log(1-p)]. Lower is better. Range [0, inf).

    Strictly proper scoring rule (unlike raw accuracy); rewards
    probabilistic honesty.
    """
    p_arr = np.asarray(p, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if p_arr.shape != y_arr.shape:
        raise ValueError(f"shape mismatch p={p_arr.shape} y={y_arr.shape}")
    if p_arr.size == 0:
        return float("nan")
    p_clip = np.clip(p_arr, _EPS, 1.0 - _EPS)
    ll = y_arr * np.log(p_clip) + (1.0 - y_arr) * np.log(1.0 - p_clip)
    return float(-np.mean(ll))


def reliability_diagram(
    p: Sequence[float],
    y: Sequence[int],
    n_bins: int = 10,
) -> ReliabilityDiagram:
    """Bin (p, y) into `n_bins` equal-width buckets and report per-bin
    confidence, accuracy, and the classic ECE / MCE.

    ECE = sum_{b} (n_b / N) * |acc_b - conf_b|
    MCE = max_{b} |acc_b - conf_b|
    """
    p_arr = np.asarray(p, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if p_arr.shape != y_arr.shape:
        raise ValueError(f"shape mismatch p={p_arr.shape} y={y_arr.shape}")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[ReliabilityBin] = []
    weighted_err_sum = 0.0
    max_err = 0.0
    n = int(p_arr.size)

    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == n_bins - 1:
            mask = (p_arr >= lo) & (p_arr <= hi)
        else:
            mask = (p_arr >= lo) & (p_arr < hi)
        cnt = int(mask.sum())
        if cnt > 0:
            conf = float(p_arr[mask].mean())
            acc = float(y_arr[mask].mean())
        else:
            conf = float("nan")
            acc = float("nan")
        err = abs(acc - conf) if cnt > 0 else 0.0
        weighted_err_sum += (cnt / max(n, 1)) * err
        if cnt > 0:
            max_err = max(max_err, err)
        bins.append(ReliabilityBin(
            bin_index=i, lower=lo, upper=hi,
            confidence=conf, accuracy=acc, count=cnt,
        ))

    return ReliabilityDiagram(
        bins=bins,
        ece=float(weighted_err_sum),
        mce=float(max_err),
        n_points=n,
    )


def expected_calibration_error(
    p: Sequence[float], y: Sequence[int], n_bins: int = 10,
) -> float:
    """Convenience wrapper: just the ECE scalar."""
    return reliability_diagram(p, y, n_bins=n_bins).ece


def maximum_calibration_error(
    p: Sequence[float], y: Sequence[int], n_bins: int = 10,
) -> float:
    """Convenience wrapper: just the MCE scalar."""
    return reliability_diagram(p, y, n_bins=n_bins).mce


def aggregate(
    points: Sequence[Tuple[float, int]],
    n_bins: int = 10,
) -> dict:
    """Take an iterable of (p, y) pairs and return a one-shot summary dict:
    brier, log_score, ece, mce, plus the ReliabilityDiagram.
    """
    if not points:
        return {
            "n": 0,
            "brier": float("nan"),
            "log_score": float("nan"),
            "ece": float("nan"),
            "mce": float("nan"),
            "reliability": None,
        }
    p = [pt[0] for pt in points]
    y = [pt[1] for pt in points]
    rd = reliability_diagram(p, y, n_bins=n_bins)
    return {
        "n": len(points),
        "brier": brier_score(p, y),
        "log_score": log_score(p, y),
        "ece": rd.ece,
        "mce": rd.mce,
        "reliability": rd,
    }
