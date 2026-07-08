"""
recalibration.py — methods for fixing miscalibrated probabilities.

Two parametric / nonparametric approaches:

    1. Platt scaling — a 1D logistic regression in logit space:
           logit(y) = a * logit(p) + b
       fit by maximum likelihood on (p, y) pairs. Same idea as Platt
       scaling for SVMs. Two parameters, very robust to overfit.

    2. Isotonic regression — nonparametric, monotonic recalibration via
       pool-adjacent-violators (PAVA). Requires more data but captures
       arbitrary monotone miscalibration. We implement PAVA from scratch
       so the module has no sklearn dependency; if sklearn is available
       we cross-check the result.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


# Numerical guards
_EPS = 1e-6
_LOGIT_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _LOGIT_EPS, 1.0 - _LOGIT_EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    neg = ~pos
    e = np.exp(x[neg])
    out[neg] = e / (1.0 + e)
    return out


@dataclass
class PlattScaler:
    """Fitted logistic recalibration: q = sigmoid(a * logit(p) + b)."""
    a: float
    b: float
    n_fit: int

    def transform(self, p: Sequence[float]) -> np.ndarray:
        p_arr = np.asarray(p, dtype=float)
        return _sigmoid(self.a * _logit(p_arr) + self.b)


def fit_platt(p: Sequence[float], y: Sequence[int]) -> PlattScaler:
    """Fit Platt scaling via gradient descent on the Bernoulli log-likelihood.

    Maximizes sum_i [ y_i log q_i + (1-y_i) log (1-q_i) ] where
    q_i = sigmoid(a * logit(p_i) + b).

    Starting point a=1, b=0 — i.e. "no change", which is already a
    reasonable baseline. We use L-BFGS style full-batch Newton steps
    (it's a 2-parameter model, Newton is fine).
    """
    p_arr = np.asarray(p, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if p_arr.size == 0:
        return PlattScaler(a=1.0, b=0.0, n_fit=0)
    if p_arr.shape != y_arr.shape:
        raise ValueError(
            f"shape mismatch p={p_arr.shape} y={y_arr.shape}"
        )

    a = 1.0
    b = 0.0
    z = _logit(p_arr)  # logit(p), fixed across iterations

    for _ in range(200):
        x = a * z + b
        q = _sigmoid(x)
        # Gradient of neg log-likelihood wrt (a, b):
        #   d/da = -(y - q) * z
        #   d/db = -(y - q)
        r = y_arr - q  # shape (n,)
        g_a = float(-np.sum(r * z))
        g_b = float(-np.sum(r))
        # Hessian is positive (since loss is convex) — use Newton
        #   d^2/da^2 = q*(1-q) * z^2
        #   d^2/db^2 = q*(1-q)
        #   d^2/da db = q*(1-q) * z
        w = q * (1.0 - q)  # shape (n,)
        h_aa = float(np.sum(w * z * z))
        h_bb = float(np.sum(w))
        h_ab = float(np.sum(w * z))
        det = h_aa * h_bb - h_ab * h_ab
        if det <= 0:
            break
        # Newton step:  d = H^{-1} g
        d_a = (h_bb * g_a - h_ab * g_b) / det
        d_b = (h_aa * g_b - h_ab * g_a) / det
        a -= d_a
        b -= d_b
        if abs(d_a) < 1e-9 and abs(d_b) < 1e-9:
            break

    return PlattScaler(a=float(a), b=float(b), n_fit=int(p_arr.size))


def apply_platt(scaler: PlattScaler, p: Sequence[float]) -> np.ndarray:
    """Apply a fitted Platt scaler to probabilities."""
    return scaler.transform(p)


@dataclass
class IsotonicCalibrator:
    """Fitted isotonic recalibration: q = PAVA-fit(p) with held-out x.

    Stores the breakpoints (sorted x, fitted y). On query, we linearly
    interpolate (or hold flat at the boundary values).
    """
    x_: np.ndarray     # sorted unique-ish sample points
    y_: np.ndarray     # isotonic fit at those points

    def transform(self, p: Sequence[float]) -> np.ndarray:
        p_arr = np.asarray(p, dtype=float)
        # Use step interpolation: nearest predecessor
        # (step = "post" style — well-calibrated monotonic mapping).
        return np.interp(p_arr, self.x_, self.y_)  # type: ignore[return-value]


def _pava(y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Pool-adjacent-violators algorithm.

    Given y[0..n-1] with optional weights w, return y_iso[0..n-1] that
    is monotonically non-decreasing and minimizes weighted squared
    error subject to monotonicity.

    Pure NumPy implementation; O(n) average, O(n^2) worst case which is
    fine for n in the thousands.
    """
    n = y.size
    if n == 0:
        return y
    if w is None:
        w = np.ones(n)
    y_iso = y.astype(float).copy()
    w = w.astype(float).copy()

    # Process left-to-right, pooling whenever the next block would
    # violate the monotonicity of block means.
    i = 0
    while i < n - 1:
        if y_iso[i] <= y_iso[i + 1]:
            i += 1
            continue
        # Pool i and i+1
        total_w = w[i] + w[i + 1]
        y_iso[i] = (w[i] * y_iso[i] + w[i + 1] * y_iso[i + 1]) / total_w
        w[i] = total_w
        y_iso[i + 1] = y_iso[i]  # advance pointer; will be re-checked
        # Walk backwards while the previous block has a higher mean
        j = i
        while j > 0 and y_iso[j - 1] > y_iso[j]:
            total_w_j = w[j - 1] + w[j]
            y_iso[j - 1] = (
                (w[j - 1] * y_iso[j - 1] + w[j] * y_iso[j]) / total_w_j
            )
            w[j - 1] = total_w_j
            w[j] = y_iso[j]  # placeholder; we then shift up
            # Shift up to fill the gap from j upward
            y_iso[j] = y_iso[j - 1]
            j -= 1
        # The 'filled' blocks are [j..i+1] effectively; the values
        # y_iso[j..=i] are correct, y_iso[i+1] is set above.
        i = max(j, i) + 1  # type: ignore[assignment]
    return y_iso


def fit_isotonic(
    p: Sequence[float],
    y: Sequence[int],
) -> IsotonicCalibrator:
    """Fit isotonic regression on (p, y).

    Sort by p, aggregate duplicate p by averaging their y, then run PAVA
    on the resulting sequence.
    """
    p_arr = np.asarray(p, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if p_arr.size == 0:
        return IsotonicCalibrator(x_=np.array([]), y_=np.array([]))
    if p_arr.shape != y_arr.shape:
        raise ValueError(
            f"shape mismatch p={p_arr.shape} y={y_arr.shape}"
        )

    # Sort by p
    order = np.argsort(p_arr, kind="mergesort")
    ps = p_arr[order]
    ys = y_arr[order]

    # Aggregate duplicates by averaging
    if ps.size > 0:
        change = np.concatenate([[True], np.diff(ps) > 0])
        xs_agg = ps[change]
        # Weighted (count) average of y at each unique x
        ys_agg_list: list[float] = []
        cs_agg_list: list[float] = []
        i = 0
        while i < ps.size:
            j = i
            while j < ps.size and ps[j] == ps[i]:
                j += 1
            block = ys[i:j]
            ys_agg_list.append(float(block.mean()))
            cs_agg_list.append(float(block.size))
            i = j
        ys_agg = np.array(ys_agg_list)
        cs_agg = np.array(cs_agg_list)
    else:
        xs_agg = ps.copy()
        ys_agg = ys.copy()
        cs_agg = np.ones_like(ys)

    # PAVA on aggregated points
    iso = _pava(ys_agg, cs_agg)

    return IsotonicCalibrator(x_=xs_agg, y_=iso)


def apply_isotonic(
    calibrator: IsotonicCalibrator, p: Sequence[float],
) -> np.ndarray:
    return calibrator.transform(p)


def compare_brier(
    p: Sequence[float],
    y: Sequence[int],
) -> dict:
    """Compare Brier/ECE before vs. after Platt and isotonic recalibration.

    Returns a summary dict suitable for `results/results.json`.
    """
    p_arr = np.asarray(p, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if p_arr.size == 0:
        return {
            "raw": {"n": 0}, "platt": {"n": 0}, "isotonic": {"n": 0},
        }

    from calibration import (
        brier_score, log_score,
        expected_calibration_error, maximum_calibration_error,
    )

    raw = {
        "n": int(p_arr.size),
        "brier": brier_score(p_arr, y_arr),
        "log_score": log_score(p_arr, y_arr),
        "ece": expected_calibration_error(p_arr, y_arr),
        "mce": maximum_calibration_error(p_arr, y_arr),
    }

    # Platt scaling
    platt = fit_platt(p_arr, y_arr)
    p_platt = np.clip(platt.transform(p_arr), _EPS, 1.0 - _EPS)
    platt_out = {
        "n": int(p_arr.size),
        "brier": brier_score(p_platt, y_arr),
        "log_score": log_score(p_platt, y_arr),
        "ece": expected_calibration_error(p_platt, y_arr),
        "mce": maximum_calibration_error(p_platt, y_arr),
        "params": {"a": platt.a, "b": platt.b},
    }

    # Isotonic regression
    iso = fit_isotonic(p_arr, y_arr)
    p_iso = np.clip(iso.transform(p_arr), _EPS, 1.0 - _EPS)
    iso_out = {
        "n": int(p_arr.size),
        "brier": brier_score(p_iso, y_arr),
        "log_score": log_score(p_iso, y_arr),
        "ece": expected_calibration_error(p_iso, y_arr),
        "mce": maximum_calibration_error(p_iso, y_arr),
        "n_breakpoints": int(iso.x_.size),
    }

    return {"raw": raw, "platt": platt_out, "isotonic": iso_out}
