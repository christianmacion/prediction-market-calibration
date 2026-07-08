"""
data_loader.py — fetch prediction-market data with a guaranteed fallback.

Real path:
    Try the Manifold Markets public API
        GET https://api.manifold.markets/v0/markets?limit=500&filter=resolved
    and turn each resolved market into a series of (probability, outcome)
    pairs using whatever probability history we can salvage.

Fallback path (always succeeds):
    Load `data/sample_markets.json` from the repo. That dataset has 60
    markets, each with a small (predicted_prob, resolved_outcome) history.

The function returns a uniform list of records:
    {
        "id":           str,
        "question":     str,
        "outcome":      bool,            # final resolution
        "snapshots":    [(float, bool)]  # (predicted_prob, eventual outcome) per snapshot
    }
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests


DEFAULT_TIMEOUT_SEC = 8
MANIFOLD_URL = "https://api.manifold.markets/v0/markets"


def _load_sample(base: Path) -> list[dict]:
    """Load the embedded sample dataset.

    Two supported shapes:
      - top-level list
      - top-level dict with a "markets" key
    """
    path = base / "data" / "sample_markets.json"
    if not path.exists():
        return []
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "markets" in raw:
        markets = raw["markets"]
    else:
        markets = raw

    records: list[dict] = []
    for m in markets:
        snaps = [
            (float(s["predicted_prob"]), bool(s["resolved"]))
            for s in m.get("calibration_data", [])
        ]
        records.append({
            "id": m.get("id", ""),
            "question": m.get("question", ""),
            "outcome": bool(m.get("outcome", False)),
            "snapshots": snaps,
        })
    return records


def _fetch_manifold(limit: int = 500) -> list[dict] | None:
    """Try the Manifold Markets API.

    Returns None on any failure, or a list of records on success.
    """
    try:
        r = requests.get(
            MANIFOLD_URL,
            params={"limit": limit, "filter": "resolved"},
            timeout=DEFAULT_TIMEOUT_SEC,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        if not isinstance(data, list) or not data:
            return None

        records: list[dict] = []
        for m in data:
            try:
                outcome_str = m.get("resolution")
                if outcome_str not in ("YES", "NO"):
                    continue
                outcome_bool = outcome_str == "YES"

                # Manifold's API gives the current probability for the
                # market, not a historical series. We synthesize a
                # monotonic-approach trajectory using probabilities[] if
                # available, otherwise the single current probability.
                probs = m.get("probTrajectory") or []
                if probs and isinstance(probs, list):
                    # probs is typically [(timestamp, p), ...] or just [p, ...]
                    p_series: list[float] = []
                    for entry in probs:
                        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                            p_series.append(float(entry[1]))
                        elif isinstance(entry, dict) and "p" in entry:
                            p_series.append(float(entry["p"]))
                        elif isinstance(entry, (int, float)):
                            p_series.append(float(entry))
                else:
                    p_series = [float(m.get("probability", 0.5))]

                # Snapshots approaching the resolution
                snaps = [
                    (max(0.01, min(0.99, float(p))), outcome_bool)
                    for p in p_series[:10]  # cap at 10 snapshots per market
                ]
                if not snaps:
                    continue

                records.append({
                    "id": str(m.get("id", m.get("slug", ""))),
                    "question": str(m.get("question", "")),
                    "outcome": outcome_bool,
                    "snapshots": snaps,
                })
            except Exception:
                # Bad record — skip
                continue

        # Require a reasonable number of resolved markets with usable data
        if len(records) < 25:
            return None
        return records
    except Exception:
        return None


def load_markets(base_dir: str | Path | None = None) -> tuple[list[dict], dict]:
    """Load markets from the Manifold API or fall back to the embedded sample.

    Returns:
        (records, source_info)

    `records` is a list of dicts with fields id, question, outcome, snapshots.
    `source_info` is metadata about where the data came from.
    """
    base = Path(base_dir) if base_dir else Path(__file__).parent

    manifold = _fetch_manifold()
    if manifold:
        return manifold, {
            "source": "manifold_api",
            "url": MANIFOLD_URL,
            "n_markets": len(manifold),
        }

    sample = _load_sample(base)
    return sample, {
        "source": "embedded_sample",
        "path": str(base / "data" / "sample_markets.json"),
        "n_markets": len(sample),
    }


def flatten_points(records: list[dict]) -> list[tuple[float, int]]:
    """Turn a list of records into a flat list of (predicted_prob, outcome_int).

    Each snapshot of each market becomes one calibration point. The
    outcome is the *eventual* market resolution — i.e. every snapshot
    for market m shares the same y. This is what we want for the
    "given the prediction at time t, did the event actually resolve
    the way the market predicted it would?" question.
    """
    points: list[tuple[float, int]] = []
    for rec in records:
        y = 1 if rec.get("outcome") else 0
        for p, _ in rec.get("snapshots", []):
            points.append((float(p), y))
    return points


def per_market_points(
    records: list[dict],
    pick: str = "final",
) -> list[tuple[float, int]]:
    """Per-market calibration points (one per market, not per snapshot).

    `pick` controls which snapshot is used as the prediction:
      - "final"   — the last snapshot (often the most confident)
      - "first"   — the first snapshot (earliest, often most uncertain)
      - "mean"    — the average prediction across snapshots
    """
    out: list[tuple[float, int]] = []
    for rec in records:
        snaps = rec.get("snapshots", [])
        if not snaps:
            continue
        y = 1 if rec.get("outcome") else 0
        if pick == "first":
            p = snaps[0][0]
        elif pick == "mean":
            p = sum(s[0] for s in snaps) / len(snaps)
        else:  # "final"
            p = snaps[-1][0]
        out.append((float(p), int(y)))
    return out
