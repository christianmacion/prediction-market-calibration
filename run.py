#!/usr/bin/env python3
"""
run.py — end-to-end calibration analysis for prediction markets.

Pipeline:
    1. Load data (try Manifold API, fall back to data/sample_markets.json).
    2. Per-market: Brier, log score.
    3. Aggregate over all (snapshot, outcome) points: reliability
       diagram, ECE, MCE.
    4. Fit Platt scaling and isotonic regression recalibration.
    5. Recompute Brier/ECE after recalibration.
    6. Write results/results.json and call plot_results.py to render
       results/figure.png.
    7. Print the headline metric:

           "Market overconfidence exposed: raw Brier 0.XXXX →
            Platt Brier Y.YYYY (improvement Z%); raw ECE A.A →
            recalibrated ECE B.B"

Run:
    python3 run.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

# Local module imports
sys.path.insert(0, str(Path(__file__).parent))
from calibration import (
    brier_score,
    log_score,
    reliability_diagram,
    expected_calibration_error,
    maximum_calibration_error,
    ReliabilityBin,
)
from data_loader import load_markets, flatten_points, per_market_points
from recalibration import (
    fit_platt, apply_platt,
    fit_isotonic, apply_isotonic,
    compare_brier,
)


def _bins_to_json(bins) -> list[dict]:
    out = []
    for b in bins:
        if isinstance(b, ReliabilityBin):
            out.append({
                "index": b.bin_index,
                "lower": b.lower,
                "upper": b.upper,
                "confidence": b.confidence,
                "accuracy": b.accuracy,
                "count": b.count,
            })
        elif is_dataclass(b):
            out.append(asdict(b))
        elif isinstance(b, dict):
            out.append(b)
        else:
            out.append({"confidence": float(b[0]) if hasattr(b, "__getitem__") else None})
    return out


def _percent_change(old: float, new: float) -> float:
    if abs(old) < 1e-12:
        return 0.0
    return 100.0 * (new - old) / old


def analyze_markets(base: Path) -> tuple[dict, dict]:
    """Compute all metrics and return (results_dict, headline_metric_dict)."""
    records, source = load_markets(base)

    # Per-market metrics (one Brier / log score per market, using the
    # final snapshot as the prediction; this is the conventional way
    # to summarize a market's calibration).
    per_market_metrics: list[dict] = []
    per_market_pp = per_market_points(records, pick="final")
    for rec, (p, y) in zip(records, per_market_pp):
        snaps = rec.get("snapshots", [])
        per_market_metrics.append({
            "id": rec.get("id", ""),
            "question": rec.get("question", ""),
            "outcome": bool(rec.get("outcome", False)),
            "final_pred": float(p),
            "snapshots": snaps,
            "brier": float(brier_score([p], [y])),
            "log_score": float(log_score([p], [y])),
        })

    # Aggregate over ALL (snapshot, outcome) points
    all_pp = flatten_points(records)
    all_p = [pt[0] for pt in all_pp]
    all_y = [pt[1] for pt in all_pp]

    raw_rel = reliability_diagram(all_p, all_y, n_bins=10)

    # Recalibration comparison
    cmp = compare_brier(all_p, all_y)

    # Compute per-market aggregate metrics directly so they're
    # available even if the API ever returns only one snapshot per
    # market (where flatten_points would still work but per-market
    # would degenerate).
    pm_briers = np.array([m["brier"] for m in per_market_metrics])
    pm_logs = np.array([m["log_score"] for m in per_market_metrics])

    # Pre-vs-post headline metrics (we use the snapshot-level Brier for
    # the headline — that's the canonical metric for a population of
    # predictions, not the per-market average).
    raw_brier = cmp["raw"]["brier"]
    platt_brier = cmp["platt"]["brier"]
    iso_brier = cmp["isotonic"]["brier"]
    raw_ece = cmp["raw"]["ece"]
    platt_ece = cmp["platt"]["ece"]
    iso_ece = cmp["isotonic"]["ece"]

    brier_improvement_platt = _percent_change(raw_brier, platt_brier)
    brier_improvement_iso = _percent_change(raw_brier, iso_brier)
    ece_improvement_platt = _percent_change(raw_ece, platt_ece)
    ece_improvement_iso = _percent_change(raw_ece, iso_ece)

    # Pick the headline scorer:
    headline_method = "platt"
    if iso_ece < platt_ece:
        headline_method = "isotonic"

    headline = {
        "raw_brier": raw_brier,
        "platt_brier": platt_brier,
        "isotonic_brier": iso_brier,
        "raw_ece": raw_ece,
        "platt_ece": platt_ece,
        "isotonic_ece": iso_ece,
        "brier_improvement_platt_pct": brier_improvement_platt,
        "brier_improvement_isotonic_pct": brier_improvement_iso,
        "ece_improvement_platt_pct": ece_improvement_platt,
        "ece_improvement_isotonic_pct": ece_improvement_iso,
        "n_snapshots": len(all_pp),
        "n_markets": len(records),
        "headline_method": headline_method,
        "headline_text": (
            f"Market overconfidence exposed: "
            f"raw Brier {raw_brier:.4f} -> "
            f"{'Platt' if headline_method == 'platt' else 'isotonic'} "
            f"Brier "
            f"{(platt_brier if headline_method == 'platt' else iso_brier):.4f} "
            f"(improvement "
            f"{(brier_improvement_platt if headline_method == 'platt' else brier_improvement_iso):.1f}%); "
            f"raw ECE {raw_ece:.3f} -> recalibrated ECE "
            f"{(platt_ece if headline_method == 'platt' else iso_ece):.3f}"
        ),
        "source": source,
    }

    results = {
        "headline": headline,
        "n_snapshots": int(len(all_pp)),
        "n_markets": int(len(records)),
        "source": source,
        "per_market": per_market_metrics,
        "per_market_summary": {
            "mean_brier": float(pm_briers.mean()) if pm_briers.size else None,
            "median_brier": float(np.median(pm_briers)) if pm_briers.size else None,
            "mean_log_score": float(pm_logs.mean()) if pm_logs.size else None,
        },
        "raw": {
            "n": int(cmp["raw"]["n"]),
            "brier": float(cmp["raw"]["brier"]),
            "log_score": float(cmp["raw"]["log_score"]),
            "ece": float(cmp["raw"]["ece"]),
            "mce": float(cmp["raw"]["mce"]),
            "bins": _bins_to_json(raw_rel.bins),
        },
        "platt": {
            "n": int(cmp["platt"]["n"]),
            "brier": float(cmp["platt"]["brier"]),
            "log_score": float(cmp["platt"]["log_score"]),
            "ece": float(cmp["platt"]["ece"]),
            "mce": float(cmp["platt"]["mce"]),
            "params": cmp["platt"]["params"],
            "bins": _bins_to_json(reliability_diagram(
                apply_platt(fit_platt(all_p, all_y), all_p).tolist(),
                all_y, n_bins=10,
            ).bins),
        },
        "isotonic": {
            "n": int(cmp["isotonic"]["n"]),
            "brier": float(cmp["isotonic"]["brier"]),
            "log_score": float(cmp["isotonic"]["log_score"]),
            "ece": float(cmp["isotonic"]["ece"]),
            "mce": float(cmp["isotonic"]["mce"]),
            "n_breakpoints": int(cmp["isotonic"].get("n_breakpoints", 0)),
        },
    }
    return results, headline


def render_text_report(results: dict, headline: dict) -> str:
    rows = []
    rows.append("\n" + "=" * 76)
    rows.append("PREDICTION-MARKET CALIBRATION — HEADLINE")
    rows.append("=" * 76)
    rows.append(headline["headline_text"])
    rows.append(f"Source: {headline['source'].get('source')} · "
                f"N={headline['n_markets']} markets, "
                f"{headline['n_snapshots']} (snapshot, outcome) points")
    rows.append("")
    rows.append("-" * 76)
    rows.append("PRE / POST RECALIBRATION (snapshot-level)")
    rows.append("-" * 76)
    rows.append(f"  {'Metric':<10} {'Raw':>10} {'Platt':>10} {'Isotonic':>10} "
                f"{'dPlatt %':>10} {'dIso %':>10}")
    rows.append(
        f"  {'Brier':<10} "
        f"{results['raw']['brier']:>10.4f} "
        f"{results['platt']['brier']:>10.4f} "
        f"{results['isotonic']['brier']:>10.4f} "
        f"{headline['brier_improvement_platt_pct']:>9.1f}% "
        f"{headline['brier_improvement_isotonic_pct']:>9.1f}%"
    )
    rows.append(
        f"  {'ECE':<10} "
        f"{results['raw']['ece']:>10.4f} "
        f"{results['platt']['ece']:>10.4f} "
        f"{results['isotonic']['ece']:>10.4f} "
        f"{headline['ece_improvement_platt_pct']:>9.1f}% "
        f"{headline['ece_improvement_isotonic_pct']:>9.1f}%"
    )
    rows.append(
        f"  {'MCE':<10} "
        f"{results['raw']['mce']:>10.4f} "
        f"{results['platt']['mce']:>10.4f} "
        f"{results['isotonic']['mce']:>10.4f} "
        f"     —       —"
    )
    rows.append(
        f"  {'LogLoss':<10} "
        f"{results['raw']['log_score']:>10.4f} "
        f"{results['platt']['log_score']:>10.4f} "
        f"{results['isotonic']['log_score']:>10.4f} "
        f"     —       —"
    )
    rows.append("")
    rows.append("Per-market summary:")
    pms = results["per_market_summary"]
    if pms.get("mean_brier") is not None:
        rows.append(f"  Mean per-market Brier: {pms['mean_brier']:.4f}")
        rows.append(f"  Median per-market Brier: {pms['median_brier']:.4f}")
        rows.append(f"  Mean per-market log score: {pms['mean_log_score']:.4f}")
    rows.append("=" * 76)
    return "\n".join(rows)


def main() -> int:
    base = Path(__file__).parent
    out_dir = base / "results"
    out_dir.mkdir(exist_ok=True)

    print("Loading markets...")
    results, headline = analyze_markets(base)

    text = render_text_report(results, headline)
    print(text)

    # Persist
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(out_dir / "demo_output.txt", "w") as f:
        f.write(text + "\n")

    print(f"\nWrote {out_dir/'results.json'}")
    print(f"Wrote {out_dir/'demo_output.txt'}")

    # Render figure
    import plot_results
    rc = plot_results.main()
    if rc != 0:
        print(f"WARNING: plot_results.main() returned {rc}")
        return rc

    # Sanity check: figure exists
    fig_path = out_dir / "figure.png"
    if not fig_path.exists():
        print(f"ERROR: {fig_path} not created")
        return 1

    print(f"OK — {fig_path} present.")
    print(f"Headline: {headline['headline_text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
