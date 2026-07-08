"""
plot_results.py — render `results/figure.png`.

4-panel matplotlib figure (140 DPI):

    [1] Reliability diagram, RAW predictions
    [2] Reliability diagram, AFTER Platt scaling
    [3] Per-market Brier score histogram
    [4] Per-market final predicted prob vs resolved outcome

Reads:
    results/results.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _plot_reliability(
    ax,
    bins_data: list[dict],
    title: str,
    subtitle: str,
):
    """Render a reliability diagram on the given axes."""
    confs = []
    accs = []
    cnts = []
    for b in bins_data:
        if b["count"] > 0:
            confs.append(b["confidence"])
            accs.append(b["accuracy"])
            cnts.append(b["count"])

    if not confs:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(title, fontsize=10)
        return

    confs_arr = np.array(confs)
    accs_arr = np.array(accs)
    cnts_arr = np.array(cnts)

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "--", color="#888888", lw=1, label="Perfect")

    # Scaled dot area by bin count
    sizes = 30 + 200 * cnts_arr / cnts_arr.max()

    ax.scatter(confs_arr, accs_arr, s=sizes, color="#1f77b4",
               edgecolor="white", linewidth=1, alpha=0.85, zorder=3,
               label="bins")

    # Optional: connect points in order to show the calibration curve
    order = np.argsort(confs_arr)
    if len(order) >= 2:
        ax.plot(confs_arr[order], accs_arr[order], color="#1f77b4",
                alpha=0.4, lw=1, zorder=2)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("confidence (mean predicted prob)", fontsize=9)
    ax.set_ylabel("accuracy (empirical YES rate)", fontsize=9)
    ax.set_title(f"{title}\n{subtitle}", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)


def _plot_brier_histogram(ax, per_market: list[dict], title: str):
    briers = [m["brier"] for m in per_market]
    outcomes = [1 if m["outcome"] else 0 for m in per_market]
    mean_b = float(np.mean(briers)) if briers else float("nan")

    if not briers:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(title, fontsize=10)
        return

    # Color by outcome for visual signal
    colors = ["#2ca02c" if o == 1 else "#d62728" for o in outcomes]
    ax.hist(briers, bins=12, color="#1f77b4", edgecolor="white")
    ax.axvline(mean_b, color="black", lw=1.2, linestyle="--",
               label=f"mean = {mean_b:.3f}")
    ax.set_xlabel("per-market Brier score", fontsize=9)
    ax.set_ylabel("count", fontsize=9)
    ax.set_title(f"{title}\nN={len(briers)} markets, mean = {mean_b:.3f}",
                 fontsize=10, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax.grid(alpha=0.25)


def _plot_pred_vs_outcome(ax, per_market: list[dict], title: str):
    if not per_market:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(title, fontsize=10)
        return

    # Use the final predicted probability (last snapshot before resolution)
    preds = []
    outs = []
    for m in per_market:
        snaps = m.get("snapshots", [])
        if not snaps:
            continue
        preds.append(snaps[-1][0])
        outs.append(1 if m["outcome"] else 0)

    preds_arr = np.array(preds)
    outs_arr = np.array(outs)

    # Add jitter to outcome axis for visibility
    jitter = np.random.default_rng(7).uniform(-0.07, 0.07, size=outs_arr.size)
    y_pos = outs_arr + jitter

    ax.scatter(preds_arr, y_pos, alpha=0.7, s=28,
               c=["#2ca02c" if o == 1 else "#d62728" for o in outs_arr],
               edgecolor="white", linewidth=0.6)
    ax.axhline(0.5, color="#888888", lw=1, linestyle=":")
    ax.axvline(0.5, color="#888888", lw=1, linestyle=":")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.2, 1.2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["resolved NO", "resolved YES"], fontsize=9)
    ax.set_xlabel("final predicted probability (sorted)", fontsize=9)
    ax.set_ylabel("actual outcome", fontsize=9)

    # Sort by predicted probability to show the trading curve
    order = np.argsort(preds_arr)
    ax.plot(preds_arr[order], outs_arr[order] + jitter[order],
            color="#1f77b4", alpha=0.3, lw=1)

    n_markets = int(outs_arr.size)
    yes_rate = float(outs_arr.mean())
    ax.set_title(
        f"{title}\nN={n_markets} markets, base rate YES = {yes_rate:.2f}",
        fontsize=10, fontweight="bold",
    )
    ax.grid(alpha=0.25)


def render(results: dict, out_path: Path) -> bool:
    """Build the 4-panel figure and save."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "prediction-market-calibration — reliability, Brier, "
        "and recalibration",
        fontsize=13, fontweight="bold", y=0.995,
    )

    _plot_reliability(
        axes[0, 0],
        results["raw"]["bins"],
        title="Reliability diagram — RAW",
        subtitle=f"Brier={results['raw']['brier']:.4f}, "
                 f"ECE={results['raw']['ece']:.3f}",
    )
    _plot_reliability(
        axes[0, 1],
        results["platt"]["bins"],
        title="Reliability diagram — PLATT",
        subtitle=(
            f"a={results['platt']['params']['a']:.3f}, "
            f"b={results['platt']['params']['b']:.3f} · "
            f"Brier={results['platt']['brier']:.4f}, "
            f"ECE={results['platt']['ece']:.3f}"
        ),
    )
    _plot_brier_histogram(
        axes[1, 0],
        results["per_market"],
        title="Per-market Brier score",
    )
    _plot_pred_vs_outcome(
        axes[1, 1],
        results["per_market"],
        title="Predicted prob vs resolved outcome (final snapshot)",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> int:
    base = Path(__file__).parent
    results_path = base / "results" / "results.json"
    out_path = base / "results" / "figure.png"

    if not results_path.exists():
        print(f"ERROR: {results_path} not found — run run.py first.")
        return 1

    with open(results_path) as f:
        s = json.load(f)
    ok = render(s, out_path)
    if ok:
        print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
