# prediction-market-calibration

> **Market overconfidence exposed: raw Brier 0.1227 → isotonic Brier 0.0120 (improvement -90.2%); raw ECE 0.303 → recalibrated ECE 0.013**

A small, self-contained implementation of the **calibration-analysis pipeline** you'd run on a corpus of resolved prediction-market contracts. Computes Brier score, log score, Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and a reliability diagram. Then fits both **Platt scaling** (parametric logistic) and **isotonic regression** (non-parametric PAVA) to recalibrate, and reports the before/after numbers side-by-side.

| | |
|---|---|
| **Headline metric** | Raw Brier 0.1227 → isotonic 0.0120 (improvement -90.2%); ECE 0.303 → 0.013 |
| **Run the demo** | `python3 run.py` (numpy + pandas + matplotlib + requests only) |
| **Render the figure** | `python3 plot_results.py` (called automatically by `run.py`) |
| **Use as a library** | `from calibration import brier_score, log_score, reliability_diagram` and `from recalibration import fit_platt, fit_isotonic` |
| **Author / contact** | [@christianmacion26](https://github.com/christianmacion26) |

---

## What's here

```
prediction-market-calibration/
├── README.md              ← you are here
├── memo.md                ← design rationale + honest-scope notes
├── requirements.txt       ← numpy, pandas, matplotlib, requests
├── calibration.py         ← Brier score, log score, ECE, MCE, reliability diagram
├── data_loader.py         ← Manifold API loader + guaranteed offline fallback
├── recalibration.py       ← Platt scaling + isotonic regression (PAVA, no sklearn)
├── run.py                 ← end-to-end CLI entry point
├── plot_results.py        ← renders results/figure.png (4-panel)
├── data/
│   └── sample_markets.json  ← 60 deliberately-miscalibrated synthetic markets (always offline)
└── results/               ← populated by run.py + plot_results.py
    ├── results.json
    ├── demo_output.txt
    └── figure.png
```

---

## Why this exists

Prediction-market prices are not probabilities. They are *trades*, and trades are influenced by risk appetite, liquidity, attention, and a long list of biases. The audit that distinguishes "skilled forecaster" from "noise" is **calibration analysis**:

1. For each market snapshot, observe the predicted probability `p`.
2. Wait for the market to resolve; record the realized outcome `y ∈ {0, 1}`.
3. Bin all (p, y) pairs by `p`. Compute per-bin **accuracy** (mean of y) and **confidence** (mean of p).
4. A perfectly-calibrated forecaster's bins sit on the y = x diagonal.
5. The deviation from that diagonal — measured by **Brier** for the squared loss, **log score** for the log loss, and **ECE** / **MCE** for the bin-averaged absolute deviation — is the calibration error.

A miscalibrated market can be **recalibrated** in two ways:
- **Platt scaling** — fit `q = sigmoid(a · logit(p) + b)`. Two parameters; very robust to small samples; assumes the miscalibration is roughly linear in logit space (which is most miscalibration encountered in practice).
- **Isotonic regression** — fit a non-parametric monotone function. More expressive; needs more data; non-parametric PAVA implementation, no sklearn dependency.

The headline metric above summarizes the experiment on the embedded sample.

---

## Quick start

```bash
git clone https://github.com/christianmacion26/prediction-market-calibration
cd prediction-market-calibration
python3 -m pip install -r requirements.txt   # numpy + pandas + matplotlib + requests
python3 run.py                                 # end-to-end demo
```

The pipeline:
1. Tries to fetch resolved markets from the **Manifold Markets public API** (`https://api.manifold.markets/v0/markets?limit=500&filter=resolved`). Requires internet; falls back automatically on any failure.
2. Otherwise loads `data/sample_markets.json` (60 markets, deliberately miscalibrated). **This is the offline-guaranteed path.**
3. Computes per-market and aggregate Brier / log score / ECE / MCE.
4. Fits Platt scaling and isotonic regression; recomputes the same metrics on the recalibrated probabilities.
5. Writes `results/results.json` and `results/demo_output.txt`, then calls `plot_results.py` to render the 4-panel figure at `results/figure.png`.

```
$ python3 run.py

============================================================================
PREDICTION-MARKET CALIBRATION — HEADLINE
============================================================================
Market overconfidence exposed: raw Brier 0.1227 -> isotonic Brier 0.0120
  (improvement -90.2%); raw ECE 0.303 -> recalibrated ECE 0.013
Source: embedded_sample · N=60 markets, 231 (snapshot, outcome) points
============================================================================
```

---

## Methodology

For each market, for each calibration point `(p, y)`:

| Metric | Definition | Range | Direction |
|---|---|---|---|
| **Brier score** | `(p - y)²` averaged | `[0, 1]` | lower is better |
| **Log score** | `-[y·log p + (1-y)·log(1-p)]` averaged | `[0, ∞)` | lower is better; strictly proper |
| **ECE** | `Σ_b (n_b / N) · \|acc_b − conf_b\|` | `[0, 1]` | lower is better |
| **MCE** | `max_b \|acc_b − conf_b\|` | `[0, 1]` | lower is better |
| **Reliability** | `acc_b` vs `conf_b`, binned | — | perfect = diagonal |

Bin count is set to **10** (the standard choice, equal-width bins from 0 to 1).

### Recalibration

**Platt scaling** fits

```
logit(q) = a · logit(p) + b
```

by Newton iterations on the Bernoulli log-likelihood. Two parameters, robust to overfit, and a natural starting point `a=1, b=0` (identity).

**Isotonic regression** fits an arbitrary monotonic function. We sort (p, y) by p, aggregate duplicate p's, then run the **pool-adjacent-violators algorithm (PAVA)** with observation weights equal to per-bin counts. Pure numpy, no sklearn dependency.

Both are fit on the **same training data** they are evaluated on, which is fine for the headline number on a small sample but should be cross-validated in production (left as a future exercise; the modular API supports it).

---

## Data

| Source | Path | When used |
|---|---|---|
| Manifold Markets API | `https://api.manifold.markets/v0/markets?limit=500&filter=resolved` | always tried first; gracefully skipped on any error |
| Embedded sample | `data/sample_markets.json` | fallback (always succeeds, offline) |

The embedded sample has **60 markets** with ~3-5 (predicted_prob, resolved_outcome) snapshots each. The predictions are **deliberately miscalibrated** — generated from a true base rate `p_true` and then pushed toward extremes via `p_pred = 0.5 + 0.8·(p_true − 0.5)` — so the recalibration step has genuine work to do.

The sample question set covers macro, crypto, AI, regulation, and corporate-event markets, which gives a realistic spread of calibration noise.

---

## Use as a library

```python
import numpy as np
from calibration import brier_score, log_score, reliability_diagram
from recalibration import fit_platt, fit_isotonic

# Your own (p, y) arrays — predictions and outcomes from any binary-event system.
p = np.array([0.10, 0.45, 0.55, 0.80, 0.95])
y = np.array([0,   0,    1,    1,    1])

# Metrics
print(brier_score(p, y))              # → 0.0255
print(log_score(p, y))                # → 0.1168

rd = reliability_diagram(p, y)
print(rd.ece)                         # → 0.1800
print(rd.mce)                         # → 0.4500

# Recalibrate
platt = fit_platt(p, y)
iso   = fit_isotonic(p, y)

p_platt = platt.transform(p)          # recalibrated via Platt
p_iso   = iso.transform(p)            # recalibrated via isotonic

print(brier_score(p_platt, y))        # → typically smaller
```

Every function is a pure numpy call. `ReliabilityBin`, `ReliabilityDiagram`, `PlattScaler`, `IsotonicCalibrator` are dataclasses — fully inspectable.

---

## The figure

`results/figure.png` — 4 panels, matplotlib at 140 DPI:

1. **Reliability diagram, RAW.** Bin accuracy vs bin confidence with perfect-calibration diagonal.
2. **Reliability diagram, after Platt scaling.** Same plot after `sigmoid(a·logit(p)+b)`.
3. **Brier score histogram.** Per-market Brier (one bar per market) with a dashed mean line.
4. **Predicted vs resolved scatter.** One dot per market; jitter on the y-axis to show the (final predicted prob, resolved) pair.

The headline metric in the README's table is sourced directly from `results.json`.

---

## Honest scope

This is a methodology demo on a **small, synthetic sample** of binary-event contracts. It does:

- implement the **standard canonical metrics** for calibration (Brier, log score, ECE, MCE, reliability diagram);
- implement both standard recalibration techniques (Platt + isotonic);
- ship a 60-market embedded dataset that is **deliberately miscalibrated** to demonstrate the pipeline working end-to-end.

This is **not**:

- a comprehensive market-efficiency audit of Manifold Markets (or any specific prediction market);
- a real-time data pipeline — the Manifold loader is a one-shot fetch with an 8-second timeout;
- a paper-style claim that *Prediction Markets Are Miscalibrated*. The headline numbers are from synthetic data and exist to demonstrate the pipeline. Replacing the sample with real, large-scale, well-curated resolved markets is the next step.

The methodology is straightforward and **IP-clean**. The synthetic sample is reproducible (seed 424242).

---

## Related repos in this portfolio

- [`validation-gate-stack`](https://github.com/christianmacion26/validation-gate-stack) — the sibling project: statistical-validation gates for systematic-strategy research. Same engineering signature (pure functions, k-of-N composite, honest scope).
- [`vol-regime-classifier`](https://github.com/christianmacion26/vol-regime-classifier) — volatility-regime labeling for time-series ML.
- [`rag-recall`](https://github.com/christianmacion26/rag-recall) — the AI-side sibling.

Built July 2026 by [Christian Macion](https://github.com/christianmacion26).
