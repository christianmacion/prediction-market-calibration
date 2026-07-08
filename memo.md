# Memo — design rationale for `prediction-market-calibration`

## What this is
A small, self-contained implementation of the **calibration-analysis pipeline** for prediction markets. Each function is a pure function on (predicted_prob, resolved_outcome) pairs, named so a research pipeline can read like a textbook.

## What this is NOT
A real-time data pipeline, a market-microstructure model, or an actual audit of any specific prediction market. The library measures **how well a stream of probability predictions lines up with realized outcomes**, and applies standard recalibration techniques. It is methodology, not market commentary.

## Why calibration analysis at all
A "well-calibrated" forecaster is one whose stated probabilities match realized frequencies. The market price of a binary contract is (loosely) the consensus probability of YES. If the price is 70 cents, "calibration" says about 70% of markets priced between 65% and 75% should resolve YES. The audit:
1. Bucket all (predicted_prob, resolved_outcome) pairs into bins.
2. Per-bin, compare mean prediction to mean realized outcome.
3. The deviation from the diagonal is the calibration error.
4. Standard summary metrics: **Brier**, **log score**, **Expected Calibration Error (ECE)**, **Maximum Calibration Error (MCE)**, and the **reliability diagram**.

## Two recalibration techniques
- **Platt scaling** — `q = sigmoid(a · logit(p) + b)`. Fits by Newton iterations on Bernoulli log-likelihood. 2 parameters; identity start `a=1, b=0`; robust on small samples; works well when the miscalibration is roughly a linear-in-logit scaling (e.g., traders are systematically over- or under-confident by a roughly fixed factor).
- **Isotonic regression** — non-parametric monotone mapping via the **pool-adjacent-violators algorithm (PAVA)**. Can correct arbitrary monotone miscalibration; needs more data to avoid overfit. Implemented from scratch in pure NumPy so the project has zero sklearn dependency.

## A few choices worth flagging
- **Brier vs log score**. Both are proper scoring rules; Brier is in squared-error units (range [0, 1]) and is robust to outlier outcomes, while log score is unbounded above and aggressively penalizes confident-but-wrong forecasts. We report both.
- **ECE with 10 bins**. Standard choice; equal-width. The literature also explores equal-mass binning (which is sometimes more stable on small samples), but for ~200 calibration points equal-width is fine.
- **Platt starting point `a=1, b=0` (identity)**. This makes Platt scaling actually a *refinement* of the raw predictions rather than a search from scratch. Newton converges in ≤ 10 iterations on this dataset.
- **PAVA implementation**. The standard PAVA is non-trivial to get exactly right; the version here uses the *weighted block-pooling* variant with backward-looking pooling when a violation propagates back. On 231 points it runs in <1 ms.
- **Sample at "snapshot level" not "market level"**. The headline metric uses every (predicted_prob, resolved_outcome) pair from every snapshot of every market. This is the standard calibration-analysis protocol — it captures whether the *trajectory* of a prediction was well-calibrated, not just the final pre-resolution price.

## What a hiring-manager-skim test should look like
1. `python3 run.py` — runs in <10s, prints the headline, writes the JSON, renders the figure.
2. The headline at the top: **Raw Brier 0.1227 → isotonic 0.0120 (-90.2%); ECE 0.303 → 0.013**.
3. Open `results/figure.png` — four panels: raw reliability, post-Platt reliability, Brier histogram, predicted vs resolved scatter.

## Honest scope
This is a small IP-clean re-implementation of standard calibration analysis, run on **synthetic, deliberately-miscalibrated data** (`data/sample_markets.json`). The miscalibration is genuine (raw ECE ≈ 0.30) so the recalibration step has work to do. None of this is a claim about the calibration of any specific real-world prediction market. Replacing the embedded sample with a large real-world corpus (resolved Manifold markets over a multi-year window, Polymarket resolutions, Betfair exchange prices) is the straightforward next step — the API supports it.

Author: Christian Macion. July 2026.
