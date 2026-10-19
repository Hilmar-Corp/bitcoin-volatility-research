# HC-BTCVOL-001 — Pre-holdout inference freeze

The initial research specification is protocol v1.0.0.
This document records the inference implementation
before the first evaluation of 2025–2026 data.

## Primary hypotheses

1. GARCH(1,1) versus EWMA of daily squared returns.
2. Log HAR-RV(1,7,30) versus EWMA-RV.

No other comparisons determine confirmatory claims.

## Forecasting

- One-day-ahead predictions, strictly causal.
- 730-calendar-day rolling estimation where specified.
- No target or parameter selection using the holdout.
- Frozen model implementations and dependencies.
- Holdout dates: 2025-01-01 through 2026-09-30.
- All 638 target dates must have valid positive
  forecasts from all seven models.
- A missing or nonconverged forecast blocks the
  confirmatory analysis; no silent date deletion.

## Confirmatory inference

- Target: daily realized variance from 5m returns.
- Loss: log(forecast variance) + RV/forecast variance.
- Difference: benchmark loss minus candidate loss.
- Positive differences favor the candidate.
- Two-sided normal-approximation Diebold-Mariano
  tests with Newey-West Bartlett HAC, lag 7.
- Holm step-down familywise correction, alpha 0.05,
  across exactly two primary hypotheses.
- Supplementary moving-block percentile interval:
  block length 14 days, 10,000 replications,
  fixed seed 20261019.
- Bootstrap intervals are not the decision rule.

## Validation interpretation

The preliminary HAC intervals on 2023-2024 both
include zero. Validation cannot establish a
statistically significant improvement.

Validation samples contain 730 daily-family
comparisons and 700 intraday-family comparisons.
Scores across these samples must not be compared
without aligning evaluation dates.

## Audit trail

See configs/inference_v1.json,
docs/validation_summary_v1.json and
docs/model_freeze_manifest_v1.json.

Any substantive change after this freeze requires
an explicitly versioned protocol amendment.
