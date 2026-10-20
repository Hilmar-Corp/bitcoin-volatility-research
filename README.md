# Bitcoin Volatility Research

HilmarCorp Research — HC-BTCVOL-001

Research question:
Can statistical models improve one-day-ahead
Bitcoin volatility forecasts out of sample?

Models:
- Rolling variance
- EWMA
- ARCH(1)
- GARCH(1,1)
- Rolling realized variance
- EWMA-RV
- HAR-RV

Data: Binance Spot BTCUSDT, 5-minute candles.
Period: 2020-01-01 to 2026-09-30 UTC.
Primary loss: QLIKE.
Evaluation: chronological walk-forward.

Raw data and full reconstructed datasets are excluded
from GitHub. Final forecast artifacts are in `results/`.

Code license: Apache License 2.0.

## Verified research dataset

Data quality policy: v1.1

- Binance BTCUSDT spot, 5-minute candles
- Study period: 2020-01-01 to 2026-09-30 UTC
- 82 SHA-256-verified source archives
- 709,920 expected 5-minute timestamps
- 709,458 observed candles
- 462 missing 5-minute candles
- 7 shortened candles
- 1 candle with invalid negative duration
- 2,465 daily observations
- 2,449 admissible realized-variance observations
- 16 documented invalid realized-variance days
- 2,465 admissible daily close-to-close returns

The final out-of-sample period, 2025-01-01 through
2026-09-30, contains 638 valid daily RV targets.

Missing observations are never interpolated.
The full calendar is preserved.

The original protocol is preserved under
`protocol-v1.0.0`.

The data-quality amendment is versioned separately.

See:
- `docs/data_quality_policy_v1.1.md`
- `docs/data_manifest_v1_1.json`
- `configs/source_quality_reference_v1.json`

Final holdout results: `docs/research_findings_v1.md`.

## Post-holdout scientific robustness review

The independent numerical replay, exploratory
robustness diagnostics, figures and academic discussion
are archived separately from the first-look results.

- `docs/postholdout_scientific_review_v1.md`
- `docs/postholdout_diagnostics_v1.json`
- `figures/`

The preregistered results remain unchanged under
`final-results-v1.0.0`.

Alternative HAC lags, bootstrap blocks and outlier
exclusions are post-holdout exploratory analyses.
They do not constitute new confirmatory tests.

This is an internally audited research artifact,
not an external peer-reviewed publication or
institutional model validation.
