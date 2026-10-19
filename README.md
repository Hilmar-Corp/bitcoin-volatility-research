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

Raw data and generated datasets are excluded
from GitHub. No performance results are claimed.

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

No forecasting performance is claimed.
