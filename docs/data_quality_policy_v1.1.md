# HC-BTCVOL-001 — Data Quality Policy v1.1

## Source

Binance BTCUSDT Spot, 5-minute OHLCV archives.

Period: 2020-01-01 through 2026-09-30 UTC.

82 monthly source archives were verified using
their published SHA-256 checksums.

## Observed anomalies

- 462 missing five-minute candles
- 7 shortened candles
- 1 invalid negative-duration candle
- 16 affected daily realized-variance targets

## Processing rules

1. Preserve all expected UTC timestamps.
2. Never interpolate missing prices.
3. Never forward-fill absent market observations.
4. Reject invalid or incomplete candles for RV.
5. Reject five-minute returns crossing data gaps.
6. Require 288 valid returns per daily RV.
7. Store invalid realized variance as NaN.
8. Compute daily close-to-close returns separately.
9. Require valid consecutive daily close prices.
10. Reject unregistered source hash changes.

## Known exception

2020-12-21 14:05 UTC:
negative candle duration of -1059479 milliseconds.

The source record is retained for audit purposes,
but its five-minute return is excluded from RV.

## Statistical implications

The daily-return and realized-variance model families
may have different admissible estimation samples.

All missing-data handling must remain causal.

Comparisons between models using identical inputs
must use identical evaluation dates.

The final 2025-2026 OOS window contains 638 valid
daily realized-variance observations.

## Reproducibility

See:
- configs/source_quality_reference_v1.json
- docs/data_manifest_v1_1.json

The original protocol v1.0.0 remains unchanged.

No backtest or forecasting performance is claimed.
