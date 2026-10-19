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
