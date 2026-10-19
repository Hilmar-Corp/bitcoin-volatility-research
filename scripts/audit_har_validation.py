"""HC-BTCVOL-001: validation-only HAR-RV research audit."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hc_vol.har_models import design, forecast_one

BASE = ROOT / "data/processed/har_validation_v1"
DATA = ROOT / (
    "data/processed/quality_v1_1_through_2026-09"
)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1 << 20), b""
        ):
            h.update(chunk)
    return h.hexdigest()


report = json.loads(
    (BASE / "validation_report.json").read_text()
)
manifest = json.loads(
    (DATA / "manifest.json").read_text()
)

source = DATA / "daily_volatility.parquet"
trace_path = BASE / "har_parameter_trace.parquet"
pred_path = BASE / "validation_predictions.parquet"

assert report["holdout_evaluated"] is False
assert report["phase"] == "validation_only"
assert report["input_sha256"] == sha256(source)
assert report["trace_sha256"] == sha256(trace_path)
assert report["predictions_sha256"] == sha256(pred_path)

assert manifest["output_sha256"][
    "daily_volatility.parquet"
] == sha256(source)

trace = pd.read_parquet(trace_path)
pred = pd.read_parquet(pred_path)

expected_dates = pd.date_range(
    "2023-01-01",
    "2025-01-01",
    freq="D",
    inclusive="left",
    tz="UTC",
)

for frame in (trace, pred):
    assert len(frame) == 731
    assert frame.date_utc.is_unique
    assert pd.DatetimeIndex(
        frame.date_utc
    ).equals(expected_dates)

print("=== 1. FORECAST INTEGRITY ===")

assert int(trace.success.sum()) == 701

invalid = trace.loc[
    ~trace.success, "date_utc"
]

expected_invalid = pd.date_range(
    "2023-03-25",
    periods=30,
    freq="D",
    tz="UTC",
)

assert pd.DatetimeIndex(
    invalid
).equals(expected_invalid)

assert (
    trace.loc[
        ~trace.success, "reason"
    ] == "invalid_features"
).all()

successful = trace.loc[
    trace.success
]

assert successful.n_train.between(
    500, 730
).all()

parameters = [
    "beta_0", "beta_d", "beta_w",
    "beta_m", "smearing",
    "condition_number", "residual_rms",
]

assert np.isfinite(
    successful[parameters].to_numpy()
).all()

assert (successful.smearing > 0).all()
assert (
    successful.condition_number < 1e8
).all()

print("SUCCESSFUL_FORECASTS=701")
print("INVALID_FEATURE_DATES=30")
print("NUMERICAL_INTEGRITY=PASS")

print("=== 2. COEFFICIENT DISTRIBUTIONS ===")

for col in parameters:
    values = successful[col].to_numpy()
    p05, p50, p95 = np.quantile(
        values, [0.05, 0.50, 0.95]
    )
    print(
        f"{col}: "
        f"P05={p05:.6f} "
        f"MEDIAN={p50:.6f} "
        f"P95={p95:.6f}"
    )

print("=== 3. INDEPENDENT FORECAST REPLAY ===")

daily = pd.read_parquet(source)
daily = daily.loc[
    daily.date_utc
    < pd.Timestamp("2025-01-01T00:00:00Z")
].reset_index(drop=True)

log_rv, features = design(
    daily.rv_5m.to_numpy(dtype=float)
)

for date in (
    "2023-01-01",
    "2023-04-24",
    "2024-12-31",
):
    timestamp = pd.Timestamp(date, tz="UTC")

    index = int(
        np.flatnonzero(
            daily.date_utc == timestamp
        )[0]
    )

    recalculated = forecast_one(
        log_rv, features, index
    )

    recorded = trace.loc[
        trace.date_utc == timestamp
    ].iloc[0]

    assert recalculated["success"]

    np.testing.assert_allclose(
        recalculated["forecast"],
        recorded.forecast,
        rtol=1e-12,
        atol=0,
    )

    print(f"REPLAY_{date}=PASS")

print("=== 4. PAIRED QLIKE AUDIT ===")

cols = [
    "rv_target",
    "I0_rolling_rv_30",
    "I1_ewma_rv_094",
    "HAR_RV",
]

common = pred.loc[
    pred[cols].notna().all(axis=1)
].copy()

assert len(common) == 700
assert (
    common[cols[1:]] > 0
).all().all()

y = common.rv_target.to_numpy()

def loss(name):
    f = common[name].to_numpy()
    return np.log(f) + y/f


loss_rolling = loss("I0_rolling_rv_30")
loss_ewma = loss("I1_ewma_rv_094")
loss_har = loss("HAR_RV")

for name, values in (
    ("I0_rolling_rv_30", loss_rolling),
    ("I1_ewma_rv_094", loss_ewma),
    ("HAR_RV", loss_har),
):
    actual = float(values.mean())
    expected = report["scores"][name]["mean_qlike"]

    np.testing.assert_allclose(
        actual, expected,
        rtol=0, atol=1e-10
    )

    print(f"{name}_QLIKE={actual:.8f}")

# Positive values favour HAR.
diff = loss_ewma - loss_har

n = len(diff)
gain = float(diff.mean())

assert abs(
    gain - report["har_gain_over_ewma_rv"]
) < 1e-10

print("=== 5. EXPLORATORY HAC UNCERTAINTY ===")

# Bartlett/Newey-West, seven daily lags.
lags = 7
centered = diff - gain

long_run = float(
    np.dot(centered, centered)/n
)

for lag in range(1, lags+1):
    covariance = float(
        np.dot(
            centered[lag:],
            centered[:-lag]
        ) / n
    )
    long_run += (
        2 * (1-lag/(lags+1)) * covariance
    )

assert long_run > 0

se = float(np.sqrt(long_run/n))
low = gain - 1.96*se
high = gain + 1.96*se

print(f"HAR_QLIKE_GAIN={gain:.8f}")
print(f"HAC_STANDARD_ERROR={se:.8f}")
print(f"HAC_95_CI=[{low:.8f}, {high:.8f}]")

for year, expected_n in (
    (2023, 334),
    (2024, 366),
):
    mask = (
        common.date_utc.dt.year.to_numpy()
        == year
    )

    assert int(mask.sum()) == expected_n

    print(
        f"HAR_GAIN_{year}="
        f"{float(diff[mask].mean()):.8f} "
        f"N={expected_n}"
    )

audit = {
    "protocol": "HC-BTCVOL-001",
    "stage": "validation_only",
    "model": "HAR_RV",
    "common_sample": n,
    "har_gain_over_ewma_rv": gain,
    "hac_lags": lags,
    "hac_standard_error": se,
    "hac_95_ci": [low, high],
    "holdout_evaluated": False,
    "confirmatory_inference": False,
    "parameter_medians": {
        col: float(successful[col].median())
        for col in parameters
    },
}

(BASE / "har_diagnostic.json").write_text(
    json.dumps(audit, indent=2) + "\n"
)

print()
print("=== FINAL HAR AUDIT ===")
print("PARAMETER_INTEGRITY=PASS")
print("INDEPENDENT_REPLAY=PASS")
print("COMMON_SAMPLE=700")
print("HOLDOUT_EVALUATED=NO")
print("HAR_AUDIT=PASS")
