"""HC-BTCVOL-001: HAR-RV validation, holdout protected."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hc_vol.har_models import design, forecast_one
from hc_vol.baselines import qlike

DATA = ROOT / "data/processed/quality_v1_1_through_2026-09"
BASE = ROOT / "data/processed/baselines_validation_v1"
OUT = ROOT / "data/processed/har_validation_v1"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


manifest = json.loads(
    (DATA / "manifest.json").read_text()
)
baseline_report = json.loads(
    (BASE / "validation_report.json").read_text()
)

source = DATA / "daily_volatility.parquet"
baseline_path = BASE / "validation_predictions.parquet"

assert digest(source) == manifest[
    "output_sha256"
]["daily_volatility.parquet"]

assert baseline_report["input_sha256"] == digest(source)
assert digest(baseline_path) == (
    baseline_report["predictions_sha256"]
)

full = pd.read_parquet(source)

# Holdout excluded BEFORE construction of features.
cutoff = pd.Timestamp("2025-01-01T00:00:00Z")
start = pd.Timestamp("2023-01-01T00:00:00Z")

daily = full.loc[
    full.date_utc < cutoff
].reset_index(drop=True)

assert len(daily) == 1827
assert daily.date_utc.max() < cutoff

log_rv, features = design(
    daily.rv_5m.to_numpy(dtype=float)
)

indices = np.flatnonzero(
    daily.date_utc >= start
)

assert len(indices) == 731
assert indices.min() >= 730

print("=== HAR-RV WALK-FORWARD ===", flush=True)

records = []

for n, t in enumerate(indices, 1):
    result = forecast_one(
        log_rv, features, int(t),
        window=730,
        min_train=500,
    )

    beta = result["beta"]

    records.append({
        "date_utc": daily.loc[t, "date_utc"],
        "success": result["success"],
        "reason": result["reason"],
        "forecast": result["forecast"],
        "n_train": result["n_train"],
        "beta_0": beta[0] if beta is not None else np.nan,
        "beta_d": beta[1] if beta is not None else np.nan,
        "beta_w": beta[2] if beta is not None else np.nan,
        "beta_m": beta[3] if beta is not None else np.nan,
        "smearing": result["smearing"],
        "condition_number": result["condition_number"],
        "residual_rms": result["residual_rms"],
    })

    if n % 100 == 0 or n == len(indices):
        print(
            f"HAR_PROGRESS={n}/{len(indices)}",
            flush=True,
        )

trace = pd.DataFrame(records)

unexpected = trace[
    ~trace.success
    & (trace.reason != "invalid_features")
]

if len(unexpected):
    print(unexpected.to_string(index=False))
    raise RuntimeError("HAR_ESTIMATION_FAILURE")

baseline = pd.read_parquet(baseline_path)

merged = baseline.merge(
    trace[["date_utc", "forecast"]].rename(
        columns={"forecast": "HAR_RV"}
    ),
    on="date_utc",
    validate="one_to_one",
)

assert len(merged) == 731

columns = [
    "rv_target",
    "I0_rolling_rv_30",
    "I1_ewma_rv_094",
    "HAR_RV",
]

common_mask = merged[columns].notna().all(axis=1)
benchmark_mask = merged[columns[:-1]].notna().all(axis=1)

if not common_mask.equals(benchmark_mask):
    raise RuntimeError("COMMON_SAMPLE_MISMATCH")

common = merged.loc[common_mask].copy()

assert len(common) == 700
assert (
    common[columns[1:]] > 0
).all().all()

scores = {
    name: qlike(
        common.rv_target.to_numpy(),
        common[name].to_numpy(),
    )
    for name in columns[1:]
}

def losses(name):
    y = common.rv_target.to_numpy()
    p = common[name].to_numpy()
    return np.log(p) + y / p


gain = float(np.mean(
    losses("I1_ewma_rv_094")
    - losses("HAR_RV")
))

OUT.mkdir(parents=True, exist_ok=True)

trace_file = OUT / "har_parameter_trace.parquet"
prediction_file = OUT / "validation_predictions.parquet"

trace.to_parquet(trace_file, index=False)
merged.to_parquet(prediction_file, index=False)

report = {
    "protocol": "HC-BTCVOL-001",
    "phase": "validation_only",
    "model": "log_HAR_RV_1_7_30",
    "estimation_window_calendar_days": 730,
    "minimum_training_rows": 500,
    "smearing": "training_residuals_only",
    "validation_dates": 731,
    "common_sample": len(common),
    "successful_forecasts": int(trace.success.sum()),
    "invalid_feature_dates": int(
        (trace.reason == "invalid_features").sum()
    ),
    "training_rows_min": int(
        trace.loc[trace.success, "n_train"].min()
    ),
    "condition_number_max": float(
        trace.loc[trace.success, "condition_number"].max()
    ),
    "scores": scores,
    "har_gain_over_ewma_rv": gain,
    "input_sha256": digest(source),
    "trace_sha256": digest(trace_file),
    "predictions_sha256": digest(prediction_file),
    "holdout_evaluated": False,
}

(OUT / "validation_report.json").write_text(
    json.dumps(report, indent=2) + "\n"
)

print()
print("=== HAR-RV VALIDATION ===")

for name, value in scores.items():
    print(
        f"{name}: N={value['n']} "
        f"QLIKE={value['mean_qlike']:.8f}"
    )

print(f"HAR_GAIN_OVER_EWMA_RV={gain:.8f}")
print(
    "SUCCESSFUL_FORECASTS="
    f"{report['successful_forecasts']}"
)
print(
    "INVALID_FEATURE_DATES="
    f"{report['invalid_feature_dates']}"
)
print(
    "TRAINING_ROWS_MIN="
    f"{report['training_rows_min']}"
)
print(
    "MAX_CONDITION_NUMBER="
    f"{report['condition_number_max']:.4f}"
)
print("COMMON_SAMPLE=700")
print("SMEARING=TRAIN_ONLY")
print("HOLDOUT_EVALUATED=NO")
print("HAR_VALIDATION=PASS")
