"""Validation-only ARCH/GARCH walk-forward: 2023-2024."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hc_vol.garch_models import WINDOW, estimate
from hc_vol.baselines import qlike

DATA = ROOT / (
    "data/processed/quality_v1_1_through_2026-09"
)
BASELINES = ROOT / (
    "data/processed/baselines_validation_v1"
)
OUT = ROOT / "data/processed/garch_validation_v1"

OUT.mkdir(parents=True, exist_ok=True)

manifest = json.loads(
    (DATA / "manifest.json").read_text()
)

source = DATA / "daily_volatility.parquet"

h = hashlib.sha256(source.read_bytes()).hexdigest()

assert h == manifest["output_sha256"][
    "daily_volatility.parquet"
], "Source fingerprint mismatch"

baseline_report = json.loads(
    (BASELINES / "validation_report.json").read_text()
)

assert baseline_report["input_sha256"] == h

daily = pd.read_parquet(source)

cutoff = pd.Timestamp("2025-01-01T00:00:00Z")
start = pd.Timestamp("2023-01-01T00:00:00Z")

# Explicitly exclude final holdout before estimation.
daily = daily.loc[
    daily["date_utc"] < cutoff
].reset_index(drop=True)

assert daily["date_utc"].max() < cutoff
assert daily["daily_log_return"].notna().all()

returns = daily[
    "daily_log_return"
].to_numpy(dtype=float)

validation_indices = daily.index[
    daily["date_utc"] >= start
].to_numpy()

assert len(validation_indices) == 731
assert validation_indices.min() >= WINDOW

records = []

for model in ("ARCH", "GARCH"):
    warm = None

    for n, index in enumerate(
        validation_indices, start=1
    ):
        history = returns[
            index-WINDOW:index
        ]

        result = estimate(
            history, model, warm=warm
        )

        warm = (
            result["theta"]
            if result["success"]
            else None
        )

        row = {
            "date_utc": daily.loc[
                index, "date_utc"
            ],
            "model": model,
            "success": result["success"],
            "forecast": result["forecast"],
            "omega_pct_squared": result.get(
                "omega_pct_squared", np.nan
            ),
            "alpha": result.get(
                "alpha", np.nan
            ),
            "beta": result.get(
                "beta", np.nan
            ),
            "persistence": result.get(
                "persistence", np.nan
            ),
            "optimizer_iterations": result.get(
                "optimizer_iterations", np.nan
            ),
            "objective": result.get(
                "objective", np.nan
            ),
        }

        records.append(row)

        if n % 100 == 0 or n == 731:
            print(
                f"{model}: {n}/731 "
                "estimations completed",
                flush=True,
            )

results = pd.DataFrame(records)

path = OUT / "parameter_trace.parquet"
results.to_parquet(path, index=False)

failures = results.loc[
    ~results["success"]
]

convergence = {
    model: {
        "attempted": 731,
        "converged": int(
            results.loc[
                results["model"] == model,
                "success",
            ].sum()
        ),
    }
    for model in ("ARCH", "GARCH")
}

(OUT / "convergence_report.json").write_text(
    json.dumps({
        "models": convergence,
        "failed_dates": [
            {
                "model": row.model,
                "date": row.date_utc.isoformat(),
            }
            for row in failures.itertuples()
        ],
    }, indent=2) + "\n"
)

if len(failures):
    print("ESTIMATION=BLOCKED_NONCONVERGENCE")
    print(f"FAILED_ESTIMATIONS={len(failures)}")
    raise SystemExit(2)

pred = results.pivot(
    index="date_utc",
    columns="model",
    values="forecast",
)

baseline_path = (
    BASELINES / "validation_predictions.parquet"
)
baseline = pd.read_parquet(
    baseline_path
).set_index("date_utc")

joined = baseline.join(
    pred, how="inner", validate="one_to_one"
)

assert len(joined) == 731

# Common evaluation sample within daily-return family.
common = joined[
    ["rv_target", "D1_ewma_094", "ARCH", "GARCH"]
].dropna()

assert len(common) == 730
assert (common[["D1_ewma_094", "ARCH", "GARCH"]] > 0).all().all()

scores = {
    model: qlike(
        common["rv_target"].to_numpy(),
        common[model].to_numpy(),
    )
    for model in (
        "D1_ewma_094",
        "ARCH",
        "GARCH",
    )
}

joined.reset_index().to_parquet(
    OUT / "validation_predictions.parquet",
    index=False,
)

report = {
    "protocol": "HC-BTCVOL-001",
    "stage": "validation_only",
    "method": "Gaussian QMLE",
    "mean_model": "zero",
    "window_calendar_days": WINDOW,
    "returns_scale_for_estimation": 100,
    "forecast_rescaling_factor": 10000,
    "persistence_cap": 0.999,
    "validation_dates": 731,
    "common_qlike_dates": len(common),
    "holdout_evaluated": False,
    "convergence": convergence,
    "scores": scores,
}

(OUT / "validation_report.json").write_text(
    json.dumps(report, indent=2) + "\n"
)

print()
print("=== ARCH/GARCH VALIDATION ===")

for model, result in scores.items():
    print(
        f"{model}: N={result['n']} "
        f"QLIKE={result['mean_qlike']:.8f}"
    )

print("COMMON_SAMPLE=PASS")
print("CONVERGENCE=PASS")
print("HOLDOUT_EVALUATED=NO")
print("GARCH_VALIDATION=PASS")
