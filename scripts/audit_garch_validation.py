"""HC-BTCVOL-001: ARCH/GARCH validation audit."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

BASE = ROOT / "data/processed/baselines_validation_v1"
GARCH = ROOT / "data/processed/garch_validation_v1"

trace = pd.read_parquet(
    GARCH / "parameter_trace.parquet"
)

pred = pd.read_parquet(
    GARCH / "validation_predictions.parquet"
)

report = json.loads(
    (GARCH / "validation_report.json").read_text()
)

convergence = json.loads(
    (GARCH / "convergence_report.json").read_text()
)

assert report["holdout_evaluated"] is False
assert report["window_calendar_days"] == 730
assert len(trace) == 1462
assert len(pred) == 731
assert pred["date_utc"].is_unique

assert pred["date_utc"].min() == pd.Timestamp(
    "2023-01-01T00:00:00Z"
)
assert pred["date_utc"].max() == pd.Timestamp(
    "2024-12-31T00:00:00Z"
)

print("=== 1. CONVERGENCE ===")

for model in ("ARCH", "GARCH"):
    sub = trace.loc[
        trace["model"] == model
    ].copy()

    assert len(sub) == 731
    assert sub["date_utc"].is_unique
    assert sub["success"].all()

    assert (
        convergence["models"][model]["attempted"]
        == 731
    )
    assert (
        convergence["models"][model]["converged"]
        == 731
    )

    for col in (
        "forecast",
        "omega_pct_squared",
        "alpha",
        "beta",
        "persistence",
        "objective",
    ):
        assert np.isfinite(sub[col]).all(), (
            model, col
        )

    assert (sub["forecast"] > 0).all()
    assert (sub["omega_pct_squared"] > 0).all()
    assert (sub["alpha"] >= 0).all()
    assert (sub["beta"] >= 0).all()
    assert (sub["persistence"] < 1).all()

    np.testing.assert_allclose(
        sub["persistence"].to_numpy(),
        (sub["alpha"] + sub["beta"]).to_numpy(),
        atol=1e-12,
    )

    if model == "ARCH":
        assert (sub["beta"] == 0).all()

    print(f"{model}_CONVERGENCE=731/731")

    print(f"=== {model} PARAMETER DISTRIBUTION ===")

    for col in (
        "omega_pct_squared",
        "alpha",
        "beta",
        "persistence",
    ):
        values = sub[col].to_numpy()
        quantiles = np.quantile(
            values, [0.05, 0.50, 0.95]
        )

        print(
            f"{col}: "
            f"P05={quantiles[0]:.8f} "
            f"MEDIAN={quantiles[1]:.8f} "
            f"P95={quantiles[2]:.8f}"
        )

    print(
        "NEAR_PERSISTENCE_CAP="
        f"{int((sub['persistence'] >= 0.995).sum())}"
    )
    print(
        "NEAR_ZERO_ALPHA="
        f"{int((sub['alpha'] <= 1e-5).sum())}"
    )
    print(
        "MAX_OPTIMIZER_ITERATIONS="
        f"{int(sub['optimizer_iterations'].max())}"
    )

print("=== 2. COMMON-SAMPLE EVALUATION ===")

required = [
    "date_utc",
    "rv_target",
    "D0_rolling_30",
    "D1_ewma_094",
    "ARCH",
    "GARCH",
]

common = pred[required].dropna()

assert len(common) == 730
assert (
    common[[
        "D0_rolling_30",
        "D1_ewma_094",
        "ARCH",
        "GARCH",
    ]] > 0
).all().all()

target = common["rv_target"].to_numpy()

losses = {}

for name in required[2:]:
    variance = common[name].to_numpy()

    loss = (
        np.log(variance)
        + target / variance
    )

    assert np.isfinite(loss).all()

    losses[name] = loss

    mean = float(loss.mean())
    print(f"{name}_QLIKE={mean:.8f}")

    if name in report["scores"]:
        np.testing.assert_allclose(
            mean,
            report["scores"][name]["mean_qlike"],
            rtol=0,
            atol=1e-10,
        )

print("=== 3. GARCH VS EWMA ===")

# Positive differences favour GARCH.
diff = (
    losses["D1_ewma_094"]
    - losses["GARCH"]
)

n = len(diff)
mean_diff = float(diff.mean())

# Newey-West long-run variance estimator,
# Bartlett kernel with 7 daily lags.
lags = 7
centered = diff - mean_diff

long_run_var = float(
    np.dot(centered, centered) / n
)

for lag in range(1, lags + 1):
    covariance = float(
        np.dot(
            centered[lag:],
            centered[:-lag],
        ) / n
    )

    weight = 1 - lag / (lags + 1)

    long_run_var += (
        2 * weight * covariance
    )

assert long_run_var > 0

standard_error = np.sqrt(
    long_run_var / n
)

ci_low = mean_diff - 1.96 * standard_error
ci_high = mean_diff + 1.96 * standard_error

print(f"MEAN_QLIKE_GAIN={mean_diff:.8f}")
print(f"HAC_STANDARD_ERROR={standard_error:.8f}")
print(f"HAC_95_CI=[{ci_low:.8f}, {ci_high:.8f}]")

for year in (2023, 2024):
    mask = (
        common["date_utc"].dt.year.to_numpy()
        == year
    )

    yearly_gain = float(diff[mask].mean())

    print(
        f"QLIKE_GAIN_{year}={yearly_gain:.8f} "
        f"N={int(mask.sum())}"
    )

audit = {
    "protocol": "HC-BTCVOL-001",
    "stage": "validation_only",
    "model": "GARCH_vs_EWMA",
    "common_dates": n,
    "mean_qlike_gain": mean_diff,
    "hac_lags": lags,
    "hac_standard_error": float(
        standard_error
    ),
    "hac_95_ci": [
        float(ci_low),
        float(ci_high),
    ],
    "holdout_evaluated": False,
    "confirmatory_test": False,
}

destination = (
    GARCH / "parameter_diagnostic.json"
)
destination.write_text(
    json.dumps(audit, indent=2) + "\n"
)

print()
print("=== FINAL AUDIT ===")
print("PARAMETER_CONSTRAINTS=PASS")
print("FORECAST_INTEGRITY=PASS")
print("COMMON_SAMPLE=730")
print("HOLDOUT_EVALUATED=NO")
print("GARCH_AUDIT=PASS")
