"""Post-holdout independent numerical review. No refitting choices."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/quality_v1_1_through_2026-09"
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def equal(actual, expected, label, tol=2e-9):
    if not np.allclose(
        actual, expected, rtol=tol,
        atol=1e-12, equal_nan=True
    ):
        raise AssertionError(f"Numerical mismatch: {label}")


audit = json.loads(
    (DOCS / "final_results_audit_v1.json").read_text()
)
report = json.loads(
    (RESULTS / "final_report.json").read_text()
)
spec = json.loads(
    (ROOT / "configs/inference_v1.json").read_text()
)
manifest = json.loads(
    (SOURCE / "manifest.json").read_text()
)

for key, filename in (
    ("original_predictions_sha256", "final_predictions.parquet"),
    ("original_parameter_trace_sha256", "final_parameter_trace.parquet"),
    ("original_final_report_sha256", "final_report.json"),
):
    assert sha(RESULTS / filename) == audit[key]

assert sha(
    SOURCE / "daily_volatility.parquet"
) == manifest["output_sha256"]["daily_volatility.parquet"]

assert sha(
    SOURCE / "bars_5m.parquet"
) == manifest["output_sha256"]["bars_5m.parquet"]

assert report["model_reoptimization"] is False
assert report["holdout_evaluated"] is True

pred = pd.read_parquet(
    RESULTS / "final_predictions.parquet"
)
trace = pd.read_parquet(
    RESULTS / "final_parameter_trace.parquet"
).set_index(["date_utc", "model"])

daily = pd.read_parquet(
    SOURCE / "daily_volatility.parquet"
)

dates = pd.date_range(
    "2025-01-01", "2026-10-01",
    freq="D", inclusive="left", tz="UTC"
)

assert len(pred) == 638
assert pd.DatetimeIndex(pred.date_utc).equals(dates)
assert len(trace) == 1914
assert len(daily) == 2465

idx = np.flatnonzero(daily.date_utc.isin(dates))
assert len(idx) == 638

rv = daily.rv_5m.to_numpy(dtype=float)
returns = daily.daily_log_return.to_numpy(dtype=float)

print("=== 1. INDEPENDENT SOURCE RECONSTRUCTION ===")

bars = pd.read_parquet(
    SOURCE / "bars_5m.parquet",
    columns=["time_utc", "close_usdt", "complete_candle"],
)

interval = bars.loc[
    (bars.time_utc >= pd.Timestamp("2024-12-31T23:55:00Z"))
    & (bars.time_utc < pd.Timestamp("2026-10-01T00:00:00Z"))
]

assert len(interval) == 638 * 288 + 1
assert interval.complete_candle.all()

prices = interval.close_usdt.to_numpy(dtype=float)
assert np.isfinite(prices).all()
assert (prices > 0).all()

recomputed_rv = np.sum(
    np.diff(np.log(prices)).reshape(638, 288) ** 2,
    axis=1,
)

equal(
    recomputed_rv,
    pred.rv_target.to_numpy(),
    "RV from source prices",
    tol=1e-10,
)

equal(recomputed_rv, rv[idx], "daily RV", tol=1e-10)

close = daily.close_usdt.to_numpy(dtype=float)

equal(
    np.log(close[idx] / close[idx - 1]),
    returns[idx],
    "close-to-close daily returns",
)

print("INDEPENDENT_5M_RV_REPLAY=PASS")

print("=== 2. FOUR BENCHMARKS ===")


def rolling(x, t):
    previous = x[t-30:t]
    assert len(previous) == 30
    return float(np.mean(previous))


def independent_ewma(x):
    result = np.full(len(x), np.nan)
    count = 0
    state = np.nan

    for t in range(1, len(x)):
        previous = x[t-1]

        if not np.isfinite(previous) or previous < 0:
            state = np.nan
            count = 0
        else:
            count += 1
            if count == 30:
                state = float(np.mean(x[t-30:t]))
            elif count > 30:
                state = 0.94 * state + 0.06 * previous

        if count >= 30:
            result[t] = state

    return result


squared = returns ** 2

benchmarks = {
    "D0_rolling_30": np.array([
        rolling(squared, t) for t in idx
    ]),
    "D1_ewma_094": independent_ewma(squared)[idx],
    "I0_rolling_rv_30": np.array([
        rolling(rv, t) for t in idx
    ]),
    "I1_ewma_rv_094": independent_ewma(rv)[idx],
}

for model, values in benchmarks.items():
    equal(values, pred[model].to_numpy(), model)

print("INDEPENDENT_BASELINES=PASS")

print("=== 3. ARCH/GARCH RECURSION ===")

for model in ("ARCH", "GARCH"):
    values = []

    for t, date in zip(idx, dates):
        p = trace.loc[(date, model)]

        omega = float(p.omega_pct_squared)
        alpha = float(p.alpha)
        beta = float(p.beta)

        assert omega > 0
        assert alpha >= 0 and beta >= 0
        assert alpha + beta < 1

        if model == "ARCH":
            assert beta == 0

        x2 = (100 * returns[t-730:t]) ** 2
        h = float(x2.mean())

        for k in range(1, len(x2)):
            h = (
                omega
                + alpha * x2[k-1]
                + beta * h
            )

        values.append(
            (omega + alpha * x2[-1] + beta * h)
            / 10000
        )

    equal(
        np.array(values),
        pred[model].to_numpy(),
        model,
        tol=1e-8,
    )

    print(f"INDEPENDENT_{model}_RECURSION=PASS")

print("=== 4. INDEPENDENT HAR OLS ===")

with np.errstate(invalid="ignore", divide="ignore"):
    logrv = np.log(rv)

X = np.full((len(rv), 4), np.nan)
X[:, 0] = 1.0

for t in range(1, len(rv)):
    X[t, 1] = logrv[t-1]

for window, column in ((7, 2), (30, 3)):
    for t in range(window, len(rv)):
        previous = logrv[t-window:t]
        if np.isfinite(previous).all():
            X[t, column] = previous.mean()

har_predictions = []

for t, date in zip(idx, dates):
    train_x = X[t-730:t]
    train_y = logrv[t-730:t]

    valid = (
        np.isfinite(train_x).all(axis=1)
        & np.isfinite(train_y)
    )

    assert valid.sum() >= 500
    assert np.isfinite(X[t]).all()

    beta, _, rank, _ = np.linalg.lstsq(
        train_x[valid],
        train_y[valid],
        rcond=None,
    )

    assert rank == 4

    residual = (
        train_y[valid] - train_x[valid] @ beta
    )

    smearing = float(np.exp(residual).mean())
    estimate = float(
        np.exp(X[t] @ beta) * smearing
    )

    recorded = trace.loc[(date, "HAR_RV")]

    assert int(recorded.n_train) == int(valid.sum())
    equal(smearing, float(recorded.smearing), "smearing")

    har_predictions.append(estimate)

equal(
    np.array(har_predictions),
    pred.HAR_RV.to_numpy(),
    "independent HAR",
    tol=1e-8,
)

print("INDEPENDENT_HAR_OLS=PASS")

print("=== 5. INDEPENDENT INFERENCE ===")

y = pred.rv_target.to_numpy()

losses = {
    name: (
        np.log(pred[name].to_numpy())
        + y / pred[name].to_numpy()
    )
    for name in spec["models"]
}

for name, loss in losses.items():
    equal(
        float(loss.mean()),
        report["scores"][name]["mean_qlike"],
        f"QLIKE {name}",
    )


def hac(d, lags):
    n = len(d)
    mean = float(np.mean(d))
    centered = d - mean

    variance = float(centered @ centered / n)

    for lag in range(1, lags+1):
        variance += (
            2 * (1 - lag/(lags+1))
            * float(
                centered[lag:] @ centered[:-lag] / n
            )
        )

    assert variance > 0

    se = math.sqrt(variance / n)
    p = math.erfc(abs(mean / se) / math.sqrt(2))

    return {
        "gain": mean,
        "se": se,
        "p": p,
    }


def holm(p_values):
    names = sorted(p_values, key=p_values.get)

    first = min(1.0, 2 * p_values[names[0]])

    return {
        names[0]: first,
        names[1]: max(first, p_values[names[1]]),
    }


contrasts = spec["primary_contrasts"]

original = {
    name: hac(
        losses[c["benchmark"]] - losses[c["candidate"]],
        7,
    )
    for name, c in contrasts.items()
}

adjusted = holm({
    name: result["p"]
    for name, result in original.items()
})

for name, result in original.items():
    stored = report["primary_contrasts"][name]

    equal(
        result["gain"],
        stored["dm_hac"]["mean_gain"],
        "mean gain",
    )
    equal(
        result["se"],
        stored["dm_hac"]["hac_standard_error"],
        "HAC standard error",
    )
    equal(
        result["p"],
        stored["dm_hac"]["p_value_two_sided"],
        "DM p-value",
    )
    equal(
        adjusted[name],
        stored["holm"]["p_holm"],
        "Holm adjustment",
    )

print("INDEPENDENT_DM_HAC_HOLM=PASS")

print("=== 6. POST-HOLDOUT SENSITIVITY ===")

hac_grid = {}

for lag in (0, 1, 3, 7, 14, 21, 30):
    values = {
        name: hac(
            losses[c["benchmark"]]
            - losses[c["candidate"]],
            lag,
        )
        for name, c in contrasts.items()
    }

    corrected = holm({
        name: result["p"]
        for name, result in values.items()
    })

    hac_grid[str(lag)] = {
        name: {**result, "p_holm": corrected[name]}
        for name, result in values.items()
    }

bootstrap_grid = {}

for block in (7, 14, 21, 28):
    bootstrap_grid[str(block)] = {}

    for j, (name, c) in enumerate(contrasts.items()):
        d = (
            losses[c["benchmark"]]
            - losses[c["candidate"]]
        )

        rng = np.random.default_rng(
            20261020 + block * 100 + j
        )

        reps = 2000
        draws = math.ceil(len(d) / block)

        starts = rng.integers(
            0, len(d) - block + 1,
            size=(reps, draws),
        )

        positions = (
            starts[:, :, None]
            + np.arange(block)[None, None, :]
        ).reshape(reps, -1)[:, :len(d)]

        ci = np.quantile(
            d[positions].mean(axis=1),
            [0.025, 0.975],
        )

        bootstrap_grid[str(block)][name] = {
            "ci_95": [float(ci[0]), float(ci[1])],
            "replications": reps,
        }

years = {}

for year in (2025, 2026):
    mask = dates.year == year

    years[str(year)] = {
        "n": int(mask.sum()),
        "gains": {
            name: float(
                (
                    losses[c["benchmark"]]
                    - losses[c["candidate"]]
                )[mask].mean()
            )
            for name, c in contrasts.items()
        },
    }

extremes = {}

for name, c in contrasts.items():
    d = (
        losses[c["benchmark"]]
        - losses[c["candidate"]]
    )

    extremes[name] = {}

    for pct in (1, 5):
        rv_mask = y <= np.quantile(
            y, 1 - pct/100
        )
        diff_mask = np.abs(d) <= np.quantile(
            np.abs(d), 1 - pct/100
        )

        extremes[name][f"exclude_top_{pct}pct_rv"] = {
            "n": int(rv_mask.sum()),
            "gain": float(d[rv_mask].mean()),
        }

        extremes[name][f"exclude_top_{pct}pct_abs_loss_diff"] = {
            "n": int(diff_mask.sum()),
            "gain": float(d[diff_mask].mean()),
        }

diagnostics = {
    "protocol": "HC-BTCVOL-001",
    "status": "POST_HOLDOUT_EXPLORATORY",
    "original_report_sha256": sha(
        RESULTS / "final_report.json"
    ),
    "original_predictions_sha256": sha(
        RESULTS / "final_predictions.parquet"
    ),
    "independent_replay": {
        "rv_from_5m_close": True,
        "four_baselines": True,
        "arch_garch_recursion": True,
        "har_ols_smearing": True,
        "qlike_dm_hac_holm": True,
    },
    "confirmatory_frozen_lag7": {
        name: {**result, "p_holm": adjusted[name]}
        for name, result in original.items()
    },
    "exploratory_hac_lag_grid": hac_grid,
    "exploratory_bootstrap_block_grid": bootstrap_grid,
    "exploratory_year_split": years,
    "exploratory_outlier_sensitivity": extremes,
    "caveat": (
        "All sensitivity variants and exclusions "
        "were specified after the holdout was viewed."
    ),
}

(DOCS / "postholdout_diagnostics_v1.json").write_text(
    json.dumps(diagnostics, indent=2) + "\n"
)

print()
print("=== POST-HOLDOUT ROBUSTNESS ===")

for name, result in original.items():
    print(
        f"FROZEN_{name}: "
        f"GAIN={result['gain']:.8f} "
        f"P_HOLM={adjusted[name]:.8f}"
    )

for lag, entries in hac_grid.items():
    print(
        f"HAC_LAGS={lag} "
        + " ".join(
            f"{name}_P_HOLM={v['p_holm']:.6g}"
            for name, v in entries.items()
        )
    )

for block, entries in bootstrap_grid.items():
    print(
        f"BOOTSTRAP_BLOCK={block} "
        + " ".join(
            f"{name}_CI=[{v['ci_95'][0]:.5f},"
            f"{v['ci_95'][1]:.5f}]"
            for name, v in entries.items()
        )
    )

for name, entries in extremes.items():
    for key, value in entries.items():
        print(
            f"EXPLORATORY_{name}_{key}: "
            f"N={value['n']} "
            f"GAIN={value['gain']:.8f}"
        )

print("FROZEN_MODELS_CHANGED=NO")
print("INDEPENDENT_NUMERICAL_REPLAY=PASS")
print("ROBUSTNESS_DIAGNOSTICS=PASS")
