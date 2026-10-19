"""HC-BTCVOL-001 — frozen-model final holdout orchestration.

No parameter tuning is allowed in this script.
Preflight and synthetic tests do not score holdout observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hc_vol.baselines import rolling_30, ewma
from hc_vol.garch_models import estimate
from hc_vol.har_models import design, forecast_one
from hc_vol.inference import (
    qlike_losses,
    hac_mean_test,
    holm_adjust,
    moving_block_ci,
)

SOURCE = ROOT / (
    "data/processed/quality_v1_1_through_2026-09"
)

OUTPUT = ROOT / (
    "data/processed/final_holdout_v1"
)

START = pd.Timestamp("2025-01-01T00:00:00Z")
END = pd.Timestamp("2026-10-01T00:00:00Z")

MODELS = (
    "D0_rolling_30",
    "D1_ewma_094",
    "I0_rolling_rv_30",
    "I1_ewma_rv_094",
    "ARCH",
    "GARCH",
    "HAR_RV",
)

FROZEN_COMMIT = (
    "166bb357f6c15c46d7e9a81d95609ed8b8f522dd"
)


def digest(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for part in iter(
            lambda: f.read(1 << 20), b""
        ):
            h.update(part)

    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def preflight():
    frozen = json.loads(
        (ROOT / "docs/model_freeze_manifest_v1.json")
        .read_text()
    )

    spec = json.loads(
        (ROOT / "configs/inference_v1.json")
        .read_text()
    )

    local_manifest = json.loads(
        (SOURCE / "manifest.json").read_text()
    )

    published_manifest = json.loads(
        (ROOT / "docs/data_manifest_v1_1.json")
        .read_text()
    )

    require(
        frozen["release"] == "models-inference-v1.0.0",
        "FREEZE_RELEASE_MISMATCH",
    )

    for name, expected in (
        frozen["locked_file_sha256"].items()
    ):
        path = ROOT / name

        require(
            path.is_file() and digest(path) == expected,
            f"FROZEN_FILE_DRIFT={name}",
        )

    require(
        local_manifest == published_manifest,
        "DATA_MANIFEST_DRIFT",
    )

    source_file = (
        SOURCE / "daily_volatility.parquet"
    )

    require(
        digest(source_file)
        == local_manifest["output_sha256"][
            "daily_volatility.parquet"
        ],
        "DATA_SHA256_MISMATCH",
    )

    require(
        spec["protocol"] == "HC-BTCVOL-001",
        "PROTOCOL_MISMATCH",
    )

    require(
        tuple(spec["models"]) == MODELS,
        "MODEL_LIST_MISMATCH",
    )

    require(
        spec["holdout_start"] == "2025-01-01"
        and spec["holdout_end"] == "2026-09-30"
        and spec["required_common_sample"] == 638,
        "HOLDOUT_SPECIFICATION_MISMATCH",
    )

    require(
        spec["hac_lags"] == 7
        and spec["familywise_alpha"] == 0.05,
        "INFERENCE_PARAMETERS_CHANGED",
    )

    expected_hypotheses = {
        "GARCH_vs_EWMA": {
            "benchmark": "D1_ewma_094",
            "candidate": "GARCH",
        },
        "HAR_RV_vs_EWMA_RV": {
            "benchmark": "I1_ewma_rv_094",
            "candidate": "HAR_RV",
        },
    }

    require(
        spec["primary_contrasts"]
        == expected_hypotheses,
        "PRIMARY_HYPOTHESES_CHANGED",
    )

    bootstrap = spec["supplementary_bootstrap"]

    require(
        bootstrap["block_length_days"] == 14
        and bootstrap["replications"] == 10000
        and bootstrap["random_seed"] == 20261019,
        "BOOTSTRAP_PARAMETERS_CHANGED",
    )

    require(
        spec["holdout_evaluated"] is False,
        "FROZEN_PROTOCOL_STATUS_CHANGED",
    )

    tag_commit = subprocess.check_output(
        [
            "git",
            "-C", str(ROOT),
            "rev-parse",
            "models-inference-v1.0.0^{commit}",
        ],
        text=True,
    ).strip()

    require(
        tag_commit == FROZEN_COMMIT,
        "MODEL_FREEZE_TAG_DRIFT",
    )

    print("FROZEN_MODELS_SHA256=PASS")
    print("FROZEN_DATA_SHA256=PASS")
    print("INFERENCE_PARAMETERS=PASS")
    print("MODEL_FREEZE_TAG=PASS")
    print("HOLDOUT_EVALUATED=NO")

    return spec, local_manifest


def produce_forecasts(daily):
    calendar = pd.DatetimeIndex(
        daily["date_utc"]
    )

    expected = pd.date_range(
        "2020-01-01",
        "2026-10-01",
        freq="D",
        inclusive="left",
        tz="UTC",
    )

    require(
        calendar.equals(expected),
        "DATA_CALENDAR_MISMATCH",
    )

    indices = np.flatnonzero(
        (calendar >= START)
        & (calendar < END)
    )

    require(
        len(indices) == 638
        and indices.min() >= 730,
        "INCORRECT_HOLDOUT_WINDOW",
    )

    returns = daily[
        "daily_log_return"
    ].to_numpy(dtype=float)

    rv = daily[
        "rv_5m"
    ].to_numpy(dtype=float)

    require(
        np.isfinite(returns).all(),
        "INCOMPLETE_DAILY_RETURNS",
    )

    require(
        np.isfinite(rv[indices]).all()
        and (rv[indices] > 0).all(),
        "INVALID_HOLDOUT_TARGETS",
    )

    valid_rv = rv[np.isfinite(rv)]

    require(
        (valid_rv > 0).all(),
        "INVALID_HISTORICAL_RV",
    )

    # Frozen causal functions: forecasts at t
    # depend exclusively on observations before t.
    baselines = {
        "D0_rolling_30": rolling_30(
            returns**2
        ),
        "D1_ewma_094": ewma(
            returns**2
        ),
        "I0_rolling_rv_30": rolling_30(
            rv
        ),
        "I1_ewma_rv_094": ewma(
            rv
        ),
    }

    log_rv, features = design(rv)

    warm = {
        "ARCH": None,
        "GARCH": None,
    }

    predictions = []
    parameter_trace = []

    for step, t in enumerate(indices, 1):
        date = calendar[t]

        row = {
            "date_utc": date,
            "rv_target": float(rv[t]),
        }

        for model, values in baselines.items():
            row[model] = float(values[t])

        for model in ("ARCH", "GARCH"):
            # Cold start on first holdout day;
            # warm start thereafter, as in validation.
            fitted = estimate(
                returns[t-730:t],
                model,
                warm=warm[model],
            )

            warm[model] = (
                fitted["theta"]
                if fitted["success"]
                else None
            )

            if not fitted["success"]:
                raise RuntimeError(
                    f"NONCONVERGENCE={date.date()}_{model}"
                )

            row[model] = float(
                fitted["forecast"]
            )

            parameter_trace.append({
                "date_utc": date,
                "model": model,
                "omega_pct_squared": fitted[
                    "omega_pct_squared"
                ],
                "alpha": fitted["alpha"],
                "beta": fitted["beta"],
                "persistence": fitted["persistence"],
                "optimizer_iterations": fitted[
                    "optimizer_iterations"
                ],
            })

        har = forecast_one(
            log_rv,
            features,
            int(t),
            window=730,
            min_train=500,
        )

        if not har["success"]:
            raise RuntimeError(
                f"HAR_FAILURE={date.date()}_"
                f"{har['reason']}"
            )

        row["HAR_RV"] = float(
            har["forecast"]
        )

        parameter_trace.append({
            "date_utc": date,
            "model": "HAR_RV",
            "n_train": har["n_train"],
            "smearing": har["smearing"],
            "condition_number": (
                har["condition_number"]
            ),
        })

        for name in MODELS:
            value = row[name]

            require(
                np.isfinite(value) and value > 0,
                f"INVALID_FORECAST={date.date()}_{name}",
            )

        predictions.append(row)

        if step % 100 == 0 or step == 638:
            print(
                f"HOLDOUT_FORECAST_PROGRESS={step}/638",
                flush=True,
            )

    frame = pd.DataFrame(predictions)

    require(
        len(frame) == 638
        and pd.DatetimeIndex(
            frame["date_utc"]
        ).equals(calendar[indices]),
        "FORECAST_DATE_MISMATCH",
    )

    require(
        frame[list(MODELS)].notna().all().all(),
        "MISSING_FORECAST",
    )

    return frame, pd.DataFrame(parameter_trace)


def inference(frame, spec):
    target = frame[
        "rv_target"
    ].to_numpy(dtype=float)

    losses = {
        model: qlike_losses(
            target,
            frame[model].to_numpy(dtype=float),
        )
        for model in MODELS
    }

    scores = {
        model: {
            "n": 638,
            "mean_qlike": float(
                values.mean()
            ),
        }
        for model, values in losses.items()
    }

    comparisons = {}
    p_values = {}

    bootstrap = spec[
        "supplementary_bootstrap"
    ]

    for name, contrast in (
        spec["primary_contrasts"].items()
    ):
        difference = (
            losses[contrast["benchmark"]]
            - losses[contrast["candidate"]]
        )

        statistic = hac_mean_test(
            difference,
            lags=spec["hac_lags"],
        )

        interval = moving_block_ci(
            difference,
            block=bootstrap["block_length_days"],
            replications=bootstrap["replications"],
            seed=bootstrap["random_seed"],
        )

        comparisons[name] = {
            "benchmark": contrast["benchmark"],
            "candidate": contrast["candidate"],
            "dm_hac": statistic,
            "bootstrap": interval,
        }

        p_values[name] = statistic[
            "p_value_two_sided"
        ]

    corrected = holm_adjust(
        p_values,
        alpha=spec["familywise_alpha"],
    )

    for name, decision in corrected.items():
        comparisons[name]["holm"] = decision

        comparisons[name][
            "candidate_superior_confirmatory"
        ] = bool(
            decision["reject_at_0_05"]
            and comparisons[name][
                "dm_hac"
            ]["mean_gain"] > 0
        )

    return scores, comparisons


def run(spec, manifest):
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # An exclusive marker prevents unacknowledged
    # repeat evaluations of the final holdout.
    marker = (
        OUTPUT / "first_holdout_opened.json"
    )

    with marker.open("x") as file:
        file.write(json.dumps({
            "protocol": spec["protocol"],
            "frozen_commit": FROZEN_COMMIT,
            "status": "first_holdout_opened",
        }, indent=2) + "\n")
        file.flush()
        os.fsync(file.fileno())

    daily = pd.read_parquet(
        SOURCE / "daily_volatility.parquet"
    )

    forecasts, trace = produce_forecasts(
        daily
    )

    require(
        len(forecasts) == spec[
            "required_common_sample"
        ],
        "COMMON_SAMPLE_FAILURE",
    )

    scores, comparisons = inference(
        forecasts, spec
    )

    predictions_path = (
        OUTPUT / "final_predictions.parquet"
    )

    trace_path = (
        OUTPUT / "final_parameter_trace.parquet"
    )

    forecasts.to_parquet(
        predictions_path,
        index=False,
        compression="zstd",
    )

    trace.to_parquet(
        trace_path,
        index=False,
        compression="zstd",
    )

    report = {
        "protocol": spec["protocol"],
        "stage": "final_holdout_first_look",
        "n": 638,
        "frozen_models_commit": FROZEN_COMMIT,
        "data_source_sha256": (
            manifest["output_sha256"][
                "daily_volatility.parquet"
            ]
        ),
        "scores": scores,
        "primary_contrasts": comparisons,
        "holdout_evaluated": True,
        "model_reoptimization": False,
        "output_sha256": {
            predictions_path.name: digest(
                predictions_path
            ),
            trace_path.name: digest(
                trace_path
            ),
        },
    }

    (OUTPUT / "final_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    print()
    print("=== FINAL HOLDOUT FIRST LOOK ===")

    for name in MODELS:
        value = scores[name]["mean_qlike"]

        print(
            f"{name}: N=638 "
            f"QLIKE={value:.8f}"
        )

    for name, result in comparisons.items():
        statistic = result["dm_hac"]
        holm = result["holm"]

        print(
            f"{name}: "
            f"GAIN={statistic['mean_gain']:.8f} "
            f"P_RAW={statistic['p_value_two_sided']:.6g} "
            f"P_HOLM={holm['p_holm']:.6g} "
            f"SUPERIOR="
            f"{result['candidate_superior_confirmatory']}"
        )

    print("COMMON_SAMPLE=638")
    print("MODEL_REOPTIMIZATION=NONE")
    print("FINAL_HOLDOUT=COMPLETE")
    print(f"OUTPUT={OUTPUT}")


def synthetic_test():
    rng = np.random.default_rng(1701)

    rv = np.exp(
        rng.normal(-6.7, 0.5, 1000)
    )

    returns = rng.normal(
        0, 0.03, 1000
    )

    t = 900

    def four_baselines(r, v):
        return np.array([
            rolling_30(r**2)[t],
            ewma(r**2)[t],
            rolling_30(v)[t],
            ewma(v)[t],
        ])

    before = four_baselines(
        returns, rv
    )

    changed_returns = returns.copy()
    changed_rv = rv.copy()

    changed_returns[t:] *= 100
    changed_rv[t:] *= 100

    after = four_baselines(
        changed_returns, changed_rv
    )

    np.testing.assert_array_equal(
        before, after
    )

    y1, x1 = design(rv)
    y2, x2 = design(changed_rv)

    h1 = forecast_one(y1, x1, t)
    h2 = forecast_one(y2, x2, t)

    require(
        h1["success"] and h2["success"],
        "SYNTHETIC_HAR_FAILURE",
    )

    np.testing.assert_allclose(
        h1["forecast"],
        h2["forecast"],
        rtol=0,
        atol=0,
    )

    for model in ("ARCH", "GARCH"):
        original = estimate(
            returns[t-730:t], model
        )

        modified = estimate(
            changed_returns[t-730:t], model
        )

        require(
            original["success"]
            and modified["success"],
            "SYNTHETIC_QMLE_FAILURE",
        )

        np.testing.assert_allclose(
            original["forecast"],
            modified["forecast"],
            rtol=0,
            atol=0,
        )

    differences = rng.normal(
        0.02, 1, 100
    )

    require(
        np.isfinite(
            hac_mean_test(
                differences
            )["p_value_two_sided"]
        ),
        "SYNTHETIC_HAC_FAILURE",
    )

    bootstrap = moving_block_ci(
        differences,
        block=14,
        replications=100,
        seed=11,
    )

    require(
        len(bootstrap["ci_percentile_95"]) == 2,
        "SYNTHETIC_BOOTSTRAP_FAILURE",
    )

    print("SYNTHETIC_ORCHESTRATION=PASS")
    print("ANTI_LEAKAGE=PASS")
    print("HOLDOUT_EVALUATED=NO")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "action",
        choices=(
            "self-test",
            "preflight",
            "run",
        ),
    )

    action = parser.parse_args().action

    if action == "self-test":
        synthetic_test()

    else:
        spec, manifest = preflight()

        if action == "preflight":
            print(
                "READY_FOR_FIRST_HOLDOUT=YES"
            )

        else:
            run(spec, manifest)
