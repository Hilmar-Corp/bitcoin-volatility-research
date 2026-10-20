"""HC-BTCVOL-001: independent second-solver QMLE audit.

POST-HOLDOUT DIAGNOSTIC ONLY.

Reestimates all 638 ARCH and 638 GARCH windows
using SLSQP, independent starting parameters,
and a normalized likelihood formulation.

Does not import or modify the frozen GARCH engine.
Does not replace the first-look forecasts.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / (
    "data/processed/quality_v1_1_through_2026-09"
)
RESULTS = ROOT / "results"
OUT = ROOT / (
    "data/processed/second_solver_qmle_v1"
)

OBJECTIVE_TOL = 1e-4
FORECAST_REL_TOL = 0.05
CAP = 0.999


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1 << 20), b""
        ):
            h.update(chunk)
    return h.hexdigest()


def likelihood(params, z, model):
    """Normalized Gaussian QMLE; h0 = sample variance."""

    omega, alpha = params[:2]
    beta = params[2] if model == "GARCH" else 0.0

    if (
        omega <= 0
        or alpha < 0
        or beta < 0
        or alpha + beta >= CAP
    ):
        return 1e12

    innovations = np.empty(len(z))
    innovations[0] = 1.0

    innovations[1:] = (
        omega + alpha * z[:-1]
    )

    h = lfilter(
        [1.0],
        [1.0, -beta],
        innovations,
    )

    if (
        not np.isfinite(h).all()
        or (h <= 0).any()
    ):
        return 1e12

    return float(
        0.5 * np.mean(
            np.log(h) + z / h
        )
    )


def scalar_recursion(params, z, model):
    """Independent scalar replay, without lfilter."""

    omega, alpha = params[:2]
    beta = params[2] if model == "GARCH" else 0.0

    h = 1.0
    values = [h]

    for i in range(1, len(z)):
        h = (
            omega
            + alpha * z[i-1]
            + beta * h
        )
        values.append(h)

    values = np.asarray(values)

    objective = float(
        0.5 * np.mean(
            np.log(values)
            + z / values
        )
    )

    next_variance = (
        omega
        + alpha * z[-1]
        + beta * h
    )

    return objective, float(next_variance)


def second_solver(z, model):
    """Independent multistart constrained SLSQP."""

    if model == "ARCH":
        starts = [
            [0.85, 0.15],
            [0.50, 0.40],
            [0.10, 0.80],
        ]

        bounds = [
            (1e-9, 20.0),
            (0.0, 0.999),
        ]

    elif model == "GARCH":
        starts = [
            [0.10, 0.08, 0.80],
            [0.25, 0.20, 0.55],
            [0.55, 0.30, 0.10],
        ]

        bounds = [
            (1e-9, 20.0),
            (0.0, 0.999),
            (0.0, 0.999),
        ]

    else:
        raise ValueError("Unknown model")

    constraints = ({
        "type": "ineq",
        "fun": lambda p: CAP - np.sum(p[1:]),
    },)

    candidates = []

    for start in starts:
        result = minimize(
            likelihood,
            x0=np.asarray(start, dtype=float),
            args=(z, model),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "maxiter": 500,
                "ftol": 1e-11,
            },
        )

        if (
            result.success
            and np.isfinite(result.fun)
            and result.fun < 1e11
        ):
            candidates.append(result)

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda result: result.fun,
    )


def self_test():
    rng = np.random.default_rng(20261020)

    returns = rng.normal(
        loc=0.0,
        scale=0.025,
        size=730,
    )

    squared = (100 * returns) ** 2
    z = squared / squared.mean()

    for model in ("ARCH", "GARCH"):
        fitted = second_solver(z, model)

        assert fitted is not None

        scalar_obj, forecast = scalar_recursion(
            fitted.x, z, model
        )

        np.testing.assert_allclose(
            scalar_obj,
            likelihood(fitted.x, z, model),
            rtol=0,
            atol=1e-11,
        )

        assert forecast > 0
        assert np.isfinite(forecast)
        assert sum(fitted.x[1:]) < CAP

        print(
            f"SYNTHETIC_{model}=PASS "
            f"ITERATIONS={fitted.nit}"
        )

    print("SECOND_SOLVER_SELF_TEST=PASS")


def main():
    audit = json.loads(
        (
            ROOT / "docs/final_results_audit_v1.json"
        ).read_text()
    )

    manifest = json.loads(
        (DATA / "manifest.json").read_text()
    )

    source = (
        DATA / "daily_volatility.parquet"
    )

    assert sha256(source) == manifest[
        "output_sha256"
    ]["daily_volatility.parquet"]

    assert sha256(
        RESULTS / "final_predictions.parquet"
    ) == audit["original_predictions_sha256"]

    assert sha256(
        RESULTS / "final_parameter_trace.parquet"
    ) == audit[
        "original_parameter_trace_sha256"
    ]

    daily = pd.read_parquet(source)

    pred = pd.read_parquet(
        RESULTS / "final_predictions.parquet"
    )

    trace = pd.read_parquet(
        RESULTS / "final_parameter_trace.parquet"
    )

    dates = pd.date_range(
        "2025-01-01",
        "2026-10-01",
        freq="D",
        inclusive="left",
        tz="UTC",
    )

    assert len(pred) == 638
    assert pd.DatetimeIndex(
        pred.date_utc
    ).equals(dates)

    assert len(daily) == 2465
    assert len(trace) == 1914

    assert not trace.duplicated(
        ["date_utc", "model"]
    ).any()

    trace = trace.set_index(
        ["date_utc", "model"]
    )

    daily_index = pd.DatetimeIndex(
        daily.date_utc
    )

    indices = daily_index.get_indexer(dates)

    assert (indices >= 730).all()
    assert len(indices) == 638

    returns = daily.daily_log_return.to_numpy(
        dtype=float
    )

    records = []

    print("=== INDEPENDENT FULL QMLE REFIT ===")
    print("ORIGINAL_OPTIMIZER=L-BFGS-B")
    print("SECOND_OPTIMIZER=SLSQP")
    print("INDEPENDENT_STARTS=3")
    print("FROZEN_FORECASTS_MODIFIED=NO")

    for step, (date, t) in enumerate(
        zip(dates, indices), 1
    ):
        history = returns[t-730:t]

        assert len(history) == 730
        assert np.isfinite(history).all()

        squared = (100 * history) ** 2
        scale = float(squared.mean())

        assert scale > 0
        z = squared / scale

        for model in ("ARCH", "GARCH"):
            row = trace.loc[(date, model)]

            original_params = np.array([
                float(row.omega_pct_squared)
                / scale,
                float(row.alpha),
            ])

            if model == "GARCH":
                original_params = np.r_[
                    original_params,
                    float(row.beta),
                ]

            original_obj, original_variance = (
                scalar_recursion(
                    original_params,
                    z,
                    model,
                )
            )

            original_forecast = (
                original_variance
                * scale / 10000
            )

            published = float(
                pred.loc[
                    step-1, model
                ]
            )

            np.testing.assert_allclose(
                original_forecast,
                published,
                rtol=1e-8,
                atol=1e-12,
            )

            np.testing.assert_allclose(
                original_obj,
                likelihood(
                    original_params, z, model
                ),
                rtol=0,
                atol=1e-10,
            )

            fitted = second_solver(z, model)

            if fitted is None:
                records.append({
                    "date_utc": date,
                    "model": model,
                    "second_solver_converged": False,
                    "original_objective": original_obj,
                    "independent_objective": np.nan,
                    "objective_gap": np.nan,
                    "forecast_rel_difference": np.nan,
                    "review_required": True,
                    "reason": "solver_nonconvergence",
                })
                continue

            new_obj, new_variance = scalar_recursion(
                fitted.x,
                z,
                model,
            )

            np.testing.assert_allclose(
                new_obj,
                fitted.fun,
                rtol=0,
                atol=1e-8,
            )

            new_forecast = (
                new_variance
                * scale / 10000
            )

            # Positive gap: second solver finds
            # a lower objective than original.
            gap = float(
                original_obj - new_obj
            )

            relative = float(
                abs(new_forecast - published)
                / published
            )

            reasons = []

            if abs(gap) > OBJECTIVE_TOL:
                reasons.append(
                    "objective_mismatch"
                )

            if relative > FORECAST_REL_TOL:
                reasons.append(
                    "forecast_instability"
                )

            records.append({
                "date_utc": date,
                "model": model,
                "second_solver_converged": True,
                "original_objective": original_obj,
                "independent_objective": new_obj,
                "objective_gap": gap,
                "forecast_rel_difference": relative,
                "independent_omega_pct_squared": (
                    float(fitted.x[0] * scale)
                ),
                "independent_alpha": float(
                    fitted.x[1]
                ),
                "independent_beta": float(
                    fitted.x[2]
                ) if model == "GARCH" else 0.0,
                "independent_optimizer_iterations": (
                    int(fitted.nit)
                ),
                "review_required": bool(reasons),
                "reason": (
                    ",".join(reasons)
                    if reasons else "ok"
                ),
            })

        if (
            step % 50 == 0
            or step == 638
        ):
            print(
                f"QMLE_REESTIMATION={step}/638_DATES",
                flush=True,
            )

    results = pd.DataFrame(records)

    assert len(results) == 1276

    problems = results[
        results.review_required
    ].copy()

    summary = {
        "protocol": "HC-BTCVOL-001",
        "classification": (
            "POST_HOLDOUT_NUMERICAL_AUDIT"
        ),
        "original_forecasts_sha256": audit[
            "original_predictions_sha256"
        ],
        "second_optimizer": "SLSQP",
        "optimization_library": (
            "SciPy " + scipy.__version__
        ),
        "independent_initializations": 3,
        "windows_per_model": 638,
        "attempts_total": 1276,
        "objective_gap_alert_threshold": (
            OBJECTIVE_TOL
        ),
        "forecast_relative_alert_threshold": (
            FORECAST_REL_TOL
        ),
        "converged": int(
            results.second_solver_converged.sum()
        ),
        "review_required": len(problems),
        "alerts": {
            model: int(
                problems.model.eq(model).sum()
            )
            for model in ("ARCH", "GARCH")
        },
        "original_models_modified": False,
        "confirmatory_inference_modified": False,
        "external_peer_review": False,
        "status": (
            "PASS"
            if len(problems) == 0
            else "REVIEW_REQUIRED"
        ),
    }

    OUT.mkdir(
        parents=True,
        exist_ok=False,
    )

    results.to_parquet(
        OUT / "independent_qmle_refits.parquet",
        index=False,
    )

    summary["refits_sha256"] = sha256(
        OUT / "independent_qmle_refits.parquet"
    )

    (OUT / "audit_report.json").write_text(
        json.dumps(
            summary,
            indent=2,
            allow_nan=False,
        ) + "\n"
    )

    print()
    print("=== SECOND SOLVER QMLE AUDIT ===")
    print("TOTAL_FITS=1276")
    print(
        "SECOND_SOLVER_CONVERGED="
        f"{summary['converged']}/1276"
    )

    for model in ("ARCH", "GARCH"):
        subset = results.loc[
            results.model == model
        ]

        print(
            f"{model}_ALERTS="
            f"{summary['alerts'][model]}"
        )

        finite_gaps = (
            subset.objective_gap.dropna()
        )

        finite_rel = (
            subset.forecast_rel_difference.dropna()
        )

        if len(finite_gaps):
            print(
                f"{model}_MAX_ABS_OBJECTIVE_GAP="
                f"{finite_gaps.abs().max():.10f}"
            )

        if len(finite_rel):
            print(
                f"{model}_MAX_FORECAST_REL_DIFF="
                f"{finite_rel.max():.8f}"
            )

    if len(problems):
        print("=== FIRST REVIEW ITEMS ===")
        print(
            problems[
                [
                    "date_utc",
                    "model",
                    "objective_gap",
                    "forecast_rel_difference",
                    "reason",
                ]
            ].head(20).to_string(index=False)
        )

        print(
            "SECOND_SOLVER_QMLE=REVIEW_REQUIRED"
        )
        raise SystemExit(2)

    print("INDEPENDENT_ARCH_QMLE=PASS")
    print("INDEPENDENT_GARCH_QMLE=PASS")
    print("FROZEN_RESULTS_UNCHANGED=YES")
    print("SECOND_SOLVER_QMLE=PASS")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
