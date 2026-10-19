"""HC-BTCVOL-001: causal variance forecasting benchmarks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW = 30
LAMBDA = 0.94

VAL_START = pd.Timestamp("2023-01-01T00:00:00Z")
VAL_END = pd.Timestamp("2025-01-01T00:00:00Z")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            h.update(block)
    return h.hexdigest()


def rolling_30(
    x: np.ndarray,
    window: int = WINDOW,
) -> np.ndarray:
    """Mean of the preceding 30 observations, never current."""

    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)

    for t in range(window, len(x)):
        previous = x[t-window:t]

        if (
            np.isfinite(previous).all()
            and (previous >= 0).all()
        ):
            out[t] = previous.mean()

    return out


def ewma(
    x: np.ndarray,
    decay: float = LAMBDA,
    seed_days: int = WINDOW,
) -> np.ndarray:
    """Causal EWMA, restarted after missing observations.

    Initialization at day t uses x[t-30:t].
    Updates use the previous day's observation only.
    """

    if not 0 < decay < 1 or seed_days < 1:
        raise ValueError("Invalid EWMA parameters")

    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)

    state = np.nan
    consecutive = 0

    for t in range(1, len(x)):
        previous = x[t-1]

        if (
            not np.isfinite(previous)
            or previous < 0
        ):
            consecutive = 0
            state = np.nan

        else:
            consecutive += 1

            if consecutive == seed_days:
                state = x[t-seed_days:t].mean()

            elif consecutive > seed_days:
                state = (
                    decay * state
                    + (1-decay) * previous
                )

        if consecutive >= seed_days:
            out[t] = state

    return out


def qlike(
    y: np.ndarray,
    prediction: np.ndarray,
) -> dict:
    """Mean QLIKE with no artificial missing-data filling."""

    y = np.asarray(y, dtype=float)
    prediction = np.asarray(
        prediction, dtype=float
    )

    if y.shape != prediction.shape:
        raise ValueError("Incompatible array shapes")

    valid = (
        np.isfinite(y)
        & np.isfinite(prediction)
    )

    if not valid.any():
        raise ValueError("No valid evaluation rows")

    if (
        (prediction[valid] <= 0).any()
        or (y[valid] < 0).any()
    ):
        raise ValueError("Invalid variance values")

    loss = (
        np.log(prediction[valid])
        + y[valid] / prediction[valid]
    )

    return {
        "n": int(valid.sum()),
        "mean_qlike": float(loss.mean()),
    }


def run(root: Path):
    dataset = (
        root / "data/processed/"
        "quality_v1_1_through_2026-09"
    )

    protocol = json.loads(
        (root / "configs/protocol_v1.json").read_text()
    )

    assert protocol["protocol_id"] == "HC-BTCVOL-001"
    assert protocol["ewma_lambda"] == LAMBDA
    assert protocol["validation_end"] == "2024-12-31"

    manifest = json.loads(
        (dataset / "manifest.json").read_text()
    )

    if manifest["status"] != (
        "PASS_WITH_DOCUMENTED_EXCLUSIONS"
    ):
        raise ValueError("Dataset is not validated")

    source = dataset / "daily_volatility.parquet"

    if sha256(source) != (
        manifest["output_sha256"][
            "daily_volatility.parquet"
        ]
    ):
        raise ValueError("Dataset fingerprint mismatch")

    daily = pd.read_parquet(source)

    expected_dates = pd.date_range(
        "2020-01-01",
        "2026-10-01",
        freq="D",
        inclusive="left",
        tz="UTC",
    )

    if not pd.DatetimeIndex(
        daily["date_utc"]
    ).equals(expected_dates):
        raise ValueError("Daily calendar mismatch")

    if (
        daily["rv_valid"].sum() != 2449
        or daily["daily_return_valid"].sum() != 2465
    ):
        raise ValueError("Dataset validity drift")

    # The holdout period is removed before forecasting.
    trainval = daily.loc[
        daily["date_utc"] < VAL_END
    ].copy()

    returns = trainval[
        "daily_log_return"
    ].to_numpy(dtype=float)

    realized = trainval[
        "rv_5m"
    ].to_numpy(dtype=float)

    if not np.isfinite(returns).all():
        raise ValueError("Invalid daily returns")

    valid_rv = realized[np.isfinite(realized)]

    if (valid_rv <= 0).any():
        raise ValueError("Nonpositive realized variance")

    daily_squared = returns ** 2

    forecasts = {
        "D0_rolling_30": rolling_30(daily_squared),
        "D1_ewma_094": ewma(daily_squared),
        "I0_rolling_rv_30": rolling_30(realized),
        "I1_ewma_rv_094": ewma(realized),
    }

    results = pd.DataFrame({
        "date_utc": trainval["date_utc"],
        "rv_target": realized,
    })

    for name, values in forecasts.items():
        if len(values) != len(results):
            raise ValueError("Forecast length mismatch")

        results[name] = values

    validation = results.loc[
        results["date_utc"] >= VAL_START
    ].copy()

    if (
        len(validation) != 731
        or validation["date_utc"].max() >= VAL_END
    ):
        raise ValueError("Validation window mismatch")

    scores = {
        name: qlike(
            validation["rv_target"].to_numpy(),
            validation[name].to_numpy(),
        )
        for name in forecasts
    }

    if not all(
        scores[name]["n"] == 730
        for name in (
            "D0_rolling_30",
            "D1_ewma_094",
        )
    ):
        raise ValueError("Daily evaluation count mismatch")

    if not all(
        scores[name]["n"] == 700
        for name in (
            "I0_rolling_rv_30",
            "I1_ewma_rv_094",
        )
    ):
        raise ValueError("RV evaluation count mismatch")

    output = (
        root / "data/processed/"
        "baselines_validation_v1"
    )
    output.mkdir(parents=True, exist_ok=True)

    predictions = output / "validation_predictions.parquet"

    validation.to_parquet(
        predictions,
        compression="zstd",
        index=False,
    )

    report = {
        "protocol": "HC-BTCVOL-001",
        "phase": "validation_only",
        "input_sha256": sha256(source),
        "predictions_sha256": sha256(predictions),
        "validation_start": "2023-01-01",
        "validation_end": "2024-12-31",
        "initialization": (
            "30 consecutive valid observations; "
            "restart after a missing observation"
        ),
        "rolling_daily_definition": (
            "mean squared daily log return, "
            "conditional mean fixed at zero"
        ),
        "ewma_lambda": LAMBDA,
        "scores": scores,
    }

    (output / "validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    print("=== BASELINE VALIDATION ===")
    print("HOLDOUT_EVALUATED=NO")
    print("VALIDATION_DATES=731")

    for name, result in scores.items():
        print(
            f"{name}: "
            f"N={result['n']} "
            f"QLIKE={result['mean_qlike']:.8f}"
        )

    print("BASELINES_VALIDATION=PASS")
    print(f"OUTPUT={output}")


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[2])
