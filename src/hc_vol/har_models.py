"""HC-BTCVOL-001: causal log-HAR-RV with smearing."""

import numpy as np
import pandas as pd

WINDOW = 730
MIN_TRAIN = 500
FEATURE_NAMES = (
    "intercept",
    "log_rv_d",
    "log_rv_w",
    "log_rv_m",
)


def design(realized_variance):
    rv = np.asarray(realized_variance, dtype=float)

    if rv.ndim != 1:
        raise ValueError("RV must be one-dimensional")

    finite = np.isfinite(rv)

    if np.isinf(rv).any() or (rv[finite] <= 0).any():
        raise ValueError("Invalid realized variance")

    log_rv = np.full(len(rv), np.nan)
    log_rv[finite] = np.log(rv[finite])

    s = pd.Series(log_rv)

    x = pd.DataFrame({
        "intercept": np.ones(len(s)),
        "log_rv_d": s.shift(1),
        "log_rv_w": (
            s.rolling(7, min_periods=7)
            .mean().shift(1)
        ),
        "log_rv_m": (
            s.rolling(30, min_periods=30)
            .mean().shift(1)
        ),
    })

    return log_rv, x.to_numpy(dtype=float)


def forecast_one(
    log_rv,
    features,
    t,
    window=WINDOW,
    min_train=MIN_TRAIN,
):
    """Forecast day t from information strictly before t."""

    if not window <= t < len(log_rv):
        raise ValueError("Invalid forecast index")

    if features.shape != (len(log_rv), 4):
        raise ValueError("Invalid design matrix")

    empty = {
        "success": False,
        "forecast": np.nan,
        "n_train": 0,
        "reason": None,
        "beta": None,
        "smearing": np.nan,
        "condition_number": np.nan,
        "residual_rms": np.nan,
    }

    current = features[t]

    if not np.isfinite(current).all():
        return {
            **empty,
            "reason": "invalid_features",
        }

    # Exactly 730 calendar target days.
    x = features[t-window:t]
    y = log_rv[t-window:t]

    usable = (
        np.isfinite(y)
        & np.isfinite(x).all(axis=1)
    )

    n = int(usable.sum())

    if n < min_train:
        return {
            **empty,
            "n_train": n,
            "reason": "insufficient_training",
        }

    x = x[usable]
    y = y[usable]

    beta, _, rank, singular = np.linalg.lstsq(
        x, y, rcond=None
    )

    if rank != 4 or singular[-1] <= 0:
        return {
            **empty,
            "n_train": n,
            "reason": "rank_deficient",
        }

    condition = float(
        singular[0] / singular[-1]
    )

    if not np.isfinite(condition) or condition > 1e8:
        return {
            **empty,
            "n_train": n,
            "reason": "ill_conditioned",
        }

    residual = y - x @ beta

    # In-sample Duan smearing factor.
    # No residual from day t or beyond enters the fit.
    smear = float(np.mean(np.exp(residual)))

    log_prediction = float(current @ beta)
    prediction = float(
        np.exp(log_prediction) * smear
    )

    if (
        not np.isfinite(prediction)
        or prediction <= 0
        or not np.isfinite(smear)
        or smear <= 0
    ):
        return {
            **empty,
            "n_train": n,
            "reason": "invalid_prediction",
        }

    return {
        "success": True,
        "forecast": prediction,
        "n_train": n,
        "reason": "ok",
        "beta": beta,
        "smearing": smear,
        "condition_number": condition,
        "residual_rms": float(
            np.sqrt(np.mean(residual**2))
        ),
    }
