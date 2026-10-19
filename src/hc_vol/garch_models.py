"""Causal Gaussian QMLE for ARCH(1) and GARCH(1,1)."""

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import expit, logsumexp

PERSISTENCE_CAP = 0.999
WINDOW = 730


def parameters(theta, model):
    omega = float(np.exp(theta[0]))

    if model == "ARCH":
        alpha = float(
            PERSISTENCE_CAP * expit(theta[1])
        )
        return omega, alpha, 0.0

    if model != "GARCH":
        raise ValueError("Unknown volatility model")

    normalizer = logsumexp(
        [0.0, theta[1], theta[2]]
    )
    alpha = float(
        PERSISTENCE_CAP
        * np.exp(theta[1] - normalizer)
    )
    beta = float(
        PERSISTENCE_CAP
        * np.exp(theta[2] - normalizer)
    )
    return omega, alpha, beta


def initial_guess(variance, model, alpha, beta=0.0):
    omega = max(
        variance * (1.0-alpha-beta), 1e-10
    )

    if model == "ARCH":
        return np.array([
            np.log(omega),
            np.log(alpha / (PERSISTENCE_CAP-alpha))
        ])

    rest = PERSISTENCE_CAP-alpha-beta
    return np.array([
        np.log(omega),
        np.log(alpha/rest),
        np.log(beta/rest),
    ])


def conditional_path(x2, omega, alpha, beta):
    """h[0]=mean(x2); h[t]=omega+alpha*x2[t-1]+beta*h[t-1]."""
    u = np.empty(len(x2), dtype=float)
    u[0] = x2.mean()
    u[1:] = omega + alpha*x2[:-1]

    return lfilter(
        [1.0], [1.0, -beta], u
    )


def estimate(history, model, warm=None):
    """One-day forecast using exactly 730 past daily returns."""
    x = np.asarray(history, dtype=float)

    if (
        x.ndim != 1
        or len(x) != WINDOW
        or not np.isfinite(x).all()
    ):
        raise ValueError("Invalid 730-day estimation window")

    # Convert decimal daily returns into percentage units.
    x2 = (100.0*x)**2
    variance = float(x2.mean())

    if variance <= 1e-12:
        raise ValueError("Degenerate estimation sample")

    def objective(theta):
        omega, alpha, beta = parameters(
            theta, model
        )
        h = conditional_path(
            x2, omega, alpha, beta
        )

        if (
            not np.isfinite(h).all()
            or (h <= 1e-12).any()
        ):
            return 1e20

        # Gaussian QMLE, omitting constant log(2*pi).
        return float(
            0.5*np.mean(np.log(h) + x2/h)
        )

    if model == "ARCH":
        starts = [
            initial_guess(variance, model, 0.05),
            initial_guess(variance, model, 0.25),
        ]
        bounds = [(-18, 10), (-15, 15)]
    elif model == "GARCH":
        starts = [
            initial_guess(variance, model, 0.07, 0.87),
            initial_guess(variance, model, 0.17, 0.55),
        ]
        bounds = [
            (-18, 10), (-15, 15), (-15, 15)
        ]
    else:
        raise ValueError("Unknown model")

    if warm is not None:
        starts.insert(
            0, np.asarray(warm, dtype=float)
        )

    candidates = []

    for start in starts:
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": 400,
                "ftol": 1e-11,
            },
        )

        if (
            result.success
            and np.isfinite(result.fun)
        ):
            candidates.append(result)

    if not candidates:
        return {
            "success": False,
            "forecast": np.nan,
            "theta": None,
        }

    best = min(
        candidates, key=lambda result: result.fun
    )

    omega, alpha, beta = parameters(
        best.x, model
    )

    h = conditional_path(
        x2, omega, alpha, beta
    )

    forecast = (
        omega
        + alpha*x2[-1]
        + beta*h[-1]
    ) / 10000.0

    valid = (
        np.isfinite(forecast)
        and forecast > 0
        and alpha >= 0
        and beta >= 0
        and alpha+beta < 1
    )

    return {
        "success": bool(valid),
        "forecast": (
            float(forecast) if valid else np.nan
        ),
        "omega_pct_squared": omega,
        "alpha": alpha,
        "beta": beta,
        "persistence": alpha+beta,
        "optimizer_iterations": int(best.nit),
        "objective": float(best.fun),
        "theta": best.x.copy() if valid else None,
    }
