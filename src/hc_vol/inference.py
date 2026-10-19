"""Frozen confirmatory inference for HC-BTCVOL-001."""

from __future__ import annotations

import math
import numpy as np


def qlike_losses(target, forecast):
    y = np.asarray(target, dtype=float)
    p = np.asarray(forecast, dtype=float)

    if (
        y.ndim != 1
        or p.ndim != 1
        or y.shape != p.shape
        or len(y) == 0
    ):
        raise ValueError("Expected aligned nonempty 1D arrays")

    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Non-finite target or forecast")

    if (y < 0).any() or (p <= 0).any():
        raise ValueError("Invalid variance")

    return np.log(p) + y / p


def hac_mean_test(differences, lags=7):
    """Two-sided DM-style test, Bartlett HAC variance."""

    x = np.asarray(differences, dtype=float)

    if not isinstance(lags, int) or lags < 0:
        raise ValueError("Invalid lag")

    if (
        x.ndim != 1
        or not np.isfinite(x).all()
        or len(x) <= lags + 1
    ):
        raise ValueError("Invalid loss differences")

    n = len(x)
    mean = float(np.mean(x))
    centered = x - mean

    long_run = float(
        np.dot(centered, centered) / n
    )

    for lag in range(1, lags + 1):
        covariance = float(
            np.dot(
                centered[lag:],
                centered[:-lag]
            ) / n
        )

        long_run += (
            2 * (1 - lag / (lags + 1))
            * covariance
        )

    if not np.isfinite(long_run) or long_run <= 0:
        raise ValueError("Degenerate long-run variance")

    standard_error = math.sqrt(long_run / n)
    z = mean / standard_error

    p_value = math.erfc(
        abs(z) / math.sqrt(2)
    )

    return {
        "n": n,
        "mean_gain": mean,
        "hac_lags": lags,
        "hac_standard_error": standard_error,
        "z": z,
        "p_value_two_sided": p_value,
    }


def holm_adjust(p_values, alpha=0.05):
    """Holm correction for exactly two primary contrasts."""

    required = {
        "GARCH_vs_EWMA",
        "HAR_RV_vs_EWMA_RV",
    }

    if set(p_values) != required:
        raise ValueError("Invalid hypothesis family")

    if not 0 < alpha < 1:
        raise ValueError("Invalid alpha")

    ordered = sorted(
        p_values,
        key=lambda name: (p_values[name], name)
    )

    adjusted = {}
    running = 0.0
    m = len(ordered)

    for i, name in enumerate(ordered):
        p = float(p_values[name])

        if not np.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("Invalid p-value")

        running = max(
            running,
            min(1.0, p * (m - i))
        )

        adjusted[name] = running

    return {
        name: {
            "p_holm": adjusted[name],
            "reject_at_0_05": bool(
                adjusted[name] <= alpha
            ),
        }
        for name in p_values
    }


def moving_block_ci(
    differences,
    block=14,
    replications=10000,
    seed=20261019,
):
    """Supplementary moving-block percentile interval."""

    d = np.asarray(differences, dtype=float)

    if (
        d.ndim != 1
        or not np.isfinite(d).all()
        or len(d) < 2 * block
    ):
        raise ValueError("Invalid bootstrap sample")

    if (
        block < 1
        or replications < 100
        or not isinstance(seed, int)
    ):
        raise ValueError("Invalid bootstrap settings")

    n = len(d)
    blocks_needed = math.ceil(n / block)
    rng = np.random.default_rng(seed)
    offsets = np.arange(block)

    means = np.empty(replications)

    for i in range(replications):
        starts = rng.integers(
            0,
            n - block + 1,
            size=blocks_needed
        )

        positions = (
            starts[:, None] + offsets
        ).reshape(-1)[:n]

        means[i] = float(
            np.mean(d[positions])
        )

    low, high = np.quantile(
        means, [0.025, 0.975]
    )

    return {
        "block_length": block,
        "replications": replications,
        "seed": seed,
        "ci_percentile_95": [
            float(low),
            float(high),
        ],
    }
