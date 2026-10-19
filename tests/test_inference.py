import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

from hc_vol.inference import (
    qlike_losses,
    hac_mean_test,
    holm_adjust,
    moving_block_ci,
)


def test_qlike_definition():
    actual = qlike_losses(
        [2.0, 4.0],
        [2.0, 2.0]
    )

    np.testing.assert_allclose(
        actual,
        [np.log(2) + 1, np.log(2) + 2]
    )


def test_qlike_rejects_invalid_forecast():
    with pytest.raises(ValueError):
        qlike_losses([2.0], [0.0])


def test_qlike_rejects_missing():
    with pytest.raises(ValueError):
        qlike_losses([np.nan], [1.0])


def test_hac_lag_zero():
    d = np.arange(1.0, 10.0)

    result = hac_mean_test(d, lags=0)

    variance = np.mean(
        (d - d.mean()) ** 2
    )

    expected_se = np.sqrt(
        variance / len(d)
    )

    assert result["mean_gain"] == pytest.approx(5.0)
    assert result["hac_standard_error"] == pytest.approx(
        expected_se
    )


def test_hac_rejects_degenerate_series():
    with pytest.raises(ValueError):
        hac_mean_test(
            np.ones(20), lags=7
        )


def test_holm_step_down():
    result = holm_adjust({
        "GARCH_vs_EWMA": 0.01,
        "HAR_RV_vs_EWMA_RV": 0.04,
    })

    assert result["GARCH_vs_EWMA"]["p_holm"] == pytest.approx(
        0.02
    )
    assert result["HAR_RV_vs_EWMA_RV"]["p_holm"] == pytest.approx(
        0.04
    )
    assert all(
        x["reject_at_0_05"]
        for x in result.values()
    )


def test_holm_rejects_incomplete_family():
    with pytest.raises(ValueError):
        holm_adjust({
            "GARCH_vs_EWMA": 0.01
        })


def test_moving_block_bootstrap_reproducible():
    d = np.arange(1.0, 101.0)

    a = moving_block_ci(
        d, block=14, replications=200, seed=11
    )
    b = moving_block_ci(
        d, block=14, replications=200, seed=11
    )

    assert a == b
    assert a["ci_percentile_95"][0] <= (
        a["ci_percentile_95"][1]
    )
