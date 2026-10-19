import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

from hc_vol.baselines import rolling_30, ewma, qlike


def test_rolling_strictly_lagged():
    x = np.arange(1, 51, dtype=float)
    forecast = rolling_30(x)

    assert np.isnan(forecast[:30]).all()

    assert forecast[30] == pytest.approx(
        np.mean(x[:30])
    )

    assert forecast[31] == pytest.approx(
        np.mean(x[1:31])
    )


def test_ewma_seed_and_update():
    x = np.arange(1, 51, dtype=float) / 10000
    forecast = ewma(x)

    assert np.isnan(forecast[:30]).all()

    assert forecast[30] == pytest.approx(
        np.mean(x[:30])
    )

    assert forecast[31] == pytest.approx(
        0.94 * forecast[30] + 0.06 * x[30]
    )


def test_missing_observation_forces_restart():
    x = np.repeat(0.01, 90)
    x[40] = np.nan

    for model in (rolling_30, ewma):
        result = model(x)

        assert np.isfinite(result[40])
        assert np.isnan(result[41:71]).all()
        assert np.isfinite(result[71])


@pytest.mark.parametrize(
    "forecaster",
    [rolling_30, ewma],
)
def test_no_future_information_leakage(forecaster):
    x = np.arange(1, 90, dtype=float) / 1000

    baseline = forecaster(x)

    modified = x.copy()
    modified[45:] *= 1000

    alternative = forecaster(modified)

    np.testing.assert_allclose(
        baseline[:46],
        alternative[:46],
        equal_nan=True,
    )

    assert baseline[46] != pytest.approx(
        alternative[46]
    )


def test_qlike_computation():
    y = np.array([2.0, 4.0, np.nan])
    prediction = np.array([2.0, 2.0, 3.0])

    result = qlike(y, prediction)

    assert result["n"] == 2

    assert result["mean_qlike"] == pytest.approx(
        np.log(2) + 1.5
    )

    with pytest.raises(ValueError):
        qlike(
            np.array([2.0]),
            np.array([0.0]),
        )
