import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

from hc_vol.har_models import design, forecast_one


def fixture():
    rng = np.random.default_rng(2026)
    return np.exp(
        rng.normal(-7.0, 0.55, 950)
    )


def test_feature_lags():
    rv = fixture()
    y, x = design(rv)

    t = 100

    np.testing.assert_allclose(
        x[t],
        [
            1.0,
            y[t-1],
            y[t-7:t].mean(),
            y[t-30:t].mean(),
        ],
    )


def test_har_prediction_positive():
    y, x = design(fixture())
    result = forecast_one(y, x, 800)

    assert result["success"]
    assert result["forecast"] > 0
    assert result["n_train"] == 730
    assert result["smearing"] > 0


def test_future_is_inaccessible():
    rv = fixture()

    y1, x1 = design(rv)
    original = forecast_one(y1, x1, 800)

    modified = rv.copy()
    modified[800:] *= 100

    y2, x2 = design(modified)
    alternative = forecast_one(y2, x2, 800)

    assert original["success"]
    assert alternative["success"]

    assert original["forecast"] == pytest.approx(
        alternative["forecast"], rel=1e-12
    )


def test_missing_rv_propagation():
    rv = fixture()

    # Missing observation immediately before
    # the forecast date.
    rv[799] = np.nan

    y, x = design(rv)

    # A missing RV contaminates the next 30
    # lagged feature vectors, never future data.
    assert np.isnan(
        x[800:830, 1:]
    ).any(axis=1).all()

    assert np.isfinite(x[830]).all()

    # Valid forecast index under the frozen
    # 730-calendar-day estimation protocol.
    result = forecast_one(y, x, 800)

    assert not result["success"]
    assert result["reason"] == "invalid_features"

def test_smearing_retransformation():
    y, x = design(fixture())
    t = 800

    result = forecast_one(y, x, t)

    train_x = x[t-730:t]
    train_y = y[t-730:t]

    beta = np.linalg.lstsq(
        train_x, train_y, rcond=None
    )[0]

    residual = train_y - train_x @ beta

    expected = (
        np.exp(x[t] @ beta)
        * np.mean(np.exp(residual))
    )

    assert result["forecast"] == pytest.approx(
        expected, rel=1e-12
    )


def test_invalid_rv_rejected():
    rv = fixture()
    rv[20] = 0

    with pytest.raises(ValueError):
        design(rv)
