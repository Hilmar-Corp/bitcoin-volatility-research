import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

from hc_vol.data import START, aggregate, parse_month


def mock_month(path, month):
    unit = "us" if month >= "2025-01" else "ms"
    width = 300_000_000 if unit == "us" else 300_000

    first = pd.Timestamp(month + "-01", tz="UTC")
    last = first + pd.offsets.MonthBegin(1)

    t = pd.date_range(
        first, last, freq="5min", inclusive="left"
    )

    with zipfile.ZipFile(path, "w") as z:
        rows = []

        for stamp in t:
            open_time = stamp.value // (
                1000 if unit == "us" else 1000000
            )
            rows.append(
                f"{open_time},100,101,99,100,1,"
                f"{open_time+width-1},100,3,0.5,50,0"
            )

        z.writestr("candles.csv", "\n".join(rows))


@pytest.mark.parametrize(
    "month", ["2020-01", "2025-01"]
)
def test_timestamp_units_and_grid(month, tmp_path):
    path = tmp_path / "candles.zip"
    mock_month(path, month)

    df = parse_month(path, month)

    assert len(df) == 31 * 288
    assert df.time_utc.iloc[0] == pd.Timestamp(
        month + "-01", tz="UTC"
    )


def test_daily_rv():
    t = pd.date_range(
        pd.Timestamp("2019-12-31T23:55:00Z"),
        periods=31 * 288 + 1,
        freq="5min",
        tz="UTC"
    )

    prices = 100 * np.exp(
        np.arange(len(t)) * 0.0001
    )

    bars = pd.DataFrame({
        "time_utc": t,
        "close_usdt": prices
    })

    intraday, daily = aggregate(
        bars, "2020-01"
    )

    assert len(intraday) == 31 * 288
    assert len(daily) == 31

    np.testing.assert_allclose(
        daily.rv_5m.to_numpy(),
        288 * 1e-8,
        atol=1e-13
    )

    np.testing.assert_allclose(
        daily.daily_log_return.to_numpy(),
        288 * 0.0001,
        atol=1e-11
    )


def test_gap_rejected():
    t = pd.date_range(
        pd.Timestamp("2019-12-31T23:55:00Z"),
        periods=31 * 288 + 1,
        freq="5min",
        tz="UTC"
    )

    bars = pd.DataFrame({
        "time_utc": t,
        "close_usdt": np.ones(len(t)) * 100
    })

    with pytest.raises(
        ValueError, match="Combined history"
    ):
        aggregate(
            bars.drop(index=100),
            "2020-01"
        )
