import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

from hc_vol.quality_data import form_dataset


START = pd.Timestamp("2020-01-01T00:00:00Z")
END = pd.Timestamp("2020-01-03T00:00:00Z")


def sample_data():
    timestamps = pd.date_range(
        START, END, freq="5min", inclusive="left"
    )
    prices = 100 * np.exp(
        np.arange(
            1, len(timestamps) + 1
        ) * 0.0001
    )
    return pd.DataFrame({
        "close_usdt": prices,
        "complete_candle": True,
    }, index=timestamps)


def test_complete_realized_variance():
    bars, daily = form_dataset(
        sample_data(), 100.0, START, END
    )

    assert len(bars) == 576
    assert len(daily) == 2
    assert daily.rv_valid.tolist() == [True, True]

    np.testing.assert_allclose(
        daily.rv_5m.to_numpy(),
        288 * 1e-8,
        atol=1e-13,
    )

    np.testing.assert_allclose(
        daily.daily_log_return.to_numpy(),
        288 * 0.0001,
        atol=1e-11,
    )


def test_gap_invalidates_rv_not_daily_return():
    missing = pd.Timestamp(
        "2020-01-01T09:00:00Z"
    )
    df = sample_data().drop(missing)

    bars, daily = form_dataset(
        df, 100.0, START, END
    )

    assert daily.rv_valid.tolist() == [
        False, True
    ]

    assert np.isnan(daily.rv_5m.iloc[0])
    assert np.isfinite(
        daily.daily_log_return.iloc[0]
    )

    indexed = bars.set_index("time_utc")

    assert np.isnan(
        indexed.loc[
            "2020-01-01T09:00:00Z",
            "log_return_5m",
        ]
    )

    assert np.isnan(
        indexed.loc[
            "2020-01-01T09:05:00Z",
            "log_return_5m",
        ]
    )

    assert np.isfinite(
        indexed.loc[
            "2020-01-01T09:10:00Z",
            "log_return_5m",
        ]
    )


def test_short_candle_invalidates_rv():
    df = sample_data()
    timestamp = pd.Timestamp(
        "2020-01-01T11:35:00Z"
    )

    df.loc[
        timestamp, "complete_candle"
    ] = False

    bars, daily = form_dataset(
        df, 100.0, START, END
    )

    assert daily.rv_valid.tolist() == [
        False, True
    ]

    assert np.isfinite(
        daily.daily_log_return.iloc[0]
    )

    indexed = bars.set_index("time_utc")

    assert np.isnan(
        indexed.loc[
            timestamp, "log_return_5m"
        ]
    )

    assert np.isnan(
        indexed.loc[
            timestamp + pd.offsets.Minute(5),
            "log_return_5m",
        ]
    )


def test_missing_daily_close_propagates():
    df = sample_data().drop(
        pd.Timestamp(
            "2020-01-01T23:55:00Z"
        )
    )

    _, daily = form_dataset(
        df, 100.0, START, END
    )

    assert not daily.rv_valid.iloc[0]
    assert not daily.rv_valid.iloc[1]
    assert not daily.daily_return_valid.iloc[0]
    assert not daily.daily_return_valid.iloc[1]


def build_negative_duration_fixture(tmp_path, bad_row):
    """Synthetic December 2020 archive."""
    import zipfile

    from hc_vol.quality_data import parse_archive

    first = pd.Timestamp("2020-12-01T00:00:00Z")
    first_ms = first.value // 1_000_000
    lines = []

    for i in range(31 * 288):
        opened = first_ms + i * 300000
        closed = (
            opened - 1059479
            if i == bad_row
            else opened + 299999
        )
        lines.append(
            f"{opened},100,101,99,100,1,"
            f"{closed},100,1,0.5,50,0"
        )

    archive = tmp_path / "synthetic.zip"

    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(
            "BTCUSDT-5m-2020-12.csv",
            "\n".join(lines) + "\n",
        )

    return parse_archive(archive, "2020-12")


def test_known_negative_duration_quarantined(tmp_path):
    # 21 December 2020 at 14:05 UTC
    observed, metrics = build_negative_duration_fixture(
        tmp_path, bad_row=5929
    )

    assert metrics["negative_duration"] == 1
    assert metrics["short_duration"] == 0
    assert metrics["bad_duration"] == 1

    timestamp = pd.Timestamp(
        "2020-12-21T14:05:00Z"
    )
    assert not bool(
        observed.loc[timestamp, "complete_candle"]
    )


def test_unknown_negative_duration_rejected(tmp_path):
    import pytest

    with pytest.raises(
        ValueError,
        match="Unexpected negative close_time",
    ):
        build_negative_duration_fixture(
            tmp_path, bad_row=5930
        )
