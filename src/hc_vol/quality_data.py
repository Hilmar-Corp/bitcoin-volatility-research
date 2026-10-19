"""HC-BTCVOL-001: quality-aware realized-variance dataset."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

START = pd.Timestamp("2020-01-01T00:00:00Z")
END = pd.Timestamp("2026-10-01T00:00:00Z")
ANCHOR = pd.Timestamp("2019-12-31T23:55:00Z")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            h.update(block)
    return h.hexdigest()


def parse_archive(path: Path, month: str):
    with zipfile.ZipFile(path) as z:
        members = [
            n for n in z.namelist()
            if n.endswith(".csv")
            and not n.startswith("__MACOSX/")
        ]
        if len(members) != 1:
            raise ValueError(f"Invalid ZIP structure: {month}")

        with z.open(members[0]) as source:
            data = pd.read_csv(
                source, header=None, low_memory=False
            )

    if data.shape[1] != 12:
        raise ValueError(f"Invalid schema: {month}")

    unit = "us" if month >= "2025-01" else "ms"
    width = 300_000_000 if unit == "us" else 300_000

    opened = pd.to_numeric(
        data[0], errors="raise"
    ).astype("int64")

    closed = pd.to_numeric(
        data[6], errors="raise"
    ).astype("int64")

    duration = closed - opened
    timestamps = pd.DatetimeIndex(
        pd.to_datetime(opened, unit=unit, utc=True)
    )

    first = pd.Timestamp(f"{month}-01T00:00:00Z")
    end = first + pd.offsets.MonthBegin(1)

    grid = pd.date_range(
        first, end, freq="5min", inclusive="left"
    )

    negative_positions = np.flatnonzero(
        (duration < 0).to_numpy()
    )

    if len(negative_positions):
        pos = int(negative_positions[0])
        known_incident = (
            month == "2020-12"
            and len(negative_positions) == 1
            and timestamps[pos]
            == pd.Timestamp("2020-12-21T14:05:00Z")
            and int(duration.iloc[pos]) == -1059479
        )

        if not known_incident:
            raise ValueError(
                f"Unexpected negative close_time: {month}"
            )

    if (
        timestamps.has_duplicates
        or not timestamps.is_monotonic_increasing
        or not timestamps.isin(grid).all()
        or (opened % width != 0).any()
        or (duration > width - 1).any()
    ):
        raise ValueError(
            f"Unexpected timestamp anomaly: {month}"
        )

    numbers = data[[1, 2, 3, 4, 5, 8]].apply(
        pd.to_numeric, errors="raise"
    )
    prices = numbers[[1, 2, 3, 4]]

    invalid = (
        not np.isfinite(numbers.to_numpy()).all()
        or (prices <= 0).any().any()
        or (numbers[5] < 0).any()
        or (numbers[8] < 0).any()
        or (
            numbers[3]
            > prices[[1, 2, 4]].min(axis=1)
        ).any()
        or (
            numbers[2]
            < prices[[1, 3, 4]].max(axis=1)
        ).any()
    )

    if invalid:
        raise ValueError(
            f"Unexpected OHLCV anomaly: {month}"
        )

    observations = pd.DataFrame(
        {
            "close_usdt": numbers[4].to_numpy(),
            "complete_candle": (
                duration == width - 1
            ).to_numpy(),
        },
        index=timestamps,
    )

    metrics = {
        "month": month,
        "expected": len(grid),
        "observed": len(observations),
        "missing": len(grid) - len(observations),
        "bad_duration": int(
            (duration != width - 1).sum()
        ),
        "short_duration": int(
            ((duration >= 0) & (duration < width - 1)).sum()
        ),
        "negative_duration": int(
            (duration < 0).sum()
        ),
    }

    return observations, metrics


def form_dataset(
    observed: pd.DataFrame,
    anchor_close: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
):
    grid = pd.date_range(
        start, end, freq="5min", inclusive="left"
    )

    if (
        observed.index.has_duplicates
        or not observed.index.isin(grid).all()
    ):
        raise ValueError("Invalid combined temporal grid")

    if (
        not np.isfinite(anchor_close)
        or anchor_close <= 0
    ):
        raise ValueError("Invalid preceding-day close")

    close = observed["close_usdt"].reindex(grid)
    complete = observed["complete_candle"].reindex(
        grid, fill_value=False
    )
    present = close.notna()

    # Reconstruct the full clock grid before computing
    # returns, so gaps cannot become multi-hour 5m returns.
    anchor_time = pd.date_range(
        end=start, periods=2, freq="5min"
    )[-2]

    anchored_close = pd.concat([
        pd.Series({anchor_time: anchor_close}),
        close,
    ])

    log_return = (
        np.log(anchored_close)
        .diff()
        .reindex(grid)
    )

    previous_complete = pd.Series(
        np.r_[
            True,
            complete.to_numpy(dtype=bool)[:-1],
        ],
        index=grid,
    )

    valid_return = (
        complete
        & previous_complete
        & present
        & log_return.notna()
    )

    log_return = log_return.where(valid_return)

    bars = pd.DataFrame({
        "time_utc": grid,
        "close_usdt": close.to_numpy(),
        "bar_present": present.to_numpy(),
        "complete_candle": complete.to_numpy(),
        "log_return_5m": log_return.to_numpy(),
        "squared_return_5m": (
            log_return.pow(2).to_numpy()
        ),
    })

    dates = pd.date_range(
        start, end, freq="D", inclusive="left"
    )
    keys = grid.floor("D")

    n_bars = (
        present.groupby(keys)
        .sum()
        .reindex(dates, fill_value=0)
        .astype(int)
    )

    n_invalid_duration = (
        (present & ~complete)
        .groupby(keys)
        .sum()
        .reindex(dates, fill_value=0)
        .astype(int)
    )

    n_returns = (
        valid_return.groupby(keys)
        .sum()
        .reindex(dates, fill_value=0)
        .astype(int)
    )

    rv_raw = (
        log_return.pow(2)
        .groupby(keys)
        .sum(min_count=1)
        .reindex(dates)
    )

    rv_valid = (
        (n_bars == 288)
        & (n_invalid_duration == 0)
        & (n_returns == 288)
    )

    rv = rv_raw.where(rv_valid)

    # Exact 23:55 closing prices; never forward-fill
    # a missing daily close.
    day_end = dates + pd.offsets.Minute(1435)
    daily_close = close.reindex(day_end).to_numpy()

    previous_close = np.concatenate((
        [anchor_close],
        daily_close[:-1],
    ))

    with np.errstate(
        invalid="ignore", divide="ignore"
    ):
        daily_return = (
            np.log(daily_close)
            - np.log(previous_close)
        )

    daily = pd.DataFrame({
        "date_utc": dates,
        "close_usdt": daily_close,
        "daily_log_return": daily_return,
        "rv_5m": rv.to_numpy(),
        "rv_vol_annualized": (
            np.sqrt(365 * rv.to_numpy())
        ),
        "n_5m_observed": n_bars.to_numpy(),
        "n_invalid_duration_candles": n_invalid_duration.to_numpy(),
        "n_valid_5m_returns": (
            n_returns.to_numpy()
        ),
        "rv_valid": rv_valid.to_numpy(),
        "daily_return_valid": (
            np.isfinite(daily_return)
        ),
    })

    return bars, daily


def build(root: Path):
    sys.path.insert(0, str(root / "src"))
    from hc_vol.data import month_archive

    reference_path = (
        root / "configs/source_quality_reference_v1.json"
    )
    reference = json.loads(
        reference_path.read_text()
    )

    reference_months = {
        item["month"]: item
        for item in reference["monthly"]
    }

    months = [
        str(month)
        for month in pd.period_range(
            "2019-12", "2026-09", freq="M"
        )
    ]

    if (
        set(reference_months) != set(months)
        or len(reference["monthly"]) != 82
    ):
        raise ValueError(
            "Source reference must contain 82 months"
        )

    raw = (
        root / "data/raw/binance_spot_btcusdt_5m"
    )

    frames = []
    source_hashes = []
    anchor_close = None
    total_short = 0
    total_negative = 0

    for index, month in enumerate(months, 1):
        path, metadata = month_archive(raw, month)
        frame, metrics = parse_archive(path, month)
        ref = reference_months[month]

        if (
            metadata["sha256"]
            != ref["source_sha256"]
        ):
            raise ValueError(
                f"Source hash diverges: {month}"
            )

        for key in (
            "expected",
            "observed",
            "missing",
            "bad_duration",
        ):
            if metrics[key] != ref[key]:
                raise ValueError(
                    f"Source audit drift: {month}/{key}"
                )

        total_short += metrics["short_duration"]
        total_negative += metrics["negative_duration"]

        if month == "2019-12":
            if (
                ANCHOR not in frame.index
                or not bool(
                    frame.loc[
                        ANCHOR, "complete_candle"
                    ]
                )
            ):
                raise ValueError(
                    "Missing or invalid warmup candle"
                )
            anchor_close = float(
                frame.loc[ANCHOR, "close_usdt"]
            )
        else:
            frames.append(frame)

        source_hashes.append(metadata)

        print(
            f"[{index:02d}/82] {month} "
            "SHA256=PASS INVENTORY=PASS",
            flush=True,
        )

    observations = pd.concat(
        frames
    ).sort_index()

    bars, daily = form_dataset(
        observations,
        anchor_close,
        START,
        END,
    )

    invalid_rv = daily.loc[
        ~daily.rv_valid, "date_utc"
    ].dt.strftime("%Y-%m-%d").tolist()

    invalid_daily_return = daily.loc[
        ~daily.daily_return_valid, "date_utc"
    ].dt.strftime("%Y-%m-%d").tolist()

    if (
        invalid_rv != reference["invalid_rv_days"]
        or invalid_daily_return
        != reference["invalid_daily_return_days"]
    ):
        raise ValueError(
            "Calculated invalid days differ "
            "from source audit"
        )

    actual = {
        "short_candles": total_short,
        "negative_duration_candles": total_negative,
        "expected_5m": len(bars),
        "observed_5m": int(
            bars.bar_present.sum()
        ),
        "missing_5m": int(
            (~bars.bar_present).sum()
        ),
        "invalid_duration_candles": int(
            (
                bars.bar_present
                & ~bars.complete_candle
            ).sum()
        ),
        "daily_dates": len(daily),
        "valid_rv_days": int(
            daily.rv_valid.sum()
        ),
        "invalid_rv_days": invalid_rv,
        "valid_daily_returns": int(
            daily.daily_return_valid.sum()
        ),
        "invalid_daily_return_days": (
            invalid_daily_return
        ),
    }

    expected = {
        "short_candles": 7,
        "negative_duration_candles": 1,
        "expected_5m": 709920,
        "observed_5m": 709458,
        "missing_5m": 462,
        "invalid_duration_candles": 8,
        "daily_dates": 2465,
        "valid_rv_days": 2449,
        "valid_daily_returns": 2465,
    }

    for key, value in expected.items():
        if actual[key] != value:
            raise ValueError(
                f"Unexpected {key}: "
                f"{actual[key]} != {value}"
            )

    dest = (
        root / "data/processed/"
        "quality_v1_1_through_2026-09"
    )
    dest.mkdir(parents=True, exist_ok=True)

    for name, dataframe in (
        ("bars_5m.parquet", bars),
        ("daily_volatility.parquet", daily),
    ):
        temp = dest / (name + ".part")
        dataframe.to_parquet(
            temp, index=False, compression="zstd"
        )
        os.replace(temp, dest / name)

    csv_temp = (
        dest / "daily_volatility.csv.part"
    )
    daily.to_csv(
        csv_temp,
        index=False,
        float_format="%.15g",
        na_rep="",
    )
    os.replace(
        csv_temp,
        dest / "daily_volatility.csv"
    )

    qa = {
        "status": "PASS_WITH_DOCUMENTED_EXCLUSIONS",
        **actual,
    }

    (dest / "qa_report.json").write_text(
        json.dumps(qa, indent=2) + "\n"
    )

    outputs = [
        "bars_5m.parquet",
        "daily_volatility.parquet",
        "daily_volatility.csv",
        "qa_report.json",
    ]

    manifest = {
        "protocol_id": "HC-BTCVOL-001",
        "data_policy": "v1.1",
        "status": (
            "PASS_WITH_DOCUMENTED_EXCLUSIONS"
        ),
        "source_reference_sha256": sha256(
            reference_path
        ),
        "source_archives": source_hashes,
        "quality": actual,
        "output_sha256": {
            name: sha256(dest / name)
            for name in outputs
        },
    }

    (dest / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print(
        "DATASET_V1_1=PASS "
        f"VALID_RV={actual['valid_rv_days']} "
        f"INVALID_RV={len(invalid_rv)}"
    )
    print(f"OUTPUT={dest}")


if __name__ == "__main__":
    build(
        Path(__file__).resolve().parents[2]
    )
