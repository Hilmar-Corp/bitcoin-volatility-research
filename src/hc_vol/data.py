"""HC-BTCVOL-001: verified Binance data to daily realized variance."""

import argparse
import certifi
import ssl
import hashlib
import json
import os
import platform
import re
import time
import urllib.request
import zipfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/5m"
COLS = [
    "open_time", "open", "high", "low", "close",
    "volume", "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore"
]
START = pd.Timestamp("2020-01-01", tz="UTC")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, dest, replace=False):
    if dest.exists() and not replace:
        return
    part = dest.with_name(dest.name + ".part")
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "HilmarCorp-Research/1.0"}
            )
            with urllib.request.urlopen(req, timeout=90, context=ssl.create_default_context(cafile=certifi.where())) as r, \
                    part.open("wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            os.replace(part, dest)
            return
        except Exception:
            part.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def month_archive(raw, month):
    name = f"BTCUSDT-5m-{month}.zip"
    path = raw / name
    check = raw / (name + ".CHECKSUM")

    fetch(f"{BASE}/{name}.CHECKSUM", check, replace=True)
    fields = check.read_text().strip().split()

    if not fields or not re.fullmatch(
        r"[a-fA-F0-9]{64}", fields[0]
    ):
        raise ValueError(f"Invalid checksum format: {month}")

    if len(fields) > 1 and Path(
        fields[1].lstrip("*")
    ).name != name:
        raise ValueError(f"Wrong checksum target: {month}")

    expected = fields[0].lower()

    for attempt in range(2):
        fetch(f"{BASE}/{name}", path)
        actual = digest(path)

        if actual == expected:
            return path, {
                "month": month,
                "url": f"{BASE}/{name}",
                "sha256": actual
            }

        path.unlink(missing_ok=True)

    raise ValueError(f"SHA256 mismatch: {month}")


def parse_month(path, month):
    with zipfile.ZipFile(path) as z:
        names = [
            n for n in z.namelist()
            if n.endswith(".csv")
            and not n.startswith("__MACOSX/")
        ]
        if len(names) != 1:
            raise ValueError(f"Invalid ZIP content: {month}")

        with z.open(names[0]) as f:
            df = pd.read_csv(
                f, header=None, low_memory=False
            )

    if df.shape[1] != 12:
        raise ValueError(f"Invalid schema: {month}")

    df.columns = COLS

    for col in ("open_time", "close_time", "trades"):
        df[col] = pd.to_numeric(
            df[col], errors="raise"
        ).astype("int64")

    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(
            df[col], errors="raise"
        ).astype("float64")

    unit = "us" if month >= "2025-01" else "ms"
    width = 300_000_000 if unit == "us" else 300_000

    valid_time = (
        (df.close_time - df.open_time == width - 1)
        & (df.open_time % width == 0)
    )

    if not valid_time.all():
        raise ValueError(f"Invalid timestamp format: {month}")

    ts = pd.DatetimeIndex(
        pd.to_datetime(df.open_time, unit=unit, utc=True)
    )

    first = pd.Timestamp(month + "-01", tz="UTC")
    last = first + pd.offsets.MonthBegin(1)
    expected = pd.date_range(
        first, last, freq="5min", inclusive="left"
    )

    if not ts.equals(expected):
        missing = len(expected.difference(ts))
        duplicate = int(ts.duplicated().sum())
        raise ValueError(
            f"Invalid temporal grid {month}: "
            f"missing={missing}, duplicate={duplicate}, "
            f"observed={len(ts)}, expected={len(expected)}"
        )

    p = df[["open", "high", "low", "close"]]

    invalid = (
        not np.isfinite(p.to_numpy()).all()
        or (p <= 0).any().any()
        or not np.isfinite(df.volume.to_numpy()).all()
        or (df.volume < 0).any()
        or (df.trades < 0).any()
        or (
            df.low
            > p[["open", "close", "high"]].min(axis=1)
        ).any()
        or (
            df.high
            < p[["open", "close", "low"]].max(axis=1)
        ).any()
    )

    if invalid:
        raise ValueError(f"Invalid OHLCV/trades: {month}")

    return pd.DataFrame({
        "time_utc": ts,
        "close_usdt": df.close.to_numpy()
    })


def aggregate(bars, through):
    end = pd.Timestamp(
        str(pd.Period(through, freq="M") + 1) + "-01",
        tz="UTC"
    )

    warmup = pd.Timestamp("2019-12-31T23:55:00Z")
    expected = pd.date_range(
        warmup, end, freq="5min", inclusive="left"
    )
    times = pd.DatetimeIndex(bars.time_utc)

    if not times.equals(expected):
        raise ValueError(
            "Combined history contains a missing, duplicate "
            "or unexpected interval"
        )

    prices = bars.close_usdt.to_numpy(dtype="float64")
    returns = np.diff(np.log(prices))

    if not np.isfinite(returns).all():
        raise ValueError("Non-finite log returns")

    intraday = pd.DataFrame({
        "time_utc": times[1:],
        "close_usdt": prices[1:],
        "log_return_5m": returns,
        "squared_log_return_5m": returns ** 2
    })

    intraday["date_utc"] = (
        intraday.time_utc.dt.floor("D")
    )

    daily = intraday.groupby(
        "date_utc", as_index=False, sort=True
    ).agg(
        close_usdt=("close_usdt", "last"),
        daily_log_return=("log_return_5m", "sum"),
        rv_5m=("squared_log_return_5m", "sum"),
        n_5m=("close_usdt", "size")
    )

    expected_days = len(pd.date_range(
        START, end, freq="D", inclusive="left"
    ))

    if (
        len(daily) != expected_days
        or not (daily.n_5m == 288).all()
    ):
        raise ValueError("Incomplete daily aggregation")

    actual = daily.daily_log_return.to_numpy()[1:]
    reconciled = np.diff(
        np.log(daily.close_usdt.to_numpy())
    )

    if not np.allclose(
        actual, reconciled, rtol=1e-10, atol=1e-12
    ):
        raise ValueError(
            "Daily close-to-close reconciliation failed"
        )

    daily["rv_vol_annualized"] = np.sqrt(
        365 * daily.rv_5m
    )

    return intraday, daily


def build(root, through):
    if (
        not re.fullmatch(
            r"20\d\d-(0[1-9]|1[0-2])", through
        )
        or not "2020-01" <= through <= "2026-09"
    ):
        raise ValueError(
            "Invalid --through: use 2020-01 to 2026-09"
        )

    raw = (
        root / "data/raw/binance_spot_btcusdt_5m"
    )
    dest = (
        root / f"data/processed/through_{through}"
    )

    raw.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)

    months = [
        "2019-12",
        *(
            str(p) for p in pd.period_range(
                "2020-01", through, freq="M"
            )
        )
    ]

    frames = []
    sources = []

    for i, month in enumerate(months, 1):
        path, metadata = month_archive(raw, month)
        frame = parse_month(path, month)

        if month == "2019-12":
            frame = frame.tail(1)

        frames.append(frame)
        sources.append(metadata)

        print(
            f"[{i:02d}/{len(months)}] {month} "
            "SHA256=PASS GRID=PASS",
            flush=True
        )

    intraday, daily = aggregate(
        pd.concat(frames, ignore_index=True),
        through
    )

    outputs = {
        "bars_5m.parquet": intraday,
        "daily_volatility.parquet": daily
    }

    for name, df in outputs.items():
        target = dest / name
        temp = dest / (name + ".part")
        df.to_parquet(
            temp, index=False, compression="zstd"
        )
        os.replace(temp, target)

    csv = dest / "daily_volatility.csv"
    temp = dest / "daily_volatility.csv.part"

    daily.to_csv(
        temp, index=False, float_format="%.15g"
    )
    os.replace(temp, csv)

    qa = {
        "status": "PASS",
        "observed_intervals": len(intraday),
        "observed_days": len(daily),
        "missing_intervals": 0,
        "duplicate_intervals": 0,
        "invalid_ohlcv": 0,
        "timezone": "UTC"
    }

    (dest / "qa_report.json").write_text(
        json.dumps(qa, indent=2) + "\n"
    )

    manifest = {
        "protocol_id": "HC-BTCVOL-001",
        "through": through,
        "rows_5m": len(intraday),
        "rows_daily": len(daily),
        "formula": (
            "RV=sum of 288 squared 5-min close-to-close "
            "log returns per UTC calendar day"
        ),
        "full_study_period": through == "2026-09",
        "archive_sha256": sources,
        "output_sha256": {
            name: digest(dest / name)
            for name in [
                *outputs, "daily_volatility.csv"
            ]
        },
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "pyarrow": version("pyarrow"),
        "qa": "PASS"
    }

    (dest / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print(
        "DATASET_READY=PASS "
        f"ROWS_5M={len(intraday)} "
        f"ROWS_DAILY={len(daily)}"
    )

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--through", default="2026-09"
    )
    args = parser.parse_args()

    build(
        Path(__file__).resolve().parents[2],
        args.through
    )
