"""DATA-QA-001: independent Binance February 2020 source audit."""

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
AUDIT = RAW / "forensics_2020_02"
OUTPUT = ROOT / "data/processed/forensics_2020_02"
BASE = "https://data.binance.vision/data/spot"

AUDIT.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_verified(category, name):
    url = f"{BASE}/{category}/{name}.zip"
    path = AUDIT / f"{name}.zip"
    check = AUDIT / f"{name}.zip.CHECKSUM"

    for target, remote in (
        (check, url + ".CHECKSUM"),
        (path, url),
    ):
        if target.exists() and target.stat().st_size:
            continue

        part = target.with_name(target.name + ".part")
        subprocess.run(
            [
                "curl", "-fLsS", "--retry", "3",
                "--connect-timeout", "15",
                "--max-time", "180",
                "-o", str(part), remote,
            ],
            check=True,
        )
        part.replace(target)

    parts = check.read_text().split()
    assert parts and len(parts[0]) == 64

    if len(parts) > 1:
        assert Path(parts[1].lstrip("*")).name == path.name

    assert sha256(path) == parts[0].lower(), name
    print(f"ARCHIVE_SHA256=PASS {name}")
    return path


def read_klines(path):
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        assert len(names) == 1

        with z.open(names[0]) as f:
            df = pd.read_csv(f, header=None)

    assert df.shape[1] == 12
    df["open_utc"] = pd.to_datetime(
        pd.to_numeric(df[0]), unit="ms", utc=True
    )
    df["close_price"] = pd.to_numeric(df[4])
    df["duration_ms"] = (
        pd.to_numeric(df[6]) - pd.to_numeric(df[0])
    )

    assert not df["open_utc"].duplicated().any()
    assert df["open_utc"].is_monotonic_increasing
    assert np.isfinite(df["close_price"]).all()
    assert (df["close_price"] > 0).all()

    return df.set_index("open_utc")


print("=== 1. MONTHLY SOURCES ===")

monthly5 = read_klines(
    RAW / "binance_spot_btcusdt_5m"
    / "BTCUSDT-5m-2020-02.zip"
)

monthly1 = None
try:
    monthly1 = read_klines(fetch_verified(
        "monthly/klines/BTCUSDT/1m",
        "BTCUSDT-1m-2020-02",
    ))
except (OSError, subprocess.CalledProcessError) as exc:
    print(f"ONE_MINUTE_SOURCE=UNAVAILABLE {exc}")

start = pd.Timestamp("2020-02-01", tz="UTC")
end = pd.Timestamp("2020-03-01", tz="UTC")

grid5 = pd.date_range(
    start, end, freq="5min", inclusive="left"
)
missing5 = grid5.difference(monthly5.index)

print(f"EXPECTED_5M={len(grid5)}")
print(f"OBSERVED_5M={len(monthly5)}")
print(f"MISSING_5M={len(missing5)}")

report = {
    "incident": "DATA-QA-001",
    "month": "2020-02",
    "monthly_5m_missing": len(missing5),
    "missing_by_day": {},
    "short_candles": [],
    "one_minute": None,
    "daily_sources": {},
    "decision": "AUDIT_ONLY_NO_REPAIR",
}

for day, count in (
    pd.Series(1, index=missing5)
    .groupby(missing5.floor("D")).sum()
    .items()
):
    key = day.strftime("%Y-%m-%d")
    report["missing_by_day"][key] = int(count)
    print(f"MISSING_DAY={key} COUNT={count}")

print("=== 2. SHORT CANDLE ===")

short = monthly5[
    monthly5.duration_ms != 299999
]
for ts, row in short.iterrows():
    record = {
        "open_utc": ts.isoformat(),
        "duration_ms": int(row.duration_ms),
        "close_price": float(row.close_price),
    }
    report["short_candles"].append(record)
    print(f"SHORT_CANDLE={record}")

print("=== 3. ONE-MINUTE SOURCE CHECK ===")

if monthly1 is not None:
    grid1 = pd.date_range(
        start, end, freq="1min", inclusive="left"
    )
    missing1 = grid1.difference(monthly1.index)

    recoverable = []
    unavailable = []

    for ts in missing5:
        wanted = pd.date_range(
            ts, periods=5, freq="1min"
        )

        if all(t in monthly1.index for t in wanted):
            segment = monthly1.loc[wanted]
            if (segment.duration_ms == 59999).all():
                recoverable.append(ts)
            else:
                unavailable.append(ts)
        else:
            unavailable.append(ts)

    # Cross-check 1m-derived and native 5m closes.
    minute_last = monthly1["close_price"].reindex(
        monthly5.index + pd.Timedelta(minutes=4)
    )
    prices5 = monthly5["close_price"].to_numpy()

    valid = minute_last.notna().to_numpy()
    matched = np.isclose(
        minute_last.to_numpy()[valid],
        prices5[valid],
        rtol=0,
        atol=1e-6,
    )

    report["one_minute"] = {
        "rows": len(monthly1),
        "missing_minutes": len(missing1),
        "recoverable_5m_intervals": len(recoverable),
        "unrecoverable_5m_intervals": len(unavailable),
        "overlap_compared": int(valid.sum()),
        "overlap_price_mismatches": int(
            (~matched).sum()
        ),
        "recoverable_timestamps": [
            t.isoformat() for t in recoverable
        ],
    }

    for key, value in report["one_minute"].items():
        if key != "recoverable_timestamps":
            print(f"{key.upper()}={value}")

    for ts, row in short.iterrows():
        wanted = pd.date_range(
            ts, periods=5, freq="1min"
        )
        complete = (
            all(t in monthly1.index for t in wanted)
            and (
                monthly1.loc[wanted, "duration_ms"]
                == 59999
            ).all()
        )
        print(
            f"SHORT_CANDLE_1M_COMPLETE={complete}"
            f" DATE={ts.isoformat()}"
        )

print("=== 4. DAILY VS MONTHLY ===")

for day in ("2020-02-09", "2020-02-19"):
    name = f"BTCUSDT-5m-{day}"

    try:
        daily = read_klines(fetch_verified(
            "daily/klines/BTCUSDT/5m",
            name,
        ))
    except (OSError, subprocess.CalledProcessError) as exc:
        report["daily_sources"][day] = {
            "status": "UNAVAILABLE",
            "reason": str(exc),
        }
        print(f"DAILY_ARCHIVE_UNAVAILABLE={day}")
        continue

    month_day = monthly5[
        monthly5.index.strftime("%Y-%m-%d") == day
    ]
    common = daily.index.intersection(month_day.index)

    price_diff = ~np.isclose(
        daily.loc[common, "close_price"].to_numpy(),
        month_day.loc[common, "close_price"].to_numpy(),
        atol=1e-6,
        rtol=0,
    )
    duration_diff = (
        daily.loc[common, "duration_ms"].to_numpy()
        != month_day.loc[common, "duration_ms"].to_numpy()
    )

    extra = daily.index.difference(month_day.index)

    result = {
        "status": "VERIFIED",
        "daily_rows": len(daily),
        "monthly_rows": len(month_day),
        "daily_only_intervals": len(extra),
        "overlap_close_disagreements": int(
            price_diff.sum()
        ),
        "overlap_duration_disagreements": int(
            duration_diff.sum()
        ),
    }
    report["daily_sources"][day] = result
    print(f"DAILY_COMPARISON_{day}={result}")

(OUTPUT / "source_audit.json").write_text(
    json.dumps(report, indent=2) + "\n"
)

print("=== FINAL ===")
print("SOURCE_FORENSICS=COMPLETE")
print("RECONSTRUCTION=NOT_APPLIED")
print(f"AUDIT_REPORT={OUTPUT / 'source_audit.json'}")
