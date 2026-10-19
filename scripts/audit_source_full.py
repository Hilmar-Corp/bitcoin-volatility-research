"""HC-BTCVOL-001: full historical source-quality inventory."""

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hc_vol.data import month_archive

RAW = ROOT / "data/raw/binance_spot_btcusdt_5m"
OUT = ROOT / "data/processed/source_audit_full"
OUT.mkdir(parents=True, exist_ok=True)

months = pd.period_range("2019-12", "2026-09", freq="M")

problem_days = set()
valid_last_close_days = set()
monthly_reports = []

for i, period in enumerate(months, 1):
    month = str(period)

    # Original source and independently verified checksum.
    path, source_meta = month_archive(RAW, month)

    with zipfile.ZipFile(path) as z:
        names = [
            n for n in z.namelist()
            if n.endswith(".csv")
            and not n.startswith("__MACOSX/")
        ]
        assert len(names) == 1, month

        with z.open(names[0]) as f:
            df = pd.read_csv(
                f, header=None, low_memory=False
            )

    assert df.shape[1] == 12, month

    unit = "us" if month >= "2025-01" else "ms"
    width = 300_000_000 if unit == "us" else 300_000

    first = period.start_time.tz_localize("UTC")
    last = (period + 1).start_time.tz_localize("UTC")

    opens = pd.to_numeric(
        df[0], errors="raise"
    ).astype("int64")

    closes = pd.to_numeric(
        df[6], errors="raise"
    ).astype("int64")

    ts = pd.DatetimeIndex(
        pd.to_datetime(
            opens, unit=unit, utc=True
        )
    )

    expected = pd.date_range(
        first, last, freq="5min", inclusive="left"
    )

    missing = expected.difference(ts)
    extra = ts.difference(expected)
    duplicates = ts.duplicated()

    misaligned = (opens % width != 0).to_numpy()
    wrong_duration = (
        (closes - opens) != width - 1
    ).to_numpy()

    values = df[[1, 2, 3, 4, 5]].apply(
        pd.to_numeric, errors="raise"
    )

    bad_numeric = ~np.isfinite(
        values.to_numpy()
    ).all(axis=1)

    bad_prices = (
        values[[1, 2, 3, 4]] <= 0
    ).any(axis=1).to_numpy()

    bad_ohlc = (
        (
            values[3]
            > values[[1, 2, 4]].min(axis=1)
        )
        |
        (
            values[2]
            < values[[1, 3, 4]].max(axis=1)
        )
    ).to_numpy()

    bad_volume = (
        values[5] < 0
    ).to_numpy()

    bad_trades = (
        pd.to_numeric(df[8], errors="raise") < 0
    ).to_numpy()

    invalid = (
        bad_numeric | bad_prices | bad_ohlc
        | bad_volume | bad_trades
    )

    bad_rows = (
        wrong_duration | invalid | misaligned
        | duplicates
        | ~ts.isin(expected)
    )

    current_problem = set(
        missing.floor("D").strftime("%Y-%m-%d")
    )
    current_problem.update(
        ts[bad_rows].floor("D").strftime("%Y-%m-%d")
    )

    problem_days.update(current_problem)

    # A valid 23:55 candle is required for the
    # next day's first five-minute return.
    is_last = (
        (ts.hour == 23)
        & (ts.minute == 55)
    )

    valid_last = is_last & ~bad_rows

    valid_last_close_days.update(
        ts[valid_last].strftime("%Y-%m-%d")
    )

    record = {
        "month": month,
        "source_sha256": source_meta["sha256"],
        "expected": len(expected),
        "observed": len(df),
        "missing": len(missing),
        "extra": len(extra),
        "duplicate": int(duplicates.sum()),
        "misaligned": int(misaligned.sum()),
        "bad_duration": int(wrong_duration.sum()),
        "invalid_ohlcv": int(invalid.sum()),
        "affected_days": sorted(current_problem),
        "first_missing": [
            t.isoformat() for t in missing[:10]
        ],
    }

    monthly_reports.append(record)

    status = (
        "ANOMALY" if (
            current_problem or len(extra)
        ) else "PASS"
    )

    print(
        f"[{i:02d}/{len(months)}] {month} "
        f"SHA256=PASS QA={status} "
        f"MISSING={len(missing)} "
        f"BAD_DURATION={int(wrong_duration.sum())}",
        flush=True,
    )

study_days = pd.date_range(
    "2020-01-01",
    "2026-09-30",
    freq="D",
    tz="UTC",
)

invalid_rv_days = []
invalid_return_days = []

for day in study_days:
    current = day.strftime("%Y-%m-%d")
    previous = (
        day - pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    if (
        current in problem_days
        or previous not in valid_last_close_days
    ):
        invalid_rv_days.append(current)

    if (
        current not in valid_last_close_days
        or previous not in valid_last_close_days
    ):
        invalid_return_days.append(current)

report = {
    "protocol": "HC-BTCVOL-001",
    "type": "complete_source_inventory",
    "archives": len(months),
    "expected_daily_dates": len(study_days),
    "candidate_valid_rv_days": (
        len(study_days) - len(invalid_rv_days)
    ),
    "invalid_rv_days": invalid_rv_days,
    "invalid_daily_return_days": invalid_return_days,
    "all_problem_days": sorted(problem_days),
    "monthly": monthly_reports,
    "reconstruction_performed": False,
}

path = OUT / "source_inventory.json"
path.write_text(
    json.dumps(report, indent=2) + "\n"
)

print()
print("=== HILMARCORP FULL SOURCE AUDIT ===")
print(f"ARCHIVES={len(months)}")
print(
    "TOTAL_MISSING_5M="
    f"{sum(x['missing'] for x in monthly_reports)}"
)
print(
    "TOTAL_BAD_DURATION="
    f"{sum(x['bad_duration'] for x in monthly_reports)}"
)
print(f"TOTAL_DAYS={len(study_days)}")
print(
    "CANDIDATE_VALID_RV_DAYS="
    f"{report['candidate_valid_rv_days']}"
)
print(f"INVALID_RV_DAYS={len(invalid_rv_days)}")

print("INVALID_DATES:")
for day in invalid_rv_days:
    print(f"  {day}")

print(
    "INVALID_DAILY_RETURN_DAYS="
    f"{len(invalid_return_days)}"
)

print("SOURCE_INVENTORY=COMPLETE")
print("RECONSTRUCTION_PERFORMED=NO")
print(f"REPORT={path}")
