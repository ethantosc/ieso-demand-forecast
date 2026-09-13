#!/usr/bin/env python3
"""Land IESO hourly demand CSVs as a single staging Parquet.

Run from the repo root:

    python -m src.ingest.demand

Reads every ``data/raw/demand/PUB_Demand_<year>.csv`` and writes
``data/staging/demand_hourly.parquet``. Raw files are never modified, so this
step is always safe to re-run.

--------------------------------------------------------------------------
THE TIME CONVENTION - read this before changing anything below
--------------------------------------------------------------------------
The CSV gives a Date and an Hour of 1-24. Hour 1 is the hour *ending* at
01:00, i.e. it covers 00:00-01:00. We relabel to hour-beginning so that an
"hour 0" row means midnight-to-1am, which is what every calendar feature and
every lag downstream will assume.

We also assume the CSV timeline is fixed EST (UTC-5 all year), not local
wall-clock time. That assumption is not decoration - it decides whether the
series is continuous. It rests on one observation: every day in these files
has exactly 24 rows, including the March and November DST transition days. A
wall-clock series cannot do that; it has a 23-hour day and a 25-hour day.

``check_demand.py`` re-tests that assumption against the real files. If the
test fails, this script is wrong and must be rewritten before anything is
modelled on top of it. Do not skip the check.

Local Toronto wall-clock columns are derived afterwards, because human load
behaviour (work hours, lighting, air conditioning) follows the clock on the
wall, not a fixed offset. Feature engineering should use ``local_hour``;
lag arithmetic and joins should use ``ts_utc``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import polars as pl

RAW_DIR = Path("data/raw/demand")
OUT_PATH = Path("data/staging/demand_hourly.parquet")

LOCAL_TZ = "America/Toronto"
EST_OFFSET_HOURS = 5  # EST is UTC-5, fixed, no DST

HEADER_ROWS = 3  # three junk lines before the real header row
FILE_PATTERN = re.compile(r"^PUB_Demand_(?P<year>\d{4})\.csv$", re.IGNORECASE)

# Normalised source column -> our name.
COLUMN_MAP = {
    "date": "delivery_date",
    "hour": "hour_ending",
    "market_demand": "market_demand_mw",
    "ontario_demand": "ontario_demand_mw",
}


def normalise(name: str) -> str:
    """Fold a CSV header into a stable key: lowercase, underscores, no spaces."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def read_one(path: Path) -> pl.DataFrame:
    """Read one annual CSV as strings, then cast explicitly.

    Everything is read as Utf8 first (``infer_schema_length=0``) so that a
    single odd row in one year cannot silently change a column's type and
    make two years unstackable.
    """
    frame = pl.read_csv(
        path,
        skip_rows=HEADER_ROWS,
        infer_schema_length=0,
        truncate_ragged_lines=True,
        ignore_errors=False,
    )

    renames = {}
    for column in frame.columns:
        key = normalise(column)
        if key in COLUMN_MAP:
            renames[column] = COLUMN_MAP[key]
    frame = frame.rename(renames)

    missing = set(COLUMN_MAP.values()) - set(frame.columns)
    if missing:
        raise ValueError(
            f"{path.name}: expected columns not found: {sorted(missing)}. "
            f"Header row read as: {frame.columns}"
        )

    # Drop fully blank trailing rows some IESO exports carry.
    frame = frame.filter(pl.col("delivery_date").is_not_null() & (pl.col("delivery_date") != ""))

    parsed = None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"):
        candidate = frame.select(
            pl.col("delivery_date").str.strptime(pl.Date, fmt, strict=False)
        ).to_series()
        if candidate.null_count() == 0:
            parsed = candidate
            break
    if parsed is None:
        sample = frame["delivery_date"].head(3).to_list()
        raise ValueError(
            f"{path.name}: could not parse the Date column. Sample values: {sample}. "
            "Add the right format to the list in read_one()."
        )
    frame = frame.with_columns(parsed.alias("delivery_date"))

    return frame.select(
        pl.col("delivery_date"),
        pl.col("hour_ending").cast(pl.Int16),
        pl.col("market_demand_mw").cast(pl.Float64),
        pl.col("ontario_demand_mw").cast(pl.Float64),
        pl.lit(path.name).alias("source_file"),
    )


def add_time_columns(frame: pl.DataFrame) -> pl.DataFrame:
    """Turn (delivery_date, hour_ending) into a real instant plus local labels."""
    return (
        frame.with_columns(
            # hour-ending 1..24  ->  hour-beginning 0..23 on the EST timeline
            (
                pl.col("delivery_date").cast(pl.Datetime("us"))
                + pl.duration(hours=pl.col("hour_ending") - 1)
            ).alias("ts_est")
        )
        .with_columns(
            (pl.col("ts_est") + pl.duration(hours=EST_OFFSET_HOURS)).alias("ts_utc")
        )
        .with_columns(
            pl.col("ts_utc")
            .dt.replace_time_zone("UTC")
            .dt.convert_time_zone(LOCAL_TZ)
            .dt.replace_time_zone(None)
            .alias("ts_local")
        )
        .with_columns(
            pl.col("ts_local").dt.date().alias("local_date"),
            pl.col("ts_local").dt.hour().cast(pl.Int8).alias("local_hour"),
            (
                (pl.col("ts_local") - pl.col("ts_utc")).dt.total_minutes() // 60
            ).cast(pl.Int8).alias("utc_offset_hours"),
        )
        .drop("ts_est")
        .select(
            "delivery_date",
            "hour_ending",
            "ts_utc",
            "ts_local",
            "local_date",
            "local_hour",
            "utc_offset_hours",
            "market_demand_mw",
            "ontario_demand_mw",
            "source_file",
        )
        .sort("ts_utc")
    )


def main() -> int:
    if not RAW_DIR.is_dir():
        print(f"raw directory not found: {RAW_DIR.resolve()}", file=sys.stderr)
        return 1

    paths = sorted(p for p in RAW_DIR.iterdir() if FILE_PATTERN.match(p.name))
    if not paths:
        print(f"no PUB_Demand_<year>.csv files in {RAW_DIR.resolve()}", file=sys.stderr)
        return 1

    frames = []
    for path in paths:
        frame = read_one(path)
        print(f"  {path.name:28s} rows={frame.height:>6,}")
        frames.append(frame)

    combined = add_time_columns(pl.concat(frames, how="vertical"))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(OUT_PATH, compression="zstd")

    span_start = combined["ts_local"].min()
    span_end = combined["ts_local"].max()
    print(
        f"\nwrote {OUT_PATH}  rows={combined.height:,}  "
        f"local span {span_start} .. {span_end}  "
        f"({OUT_PATH.stat().st_size / 1e6:.1f} MB)"
    )
    print("\nNow run:  python -m src.ingest.check_demand")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
