#!/usr/bin/env python3
"""Parse the harvested IESO day-ahead forecasts into one tidy table.

    python -m src.ingest.forecasts

Reads every `data/raw/forecasts/DATotals/PUB_DATotals_YYYYMMDD.xml` and writes
`data/staging/official_forecast.parquet` with one row per delivery hour:
delivery_date, hour_ending, total_load_mw, total_energy_mw, created_at.

Two traps documented in spec section 2.4, both handled here:

1. The XML carries a default namespace, `http://www.ieso.ca/schema`. A plain
   `find("DocBody")` returns None without raising. Every lookup below goes
   through the namespace map, and a missing element raises instead.
2. The file-name date is the DeliveryDate, the day being forecast. The forecast
   itself was made the day before, at `CreatedAt`.

`Total Load` is the series to compare against Market Demand; spec section 1
established that by measuring it (ME +61 MW against Market Demand, +2,911 MW
against Ontario Demand).
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from datetime import date, datetime
from pathlib import Path

import polars as pl

SRC = Path("data/raw/forecasts/DATotals")
OUT = Path("data/staging/official_forecast.parquet")
NS = {"i": "http://www.ieso.ca/schema"}
PATTERN = "PUB_DATotals_[0-9]*.xml"
WANTED = {"Total Load": "total_load_mw", "Total Energy": "total_energy_mw"}


def _find(node: ET.Element, path: str) -> ET.Element:
    found = node.find(path, NS)
    if found is None:
        raise ValueError(f"missing element {path!r} - namespace or schema change?")
    return found


def parse_file(path: Path) -> list[dict]:
    root = ET.parse(path).getroot()
    created = datetime.fromisoformat(_find(root, "i:DocHeader/i:CreatedAt").text)
    body = _find(root, "i:DocBody")
    day = date.fromisoformat(_find(body, "i:DeliveryDate").text)
    rows = []
    for hourly in body.findall("i:Energies/i:HourlyEnergy", NS):
        row = {"delivery_date": day, "hour_ending": int(_find(hourly, "i:DeliveryHour").text),
               "created_at": created, "source_file": path.name}
        for mq in hourly.findall("i:MQ", NS):
            name = _find(mq, "i:MarketQuantity").text
            if name in WANTED:
                row[WANTED[name]] = float(_find(mq, "i:EnergyMW").text)
        rows.append(row)
    if len(rows) != 24:
        raise ValueError(f"{path.name}: expected 24 delivery hours, found {len(rows)}")
    return rows


def load(src: Path = SRC) -> pl.DataFrame:
    paths = sorted(p for p in src.glob(PATTERN) if len(p.stem) == len("PUB_DATotals_20260614"))
    if not paths:
        return pl.DataFrame(schema={"delivery_date": pl.Date, "hour_ending": pl.Int64,
                                    "created_at": pl.Datetime, "source_file": pl.Utf8,
                                    "total_load_mw": pl.Float64, "total_energy_mw": pl.Float64})
    rows = [r for p in paths for r in parse_file(p)]
    frame = pl.DataFrame(rows).sort("delivery_date", "hour_ending")
    stamped = frame["source_file"].str.slice(13, 8).str.strptime(pl.Date, "%Y%m%d")
    if (stamped != frame["delivery_date"]).any():
        raise ValueError("a file's name disagrees with its DeliveryDate")
    return frame


def main() -> int:
    frame = load()
    if frame.is_empty():
        print(f"no DATotals files under {SRC.resolve()}", file=sys.stderr)
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(OUT)
    days = frame["delivery_date"].unique().sort()
    print(f"wrote {OUT}  {frame.height:,} hours  {days.len()} days  {days.min()} .. {days.max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
