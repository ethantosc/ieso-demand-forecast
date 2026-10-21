#!/usr/bin/env python3
"""Hourly temperature from Environment and Climate Change Canada, for Track B only.

    python -m src.ingest.fetch_weather

Downloads ECCC bulk hourly CSVs into `data/raw/weather/` (one file per station
per month, cached), splices each city's old and new airport stations, and
writes `data/staging/weather_hourly.parquet`: ts_est, temp (city mean), one
column per city, and n_cities.

WHY THIS IS TRACK B ONLY. These are observed temperatures. At the 12:30 origin
nobody knows tomorrow's observed temperature, only a forecast of it, and
archived weather forecasts are not available here. Anything trained on these
is an upper bound, never a forecast. Spec section 4.

TIME BASE. ECCC hourly data is stamped in Local Standard Time all year, with no
daylight saving. For Ontario that is fixed EST, the same base as the demand
files, so the join is on EST hour with no conversion.

STATIONS. Each airport changed station ID once (the new automatic station took
over around 2011-2013). Both are listed, newest first; for each hour the newest
station with a value wins. The IDs below are the commonly cited ones, but they
could not be verified from the environment this was written in, so every file
is checked: the station name inside it must contain the expected city, and
months returning no temperatures are counted. A wrong ID costs that city's
coverage for those years and shows up in the coverage report; it cannot
silently put Ottawa's weather in Toronto's column.

Cities are averaged with equal weight. That is crude - Toronto carries far more
load than London - and it is listed as a limitation rather than tuned.
"""

from __future__ import annotations

import io
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import polars as pl

RAW = Path("data/raw/weather")
OUT = Path("data/staging/weather_hourly.parquet")
COVERAGE = Path("data/staging/weather_coverage.csv")
URL = ("https://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv"
       "&stationID={sid}&Year={y}&Month={m}&Day=14&timeframe=1&submit=Download+Data")
USER_AGENT = "ieso-demand-forecast/1.0 (personal research project)"
WORKERS = 4
RETRIES = 4
TIMEOUT = 30
MIN_CITY_COVERAGE = 0.90   # a city below this over the span is dropped, not averaged in
PROBE = (51459, 2024, 1)   # one known station-month, fetched first
FIRST_MONTH = (2003, 1)


@dataclass(frozen=True)
class Station:
    sid: int
    city_token: str
    first: tuple[int, int]
    last: tuple[int, int] | None = None


STATIONS: dict[str, list[Station]] = {
    "toronto": [Station(51459, "TORONTO", (2013, 6)), Station(5097, "TORONTO", (2003, 1), (2013, 7))],
    "ottawa": [Station(49568, "OTTAWA", (2011, 12)), Station(4337, "OTTAWA", (2003, 1), (2012, 1))],
    "london": [Station(50093, "LONDON", (2012, 1)), Station(4789, "LONDON", (2003, 1), (2013, 1))],
}


def _months(first: tuple[int, int], last: tuple[int, int]):
    y, m = first
    while (y, m) <= last:
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def _path(sid: int, y: int, m: int) -> Path:
    return RAW / f"{sid}_{y}_{m:02d}.csv"


def _download(job: tuple[int, int, int, bool]) -> str:
    sid, y, m, refresh = job
    path = _path(sid, y, m)
    if path.exists() and not refresh:
        return "cached"
    url = URL.format(sid=sid, y=y, m=m)
    for attempt in range(RETRIES):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                path.write_bytes(response.read())
            return "downloaded"
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt == RETRIES - 1:
                return f"failed: {error}"
            time.sleep(3 * (attempt + 1))
    return "failed"


def fetch(today: date | None = None) -> dict[str, int]:
    """Download every needed station-month not already cached. Returns status counts."""
    today = today or date.today()
    current = (today.year, today.month)
    RAW.mkdir(parents=True, exist_ok=True)
    jobs = []
    for stations in STATIONS.values():
        for st in stations:
            first = max(st.first, FIRST_MONTH)
            last = min(st.last or current, current)
            for y, m in _months(first, last):
                jobs.append((st.sid, y, m, (y, m) == current))
    # Fail fast: if one known file cannot be fetched, do not queue hundreds of
    # requests that will each time out. Track B is skipped and the run goes on.
    probe = _download((*PROBE, True))
    if probe.startswith("failed"):
        cached = len(list(RAW.glob("*.csv")))
        if not cached:
            raise ConnectionError(f"ECCC unreachable ({probe}) - Track B will not run")
        print(f"WARNING: ECCC unreachable ({probe}); using {cached} cached files", file=sys.stderr)
        return {"cached": cached, "unreachable": 1}
    counts: dict[str, int] = {}
    failed = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for i, (job, status) in enumerate(zip(jobs, pool.map(_download, jobs)), start=1):
            key = status.split(":")[0]
            counts[key] = counts.get(key, 0) + 1
            if key == "failed":
                failed.append(f"{job[0]}_{job[1]}_{job[2]:02d}")
            if i % 100 == 0 or i == len(jobs):
                print(f"  weather files {i}/{len(jobs)}", flush=True)
    # A failed station-month is not fatal, but it is not harmless either: that city
    # drops out of the average for the month and the mean shifts (Ottawa runs colder).
    if failed:
        print(f"WARNING: {len(failed)} weather file(s) failed to download: {', '.join(failed[:12])}"
              + (" ..." if len(failed) > 12 else ""), file=sys.stderr)
    return counts


def complete_through(hourly: pl.DataFrame | None):
    """Last EST day whose final hour (23:00) has an observed temperature.

    Observations are published hour by hour, so every earlier day is as complete
    as it will get, while later days may still be filling in. Track B is an upper
    bound only when it sees the whole observed day, so it is not scored after
    this day (spec-v2 section 7.2). None if there is no observation at all.
    """
    if hourly is None:
        return None
    have = hourly.filter(pl.col("temp").is_not_null())
    if have.is_empty():
        return None
    last = have["ts_est"].max()
    return last.date() if last.hour == 23 else last.date() - timedelta(days=1)


def parse_file(path: Path, city_token: str) -> pl.DataFrame | None:
    """One ECCC monthly CSV -> (ts_est, temp). None if empty or the wrong station."""
    text = path.read_bytes().decode("utf-8-sig", errors="replace")
    if "Date/Time" not in text[:2000]:
        return None
    frame = pl.read_csv(io.StringIO(text), infer_schema_length=0, truncate_ragged_lines=True)
    cols = {c.strip().strip('"'): c for c in frame.columns}
    name_col = next((cols[c] for c in cols if c.startswith("Station Name")), None)
    time_col = next((cols[c] for c in cols if c.startswith("Date/Time")), None)
    temp_col = next((cols[c] for c in cols if c.startswith("Temp (")), None)
    if not (name_col and time_col and temp_col) or frame.is_empty():
        return None
    names = frame[name_col].drop_nulls().unique().to_list()
    if not names or not all(city_token in n.upper() for n in names):
        raise ValueError(f"{path.name}: station name {names} does not contain {city_token}")
    out = frame.select(
        pl.col(time_col).str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M", strict=False).alias("ts_est"),
        pl.col(temp_col).str.strip_chars().replace("", None).cast(pl.Float64, strict=False).alias("temp"),
    ).drop_nulls("ts_est")
    return out if out["temp"].null_count() < out.height else None


def combine() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Splice stations per city, average cities. Returns (hourly, coverage by city-year).

    A file whose station name does not match its city is dropped and listed,
    never merged: that is the guard against a wrong station ID.
    """
    cities, mismatched = {}, []
    for city, stations in STATIONS.items():
        parts = []
        for rank, st in enumerate(stations):
            for path in sorted(RAW.glob(f"{st.sid}_*.csv")):
                try:
                    parsed = parse_file(path, st.city_token)
                except ValueError as error:
                    mismatched.append(str(error))
                    continue
                if parsed is not None:
                    parts.append(parsed.with_columns(pl.lit(rank).alias("rank")))
        if not parts:
            continue
        cities[city] = (
            pl.concat(parts).drop_nulls("temp").sort("ts_est", "rank")
            .unique("ts_est", keep="first").select("ts_est", pl.col("temp").alias(city))
        )
    if mismatched:
        print(f"WARNING: {len(mismatched)} weather file(s) dropped for a station-name mismatch, e.g.",
              mismatched[0], file=sys.stderr)
    if not cities:
        raise RuntimeError("no usable weather data in " + str(RAW.resolve()))
    hourly = None
    for frame in cities.values():
        hourly = frame if hourly is None else hourly.join(frame, on="ts_est", how="full", coalesce=True)
    names = list(cities)
    hourly = hourly.sort("ts_est").with_columns(
        pl.mean_horizontal(names).alias("temp"),
        pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int8) for c in names]).alias("n_cities"),
    )
    span = pl.DataFrame({"ts_est": pl.datetime_range(
        hourly["ts_est"].min(), hourly["ts_est"].max(), "1h", eager=True)})
    full = span.join(hourly, on="ts_est", how="left")
    # A city whose coverage is poor would make the average jump by a degree or so
    # each time it drops in or out (Ottawa runs colder than Toronto). Drop it.
    weak = [c for c in names if full[c].is_not_null().mean() < MIN_CITY_COVERAGE]
    if weak and len(weak) < len(names):
        print(f"WARNING: dropping {weak} - coverage below {MIN_CITY_COVERAGE:.0%}", file=sys.stderr)
        names = [c for c in names if c not in weak]
        hourly = hourly.drop(weak).with_columns(
            pl.mean_horizontal(names).alias("temp"),
            pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int8) for c in names]).alias("n_cities"),
        )
        full = span.join(hourly, on="ts_est", how="left")
    coverage = (
        full.with_columns(pl.col("ts_est").dt.year().alias("year"))
        .group_by("year").agg([pl.col(c).is_not_null().mean().round(3).alias(c) for c in names + ["temp"]])
        .sort("year")
    )
    return hourly, coverage


def main(skip_download: bool = False) -> int:
    if not skip_download:
        counts = fetch()
        print("weather files:", counts)
        if counts.get("failed", 0) and not counts.get("downloaded", 0) and not counts.get("cached", 0):
            print("every download failed - Track B cannot run", file=sys.stderr)
            return 1
    hourly, coverage = combine()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    hourly.write_parquet(OUT)
    coverage.write_csv(COVERAGE)
    print(f"wrote {OUT}  {hourly.height:,} hours  {hourly['ts_est'].min()} .. {hourly['ts_est'].max()}")
    print(coverage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(skip_download="--skip-download" in sys.argv))
