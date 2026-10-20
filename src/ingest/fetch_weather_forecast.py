#!/usr/bin/env python3
"""Archived GFS temperature forecasts from Open-Meteo, for Track C (spec-v2 section 3).

    python -m src.ingest.fetch_weather_forecast             # probe, download, verify, build
    python -m src.ingest.fetch_weather_forecast --skip-download

Writes
    data/raw/weather_fc/{model}_{city}_{year}.json     raw API responses (cached)
    data/staging/weather_fc_hourly.parquet             ts_est, temp (city mean), per city
    data/staging/weather_fc_meta.json                  model id, rule, verification, start month

WHICH FORECAST IS LEGAL. `temperature_2m_previous_dayN` at valid time t holds
the value from the model run started about 24*N hours before t. GFS runs every
six hours. At the day-ahead origin (D-1, cutoff 11:00 EST = 16:00 UTC) the
latest run that has been published is D-1 06Z; the 12Z run appears from about
16-18Z and is not counted. So:

    delivery-day EST hours 0-6   -> previous_day1  (runs D-1 00Z/06Z, lead 24-29 h)
    delivery-day EST hours 7-23  -> previous_day2  (runs D-2 12Z .. D-1 00Z, lead 48-53 h)

`is_legal()` encodes that check and tests/test_weather_forecast.py runs it on
every hour. The "about 24*N hours" part comes from Open-Meteo's documentation,
not from a guarantee, so `verify()` checks it against the Single Runs API,
which serves individual runs by start time: each archived day1/day2 value must
match the run the rule predicts. If it does not, this script exits non-zero
and Track C does not run. If the Single Runs API has no GFS runs, every hour
falls back to previous_day2, which is legal by the documentation's own
definition (predicted at least 48 hours ahead).

WHY GFS GLOBAL, NOT SEAMLESS. Seamless blends HRRR for short lead times, and
HRRR's archive starts at a different date, so the history would change
character partway through. The model ID is not documented for this API, so a
short list of candidates is probed and the one with the longest archive wins.

Data: Open-Meteo, CC BY 4.0. Free for non-commercial use.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

from src.transform.features import CUTOFF_HOUR_EST

PREVIOUS_RUNS = "https://previous-runs-api.open-meteo.com/v1/forecast"
SINGLE_RUNS = "https://single-runs-api.open-meteo.com/v1/forecast"
MODEL_CANDIDATES = ("ncep_gfs_global", "gfs_global", "ncep_gfs013", "ncep_gfs025")
CITIES = {                       # the same three airports as the observed weather
    "toronto": (43.677, -79.631),
    "ottawa": (45.322, -75.669),
    "london": (43.033, -81.151),
}
RAW = Path("data/raw/weather_fc")
OUT = Path("data/staging/weather_fc_hourly.parquet")
META = Path("data/staging/weather_fc_meta.json")
FIRST_YEAR = 2021

CYCLE_HOURS = 6
PUBLISH_DELAY_HOURS = 5          # a run is usable this long after it starts (Open-Meteo: 4-6 h)
DAY1_MAX_EST_HOUR = 6
MIN_MONTH_COVERAGE = 0.95
VERIFY_DATES = (date(2026, 5, 12), date(2026, 6, 16), date(2026, 7, 21),
                date(2026, 8, 25), date(2026, 9, 29))
VERIFY_TOLERANCE_C = 0.15
VERIFY_PASS_SHARE = 0.95
EXPECTED_LEADS = {1: range(24, 30), 2: range(48, 54)}

USER_AGENT = "ieso-demand-forecast/2.0 (personal research project)"
TIMEOUT = 60
RETRIES = 3
PAUSE = 1.0


# ---------------------------------------------------------------- the rule


def legal_day(est_hour: int) -> int:
    """Which previous_dayN to use for a delivery-day hour (EST)."""
    return 1 if est_hour <= DAY1_MAX_EST_HOUR else 2


def implied_run(t_utc: datetime, n: int) -> datetime:
    """The run that previous_dayN reports at valid time t: started ~24*N h earlier, on a 6 h cycle."""
    start = t_utc - timedelta(hours=24 * n)
    return start.replace(hour=start.hour - start.hour % CYCLE_HOURS, minute=0, second=0, microsecond=0)


def is_legal(t_est: datetime, n: int) -> bool:
    """Was that run published before the origin of t's delivery day?"""
    day = t_est.date()
    origin_utc = datetime.combine(day - timedelta(days=1), datetime.min.time()) + timedelta(
        hours=CUTOFF_HOUR_EST + 5)
    run = implied_run(t_est + timedelta(hours=5), n)
    return run + timedelta(hours=PUBLISH_DELAY_HOURS) <= origin_utc


# ---------------------------------------------------------------- http


def _get_json(url: str) -> dict | list:
    """GET a JSON document. Open-Meteo reports bad requests as HTTP 400 with a reason."""
    last: Exception | None = None
    for attempt in range(RETRIES):
        if attempt:
            time.sleep(3 * attempt)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            if 400 <= error.code < 500:                    # a real refusal: do not retry
                try:
                    reason = json.loads(body).get("reason", body)
                except json.JSONDecodeError:
                    reason = body
                raise ValueError(f"HTTP {error.code}: {reason}") from error
            last = error
        except (urllib.error.URLError, TimeoutError) as error:
            last = error
    raise ConnectionError(f"could not reach {url.split('?')[0]}: {last}")


def _url(base: str, **params) -> str:
    return base + "?" + urllib.parse.urlencode(params, safe=",:")


def _series(doc: dict, variable: str) -> pl.DataFrame:
    hourly = doc.get("hourly") or {}
    times, values = hourly.get("time", []), hourly.get(variable, [])
    return pl.DataFrame({"ts_utc": times, variable: values},
                        schema={"ts_utc": pl.Utf8, variable: pl.Float64}).with_columns(
        pl.col("ts_utc").str.strptime(pl.Datetime("us"), "%Y-%m-%dT%H:%M"))


# ---------------------------------------------------------------- probe


def probe_model() -> tuple[str, dict]:
    """Pick the GFS model ID with the longest archive. Returns (model, probe log)."""
    lat, lon = CITIES["toronto"]
    log: dict[str, dict] = {}
    for model in MODEL_CANDIDATES:
        entry = {}
        for label, day in (("2021-04", date(2021, 4, 1)), ("2024-07", date(2024, 7, 1))):
            try:
                doc = _get_json(_url(PREVIOUS_RUNS, latitude=lat, longitude=lon, models=model,
                                     hourly="temperature_2m_previous_day1", timezone="GMT",
                                     start_date=day.isoformat(), end_date=(day + timedelta(days=1)).isoformat()))
                s = _series(doc, "temperature_2m_previous_day1")
                entry[label] = int(s["temperature_2m_previous_day1"].is_not_null().sum())
            except ValueError as error:
                entry[label] = f"refused: {error}"
            time.sleep(PAUSE)
        log[model] = entry
    def has(entry, label):
        return isinstance(entry.get(label), int) and entry[label] > 0
    for label in ("2021-04", "2024-07"):
        for model in MODEL_CANDIDATES:
            if has(log[model], label):
                return model, log
    raise RuntimeError(f"no GFS model ID returned data: {log}")


# ---------------------------------------------------------------- download


def _raw_path(model: str, city: str, year: int) -> Path:
    return RAW / f"{model}_{city}_{year}.json"


def fetch(model: str, today: date | None = None) -> dict[str, int]:
    """One request per city per year; settled years are cached, the current year is refreshed."""
    today = today or date.today()
    RAW.mkdir(parents=True, exist_ok=True)
    counts = {"downloaded": 0, "cached": 0}
    for city, (lat, lon) in CITIES.items():
        for year in range(FIRST_YEAR, today.year + 1):
            path = _raw_path(model, city, year)
            if path.exists() and year < today.year:
                counts["cached"] += 1
                continue
            end = min(date(year, 12, 31), today)
            doc = _get_json(_url(PREVIOUS_RUNS, latitude=lat, longitude=lon, models=model,
                                 hourly="temperature_2m_previous_day1,temperature_2m_previous_day2",
                                 timezone="GMT", start_date=f"{year}-01-01", end_date=end.isoformat()))
            path.write_text(json.dumps(doc))
            counts["downloaded"] += 1
            print(f"  forecast {city} {year}: "
                  f"{sum(v is not None for v in doc['hourly']['temperature_2m_previous_day2'])} hours",
                  flush=True)
            time.sleep(PAUSE)
    return counts


def _city_frame(model: str, city: str) -> pl.DataFrame | None:
    parts = []
    for path in sorted(RAW.glob(f"{model}_{city}_*.json")):
        doc = json.loads(path.read_text())
        d1 = _series(doc, "temperature_2m_previous_day1")
        d2 = _series(doc, "temperature_2m_previous_day2")
        parts.append(d1.join(d2, on="ts_utc", how="full", coalesce=True))
    if not parts:
        return None
    return (pl.concat(parts).unique("ts_utc", keep="last").sort("ts_utc")
            .rename({"temperature_2m_previous_day1": "day1", "temperature_2m_previous_day2": "day2"}))


# ---------------------------------------------------------------- verify


def verify(model: str, dates: tuple[date, ...] = VERIFY_DATES) -> dict:
    """Check that day1/day2 values come from the runs the rule says they do.

    For each sample date X, the four runs of X are fetched individually. Every
    archived day1 value on X+1 and day2 value on X+2 (Toronto) is matched
    against those runs; the lead time of the matching run is recorded. The
    rule predicts leads of 24-29 h for day1 and 48-53 h for day2.
    """
    lat, lon = CITIES["toronto"]
    archived = _city_frame(model, "toronto")
    if archived is None:
        return {"verdict": "unavailable", "reason": "no archived Toronto forecasts on disk"}
    runs: dict[datetime, dict[datetime, float]] = {}
    refused = []
    for day in dates:
        for hour in range(0, 24, CYCLE_HOURS):
            run = datetime.combine(day, datetime.min.time()) + timedelta(hours=hour)
            try:
                doc = _get_json(_url(SINGLE_RUNS, latitude=lat, longitude=lon, models=model,
                                     hourly="temperature_2m", timezone="GMT", forecast_days=4,
                                     run=run.strftime("%Y-%m-%dT%H:%M")))
            except (ValueError, ConnectionError) as error:
                refused.append(f"{run:%Y-%m-%d %HZ}: {error}")
                continue
            s = _series(doc, "temperature_2m").drop_nulls()
            runs[run] = dict(zip(s["ts_utc"].to_list(), s["temperature_2m"].to_list()))
            time.sleep(PAUSE)
    if not runs:
        return {"verdict": "unavailable", "reason": "Single Runs API returned no GFS runs",
                "examples": refused[:3]}
    return score_matches(archived, runs, dates, refused)


def score_matches(archived: pl.DataFrame, runs: dict[datetime, dict[datetime, float]],
                  dates: tuple[date, ...], refused: list[str] | None = None) -> dict:
    """Pure scoring step, separated so it can be tested without a network."""
    lookup = {r["ts_utc"]: r for r in archived.iter_rows(named=True)}
    result = {}
    for n, column in ((1, "day1"), (2, "day2")):
        leads: dict[str, int] = {}
        tested = expected = 0
        for day in dates:
            target_day = day + timedelta(days=n)
            for hour in range(24):
                t = datetime.combine(target_day, datetime.min.time()) + timedelta(hours=hour)
                value = (lookup.get(t) or {}).get(column)
                candidates = {run: vals[t] for run, vals in runs.items() if t in vals}
                if value is None or not candidates:
                    continue
                tested += 1
                matched = sorted(int((t - run).total_seconds() // 3600) for run, v in candidates.items()
                                 if abs(v - value) <= VERIFY_TOLERANCE_C)
                if any(lead in EXPECTED_LEADS[n] for lead in matched):
                    expected += 1
                key = ",".join(map(str, matched)) if matched else "none"
                leads[key] = leads.get(key, 0) + 1
        share = expected / tested if tested else 0.0
        result[f"day{n}"] = {"tested": tested, "matched_expected_lead": expected,
                             "share": round(share, 3), "matching_leads": dict(sorted(leads.items()))}
    ok = all(result[k]["tested"] >= 24 and result[k]["share"] >= VERIFY_PASS_SHARE for k in ("day1", "day2"))
    return {"verdict": "pass" if ok else "fail", **result, "runs_fetched": len(runs),
            "runs_refused": len(refused or [])}


# ---------------------------------------------------------------- build


def combine(model: str, rule: str) -> tuple[pl.DataFrame, pl.DataFrame, str | None]:
    """City-mean hourly forecast series under the legality rule.

    Returns (hourly frame on ts_est, monthly coverage, first month with >= 95% coverage).
    """
    cities = {}
    for city in CITIES:
        frame = _city_frame(model, city)
        if frame is None:
            continue
        frame = frame.with_columns((pl.col("ts_utc") - pl.duration(hours=5)).alias("ts_est"))
        if rule == "day1_day2":
            value = pl.when(pl.col("ts_est").dt.hour() <= DAY1_MAX_EST_HOUR).then(pl.col("day1")).otherwise(pl.col("day2"))
        else:
            value = pl.col("day2")
        cities[city] = frame.select("ts_est", value.alias(city))
    if not cities:
        raise RuntimeError(f"no forecast files under {RAW.resolve()}")
    hourly = None
    for frame in cities.values():
        hourly = frame if hourly is None else hourly.join(frame, on="ts_est", how="full", coalesce=True)
    names = list(cities)
    hourly = hourly.sort("ts_est").with_columns(
        pl.mean_horizontal(names).alias("temp"),
        pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int8) for c in names]).alias("n_cities"),
    )
    span = pl.DataFrame({"ts_est": pl.datetime_range(hourly["ts_est"].min(), hourly["ts_est"].max(),
                                                     "1h", eager=True)})
    full = span.join(hourly, on="ts_est", how="left")
    coverage = (full.with_columns(pl.col("ts_est").dt.strftime("%Y-%m").alias("month"))
                .group_by("month").agg(pl.col("temp").is_not_null().mean().round(3).alias("coverage"))
                .sort("month"))
    good = coverage.filter(pl.col("coverage") >= MIN_MONTH_COVERAGE)
    first = good["month"][0] if good.height else None
    return hourly.drop_nulls("temp"), coverage, first


def main(skip_download: bool = False, skip_verify: bool = False) -> int:
    meta: dict = {}
    if skip_download:
        model = next((m for m in MODEL_CANDIDATES if list(RAW.glob(f"{m}_*.json"))), None)
        if model is None:
            print(f"no cached forecast files under {RAW.resolve()}", file=sys.stderr)
            return 1
        meta["probe"] = "skipped (cached files)"
    else:
        model, probe = probe_model()
        meta["probe"] = probe
        print("forecast model:", model, "| probe:", probe, flush=True)
        meta["files"] = fetch(model)
    meta["model"] = model

    if skip_verify:
        check = {"verdict": "skipped"}
    else:
        check = verify(model)
    meta["verification"] = check
    print("verification:", json.dumps(check, indent=1), flush=True)
    if check["verdict"] == "fail":
        META.parent.mkdir(parents=True, exist_ok=True)
        META.write_text(json.dumps(meta, indent=1, default=str))
        print("VERIFICATION FAILED: the archived values do not come from the runs the legality rule "
              "assumes. Track C must not run until the rule is corrected.", file=sys.stderr)
        return 2
    rule = "day1_day2" if check["verdict"] in ("pass", "skipped") else "day2_only"
    meta["rule"] = rule

    hourly, coverage, first = combine(model, rule)
    meta["first_full_month"] = first
    OUT.parent.mkdir(parents=True, exist_ok=True)
    hourly.write_parquet(OUT)
    coverage.write_csv(OUT.with_name("weather_fc_coverage.csv"))
    META.write_text(json.dumps(meta, indent=1, default=str))
    print(f"wrote {OUT}  {hourly.height:,} hours  rule={rule}  first full month={first}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(skip_download="--skip-download" in sys.argv,
                          skip_verify="--skip-verify" in sys.argv))
