"""The Track C legality rule and its verification (spec-v2 section 3)."""

import json
from datetime import date, datetime, timedelta

import polars as pl

from src.ingest import fetch_weather_forecast as fwf

DAY = date(2026, 7, 15)


def _est(hour: int) -> datetime:
    return datetime.combine(DAY, datetime.min.time()) + timedelta(hours=hour)


def test_chosen_forecast_is_published_before_the_origin_for_every_hour():
    for hour in range(24):
        assert fwf.is_legal(_est(hour), fwf.legal_day(hour)), f"hour {hour} would leak"


def test_day1_really_would_leak_from_07_est():
    """The rule is tight: day1 is legal up to 06:00 EST and not one hour later."""
    assert all(fwf.is_legal(_est(h), 1) for h in range(0, 7))
    assert not any(fwf.is_legal(_est(h), 1) for h in range(7, 24))
    assert all(fwf.is_legal(_est(h), 2) for h in range(24))


def test_implied_run():
    assert fwf.implied_run(datetime(2026, 7, 2, 5), 1) == datetime(2026, 7, 1, 0)
    assert fwf.implied_run(datetime(2026, 7, 2, 11), 1) == datetime(2026, 7, 1, 6)
    assert fwf.implied_run(datetime(2026, 7, 2, 12), 1) == datetime(2026, 7, 1, 12)
    assert fwf.implied_run(datetime(2026, 7, 3, 4), 2) == datetime(2026, 7, 1, 0)


def _runs_for(dates):
    """Fake single runs: every run has its own distinct value at every valid time."""
    runs = {}
    for day in dates:
        for hour in (0, 6, 12, 18):
            run = datetime.combine(day, datetime.min.time()) + timedelta(hours=hour)
            runs[run] = {run + timedelta(hours=k): round(run.day * 10 + hour + k * 0.37, 2) for k in range(96)}
    return runs


def _archive(runs, lead_shift: int = 0) -> pl.DataFrame:
    """Archive built the way the rule assumes (shift 0) or misaligned by `lead_shift` hours."""
    rows = {}
    for n, col in ((1, "day1"), (2, "day2")):
        for vals in runs.values():
            for t in vals:
                run = fwf.implied_run(t + timedelta(hours=lead_shift), n)
                if run in runs and t in runs[run]:
                    rows.setdefault(t, {})[col] = runs[run][t]
    return pl.DataFrame([{"ts_utc": t, "day1": r.get("day1"), "day2": r.get("day2")}
                         for t, r in sorted(rows.items())])


def test_verification_passes_when_archive_matches_the_rule():
    dates = (date(2026, 5, 12), date(2026, 6, 16))
    runs = _runs_for(dates)
    result = fwf.score_matches(_archive(runs), runs, dates)
    assert result["verdict"] == "pass", result
    assert result["day1"]["share"] == 1.0 and result["day2"]["share"] == 1.0


def test_verification_fails_when_archive_is_one_cycle_fresher():
    """If day1 actually came from runs 6 h later than assumed, early hours would leak - must fail."""
    dates = (date(2026, 5, 12), date(2026, 6, 16))
    runs = _runs_for(dates)
    result = fwf.score_matches(_archive(runs, lead_shift=6), runs, dates)
    assert result["verdict"] == "fail", result


def test_combine_applies_day1_before_07_est_and_day2_after(tmp_path, monkeypatch):
    monkeypatch.setattr(fwf, "RAW", tmp_path)
    times = [datetime(2026, 7, 1) + timedelta(hours=i) for i in range(72)]
    for city in fwf.CITIES:
        doc = {"hourly": {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
                          "temperature_2m_previous_day1": [1.0] * len(times),
                          "temperature_2m_previous_day2": [2.0] * len(times)}}
        (tmp_path / f"m_{city}_2026.json").write_text(json.dumps(doc))
    hourly, _, _ = fwf.combine("m", "day1_day2")
    by_hour = dict(hourly.group_by(pl.col("ts_est").dt.hour().alias("h")).agg(pl.col("temp").first()).iter_rows())
    assert all(by_hour[h] == 1.0 for h in range(0, 7))
    assert all(by_hour[h] == 2.0 for h in range(7, 24))
    safe, _, _ = fwf.combine("m", "day2_only")
    assert safe["temp"].unique().to_list() == [2.0]


def test_verify_end_to_end_with_a_mocked_api(tmp_path, monkeypatch):
    """verify() builds the right requests and reads the right fields, not just the scoring."""
    import urllib.parse

    monkeypatch.setattr(fwf, "RAW", tmp_path)
    monkeypatch.setattr(fwf, "PAUSE", 0)
    dates = (date(2026, 5, 12), date(2026, 6, 16))
    runs = _runs_for(dates)
    archive = _archive(runs)
    doc = {"hourly": {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in archive["ts_utc"]],
                      "temperature_2m_previous_day1": archive["day1"].to_list(),
                      "temperature_2m_previous_day2": archive["day2"].to_list()}}
    (tmp_path / "m_toronto_2026.json").write_text(json.dumps(doc))
    requested = []

    def fake_get_json(url):
        params = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
        requested.append(params)
        assert url.startswith(fwf.SINGLE_RUNS) and params["models"] == "m" and params["timezone"] == "GMT"
        run = datetime.strptime(params["run"], "%Y-%m-%dT%H:%M")
        series = runs[run]
        return {"hourly": {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in series],
                           "temperature_2m": list(series.values())}}

    monkeypatch.setattr(fwf, "_get_json", fake_get_json)
    result = fwf.verify("m", dates)
    assert result["verdict"] == "pass", result
    assert len(requested) == 8 and {p["run"][-5:] for p in requested} == {"00:00", "06:00", "12:00", "18:00"}
