"""The leakage guard. If this fails, every result downstream is optimistic.

Run from the repo root:  python -m pytest tests -q
"""

from datetime import date, datetime, timedelta

import polars as pl
import pytest

from src.transform import features as F


TARGET_DAY = date(2025, 3, 15)


def _cutoff(day: date) -> datetime:
    return datetime.combine(day - timedelta(days=1), datetime.min.time()) + timedelta(hours=F.CUTOFF_HOUR_EST)


def _features_for(staging: pl.DataFrame, day: date) -> pl.DataFrame:
    cols = list(dict.fromkeys(["ts_est"] + F.CALENDAR_FEATURES + F.LAG_FEATURES + F.RELATIVE_FEATURES))
    return F.build(staging).filter(pl.col("delivery_date") == day).select(cols).sort("ts_est")


def _corrupt(staging: pl.DataFrame, from_est: datetime, value: float) -> pl.DataFrame:
    boundary = from_est + timedelta(hours=5)
    return staging.with_columns(
        pl.when(pl.col("ts_utc") >= boundary).then(value)
        .otherwise(pl.col("market_demand_mw")).alias("market_demand_mw")
    )


def test_nothing_after_the_cutoff_reaches_day_d(make_staging):
    clean = make_staging()
    before = _features_for(clean, TARGET_DAY)
    after = _features_for(_corrupt(clean, _cutoff(TARGET_DAY), 999_999.0), TARGET_DAY)
    assert before.height == 24
    assert before.equals(after), "a feature for day D changed when only post-cutoff data changed"


def test_the_last_pre_cutoff_hour_is_used(make_staging):
    """Control: corrupting from one hour earlier must change something, or the test above proves nothing."""
    clean = make_staging()
    before = _features_for(clean, TARGET_DAY)
    after = _features_for(_corrupt(clean, _cutoff(TARGET_DAY) - timedelta(hours=1), 999_999.0), TARGET_DAY)
    assert not before.equals(after)
    assert after["origin_last_obs"].unique().to_list() == [999_999.0]


def test_lag24_only_where_legal(make_staging):
    frame = F.build(make_staging()).filter(pl.col("delivery_date") == TARGET_DAY)
    legal = frame.filter(pl.col("h") < F.CUTOFF_HOUR_EST)["lag24_if_known"]
    illegal = frame.filter(pl.col("h") >= F.CUTOFF_HOUR_EST)["lag24_if_known"]
    assert legal.null_count() == 0
    assert illegal.null_count() == illegal.len()


def test_missing_hour_does_not_misalign_shifts(make_staging):
    """Drop one hour; lag168 a week later must still point at the same clock hour."""
    staging = make_staging()
    hole = datetime(2025, 3, 1, 7) + timedelta(hours=5)
    gappy = staging.filter(pl.col("ts_utc") != hole)
    frame = F.build(gappy)
    week_later = frame.filter(pl.col("ts_est") == datetime(2025, 3, 8, 7))
    assert week_later["lag168"].item() is None
    next_hour = frame.filter(pl.col("ts_est") == datetime(2025, 3, 8, 8))
    expected = staging.filter(pl.col("ts_utc") == datetime(2025, 3, 1, 13))["market_demand_mw"].item()
    assert next_hour["lag168"].item() == pytest.approx(expected)


def test_excluded_dates_feed_nothing(make_staging):
    staging = make_staging()
    frame = F.build(staging, excluded_local_dates=[date(2025, 3, 1)])
    assert frame.filter(pl.col("ts_est") == datetime(2025, 3, 1, 12))["y"].item() is None
    assert frame.filter(pl.col("ts_est") == datetime(2025, 3, 8, 12))["lag168"].item() is None


def test_weather_week_deltas_survive_a_missing_day(make_staging):
    """Temperature = hours since start, so every one-week delta must be exactly 168."""
    staging = make_staging()
    start = datetime(2025, 2, 1)
    hours = [start + timedelta(hours=i) for i in range(60 * 24)]
    weather = pl.DataFrame({"ts_est": hours, "temp": [float(i) for i in range(len(hours))]})
    weather = weather.filter(pl.col("ts_est").dt.date() != date(2025, 3, 3))   # a whole day missing
    frame = F.build(staging, weather=weather).filter(pl.col("delivery_date") == date(2025, 3, 15))
    assert frame["temp_delta_168"].to_list() == [168.0] * 24
    assert frame["temp_day_mean_delta_7d"].unique().to_list() == [168.0]
