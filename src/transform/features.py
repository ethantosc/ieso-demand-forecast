#!/usr/bin/env python3
"""Origin-aware features: everything a forecast made at the day-ahead origin may see.

--------------------------------------------------------------------------
THE CLOCK - read this before changing anything below
--------------------------------------------------------------------------
A forecast for delivery day D is made the day before. The IESO day-ahead
forecast files are stamped `CreatedAt` 12:31 on 125 of the 127 archived days,
which is the origin used in the spec: D-1 at about 12:30.

What is known at that moment is demand up to roughly noon on D-1. Whether
`CreatedAt` is EST or local EDT is not stated, so the cutoff is set to the
conservative reading: hours of D-1 that *begin* before 11:00 EST are known,
anything later is not. `CUTOFF_HOUR_EST = 11`. Being an hour too strict can
only make our models look slightly worse; being an hour too lenient would leak.

The consequence the spec missed. Spec section 5 defines naive as y(t-24) and
section 3 lists a 24-hour lag. For a target hour h on day D, t-24 is hour h of
D-1, which is still in the future at the origin whenever h >= 11. So:

  * lag 24 is only legal for target hours 0-10 and is null elsewhere;
  * lag 48, 168 and 336 are legal for every hour (they land on D-2 or earlier);
  * rolling statistics are taken as of the cutoff, not as of the target hour.

`tests/test_features.py` proves this by corrupting every observation at or
after the cutoff and checking that no feature for day D changes.

Days are IESO delivery days: EST dates, 24 hours each, hour-ending 1-24
relabelled 0-23. That is the unit IESO forecasts in and the unit we score in,
and it never has a 23- or 25-hour day. Behavioural calendar features (local
hour, weekday, holidays) are taken from Toronto local time.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

CUTOFF_HOUR_EST = 11          # hours of D-1 beginning before this are known
EST_OFFSET = timedelta(hours=5)
LOCAL_TZ = "America/Toronto"
HDD_CDD_BASE_C = 18.0

CALENDAR_FEATURES = [
    "h", "local_hour", "weekday", "month", "doy_sin", "doy_cos",
    "is_weekend", "is_stat_holiday", "is_observed_holiday",
    "is_day_before_holiday", "is_day_after_holiday",
]
LAG_FEATURES = [
    "lag24_if_known", "lag48", "lag168", "lag336",
    "origin_last_obs", "origin_mean_24h", "origin_mean_168h", "origin_wow_ratio",
]
# Level-free versions of the lags, for models that predict the weekly change.
# A model fed raw levels learns "high level -> expect a fall" from training
# years, then predicts falls that never come once the whole series trends up.
RELATIVE_FEATURES = [
    "rel_d24", "rel_d48", "rel_d336",
    "origin_wow_ratio", "rel_origin_24_168", "rel_origin_last_wow",
]
WEATHER_FEATURES = ["temp", "hdd", "cdd", "temp_day_mean", "temp_day_max", "temp_day_min",
                    "temp_delta_168", "temp_day_mean_delta_7d"]


# ------------------------------------------------------------------ grid


def hourly_grid(staging: pl.DataFrame, excluded_local_dates: list[date] = ()) -> pl.DataFrame:
    """A complete EST hourly grid, so that shifting by k rows is shifting by k hours.

    Missing hours (2025-05-01 00:00 EST) become rows with a null target rather
    than vanishing, which would silently misalign every shift after them.
    Excluded dates (the 2003 blackout window) are nulled the same way: the rows
    stay, the values do not feed any model or any lag.
    """
    frame = staging.select(
        (pl.col("ts_utc") - pl.duration(hours=5)).alias("ts_est"),
        pl.col("market_demand_mw").alias("y_raw"),
        "local_date",
    )
    full = pl.DataFrame({
        "ts_est": pl.datetime_range(frame["ts_est"].min(), frame["ts_est"].max(),
                                    interval="1h", eager=True)
    }).with_columns(pl.col("ts_est").cast(frame.schema["ts_est"]))
    grid = full.join(frame, on="ts_est", how="left")
    excluded = list(excluded_local_dates)
    return grid.with_columns(
        pl.when(pl.col("local_date").is_in(excluded) if excluded else pl.lit(False))
        .then(None).otherwise(pl.col("y_raw")).alias("y")
    ).drop("local_date").sort("ts_est")


# -------------------------------------------------------------- calendar


def _holiday_tables(years: list[int]) -> tuple[set[date], set[date]]:
    import holidays
    stat = holidays.country_holidays("CA", subdiv="ON", years=years)
    every = holidays.country_holidays("CA", subdiv="ON", years=years,
                                      categories=("public", "optional"))
    return set(stat), set(every) - set(stat)


def calendar(grid: pl.DataFrame) -> pl.DataFrame:
    """Calendar columns for each hour. Local time drives behaviour; EST drives the day."""
    local = (
        (pl.col("ts_est") + pl.duration(hours=5))
        .dt.replace_time_zone("UTC").dt.convert_time_zone(LOCAL_TZ).dt.replace_time_zone(None)
    )
    out = grid.with_columns(
        pl.col("ts_est").dt.date().alias("delivery_date"),
        pl.col("ts_est").dt.hour().cast(pl.Int8).alias("h"),
        local.alias("ts_local"),
    ).with_columns(
        pl.col("ts_local").dt.date().alias("local_date"),
        pl.col("ts_local").dt.hour().cast(pl.Int8).alias("local_hour"),
        pl.col("ts_local").dt.weekday().cast(pl.Int8).alias("weekday"),
        pl.col("ts_local").dt.month().cast(pl.Int8).alias("month"),
        pl.col("ts_local").dt.ordinal_day().alias("_doy"),
    )
    years = sorted(set(out["local_date"].dt.year().to_list()))
    stat, observed = _holiday_tables(years)
    stat_l, any_l = list(stat), list(stat | observed)
    before = [d - timedelta(days=1) for d in stat]
    after = [d + timedelta(days=1) for d in stat]
    two_pi = 2 * 3.141592653589793
    return out.with_columns(
        (two_pi * pl.col("_doy") / 365.25).sin().alias("doy_sin"),
        (two_pi * pl.col("_doy") / 365.25).cos().alias("doy_cos"),
        (pl.col("weekday") >= 6).cast(pl.Int8).alias("is_weekend"),
        pl.col("local_date").is_in(stat_l).cast(pl.Int8).alias("is_stat_holiday"),
        (pl.col("local_date").is_in(any_l) & ~pl.col("local_date").is_in(stat_l))
        .cast(pl.Int8).alias("is_observed_holiday"),
        pl.col("local_date").is_in(before).cast(pl.Int8).alias("is_day_before_holiday"),
        pl.col("local_date").is_in(after).cast(pl.Int8).alias("is_day_after_holiday"),
    ).drop("_doy")


# --------------------------------------------------------------- lags


def lag_features(grid: pl.DataFrame) -> pl.DataFrame:
    """Target-relative lags plus origin-level summaries, all legal at the cutoff."""
    g = grid.sort("ts_est").with_columns(
        pl.col("ts_est").dt.date().alias("delivery_date"),
        pl.col("ts_est").dt.hour().alias("_h"),
    ).with_columns(
        pl.when(pl.col("_h") < CUTOFF_HOUR_EST).then(pl.col("y").shift(24))
        .otherwise(None).alias("lag24_if_known"),
        pl.col("y").shift(48).alias("lag48"),
        pl.col("y").shift(168).alias("lag168"),
        pl.col("y").shift(336).alias("lag336"),
        pl.col("y").rolling_mean(24, min_samples=18).alias("_r24"),
        pl.col("y").rolling_mean(168, min_samples=140).alias("_r168"),
        pl.col("y").rolling_mean(CUTOFF_HOUR_EST, min_samples=8).alias("_rpart"),
    )
    last_known = CUTOFF_HOUR_EST - 1
    origin = (
        g.with_columns(pl.col("_rpart").shift(168).alias("_rpart_wk"))
        .with_columns(pl.col("y").shift(168).alias("_y_wk"))
        .filter(pl.col("_h") == last_known)
        .select(
            (pl.col("delivery_date") + pl.duration(days=1)).cast(pl.Date).alias("delivery_date"),
            pl.col("y").alias("origin_last_obs"),
            (pl.col("y") / pl.col("_y_wk")).alias("rel_origin_last_wow"),
            pl.col("_r24").alias("origin_mean_24h"),
            pl.col("_r168").alias("origin_mean_168h"),
            (pl.col("_rpart") / pl.col("_rpart_wk")).alias("origin_wow_ratio"),
        )
    )
    return (
        g.drop("_r24", "_r168", "_rpart", "_h")
        .join(origin, on="delivery_date", how="left")
        .with_columns(
            (pl.col("lag24_if_known") - pl.col("lag168")).alias("rel_d24"),
            (pl.col("lag48") - pl.col("lag168")).alias("rel_d48"),
            (pl.col("lag168") - pl.col("lag336")).alias("rel_d336"),
            (pl.col("origin_mean_24h") / pl.col("origin_mean_168h")).alias("rel_origin_24_168"),
        )
        .sort("ts_est")
    )


# ------------------------------------------------------------- weather


def weather_features(frame: pl.DataFrame, weather: pl.DataFrame | None) -> pl.DataFrame:
    """Join actual temperature. Perfect foresight: Track B only, an upper bound.

    `weather` has columns ts_est and temp (city-mean, degrees C). Daily stats are
    over the EST delivery day, the same day being forecast.
    """
    if weather is None:
        return frame.with_columns([pl.lit(None, dtype=pl.Float64).alias(c) for c in WEATHER_FEATURES])
    w = weather.select("ts_est", "temp").with_columns(
        pl.col("ts_est").cast(frame.schema["ts_est"]),
        pl.col("ts_est").dt.date().alias("_d"),
    )
    daily = w.group_by("_d").agg(
        pl.col("temp").mean().alias("temp_day_mean"),
        pl.col("temp").max().alias("temp_day_max"),
        pl.col("temp").min().alias("temp_day_min"),
    )
    # A join on the date a week earlier, not shift(7): a day with no weather at all
    # would otherwise misalign every row after it.
    week_ago = daily.select((pl.col("_d") + pl.duration(days=7)).cast(pl.Date).alias("_d"),
                            pl.col("temp_day_mean").alias("_wk"))
    daily = daily.join(week_ago, on="_d", how="left").with_columns(
        (pl.col("temp_day_mean") - pl.col("_wk")).alias("temp_day_mean_delta_7d")
    ).drop("_wk")
    # Sort after the last join and before any shift: polars does not promise that
    # a join keeps row order, and shift(168) is only "one week" on a sorted,
    # complete hourly grid.
    return (
        frame.join(w.drop("_d"), on="ts_est", how="left")
        .join(daily, left_on="delivery_date", right_on="_d", how="left")
        .sort("ts_est")
        .with_columns(
            (HDD_CDD_BASE_C - pl.col("temp")).clip(lower_bound=0).alias("hdd"),
            (pl.col("temp") - HDD_CDD_BASE_C).clip(lower_bound=0).alias("cdd"),
            (pl.col("temp") - pl.col("temp").shift(168)).alias("temp_delta_168"),
        )
    )


# ---------------------------------------------------------------- build


def build(staging: pl.DataFrame, excluded_local_dates: list[date] = (),
          weather: pl.DataFrame | None = None) -> pl.DataFrame:
    """One row per EST hour with target, calendar, lag and (optional) weather columns."""
    grid = hourly_grid(staging, excluded_local_dates)
    frame = lag_features(grid).join(
        calendar(grid).drop("y", "y_raw"), on="ts_est", how="left"
    ).with_columns(pl.col("delivery_date").cast(pl.Date))
    if "delivery_date_right" in frame.columns:
        frame = frame.drop("delivery_date_right")
    return weather_features(frame, weather).sort("ts_est")
