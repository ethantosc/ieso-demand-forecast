#!/usr/bin/env python3
"""The probabilistic baseline (spec-v2 section 6).

A quantile model has to beat something simple too. The baseline takes the
point forecast of the threshold model (the better of the two naive forecasts,
chosen on the development window) and adds the empirical quantiles of that
model's own errors over the 365 days before each monthly refit, separately for
each EST delivery hour. Like every model here it refits monthly and uses only
days fully known at the month's first origin.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
LOOKBACK_DAYS = 365


def point_forecast(frame: pl.DataFrame, base: str) -> pl.Expr:
    if base == "naive":
        return pl.coalesce("lag24_if_known", "lag48")
    if base == "seasonal_naive":
        return pl.col("lag168")
    raise ValueError(base)


def empirical_quantiles(frame: pl.DataFrame, days: list[date], base: str,
                        quantiles: tuple[float, ...] = QUANTILES) -> pl.DataFrame:
    """Returns ts_est, q, yhat for the requested delivery days."""
    with_pred = frame.with_columns(point_forecast(frame, base).alias("_pred"))
    by_month: dict[tuple[int, int], list[date]] = {}
    for day in sorted(days):
        by_month.setdefault((day.year, day.month), []).append(day)
    parts = []
    for month_days in by_month.values():
        refit = month_days[0]
        end = refit - timedelta(days=2)
        history = with_pred.filter(
            pl.col("delivery_date").is_between(end - timedelta(days=LOOKBACK_DAYS - 1), end)
            & pl.col("y").is_not_null() & pl.col("_pred").is_not_null()
        )
        if history.height < 24 * 60:
            continue
        offsets = history.group_by("h").agg(
            [(pl.col("y") - pl.col("_pred")).quantile(q, interpolation="linear").alias(str(q)) for q in quantiles]
        )
        target = with_pred.filter(pl.col("delivery_date").is_in(month_days)).join(offsets, on="h", how="left")
        for q in quantiles:
            parts.append(target.select("ts_est", pl.lit(q).alias("q"),
                                       (pl.col("_pred") + pl.col(str(q))).alias("yhat")))
    if not parts:
        return pl.DataFrame(schema={"ts_est": frame.schema["ts_est"], "q": pl.Float64, "yhat": pl.Float64})
    return pl.concat(parts).sort("ts_est", "q")
