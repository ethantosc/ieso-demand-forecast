#!/usr/bin/env python3
"""Scoring: headline metrics, the spec section 5 decomposition, and a significance test.

SIGN CONVENTION. ME = mean(forecast - actual). Positive means over-forecast.
That matches spec section 1, where IESO's Total Load came out +61 MW against
Market Demand. (Spec section 5 describes the MSc thesis errors as "negative =
over-estimate", which is the opposite convention; the thesis used actual minus
forecast. Everything in this repo uses forecast minus actual.)

COMMON SAMPLE. Models are only compared on hours every one of them forecast.
A model that skipped hard hours would otherwise look better for it.
"""

from __future__ import annotations

import math

import polars as pl

SEASONS = {12: "winter", 1: "winter", 2: "winter", 6: "summer", 7: "summer", 8: "summer"}


def common_sample(pred: pl.DataFrame) -> pl.DataFrame:
    """Keep only hours with an actual and a forecast from every model present."""
    pred = pred.filter(pl.col("y").is_not_null() & pl.col("yhat").is_not_null())
    n_models = pred["model"].n_unique()
    keep = pred.group_by("ts_est").agg(pl.len().alias("_n")).filter(pl.col("_n") == n_models)
    return pred.join(keep.select("ts_est"), on="ts_est", how="semi")


def _agg() -> list[pl.Expr]:
    e = pl.col("yhat") - pl.col("y")
    return [
        pl.len().alias("n_hours"),
        e.abs().mean().round(1).alias("MAE"),
        (e.pow(2).mean().sqrt()).round(1).alias("RMSE"),
        (e.abs() / pl.col("y")).mean().mul(100).round(2).alias("MAPE"),
        e.mean().round(1).alias("ME"),
    ]


def summary(pred: pl.DataFrame, by: list[str] = ()) -> pl.DataFrame:
    return pred.group_by(["model", *by]).agg(_agg()).sort([*by, "MAE"])


def with_labels(pred: pl.DataFrame, temp_daily: pl.DataFrame | None = None) -> pl.DataFrame:
    """Attach the decomposition dimensions from spec section 5."""
    out = pred.with_columns(
        (pl.col("h") + 1).alias("hour_ending"),
        pl.col("month").replace_strict(SEASONS, default="shoulder", return_dtype=pl.Utf8).alias("season"),
        pl.when(pl.col("is_stat_holiday") == 1).then(pl.lit("holiday"))
        .when(pl.col("is_weekend") == 1).then(pl.lit("weekend"))
        .otherwise(pl.lit("weekday")).alias("day_type"),
    )
    if temp_daily is None or temp_daily.is_empty():
        return out.with_columns(pl.lit(None, dtype=pl.Utf8).alias("temp_band"))
    days = out.select("delivery_date").unique().join(temp_daily, on="delivery_date", how="left")
    lo, hi = days["temp_day_mean"].quantile(0.05), days["temp_day_mean"].quantile(0.95)
    band = days.with_columns(
        pl.when(pl.col("temp_day_mean") >= hi).then(pl.lit("extreme hot (top 5%)"))
        .when(pl.col("temp_day_mean") <= lo).then(pl.lit("extreme cold (bottom 5%)"))
        .when(pl.col("temp_day_mean").is_null()).then(None)
        .otherwise(pl.lit("normal")).alias("temp_band")
    ).select("delivery_date", "temp_band")
    return out.join(band, on="delivery_date", how="left")


def diebold_mariano(pred: pl.DataFrame, model: str, benchmark: str) -> dict:
    """Is `model` really better than `benchmark`, or is the gap noise?

    Loss is each day's MAE. The test statistic uses a Newey-West variance,
    because daily errors are autocorrelated (a cold snap spans several days)
    and an ordinary t-test would overstate the evidence. Negative mean_diff
    means `model` has the lower error.
    """
    daily = (
        pred.filter(pl.col("model").is_in([model, benchmark]))
        .group_by("delivery_date", "model").agg((pl.col("yhat") - pl.col("y")).abs().mean().alias("loss"))
        .pivot(on="model", index="delivery_date", values="loss").drop_nulls().sort("delivery_date")
    )
    d = (daily[model] - daily[benchmark]).to_list()
    n = len(d)
    if n < 10:
        return {"model": model, "benchmark": benchmark, "n_days": n, "mean_diff": None, "dm_stat": None, "p_value": None}
    mean = sum(d) / n
    lag = int(math.floor(4 * (n / 100) ** (2 / 9)))
    def gamma(k: int) -> float:
        return sum((d[i] - mean) * (d[i - k] - mean) for i in range(k, n)) / n
    var = gamma(0) + 2 * sum((1 - k / (lag + 1)) * gamma(k) for k in range(1, lag + 1))
    stat = mean / math.sqrt(var / n) if var > 0 else float("nan")
    p = math.erfc(abs(stat) / math.sqrt(2)) if stat == stat else float("nan")
    return {"model": model, "benchmark": benchmark, "n_days": n, "mean_diff": round(mean, 1),
            "dm_stat": round(stat, 2), "p_value": float(f"{p:.2g}") if p == p else None}
