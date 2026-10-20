#!/usr/bin/env python3
"""Rung 5: one global LightGBM model across all 24 delivery hours (spec section 5).

Rolling origin with a monthly refit. For every month in an evaluation window
the model is retrained on target days up to two days before the month starts -
the last days fully known at the first origin of that month - then used for
every day of the month, each day with features taken as of its own origin.
Refitting daily would change little and cost thirty times as much; the
cadence is a stated choice, not a shortcut around the origin.

Two training windows, as spec section 5 asks: a rolling three years, and an
expanding window from 2003 (the first full year of market data). Two tracks:
A uses calendar and lag features only and is the real forecast; B adds
observed temperature and is an upper bound, never a forecast (spec section 4).

TARGET. The trees predict y - lag168, the change from the same hour last week,
and the forecast is lag168 plus that change. A tree cannot predict a value
outside the range of targets it was trained on, and Ontario demand is rising:
the 2026 peak was the highest since 2011. Trained on levels, the model would
cap its forecasts at historical values exactly when demand sets new ones.
Trained on weekly changes, the level comes from the lag and only the change has
to be learned. For the same reason the inputs are level-free too (differences
and ratios between lags, see RELATIVE_FEATURES): with raw levels as inputs, a
synthetic test with a built-in upward trend produced a -90 MW bias in every
month, because the trees had learned that a high level means a fall is coming.
Both choices rest on that structural argument and were fixed before any real
result existed, not picked by comparing scores.

Hyperparameters are fixed below and were not tuned on any evaluation window.
Tuning them on the test years would be the same kind of look-ahead the feature
layer exists to prevent.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from src.transform.features import CALENDAR_FEATURES, RELATIVE_FEATURES, WEATHER_FEATURES

EXPANDING_START = date(2003, 1, 1)
PARAMS = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 0,
}
NUM_ROUNDS = 600


def feature_set(track: str) -> list[str]:
    """A: calendar and lags. B and C add the weather features - observed for B,
    legal archived forecasts for C. Which weather a model sees is decided by the
    frame it is given, not by the column names, which are the same."""
    base = CALENDAR_FEATURES + RELATIVE_FEATURES
    return base + WEATHER_FEATURES if track in ("B", "C") else base


def _shift_years(day: date, years: int) -> date:
    try:
        return day.replace(year=day.year - years)
    except ValueError:                       # 29 February
        return day.replace(year=day.year - years, day=28)


def _training_start(refit: date, window: str, train_start: date | None) -> date:
    if window == "rolling_3y":
        return _shift_years(refit, 3)
    if window == "since":
        if train_start is None:
            raise ValueError("window='since' needs train_start")
        return train_start
    return EXPANDING_START


def _monthly_fits(frame: pl.DataFrame, days: list[date], track: str, window: str,
                  train_start: date | None, min_train_days: int):
    """Yield (month_days, training rows, target rows) for each monthly refit.

    Training rows are target days up to two days before the month starts: the
    last days fully known at the month's first origin.
    """
    by_month: dict[tuple[int, int], list[date]] = {}
    for day in sorted(days):
        by_month.setdefault((day.year, day.month), []).append(day)
    for month_days in by_month.values():
        refit = month_days[0]
        train = frame.filter(
            pl.col("delivery_date").is_between(_training_start(refit, window, train_start),
                                               refit - timedelta(days=2))
            & pl.col("y").is_not_null() & pl.col("lag168").is_not_null()
        )
        if track in ("B", "C"):
            train = train.filter(pl.col("temp").is_not_null())
        if train.height < 24 * min_train_days:
            continue
        yield month_days, train, frame.filter(pl.col("delivery_date").is_in(month_days))


def _train(train: pl.DataFrame, cols: list[str], n_jobs: int, quantile: float | None = None):
    import lightgbm as lgb

    params = {**PARAMS, "num_threads": n_jobs if n_jobs > 0 else 0}
    if quantile is not None:
        params.update(objective="quantile", alpha=quantile)
    return lgb.train(
        params,
        lgb.Dataset(train.select(cols).to_numpy(), label=(train["y"] - train["lag168"]).to_numpy(),
                    feature_name=cols),
        num_boost_round=NUM_ROUNDS,
    )


def forecast(frame: pl.DataFrame, days: list[date], track: str, window: str,
             n_jobs: int = -1, *, train_start: date | None = None,
             min_train_days: int = 60) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Returns (predictions: ts_est, yhat) and (gain importance from the last refit).

    window: "rolling_3y", "expanding" (from 2003), or "since" (from train_start).
    """
    cols = feature_set(track)
    preds, importance = [], None
    for _, train, target in _monthly_fits(frame, days, track, window, train_start, min_train_days):
        model = _train(train, cols, n_jobs)
        change = pl.Series("change", model.predict(target.select(cols).to_numpy()))
        preds.append(target.select("ts_est", "lag168").with_columns(change)
                     .select("ts_est", (pl.col("lag168") + pl.col("change")).alias("yhat")))
        importance = pl.DataFrame({"feature": cols, "gain": model.feature_importance("gain")})
    if not preds:
        return pl.DataFrame(schema={"ts_est": frame.schema["ts_est"], "yhat": pl.Float64}), pl.DataFrame()
    gain = importance.with_columns((pl.col("gain") / pl.col("gain").sum() * 100).round(1).alias("gain_pct"))
    return pl.concat(preds), gain.sort("gain_pct", descending=True).select("feature", "gain_pct")


def forecast_quantiles(frame: pl.DataFrame, days: list[date], track: str, window: str,
                       quantiles: tuple[float, ...], n_jobs: int = -1, *,
                       train_start: date | None = None, min_train_days: int = 60) -> pl.DataFrame:
    """One quantile model per q at every monthly refit. Returns ts_est, q, yhat.

    Separately fitted quantiles can cross (q05 above q25 for some hour). They are
    sorted within each hour, which keeps each hour's set of values and only
    reassigns them to quantile levels in order.
    """
    cols = feature_set(track)
    parts = []
    for _, train, target in _monthly_fits(frame, days, track, window, train_start, min_train_days):
        x = target.select(cols).to_numpy()
        base = target.select("ts_est", "lag168")
        for q in quantiles:
            change = _train(train, cols, n_jobs, quantile=q).predict(x)
            parts.append(base.with_columns(pl.Series("change", change), pl.lit(q).alias("q"))
                         .select("ts_est", "q", (pl.col("lag168") + pl.col("change")).alias("yhat")))
    if not parts:
        return pl.DataFrame(schema={"ts_est": frame.schema["ts_est"], "q": pl.Float64, "yhat": pl.Float64})
    long = pl.concat(parts)
    ordered = (long.sort("ts_est", "q")
               .with_columns(pl.col("yhat").sort().over("ts_est").alias("yhat")))
    return ordered
