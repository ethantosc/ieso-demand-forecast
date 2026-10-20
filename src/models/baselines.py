#!/usr/bin/env python3
"""Rungs 1-4 of the baseline ladder (spec section 5).

Every function returns (ts_est, yhat) for the hours of the requested delivery
days, using only what was known at that day's origin.

Rung 1, naive. The spec wrote y(t-24). Under the 12:30 origin that value does
not exist yet for target hours from 11:00 on, so the legal naive is "the most
recent observed value at the same clock hour": D-1 for early hours, D-2 for the
rest. `naive_leaky` keeps the spec's original definition only as a diagnostic,
to measure what that leak would have been worth. It is never a contender.

Rung 2, seasonal naive. y(t-168): the same hour last week. Always legal. This
is the threshold every model has to beat.

Rungs 3-4, classical. statsforecast MSTL with daily (24) and weekly (168)
seasonality, refitted at every origin on the trailing eight weeks:
  * MSTL + AutoETS on the deseasonalised remainder (rung 3, ETS/decomposition)
  * MSTL + AutoARIMA on the remainder (rung 4). This is ARIMA after removing
    multiple seasonality, not a seasonal SARIMA: a period-168 SARIMA does not
    fit at every origin in a reasonable time, and MSTL handles the 24 and 168
    cycles that a single-period SARIMA cannot hold together anyway.
A missing hour inside a training window is linearly interpolated for these two
models only; they cannot take gaps. The trees and the naive rungs see the gap.
"""

from __future__ import annotations

import warnings
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import polars as pl

from src.transform.features import CUTOFF_HOUR_EST

CLASSICAL_WINDOW_HOURS = 8 * 7 * 24
CHUNK_DAYS = 100
STEPS_TO_END_OF_DAY = (24 - CUTOFF_HOUR_EST) + 24   # from the last known hour to 23:00 on D


def _rows(frame: pl.DataFrame, days: list[date]) -> pl.DataFrame:
    return frame.filter(pl.col("delivery_date").is_in(days))


def naive(frame: pl.DataFrame, days: list[date]) -> pl.DataFrame:
    return _rows(frame, days).select("ts_est", pl.coalesce("lag24_if_known", "lag48").alias("yhat"))


def naive_leaky(frame: pl.DataFrame, days: list[date]) -> pl.DataFrame:
    shifted = frame.sort("ts_est").with_columns(pl.col("y").shift(24).alias("yhat"))
    return _rows(shifted, days).select("ts_est", "yhat")


def seasonal_naive(frame: pl.DataFrame, days: list[date]) -> pl.DataFrame:
    return _rows(frame, days).select("ts_est", pl.col("lag168").alias("yhat"))


def _panel(frame: pl.DataFrame, days: list[date], window: int) -> pd.DataFrame:
    series = frame.sort("ts_est").select("ts_est", "y")
    ts = series["ts_est"].to_list()
    y = series["y"].to_numpy(allow_copy=True).astype(float)
    index = {t: i for i, t in enumerate(ts)}
    parts = []
    for day in days:
        cutoff = datetime.combine(day - timedelta(days=1), datetime.min.time()) + timedelta(hours=CUTOFF_HOUR_EST)
        end = index.get(cutoff)
        if end is None or end < window:
            continue
        seg = y[end - window:end].copy()
        bad = np.isnan(seg)
        if bad.mean() > 0.05:
            continue
        if bad.any():
            good = np.flatnonzero(~bad)
            seg[bad] = np.interp(np.flatnonzero(bad), good, seg[good])
        parts.append(pd.DataFrame({"unique_id": day.isoformat(), "ds": ts[end - window:end], "y": seg}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["unique_id", "ds", "y"])


def classical(frame: pl.DataFrame, days: list[date], n_jobs: int = -1,
              window: int = CLASSICAL_WINDOW_HOURS,
              which: tuple[str, ...] = ("mstl_ets", "mstl_arima")) -> dict[str, pl.DataFrame]:
    """MSTL+ETS and/or MSTL+ARIMA forecasts for every day, refitted at every origin.

    The two are independent models fitted side by side, so asking for one gives
    exactly the forecasts it would have had alongside the other.
    """
    from statsforecast import StatsForecast
    from statsforecast.models import MSTL, AutoARIMA, AutoETS

    if len(days) > CHUNK_DAYS:
        # Fit in chunks so a long window reports progress. Each origin is its own
        # series and is fitted independently, so chunking changes no forecast.
        parts: dict[str, list[pl.DataFrame]] = {}
        for start in range(0, len(days), CHUNK_DAYS):
            for name, out in classical(frame, days[start:start + CHUNK_DAYS], n_jobs, window, which).items():
                parts.setdefault(name, []).append(out)
            print(f"  classical models {min(start + CHUNK_DAYS, len(days))}/{len(days)} days", flush=True)
        return {name: pl.concat(outs) for name, outs in parts.items()}

    panel = _panel(frame, days, window)
    if panel.empty:
        return {}
    available = {
        "mstl_ets": lambda: MSTL(season_length=[24, 168], trend_forecaster=AutoETS(model="ZZN"), alias="mstl_ets"),
        "mstl_arima": lambda: MSTL(season_length=[24, 168], trend_forecaster=AutoARIMA(), alias="mstl_arima"),
    }
    models = [available[name]() for name in which]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = StatsForecast(models=models, freq="h", n_jobs=n_jobs).forecast(df=panel, h=STEPS_TO_END_OF_DAY)
    out = pl.from_pandas(out.reset_index() if "unique_id" not in out.columns else out)
    out = out.with_columns(
        pl.col("unique_id").cast(pl.Utf8).str.strptime(pl.Date, "%Y-%m-%d").alias("delivery_date"),
        pl.col("ds").cast(frame.schema["ts_est"]).alias("ts_est"),
    ).filter(pl.col("ts_est").dt.date() == pl.col("delivery_date"))
    return {name: out.select("ts_est", pl.col(name).alias("yhat")) for name in which}
