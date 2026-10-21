"""v2 building blocks: Holm, pinball and coverage, ensemble weights, the quantile baseline."""

from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from src.evaluate import metrics
from src.models import ensemble, lgbm, probabilistic
from src.transform import features as F


def test_holm_matches_hand_calculation():
    assert metrics.holm([0.01, 0.04, 0.03, None]) == pytest.approx([0.03, 0.06, 0.06, None])
    assert metrics.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_pinball_and_coverage():
    rows = []
    for t, y in ((datetime(2026, 1, 1, 0), 10.0), (datetime(2026, 1, 1, 1), 20.0)):
        for q, yhat in ((0.05, 9.0), (0.25, 9.5), (0.5, 10.0), (0.75, 10.5), (0.95, 11.0)):
            rows.append({"ts_est": t, "model": "m", "q": q, "yhat": yhat, "y": y})
    out = metrics.quantile_summary(pl.DataFrame(rows)).row(0, named=True)
    assert out["coverage_90_pct"] == 50.0           # 10 inside [9, 11], 20 outside
    assert out["width_90_mw"] == 2.0
    # hour 0: losses 0.05, 0.125, 0, 0.125, 0.05; hour 1: q*(20-yhat) = .55, 2.625, 5, 7.125, 8.55
    assert out["mean_pinball"] == pytest.approx(round((0.35 + 23.85) / 10, 1))


def _dev(weight_summer: float, weight_other: float) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    ts = [datetime(2023, 1, 1) + timedelta(hours=i) for i in range(24 * 365)]
    a, b = rng.normal(17000, 900, len(ts)), rng.normal(17000, 900, len(ts))
    w = np.where([t.month in (6, 7, 8) for t in ts], weight_summer, weight_other)
    y = w * a + (1 - w) * b
    return pl.concat([
        pl.DataFrame({"ts_est": ts, "model": "lgbm", "yhat": a, "y": y}),
        pl.DataFrame({"ts_est": ts, "model": "ets", "yhat": b, "y": y}),
    ])


def test_ensemble_weights_recover_a_known_mix():
    dev = _dev(0.8, 0.2)
    assert ensemble.fit_weights(dev, "lgbm", "ets", "seasonal") == {"other": 0.2, "summer": 0.8}
    assert ensemble.fit_weights(_dev(0.3, 0.3), "lgbm", "ets", "global") == {"all": 0.3}


def test_ensemble_applies_frozen_weights_without_refitting():
    preds = _dev(0.8, 0.2).with_columns(pl.lit(0.0).alias("y"))     # different truth must not matter
    out = ensemble.combine(preds, "lgbm", "ets", {"all": 0.5}, "ens")
    wide = preds.pivot(on="model", index="ts_est", values="yhat")
    expected = wide.select("ts_est", (0.5 * pl.col("lgbm") + 0.5 * pl.col("ets")).alias("e"))
    joined = out.join(expected, on="ts_est")
    assert (joined["yhat"] - joined["e"]).abs().max() < 1e-9


def test_quantile_baseline_offsets_use_only_past_actuals(make_staging):
    frame = F.build(make_staging(days=440))          # 2025-02-01 .. 2026-04-15
    march = [date(2026, 3, 1) + timedelta(d) for d in range(31)]
    clean = probabilistic.empirical_quantiles(frame, march, "seasonal_naive")
    # Overwrite actuals from the day before the refit onward; lag columns stay as built,
    # so only information the baseline must not use is changed.
    corrupted = frame.with_columns(
        pl.when(pl.col("delivery_date") >= date(2026, 2, 28)).then(1e9).otherwise(pl.col("y")).alias("y"))
    after = probabilistic.empirical_quantiles(corrupted, march, "seasonal_naive")
    assert clean.height == 31 * 24 * 5
    assert clean.equals(after)


def test_quantile_forecasts_are_ordered(make_staging):
    frame = F.build(make_staging(days=120))
    days = [date(2025, 5, 20) + timedelta(d) for d in range(3)]
    q = lgbm.forecast_quantiles(frame, days, "A", "since", (0.05, 0.5, 0.95), n_jobs=1,
                                train_start=date(2025, 2, 1), min_train_days=20)
    assert q.height == 3 * 24 * 3
    ordered = q.sort("ts_est", "q").group_by("ts_est").agg(pl.col("yhat"))
    assert all(v == sorted(v) for v in ordered["yhat"].to_list())
