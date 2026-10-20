#!/usr/bin/env python3
"""LightGBM + MSTL+ETS combinations (spec-v2 section 4).

    combined = w * LightGBM + (1 - w) * MSTL+ETS,   w in {0, 0.1, ..., 1}

w is chosen on the development window only (2023) by lowest MAE, ties going
to the smaller w, and is then frozen for every other window. Two variants
were declared in advance: one global weight, and one weight for summer
(June-August) with another for the rest of the year. Nothing else is tried.
"""

from __future__ import annotations

import polars as pl

GRID = tuple(round(i / 10, 1) for i in range(11))
SUMMER = (6, 7, 8)


def _group(variant: str) -> pl.Expr:
    if variant == "global":
        return pl.lit("all")
    if variant == "seasonal":
        return pl.when(pl.col("ts_est").dt.month().is_in(SUMMER)).then(pl.lit("summer")).otherwise(pl.lit("other"))
    raise ValueError(variant)


def _wide(preds: pl.DataFrame, lgbm: str, ets: str) -> pl.DataFrame:
    """Hours where both components have a forecast (and, if present, an actual)."""
    keep = preds.filter(pl.col("model").is_in([lgbm, ets]))
    wide = keep.pivot(on="model", index="ts_est", values="yhat").drop_nulls([lgbm, ets])
    if "y" in preds.columns:
        actual = preds.select("ts_est", "y").unique("ts_est")
        wide = wide.join(actual, on="ts_est", how="left")
    return wide


def fit_weights(dev: pl.DataFrame, lgbm: str, ets: str, variant: str) -> dict[str, float]:
    """Grid-search w on the development window. `dev` is long: ts_est, model, yhat, y."""
    wide = _wide(dev, lgbm, ets).drop_nulls("y").with_columns(_group(variant).alias("g"))
    weights = {}
    for (group,), part in wide.group_by(["g"], maintain_order=True):
        best = None
        for w in GRID:
            mae = (w * part[lgbm] + (1 - w) * part[ets] - part["y"]).abs().mean()
            if best is None or mae < best[1] - 1e-9:
                best = (w, mae)
        weights[group] = best[0]
    return dict(sorted(weights.items()))


def combine(preds: pl.DataFrame, lgbm: str, ets: str, weights: dict[str, float], name: str) -> pl.DataFrame:
    """Apply frozen weights. Returns ts_est, yhat, model."""
    variant = "global" if set(weights) == {"all"} else "seasonal"
    wide = _wide(preds, lgbm, ets).with_columns(_group(variant).alias("g"))
    w = pl.col("g").replace_strict(weights, return_dtype=pl.Float64)
    return wide.select("ts_est", (w * pl.col(lgbm) + (1 - w) * pl.col(ets)).alias("yhat"),
                       pl.lit(name).alias("model"))
