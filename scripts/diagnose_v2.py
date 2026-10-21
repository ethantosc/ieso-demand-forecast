#!/usr/bin/env python3
"""Post-hoc diagnostics for the first v2 run. Exploratory, not pre-registered.

The first v2 report raised two questions it cannot answer by itself:

1. Track C (GFS forecast temperature) scored better than its observed-temperature
   control B_2021, which should be an upper bound. Noise, or something specific:
   certain hours, certain months, gaps in the observed data?
2. LightGBM B scored 644 MW on the official window in v2 against 613 MW in v1.
   The only new day is 2026-10-19. Was its observed weather incomplete when the
   run happened?

This reads the prediction cache the run left behind and the staging weather
files as they are. It downloads nothing - a refetch would replace the weather
the run actually saw - and it lives outside src/, so the freeze date and the
prospective window are untouched. Nothing here feeds back into the v2 results.

    python scripts/diagnose_v2.py [--cache data/curated/v2_cache/<commit>]

Prints a Markdown report and writes it to reports/v2/diagnostics.md.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import polars as pl  # noqa: E402

from src.analysis import eda  # noqa: E402
from src.evaluate import metrics  # noqa: E402
from src.evaluate.run import md_table  # noqa: E402
from src.transform import features as F  # noqa: E402

CACHE_ROOT = Path("data/curated/v2_cache")
STAGING = Path("data/staging/demand_hourly.parquet")
WEATHER = Path("data/staging/weather_hourly.parquet")
WEATHER_FC = Path("data/staging/weather_fc_hourly.parquet")
OUT = Path("reports/v2/diagnostics.md")

WINDOWS = ("holdout", "official", "seen_backtest")
MODELS = {"lgbm_A_2021": "A_2021", "lgbm_C_2021": "C", "lgbm_B_2021": "B_2021", "lgbm_B_expanding": "B_2003"}
DAY1_MAX_EST_HOUR = 6        # EST hours 0-6 use previous_day1, 7-23 previous_day2 (spec-v2 section 3)
FOCUS_DAYS = (date(2026, 10, 19),)   # the one day the first run added to v1's IESO window


def pick_cache(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    dirs = [d for d in CACHE_ROOT.glob("*") if d.is_dir() and not d.name.endswith("-quick")]
    if not dirs:
        sys.exit(f"no prediction cache under {CACHE_ROOT} - this needs the runtime that ran the v2 notebook")
    return max(dirs, key=lambda d: d.stat().st_mtime)


def load_predictions(cache: Path, window: str, frame: pl.DataFrame) -> pl.DataFrame:
    parts = []
    for model, short in MODELS.items():
        files = sorted(cache.glob(f"{window}__*__market__{model}.parquet"))
        if files:
            parts.append(pl.read_parquet(files[-1]).select("ts_est", "yhat")
                         .with_columns(pl.lit(short).alias("model")))
    if not parts:
        return pl.DataFrame()
    preds = pl.concat(parts, how="vertical_relaxed")
    preds = preds.with_columns(pl.col("ts_est").cast(frame.schema["ts_est"]))
    joined = preds.join(frame.select("ts_est", "delivery_date", "y", "h", "month"), on="ts_est", how="left")
    return metrics.common_sample(joined)


def hourly_weather(path: Path, dtype) -> pl.DataFrame:
    w = pl.read_parquet(path)
    cols = ["ts_est", "temp"] + (["n_cities"] if "n_cities" in w.columns else [])
    return w.select(cols).with_columns(pl.col("ts_est").cast(dtype)).filter(pl.col("temp").is_not_null())


def day_completeness(days: list[date], obs: pl.DataFrame, fc: pl.DataFrame) -> pl.DataFrame:
    grid = pl.DataFrame({"delivery_date": days}).with_columns(pl.col("delivery_date").cast(pl.Date))
    def per_day(w: pl.DataFrame, name: str) -> pl.DataFrame:
        aggs = [pl.len().alias(f"{name}_hours")]
        if name == "obs" and "n_cities" in w.columns:
            aggs.append(pl.col("n_cities").min().alias("obs_min_cities"))
        return w.with_columns(pl.col("ts_est").dt.date().alias("delivery_date")).group_by("delivery_date").agg(aggs)
    out = grid.join(per_day(obs, "obs"), on="delivery_date", how="left").join(
        per_day(fc, "fc"), on="delivery_date", how="left")
    return out.with_columns(pl.col("obs_hours").fill_null(0), pl.col("fc_hours").fill_null(0)).sort("delivery_date")


def mae_wide(pred: pl.DataFrame, by: str | None = None) -> pl.DataFrame:
    keys = [by] if by else []
    m = pred.group_by(["model", *keys]).agg((pl.col("yhat") - pl.col("y")).abs().mean().round(0).alias("MAE"))
    if not by:
        m = m.with_columns(pl.lit("all").alias("_k"))
        by = "_k"
    present = [s for s in MODELS.values() if s in m["model"].unique().to_list()]
    out = m.pivot(on="model", index=by, values="MAE").sort(by)
    return out.select([by, *present])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache")
    cache = pick_cache(ap.parse_args().cache)

    staging = pl.read_parquet(STAGING)
    excluded = eda.exclusion_dates(eda.blackout_window(eda.load()))
    frame = F.build(staging, excluded)
    dtype = frame.schema["ts_est"]
    obs = hourly_weather(WEATHER, dtype)
    fc = hourly_weather(WEATHER_FC, dtype)

    lines = ["# v2 diagnostics - exploratory, post hoc", "",
             "Not pre-registered. Generated by `scripts/diagnose_v2.py` from the prediction cache "
             f"`{cache}` and the staging weather files the run used. p-values are raw: these tests "
             "are not part of the spec-v2 family and carry no Holm correction.", ""]
    def section(title: str, text: str = "") -> None:
        lines.extend([f"## {title}", ""] + ([text, ""] if text else []))
    def table(df: pl.DataFrame, max_rows: int = 60) -> None:
        lines.extend([md_table(df, max_rows), ""])

    preds = {w: load_predictions(cache, w, frame) for w in WINDOWS}
    preds = {w: p for w, p in preds.items() if not p.is_empty()}
    days = {w: sorted(p["delivery_date"].unique().to_list()) for w, p in preds.items()}
    complete = {}

    section("1. Weather the run saw")
    lines.append(f"- Last hour with observed temperature: **{obs['ts_est'].max()}** EST")
    lines.append(f"- Last hour with forecast temperature: **{fc['ts_est'].max()}** EST")
    lines.append("")
    rows = []
    for w, d in days.items():
        c = day_completeness(d, obs, fc)
        full = pl.col("obs_hours") == 24
        if "obs_min_cities" in c.columns:
            full = full & (pl.col("obs_min_cities") == 3)
        complete[w] = c.filter(full)["delivery_date"].to_list()
        row = {"window": w, "days": len(d), "days_obs_incomplete": c.filter(pl.col("obs_hours") < 24).height,
               "hours_obs_missing": int((24 - c["obs_hours"]).clip(lower_bound=0).sum()),
               "hours_fc_missing": int((24 - c["fc_hours"]).clip(lower_bound=0).sum())}
        if "obs_min_cities" in c.columns:
            row["days_with_a_city_missing"] = c.filter(pl.col("obs_min_cities") < 3).height
        rows.append(row)
    table(pl.DataFrame(rows))
    if "official" in days:
        lines.append("Last five days of the official window:")
        lines.append("")
        table(day_completeness(days["official"][-5:], obs, fc))

    if "official" in preds:
        section("2. Official window - daily MAE, last seven days and 2026-10-19",
                "If B collapses only on days with incomplete observed weather, the v1-v2 gap is a data-timing "
                "artefact, not a model change. 2026-10-19 was the last day of the first run's window.")
        p = preds["official"]
        last = sorted(set(days["official"][-7:]) | {d for d in FOCUS_DAYS if d in days["official"]})
        daily = mae_wide(p.filter(pl.col("delivery_date").is_in(last)), "delivery_date")
        comp = day_completeness(last, obs, fc).select("delivery_date", "obs_hours")
        table(daily.join(comp, on="delivery_date", how="left").sort("delivery_date"))

    section("3. MAE on all days vs days with complete observed weather",
            "Complete = all 24 hours have an observed value and all three cities report in every hour.")
    rows = []
    for w, p in preds.items():
        for label, keep in (("all days", days[w]), ("complete observed weather", complete[w])):
            sub = p.filter(pl.col("delivery_date").is_in(keep))
            if sub.is_empty():
                continue
            rows.append(mae_wide(sub).drop("_k").with_columns(
                pl.lit(w).alias("window"), pl.lit(label).alias("sample"), pl.lit(len(keep)).alias("days")))
    if rows:
        out = pl.concat(rows, how="diagonal_relaxed")
        table(out.select(["window", "sample", "days", *[c for c in MODELS.values() if c in out.columns]]))

    section("4. Forecast vs observed temperature",
            "City-mean temperature, both series as the models saw them. Bias = forecast - observed. "
            "EST hours 0-6 use previous_day1 (lead 24-29 h), 7-23 use previous_day2 (lead 48-53 h).")
    both = fc.select("ts_est", pl.col("temp").alias("fc")).join(obs.select("ts_est", pl.col("temp").alias("obs")),
                                                               on="ts_est", how="inner")
    both = both.with_columns(
        pl.col("ts_est").dt.date().alias("delivery_date"),
        pl.when(pl.col("ts_est").dt.hour() <= DAY1_MAX_EST_HOUR).then(pl.lit("day1 (EST 0-6)"))
        .otherwise(pl.lit("day2 (EST 7-23)")).alias("series"),
    )
    rows = []
    for w, d in days.items():
        sub = both.filter(pl.col("delivery_date").is_in(d))
        rows.append(sub.group_by("series").agg(
            pl.len().alias("hours"),
            (pl.col("fc") - pl.col("obs")).abs().mean().round(2).alias("MAE_C"),
            (pl.col("fc") - pl.col("obs")).mean().round(2).alias("bias_C"),
            pl.corr("fc", "obs").round(4).alias("corr"),
        ).with_columns(pl.lit(w).alias("window")))
    if rows:
        table(pl.concat(rows).select("window", "series", "hours", "MAE_C", "bias_C", "corr").sort("window", "series"))

    section("5. C against its controls - Diebold-Mariano, exploratory",
            "Negative mean_diff = C better. Same daily-MAE loss and Newey-West variance as the main report.")
    rows = []
    for w, p in preds.items():
        present = p["model"].unique().to_list()
        for label, keep in (("all days", days[w]), ("complete observed weather", complete[w])):
            sub = p.filter(pl.col("delivery_date").is_in(keep))
            for bench in ("B_2021", "B_2003", "A_2021"):
                if bench in present and "C" in present:
                    r = metrics.diebold_mariano(sub, "C", bench)
                    rows.append({"window": w, "sample": label, **r})
    if rows:
        table(pl.DataFrame(rows))

    section("6. MAE by month")
    for w, p in preds.items():
        lines.append(f"**{w}**")
        lines.append("")
        by_month = p.with_columns(pl.col("delivery_date").dt.strftime("%Y-%m").alias("month_key"))
        out = mae_wide(by_month, "month_key")
        fc_gap = (day_completeness(days[w], obs, fc)
                  .with_columns(pl.col("delivery_date").dt.strftime("%Y-%m").alias("month_key"))
                  .group_by("month_key").agg((24 - pl.col("fc_hours")).clip(lower_bound=0).sum().alias("fc_hours_missing")))
        table(out.join(fc_gap, on="month_key", how="left").sort("month_key"))

    section("7. By EST hour group",
            "C's day1 hours use a forecast about a day old; its day2 hours one about two days old.")
    for w, p in preds.items():
        grouped = p.with_columns(pl.when(pl.col("h") <= DAY1_MAX_EST_HOUR).then(pl.lit("EST 0-6"))
                                 .otherwise(pl.lit("EST 7-23")).alias("hours"))
        lines.append(f"**{w}**")
        lines.append("")
        table(mae_wide(grouped, "hours"))

    text = "\n".join(lines) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(text)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
