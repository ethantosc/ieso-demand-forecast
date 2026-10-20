#!/usr/bin/env python3
"""Milestones 3-7 in one run: backtest, baseline ladder, LightGBM tracks A and B, official comparison.

Run from the repo root, after the staging Parquet exists (notebook 00 or
`python -m src.ingest.fetch_demand && python -m src.ingest.demand`):

    python -m src.evaluate.run              # full run, about 20-30 minutes on Colab
    python -m src.evaluate.run --no-weather # skip Track B and the temperature split
    python -m src.evaluate.run --quick      # two months, for checking the plumbing

Writes
    reports/findings_auto.md        every number, machine-written; do not edit by hand
    reports/results/*.csv           the tables behind it
    reports/figures/*.png           the figures
    data/curated/predictions_*.parquet   every forecast, for your own digging (not in git)

Two evaluation windows:
    backtest   2024-01-01 .. 2025-12-31, two complete years (spec section 5)
    official   every delivery day in the harvested DATotals archive that has
               actuals - the comparison against IESO's own day-ahead forecast

The evaluation design was fixed before any model was run (spec section 8:
framework before models). Nothing here was adjusted after seeing results.
"""

from __future__ import annotations

import argparse
import math
import platform
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

from src.evaluate import metrics, plots
from src.ingest import forecasts as official_io
from src.models import baselines, lgbm
from src.transform import features as F

STAGING = Path("data/staging/demand_hourly.parquet")
WEATHER = Path("data/staging/weather_hourly.parquet")
WEATHER_COVERAGE = Path("data/staging/weather_coverage.csv")
REPORT = Path("reports/findings_auto.md")
RESULTS = Path("reports/results")
FIGURES = Path("reports/figures")
CURATED = Path("data/curated")

BACKTEST = (date(2024, 1, 1), date(2025, 12, 31))
QUICK_BACKTEST = (date(2025, 1, 1), date(2025, 2, 28))

LABELS = {
    "naive": "1  Naive (legal: D-1 or D-2)",
    "seasonal_naive": "2  Seasonal naive, y(t-168)",
    "mstl_ets": "3  MSTL + ETS",
    "mstl_arima": "4  MSTL + ARIMA",
    "lgbm_A_rolling_3y": "5  LightGBM A, 3-year window",
    "lgbm_A_expanding": "5  LightGBM A, expanding window",
    "lgbm_B_rolling_3y": "B  LightGBM, 3-year (UPPER BOUND)",
    "lgbm_B_expanding": "B  LightGBM, expanding (UPPER BOUND)",
    "naive_leaky": "x  Naive y(t-24) - LEAKS, diagnostic",
    "ieso_official": "IESO day-ahead forecast",
}
TRACK_A = ["naive", "seasonal_naive", "mstl_ets", "mstl_arima", "lgbm_A_rolling_3y", "lgbm_A_expanding"]
TRACK_B = ["lgbm_B_rolling_3y", "lgbm_B_expanding"]
DIAGNOSTIC = ["naive_leaky"]
DIMENSIONS = ["hour_ending", "local_hour", "season", "day_type", "temp_band"]


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def md_table(frame: pl.DataFrame, max_rows: int = 60) -> str:
    if frame.is_empty():
        return "_no rows_\n"
    head = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, float):
            if v != v:
                return "nan"
            if abs(v) < 1:
                return f"{v:.3g}"
            return f"{v:,.2f}" if abs(v) < 10 else f"{v:,.0f}"
        return str(v)
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.rows()[:max_rows]]
    more = f"\n_({frame.height} rows, first {max_rows} shown)_" if frame.height > max_rows else ""
    return "\n".join([head, rule, *body]) + more + "\n"


def pct(a: float, b: float) -> str:
    return f"{(a / b - 1) * 100:+.1f}%"


# ------------------------------------------------------------------ inputs


def load_inputs(use_weather: bool) -> dict:
    from src.analysis import eda

    if not STAGING.exists():
        sys.exit(f"missing {STAGING}; run the ingest first (notebook 00, cells 1-5)")
    staging = pl.read_parquet(STAGING)
    eda_frame = eda.load()
    window = eda.blackout_window(eda_frame)
    excluded = eda.exclusion_dates(window)
    log(f"staging {staging.height:,} rows; blackout exclusion {len(excluded)} days "
        f"({excluded[0]} .. {excluded[-1]})" if excluded else "no exclusion")

    weather, weather_status, coverage = None, "not requested (--no-weather)", None
    if use_weather:
        try:
            from src.ingest import fetch_weather
            code = fetch_weather.main()
            if code != 0 or not WEATHER.exists():
                raise RuntimeError(f"fetch_weather exited {code}")
            weather = pl.read_parquet(WEATHER).select("ts_est", "temp")
            coverage = pl.read_csv(WEATHER_COVERAGE)
            weather_status = "available"
        except Exception as error:  # Track B is optional; Track A must still finish
            weather_status = f"unavailable - {type(error).__name__}: {error}"
            log("weather " + weather_status)

    frame = F.build(staging, excluded, weather)
    official = official_io.load()
    return {"staging": staging, "frame": frame, "excluded": excluded, "blackout": window,
            "eda_frame": eda_frame, "weather": weather, "weather_status": weather_status,
            "coverage": coverage, "official": official}


def day_list(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def official_days(frame: pl.DataFrame, official: pl.DataFrame, quick: bool) -> list[date]:
    if official.is_empty():
        return []
    have = (frame.filter(pl.col("y").is_not_null()).group_by("delivery_date")
            .agg(pl.len().alias("n")).filter(pl.col("n") >= 23)["delivery_date"])
    days = sorted(set(official["delivery_date"].to_list()) & set(have.to_list()))
    return days[-21:] if quick else days


# ------------------------------------------------------------------ models


def run_models(frame: pl.DataFrame, days: list[date], with_weather: bool, n_jobs: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    preds, gains = [], []

    def add(name: str, out: pl.DataFrame) -> None:
        preds.append(out.with_columns(pl.lit(name).alias("model")))

    for name, fn in [("naive", baselines.naive), ("naive_leaky", baselines.naive_leaky),
                     ("seasonal_naive", baselines.seasonal_naive)]:
        add(name, fn(frame, days))
    t = time.time()
    for name, out in baselines.classical(frame, days, n_jobs=n_jobs).items():
        add(name, out)
    log(f"  classical models: {time.time() - t:.0f}s")
    tracks = ["A", "B"] if with_weather else ["A"]
    for track in tracks:
        for window in ("rolling_3y", "expanding"):
            t = time.time()
            out, gain = lgbm.forecast(frame, days, track, window, n_jobs=n_jobs)
            name = f"lgbm_{track}_{window}"
            add(name, out)
            if not gain.is_empty():
                gains.append(gain.with_columns(pl.lit(name).alias("model")))
            log(f"  {name}: {time.time() - t:.0f}s")
    gain_table = pl.concat(gains) if gains else pl.DataFrame()
    return pl.concat(preds, how="vertical_relaxed"), gain_table


def attach(preds: pl.DataFrame, frame: pl.DataFrame, temp_daily: pl.DataFrame | None) -> pl.DataFrame:
    cols = ["ts_est", "delivery_date", "y", "h", "local_hour", "month", "is_weekend", "is_stat_holiday"]
    joined = preds.join(frame.select(cols), on="ts_est", how="left")
    return metrics.with_labels(joined, temp_daily)


# ------------------------------------------------------------------ report


class Report:
    def __init__(self) -> None:
        self.parts: list[str] = []

    def h(self, text: str) -> None:
        self.parts.append(f"\n## {text}\n")

    def p(self, text: str) -> None:
        self.parts.append(text.rstrip() + "\n")

    def table(self, frame: pl.DataFrame, max_rows: int = 60) -> None:
        self.parts.append(md_table(frame, max_rows))

    def text(self) -> str:
        return "\n".join(self.parts)


def wide(summary: pl.DataFrame, dim: str, value: str, order: list[str]) -> pl.DataFrame:
    present = [m for m in order if m in summary["model"].unique().to_list()]
    out = summary.pivot(on="model", index=dim, values=value).sort(dim)
    return out.select([dim, *present]).rename({m: LABELS.get(m, m).split("  ", 1)[-1] for m in present})


def evaluate_window(name: str, scored: pl.DataFrame, rep: Report, figs: dict, contenders: list[str]) -> dict:
    """Common-sample metrics for one window. Returns the headline facts."""
    diag = scored.filter(pl.col("model").is_in(DIAGNOSTIC))
    main = metrics.common_sample(scored.filter(pl.col("model").is_in(contenders)))
    hours = main.select("ts_est").unique()
    diag = diag.join(hours, on="ts_est", how="semi").filter(pl.col("yhat").is_not_null())

    overall = metrics.summary(main)
    overall.write_csv(RESULTS / f"{name}_summary.csv")
    facts = {"overall": overall, "n_days": main["delivery_date"].n_unique(), "n_hours": hours.height}

    snaive = overall.filter(pl.col("model") == "seasonal_naive")["MAE"]
    facts["snaive_mae"] = snaive.item() if snaive.len() else None
    # Seasonal naive is a candidate too: if nothing beats the threshold, the
    # threshold is the best Track A model, and the report must say so.
    a_models = overall.filter(pl.col("model").is_in(TRACK_A))
    best_a = a_models.sort("MAE").row(0, named=True) if a_models.height else None
    facts["best_a"] = best_a
    lgbm_a = overall.filter(pl.col("model").str.starts_with("lgbm_A")).sort("MAE")
    facts["best_lgbm_a"] = lgbm_a.row(0, named=True) if lgbm_a.height else None
    b_models = overall.filter(pl.col("model").is_in(TRACK_B)).sort("MAE")
    facts["best_b"] = b_models.row(0, named=True) if b_models.height else None
    off = overall.filter(pl.col("model") == "ieso_official")
    facts["official"] = off.row(0, named=True) if off.height else None

    by = {}
    for dim in DIMENSIONS:
        if main[dim].null_count() == main.height:
            continue
        table = metrics.summary(main, [dim])
        table.write_csv(RESULTS / f"{name}_by_{dim}.csv")
        by[dim] = table
    facts["by"] = by

    tests = [metrics.diebold_mariano(main, m, "seasonal_naive") for m in contenders if m != "seasonal_naive"]
    if facts["official"] and best_a:
        tests.append(metrics.diebold_mariano(main, best_a["model"], "ieso_official"))
    facts["dm"] = pl.DataFrame(tests)
    facts["dm"].write_csv(RESULTS / f"{name}_dm_tests.csv")

    if not diag.is_empty():
        leak = metrics.summary(pl.concat([main.filter(pl.col("model") == "naive"), diag]))
        leak.write_csv(RESULTS / f"{name}_leakage_cost.csv")
        facts["leak"] = leak
    facts["main"] = main
    return facts


def write_window_section(rep: Report, title: str, facts: dict, figs: dict, name: str) -> None:
    rep.h(title)
    rep.p(f"{facts['n_days']} delivery days, {facts['n_hours']:,} hours, scored on the common sample "
          f"(only hours every model forecast). MAE/RMSE/ME in MW; ME = forecast - actual.")
    shown = facts["overall"].with_columns(
        pl.col("model").replace_strict(LABELS, default=pl.col("model")).alias("model"))
    rep.table(shown)
    if name in figs:
        rep.p(f"![ladder](figures/{figs[name].name})")
    rep.p("\n**Is each gap real?** Diebold-Mariano test on daily MAE, Newey-West variance. "
          "Negative mean_diff = the model beat the benchmark.\n")
    rep.table(facts["dm"])

    order = [m for m in LABELS if m in facts["overall"]["model"].to_list()]
    for dim, label in [("hour_ending", "MAE by delivery hour (hour-ending 1-24 EST; lead time from the last known hour is hour-ending + 13 h)"),
                       ("local_hour", "MAE by local clock hour"),
                       ("season", "MAE by season"),
                       ("day_type", "MAE by day type"),
                       ("temp_band", "MAE on extreme-temperature days (needs weather)")]:
        if dim in facts["by"]:
            rep.p(f"\n**{label}**\n")
            rep.table(wide(facts["by"][dim], dim, "MAE", order), max_rows=30)
    if "season" in facts["by"]:
        rep.p("\n**Bias (ME) by season** - positive = over-forecast\n")
        rep.table(wide(facts["by"]["season"], "season", "ME", order))


def residual_section(rep: Report, main: pl.DataFrame, model: str, name: str) -> None:
    part = main.filter(pl.col("model") == model).with_columns((pl.col("yhat") - pl.col("y")).alias("e"))
    daily = (part.group_by("delivery_date").agg(pl.col("e").mean().alias("me"))
             .sort("delivery_date").with_columns(pl.col("me").rolling_mean(30, min_samples=10).alias("me_30d")))
    series = daily["me"].to_list()
    n, mean = len(series), sum(series) / max(len(series), 1)
    var = sum((x - mean) ** 2 for x in series) / max(n, 1)
    acf = [sum((series[i] - mean) * (series[i - k] - mean) for i in range(k, n)) / n / var if var else 0.0
           for k in range(1, 8)]
    path = plots.residuals(daily, acf, part["e"], LABELS.get(model, model), FIGURES / f"{name}_residuals.png")
    monthly = (part.with_columns(pl.col("delivery_date").dt.strftime("%Y-%m").alias("month"))
               .group_by("month").agg(pl.col("e").mean().round(0).alias("ME"), pl.col("e").abs().mean().round(0).alias("MAE"))
               .sort("month"))
    rep.h(f"Residual diagnostics - {LABELS.get(model, model)} ({name})")
    rep.p(f"![residuals](figures/{path.name})")
    rep.p("Autocorrelation of the daily mean error, lags 1-7 days (|value| above "
          f"{1.96 / math.sqrt(max(n, 1)):.2f} is outside the 95% noise band):\n")
    rep.table(pl.DataFrame({"lag_days": list(range(1, 8)), "acf": [round(a, 3) for a in acf]}))
    q = part["e"].quantile
    rep.p(f"\nHourly error quantiles (MW): 5% {q(0.05):,.0f} · 25% {q(0.25):,.0f} · median {q(0.5):,.0f} · "
          f"75% {q(0.75):,.0f} · 95% {q(0.95):,.0f}\n")
    rep.p("\nMonthly bias:\n")
    rep.table(monthly, max_rows=30)


def headline(rep: Report, bt: dict, off: dict | None, weather_status: str) -> None:
    rep.h("1. Headline (generated from the numbers below, no interpretation)")
    lines = []
    s = bt["snaive_mae"]
    lines.append(f"- Threshold: seasonal naive scored MAE **{s:,.0f} MW** over the backtest.")
    la = bt["best_lgbm_a"]
    if la:
        p = bt["dm"].filter(pl.col("model") == la["model"])["p_value"]
        p = p.item() if p.len() else None
        if la["MAE"] >= s:
            verdict = "**did not beat**"
        elif p is not None and p < 0.05:
            verdict = "**beat**, by a gap the Diebold-Mariano test separates from noise,"
        else:
            verdict = "had a lower MAE than, but **not distinguishably better than**,"
        lines.append(f"- Spec rule (rung 5 must beat rung 2): LightGBM Track A {verdict} seasonal naive - "
                     f"{LABELS[la['model']].split('  ')[-1]} MAE {la['MAE']:,.0f} MW ({pct(la['MAE'], s)}), "
                     f"Diebold-Mariano p {'< 0.0001' if p is not None and p < 1e-4 else '= ' + str(p)}.")
    ba = bt["best_a"]
    if ba:
        lines.append(f"- Best Track A model overall: {LABELS[ba['model']].split('  ')[-1]}, "
                     f"MAE {ba['MAE']:,.0f} MW, ME {ba['ME']:+,.0f} MW.")
    if "leak" in bt:
        lk = {r["model"]: r["MAE"] for r in bt["leak"].iter_rows(named=True)}
        if "naive" in lk and "naive_leaky" in lk:
            lines.append(f"- Cost of the clock: the spec's original naive y(t-24), which reads values not yet "
                         f"published at 12:30, would have scored {lk['naive_leaky']:,.0f} MW against the legal "
                         f"naive's {lk['naive']:,.0f} MW. That gap is what the leak was worth.")
    bb = bt["best_b"]
    if bb and ba:
        lines.append(f"- Track B (UPPER BOUND, not a forecast - uses observed temperature): best MAE "
                     f"{bb['MAE']:,.0f} MW. Perfect weather knowledge would have been worth "
                     f"{ba['MAE'] - bb['MAE']:,.0f} MW over the best Track A model.")
    elif not bb:
        lines.append(f"- Track B did not run: weather {weather_status}.")
    if off and off.get("official"):
        o, oba = off["official"], off["best_a"]
        lines.append(f"- Official comparison, {off['n_days']} days: IESO day-ahead MAE **{o['MAE']:,.0f} MW** "
                     f"(ME {o['ME']:+,.0f}); best Track A here {LABELS[oba['model']].split('  ')[-1]} "
                     f"{oba['MAE']:,.0f} MW ({pct(oba['MAE'], o['MAE'])} vs IESO); seasonal naive "
                     f"{off['snaive_mae']:,.0f} MW.")
    rep.p("\n".join(lines))


def eda_appendix(rep: Report, inputs: dict) -> None:
    import holidays as holidays_pkg
    from src.analysis import eda

    df = eda.clean(eda.mark_excluded(inputs["eda_frame"], inputs["excluded"]))
    rep.h("EDA appendix (Milestone 2, sections 0-7)")
    rep.p("**0. Blackout exclusion window** (weekday-corrected):\n")
    rep.table(inputs["blackout"].head(16))
    test = eda.dst_ramp_test(df)
    rep.p("\n**1. DST falsification test** - mean |shift| in morning-ramp hour across transitions vs a control three weeks earlier:\n")
    rep.table(test.group_by("kind").agg(pl.len().alias("n"), pl.col("shift").abs().mean().round(3).alias("mean_abs_shift"),
                                        pl.col("control_shift").abs().mean().round(3).alias("mean_abs_control")))
    prof = eda.intraday_profile(df)
    peaks = (prof.sort("mean_mw", descending=True).group_by(["season", "is_weekend"]).first()
             .select("season", "is_weekend", pl.col("local_hour").alias("peak_local_hour"), pl.col("mean_mw").alias("peak_mean_mw"))
             .sort("season", "is_weekend"))
    troughs = (prof.sort("mean_mw").group_by(["season", "is_weekend"]).first()
               .select("season", "is_weekend", pl.col("local_hour").alias("trough_local_hour"), pl.col("mean_mw").alias("trough_mean_mw")))
    rep.p("\n**2. Daily shape** - peak and trough of the mean profile:\n")
    rep.table(peaks.join(troughs, on=["season", "is_weekend"]).sort("season", "is_weekend"))
    weekly = eda.weekly_profile(df)
    wk = (weekly.with_columns((pl.col("weekday") >= 6).alias("weekend"))
          .group_by("season", "weekend").agg(pl.col("mean_mw").mean().round(0).alias("mean_mw"))
          .pivot(on="weekend", index="season", values="mean_mw")
          .rename({"false": "weekday_mw", "true": "weekend_mw"})
          .with_columns(((pl.col("weekend_mw") / pl.col("weekday_mw") - 1) * 100).round(1).alias("weekend_drop_pct")))
    rep.p("\n**3. Weekly shape** - weekend vs weekday mean:\n")
    rep.table(wk)
    ap = eda.annual_peaks(df)
    rep.p("\n**4. Annual peaks** - count by season, then every year:\n")
    rep.table(ap.group_by("peak_season").agg(pl.len().alias("years")).sort("years", descending=True))
    rep.table(ap.select("year", "peak_ts_local", "peak_mw", "peak_season", "mean_mw"), max_rows=30)
    rep.p("\n**5. Level bias by training-window length** (estimate = mean of previous W years; W=99 means all):\n")
    rep.table(eda.level_bias_by_window(df).filter(pl.col("season").is_in(["all", "summer", "winter"])))
    rep.p("\n**6. How far the series moves** (NOT a baseline score - no origin, no split):\n")
    rep.table(eda.diff_distribution(df))
    years = df["year"].unique().to_list()
    on = holidays_pkg.country_holidays("CA", subdiv="ON", years=years)
    rep.p("\n**7. Holiday effect** - daily mean vs same weekday, same month and year:\n")
    rep.table(eda.holiday_summary(eda.holiday_effect(df, set(on)), dict(on.items())))


# ------------------------------------------------------------------ main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-weather", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--jobs", type=int, default=-1)
    args = parser.parse_args()
    started = time.time()
    for d in (RESULTS, FIGURES, CURATED):
        d.mkdir(parents=True, exist_ok=True)

    log("loading inputs")
    inputs = load_inputs(use_weather=not args.no_weather)
    frame, weather = inputs["frame"], inputs["weather"]
    temp_daily = None
    if weather is not None:
        temp_daily = (frame.group_by("delivery_date").agg(pl.col("temp_day_mean").first())
                      .filter(pl.col("temp_day_mean").is_not_null()))

    windows = {"backtest": day_list(*(QUICK_BACKTEST if args.quick else BACKTEST))}
    od = official_days(frame, inputs["official"], args.quick)
    if od:
        windows["official"] = od

    rep, figs, facts = Report(), {}, {}
    for name, days in windows.items():
        log(f"window {name}: {len(days)} days {days[0]} .. {days[-1]}")
        preds, gains = run_models(frame, days, weather is not None, args.jobs)
        if name == "official":
            off = inputs["official"].with_columns(
                (pl.col("delivery_date").cast(pl.Datetime("us")) + pl.duration(hours=pl.col("hour_ending") - 1))
                .cast(frame.schema["ts_est"]).alias("ts_est")
            ).filter(pl.col("delivery_date").is_in(days))
            preds = pl.concat([preds, off.select("ts_est", pl.col("total_load_mw").alias("yhat"),
                                                 pl.lit("ieso_official").alias("model"))], how="vertical_relaxed")
        scored = attach(preds, frame, temp_daily)
        scored.write_parquet(CURATED / f"predictions_{name}.parquet")
        if not gains.is_empty():
            gains.write_csv(RESULTS / f"{name}_lgbm_importance.csv")
        contenders = TRACK_A + TRACK_B + (["ieso_official"] if name == "official" else [])
        contenders = [m for m in contenders if m in scored["model"].unique().to_list()]
        f = evaluate_window(name, scored, rep, figs, contenders)
        f["gains"] = gains
        figs[name] = plots.ladder(f["overall"], LABELS, f["best_a"]["model"] if f["best_a"] else None,
                                  f["snaive_mae"], FIGURES / f"{name}_ladder.png",
                                  f"{name}: MAE by model ({f['n_days']} days)")
        key = [m for m in ([f["best_a"]["model"]] if f["best_a"] else []) + ["seasonal_naive"]
               + (["ieso_official"] if name == "official" else ["mstl_arima"])]
        by_hour = f["by"]["hour_ending"].select("model", pl.col("hour_ending").alias("x"), pl.col("MAE").alias("value"))
        figs[f"{name}_hour"] = plots.lines_by(by_hour, "x", key, LABELS, FIGURES / f"{name}_by_hour.png",
                                              f"{name}: MAE by delivery hour", "hour-ending (EST)")
        if name == "official":
            daily = (f["main"].group_by("model", "delivery_date").agg((pl.col("yhat") - pl.col("y")).abs().mean().alias("mae"))
                     .sort("delivery_date").with_columns(pl.col("mae").rolling_mean(7, min_samples=4).over("model").alias("value"))
                     .rename({"delivery_date": "x"}))
            figs["official_daily"] = plots.lines_by(daily, "x", key, LABELS, FIGURES / "official_daily.png",
                                                    "official window: daily MAE, 7-day rolling mean", "delivery date")
        facts[name] = f

    log("writing report")
    bt, off = facts["backtest"], facts.get("official")
    rep.p("# Results - machine output\n")
    rep.p(f"Generated {datetime.now():%Y-%m-%d %H:%M} by `src/evaluate/run.py`"
          f"{' (QUICK MODE - plumbing check only, not results)' if args.quick else ''}.\n")
    rep.p("**Do not edit this file by hand.** Interpretation belongs in `findings.md`, which cites these numbers.\n")
    rep.h("0. Run")
    st = inputs["staging"]
    rep.p("\n".join([
        f"- Data: {st.height:,} hourly rows, {st['local_date'].min()} .. {st['local_date'].max()}",
        f"- Excluded (2003 blackout and conservation period): {len(inputs['excluded'])} days, "
        f"{inputs['excluded'][0]} .. {inputs['excluded'][-1]}" if inputs["excluded"] else "- Excluded: none",
        f"- Forecast origin: D-1 ~12:30; data cutoff: D-1 hours beginning before {F.CUTOFF_HOUR_EST:02d}:00 EST",
        f"- Windows: " + "; ".join(f"{k} {v[0]} .. {v[-1]} ({len(v)} days)" for k, v in windows.items()),
        "- Refits: classical models at every origin on the trailing 8 weeks; LightGBM monthly, "
        "trained on days up to two days before the month",
        f"- Weather: {inputs['weather_status']}",
        f"- Official archive: {inputs['official']['delivery_date'].n_unique()} days parsed",
        f"- Python {platform.python_version()} · polars {pl.__version__} · lightgbm {__import__('lightgbm').__version__}"
        f" · statsforecast {__import__('statsforecast').__version__}",
    ]))
    headline(rep, bt, off, inputs["weather_status"])
    bt_days = windows["backtest"]
    write_window_section(rep, f"2. Backtest {bt_days[0]} .. {bt_days[-1]} - the ladder", bt, figs, "backtest")
    rep.p(f"![by hour](figures/{figs['backtest_hour'].name})")
    if "leak" in bt:
        rep.h("3. What the leaky naive would have been worth")
        rep.p("Same hours, same days. `naive_leaky` is the spec's original y(t-24); it uses values that are "
              "not published until after the 12:30 origin for every delivery hour from 11:00 EST on.\n")
        rep.table(bt["leak"].with_columns(pl.col("model").replace_strict(LABELS, default=pl.col("model"))))
    if bt["best_a"]:
        residual_section(rep, bt["main"], bt["best_a"]["model"], "backtest")
    if not bt["gains"].is_empty():
        rep.h("Feature importance (gain %, last monthly refit of the backtest)")
        rep.table(bt["gains"].pivot(on="model", index="feature", values="gain_pct")
                  .sort(bt["gains"]["model"][0], descending=True))
    if off:
        write_window_section(rep, "4. Official comparison - against IESO's own day-ahead forecast", off, figs, "official")
        rep.p(f"![by hour](figures/{figs['official_hour'].name})")
        rep.p(f"![daily](figures/{figs['official_daily'].name})")
    else:
        rep.h("4. Official comparison")
        rep.p("No overlapping days between the DATotals archive and the actuals.")
    if inputs["coverage"] is not None:
        rep.h("Weather coverage (fraction of hours with a value, by city and year)")
        rep.table(inputs["coverage"], max_rows=30)
    eda_appendix(rep, inputs)
    rep.p(f"\n---\nRun time {(time.time() - started) / 60:.1f} min.")
    REPORT.write_text(rep.text(), encoding="utf-8")
    log(f"wrote {REPORT} and {len(list(RESULTS.glob('*.csv')))} tables, {len(list(FIGURES.glob('*.png')))} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
