#!/usr/bin/env python3
"""v2 under the pre-registered design in docs/spec-v2.md.

    python -m src.evaluate.run_v2              # about 60-80 minutes on Colab
    python -m src.evaluate.run_v2 --quick      # small windows, for checking the plumbing
    python -m src.evaluate.run_v2 --fresh      # ignore cached predictions
    python -m src.evaluate.run_v2 --no-weather # Track A only

Reads the staging files written by
    src.ingest.fetch_demand + src.ingest.demand      actuals
    src.ingest.fetch_weather                         observed temperature (Track B)
    src.ingest.fetch_weather_forecast                legal archived forecasts (Track C)

Writes reports/v2/findings_auto.md, reports/v2/results/*.csv, reports/v2/figures/*.png.

Nothing here chooses anything by looking at the holdout, prospective or seen
windows. The threshold and the ensemble weights come from the development
window (2023); everything else was fixed in the spec before this file existed.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

from src.evaluate import metrics, plots
from src.evaluate.run import Report, attach, day_list, log, md_table
from src.ingest import forecasts as official_io
from src.models import baselines, ensemble, lgbm, probabilistic
from src.transform import features as F

STAGING = Path("data/staging/demand_hourly.parquet")
WEATHER = Path("data/staging/weather_hourly.parquet")
WEATHER_FC = Path("data/staging/weather_fc_hourly.parquet")
WEATHER_FC_META = Path("data/staging/weather_fc_meta.json")
WEATHER_FC_COVERAGE = Path("data/staging/weather_fc_coverage.csv")
OUT = Path("reports/v2")
RESULTS, FIGURES, REPORT = OUT / "results", OUT / "figures", OUT / "findings_auto.md"
CACHE_ROOT = Path("data/curated/v2_cache")

WINDOWS = {
    "dev": (date(2023, 1, 1), date(2023, 12, 31)),
    "holdout": (date(2026, 1, 1), date(2026, 6, 13)),
    "seen_backtest": (date(2024, 1, 1), date(2025, 12, 31)),
}
QUICK_WINDOWS = {
    "dev": (date(2023, 1, 1), date(2023, 2, 28)),
    "holdout": (date(2026, 1, 1), date(2026, 2, 28)),
    "seen_backtest": (date(2025, 1, 1), date(2025, 2, 28)),
}
FREEZE_FALLBACK = date(2026, 10, 20)
FC_FALLBACK_START = date(2021, 3, 1)
NIGHT, DAY = (24, 1, 2, 3, 4, 5), tuple(range(8, 21))
QUANTILES = probabilistic.QUANTILES

LABELS = {
    "naive": "Naive (legal: D-1 or D-2)",
    "seasonal_naive": "Seasonal naive, y(t-168)",
    "mstl_ets": "MSTL + ETS",
    "mstl_arima": "MSTL + ARIMA",
    "lgbm_A_expanding": "LightGBM A (from 2003)",
    "lgbm_A_2021": "LightGBM A (forecast-archive window)",
    "lgbm_B_expanding": "LightGBM B, observed temp (UPPER BOUND)",
    "lgbm_B_2021": "LightGBM B, forecast-archive window (UPPER BOUND)",
    "lgbm_C_2021": "LightGBM C, GFS forecast temp",
    "ens_A_global": "Ensemble A (LightGBM A + MSTL+ETS)",
    "ens_A_seasonal": "Ensemble A, seasonal weights",
    "ens_C_global": "Ensemble C (LightGBM C + MSTL+ETS)",
    "ens_C_seasonal": "Ensemble C, seasonal weights",
    "ieso_official": "IESO day-ahead forecast",
    "q_lgbm": "Quantile LightGBM",
    "q_baseline": "Threshold + empirical error quantiles",
    "market": "Market Demand",
    "ontario": "Ontario Demand",
    "exports": "Exports (Market - Ontario)",
}
OPERATIONAL = ["naive", "seasonal_naive", "mstl_ets", "mstl_arima", "lgbm_A_expanding",
               "ens_A_global", "ens_A_seasonal", "lgbm_C_2021", "ens_C_global", "ens_C_seasonal"]
REFERENCES = ["lgbm_A_2021", "lgbm_B_2021", "lgbm_B_expanding"]


# ------------------------------------------------------------------ provenance


def git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def freeze_date() -> tuple[date, str]:
    """Date of the last commit that can change a score (spec-v2 section 1)."""
    stamp = git("log", "-1", "--format=%cs", "--", "src", ":(exclude)src/evaluate/plots.py")
    if stamp:
        return date.fromisoformat(stamp), "git: last commit touching src/ (plots excluded)"
    return FREEZE_FALLBACK, "fallback constant (no git history available)"


# ------------------------------------------------------------------ cache


class Cache:
    """Predictions keyed by code version, window span, target and model, so a
    disconnected Colab session resumes instead of starting again."""

    def __init__(self, enabled: bool, quick: bool) -> None:
        self.enabled = enabled
        self.root = CACHE_ROOT / f"{git('rev-parse', '--short', 'HEAD') or 'nogit'}{'-quick' if quick else ''}"

    def path(self, window: str, days: list[date], target: str, model: str) -> Path:
        span = f"{days[0]:%Y%m%d}-{days[-1]:%Y%m%d}-{len(days)}"
        return self.root / f"{window}__{span}__{target}__{model}.parquet"

    def get_or_run(self, window: str, days: list[date], target: str, model: str, run) -> pl.DataFrame:
        path = self.path(window, days, target, model)
        if self.enabled and path.exists():
            log(f"  {window}/{target}/{model}: cached")
            return pl.read_parquet(path)
        started = time.time()
        out = run()
        if self.enabled:
            path.parent.mkdir(parents=True, exist_ok=True)
            out.write_parquet(path)
        log(f"  {window}/{target}/{model}: {time.time() - started:.0f}s")
        return out


# ------------------------------------------------------------------ inputs


def load_inputs(use_weather: bool) -> dict:
    from src.analysis import eda

    if not STAGING.exists():
        sys.exit(f"missing {STAGING}; run the ingest first")
    staging = pl.read_parquet(STAGING)
    excluded = eda.exclusion_dates(eda.blackout_window(eda.load()))

    from src.ingest import fetch_weather

    weather_obs, weather_cities, obs_status = None, None, "not requested"
    if use_weather:
        try:
            if fetch_weather.main() != 0 or not WEATHER.exists():
                raise RuntimeError("fetch_weather failed")
            weather_cities = pl.read_parquet(WEATHER)
            weather_obs = weather_cities.select("ts_est", "temp")
            obs_status = "available"
        except Exception as error:
            obs_status = f"unavailable - {type(error).__name__}: {error}"
    obs_through = fetch_weather.complete_through(weather_obs)

    weather_fc, fc_meta, fc_start, fc_status = None, {}, None, "not requested"
    if use_weather:
        if WEATHER_FC_META.exists():
            fc_meta = json.loads(WEATHER_FC_META.read_text())
        verdict = (fc_meta.get("verification") or {}).get("verdict")
        if verdict == "fail":
            fc_status = "verification FAILED - Track C not run"
        elif not WEATHER_FC.exists() or not fc_meta.get("first_full_month"):
            fc_status = "no forecast data (run src.ingest.fetch_weather_forecast first)"
        else:
            weather_fc = pl.read_parquet(WEATHER_FC).select("ts_est", "temp")
            year, month = map(int, fc_meta["first_full_month"].split("-"))
            fc_start = date(year, month, 1)
            fc_status = f"available - model {fc_meta.get('model')}, rule {fc_meta.get('rule')}, verification {verdict}"

    frame_obs = F.build(staging, excluded, weather_obs)
    temp_daily = None
    if weather_obs is not None:
        temp_daily = (frame_obs.group_by("delivery_date").agg(pl.col("temp_day_mean").first())
                      .filter(pl.col("temp_day_mean").is_not_null()))
    return {
        "staging": staging, "excluded": excluded,
        "frame_obs": frame_obs,
        "frame_fc": F.build(staging, excluded, weather_fc) if weather_fc is not None else None,
        "frames_target": {t: F.build(staging, excluded, target=t) for t in ("ontario", "exports")},
        "obs_status": obs_status, "fc_status": fc_status, "fc_meta": fc_meta, "fc_start": fc_start,
        "has_obs": weather_obs is not None, "has_fc": weather_fc is not None,
        "obs_through": obs_through, "weather_cities": weather_cities,
        "temp_daily": temp_daily, "official": official_io.load(),
    }


def official_days(frame: pl.DataFrame, official: pl.DataFrame, quick: bool) -> list[date]:
    if official.is_empty():
        return []
    have = (frame.filter(pl.col("y").is_not_null()).group_by("delivery_date")
            .agg(pl.len().alias("n")).filter(pl.col("n") >= 23)["delivery_date"])
    days = sorted(set(official["delivery_date"].to_list()) & set(have.to_list()))
    return days[-21:] if quick else days


# ------------------------------------------------------------------ models


def point_models(window: str, days: list[date], inp: dict, cache: Cache, n_jobs: int) -> pl.DataFrame:
    """Market-demand point forecasts for one window. Which models run where is spec-v2 section 2-4."""
    fr, fc = inp["frame_obs"], inp["frame_fc"]
    start = inp["fc_start"] or FC_FALLBACK_START
    dev = window == "dev"
    runs = {
        "naive": lambda: baselines.naive(fr, days),
        "seasonal_naive": lambda: baselines.seasonal_naive(fr, days),
        "lgbm_A_expanding": lambda: lgbm.forecast(fr, days, "A", "expanding", n_jobs)[0],
    }
    if inp["has_fc"]:
        runs["lgbm_C_2021"] = lambda: lgbm.forecast(fc, days, "C", "since", n_jobs, train_start=start,
                                                     min_train_days=365)[0]
    through = inp["obs_through"]
    b_days = [d for d in days if through is not None and d <= through]
    if not dev:
        if inp["has_fc"]:
            runs["lgbm_A_2021"] = lambda: lgbm.forecast(fr, days, "A", "since", n_jobs, train_start=start,
                                                         min_train_days=365)[0]
        # Track B is an upper bound only with the whole observed day, so it skips days whose
        # observations are not fully published yet (spec-v2 s.7.2). Those hours then drop out
        # of the window's common sample for every model, and come back on the next rerun.
        if inp["has_obs"] and b_days:
            if window != "seen_backtest":       # v1 already reports the 2003-window B there (s.7.1)
                runs["lgbm_B_expanding"] = lambda: lgbm.forecast(fr, b_days, "B", "expanding", n_jobs)[0]
            if inp["has_fc"]:                   # the Track C control runs everywhere (s.7.1)
                runs["lgbm_B_2021"] = lambda: lgbm.forecast(fr, b_days, "B", "since", n_jobs, train_start=start,
                                                             min_train_days=365)[0]
    parts = [cache.get_or_run(window, b_days if name.startswith("lgbm_B") else days, "market", name, run)
             .with_columns(pl.lit(name).alias("model")) for name, run in runs.items()]
    which = ("mstl_ets",) if dev else ("mstl_ets", "mstl_arima")
    for name in which:
        out = cache.get_or_run(window, days, "market", name,
                               lambda name=name: baselines.classical(fr, days, n_jobs=n_jobs, which=(name,)).get(
                                   name, pl.DataFrame(schema={"ts_est": fr.schema["ts_est"], "yhat": pl.Float64})))
        parts.append(out.with_columns(pl.lit(name).alias("model")))
    return pl.concat(parts, how="vertical_relaxed")


def add_ensembles(preds: pl.DataFrame, weights: dict) -> pl.DataFrame:
    extra = []
    for name, (component, w) in weights.items():
        if w and component in preds["model"].unique().to_list():
            extra.append(ensemble.combine(preds, component, "mstl_ets", w, name))
    return pl.concat([preds, *extra], how="vertical_relaxed") if extra else preds


def quantile_models(window: str, days: list[date], inp: dict, cache: Cache, threshold: str,
                    n_jobs: int) -> pl.DataFrame:
    track, frame = ("C", inp["frame_fc"]) if inp["has_fc"] else ("A", inp["frame_obs"])
    start = inp["fc_start"] or FC_FALLBACK_START
    q = cache.get_or_run(window, days, "market", f"q_lgbm_{track}",
                         lambda: lgbm.forecast_quantiles(frame, days, track, "since", QUANTILES, n_jobs,
                                                         train_start=start, min_train_days=365))
    b = cache.get_or_run(window, days, "market", f"q_baseline_{threshold}",
                         lambda: probabilistic.empirical_quantiles(inp["frame_obs"], days, threshold, QUANTILES))
    out = pl.concat([q.with_columns(pl.lit("q_lgbm").alias("model")),
                     b.with_columns(pl.lit("q_baseline").alias("model"))], how="vertical_relaxed")
    return out.join(inp["frame_obs"].select("ts_est", "y"), on="ts_est", how="left")


def target_models(window: str, days: list[date], inp: dict, cache: Cache, n_jobs: int) -> pl.DataFrame:
    """Ontario Demand and exports with the same three models (spec-v2 section 5)."""
    parts = []
    for target, fr in inp["frames_target"].items():
        runs = {
            "naive": lambda fr=fr: baselines.naive(fr, days),
            "seasonal_naive": lambda fr=fr: baselines.seasonal_naive(fr, days),
            "lgbm_A_expanding": lambda fr=fr: lgbm.forecast(fr, days, "A", "expanding", n_jobs)[0],
        }
        for name, run in runs.items():
            out = cache.get_or_run(window, days, target, name, run)
            parts.append(out.with_columns(pl.lit(name).alias("model"), pl.lit(target).alias("target"))
                         .join(fr.select("ts_est", "y", "h"), on="ts_est", how="left"))
    return pl.concat(parts, how="vertical_relaxed")


# ------------------------------------------------------------------ evaluation


def ladder_for(threshold: str, present: list[str]) -> list[str]:
    order = [threshold, "mstl_ets", "mstl_arima", "lgbm_A_expanding", "ens_A_global", "lgbm_C_2021", "ens_C_global"]
    return [m for m in order if m in present]


def test_family(ladder: list[str], present: list[str], threshold: str, best_op: str | None,
                with_ieso: bool) -> list[tuple[str, str, str]]:
    """(model, benchmark, reason) - the pre-declared Diebold-Mariano family."""
    pairs: list[tuple[str, str, str]] = []
    for upper, lower in zip(ladder[1:], ladder[:-1]):
        pairs.append((upper, lower, "adjacent rung"))
    for m in ladder[1:] + [v for v in ("ens_A_seasonal", "ens_C_seasonal") if v in present]:
        pairs.append((m, threshold, "vs threshold"))
    for m, b in (("lgbm_C_2021", "lgbm_A_2021"), ("ens_A_global", "lgbm_A_expanding"),
                 ("ens_A_global", "mstl_ets")):
        pairs.append((m, b, "specified"))
    if with_ieso and best_op:
        pairs.append((best_op, "ieso_official", "specified"))
    seen, family = set(), []
    for m, b, why in pairs:
        if m in present and b in present and m != b and (m, b) not in seen:
            seen.add((m, b))
            family.append((m, b, why))
    return family


def evaluate(scored: pl.DataFrame, threshold: str, best_op: str | None, with_ieso: bool,
             decompose: bool) -> dict:
    present_all = scored["model"].unique().to_list()
    contenders = [m for m in OPERATIONAL + REFERENCES + (["ieso_official"] if with_ieso else []) if m in present_all]
    main = metrics.common_sample(scored.filter(pl.col("model").is_in(contenders)))
    present = main["model"].unique().to_list()
    out = {"main": main, "n_days": main["delivery_date"].n_unique(), "n_hours": main["ts_est"].n_unique(),
           "summary": metrics.summary(main) if main.height else pl.DataFrame()}
    if not main.height:
        return out
    ladder = ladder_for(threshold, present)
    family = test_family(ladder, present, threshold, best_op, with_ieso)
    tests = [{**metrics.diebold_mariano(main, m, b), "reason": why} for m, b, why in family]
    holm = metrics.holm([t["p_value"] for t in tests])
    out["dm"] = pl.DataFrame([{**t, "p_holm": None if h is None else float(f"{h:.2g}")} for t, h in zip(tests, holm)])
    out["ladder"] = ladder
    if decompose:
        out["by"] = {dim: metrics.summary(main, [dim]) for dim in ("hour_ending", "local_hour", "season", "day_type", "temp_band")
                     if main[dim].null_count() < main.height}
    return out


def observed_coverage(inp: dict, eval_days: list[date]) -> pl.DataFrame | None:
    """Per city and month, the share of evaluated hours with an observed temperature.

    Only months where some city is below 100% are returned. Hours after the last
    complete observed day are left out: they are missing because they are not
    published yet, which the report states separately.
    """
    cities = inp.get("weather_cities")
    if cities is None or not eval_days:
        return None
    through = inp.get("obs_through")
    keep = [d for d in eval_days if through is None or d <= through]
    names = [c for c in ("toronto", "ottawa", "london") if c in cities.columns]
    grid = (inp["frame_obs"].filter(pl.col("delivery_date").is_in(keep)).select("ts_est", "delivery_date")
            .join(cities.select("ts_est", *names).with_columns(pl.col("ts_est").cast(inp["frame_obs"].schema["ts_est"])),
                  on="ts_est", how="left"))
    out = (grid.with_columns(pl.col("delivery_date").dt.strftime("%Y-%m").alias("month"))
           .group_by("month").agg([pl.col(c).is_not_null().mean().round(3).alias(c) for c in names])
           .sort("month"))
    return out.filter(pl.any_horizontal([pl.col(c) < 1 for c in names]))


def mae_of(summary: pl.DataFrame, model: str) -> float | None:
    row = summary.filter(pl.col("model") == model)
    return row["MAE"].item() if row.height else None


def p_holm(facts: dict, model: str, benchmark: str) -> float | None:
    dm = facts.get("dm")
    if dm is None or dm.is_empty():
        return None
    row = dm.filter((pl.col("model") == model) & (pl.col("benchmark") == benchmark))
    return row["p_holm"].item() if row.height else None


def fmt_p(p: float | None) -> str:
    if p is None:
        return "n/a"
    return "< 0.0001" if p < 1e-4 else f"{p:.2g}"


# ------------------------------------------------------------------ report pieces


def ladder_table(facts: dict) -> pl.DataFrame:
    s = facts["summary"]
    order = {m: i for i, m in enumerate(OPERATIONAL + REFERENCES + ["ieso_official"])}
    return (s.with_columns(pl.col("model").replace_strict(order, default=99).alias("_o"))
            .sort("_o").drop("_o").with_columns(pl.col("model").replace_strict(LABELS, default=pl.col("model"))))


def dm_table(facts: dict) -> pl.DataFrame:
    dm = facts.get("dm")
    if dm is None or dm.is_empty():
        return pl.DataFrame()
    return dm.select("model", "benchmark", "reason", "n_days", "mean_diff", "dm_stat", "p_value", "p_holm")


def wide(summary: pl.DataFrame, dim: str, value: str, order: list[str]) -> pl.DataFrame:
    present = [m for m in order if m in summary["model"].unique().to_list()]
    out = summary.pivot(on="model", index=dim, values=value).sort(dim)
    return out.select([dim, *present]).rename({m: LABELS.get(m, m) for m in present})


def exports_decomposition(t: pl.DataFrame, market: pl.DataFrame) -> pl.DataFrame:
    """LightGBM A MAE by delivery hour for the three targets, on hours all three share."""
    m = market.filter(pl.col("model") == "lgbm_A_expanding").select(
        "ts_est", (pl.col("yhat") - pl.col("y")).abs().alias("market"))
    parts = []
    for target in ("ontario", "exports"):
        parts.append(t.filter((pl.col("model") == "lgbm_A_expanding") & (pl.col("target") == target))
                     .select("ts_est", "h", (pl.col("yhat") - pl.col("y")).abs().alias(target)))
    joined = parts[0].join(parts[1].drop("h"), on="ts_est").join(m, on="ts_est").drop_nulls()
    return (joined.group_by("h").agg(pl.col("market").mean().round(0), pl.col("ontario").mean().round(0),
                                     pl.col("exports").mean().round(0), pl.len().alias("n_hours"))
            .with_columns((pl.col("h") + 1).alias("hour_ending"),
                          (pl.col("exports") / (pl.col("exports") + pl.col("ontario"))).round(3).alias("export_share"))
            .sort("hour_ending").select("hour_ending", "market", "ontario", "exports", "export_share", "n_hours"))


def exports_verdict(table: pl.DataFrame) -> dict:
    night = table.filter(pl.col("hour_ending").is_in(NIGHT))
    day = table.filter(pl.col("hour_ending").is_in(DAY))
    share_night, share_day = night["export_share"].mean(), day["export_share"].mean()
    exp_night, ont_night = night["exports"].mean(), night["ontario"].mean()
    c1, c2 = share_night > share_day, exp_night >= ont_night
    return {"share_night": round(share_night, 3), "share_day": round(share_day, 3),
            "exports_mae_night": round(exp_night), "ontario_mae_night": round(ont_night),
            "criterion_1": c1, "criterion_2": c2, "supported": bool(c1 and c2)}


# ------------------------------------------------------------------ main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--no-weather", action="store_true")
    parser.add_argument("--jobs", type=int, default=-1)
    args = parser.parse_args()
    started = time.time()
    for d in (RESULTS, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
    cache = Cache(enabled=not args.fresh, quick=args.quick)
    frozen, frozen_how = freeze_date()

    log("loading inputs")
    inp = load_inputs(use_weather=not args.no_weather)
    log(f"observed weather: {inp['obs_status']} | forecast weather: {inp['fc_status']}")
    fr = inp["frame_obs"]
    spans = QUICK_WINDOWS if args.quick else WINDOWS
    days = {name: day_list(*span) for name, span in spans.items()}
    od = official_days(fr, inp["official"], args.quick)

    # ---- development window: the only place choices look at results ----------------
    log(f"window dev: {len(days['dev'])} days")
    dev = attach(point_models("dev", days["dev"], inp, cache, args.jobs), fr, inp["temp_daily"])
    simple = metrics.summary(metrics.common_sample(dev.filter(pl.col("model").is_in(["naive", "seasonal_naive"]))))
    threshold = simple.sort("MAE")["model"][0]
    weights = {}
    for component, prefix in (("lgbm_A_expanding", "ens_A"), ("lgbm_C_2021", "ens_C")):
        if component in dev["model"].unique().to_list():
            for variant in ("global", "seasonal"):
                weights[f"{prefix}_{variant}"] = (component, ensemble.fit_weights(dev, component, "mstl_ets", variant))
    log(f"threshold (dev): {threshold}; ensemble weights (dev): {{{', '.join(f'{k}: {v[1]}' for k, v in weights.items())}}}")

    # ---- evaluation windows -------------------------------------------------------
    facts: dict[str, dict] = {}
    scored: dict[str, pl.DataFrame] = {}
    qpred: dict[str, pl.DataFrame] = {}
    tpred: dict[str, pl.DataFrame] = {}
    plan = [("holdout", days["holdout"]), ("seen_backtest", days["seen_backtest"])]
    if od:
        plan.append(("official", od))
    for name, window_days in plan:
        log(f"window {name}: {len(window_days)} days {window_days[0]} .. {window_days[-1]}")
        preds = add_ensembles(point_models(name, window_days, inp, cache, args.jobs), weights)
        if name == "official":
            off = inp["official"].filter(pl.col("delivery_date").is_in(window_days)).with_columns(
                (pl.col("delivery_date").cast(pl.Datetime("us")) + pl.duration(hours=pl.col("hour_ending") - 1))
                .cast(fr.schema["ts_est"]).alias("ts_est"))
            preds = pl.concat([preds, off.select("ts_est", pl.col("total_load_mw").alias("yhat"),
                                                 pl.lit("ieso_official").alias("model"))], how="vertical_relaxed")
        scored[name] = attach(preds, fr, inp["temp_daily"])
        if name != "seen_backtest":          # probabilistic: holdout and official archive only (spec-v2 s.6)
            qpred[name] = quantile_models(name, window_days, inp, cache, threshold, args.jobs)
        if name != "official":
            tpred[name] = target_models(name, window_days, inp, cache, args.jobs)

    facts["holdout"] = evaluate(scored["holdout"], threshold, None, False, decompose=True)
    op = facts["holdout"]["summary"].filter(pl.col("model").is_in(OPERATIONAL)) if facts["holdout"]["summary"].height else None
    best_op = op.sort("MAE")["model"][0] if op is not None and op.height else None
    facts["seen_backtest"] = evaluate(scored["seen_backtest"], threshold, best_op, False, decompose=True)
    windows_q = {"holdout": qpred["holdout"]}
    if "official" in scored:
        off_scored = scored["official"]
        seen = off_scored.filter(pl.col("delivery_date") <= frozen)
        prosp = off_scored.filter(pl.col("delivery_date") > frozen)
        facts["official_seen"] = evaluate(seen, threshold, best_op, True, decompose=False) if seen.height else {}
        facts["prospective"] = evaluate(prosp, threshold, best_op, True, decompose=True) if prosp.height else {}
        dd = qpred["official"].with_columns(pl.col("ts_est").dt.date().alias("_d"))
        windows_q["official_seen"] = dd.filter(pl.col("_d") <= frozen).drop("_d")
        windows_q["prospective"] = dd.filter(pl.col("_d") > frozen).drop("_d")

    # ---- write results ---------------------------------------------------------------
    for name, f in facts.items():
        if f and f.get("summary") is not None and f["summary"].height:
            f["summary"].write_csv(RESULTS / f"{name}_summary.csv")
            if f.get("dm") is not None:
                f["dm"].write_csv(RESULTS / f"{name}_dm_tests.csv")
            for dim, table in (f.get("by") or {}).items():
                table.write_csv(RESULTS / f"{name}_by_{dim}.csv")
    qsum = {}
    for name, q in windows_q.items():
        if q.height and q["y"].drop_nulls().len():
            qsum[name] = (metrics.quantile_summary(q), metrics.reliability(q))
            qsum[name][0].write_csv(RESULTS / f"{name}_quantile_summary.csv")
            qsum[name][1].write_csv(RESULTS / f"{name}_reliability.csv")
    decomp = {}
    for name, t in tpred.items():
        decomp[name] = exports_decomposition(t, scored[name])
        decomp[name].write_csv(RESULTS / f"{name}_exports_by_hour.csv")
        metrics.summary(t.with_columns(pl.col("ts_est").dt.date().alias("delivery_date")), ["target"]).write_csv(
            RESULTS / f"{name}_targets_summary.csv")
    verdict = exports_verdict(decomp["holdout"]) if "holdout" in decomp and decomp["holdout"].height else None

    # ---- figures ---------------------------------------------------------------------
    figs = {}
    role = {best_op: plots.SERIES[0], threshold: plots.SERIES[1], "ieso_official": plots.SERIES[2]}
    hf = facts["holdout"]
    if hf["summary"].height:
        figs["ladder"] = plots.ladder(hf["summary"], LABELS, best_op, mae_of(hf["summary"], threshold),
                                      FIGURES / "holdout_ladder.png",
                                      f"holdout {spans['holdout'][0]} .. {spans['holdout'][1]}: MAE by model",
                                      threshold_label=f"threshold: {LABELS[threshold]}")
        by_hour = hf["by"]["hour_ending"].select("model", pl.col("hour_ending").alias("x"), pl.col("MAE").alias("value"))
        figs["by_hour"] = plots.lines_by(by_hour, "x", [best_op, threshold, "mstl_ets"], LABELS,
                                         FIGURES / "holdout_by_hour.png", "holdout: MAE by delivery hour",
                                         "hour-ending (EST)", colours={**role, "mstl_ets": plots.INK_2})
    if "holdout" in qsum:
        figs["reliability"] = plots.reliability_plot(qsum["holdout"][1], ["q_lgbm", "q_baseline"], LABELS,
                                                     FIGURES / "holdout_reliability.png", "holdout: quantile reliability")
    if "holdout" in decomp and decomp["holdout"].height:
        d = decomp["holdout"].unpivot(index="hour_ending", on=["market", "ontario", "exports"],
                                      variable_name="model", value_name="value").rename({"hour_ending": "x"})
        figs["exports"] = plots.lines_by(d, "x", ["ontario", "exports", "market"], LABELS,
                                         FIGURES / "holdout_exports_by_hour.png",
                                         "holdout: LightGBM A error by target and delivery hour", "hour-ending (EST)",
                                         colours={"ontario": plots.SERIES[0], "exports": plots.SERIES[1],
                                                  "market": plots.INK_2})
    pf = facts.get("prospective") or {}
    if pf.get("n_days", 0) >= 7:
        daily = (pf["main"].group_by("model", "delivery_date").agg((pl.col("yhat") - pl.col("y")).abs().mean().alias("value"))
                 .sort("delivery_date").rename({"delivery_date": "x"}))
        figs["prospective"] = plots.lines_by(daily, "x", [best_op, threshold, "ieso_official"], LABELS,
                                             FIGURES / "prospective_daily.png", "prospective window: daily MAE",
                                             "delivery date", colours=role)

    # ---- report ------------------------------------------------------------------------
    rep = Report()
    rep.p("# v2 results - machine output\n")
    rep.p(f"Generated {datetime.now():%Y-%m-%d %H:%M} by `src/evaluate/run_v2.py` under `docs/spec-v2.md`"
          f"{' (QUICK MODE - plumbing check only, not results)' if args.quick else ''}.\n")
    rep.p("**Do not edit this file by hand.** Interpretation belongs in `reports/findings.md`.\n")

    rep.h("0. Run")
    meta = inp["fc_meta"]
    ver = meta.get("verification") or {}
    lines = [
        f"- Code: commit {git('rev-parse', '--short', 'HEAD') or 'unknown'}; freeze date **{frozen}** ({frozen_how}); "
        f"prospective window = delivery days after {frozen}",
        f"- Windows: " + "; ".join(f"{k} {v[0]} .. {v[-1]} ({len(v)} days)" for k, v in days.items())
        + (f"; official archive {od[0]} .. {od[-1]} ({len(od)} days)" if od else ""),
        f"- Threshold (chosen on dev): **{LABELS[threshold]}** - dev MAE "
        + ", ".join(f"{LABELS[r['model']]} {r['MAE']:,.0f}" for r in simple.iter_rows(named=True)),
        "- Ensemble weights (fitted on dev, w = LightGBM share): "
        + ("; ".join(f"{k}: {v[1]}" for k, v in weights.items()) or "none"),
        f"- Observed weather: {inp['obs_status']}"
        + (f"; complete through **{inp['obs_through']}** (Track B is not scored after that day)"
           if inp["obs_through"] else ""),
        f"- Forecast weather: {inp['fc_status']}",
        f"- Forecast archive first full month: {meta.get('first_full_month', 'n/a')} "
        f"(Track C and its controls train from {inp['fc_start'] or 'n/a'})",
        f"- Verification of the forecast-run mapping: **{ver.get('verdict', 'n/a')}**"
        + (f" - day1 {ver['day1']['share']:.1%} of {ver['day1']['tested']} values, "
           f"day2 {ver['day2']['share']:.1%} of {ver['day2']['tested']} at the expected lead"
           if "day1" in ver else (f" - {ver.get('reason', '')}" if ver else "")),
        f"- Quantile model: track {'C' if inp['has_fc'] else 'A'}, q = {QUANTILES}",
        f"- Best operational model on holdout (used as 'best' in later windows): "
        f"{LABELS.get(best_op, best_op)}",
        f"- Python {platform.python_version()} · polars {pl.__version__} · lightgbm {__import__('lightgbm').__version__}"
        f" · statsforecast {__import__('statsforecast').__version__}",
    ]
    rep.p("\n".join(lines))
    if "day1" in ver:
        rep.p("\nMatching leads found in verification (hours between run start and valid time; expected "
              "24-29 for day1, 48-53 for day2):\n")
        rep.table(pl.DataFrame([{"series": k, "leads": lead, "values": n}
                                for k in ("day1", "day2") for lead, n in ver[k]["matching_leads"].items()]))

    # headline
    rep.h("1. Headline (generated, no interpretation) - clean holdout unless stated")
    s = hf["summary"]
    head = []
    if s.height:
        t_mae = mae_of(s, threshold)
        head.append(f"- Threshold ({LABELS[threshold]}, chosen on dev): holdout MAE **{t_mae:,.0f} MW**.")
        b = s.filter(pl.col("model") == best_op).row(0, named=True)
        head.append(f"- Best operational model: **{LABELS[best_op]}**, MAE {b['MAE']:,.0f} MW, ME {b['ME']:+,.0f} MW "
                    f"({(b['MAE'] / t_mae - 1) * 100:+.1f}% vs threshold; Holm-adjusted p {fmt_p(p_holm(hf, best_op, threshold))}).")
        a21, c, b21 = (mae_of(s, m) for m in ("lgbm_A_2021", "lgbm_C_2021", "lgbm_B_2021"))
        if None not in (a21, c, b21):
            line = (f"- Track C: forecast weather worth **{a21 - c:,.0f} MW** (A {a21:,.0f} -> C {c:,.0f}, same "
                    f"training window; Holm p {fmt_p(p_holm(hf, 'lgbm_C_2021', 'lgbm_A_2021'))}). ")
            if a21 - b21 <= 0:
                line += (f"Perfect weather (B {b21:,.0f}) was worth nothing here, so the share of the weather gap "
                         f"that C recovers is undefined.")
            elif c < b21:
                line += (f"C scored below the perfect-weather control (B {b21:,.0f}), so a 'share recovered' would "
                         f"exceed 100% and is not meaningful - most likely noise.")
            else:
                line += (f"Forecast error costs {c - b21:,.0f} MW against perfect weather (B {b21:,.0f}); C recovers "
                         f"{(a21 - c) / (a21 - b21):.0%} of the weather gap.")
            head.append(line)
        else:
            head.append(f"- Track C did not run: {inp['fc_status']}.")
        la, et, ea = (mae_of(s, m) for m in ("lgbm_A_expanding", "mstl_ets", "ens_A_global"))
        if None not in (la, et, ea):
            head.append(f"- Ensemble A (w = {weights['ens_A_global'][1]['all']}): MAE {ea:,.0f} vs LightGBM A {la:,.0f} "
                        f"(Holm p {fmt_p(p_holm(hf, 'ens_A_global', 'lgbm_A_expanding'))}) and MSTL+ETS {et:,.0f} "
                        f"(Holm p {fmt_p(p_holm(hf, 'ens_A_global', 'mstl_ets'))}).")
    if "holdout" in qsum:
        qs = {r["model"]: r for r in qsum["holdout"][0].iter_rows(named=True)}
        if "q_lgbm" in qs and "q_baseline" in qs:
            ql, qb = qs["q_lgbm"], qs["q_baseline"]
            head.append(f"- Probabilistic: pinball {ql['mean_pinball']:,.0f} vs baseline {qb['mean_pinball']:,.0f}; "
                        f"90% interval covers {ql['coverage_90_pct']}% (baseline {qb['coverage_90_pct']}%), "
                        f"50% covers {ql['coverage_50_pct']}% (baseline {qb['coverage_50_pct']}%).")
    if verdict:
        head.append(f"- Exports hypothesis (spec-v2 section 5): **{'supported' if verdict['supported'] else 'not supported'}** - "
                    f"export share of error {verdict['share_night']:.0%} at night vs {verdict['share_day']:.0%} by day; "
                    f"night MAE exports {verdict['exports_mae_night']:,} vs Ontario {verdict['ontario_mae_night']:,}.")
    n_p = pf.get("n_days", 0)
    if n_p:
        po = pf["summary"]
        ieso, best = mae_of(po, "ieso_official"), mae_of(po, best_op)
        head.append(f"- Prospective: {n_p} days so far{' - too few to conclude anything' if n_p < 30 else ''}; "
                    f"IESO {ieso:,.0f} MW vs {LABELS[best_op]} {best:,.0f} MW." if ieso and best else
                    f"- Prospective: {n_p} days so far.")
    else:
        head.append(f"- Prospective: no delivery days after the freeze date ({frozen}) yet. Rerun the notebook to add them.")
    rep.p("\n".join(head))

    # holdout detail
    rep.h(f"2. Holdout {spans['holdout'][0]} .. {spans['holdout'][1]} - clean")
    if s.height:
        rep.p(f"{hf['n_days']} days, {hf['n_hours']:,} hours, common sample. Track B rows are upper bounds, not forecasts.\n")
        rep.table(ladder_table(hf))
        if "ladder" in figs:
            rep.p(f"![ladder](figures/{figs['ladder'].name})")
        rep.p("\n**Pre-declared Diebold-Mariano family** (negative mean_diff = model better; conclusions use p_holm):\n")
        rep.table(dm_table(hf))
        order = [m for m in OPERATIONAL + REFERENCES if m in s["model"].to_list()]
        for dim, label in (("hour_ending", "MAE by delivery hour (EST hour-ending)"), ("season", "MAE by season"),
                           ("day_type", "MAE by day type"), ("temp_band", "MAE on extreme-temperature days"),
                           ("local_hour", "MAE by local clock hour")):
            if dim in hf.get("by", {}):
                rep.p(f"\n**{label}**\n")
                rep.table(wide(hf["by"][dim], dim, "MAE", order), max_rows=30)
        if "by_hour" in figs:
            rep.p(f"![by hour](figures/{figs['by_hour'].name})")

    # Track C across windows
    rep.h("3. Track C - the value of forecast weather")
    rows = []
    for name in ("holdout", "prospective", "seen_backtest", "official_seen"):
        f = facts.get(name) or {}
        if not f or f.get("summary") is None or not f["summary"].height:
            continue
        a21, c, b21 = (mae_of(f["summary"], m) for m in ("lgbm_A_2021", "lgbm_C_2021", "lgbm_B_2021"))
        if None in (a21, c, b21):
            continue
        rows.append({"window": name, "clean": name in ("holdout", "prospective"), "A_same_window": a21, "C": c,
                     "B_same_window": b21, "value_of_forecast_mw": round(a21 - c), "cost_of_forecast_error_mw": round(c - b21),
                     "recovered_share": round((a21 - c) / (a21 - b21), 3) if a21 - b21 > 0 and c >= b21 else None,
                     "p_holm_C_vs_A": p_holm(f, "lgbm_C_2021", "lgbm_A_2021")})
    rep.table(pl.DataFrame(rows) if rows else pl.DataFrame())

    # ensembles
    rep.h("4. Ensembles (weights fitted on dev 2023, then frozen)")
    rep.table(pl.DataFrame([{"ensemble": k, "component": v[0], "weights": json.dumps(v[1])} for k, v in weights.items()])
              if weights else pl.DataFrame())
    for name in ("holdout", "seen_backtest"):
        f = facts.get(name) or {}
        if f.get("by", {}).get("season") is not None:
            ens_order = [m for m in ("mstl_ets", "lgbm_A_expanding", "ens_A_global", "ens_A_seasonal",
                                     "lgbm_C_2021", "ens_C_global", "ens_C_seasonal") if m in f["summary"]["model"].to_list()]
            rep.p(f"\n**{name}{' (SEEN in v1)' if name == 'seen_backtest' else ''} - MAE by season**\n")
            rep.table(wide(f["by"]["season"], "season", "MAE", ens_order))

    # probabilistic
    rep.h("5. Probabilistic forecasts")
    for name, (summary_q, rel) in qsum.items():
        rep.p(f"\n**{name}{' (SEEN)' if name in ('seen_backtest', 'official_seen') else ''}**\n")
        rep.table(summary_q.with_columns(pl.col("model").replace_strict(LABELS, default=pl.col("model"))))
        rep.table(rel.pivot(on="model", index=["q", "nominal_pct"], values="pct_below").sort("q"))
    if "reliability" in figs:
        rep.p(f"![reliability](figures/{figs['reliability'].name})")

    # exports
    rep.h("6. Ontario Demand and exports (spec-v2 section 5)")
    if verdict:
        rep.p(f"Criterion on holdout, LightGBM A: (1) export share of error higher at night (hour-ending 24, 1-5) than "
              f"by day (8-20): {verdict['share_night']} vs {verdict['share_day']} -> {verdict['criterion_1']}; "
              f"(2) night export MAE >= night Ontario MAE: {verdict['exports_mae_night']} vs {verdict['ontario_mae_night']} "
              f"-> {verdict['criterion_2']}. **Hypothesis {'supported' if verdict['supported'] else 'not supported'}.**\n")
    for name, table in decomp.items():
        rep.p(f"\n**{name}{' (SEEN)' if name == 'seen_backtest' else ''} - LightGBM A MAE by delivery hour**\n")
        rep.table(table, max_rows=30)
    if "exports" in figs:
        rep.p(f"![exports](figures/{figs['exports'].name})")
    osf = facts.get("official_seen") or {}
    if osf.get("main") is not None and "seen_backtest" in decomp:
        ieso = (osf["main"].filter(pl.col("model") == "ieso_official").group_by("hour_ending")
                .agg((pl.col("yhat") - pl.col("y")).abs().mean().alias("ieso")))
        joined = ieso.join(decomp["seen_backtest"].select("hour_ending", "exports", "ontario"), on="hour_ending")
        if joined.height > 3:
            r_exp = joined.select(pl.corr("ieso", "exports")).item()
            r_ont = joined.select(pl.corr("ieso", "ontario")).item()
            rep.p(f"\nDescriptive only (24 points, different windows): correlation across delivery hours between IESO's "
                  f"error (official, seen) and our export error (2024-25) r = {r_exp:.2f}; with our Ontario error "
                  f"r = {r_ont:.2f}. IESO's forecast cannot be split into the two parts, so this cannot test the hypothesis "
                  f"for IESO.\n")

    # prospective
    rep.h("7. Prospective window - clean, grows with every rerun")
    if n_p:
        rep.p(f"{n_p} days after the freeze date ({frozen}).\n")
        rep.table(ladder_table(pf))
        rep.table(dm_table(pf))
        if "prospective" in figs:
            rep.p(f"![prospective](figures/{figs['prospective'].name})")
    else:
        rep.p(f"No delivery days after {frozen} yet.")

    # seen
    rep.h("8. Seen windows - for comparison with v1 only, not evidence")
    for name, title in (("seen_backtest", "2024-2025 backtest"), ("official_seen", "official window up to the freeze")):
        f = facts.get(name) or {}
        if f.get("summary") is not None and f["summary"].height:
            rep.p(f"\n**{title}** - {f['n_days']} days\n")
            rep.table(ladder_table(f))
            rep.table(dm_table(f))

    obs_cov = observed_coverage(inp, sorted({d for v in days.values() for d in v} | set(od)))
    if obs_cov is not None:
        rep.h("Observed-weather coverage in evaluated months (share of hours with a value)")
        if obs_cov.height:
            rep.p("Only months where a city falls short are listed. A missing city drops out of the "
                  "average, which shifts it (Ottawa runs colder), so Track B sees a biased temperature there.\n")
            rep.table(obs_cov)
        else:
            rep.p("All three cities have a value for every hour of every evaluated month.")
    if WEATHER_FC_COVERAGE.exists():
        cov = pl.read_csv(WEATHER_FC_COVERAGE).with_columns(pl.col("month").str.slice(0, 4).alias("year"))
        rep.h("Forecast-weather coverage (share of hours with a value)")
        rep.table(cov.group_by("year").agg(pl.col("coverage").min().alias("worst_month"),
                                           pl.col("coverage").mean().round(3).alias("mean")).sort("year"))
    rep.p(f"\n---\nData: Open-Meteo (CC BY 4.0), IESO, Environment and Climate Change Canada. "
          f"Run time {(time.time() - started) / 60:.1f} min.")
    REPORT.write_text(rep.text(), encoding="utf-8")
    log(f"wrote {REPORT}, {len(list(RESULTS.glob('*.csv')))} tables, {len(list(FIGURES.glob('*.png')))} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
