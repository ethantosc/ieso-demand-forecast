#!/usr/bin/env python3
"""Computations for Milestone 2 EDA. Returns tables; the notebook draws them.

Splitting calculation from plotting is what makes this testable. Every function
here takes a DataFrame and returns a DataFrame, so it can be exercised against
synthetic data without a notebook or a screen.

--------------------------------------------------------------------------
WHAT THIS DELIBERATELY DOES NOT DO
--------------------------------------------------------------------------
No model is fitted and no forecast is scored. `diff_distribution` reports how
much the series moves from one day or one week to the next, which is a property
of the data, not a baseline result: there is no rolling origin and no train/test
split behind it. The seasonal-naive threshold gets set in Milestone 4, on top of
the backtest framework built in Milestone 3, and not before. Spec section 8 puts
the framework ahead of the models on purpose; producing a number that looks like
a score now would give a later decision something to anchor on.
"""

from __future__ import annotations

from datetime import date

import polars as pl

STAGING = "data/staging/demand_hourly.parquet"

TARGET = "market_demand_mw"

BLACKOUT_START = date(2003, 8, 14)
BLACKOUT_SEARCH_END = date(2003, 9, 15)
BLACKOUT_REFERENCE_YEARS = (2002, 2004, 2005, 2006)
BLACKOUT_DEFICIT_THRESHOLD = 0.10  # 10% below the expected level

SEASONS = {
    12: "winter", 1: "winter", 2: "winter",
    3: "shoulder", 4: "shoulder", 5: "shoulder",
    6: "summer", 7: "summer", 8: "summer",
    9: "shoulder", 10: "shoulder", 11: "shoulder",
}


# ---------------------------------------------------------------- loading


def load(path: str = STAGING) -> pl.DataFrame:
    """Read the staging Parquet and attach calendar columns."""
    frame = pl.read_parquet(path)
    return frame.with_columns(
        pl.col("local_date").dt.year().alias("year"),
        pl.col("local_date").dt.month().alias("month"),
        pl.col("local_date").dt.weekday().alias("weekday"),  # 1=Mon .. 7=Sun
    ).with_columns(
        pl.col("month").replace_strict(SEASONS, return_dtype=pl.Utf8).alias("season"),
        (pl.col("weekday") >= 6).alias("is_weekend"),
    )


def daily(frame: pl.DataFrame) -> pl.DataFrame:
    """Collapse to one row per local date. Partial days are marked, not dropped."""
    return (
        frame.group_by("local_date")
        .agg(
            pl.col(TARGET).mean().alias("mean_mw"),
            pl.col(TARGET).max().alias("peak_mw"),
            pl.col(TARGET).min().alias("trough_mw"),
            pl.len().alias("hours"),
            pl.col("year").first(),
            pl.col("month").first(),
            pl.col("weekday").first(),
            pl.col("season").first(),
            pl.col("is_weekend").first(),
        )
        .sort("local_date")
    )


# ------------------------------------------------- 0. blackout exclusion


def blackout_window(frame: pl.DataFrame) -> pl.DataFrame:
    """Find how long August 2003 demand stayed suppressed, from the data.

    The blackout began 2003-08-14. Ontario then ran an emergency conservation
    period, so demand stayed below normal for days afterwards at levels well
    above the 5,000 MW floor used in the Milestone 1 checks - low enough to
    distort a profile, high enough to pass unnoticed.

    Method: express each day as a fraction of its own year's mean, which removes
    the long-run level trend, then compare 2003 against the median of the same
    calendar date in neighbouring years. Days more than 10% below that
    expectation are reported.

    LIMITATION, and it is not a small one: with no weather data in v1, this
    cannot separate conservation from a cool spell. The window is therefore
    approximate and biased towards over-exclusion. Record it in data_quality.md
    as such.
    """
    days = daily(frame)
    annual = days.group_by("year").agg(pl.col("mean_mw").mean().alias("year_mean"))

    normalised = days.join(annual, on="year").with_columns(
        (pl.col("mean_mw") / pl.col("year_mean")).alias("rel"),
        pl.col("local_date").dt.strftime("%m-%d").alias("md"),
    )

    reference = (
        normalised.filter(pl.col("year").is_in(BLACKOUT_REFERENCE_YEARS))
        .group_by("md")
        .agg(pl.col("rel").median().alias("rel_expected"))
    )

    subject = normalised.filter(
        (pl.col("local_date") >= BLACKOUT_START)
        & (pl.col("local_date") <= BLACKOUT_SEARCH_END)
    )

    return (
        subject.join(reference, on="md", how="left")
        .with_columns(
            (1 - pl.col("rel") / pl.col("rel_expected")).alias("deficit"),
        )
        .with_columns(
            (pl.col("deficit") > BLACKOUT_DEFICIT_THRESHOLD).alias("suppressed"),
        )
        .select(
            "local_date", "weekday", "mean_mw",
            pl.col("rel").round(4), pl.col("rel_expected").round(4),
            pl.col("deficit").round(4), "suppressed",
        )
        .sort("local_date")
    )


def exclusion_dates(window: pl.DataFrame) -> list[date]:
    """Contiguous run of suppressed days starting at the blackout.

    Stops at the first normal day, so an unrelated dip later in the search
    range is not swept in.
    """
    keep: list[date] = []
    for row in window.iter_rows(named=True):
        if not row["suppressed"]:
            break
        keep.append(row["local_date"])
    return keep


def mark_excluded(frame: pl.DataFrame, excluded: list[date]) -> pl.DataFrame:
    """Add `excluded_reason`. Nothing is deleted - every row stays in the table.

    Analyses filter on this column. Keeping the rows means the exclusion is
    visible and reversible rather than baked into a shorter file.
    """
    if excluded:
        flag = (
            pl.when(pl.col("local_date").is_in(excluded))
            .then(pl.lit("2003 blackout and conservation period"))
            .otherwise(pl.lit(None, dtype=pl.Utf8))
        )
    else:
        flag = pl.lit(None, dtype=pl.Utf8)
    return frame.with_columns(flag.alias("excluded_reason"))


def clean(frame: pl.DataFrame) -> pl.DataFrame:
    """Rows the EDA should use: not excluded, and not the final partial day."""
    last_full = (
        daily(frame).filter(pl.col("hours") == 24)["local_date"].max()
    )
    return frame.filter(
        pl.col("excluded_reason").is_null() & (pl.col("local_date") <= last_full)
    )


# ------------------------------------------------------ 1. the DST test


def _ramp_hour(hours: list[int], values: list[float]) -> float | None:
    """Local hour at which the morning ramp passes its halfway point.

    Linear interpolation between hourly points, because a whole-hour answer
    cannot resolve the very error being tested for.
    """
    pairs = sorted(zip(hours, values))
    window = [(h, v) for h, v in pairs if 1 <= h <= 12]
    if len(window) < 8:
        return None
    lows = [v for _, v in window]
    threshold = min(lows) + 0.5 * (max(lows) - min(lows))
    for (h0, v0), (h1, v1) in zip(window, window[1:]):
        if v0 < threshold <= v1:
            if v1 == v0:
                return float(h1)
            return h0 + (threshold - v0) / (v1 - v0) * (h1 - h0)
    return None


def morning_ramp(frame: pl.DataFrame) -> pl.DataFrame:
    """Per-date morning ramp hour, weekdays only.

    The morning ramp is driven by alarm clocks, so in a correctly converted
    series it sits at the same LOCAL hour year round. If the raw file were wall
    clock and the ingest wrongly added an hour, the ramp would jump by exactly
    one hour at each DST transition. That makes this falsifiable, which
    Milestone 1's sections 7 and 8 were not: they were derived from the same
    assumption they appeared to confirm.
    """
    rows = []
    for (day,), group in frame.filter(~pl.col("is_weekend")).group_by(
        ["local_date"], maintain_order=True
    ):
        value = _ramp_hour(group["local_hour"].to_list(), group[TARGET].to_list())
        if value is not None:
            rows.append({"local_date": day, "ramp_hour": round(value, 3)})
    return pl.DataFrame(rows).sort("local_date")


def dst_transitions(frame: pl.DataFrame) -> pl.DataFrame:
    """Dates where the UTC offset changes, read off the data itself."""
    offsets = (
        frame.select("local_date", "utc_offset_hours")
        .group_by("local_date")
        .agg(pl.col("utc_offset_hours").min().alias("off_min"),
             pl.col("utc_offset_hours").max().alias("off_max"))
        .sort("local_date")
    )
    # Both transition days span offsets -5 and -4, so the offset cannot tell them
    # apart. The hour count can: a spring-forward local date has 23 hours, a
    # fall-back date has 25.
    counts = frame.group_by("local_date").agg(pl.len().alias("hours"))
    return (
        offsets.filter(pl.col("off_min") != pl.col("off_max"))
        .join(counts, on="local_date")
        .select(
            "local_date",
            pl.when(pl.col("hours") == 23)
            .then(pl.lit("spring forward"))
            .when(pl.col("hours") == 25)
            .then(pl.lit("fall back"))
            .otherwise(pl.lit("irregular"))
            .alias("kind"),
        )
        .sort("local_date")
    )


def dst_ramp_test(frame: pl.DataFrame, days: int = 10) -> pl.DataFrame:
    """Mean ramp hour before vs after each transition, with a control.

    The control is the same comparison centred three weeks earlier, where no
    transition occurs. It shows how much the ramp hour naturally drifts between
    two nearby weeks, which is the yardstick for reading the treatment shift.
    """
    ramps = morning_ramp(frame)
    lookup = dict(zip(ramps["local_date"].to_list(), ramps["ramp_hour"].to_list()))

    def window_mean(centre: date, lo: int, hi: int) -> float | None:
        values = [
            v for d, v in lookup.items()
            if lo <= (d - centre).days <= hi
        ]
        return sum(values) / len(values) if values else None

    rows = []
    for row in dst_transitions(frame).iter_rows(named=True):
        centre = row["local_date"]
        control_centre = date.fromordinal(centre.toordinal() - 21)
        before = window_mean(centre, -days, -1)
        after = window_mean(centre, 1, days)
        c_before = window_mean(control_centre, -days, -1)
        c_after = window_mean(control_centre, 1, days)
        if None in (before, after, c_before, c_after):
            continue
        rows.append({
            "transition": centre,
            "kind": row["kind"],
            "ramp_before": round(before, 2),
            "ramp_after": round(after, 2),
            "shift": round(after - before, 2),
            "control_shift": round(c_after - c_before, 2),
        })
    return pl.DataFrame(rows)


# ------------------------------------------------------ 2-4. the shapes


def intraday_profile(frame: pl.DataFrame) -> pl.DataFrame:
    """Mean demand by local hour, split by season and weekday/weekend."""
    return (
        frame.group_by(["season", "is_weekend", "local_hour"])
        .agg(pl.col(TARGET).mean().round(0).alias("mean_mw"),
             pl.col(TARGET).quantile(0.9).round(0).alias("p90_mw"),
             pl.len().alias("n"))
        .sort(["season", "is_weekend", "local_hour"])
    )


def weekly_profile(frame: pl.DataFrame) -> pl.DataFrame:
    """Mean demand by day of week, split by season."""
    return (
        frame.group_by(["season", "weekday"])
        .agg(pl.col(TARGET).mean().round(0).alias("mean_mw"), pl.len().alias("n"))
        .sort(["season", "weekday"])
    )


def monthly_profile(frame: pl.DataFrame) -> pl.DataFrame:
    """Mean and peak demand by calendar month, per year."""
    return (
        frame.group_by(["year", "month"])
        .agg(pl.col(TARGET).mean().round(0).alias("mean_mw"),
             pl.col(TARGET).max().round(0).alias("peak_mw"))
        .sort(["year", "month"])
    )


def annual_peaks(frame: pl.DataFrame) -> pl.DataFrame:
    """When each year peaked. Tests the summer-peaking claim in spec 2.2."""
    return (
        frame.group_by("year")
        .agg(
            pl.col(TARGET).max().round(0).alias("peak_mw"),
            pl.col("ts_local").get(pl.col(TARGET).arg_max()).alias("peak_ts_local"),
            pl.col("month").get(pl.col(TARGET).arg_max()).alias("peak_month"),
            pl.col("local_hour").get(pl.col(TARGET).arg_max()).alias("peak_hour"),
            pl.col(TARGET).mean().round(0).alias("mean_mw"),
        )
        .with_columns(
            pl.col("peak_month").replace_strict(SEASONS, return_dtype=pl.Utf8)
            .alias("peak_season")
        )
        .sort("year")
    )


# -------------------------------------------- 5. how stale is old data


def level_bias_by_window(
    frame: pl.DataFrame,
    windows: tuple[int, ...] = (1, 2, 3, 5, 10, 99),
    first_eval_year: int = 2013,
) -> pl.DataFrame:
    """If you set the demand level from the last W years, how wrong are you?

    For each evaluation year Y and window length W, the estimate is the mean
    demand over years Y-W .. Y-1, and the error is estimate minus actual. This
    turns spec section 5's choice between a 3-year rolling window and a
    full-history expanding window into a measured trade-off instead of a taste:
    short windows track the level but see fewer years of weather, long windows
    are stable but carry a stale level.

    Reported separately for summer and winter because the trend has not moved
    them equally.
    """
    per_year = (
        frame.group_by(["year", "season"])
        .agg(pl.col(TARGET).mean().alias("mean_mw"))
    )
    overall = frame.group_by("year").agg(pl.col(TARGET).mean().alias("mean_mw")) \
        .with_columns(pl.lit("all").alias("season"))
    combined = pl.concat([per_year, overall], how="diagonal").sort(["season", "year"])

    years = sorted(frame["year"].unique().to_list())
    complete = [y for y in years if y != max(years)]  # drop the partial current year

    rows = []
    for season in ("all", "summer", "winter", "shoulder"):
        table = {r["year"]: r["mean_mw"] for r in
                 combined.filter(pl.col("season") == season).iter_rows(named=True)}
        for window in windows:
            errors = []
            for evaluation in complete:
                if evaluation < first_eval_year:
                    continue
                history = [table[y] for y in complete
                           if evaluation - window <= y < evaluation and y in table]
                if len(history) < min(window, 1) or not history:
                    continue
                if evaluation not in table:
                    continue
                errors.append(sum(history) / len(history) - table[evaluation])
            if not errors:
                continue
            rows.append({
                "season": season,
                "window_years": window,
                "n_eval_years": len(errors),
                "mean_bias_mw": round(sum(errors) / len(errors), 1),
                "mean_abs_bias_mw": round(sum(abs(e) for e in errors) / len(errors), 1),
                "worst_abs_mw": round(max(abs(e) for e in errors), 1),
            })
    return pl.DataFrame(rows)


# ------------------------------------ 6. how much the series moves


def diff_distribution(frame: pl.DataFrame) -> pl.DataFrame:
    """Absolute change over 24h and 168h, by season.

    NOT a baseline score. There is no rolling origin, no train/test split and no
    model here - this is the spread of the series itself, used only to calibrate
    expectations before the framework exists. The seasonal-naive threshold is
    set in Milestone 4.
    """
    ordered = frame.sort("ts_utc").with_columns(
        (pl.col(TARGET) - pl.col(TARGET).shift(24)).abs().alias("abs_d24"),
        (pl.col(TARGET) - pl.col(TARGET).shift(168)).abs().alias("abs_d168"),
    )
    rows = []
    for column in ("abs_d24", "abs_d168"):
        grouped = (
            ordered.filter(pl.col(column).is_not_null())
            .group_by("season")
            .agg(
                pl.col(column).median().round(0).alias("median_mw"),
                pl.col(column).mean().round(0).alias("mean_mw"),
                pl.col(column).quantile(0.9).round(0).alias("p90_mw"),
                pl.len().alias("n"),
            )
            .with_columns(pl.lit(column).alias("lag"))
        )
        rows.append(grouped)
    return pl.concat(rows).select(
        "lag", "season", "median_mw", "mean_mw", "p90_mw", "n"
    ).sort(["lag", "season"])


# ------------------------------------------------- 7. holiday effect


def holiday_effect(frame: pl.DataFrame, holiday_dates: set[date]) -> pl.DataFrame:
    """Holiday daily mean against the same weekday in the same month and year.

    Comparing like with like matters: Christmas Day falls on a different weekday
    each year, and a holiday landing on a Monday is not comparable to the
    weekend around it.
    """
    days = daily(frame).with_columns(
        pl.col("local_date").is_in(list(holiday_dates)).alias("is_holiday")
    )
    reference = (
        days.filter(~pl.col("is_holiday"))
        .group_by(["year", "month", "weekday"])
        .agg(pl.col("mean_mw").mean().alias("reference_mw"))
    )
    return (
        days.filter(pl.col("is_holiday"))
        .join(reference, on=["year", "month", "weekday"], how="left")
        .with_columns(
            (pl.col("mean_mw") - pl.col("reference_mw")).round(0).alias("delta_mw"),
            ((pl.col("mean_mw") / pl.col("reference_mw") - 1) * 100)
            .round(1).alias("delta_pct"),
        )
        .select("local_date", "year", "weekday", "mean_mw",
                pl.col("reference_mw").round(0), "delta_mw", "delta_pct")
        .sort("local_date")
    )


def holiday_summary(effect: pl.DataFrame, names: dict[date, str]) -> pl.DataFrame:
    """Average effect per named holiday across all years."""
    labelled = effect.with_columns(
        pl.col("local_date").map_elements(
            lambda d: names.get(d, "unknown"), return_dtype=pl.Utf8
        ).alias("holiday")
    )
    return (
        labelled.group_by("holiday")
        .agg(
            pl.len().alias("n_years"),
            pl.col("delta_pct").mean().round(1).alias("mean_delta_pct"),
            pl.col("delta_pct").min().round(1).alias("min_delta_pct"),
            pl.col("delta_pct").max().round(1).alias("max_delta_pct"),
        )
        .sort("mean_delta_pct")
    )
