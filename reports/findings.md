# Findings — Ontario hourly demand forecasting, v1

**Status:** design, data and EDA findings final · **model results pending the first
real run of `notebooks/02_results.ipynb`** (2026-10-19)

Every number in §5–§6 will come from `findings_auto.md`, the machine-written output of
`src/evaluate/run.py`. This file interprets those numbers; it never restates a figure
the machine did not produce. Data-quality detail lives in `data_quality.md`.

---

## 1. The question

Forecast Ontario's hourly **Market Demand** (MW) for every hour of tomorrow, the way
IESO's own day-ahead forecast does: made the day before at about 12:30, using only
what is known at that moment. Judge every model against a ladder of baselines, and
against IESO's published day-ahead forecast on the days the harvester has archived.

Market Demand rather than Ontario Demand because IESO's forecast is built on it: its
`Total Load` sits +61 MW from Market Demand and +2,911 MW from Ontario Demand, the
difference being scheduled exports (spec §1).

## 2. The clock, and what it changed

The forecast origin is D-1 at about 12:30. The archived IESO files say so directly:
`CreatedAt` is 12:31 on 125 of 127 days, and all 127 were created the day before
delivery. Whether that stamp is EST or EDT is not stated, so the data cutoff takes the
stricter reading: hours of D-1 beginning before **11:00 EST** are known, later ones
are not. An hour too strict can only make our models look slightly worse; an hour too
lenient would leak.

**This exposed a leak in the original spec.** It defined the naive baseline as
y(t-24) and listed a 24-hour lag as a feature. For a target hour h on day D, t-24 is
hour h on D-1 — still in the future at the origin whenever h ≥ 11. Thirteen of the
24 hours of every forecast would have read values not yet published. So:

- the legal naive is "the latest observed value at the same clock hour": D-1 for
  hours 0–10, D-2 for hours 11–23;
- the 24-hour lag is used only where it is legal, and is empty otherwise;
- every rolling statistic is taken as of the cutoff, not the target hour.

`tests/test_features.py` enforces this: it overwrites every observation at or after
the cutoff with nonsense and checks that no feature for day D changes, with a control
showing the check is not vacuous. The spec's original y(t-24) is kept as a labelled
diagnostic, `naive_leaky`, so the run reports what the leak would have been worth in
MW (`findings_auto.md` §3). This is the same family of error as the look-ahead
problem in the MSc thesis that motivated the project.

Forecast days are IESO delivery days: EST dates, always 24 hours, hour-ending 1–24.
That is the unit IESO publishes and the unit scored here, so DST never creates a 23-
or 25-hour forecast day. Behavioural features (local hour, weekday, holidays) use
Toronto local time.

## 3. What the data looks like (Milestones 1–2, real data)

**The clock is right.** The raw files are fixed EST. Evidence that could have failed
and did not: exactly 24 rows on every delivery date including DST days; the 2003
blackout collapsing in the first full hour after its documented 16:11 EDT start; and a
falsification test on the morning load ramp across all 48 DST transitions — mean
shift 0.28 h (spring) and 0.16 h (fall) against controls of 0.20 h and 0.27 h, where
a one-hour error would push the first figure towards 1.0. Details in
`data_quality.md` §3.

**The 2003 blackout lasted six days in the data, not seven hours.** Demand stayed
10–34% below expectation from 14 to 19 August 2003 during the conservation period
that followed. Those six days are flagged out of every model and every lag.

**Summer and winter are different shapes, not one shape at two heights.** The weekday
mean profiles cross twice (read from the EDA chart; exact values in the
`findings_auto.md` EDA appendix):

| | winter | summer |
|---|---|---|
| morning | steep rise from 06:00, plateau by 08:00 | slow climb into the afternoon |
| daytime | plateau with a slight midday dip | still rising |
| peak | about 18:00 local, ≈22,100 MW | about 17:00 local, ≈21,500 MW |
| overnight trough | ≈17,100 MW | ≈15,650 MW |

Winter has a morning plateau and an evening peak (lighting and heating); summer has
one broad afternoon peak (air conditioning). Hour of day has to interact with season,
which a tree model learns from hour and calendar features together and a single
fixed daily seasonal pattern cannot. The weekend drop also varies by season — about
1,200 MW at the winter peak against 1,750 MW at the summer peak — consistent with
commercial cooling switching off at weekends while residential evening load does not.

**Ontario is not purely summer-peaking.** Six of the 25 annual peaks were in winter
(2004, 2014, 2015, 2017, 2018, 2023). The spec's description was too simple, and the
seasonal error split is load-bearing.

**Demand is trending up again.** Annual means fell from 18,500–19,450 MW in 2005–08
to 17,189 MW in 2017, then rose: 18,022 MW in 2024, 18,896 MW in 2025. The
2026-07-02 peak of 27,678 MW is the highest since 2011. That trend shaped two model
decisions in §5.

Still to be read from the real run, in the `findings_auto.md` EDA appendix: the
weekend drop by season (§3), the cost of older training data (§5), the day- and
week-on-week movement of the series (§6) and which holidays matter (§7).

## 4. How models are judged

Fixed before any model was run (spec §8: framework before models):

- **Rolling origin**, one forecast per delivery day, each from its own 12:30 origin.
- **Backtest:** 2024-01-01 to 2025-12-31, two complete years with two summers and two
  winters. **Official window:** every archived day with actuals, 2026-06-14 onward.
- **Common sample:** models are compared only on hours every one of them forecast.
- **Metrics:** MAE (primary), RMSE, MAPE, ME. ME is forecast minus actual, so
  positive means over-forecast — the convention of spec §1, not of the thesis.
- **Decomposition:** by delivery hour, local clock hour, season, day type, and
  extreme-temperature days (the 5% hottest and coldest, needs weather).
- **Is the gap real?** A Diebold-Mariano test on daily MAE with a Newey-West variance,
  because errors run in multi-day spells and a plain t-test would overstate the case.

The ladder: (1) legal naive, (2) seasonal naive y(t-168) — the threshold, (3) MSTL +
ETS, (4) MSTL + ARIMA, (5) LightGBM Track A. Track B is LightGBM with observed
temperature: an upper bound, never a forecast, because no one knows tomorrow's
observed temperature at 12:30 (spec §4).

## 5. Decisions made while building, and why

**LightGBM predicts the change from last week, not the level.** A tree cannot
forecast a value outside the range of targets it trained on. With demand setting new
highs, a level model would cap its forecasts exactly when they matter most. The model
learns y − y(t−168) and adds it back to last week's value.

**Its inputs are level-free.** Predicting the change was not enough on its own. In a
synthetic test with a built-in upward trend, a model fed raw lagged levels was biased
by about −90 MW in every month: in training, a high level usually meant a temporary
spike about to revert, so once the whole series trends upward the model forecasts
falls that never come. Replacing raw levels with differences and ratios between lags
brought the bias to within ±5 MW. Both choices rest on that structural argument and
were fixed before any real result existed; they were not chosen by comparing scores.

**Track B learns from weekly temperature change.** Since the target is the weekly
change in demand, the inputs include temperature now minus temperature a week ago,
alongside the levels and heating and cooling degree-hours.

**The classical rungs are MSTL with ETS or ARIMA on the remainder**, refitted at every
origin on the trailing eight weeks. A seasonal ARIMA cannot hold the daily and weekly
cycles together and does not fit at every origin in reasonable time; MSTL removes both
cycles and the ETS or ARIMA handles what is left.

**LightGBM refits monthly**, on target days up to two days before the month starts —
the last days fully known at the month's first origin. Hyperparameters are fixed in
`src/models/lgbm.py` and were never tuned on an evaluation window.

**Two holiday flags.** Ontario's nine statutory holidays, and a second flag for Civic
Holiday, Easter Monday and Remembrance Day, which are widely observed but not
statutory. The EDA holiday table will say whether the distinction matters.

## 6. Results

**Pending.** `notebooks/02_results.ipynb` produces them in one run. When
`findings_auto.md` comes back, this section answers, in this order:

1. Did LightGBM Track A beat seasonal naive (the spec's rule), by how much, and is the
   gap distinguishable from noise?
2. Where does each model win and lose — which hours, seasons, day types?
3. Is any model biased, and does the bias move with the season or the trend?
4. Do the residuals carry structure the model missed (daily autocorrelation)?
5. 3-year window or expanding window?
6. What would perfect weather knowledge be worth (Track B minus best Track A)?
7. How close does Track A come to IESO's own day-ahead forecast?

**Official-comparison data status.** The harvester has run without a gap since its
first day: 127 consecutive delivery days, 2026-06-14 to 2026-10-18, every file created
the day before delivery. Two were created later than usual (2026-07-01 at 13:31,
2026-09-24 at 14:25), giving IESO one to two hours more information on those two days.
At about four months this is inside the spec's three-to-six-month threshold, but it
covers only one summer and no winter; the comparison should be rerun as the archive
grows.

## 7. Limitations

1. **DST.** The series is fixed EST. Local wall-clock days have 23 and 25 hours at the
   transitions, the November fall-back hour appears twice in local time, and
   seasonal naive matches the same EST hour, which is one local hour off during the
   week after a transition.
2. **No weather forecasts.** Track A uses no weather at all; Track B uses observed
   weather and is an upper bound. The realistic middle — a model fed archived weather
   *forecasts* — is not built, because those archives were not available. IESO's own
   forecast does use weather forecasts, which matters when reading §6 item 7.
3. **Station representativeness.** Three airports averaged with equal weight. Toronto
   carries far more load than London; a load-weighted or population-weighted average
   would be better and was not attempted.
4. **The blackout window is approximate.** Without weather the method cannot tell
   conservation from a cool spell.
5. **The data cutoff is an assumption.** It assumes actual demand is published within
   the hour, and takes the stricter reading of IESO's unstated timestamp zone.
6. **One missing hour** (2025-05-01 00:00 EST) is left missing; the hours that depend
   on it drop out of the common sample.
7. **The official window is short and summer-heavy** (§6).
8. **Market Demand's definition** has not been checked for changes over 24 years.

## 8. What v2 would add

Probabilistic forecasts (quantile LightGBM), archived weather forecasts for an honest
weather-aware track, a load-weighted temperature, and the official comparison over a
full year once the archive covers one.
