# Findings — Ontario hourly demand forecasting

| | Results from | Code | Plan |
|---|---|---|---|
| **v2** — Part I | run of 2026-10-21 03:08 UTC | commit `ab9d3b7` | [`docs/spec-v2.md`](../docs/spec-v2.md), pre-registered |
| **v1** — Part II, final and unchanged | run of 2026-10-20 05:02 UTC | commit `5011a08` | [`docs/spec.md`](../docs/spec.md) |

Every v2 number comes from [`v2/findings_auto.md`](v2/findings_auto.md) or the tables
beside it in `v2/results/` — machine output of `src/evaluate/run_v2.py` — cited as
*(v2 auto §n)*. Anything marked **post hoc** was written after the results were seen,
was not pre-registered, and is not evidence for any v2 conclusion; from the next rerun
on, those checks come from `scripts/diagnose_v2.py`, which writes `v2/diagnostics.md`.

---

# Part I — v2

## A. What v2 asked, and where it was judged

v1 ended with four leads: archived weather forecasts, an ensemble of LightGBM with
MSTL + ETS, Ontario Demand against exports, and probabilistic forecasts. Each one was
suggested by v1's results, so judging them on v1's evaluation windows would have been
choosing on the test set. `docs/spec-v2.md` fixed where and how v2 would be scored,
and was committed before any v2 model code existed:

- **Development year, 2023** — the only window whose results drive choices: which
  naive is the threshold, and the ensemble weights.
- **Clean holdout, 2026-01-01 to 2026-06-13** (164 days) — never scored in v1. The v2
  verdicts come from here.
- **Prospective window** — delivery days after the freeze date, 2026-10-20. Empty so
  far. Each rerun of the notebook adds the days since; any later change to model code
  moves the freeze date and empties it again.
- **Seen windows** — the 2024–25 backtest and the IESO window (2026-06-14 to
  2026-10-19) — reported for comparison with v1, not as evidence.

A gap counts only when its Diebold-Mariano p-value, Holm-corrected within the window's
pre-declared family of tests, is below 0.05. The holdout's limits were written down in
advance: winter and spring plus the first 13 days of June, and no IESO forecasts (the
archive starts 2026-06-14).

## B. Answers in brief

1. **Weather forecasts are the largest gain in the project so far.** LightGBM with
   archived GFS temperature forecasts — using only model runs published before the
   12:30 origin — scores **659 MW** on the holdout, against 856 MW for the same model
   without weather trained over the same years: **197 MW, Holm p < 0.0001**
   *(v2 auto §1, §3)*.
2. **The best operational model is Ensemble C** (0.8 × LightGBM C + 0.2 × MSTL + ETS):
   654 MW, 36.5% below the threshold's 1,030 MW. Its step over LightGBM C alone,
   4.6 MW a day, is **not significant** (Holm p 0.54).
3. **v1's summer and hot-day weakness was a missing-weather problem.** Without weather,
   LightGBM is again the worst of the non-naive models in the holdout's June days.
   With forecast temperature it beats MSTL + ARIMA, the best model without weather,
   by 239 MW (§D).
4. **Without weather, LightGBM's edge over the classical models does not hold up.** On
   the holdout MSTL + ARIMA is ahead, 824 against 864 MW (Holm p 0.5); in the 2024–25
   backtest LightGBM was ahead by 33 MW a day (Holm p 0.054). Neither gap is
   significant.
5. **The overnight-exports hypothesis from v1 is not supported** by the criterion fixed
   in advance. At night our Ontario Demand error (561 MW) is larger than our export
   error (406 MW); in fact Ontario Demand's error is the larger one at every hour of the
   holdout *(v2 auto §6)*.
6. **Quantile LightGBM is sharp but overconfident.** It beats the baseline on pinball
   loss (226 against 320), but its 90% interval contains only 67% of outcomes and its
   50% interval 30%. The baseline is close to nominal (89% and 46%) with a 90% interval
   2.4 times as wide *(v2 auto §5)*.
7. **IESO is still ahead**, on the seen window only: 470 MW against 621 MW for
   Ensemble C (Holm p < 0.0001). LightGBM A, v1's best real forecast, scores 866 MW
   there.
   A clean comparison has to come from the prospective window.

## C. Track C — what forecast weather is worth

**The data.** Open-Meteo's Previous Runs API keeps, for every hour, what earlier GFS
runs forecast for it: `previous_day1` from the run about a day before, `previous_day2`
from about two days before. At the origin (D-1 11:00 EST, 16:00 UTC) the newest
published run is D-1 06Z. EST hours 0–6 of day D therefore use `previous_day1` (runs
24–29 h before the hour) and hours 7–23 use `previous_day2` (48–53 h). For hours 7–23
this is conservative: a fresher legal run exists and is not used (spec-v2 §3). The same
three airports as Track B, equal weight. The model is trained on the forecast series as
well, so it learns how far to trust it.

**Was the rule right?** The run checked before using it *(v2 auto §0)*. It compared 240
archived values, 120 per series, with individual GFS runs from Open-Meteo's Single Runs
API. Every value matched a run at the expected lead. 133 matched exactly one run, which
was always at the expected lead; the other 107 matched several, because neighbouring
runs often agree to 0.1 °C, and the expected lead was always among them. No value
matched only a later, unpublished run *(`v2/weather_fc_meta.json`)*. The check covered five dates from May to
September 2026; that the same mapping holds back to 2021 is assumed, not checked.

**Coverage.** The archive is complete from April 2021 (March 2021: 24%), so Track C and
its controls train from 2021-04-01. One gap: December 2023 (93%) and January 2024 (39%).
It touches the end of the development year and the start of the seen backtest; the
holdout is complete *(v2 auto, coverage table)*.

**Three quantities**, each against a control trained over the same years so that the
training window is not mixed in *(v2 auto §3)*:

| MAE (MW) | Holdout (clean) | IESO window (seen) |
|---|---|---|
| A_2021 — no weather | 856 | 861 |
| **C — GFS forecast temperature** | **659** | **630** |
| B_2021 — observed temperature, the "upper bound" | 676 | 646 |
| Value of forecast weather, A − C | 197 (Holm p < 0.0001) | 231 (Holm p 0.0044) |
| Cost of forecast error, C − B | −17 | −16 |

The value of forecast weather is large in every window, and 171 MW a day in the 2024–25
backtest as well *(v2 auto §8)*.

**The cost is negative, which should not happen.** Observed temperature is meant to be
an upper bound: knowing tomorrow's weather exactly should beat a two-day-old forecast
of it. Here the forecast won, by 16–17 MW, in both windows, so the spec's "share of the
upper bound recovered" is meaningless and the report declines to give it. The
pre-registered family has no test of C against B, so on the pre-registered evidence
this is unexplained.

**Unresolved.** The post-hoc check meant to settle it needed the run's own predictions,
and those were lost when the Colab session closed. Three explanations stay open:

1. **Noise.** No pre-registered test compares C with B.
2. **Gaps in the observed series.** A station-month that fails to download is not fatal:
   that city silently drops out of the three-city average for the month, and the
   average shifts (Ottawa runs colder). The first run did not report observed coverage
   by month, so this cannot be ruled out.
3. **A better temperature, not a worse one** — *hypothesis, not tested*. A gridded model
   temperature may track the temperature that drives load better than three airport
   thermometers do.

Since this run the code reports observed coverage by city and month, retries failed
downloads, and runs the diagnostic automatically at the end of every run
(spec-v2 §7.2). The next rerun can therefore tell (2) apart from the others and give an
exploratory test of (1).

The sign is not uniform *(v2 results, by temperature band)*: on the holdout's hottest
5% of days observed temperature does win (B_2021 822 against C 883 MW), as an upper
bound should; on the coldest 5% the forecast wins by a wide margin (700 against 836),
and on ordinary days by 15 MW.

## D. Where the errors went

**By season** *(v2 auto §2)*. The holdout's "summer" is 1–13 June.

| Holdout MAE (MW) | winter | shoulder | summer, 13 days |
|---|---|---|---|
| MSTL + ETS | 835 | 889 | 1,061 |
| MSTL + ARIMA | 801 | 808 | 1,032 |
| LightGBM A | 801 | 839 | 1,324 |
| **LightGBM C** | **643** | **650** | **793** |
| Ensemble C | 636 | 647 | 786 |

Without weather, LightGBM repeats v1's pattern: level with the classical models in
winter, worst of them in June. With forecast temperature the June weakness turns into
its largest lead. The seen 2024–25 backtest, with two full summers, says the same:
LightGBM C 755 MW in summer against 1,009 for MSTL + ETS and 1,032 for LightGBM A
*(v2 auto §4)*.

**Extreme-temperature days** — the hottest and coldest 5% of holdout days, 9 days each,
too few to stand alone. Hottest: LightGBM A 1,694, MSTL + ETS 1,468, LightGBM C 883 MW.
Coldest: LightGBM A 1,037 — the worst of all models on those days, naive included —
MSTL + ARIMA 802, LightGBM C 700.

**By hour.** Forecast weather helps most in the afternoon and evening, where the errors
are: from LightGBM A to LightGBM C, hour-ending 16 falls from 1,192 to 893 MW and
hour-ending 19 from 988 to 677, against 653 to 539 at hour-ending 3. The step at hour-ending 12, where yesterday's value stops
being known, is still there for every model (LightGBM C 742 → 831; the legal naive
1,080 → 1,352).

**Holidays.** LightGBM keeps the advantage its holiday flags gave it in v1: 929 MW for
LightGBM A against 1,198–1,312 for the MSTL models, over the holdout's few holidays.

## E. Ensembles

Weights were fitted on 2023 and frozen *(v2 auto §4)*. LightGBM A gets 0.7 (seasonal
version: 0.6 in summer, 0.7 otherwise); LightGBM C gets 0.8 in both versions, so
Ensemble C's global and seasonal variants are the same forecast. On the holdout:

- **Ensemble A** (823 MW) beats LightGBM A by 41 MW a day (Holm p 0.0004), but its
  lead over MSTL + ETS, 61 MW a day, is not significant (Holm p 0.11), and it only
  matches MSTL + ARIMA (824). Seasonal weights add nothing visible (820).
- **Ensemble C** (654 MW) is not significantly better than LightGBM C alone (Holm
  p 0.54).

In the seen backtest both steps are significant (Ensemble A over LightGBM A by 30 MW
a day, Ensemble C over LightGBM C by 24) *(v2 auto §8)*. The ensemble was proposed
because MSTL + ETS covered LightGBM's summer weakness. Forecast weather removed that
weakness, the development year already put 80% of the weight on LightGBM C, and what
is left is a small gain that the holdout cannot confirm.

## F. Ontario Demand against exports

The criterion, fixed in advance (spec-v2 §5, LightGBM A on the holdout): (1) exports'
share of the combined error is higher at night (hour-ending 24 and 1–5) than by day
(8–20), **and** (2) at night the export error is at least as large as Ontario Demand's.
(1) holds, 42% against 39%; (2) fails, 406 against 561 MW. **Not supported**
*(v2 auto §6)*.

![exports](v2/figures/holdout_exports_by_hour.png)

The figure shows why. The export error is nearly flat through the day (395–562 MW);
Ontario Demand's carries the daytime peak and is the larger of the two at every hour.
Exports weigh relatively more at night only because Ontario's own error is smallest
then. In the seen 2024–25 backtest the two are level at hours-ending 3–4 (exports 442,
Ontario 438–441 MW) and Ontario's is larger everywhere else.

One more pattern, not tested: the two errors partly cancel. Over the holdout,
LightGBM A's Market Demand error (864 MW) is well below the sum of its Ontario (731) and
export (484) errors *(v2 results, targets summary)* — consistent with exports
adjusting to Ontario's own demand.

**What this says about IESO.** Its forecast cannot be split into the two parts, so
nothing here can test why its error peaks at hour-ending 24. Descriptively, across
delivery hours IESO's error profile is negatively correlated with our export error
(r = −0.42) *(v2 auto §6)*. Nothing supports v1's guess that exports explain it.

## G. Probabilistic forecasts

*(v2 auto §5)*

| Holdout | pinball | 90% interval covers | 50% interval covers | 90% width (MW) |
|---|---|---|---|---|
| **Quantile LightGBM** (track C) | **226** | 67% | 30% | 1,693 |
| Legal naive + empirical error quantiles | 320 | 89% | 46% | 4,097 |

![reliability](v2/figures/holdout_reliability.png)

The quantile model's median is well placed (49% of outcomes below it) but both tails
are too short: 17% of outcomes fall below its 5% quantile, and only 84% below its 95%
quantile. The seen IESO window repeats it (68% and 29% coverage). Pinball loss rewards
its sharpness more than it penalises the miscalibration, but an interval that holds two
outcomes in three when it claims nine in ten is not usable as stated.

A likely reason, **not tested**: quantile trees learn the spread of their own training
residuals, which is narrower than the error on new days. Calibrating the intervals on
the development year (for example by conformal adjustment) is the obvious fix and would
have to be pre-registered for v3.

## H. Against IESO — seen window only

128 delivery days, 2026-06-14 to 2026-10-19 *(v2 auto §8)*.

| | MAE (MW) | ME (MW) |
|---|---|---|
| **IESO day-ahead forecast** | **470** | −176 |
| Ensemble C | 621 | +37 |
| LightGBM C | 630 | +41 |
| LightGBM B, observed temperature (upper bound; see §I) | 644 | +68 |
| LightGBM A (v1's model) | 866 | +60 |
| Naive, legal | 1,053 | +43 |

IESO is ahead of Ensemble C by 152 MW a day (Holm p < 0.0001). In v1 the gap to the best
real forecast was about 400 MW; forecast temperature closed most of it. IESO also still
under-forecasts this window by 176 MW on average, while ours is close to unbiased.

This window is not evidence. v1 scored it, and v1's results shaped v2. The clean version
of this comparison is the prospective window, which needs reruns to fill — and a winter.

## I. Seen windows — does v2 reproduce v1?

**The backtest, exactly.** For every model v1 had, the 2024–25 numbers are unchanged:
legal naive 995, seasonal naive 1,275, MSTL + ETS 874, MSTL + ARIMA 834, LightGBM A 801
*(v2 auto §8)*. The v2 changes to the shared code did not alter v1's models on real data.
The gap v1 left untested — LightGBM A against MSTL + ARIMA — is now tested: 33 MW a day,
Holm p 0.054.

**The IESO window, almost.** It has one more day than in v1 (2026-10-19). Most models
moved by a few MW (legal naive 1,056 → 1,053, LightGBM A 868 → 866; IESO 470 both
times). LightGBM B did not: 613 → 644.

**Unresolved**, for the same reason as in §C. If B's other 127 days were unchanged, its
error on 2026-10-19 alone would have to be about 4,600 MW. There are two candidate causes,
and neither is confirmed:

- **That day's observed weather was not yet fully published** when the run fetched it. On
  synthetic data, withholding part or all of a day's observations raised B's error on
  that day by 48–106 MW, not thousands, so this alone looks unlikely — though synthetic
  data is a weak guide.
- **A station-month failed to download in this run**, which would shift B's inputs over
  many days rather than one.

Both are now guarded or reported (spec-v2 §7.2): B is not scored on days whose observed
weather is not fully published, and the report lists every evaluated month where a city
falls short. Track C uses forecasts, which exist ahead of time, and is not exposed to
either problem.

## J. Deviations from spec-v2, and limits

1. **B_2021 was not run on the seen backtest.** Spec-v2 §7.1 dropped only the 2003-window
   Track B there; the code skipped both. It affects a seen window only — the Track C
   table has no backtest row. Fixed after this run (spec-v2 §7.2); it appears from the
   next rerun.
2. **Observed-weather problems were invisible.** A failed download or an unpublished
   day reaches Track B silently. After this run, B skips days not yet fully observed and
   the report lists observed coverage by city and month (spec-v2 §7.2). These changes
   leave every existing holdout, development and backtest prediction unchanged — checked by
   running the pipeline on synthetic data before and after — and, because the change was
   committed on 2026-10-20 before any 2026-10-21 data existed, they leave the freeze date
   and the prospective window where they were.
3. **Track C trains from April 2021**, not March as the spec expected; the start is set
   by the rule (first month with ≥ 95% coverage), not by hand.
4. **The run took 96 minutes**, above the 60–80 the spec estimated.
5. **The holdout has no full summer and no IESO forecasts.** A clean summer test needs the
   prospective window to reach summer 2027.
6. **The forecast-run mapping was verified on five dates in 2026 only** (§C).
7. **Temperature only.** No humidity, cloud cover or wind, and the same equal-weight three
   airports as v1.
8. **The quantile intervals are not calibrated** (§G).
9. **Post-hoc checks** (`scripts/diagnose_v2.py`) were written after seeing the results
   and have not yet run on real data: the first attempt needed the closed Colab session.

## K. Next

1. **Rerun the notebook every month or so.** That fills the prospective window — the
   clean test against IESO, including this coming winter — and runs the post-hoc checks
   on the two open questions in §C and §I. Reruns count only while the code is
   untouched; the report prints the freeze date it used.
2. **v3 candidates**, each to be pre-registered before it is built: calibrated intervals;
   the fresher legal GFS run for hours 7–23; humidity and cloud cover; load-weighted
   temperature; yesterday's forecast error as a feature.

---

# Part II — v1

Every number here comes from [`findings_auto.md`](findings_auto.md), the
machine-written output of `src/evaluate/run.py`, or from `data_quality.md`. This file
interprets them and does not restate any figure the machine did not produce.
Section references like *(auto §2)* point into `findings_auto.md`.

---

## 1. The question

Forecast Ontario's hourly **Market Demand** (MW) for every hour of tomorrow, the way
IESO's own day-ahead forecast does: made the day before at about 12:30, using only
what is known at that moment. Judge every model against a ladder of baselines, and
against IESO's published day-ahead forecast on the 127 days the harvester archived.

## 2. Answers in brief

1. **LightGBM Track A beat the spec's threshold decisively**: MAE 801 MW against
   seasonal naive's 1,275 MW (−37%, Diebold-Mariano p < 0.0001), over 731 days.
2. **But the threshold was the wrong one.** On this data the legal naive (yesterday
   or the day before) beats seasonal naive (last week), 995 against 1,275 MW. Against
   that stronger simple baseline LightGBM's margin is 19%, and against the best
   classical model (MSTL + ARIMA, 834 MW) only 4% — a gap that was not tested for
   significance.
3. **LightGBM's edge is uneven.** It comes from winter, the shoulder seasons, holidays
   and the night and evening hours. In summer, at the afternoon peak and on the hottest
   days — the hours that matter most for a grid — the classical models match or beat
   it, in both evaluation windows (§5).
4. **The leak in the original spec would have been worth 168 MW.** Its y(t-24)
   "naive" scores 827 MW against the legal naive's 995 MW. That alone would have put
   the dumbest baseline level with the classical models.
5. **Weather is the biggest missing piece.** Perfect knowledge of tomorrow's
   temperature (Track B, an upper bound) would cut the error by 181 MW overall, 370 MW
   in summer and 429 MW on the hottest days.
6. **IESO's own forecast is far better than anything here**: MAE 470 MW against 868 MW
   for the best Track A model on the same 127 days (p < 0.0001). Even Track B, with
   perfect observed temperature, reaches only 613 MW.

## 3. The clock, and what it changed

The forecast origin is D-1 at about 12:30. The archived IESO files say so directly:
`CreatedAt` is 12:31 on 125 of 127 days, and all 127 were created the day before
delivery. The data cutoff takes the stricter reading of that unstated time zone:
hours of D-1 beginning before **11:00 EST** are known, later ones are not.

The original spec defined the naive baseline as y(t-24) and listed a 24-hour lag as
a feature. For a target hour h on day D, t-24 is hour h on D-1, still in the future at
the origin whenever h ≥ 11: thirteen of the 24 hours of every forecast would have read
values not yet published. The legal naive uses D-1 for hours 0–10 and D-2 for 11–23,
the 24-hour lag is used only where legal, and `tests/test_features.py` proves no
feature for day D changes when every post-cutoff observation is overwritten.

**What it cost to be honest about this** *(auto §3)*: the leaky naive scores
827 MW, the legal one 995 MW. The 168 MW difference is pure look-ahead. The cutoff is
also visible directly in the error profile *(auto §2, by delivery hour)*: every model's
error steps up at hour-ending 12, the first hour where yesterday's value is no longer
known — the legal naive by 309 MW (965 → 1,274), LightGBM by 128 MW, MSTL + ARIMA by
68 MW. The better the model, the less it leans on that single lag.

This is the same family of error as the look-ahead problem in the MSc thesis that
motivated the project, and it was in the plan, not just in the code.

## 4. What the data looks like (Milestones 1–2)

**The clock is right.** The raw files are fixed EST, confirmed three independent ways
including a falsification test that could have failed and did not: across the 48 DST
transitions the morning ramp moved 0.28 h (spring) and 0.16 h (fall) against controls
of 0.20 h and 0.27 h, where a one-hour error would have pushed the first figure
towards 1.0 *(auto EDA §1; data_quality.md §3)*.

**The 2003 blackout lasted six days in the data**, 14–19 August, with demand 10–34%
below expectation during the conservation period. Those days are flagged out of every
model and every lag *(auto EDA §0)*.

**Summer and winter are different shapes** *(auto EDA §2)*. Weekday winter demand
peaks at 18:00 local (22,071 MW) with a 17,123 MW overnight trough; summer peaks at
17:00 (21,496 MW) over a much deeper trough of 15,640 MW. Winter has a steep morning
rise to a plateau and an evening peak; summer climbs slowly to one broad afternoon
peak. Hour of day therefore has to interact with season.

**Weekends drop 6–8%** *(auto EDA §3)*: 8.0% in summer, 7.3% in the shoulder seasons,
5.8% in winter — commercial cooling switches off at weekends while residential winter
evening load does not.

**Ontario is mostly, not purely, summer-peaking** *(auto EDA §4)*: 18 of 25 annual
peaks in summer, 6 in winter, and one in September (2016).

**Holidays do not behave alike** *(auto EDA §7)*. New Year's Day (−13%), Christmas
(−11%) and Boxing Day (−9.5%) cut demand most; Thanksgiving, Good Friday, Canada Day
and Victoria Day sit near −7% to −8%; Family Day −4.6%; Labour Day only −2.9% and
swings from −16% to +25%, because early-September heat can outweigh the holiday.

**Yesterday predicts today better than last week does** *(auto EDA §6)*. The median
day-on-day change is smaller than the median week-on-week change in every season
(summer 761 against 1,096 MW). Weather persists over days, not weeks. That was
visible before any model ran, and it is why the legal naive beat seasonal naive (§5).

**Old data misstates the level** *(auto EDA §5)*. Setting the demand level from the
previous three years gives the smallest error (mean 344 MW); using all history is
worst (486 MW, biased +342 MW, because 2005–08 ran higher than recent years).
Demand is rising again: the annual mean went from 17,189 MW in 2017 to 18,896 MW in
2025, and the 2026-07-02 peak of 27,678 MW is the highest since 2011.

## 5. Results — the 2024–2025 backtest

731 delivery days, 17,542 hours, every model scored on the same hours *(auto §2)*.

| | MAE (MW) | MAPE | ME (MW) |
|---|---|---|---|
| Seasonal naive, y(t-168) — the spec's threshold | 1,275 | 6.8% | −30 |
| Naive, legal (D-1 or D-2) | 995 | 5.4% | −8 |
| MSTL + ETS | 874 | 4.7% | −4 |
| LightGBM A, 3-year window | 868 | 4.7% | −8 |
| MSTL + ARIMA | 834 | 4.5% | −4 |
| **LightGBM A, expanding window** | **801** | **4.3%** | −8 |
| *Track B (upper bound), 3-year* | *662* | *3.6%* | *+4* |
| *Track B (upper bound), expanding* | *620* | *3.4%* | *−14* |

**The spec's rule is met**: LightGBM Track A beat seasonal naive by 37%, and the
Diebold-Mariano test puts the gap far outside noise (p ≈ 6 × 10⁻²⁶). Every other model
also beat seasonal naive significantly.

**The rule was a low bar.** The spec named seasonal naive "the real threshold" on the
assumption that last week is the hardest simple benchmark. On Ontario data it is not:
the legal naive beats it by 279 MW a day (p ≈ 7 × 10⁻⁹), for the reason in §4. A v2
plan should set the threshold at the better of the two. Against the legal naive,
LightGBM's margin is 19%. Against MSTL + ARIMA it is 33 MW, 4%, and the run did not
test that gap; it could be real or noise. The 3-year LightGBM (868 MW) is worse than
MSTL + ARIMA. The gradient-boosted model is the best of the ladder overall, but not by
a margin the run can certify over the best classical model — and, as the breakdown
below shows, not everywhere.

**Where the models win and lose** *(auto §2 decompositions; tables in
`reports/results/`)*:

- **By hour.** Errors are lowest overnight (≈ 560–590 MW for LightGBM at hours-ending
  1–5), climb through the morning, jump at the cutoff (hour-ending 12), and peak in
  the afternoon (≈ 1,050 MW at 13:00–16:00 EST) before easing in the evening.
  **At hours-ending 12–16, MSTL + ARIMA beats LightGBM** by 15–42 MW; LightGBM wins
  every other hour.
- **By season.** Summer is hardest for everything, and there **LightGBM has no edge**:
  MSTL + ETS 1,009 MW, LightGBM 1,032, MSTL + ARIMA 1,033. LightGBM's lead is in winter
  (765 against 859 MW for MSTL + ARIMA) and the shoulder seasons (702 against 721).
- **By day type.** Holidays separate the models most clearly. LightGBM, which has
  holiday flags, scores 798 MW on statutory holidays; the MSTL models, which have no
  calendar knowledge beyond the daily and weekly cycle, score about 1,240 MW, worse
  than on any other day type.
- **On extreme-temperature days** (hottest and coldest 5% of days, 37–38 days each).
  On the coldest days LightGBM is clearly best (885 against 1,018 MW for MSTL + ARIMA).
  On the hottest days **MSTL + ETS is best** (1,074 MW, against 1,127 for LightGBM and
  1,166 for MSTL + ARIMA).

**The same pattern holds in the IESO window** *(auto §4)*, which no model was tuned on:
in summer LightGBM 958 MW against 947 for MSTL + ARIMA; at hours-ending 12–14 the
classical models lead by 49–106 MW; and on the 7 hottest days LightGBM scores 1,942 MW,
twice MSTL + ETS's 970. Seven days is a small sample, but it points the same way as
the 37-day backtest figure.

**Why, as a hypothesis — not tested.** LightGBM's forecast is last week's value plus a
learned change, and its strongest inputs compare against last week. Training teaches it
that large week-on-week jumps are rare, so when a heat wave builds it shrinks the jump
it should be forecasting. The MSTL models re-estimate from the latest eight weeks at
every origin and carry the most recent level straight forward, which is what a heat
wave needs. The afternoon loss fits too: from hour-ending 12 LightGBM loses its lag-24
input, and its error steps up by 128 MW at that point against MSTL + ARIMA's 68 MW
(§3). Testing this would mean looking at errors day by day through specific heat waves.

**3-year or expanding window?** Expanding wins for LightGBM in both tracks (801 vs 868;
620 vs 662). That reverses what the level analysis in §4 would suggest, where a
3-year window tracked the demand level best. The two are consistent: LightGBM predicts
the change from last week with level-free inputs, so a stale level in old data costs
it nothing, and more years of examples help. The classical models, which fit only the
last eight weeks, are unaffected either way.

**Bias.** Overall ME is −8 MW, essentially unbiased, which the design aimed for *(auto
§2; findings_auto residual section)*. By season LightGBM under-forecasts winter by 58
MW. By month the bias ranges from −284 MW (December 2025) to +459 MW (August 2025).
Both Junes were under-forecast (−206 and −223 MW), consistent with early heat waves
arriving before last week's anchor reflects them.

**What the residuals still hold** *(auto residual section)*. The daily mean error has
an autocorrelation of 0.33 at a lag of one day, far outside the ±0.07 noise band: a
bad day tends to be followed by another, as a weather spell runs over several days.
The −0.15 at lag 7 is mechanical — the model anchors on last week's value, so a spike
in one week reappears as an error in the opposite direction the next. Hourly errors
are symmetric around zero (median −2 MW), with 90% inside −1,744 to +1,674 MW.

**What drives LightGBM** *(auto, feature importance)*. In Track A the most recent
information dominates: the change between D-2 and last week at the same hour, and
how D-1's latest known hour compares with a week earlier, together carry about half
the gain. Calendar terms matter less than expected because the weekly anchor already
carries most of the daily and weekly shape. In Track B the temperature terms take
about 35% of the gain, led by the week-on-week change in daily mean temperature.

## 6. Results — weather, the upper bound

Track B uses observed temperature and is **not a forecast**; it measures what perfect
weather knowledge would be worth. Over the backtest: 181 MW (801 → 620), or 23%.
The value is concentrated where the errors are: 370 MW in summer (1,032 → 662),
429 MW on the hottest days (1,127 → 698), 140 MW on the coldest, 133 MW across winter.

That is the largest single improvement available to this project, and it is not
available from this data: it needs archived weather *forecasts*, which a real
day-ahead model would use. Realistic gains would be smaller than the upper bound,
because forecasts are wrong too.

## 7. Results — against IESO's own forecast

127 delivery days, 2026-06-14 to 2026-10-18, summer and autumn only *(auto §4)*.

| | MAE (MW) | MAPE | ME (MW) |
|---|---|---|---|
| **IESO day-ahead forecast** | **470** | **2.5%** | −177 |
| Track B, expanding (upper bound) | 613 | 3.2% | +32 |
| LightGBM A, expanding | 868 | 4.5% | +56 |
| MSTL + ARIMA | 894 | 4.6% | +42 |
| Seasonal naive | 1,318 | 6.8% | +139 |

**IESO wins by a wide margin**: 398 MW a day better than the best Track A model,
p < 0.0001. The ranking of our own models in this window matches the backtest, and
LightGBM's summer error here (958 MW) is close to its 2024–25 summer error (1,032 MW),
so the window looks representative of what these models do, not an unlucky stretch.

**Even perfect temperature does not close the gap.** Track B reaches 613 MW, still 30%
behind IESO. So IESO's advantage is not only weather forecasts. What else it uses is
not visible from here; plausible candidates are humidity and cloud cover, regional
weather rather than three airports, embedded solar, scheduled industrial and
dispatchable loads, and operator knowledge. None of that was tested.

**Where IESO's edge is** *(auto §4, by delivery hour)*. Through the day, hour-ending 6
to 23, IESO's error is 309–500 MW against LightGBM's 637–1,189 MW. Overnight the picture
reverses: at hours-ending 1–5 LightGBM Track A roughly matches IESO (608–678 against
541–656 MW), Track B (expanding) beats it at every one of those hours (494–533 MW), and IESO's single worst hour is
hour-ending 24 (840 MW). A one-hour misalignment between the files would put the
largest errors on the morning and evening ramps, where IESO's are smallest, so this
is not a join error.

A hypothesis for the overnight weakness, **not tested**: Market Demand includes
scheduled exports (on average 2,575 MW of the 2026 level, `data_quality_auto.md`
§12), and Ontario exports heavily overnight. Night-time error would then reflect how
far real-time export schedules drift from day-ahead ones — a trading question rather
than a consumption one. Rerunning the pipeline with Ontario Demand as the target would
test this.

**Bias.** IESO under-forecast by 177 MW on average over these 127 days, and by 312 MW
in the autumn weeks. The single-day check in spec §1 (2026-06-14) had shown +61 MW;
one day was not representative.

**The small print.** 127 days, one summer, no winter. Holidays (3 days in the window)
and extreme-temperature days (7 each) are too few to stand on their own. LightGBM's poor
holiday score here (1,512 MW) is not treated as a finding, because it contradicts the
731-day backtest; its poor hottest-day score is mentioned in §5 only because it agrees
with the backtest.
The comparison should be rerun as the archive grows; the harvester keeps adding a day
at a time.

## 8. Decisions made while building, and why

**LightGBM predicts the change from last week, with level-free inputs.** A tree cannot
forecast outside the range of targets it trained on, and demand is setting new highs.
Predicting the level would cap forecasts exactly when they matter. Predicting the
change was not enough on its own: on synthetic data with a built-in upward trend, a
model fed raw lagged levels was biased by about −90 MW in every month, having learned
that a high level means a fall is coming. Level-free inputs brought that to within
±5 MW. Both choices were fixed on that argument before any real result existed. The
real results bear them out: overall bias −8 MW, and the expanding window wins (§5).

**The classical rungs are MSTL with ETS or ARIMA on the remainder**, refitted at every
origin on the trailing eight weeks, because a seasonal ARIMA cannot hold the daily and
weekly cycles together.

**LightGBM refits monthly** on days up to two days before the month starts, with
hyperparameters fixed in `src/models/lgbm.py` and never tuned on an evaluation window.

**Two holiday flags**: Ontario's statutory holidays, and a second flag for Civic
Holiday, Easter Monday and Remembrance Day. The second flag earned almost no gain in
any model *(auto, feature importance)*; its effect was not measured separately.

## 9. Limitations

1. **No weather forecasts.** Track A has no weather; Track B has perfect weather. The
   realistic middle is not built. This is the biggest gap to IESO (§6–7).
2. **Station representativeness.** Three airports, equal weight. Toronto carries far
   more load than London; a load-weighted average would be better.
3. **DST.** The series is fixed EST. Local days have 23 and 25 hours at the
   transitions, the November fall-back hour appears twice in local time, and seasonal
   naive matches the same EST hour, which is one local hour off in the week after a
   transition.
4. **The target includes exports**, which may behave more like trading decisions than
   demand (§7, untested).
5. **Untested gaps.** LightGBM against MSTL + ARIMA, and either against the legal
   naive, were not put through the significance test.
6. **The official window is short and has no winter** (§7).
7. **The data cutoff is an assumption**: actual demand published within the hour, and
   the stricter reading of IESO's unstated timestamp zone.
8. **The blackout window is approximate**; without weather the method cannot tell
   conservation from a cool spell.
9. **One missing hour** (2025-05-01 00:00 EST) is left missing; hours that depend on it
   drop out of the common sample.
10. **Market Demand's definition** has not been checked for changes over 24 years.

## 10. What v2 would do

In rough order of expected payoff:

1. **Archived weather forecasts** for an honest weather-aware track — the 181 MW upper
   bound says this is where most of the remaining error is.
2. **Rerun with Ontario Demand as the target**, to separate consumption from exports
   and test the overnight hypothesis in §7.
3. **Set the threshold at the better of naive and seasonal naive**, and add
   significance tests between adjacent rungs, not only against the threshold.
4. **Combine LightGBM with MSTL + ETS.** Their strengths are complementary (§5):
   LightGBM in winter, on holidays and at night; MSTL + ETS in summer and on the hottest
   days. A simple average, or a weight chosen on data before the test window, is the
   obvious first try.
5. **Use recent forecast error as a feature** — the 0.33 day-to-day error
   autocorrelation says yesterday's miss carries information.
6. **Load-weighted temperature**, and probabilistic forecasts (quantile LightGBM).
7. **Repeat the IESO comparison** once the archive covers a full year, including a
   winter.
