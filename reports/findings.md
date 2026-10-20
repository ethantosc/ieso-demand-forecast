# Findings — Ontario hourly demand forecasting, v1

**Status:** final for v1 · results from the run of 2026-10-20 05:02 UTC, code at
commit `5011a08`

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
3. **The leak in the original spec would have been worth 168 MW.** Its y(t-24)
   "naive" scores 827 MW against the legal naive's 995 MW. That alone would have put
   the dumbest baseline level with the classical models.
4. **Weather is the biggest missing piece.** Perfect knowledge of tomorrow's
   temperature (Track B, an upper bound) would cut the error by 181 MW overall, 370 MW
   in summer and 429 MW on the hottest days.
5. **IESO's own forecast is far better than anything here**: MAE 470 MW against 868 MW
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
MSTL + ARIMA. The honest summary is that the gradient-boosted model is the best of the
ladder, but not by a margin the run can certify over the best classical model.

**Where the models win and lose** *(auto §2 decompositions)*:

- **By hour.** Errors are lowest overnight (≈ 560–590 MW for LightGBM at hours-ending
  1–5), climb through the morning, jump at the cutoff (hour-ending 12), and peak in
  the afternoon (≈ 1,050 MW at 13:00–16:00 EST) before easing in the evening.
- **By season.** Summer is hardest for everything (LightGBM 1,032 MW against 765 MW
  in winter and 702 MW in the shoulder seasons). Air-conditioning load swings with
  temperature, and Track A cannot see temperature.
- **By day type.** Holidays separate the models most clearly. LightGBM, which has
  holiday flags, scores 798 MW on statutory holidays; the MSTL models, which have no
  calendar knowledge beyond the daily and weekly cycle, score about 1,240 MW, worse
  than on any other day type.
- **On extreme-temperature days** (hottest and coldest 5% of days). LightGBM degrades
  to 1,127 MW on the hottest days and 885 MW on the coldest, against 778 MW on normal
  days.

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
and extreme-temperature days (about 6 each) are too few to read much into, which is
why LightGBM's poor holiday score here (1,512 MW) is not treated as a finding.
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
4. **Use recent forecast error as a feature** — the 0.33 day-to-day error
   autocorrelation says yesterday's miss carries information.
5. **Load-weighted temperature**, and probabilistic forecasts (quantile LightGBM).
6. **Repeat the IESO comparison** once the archive covers a full year, including a
   winter.
