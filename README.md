# ieso-demand-forecast

Day-ahead forecasts of Ontario's hourly electricity demand from IESO public data,
judged against a ladder of baselines and against IESO's own published day-ahead
forecast.

A skills project, not a research contribution: the point is doing baseline
comparison, time-series backtesting and residual diagnostics properly, on a
deliberately well-trodden problem, with a Python stack (Polars, DuckDB,
statsforecast, LightGBM). The plan is in [`docs/spec.md`](docs/spec.md).

## Status

| | |
|---|---|
| Milestone 0 - forecast harvester | running daily since 2026-09-13; the archive covers delivery days from 2026-06-14 |
| Milestone 1 - data | done - [`reports/data_quality.md`](reports/data_quality.md) |
| Milestone 2 - EDA | done - [`reports/findings.md`](reports/findings.md) §3 |
| Milestones 3-7 - backtest, ladder, LightGBM A and B, official comparison | done - run of 2026-10-20 |
| Milestone 8 - write-up | done - [`reports/findings.md`](reports/findings.md) |
| **v2** - weather forecasts, ensembles, exports, probabilistic | pre-registered in [`docs/spec-v2.md`](docs/spec-v2.md); results of 2026-10-22 UTC - [`reports/findings.md`](reports/findings.md) Part I |
| v2 prospective window | delivery days after the 2026-10-20 freeze; grows with each rerun of notebook 03 |

## The problem

Forecast **Market Demand** (MW) for each hour of tomorrow, made the day before at
about 12:30 - the time IESO's own day-ahead forecast is produced - using only what
is known then. Market Demand, not Ontario Demand, because IESO's forecast is built
on it (spec §1).

## Data

| Source | What | Where |
|---|---|---|
| IESO `PUB_Demand_{year}.csv` | hourly Market and Ontario demand, 2002-05-01 onward, fixed EST | downloaded, not in git |
| IESO `DATotals` | IESO's day-ahead forecast, kept by IESO for only a few months | harvested daily into `data/raw/forecasts/`, **in git** |
| Environment Canada hourly climate | temperature, Toronto, Ottawa, London airports | downloaded, Track B only |
| Open-Meteo Previous Runs API (v2) | archived NCEP GFS temperature forecasts, same three airports, from 2021 | downloaded, Track C only |

Quirks that matter, all documented in `data_quality.md`: the files are fixed EST,
not wall-clock time (tested three independent ways); one hour is missing
(2025-05-01 00:00); the 2003 blackout suppressed demand for six days, which are
flagged out of every model.

## Method

- **The clock.** Hours of the day before beginning before 11:00 EST are known at the
  origin; nothing later is. That rules out the spec's original 24-hour lag for 13 of
  the 24 hours, and `tests/test_features.py` proves no feature crosses the line.
- **Backtest.** Rolling origin over 2024-2025, one forecast per day from its own origin.
- **Ladder** (spec §5): legal naive → **seasonal naive (the threshold)** → MSTL + ETS
  → MSTL + ARIMA → LightGBM.
- **Two LightGBM tracks.** A: calendar and lags only - the real forecast. B: adds
  observed temperature - an **upper bound, not a forecast**, since no one knows
  tomorrow's observed weather at 12:30.
- **v2 adds** Track C (LightGBM with archived GFS temperature forecasts, legal at the
  origin), ensembles with MSTL + ETS, Ontario Demand and exports as separate targets,
  and quantile forecasts - all judged on windows fixed before the code was written.
- **Scoring.** MAE, RMSE, MAPE, ME (forecast minus actual) on a common sample,
  split by delivery hour, clock hour, season, day type and extreme-temperature days,
  with a Diebold-Mariano test for whether each gap is real.

## Results

### v2 - clean holdout, 2026-01-01 to 2026-06-13

164 days that v1 never scored. MAE in MW, every model on the same hours.

| | Holdout |
|---|---|
| Naive, legal (D-1 or D-2) - the threshold, chosen on 2023 | 1,030 |
| MSTL + ARIMA | 824 |
| LightGBM A - calendar and lags only | 864 |
| **LightGBM C - adds archived GFS temperature forecasts** | **659** |
| **Ensemble C** - 0.8 x LightGBM C + 0.2 x MSTL + ETS | **654** |
| *LightGBM B - observed temperature, same training years as C; upper bound only* | *654* |

- Forecast weather is worth 197 MW against the same model without it (Holm-corrected
  Diebold-Mariano p < 0.0001), using only GFS runs published before the 12:30 origin -
  a rule checked against individual model runs before use. That is 98% of what perfect
  temperature is worth on the holdout, 80% over 2024-25; the shortfall is in summer.
- Without weather, LightGBM does not beat MSTL + ARIMA significantly in either window;
  with forecast weather, v1's summer weakness disappears.
- Exports do not explain the night-time error; Ontario Demand's own error is larger at
  every hour.
- Quantile LightGBM is sharp but overconfident: its 90% interval holds 67% of outcomes.
- IESO is still ahead on the seen window (468 against 618 MW) - and ahead of the
  perfect-weather upper bound too; the clean comparison is the prospective window.

Interpretation: [`reports/findings.md`](reports/findings.md) Part I. Every number:
[`reports/v2/findings_auto.md`](reports/v2/findings_auto.md).

### v1 - backtest and IESO window

Backtest 2024-2025 (731 days) and IESO comparison (127 days, 2026-06-14 to
2026-10-18). MAE in MW; every model scored on the same hours.

| | Backtest | vs IESO window |
|---|---|---|
| Seasonal naive, y(t-168) - the spec's threshold | 1,275 | 1,318 |
| Naive, legal (D-1 or D-2) | 995 | 1,056 |
| MSTL + ARIMA | 834 | 894 |
| **LightGBM Track A** (expanding window) | **801** | **868** |
| *Track B - observed temperature, upper bound only* | *620* | *613* |
| IESO's own day-ahead forecast | - | **470** |

- LightGBM Track A beats the spec's threshold by 37% (Diebold-Mariano p < 0.0001),
  but the legal naive turns out to be the stronger simple baseline, and the margin
  over the best classical model is only 4% and untested.
- That edge is uneven: in summer, at the afternoon peak and on the hottest days, the
  classical models match or beat LightGBM, in both windows.
- The spec's original y(t-24) naive read unpublished data; that leak was worth 168 MW.
- Perfect weather knowledge would be worth 181 MW overall and 370 MW in summer.
- IESO's forecast is far better (470 vs 868 MW) - and better than the perfect-weather
  upper bound, so its edge is more than weather forecasts.

Interpretation: [`reports/findings.md`](reports/findings.md) Part II. Every number:
[`reports/findings_auto.md`](reports/findings_auto.md).

## Limitations

The full lists are in `findings.md` (Part I §J for v2, Part II §9 for v1). The
three that matter most now: there is no clean comparison with IESO yet (the holdout
predates the forecast archive, and the prospective window has only just started);
weather is temperature only, at three airports averaged with equal weight; and the
holdout has no full summer.

## Reproduce

In Google Colab, no local install needed. Open each notebook from GitHub
(**File → Open notebook → GitHub**) and use **Runtime → Run all**:

| Notebook | Does | Time |
|---|---|---|
| `notebooks/00_ingest_colab.ipynb` | download IESO demand, build and validate the Parquet | ~2 min |
| `notebooks/01_eda.ipynb` | Milestone 2 analysis | ~3 min |
| `notebooks/02_results.ipynb` | tests, full backtest, official comparison, report | ~35 min |
| `notebooks/03_v2.ipynb` | v2: forecast-weather verification, holdout, ensembles, exports, quantiles | ~100 min |

Locally, with Python 3.12+ and `pip install -r requirements.txt`:

```bash
python -m src.ingest.fetch_demand && python -m src.ingest.demand   # data
python -m pytest tests -q                                          # leakage guard and parsers
python -m src.evaluate.run                                         # everything else
```

## Layout

```
.github/workflows/harvest.yml   daily forecast harvester (GitHub Actions)
scripts/harvest_forecasts.py    the harvester itself
scripts/diagnose_v2.py          post-hoc v2 checks, outside src/ so the freeze date stays put
src/ingest/                     downloads and parsers: demand, weather, weather forecasts, IESO forecasts
src/transform/features.py       origin-aware features - the leakage boundary
src/models/                     baselines.py, lgbm.py (tracks A/B/C, quantiles), ensemble.py, probabilistic.py
src/evaluate/                   run.py (v1), run_v2.py (v2), metrics.py, plots.py
src/analysis/eda.py             Milestone 2 computations
tests/                          leakage guards (features, quantile baseline, forecast rule), parsers
notebooks/                      Colab drivers - no logic lives here
reports/                        data_quality, findings, and machine-written outputs
docs/spec.md                    the plan, v1.1
docs/spec-v2.md                 the v2 plan, written before any v2 model code
data/raw/forecasts/             the harvested IESO forecast archive (tracked)
```

## Data credits

Demand and day-ahead forecasts: IESO public reports. Observed weather: Environment and
Climate Change Canada. Weather forecasts (v2): [Open-Meteo](https://open-meteo.com/),
licensed CC BY 4.0, NCEP GFS model data.
