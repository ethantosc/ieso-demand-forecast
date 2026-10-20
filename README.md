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
| Milestone 0 - forecast harvester | running daily since 2026-09-13; 127 consecutive days archived |
| Milestone 1 - data | done - [`reports/data_quality.md`](reports/data_quality.md) |
| Milestone 2 - EDA | done - [`reports/findings.md`](reports/findings.md) §3 |
| Milestones 3-7 - backtest, ladder, LightGBM A and B, official comparison | built and tested; **awaiting the first real run** |
| Milestone 8 - write-up | [`reports/findings.md`](reports/findings.md); results section fills in after the run |

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
- **Scoring.** MAE, RMSE, MAPE, ME (forecast minus actual) on a common sample,
  split by delivery hour, clock hour, season, day type and extreme-temperature days,
  with a Diebold-Mariano test for whether each gap is real.

## Results

Pending the first real run. The machine output will be
[`reports/findings_auto.md`](reports/findings_auto.md); the interpretation goes in
[`reports/findings.md`](reports/findings.md) §6.

## Limitations

The full list is in `findings.md` §7. The three that matter most: no archived
weather forecasts (so Track A has no weather and Track B has perfect weather -
nothing realistic in between); three airport stations averaged with equal weight;
and an official comparison window of about four months, one summer and no winter.

## Reproduce

In Google Colab, no local install needed. Open each notebook from GitHub
(**File → Open notebook → GitHub**) and use **Runtime → Run all**:

| Notebook | Does | Time |
|---|---|---|
| `notebooks/00_ingest_colab.ipynb` | download IESO demand, build and validate the Parquet | ~2 min |
| `notebooks/01_eda.ipynb` | Milestone 2 analysis | ~3 min |
| `notebooks/02_results.ipynb` | tests, full backtest, official comparison, report | ~20-30 min |

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
src/ingest/                     downloads and parsers: demand, weather, IESO forecasts
src/transform/features.py       origin-aware features - the leakage boundary
src/models/                     baselines.py (rungs 1-4), lgbm.py (rung 5, tracks A/B)
src/evaluate/                   run.py (orchestrator), metrics.py, plots.py
src/analysis/eda.py             Milestone 2 computations
tests/                          leakage guard, parsers
notebooks/                      Colab drivers - no logic lives here
reports/                        data_quality, findings, and machine-written outputs
docs/spec.md                    the plan, v1.1
data/raw/forecasts/             the harvested IESO forecast archive (tracked)
```
