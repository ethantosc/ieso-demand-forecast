# Data quality — Ontario hourly demand

**Status:** skeleton · fill in after running `src/ingest/check_demand.py`

Machine-produced figures live in `data_quality_auto.md`. This file is the prose
version: what the numbers mean, what was decided, what is still unknown. Every
figure quoted here should be traceable to a section number in the auto file.

---

## 1. What is in the dataset

| | |
|---|---|
| Source | IESO public annual demand CSVs, `PUB_Demand_{year}.csv` |
| Coverage | _(auto §1)_ |
| Rows | _(auto §1)_ |
| Grain | one row per hour |
| Target variable | Market Demand (MW) — see spec §1 for why not Ontario Demand |

## 2. Time convention

The raw CSV gives `Date` plus `Hour` 1–24, hour-**ending**. Hour 1 covers
00:00–01:00.

The staging table relabels to hour-beginning and carries three time columns:

| Column | Meaning | Use it for |
|---|---|---|
| `ts_utc` | the actual instant, naive UTC | keys, joins, lags, sorting |
| `ts_local` | Toronto wall clock, naive | reading, plotting |
| `local_hour` | 0–23 wall clock | calendar features |

`ts_local` is **not unique** — the November fall-back hour appears twice each
year _(auto §8)_. Never key on it.

## 3. The fixed-EST assumption and how it was tested

**Claim:** the CSV timeline is fixed EST (UTC−5 year round), not local wall
clock.

**Why it matters:** it decides whether the series is continuous. If the files
were wall clock, one day a year would be missing an hour and another would have
a duplicate, and every 24h/168h lag crossing those days would be silently wrong.

**Test and result:**

| Test | Expectation if fixed EST | Result |
|---|---|---|
| Rows per `delivery_date` | always exactly 24 | _(auto §2)_ |
| Gaps in the UTC hourly series | none | _(auto §5)_ |
| Local dates with ≠24 hours | exactly 2 per year, on the DST Sundays | _(auto §7)_ |
| Repeated wall-clock hours | exactly 1 per year, November | _(auto §8)_ |

The last two are the positive proof: a padded file would pass the first test but
fail these.

**Status:** _(confirmed / not confirmed — state which, with the numbers)_

## 4. Series start date

_(To confirm: does Market Demand exist before market opening on 2002-05-01? If
the 1994–2002 file carries Ontario Demand only, then the v1 target variable
simply does not exist before that date, the pre-2002 file is out of scope, and
the "2002 seam" question in the spec dissolves rather than being answered.
Record the file's actual header here either way.)_

## 5. Missing and implausible values

- Nulls: _(auto §10)_
- Outside 5,000–30,000 MW: _(auto §11)_
- **Nothing has been imputed or dropped.** If that changes, the rule goes here
  and the reason goes with it.

## 6. Year-file seams

Annual files are concatenated. A definitional or unit disagreement between
files would show as a level shift at the 31 Dec → 1 Jan boundary.

_(auto §6 — record the deltas and whether any look anomalous against a typical
hour-to-hour change.)_

## 7. Market vs Ontario demand

Market Demand = Ontario Demand + scheduled exports, so the gap should be
positive and material.

- Average gap by year: _(auto §12)_
- Hours where Ontario exceeded Market: _(auto §13)_ — _(explain these if any:
  net import hours, or a data issue?)_

## 8. Known limitations

1. _(DST handling — state the convention and what a reader should watch for)_
2. _(Weather forecast gap — see spec §4, track A vs track B)_
3. _(Station representativeness — pending ECCC decision)_
4. _(Any definitional change in Market Demand over 20+ years that has not been
   checked)_

## 9. Open questions

- [ ] Pre-2002 file: does it carry Market Demand at all?
- [ ] Has the Market Demand definition changed between 2002 and today?
- [ ] ECCC station selection
