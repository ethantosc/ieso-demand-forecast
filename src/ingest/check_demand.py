#!/usr/bin/env python3
"""Test the assumptions behind demand_hourly.parquet, then write the numbers down.

Run from the repo root, after ``python -m src.ingest.demand``:

    python -m src.ingest.check_demand

This does two jobs and they are deliberately separate from ingestion:

1.  It *tests* the claims the ingest makes - that the CSV timeline is fixed
    EST, that every day is complete, that the series has no gaps or
    duplicates, that year-file boundaries join cleanly. Anything that fails
    here invalidates the Parquet, not just the report.

2.  It writes ``reports/data_quality_auto.md`` holding only machine-produced
    numbers. The prose version, ``reports/data_quality.md``, is written by
    hand around those numbers. Keeping them in separate files means no figure
    in the narrative can drift away from what the data actually said.

Queries are DuckDB SQL straight over the Parquet - no load step, no database
file. That is the point of the tool.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import duckdb

PARQUET = Path("data/staging/demand_hourly.parquet")
REPORT = Path("reports/data_quality_auto.md")

# Anything outside this band is implausible for Ontario and worth eyeballing.
# Not a cleaning rule - nothing is dropped, this only flags.
PLAUSIBLE_MW = (5_000, 30_000)


class Checks:
    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self.con = connection
        self.sections: list[str] = []
        self.failures: list[str] = []

    def sql(self, query: str):
        return self.con.sql(query)

    def note(self, heading: str, body: str) -> None:
        self.sections.append(f"### {heading}\n\n{body}\n")

    def table(self, heading: str, query: str, empty_is_pass: bool = False) -> int:
        """Run a query, record it as a markdown table, return the row count."""
        result = self.con.sql(query)
        columns = result.columns
        rows = result.fetchall()
        if not rows:
            body = "_no rows_"
        else:
            header = "| " + " | ".join(columns) + " |"
            rule = "|" + "|".join(["---"] * len(columns)) + "|"
            lines = [
                "| " + " | ".join("" if v is None else str(v) for v in row) + " |"
                for row in rows[:60]
            ]
            body = "\n".join([header, rule, *lines])
            if len(rows) > 60:
                body += f"\n\n_({len(rows):,} rows total, first 60 shown)_"
        self.note(heading, body)
        if empty_is_pass and rows:
            self.failures.append(f"{heading}: {len(rows):,} offending row(s)")
        return len(rows)


def main() -> int:
    if not PARQUET.exists():
        print(f"missing {PARQUET} - run the ingest first", file=sys.stderr)
        return 1

    con = duckdb.connect()
    con.sql("SET TimeZone='UTC'")  # determinism; all stored timestamps are naive anyway
    con.sql(f"CREATE VIEW demand AS SELECT * FROM read_parquet('{PARQUET.as_posix()}')")
    checks = Checks(con)

    # ---- 1. shape -------------------------------------------------------
    checks.table(
        "1. Coverage by source file",
        """
        SELECT source_file,
               count(*)                       AS rows,
               min(delivery_date)             AS first_date,
               max(delivery_date)             AS last_date,
               count(DISTINCT delivery_date)  AS days,
               round(count(*) / 24.0, 2)      AS days_implied
        FROM demand GROUP BY source_file ORDER BY source_file
        """,
    )

    # ---- 2. the EST assumption -----------------------------------------
    # If the CSV were wall-clock time, some delivery_date would have 23 or 25
    # rows. Fixed EST means every delivery_date has exactly 24, always.
    checks.table(
        "2. FAIL EXPECTED EMPTY - delivery_date without exactly 24 hours",
        """
        SELECT delivery_date, count(*) AS rows,
               list_sort(list(hour_ending)) AS hours
        FROM demand GROUP BY delivery_date HAVING count(*) <> 24
        ORDER BY delivery_date
        """,
        empty_is_pass=True,
    )
    checks.table(
        "3. FAIL EXPECTED EMPTY - hour_ending outside 1..24",
        "SELECT DISTINCT hour_ending FROM demand WHERE hour_ending NOT BETWEEN 1 AND 24",
        empty_is_pass=True,
    )
    checks.table(
        "4. FAIL EXPECTED EMPTY - duplicate (delivery_date, hour_ending)",
        """
        SELECT delivery_date, hour_ending, count(*) AS n
        FROM demand GROUP BY 1, 2 HAVING count(*) > 1 ORDER BY 1, 2
        """,
        empty_is_pass=True,
    )

    # ---- 3. continuity in real time -------------------------------------
    checks.table(
        "5. FAIL EXPECTED EMPTY - gaps or jumps in the UTC hourly series",
        """
        WITH stepped AS (
            SELECT ts_utc, lag(ts_utc) OVER (ORDER BY ts_utc) AS prev_ts FROM demand
        )
        SELECT prev_ts, ts_utc, date_diff('hour', prev_ts, ts_utc) AS step_hours
        FROM stepped
        WHERE prev_ts IS NOT NULL AND date_diff('hour', prev_ts, ts_utc) <> 1
        ORDER BY ts_utc
        """,
        empty_is_pass=True,
    )

    # ---- 4. year-file seams ---------------------------------------------
    # Two annual files meeting should look like any other hour. A level shift
    # here would mean the files disagree about units or definition.
    checks.table(
        "6. Year-file seams (Dec 31 23:00 EST -> Jan 1 00:00 EST)",
        """
        WITH edges AS (
            SELECT ts_utc, source_file, market_demand_mw,
                   lag(source_file)      OVER (ORDER BY ts_utc) AS prev_file,
                   lag(market_demand_mw) OVER (ORDER BY ts_utc) AS prev_mw
            FROM demand
        )
        SELECT prev_file, source_file, ts_utc,
               prev_mw AS mw_before, market_demand_mw AS mw_after,
               round(market_demand_mw - prev_mw, 1) AS delta_mw
        FROM edges
        WHERE prev_file IS NOT NULL AND prev_file <> source_file
        ORDER BY ts_utc
        """,
    )

    # ---- 5. DST, seen from the local side --------------------------------
    # Now the arithmetic flips. In local wall-clock terms a fixed-EST series
    # MUST show one 23-hour day and one 25-hour day per year. Seeing exactly
    # two such dates per year, on the right Sundays, is the positive proof
    # that check 2 was not just a well-padded file.
    checks.table(
        "7. Local dates without 24 hours - expect exactly 2 per year (DST)",
        """
        SELECT local_date, count(*) AS rows,
               min(utc_offset_hours) AS off_min, max(utc_offset_hours) AS off_max
        FROM demand GROUP BY local_date HAVING count(*) <> 24
        ORDER BY local_date
        """,
    )
    checks.table(
        "8. Repeated local wall-clock hours - the fall-back hour, 1 per year",
        """
        SELECT ts_local, count(*) AS n, list(ts_utc) AS utc_instants
        FROM demand GROUP BY ts_local HAVING count(*) > 1 ORDER BY ts_local
        """,
    )
    checks.note(
        "9. Reminder",
        "`ts_local` is **not unique** - the November fall-back hour appears twice "
        "every year. Never key, join, or sort on it. `ts_utc` is the only safe key.",
    )

    # ---- 6. nulls and plausibility ---------------------------------------
    checks.table(
        "10. FAIL EXPECTED EMPTY - null demand values",
        """
        SELECT ts_utc, source_file, market_demand_mw, ontario_demand_mw
        FROM demand
        WHERE market_demand_mw IS NULL OR ontario_demand_mw IS NULL
        ORDER BY ts_utc
        """,
        empty_is_pass=True,
    )
    checks.table(
        f"11. Values outside {PLAUSIBLE_MW[0]:,}-{PLAUSIBLE_MW[1]:,} MW (flag only, nothing dropped)",
        f"""
        SELECT ts_utc, ts_local, source_file, market_demand_mw, ontario_demand_mw
        FROM demand
        WHERE market_demand_mw NOT BETWEEN {PLAUSIBLE_MW[0]} AND {PLAUSIBLE_MW[1]}
           OR ontario_demand_mw NOT BETWEEN {PLAUSIBLE_MW[0]} AND {PLAUSIBLE_MW[1]}
        ORDER BY ts_utc
        """,
    )

    # ---- 7. the two demand series ----------------------------------------
    # Market Demand = Ontario Demand + scheduled exports, so the gap should be
    # positive and sizeable. This is the same relationship that decided the
    # target variable; it should hold in the actuals too, not just in DATotals.
    checks.table(
        "12. Market minus Ontario demand, by year",
        """
        SELECT year(local_date) AS yr,
               round(avg(market_demand_mw), 0)                          AS avg_market,
               round(avg(ontario_demand_mw), 0)                         AS avg_ontario,
               round(avg(market_demand_mw - ontario_demand_mw), 0)      AS avg_gap,
               round(min(market_demand_mw - ontario_demand_mw), 0)      AS min_gap,
               round(max(market_demand_mw - ontario_demand_mw), 0)      AS max_gap
        FROM demand GROUP BY 1 ORDER BY 1
        """,
    )
    checks.table(
        "13. Hours where Ontario demand exceeds Market demand (net import hours)",
        """
        SELECT year(local_date) AS yr, count(*) AS hours,
               round(min(market_demand_mw - ontario_demand_mw), 0) AS most_negative_gap
        FROM demand WHERE ontario_demand_mw > market_demand_mw
        GROUP BY 1 ORDER BY 1
        """,
    )

    # ---- 8. annual profile, for the EDA to come --------------------------
    checks.table(
        "14. Annual summary (Market Demand, MW)",
        """
        SELECT year(local_date) AS yr, count(*) AS hours,
               round(avg(market_demand_mw), 0) AS mean,
               round(min(market_demand_mw), 0) AS min,
               round(max(market_demand_mw), 0) AS max,
               arg_max(ts_local, market_demand_mw) AS peak_hour_local
        FROM demand GROUP BY 1 ORDER BY 1
        """,
    )

    # ---- write it out -----------------------------------------------------
    total = con.sql("SELECT count(*) FROM demand").fetchone()[0]
    header = (
        "# Data quality - machine output\n\n"
        f"Generated {datetime.now().isoformat(timespec='seconds')} by "
        "`src/ingest/check_demand.py`.\n\n"
        "**Do not edit this file by hand.** Prose belongs in `data_quality.md`, "
        "which should cite these numbers rather than restate them from memory.\n\n"
        f"Source: `{PARQUET.as_posix()}` - {total:,} rows.\n\n---\n"
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(header + "\n".join(checks.sections), encoding="utf-8")

    print(f"wrote {REPORT}  ({total:,} rows checked)")
    if checks.failures:
        print("\nFAILED:")
        for failure in checks.failures:
            print(f"  - {failure}")
        print("\nThe Parquet is not trustworthy yet. Do not build features on it.")
        return 1

    print("\nAll must-be-empty checks passed.")
    print("Still needs your eyes: sections 6, 7, 8, 11, 12, 13.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
