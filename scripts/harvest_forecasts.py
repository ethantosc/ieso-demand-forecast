#!/usr/bin/env python3
"""Harvest IESO forecast reports before they expire.

IESO keeps its forecast reports for only 30-90 days. Once a report drops out
of that window it is gone permanently. This script runs daily and saves any
file that is still inside the retention window but not yet in our archive.

The design is reconciliation-based, not "fetch the latest". Every run compares
the remote directory listing against what we already hold and downloads only
what is missing. That means a few days of failed runs cost us nothing: the
next successful run backfills everything still inside the window.

Standard library only - no pip install required.
"""

from __future__ import annotations

import os
import sys
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

BASE = "https://reports-public.ieso.ca/public"
OUT_ROOT = Path("data/raw/forecasts")

# Reports to harvest. The name is the URL path segment.
# PredispTotals is commented out until we know how fast it grows - it may
# publish several versions per day. See the repo size note in the README.
REPORTS = [
    "DATotals",   # Day-ahead totals: expected hourly load. One file per day.
    "Adequacy3",  # Adequacy: hourly requirements for today + next 34 days.
    # "PredispTotals",
]

TIMEOUT = 60
RETRIES = 3
BACKOFF = 5  # seconds, doubled on each retry
USER_AGENT = "ieso-demand-forecast-harvester/1.0 (personal research project)"


class LinkParser(HTMLParser):
    """Collect every href from a directory index page."""

    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        for name, value in attrs:
            if name == "href" and value:
                self.hrefs.append(value)


def fetch(url: str) -> bytes:
    """Fetch a URL, retrying with exponential backoff on failure."""
    last_error: Exception | None = None
    for attempt in range(RETRIES):
        if attempt:
            time.sleep(BACKOFF * (2 ** (attempt - 1)))
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as error:
            last_error = error
            print(f"    attempt {attempt + 1}/{RETRIES} failed: {error}", file=sys.stderr)
    raise RuntimeError(f"could not fetch {url} after {RETRIES} attempts") from last_error


def list_remote_files(report: str) -> list[str]:
    """List every dated file in a report directory.

    Skips the undated pointer file (e.g. PUB_DATotals.xml). That file always
    holds the most recent version, so its contents change daily. Archiving it
    would leave us with snapshots we cannot tell apart.
    """
    index_html = fetch(f"{BASE}/{report}/").decode("utf-8", errors="replace")
    parser = LinkParser()
    parser.feed(index_html)

    prefix = f"PUB_{report}_"
    files = []
    for href in parser.hrefs:
        name = href.rsplit("/", 1)[-1]
        if not name.startswith(prefix):
            continue
        if not name.lower().endswith((".xml", ".csv")):
            continue
        files.append(name)
    return sorted(set(files))


def harvest(report: str) -> tuple[int, int]:
    """Download missing files for one report. Returns (downloaded, already_held)."""
    out_dir = OUT_ROOT / report
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{report}] reading directory listing...")
    remote = list_remote_files(report)
    if not remote:
        raise RuntimeError(
            f"[{report}] listing contained no files matching PUB_{report}_*. "
            "IESO may have changed its naming or page structure. Go look at the "
            "page and fix this script - do not let it fail silently."
        )

    existing = {p.name for p in out_dir.iterdir() if p.is_file()}
    missing = [name for name in remote if name not in existing]

    print(
        f"[{report}] remote={len(remote)} "
        f"held={len(remote) - len(missing)} "
        f"to_download={len(missing)}"
    )

    downloaded = 0
    for name in missing:
        content = fetch(f"{BASE}/{report}/{name}")
        # Write to a temp name first, then rename. An interrupted download
        # leaves a .partial file rather than a truncated one that later runs
        # would wrongly treat as already harvested.
        tmp = out_dir / f".{name}.partial"
        tmp.write_bytes(content)
        tmp.rename(out_dir / name)
        downloaded += 1
        print(f"    ok  {name}  ({len(content):,} bytes)")

    return downloaded, len(remote) - len(missing)


def directory_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def main() -> int:
    total_new = 0
    failures: list[str] = []

    for report in REPORTS:
        try:
            new, _ = harvest(report)
            total_new += new
        except Exception as error:  # noqa: BLE001 - one bad report must not stop the rest
            print(f"[{report}] FAILED: {error}", file=sys.stderr)
            failures.append(report)

    size_mb = directory_size(OUT_ROOT) / 1_000_000 if OUT_ROOT.exists() else 0
    print(f"\nDownloaded {total_new} new file(s). Archive size: {size_mb:.1f} MB")

    # Write to the GitHub Actions run summary so the result is visible
    # without opening the logs.
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write("## IESO harvest\n\n")
            f.write(f"- New files: **{total_new}**\n")
            f.write(f"- Archive size: **{size_mb:.1f} MB**\n")
            if failures:
                f.write(f"- Failed reports: {', '.join(failures)}\n")

    # Only treat the run as failed if nothing at all was downloaded,
    # so a single flaky report does not spam failure notifications.
    if failures and total_new == 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
