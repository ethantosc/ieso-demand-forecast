#!/usr/bin/env python3
"""Download IESO annual hourly demand CSVs into data/raw/demand/.

    python -m src.ingest.fetch_demand              # 2002 .. current year
    python -m src.ingest.fetch_demand --from 2015  # narrower range
    python -m src.ingest.fetch_demand --force      # re-download everything

This exists so the dataset can be rebuilt from nothing. Raw files are never
edited afterwards, and every download is recorded in ``manifest.json`` with a
timestamp, byte count and SHA-256, so a file that silently changes underneath
us is detectable rather than mysterious.

Two files behave differently and the difference matters:

* A completed year (2002..last year) is settled. Once downloaded it is skipped
  on later runs.
* The current year is still being written. It is re-downloaded every run, and
  its manifest entry moves with it. Anything computed from it is provisional.

Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime
from pathlib import Path

BASE = "https://reports-public.ieso.ca/public/Demand"
OUT_DIR = Path("data/raw/demand")
MANIFEST = OUT_DIR / "manifest.json"

FIRST_YEAR = 2002  # the market opened 2002-05-01; earlier years are a different file
TIMEOUT = 60
RETRIES = 3
BACKOFF = 5  # seconds, doubled each retry
PAUSE = 0.5  # be polite between files
USER_AGENT = "ieso-demand-forecast/1.0 (personal research project)"


def fetch(url: str) -> bytes:
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


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {}


def main() -> int:
    current_year = date.today().year
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="start", type=int, default=FIRST_YEAR)
    parser.add_argument("--to", dest="end", type=int, default=current_year)
    parser.add_argument("--force", action="store_true", help="re-download settled years too")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    downloaded = skipped = 0
    for year in range(args.start, args.end + 1):
        name = f"PUB_Demand_{year}.csv"
        path = OUT_DIR / name
        is_current = year == current_year

        if path.exists() and not is_current and not args.force:
            print(f"  {name:24s} already held, skipping")
            skipped += 1
            continue

        url = f"{BASE}/{name}"
        try:
            payload = fetch(url)
        except RuntimeError as error:
            # A missing current-year file is normal in early January.
            print(f"  {name:24s} FAILED - {error}", file=sys.stderr)
            if is_current:
                continue
            return 1

        digest = hashlib.sha256(payload).hexdigest()
        previous = manifest.get(name, {})
        changed = previous.get("sha256") not in (None, digest)

        path.write_bytes(payload)
        manifest[name] = {
            "url": url,
            "downloaded_at": datetime.now().isoformat(timespec="seconds"),
            "bytes": len(payload),
            "sha256": digest,
            "settled": not is_current,
        }
        flag = "  [CONTENT CHANGED]" if changed else ""
        note = " (current year, provisional)" if is_current else ""
        print(f"  {name:24s} {len(payload):>9,} bytes{note}{flag}")
        downloaded += 1
        time.sleep(PAUSE)

    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\n{downloaded} downloaded, {skipped} already held. Manifest: {MANIFEST}")
    print("\nNext:  python -m src.ingest.demand")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
