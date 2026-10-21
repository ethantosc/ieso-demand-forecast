"""Shared test helpers, provided as fixtures.

Test modules never import from each other ("from tests.x import y"). This
folder has no __init__.py, and a third-party package that happens to be
called `tests` - Colab ships one - would take precedence over it and break
the import. pytest loads this file by location, so fixtures here are safe.
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from src.transform import features as F


def synthetic_staging(days: int = 60, seed: int = 0) -> pl.DataFrame:
    """Hourly staging rows from 2025-02-01 EST, shaped like the real staging Parquet."""
    rng = np.random.default_rng(seed)
    start = datetime(2025, 2, 1)
    ts_est = [start + timedelta(hours=i) for i in range(days * 24)]
    frame = pl.DataFrame({
        "ts_utc": [t + timedelta(hours=5) for t in ts_est],
        "market_demand_mw": 17000 + rng.normal(0, 500, len(ts_est)),
    })
    return frame.with_columns(
        pl.col("ts_utc").dt.replace_time_zone("UTC").dt.convert_time_zone(F.LOCAL_TZ)
        .dt.replace_time_zone(None).dt.date().alias("local_date")
    )


@pytest.fixture
def make_staging():
    return synthetic_staging
