"""Parsers for the two external formats: IESO DATotals XML and ECCC hourly CSV."""

from datetime import date, datetime
from pathlib import Path

import pytest

from src.ingest import fetch_weather, forecasts

ARCHIVE = Path("data/raw/forecasts/DATotals")


@pytest.mark.skipif(not (ARCHIVE / "PUB_DATotals_20261018.xml").exists(), reason="archive file not present")
def test_real_datotals_file():
    rows = forecasts.parse_file(ARCHIVE / "PUB_DATotals_20261018.xml")
    assert len(rows) == 24
    first = rows[0]
    assert first["delivery_date"] == date(2026, 10, 18)
    assert first["hour_ending"] == 1
    assert first["total_load_mw"] == pytest.approx(15894.6)
    assert first["total_load_mw"] == pytest.approx(first["total_energy_mw"] - 314.5)
    assert first["created_at"] == datetime(2026, 10, 17, 12, 31, 46)


def test_namespace_trap_raises_rather_than_returning_none(tmp_path):
    bare = tmp_path / "PUB_DATotals_20260101.xml"
    bare.write_text("<Document><DocHeader><CreatedAt>2025-12-31T12:31:00</CreatedAt></DocHeader></Document>")
    with pytest.raises(ValueError, match="missing element"):
        forecasts.parse_file(bare)


ECCC = ('﻿"Longitude (x)","Latitude (y)","Station Name","Climate ID","Date/Time (LST)","Year","Month",'
        '"Day","Time (LST)","Temp (°C)","Temp Flag"\n'
        '"-79.63","43.68","{name}","6158731","2024-07-01 00:00","2024","07","01","00:00","21.4",""\n'
        '"-79.63","43.68","{name}","6158731","2024-07-01 01:00","2024","07","01","01:00","",""\n'
        '"-79.63","43.68","{name}","6158731","2024-07-01 02:00","2024","07","01","02:00","20.1",""\n')


def test_eccc_csv_parses_in_local_standard_time(tmp_path):
    path = tmp_path / "51459_2024_07.csv"
    path.write_text(ECCC.format(name="TORONTO INTL A"), encoding="utf-8")
    frame = fetch_weather.parse_file(path, "TORONTO")
    assert frame["ts_est"].to_list()[0] == datetime(2024, 7, 1, 0, 0)
    assert frame["temp"].to_list() == [21.4, None, 20.1]


def test_wrong_station_is_refused(tmp_path):
    path = tmp_path / "51459_2024_07.csv"
    path.write_text(ECCC.format(name="OTTAWA INTL A"), encoding="utf-8")
    with pytest.raises(ValueError, match="does not contain TORONTO"):
        fetch_weather.parse_file(path, "TORONTO")


def test_empty_month_is_skipped(tmp_path):
    path = tmp_path / "51459_2024_07.csv"
    path.write_text(ECCC.format(name="TORONTO INTL A").replace('"21.4"', '""').replace('"20.1"', '""'))
    assert fetch_weather.parse_file(path, "TORONTO") is None
