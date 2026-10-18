"""Tests for the one-file-at-a-time Ecology DataStream cache."""

from datetime import date

import pandas as pd

from hydrology.data.ecology_datastream import (
    DISCHARGE_SUFFIX,
    STAGE_SUFFIX,
    EcologyDataError,
    download_queue,
    load_cached_series,
    parse_daily_table,
    parse_published_years,
    pending_downloads,
    record_status,
    refresh_published_years,
    water_years_overlapping,
    year_record_status,
)

DISCHARGE = """\
32B100--Touchet R. @ Bolles

DATE         Discharge (cfs)   QUALITY
----------   ---------------   -------
10/01/2023              51.5         1
10/02/2023              49.9         1
11/02/2023                         254

Quality Codes Key:
1 - Good quality REVIEWED data
254 - Rating table exceeded (data will not be reported)
"""

STAGE = """\
DATE         Stage (ft.)   QUALITY
----------   -----------   -------
10/01/2023          3.08         1
10/02/2023          3.07         1
"""

PROVISIONAL = DISCHARGE.replace("         1", "         2", 1)


def test_water_years_follow_the_october_boundary():
    assert water_years_overlapping(date(2024, 1, 1), date(2024, 12, 31)) == [2024, 2025]
    assert water_years_overlapping(date(2024, 10, 1), date(2025, 9, 30)) == [2025]


def test_parser_keeps_withheld_days_blank_and_stops_at_the_code_key():
    table = parse_daily_table(DISCHARGE)

    assert list(table["value"].iloc[:2]) == [51.5, 49.9]
    assert pd.isna(table["value"].iloc[2])
    assert list(table["quality"]) == [1, 1, 254]
    assert table["date"].dt.tz.zone == "America/Los_Angeles"


def test_published_years_ignore_other_stations():
    html = "32B100_2024_DSG_DV 32B100_2024_STG_DV 35B150_2024_DSG_DV"
    years = parse_published_years(html, "32b100")

    assert years == {"discharge_years": [2024], "stage_years": [2024]}


def test_queue_saves_one_file_at_a_time_and_stops_on_failure(tmp_path):
    calls = []

    def fetch(url):
        calls.append(url)
        if len(calls) == 1:
            return b"32B100_2023_DSG_DV 32B100_2024_DSG_DV 32B100_2024_STG_DV"
        if "2023_DSG" in url:
            return DISCHARGE.encode()
        raise EcologyDataError("DataStream returned HTTP 503.")

    refresh_published_years("32B100", fetch=fetch, cache_dir=tmp_path)
    pending = pending_downloads(
        "32B100",
        [2023, 2024],
        DISCHARGE_SUFFIX,
        cache_dir=tmp_path,
    )
    pauses = []
    result = download_queue(
        "32B100",
        pending,
        pause_seconds=2,
        fetch=fetch,
        sleep=pauses.append,
        cache_dir=tmp_path,
    )

    assert [item[0] for item in pending] == [2023, 2024]
    assert len(result.saved) == 1
    assert result.stopped_on == "DataStream returned HTTP 503."
    assert pauses == [2]
    assert not any("STG" in url for url in calls)
    assert pending_downloads("32B100", [2023, 2024], STAGE_SUFFIX, cache_dir=tmp_path) == []


def test_stage_waits_until_discharge_for_that_year_is_saved(tmp_path):
    refresh_published_years(
        "32B100",
        fetch=lambda url: b"32B100_2024_DSG_DV 32B100_2024_STG_DV",
        cache_dir=tmp_path,
    )
    assert pending_downloads("32B100", [2024], STAGE_SUFFIX, cache_dir=tmp_path) == []

    download_queue(
        "32B100",
        [(2024, DISCHARGE_SUFFIX)],
        pause_seconds=0,
        fetch=lambda url: DISCHARGE.encode(),
        cache_dir=tmp_path,
    )

    assert pending_downloads("32B100", [2024], STAGE_SUFFIX, cache_dir=tmp_path) == [
        (2024, STAGE_SUFFIX)
    ]
    download_queue(
        "32B100",
        [(2024, STAGE_SUFFIX)],
        pause_seconds=0,
        fetch=lambda url: STAGE.encode(),
        cache_dir=tmp_path,
    )
    frame = load_cached_series("32B100", date(2023, 10, 1), date(2023, 10, 2), tmp_path)

    assert list(frame["Discharge_cfs"]) == [51.5, 49.9]
    assert list(frame["Gage_Height_ft"]) == [3.08, 3.07]


def test_unknown_station_does_not_call_datastream(tmp_path):
    def fetch(url):
        raise AssertionError(url)

    try:
        download_queue("ZZZ999", [(2024, DISCHARGE_SUFFIX)], fetch=fetch, cache_dir=tmp_path)
    except EcologyDataError as exc:
        assert "catalog" in str(exc)
    else:
        raise AssertionError("expected a catalog error")


def test_provisional_year_is_labeled_and_reviewed_year_is_not_pending(tmp_path):
    refresh_published_years(
        "32B100",
        fetch=lambda url: b"32B100_2024_DSG_DV",
        cache_dir=tmp_path,
    )
    download_queue(
        "32B100",
        [(2024, DISCHARGE_SUFFIX)],
        fetch=lambda url: PROVISIONAL.encode(),
        cache_dir=tmp_path,
    )

    assert year_record_status("32B100", 2024, tmp_path) == "provisional"
    assert record_status({1, 8, 50}) == "reviewed"
    assert pending_downloads("32B100", [2024], DISCHARGE_SUFFIX, cache_dir=tmp_path) == []
    assert isinstance(load_cached_series("32B100", date(2023, 10, 1), date(2023, 10, 1), tmp_path), pd.DataFrame)
