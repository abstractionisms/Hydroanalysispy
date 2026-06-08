"""Groundwater data boundary tests."""

from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

from hydrology.data.groundwater import (
    DEFAULT_GROUNDWATER_PARAMETER_CODES,
    SAFE_GROUNDWATER_COLUMNS,
    fetch_usgs_groundwater_measurements,
    normalize_usgs_groundwater_measurements,
)


def test_normalize_usgs_groundwater_measurements_drops_private_fields_and_marks_depth_eligible():
    raw = pd.DataFrame(
        [
            {
                "monitoring_location_id": "USGS-12422500",
                "time": "2024-01-15T00:00:00Z",
                "parameter_code": "72019",
                "parameter_name": "Depth to water level, feet below land surface",
                "value": "12.4",
                "unit_of_measure": "ft",
                "vertical_datum": "NAVD88",
                "approval_status": "Approved",
                "latitude": 47.0,
                "longitude": -117.0,
                "owner_name": "private",
                "address": "private",
                "parcel": "private",
            }
        ]
    )

    normalized = normalize_usgs_groundwater_measurements(raw)
    row = normalized.iloc[0]

    assert list(normalized.columns) == SAFE_GROUNDWATER_COLUMNS
    assert "owner_name" not in normalized.columns
    assert "address" not in normalized.columns
    assert "parcel" not in normalized.columns
    assert row["site_id"] == "12422500"
    assert row["depth_to_water_ft"] == pytest.approx(12.4)
    assert pd.isna(row["water_level_ft"])
    assert bool(row["analysis_eligible"]) is True
    assert row["data_use"] == "analysis"


def test_normalize_usgs_groundwater_measurements_maps_water_level_elevation():
    raw = pd.DataFrame(
        [
            {
                "monitoringLocationId": "USGS-12422500",
                "dateTime": "2024-02-15",
                "parameterCode": "62611",
                "value": "2011.2",
            }
        ]
    )

    normalized = normalize_usgs_groundwater_measurements(raw)
    row = normalized.iloc[0]

    assert row["water_level_ft"] == pytest.approx(2011.2)
    assert pd.isna(row["depth_to_water_ft"])
    assert bool(row["analysis_eligible"]) is True


def test_normalize_usgs_groundwater_measurements_marks_missing_measurement_limited():
    raw = pd.DataFrame(
        [
            {
                "monitoring_location_id": "USGS-12422500",
                "time": None,
                "parameter_code": "00060",
                "value": "100",
            }
        ]
    )

    normalized = normalize_usgs_groundwater_measurements(raw)
    row = normalized.iloc[0]

    assert bool(row["analysis_eligible"]) is False
    assert row["data_use"] == "limited"
    assert "missing measurement date" in row["notes"]
    assert "missing supported groundwater measurement" in row["notes"]


def test_fetch_usgs_groundwater_measurements_uses_waterdata_client(monkeypatch):
    calls = {}

    def fake_get_field_measurements(**params):
        calls.update(params)
        raw = pd.DataFrame(
            [
                {
                    "monitoring_location_id": "USGS-12422500",
                    "time": "2024-01-15",
                    "parameter_code": "72019",
                    "value": "12.4",
                }
            ]
        )
        return raw, {"ok": True}

    fake_dataretrieval = types.SimpleNamespace(
        waterdata=types.SimpleNamespace(get_field_measurements=fake_get_field_measurements)
    )
    monkeypatch.setitem(sys.modules, "dataretrieval", fake_dataretrieval)

    normalized = fetch_usgs_groundwater_measurements(
        ["12422500"],
        start_date="2024-01-01",
        end_date="2024-02-01",
    )

    assert calls["monitoring_location_id"] == ["USGS-12422500"]
    assert calls["parameter_code"] == DEFAULT_GROUNDWATER_PARAMETER_CODES
    assert calls["time"] == "2024-01-01/2024-02-01"
    assert calls["limit"] == 50000
    assert calls["skip_geometry"] is False
    assert normalized.iloc[0]["site_id"] == "12422500"


def test_fetch_usgs_groundwater_measurements_requires_site_or_bbox():
    with pytest.raises(ValueError, match="site_ids or bbox"):
        fetch_usgs_groundwater_measurements()
