"""Public groundwater data helpers.

This module starts with USGS field measurements only. Ecology EIM and well-log
sources need separate source-specific modules because their privacy and
analysis-eligibility rules differ.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from ..core.logging_setup import get_logger

logger = get_logger(__name__)

USGS_SOURCE = "USGS"
USGS_FIELD_MEASUREMENTS = "usgs_field_measurements"
DEPTH_TO_WATER_PARAM_CODES = {"72019"}
WATER_LEVEL_ELEVATION_PARAM_CODES = {"62610", "62611", "62612"}
DEFAULT_GROUNDWATER_PARAMETER_CODES = sorted(
    DEPTH_TO_WATER_PARAM_CODES | WATER_LEVEL_ELEVATION_PARAM_CODES
)

SAFE_GROUNDWATER_COLUMNS = [
    "site_id",
    "monitoring_location_id",
    "source",
    "source_type",
    "measurement_date",
    "parameter_code",
    "parameter_name",
    "depth_to_water_ft",
    "water_level_ft",
    "unit",
    "vertical_datum",
    "approval_status",
    "qualifier",
    "measuring_agency",
    "latitude",
    "longitude",
    "location_precision",
    "data_use",
    "analysis_eligible",
    "notes",
]


def _as_list(value: str | Iterable[str] | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _usgs_monitoring_location_id(site_id: str) -> str:
    site = str(site_id)
    return site if site.upper().startswith("USGS-") else f"USGS-{site}"


def _strip_usgs_prefix(monitoring_location_id: object) -> str | None:
    if pd.isna(monitoring_location_id):
        return None
    text = str(monitoring_location_id)
    return text.split("-", 1)[1] if text.upper().startswith("USGS-") else text


def _time_interval(start_date: str | None, end_date: str | None) -> str | None:
    if start_date and end_date:
        return f"{start_date}/{end_date}"
    if start_date:
        return f"{start_date}/.."
    if end_date:
        return f"../{end_date}"
    return None


def _first_existing(row: pd.Series, names: list[str]):
    for name in names:
        if name in row and pd.notna(row[name]):
            return row[name]
    return None


def _geometry_lon_lat(geometry):
    if geometry is None:
        return None, None
    try:
        if pd.isna(geometry):
            return None, None
    except (TypeError, ValueError):
        pass
    try:
        if getattr(geometry, "geom_type", None) == "Point":
            return float(geometry.x), float(geometry.y)
    except Exception:
        return None, None
    return None, None


def _location_from_row(row: pd.Series) -> tuple[float | None, float | None]:
    lon = _first_existing(row, ["longitude", "lon", "dec_long_va", "x"])
    lat = _first_existing(row, ["latitude", "lat", "dec_lat_va", "y"])
    if lon is None or lat is None:
        geom_lon, geom_lat = _geometry_lon_lat(row.get("geometry"))
        lon = lon if lon is not None else geom_lon
        lat = lat if lat is not None else geom_lat
    try:
        return float(lat), float(lon)
    except (TypeError, ValueError):
        return None, None


def _safe_numeric(value):
    return pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]


def _normalize_row(row: pd.Series) -> dict:
    monitoring_location_id = _first_existing(row, ["monitoring_location_id", "monitoringLocationId", "site_no"])
    parameter_code = _first_existing(row, ["parameter_code", "parameterCode", "parm_cd"])
    parameter_code = str(parameter_code).split(".", 1)[0] if parameter_code is not None else None
    measurement_date = pd.to_datetime(
        _first_existing(row, ["time", "measurement_date", "dateTime", "datetime"]),
        errors="coerce",
    )
    value = _safe_numeric(_first_existing(row, ["value", "lev_va", "result_va"]))
    lat, lon = _location_from_row(row)

    depth_to_water_ft = value if parameter_code in DEPTH_TO_WATER_PARAM_CODES else pd.NA
    water_level_ft = value if parameter_code in WATER_LEVEL_ELEVATION_PARAM_CODES else pd.NA
    has_value = pd.notna(depth_to_water_ft) or pd.notna(water_level_ft)
    has_date = pd.notna(measurement_date)
    has_site = monitoring_location_id is not None

    note_parts = []
    if not has_site:
        note_parts.append("missing site ID")
    if not has_date:
        note_parts.append("missing measurement date")
    if not has_value:
        note_parts.append("missing supported groundwater measurement")

    return {
        "site_id": _strip_usgs_prefix(monitoring_location_id),
        "monitoring_location_id": monitoring_location_id,
        "source": USGS_SOURCE,
        "source_type": USGS_FIELD_MEASUREMENTS,
        "measurement_date": measurement_date if has_date else pd.NaT,
        "parameter_code": parameter_code,
        "parameter_name": _first_existing(row, ["parameter_name", "parameterName", "parameter_description"]),
        "depth_to_water_ft": depth_to_water_ft,
        "water_level_ft": water_level_ft,
        "unit": _first_existing(row, ["unit_of_measure", "unitOfMeasure", "unit"]),
        "vertical_datum": _first_existing(row, ["vertical_datum", "verticalDatum"]),
        "approval_status": _first_existing(row, ["approval_status", "approvalStatus"]),
        "qualifier": _first_existing(row, ["qualifier"]),
        "measuring_agency": _first_existing(row, ["measuring_agency", "measuringAgency"]),
        "latitude": lat,
        "longitude": lon,
        "location_precision": "measured" if lat is not None and lon is not None else "unavailable",
        "data_use": "analysis" if has_value and has_date and has_site else "limited",
        "analysis_eligible": bool(has_value and has_date and has_site),
        "notes": "; ".join(note_parts),
    }


def normalize_usgs_groundwater_measurements(raw: pd.DataFrame | None) -> pd.DataFrame:
    """Normalize USGS field measurements into HydroPlot's safe groundwater schema."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=SAFE_GROUNDWATER_COLUMNS)

    rows = [_normalize_row(row) for _, row in raw.iterrows()]
    normalized = pd.DataFrame(rows, columns=SAFE_GROUNDWATER_COLUMNS)
    if not normalized.empty:
        normalized["measurement_date"] = pd.to_datetime(normalized["measurement_date"], errors="coerce")
        normalized = normalized.sort_values(["site_id", "measurement_date"], na_position="last").reset_index(drop=True)
    return normalized


def fetch_usgs_groundwater_measurements(
    site_ids: str | Iterable[str] | None = None,
    *,
    bbox: list[float] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    parameter_codes: Iterable[str] | None = None,
    limit: int | None = 50000,
    skip_geometry: bool = False,
) -> pd.DataFrame:
    """Fetch public USGS groundwater field measurements and return safe normalized rows."""
    if site_ids is None and bbox is None:
        raise ValueError("Provide site_ids or bbox for groundwater field-measurement retrieval")

    try:
        from dataretrieval import waterdata
    except ImportError as exc:
        raise ImportError(
            "Install `dataretrieval` to fetch USGS groundwater field measurements."
        ) from exc

    monitoring_location_id = None
    if site_ids is not None:
        monitoring_location_id = [_usgs_monitoring_location_id(site_id) for site_id in _as_list(site_ids)]

    params = {
        "monitoring_location_id": monitoring_location_id,
        "parameter_code": list(parameter_codes or DEFAULT_GROUNDWATER_PARAMETER_CODES),
        "time": _time_interval(start_date, end_date),
        "bbox": bbox,
        "limit": limit,
        "skip_geometry": skip_geometry,
    }
    params = {key: value for key, value in params.items() if value is not None}

    raw, _metadata = waterdata.get_field_measurements(**params)
    return normalize_usgs_groundwater_measurements(raw)
