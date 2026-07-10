"""
Shared utilities for the Hydrology Dashboard multipage app.

Contains cached data functions, UI components, and processing helpers
used across multiple pages.
"""

import sys
from pathlib import Path

# Ensure the hydrology package is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import io

from hydrology.core import DEFAULT_DISCHARGE_CODE
from hydrology.data.inventory import load_inventory, get_site_info
from hydrology.data.usgs import (
    fetch_waterml_data, parse_waterml, fetch_daily_values, fetch_stage_data,
    fetch_instantaneous_values, DEFAULT_PARAM_DISCHARGE, DEFAULT_PARAM_STAGE,
    fetch_current_conditions, fetch_daily_percentiles, classify_condition
)
from hydrology.data.climate import fetch_climate_data, fetch_nearest_station_info
from hydrology.data.nwm import NWMClient, compare_nwm_usgs, get_forecast_skill
from hydrology.analysis.alerts import (
    AlertMonitor, AlertThreshold, create_flood_alert, create_low_flow_alert
)
from hydrology.analysis.multisite import MultiSiteAnalyzer, quick_correlation_check
from hydrology.analysis.flood_events import FloodEventAnalyzer, calculate_event_statistics
from hydrology.data.usgs import fetch_peak_streamflow, get_top_flood_events
from hydrology.data.national_inventory import get_national_inventory, get_region_inventory, get_inventory_summary
from hydrology.core.huc_regions import HUC2_REGIONS, get_region_name, get_region_center, US_CENTER
from hydrology.visualization import create_multi_plot, PlotLayout
from hydrology.visualization.plots import AVAILABLE_PLOTS
from hydrology.scripts.analyze_sites import analyze_correlation
from hydrology.core.logging_setup import get_logger
from hydrology.core.timezone import ensure_utc
from hydrology.app.plot_config import (
    multi_plot_selector, single_plot_selector,
    CLIMATE_PLOTS, DISCHARGE_PLOTS, STAGE_PLOTS, REACH_PLOTS,
    get_display_name
)
from hydrology.app.styles import (
    apply_custom_css, render_site_header, render_availability_badges,
    render_metric_cards, render_footer
)

logger = get_logger(__name__)


# =============================================================================
# CACHED DATA FUNCTIONS
# =============================================================================

@st.cache_data(ttl=86400)
def get_inventory():
    """Load and cache inventory data."""
    return load_inventory()




@st.cache_data(ttl=3600, show_spinner="Loading site conditions...")
def get_site_conditions(site_ids: list) -> dict:
    """Fetch and classify current streamflow conditions for all sites."""
    details = get_site_condition_details(site_ids)
    return {
        sid: values["percentile"]
        for sid, values in details.items()
        if values.get("percentile") is not None
    }


@st.cache_data(ttl=3600, show_spinner="Loading site condition details...")
def get_site_condition_details(site_ids: list) -> dict:
    """Fetch current flow plus condition metadata for map hovers and popups."""
    current = fetch_current_conditions(site_ids)
    percentiles = fetch_daily_percentiles(site_ids)
    details = {}

    for sid in site_ids:
        flow = current.get(sid)
        pcts = percentiles.get(sid)
        if flow is None:
            continue
        percentile = classify_condition(flow, pcts) if pcts else None
        details[sid] = {
            "flow_cfs": flow,
            "percentile": percentile,
            "source": "USGS seasonal percentile" if percentile is not None else "USGS live flow",
        }

    if details and not any(v.get("percentile") is not None for v in details.values()):
        sorted_flows = sorted(
            ((sid, values["flow_cfs"]) for sid, values in details.items()),
            key=lambda item: item[1],
        )
        total = len(sorted_flows)
        for rank, (sid, _flow) in enumerate(sorted_flows):
            details[sid]["percentile"] = 100.0 * rank / max(total - 1, 1)
            details[sid]["source"] = "Relative live-flow rank among mapped sites"

    return details


def build_site_summary(site_id: str, site_info: dict, condition: dict | None = None) -> dict:
    """Build display-ready selected-site summary data."""
    from hydrology.visualization.map_utils import get_condition_label

    condition = condition or {}
    desc = site_info.get("description") or site_id
    lat = site_info.get("latitude")
    lon = site_info.get("longitude")
    subtitle = f"USGS {site_id}"
    if lat is not None and lon is not None:
        subtitle = f"{subtitle} | {float(lat):.4f}, {float(lon):.4f}"

    chips = []
    flow = condition.get("flow_cfs")
    if flow is not None:
        chips.append({"label": f"Flow {flow:,.0f} cfs", "state": "ready"})

    pctile = condition.get("percentile")
    if pctile is not None:
        chips.append({"label": get_condition_label(pctile), "state": "ready"})

    begin_date = str(site_info.get("begin_date") or "")
    if len(begin_date) >= 4 and begin_date[:4].isdigit():
        chips.append({"label": f"Record since {begin_date[:4]}", "state": "ready"})

    return {
        "title": desc,
        "subtitle": subtitle,
        "chips": chips,
    }


@st.cache_data(ttl=86400)
def get_cached_site_info(site_id: str):
    """Get site info from inventory."""
    return get_site_info(site_id)


@st.cache_data(ttl=86400, show_spinner=False)
def get_weather_station_info(lat: float, lon: float):
    """Get nearest weather station info."""
    try:
        return fetch_nearest_station_info(lat, lon)
    except Exception as e:
        logger.warning(f"Weather station lookup failed for ({lat}, {lon}): {e}")
        return None


def extract_site_id(site_string: str) -> str:
    """
    Safely extract site ID from selection string like "12345678 - Site Description".
    Returns the site ID or None if extraction fails.
    """
    if not site_string:
        return None
    parts = site_string.split(" - ", 1)
    if parts and parts[0].strip():
        return parts[0].strip()
    return None


@st.cache_data(ttl=43200, show_spinner=False)
def get_parameter_availability(site_id: str) -> dict:
    """
    Get exact availability dates for all parameters at a site using USGS series catalog.
    Returns dict mapping param_cd to {'begin_date': str, 'end_date': str, 'count': int}.
    """
    import requests

    url = "https://waterservices.usgs.gov/nwis/site/"
    params = {
        "format": "rdb",
        "sites": site_id,
        "seriesCatalogOutput": "true",
        "outputDataTypeCd": "dv",
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        if response.status_code != 200:
            return {}

        lines = response.text.strip().split('\n')
        data_lines = [l for l in lines if not l.startswith('#') and l.strip()]

        if len(data_lines) < 2:
            return {}

        header = data_lines[0].split('\t')
        try:
            parm_idx = header.index('parm_cd')
            begin_idx = header.index('begin_date')
            end_idx = header.index('end_date')
            count_idx = header.index('count_nu')
        except ValueError:
            return {}

        result = {}
        for line in data_lines[2:]:
            cols = line.split('\t')
            if len(cols) > max(parm_idx, begin_idx, end_idx, count_idx):
                parm_cd = cols[parm_idx]
                begin_date = cols[begin_idx]
                end_date = cols[end_idx]
                count = cols[count_idx]

                if parm_cd not in result or int(count or 0) > result[parm_cd].get('count', 0):
                    result[parm_cd] = {
                        'begin_date': begin_date,
                        'end_date': end_date,
                        'count': int(count) if count else 0
                    }

        return result

    except Exception as e:
        logger.warning(f"Failed to get parameter availability for {site_id}: {e}")
        return {}


@st.cache_data(ttl=43200, show_spinner=False)
def find_availability_windows(site_id: str, param_cd: str, check_iv: bool = False):
    """
    Find data availability for a parameter using USGS series catalog.
    Returns list of (start_year, end_year) tuples.
    """
    import requests

    current_year = date.today().year
    param_info = get_parameter_availability(site_id)

    if param_cd in param_info:
        info = param_info[param_cd]
        begin_date = info.get('begin_date', '')
        end_date = info.get('end_date', '')

        if begin_date:
            try:
                start_year = int(begin_date[:4])
                if end_date:
                    end_year = int(end_date[:4])
                    if end_year >= current_year - 1:
                        return [(start_year, "present")]
                    else:
                        return [(start_year, end_year)]
                else:
                    return [(start_year, "present")]
            except (ValueError, TypeError):
                pass

    if check_iv:
        try:
            url = "https://waterservices.usgs.gov/nwis/iv/"
            params = {
                "format": "json",
                "sites": site_id,
                "parameterCd": param_cd,
                "startDT": f"{current_year - 1}-01-01",
                "endDT": date.today().isoformat(),
            }
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                json_data = response.json()
                time_series = json_data.get('value', {}).get('timeSeries', [])
                if time_series:
                    values = time_series[0].get('values', [{}])[0].get('value', [])
                    if values:
                        return [(current_year - 1, "present")]
        except Exception:
            pass

    return None


def format_availability_windows(windows: list) -> str:
    """Format availability windows as readable string."""
    if not windows:
        return "not available"

    parts = []
    for start, end in windows:
        if end == "present":
            parts.append(f"{start}-present")
        elif start == end:
            parts.append(str(start))
        else:
            parts.append(f"{start}-{end}")

    return ", ".join(parts)


def analyze_data_coverage(df: pd.DataFrame, start_date: str, end_date: str):
    """
    Analyze actual data coverage and detect gaps.
    Returns dict with coverage info and any significant gaps.
    """
    if df is None or df.empty:
        return None

    req_start = pd.Timestamp(start_date)
    req_end = pd.Timestamp(end_date)
    requested_days = (req_end - req_start).days + 1

    actual_start = df.index.min()
    actual_end = df.index.max()
    actual_days = len(df)

    coverage_pct = (actual_days / requested_days) * 100 if requested_days > 0 else 0

    gaps = []
    if len(df) > 1:
        sorted_idx = df.index.sort_values()
        diffs = sorted_idx.to_series().diff()
        gap_threshold = pd.Timedelta(days=30)
        large_gaps = diffs[diffs > gap_threshold]

        for gap_end, gap_size in large_gaps.items():
            gap_start = gap_end - gap_size
            gaps.append({
                'start': gap_start.strftime('%Y-%m-%d'),
                'end': gap_end.strftime('%Y-%m-%d'),
                'days': gap_size.days
            })

    windows = []
    if len(df) > 0:
        sorted_idx = df.index.sort_values()
        window_start = sorted_idx[0]
        prev_date = sorted_idx[0]

        for curr_date in sorted_idx[1:]:
            gap = (curr_date - prev_date).days
            if gap > 365:
                windows.append((window_start.year, prev_date.year))
                window_start = curr_date
            prev_date = curr_date

        current_year = date.today().year
        if prev_date.year >= current_year - 1:
            windows.append((window_start.year, "present"))
        else:
            windows.append((window_start.year, prev_date.year))

    return {
        'actual_start': actual_start.strftime('%Y-%m-%d') if pd.notna(actual_start) else None,
        'actual_end': actual_end.strftime('%Y-%m-%d') if pd.notna(actual_end) else None,
        'actual_days': actual_days,
        'requested_days': requested_days,
        'coverage_pct': coverage_pct,
        'gaps': gaps,
        'windows': windows
    }


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_discharge_data(site_id: str, param_cd: str, start_str: str, end_str: str):
    """Fetch and parse discharge data - cached."""
    waterml = fetch_waterml_data(site_id, param_cd, start_str, end_str)
    if not waterml:
        return None

    df = parse_waterml(waterml)
    if df is None or df.empty:
        return None

    df = df.rename(columns={'value': 'Discharge_cfs'})
    df = df[df['Discharge_cfs'] > 0]
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_climate_cached_result(
    lat: float,
    lon: float,
    start_str: str,
    end_str: str,
    site_id: str | None = None,
    include_temp: bool = True,
    include_precip: bool = True,
):
    """Fetch normalized climate data with source metadata.

    Order of preference (reliability-first for SPI/dashboard):
      1. Meteostat nearest station (no special auth, fast)
      2. Daymet watershed grid (needs NASA Earthdata; often 401 without it)

    Set HYDRO_PREFER_DAYMET=1 to try Daymet first when credentials exist.
    """
    import os

    start_dt = datetime.strptime(start_str, '%Y-%m-%d')
    end_dt = datetime.strptime(end_str, '%Y-%m-%d')

    prefer_daymet = os.environ.get("HYDRO_PREFER_DAYMET", "").strip().lower() in (
        "1", "true", "yes", "on",
    )

    def _try_daymet():
        if not site_id:
            return None
        variables = []
        if include_precip:
            variables.append("prcp")
        if include_temp:
            variables.extend(["tmin", "tmax"])
        try:
            from hydrology.data.hyriver import get_daymet_climate

            daymet = get_daymet_climate(
                site_id, start_str, end_str, variables=variables or None
            )
            normalized = normalize_climate_columns(daymet)
            if normalized is not None and not normalized.empty:
                return {
                    "data": normalized,
                    "source": "Daymet",
                    "message": "Loaded gridded Daymet climate data for the selected gage.",
                }
        except Exception as e:
            logger.info(f"Daymet climate unavailable for {site_id}: {e}")
        return None

    def _try_meteostat():
        station_climate = normalize_climate_columns(
            fetch_climate_data(
                lat,
                lon,
                pd.Timestamp(start_dt),
                pd.Timestamp(end_dt),
                include_temp=include_temp,
                include_precip=include_precip,
            )
        )
        if station_climate is not None and not station_climate.empty:
            station = None
            try:
                station = fetch_nearest_station_info(lat, lon)
            except Exception:
                pass
            dist = None
            name = None
            if station:
                dist = station.get("distance_km")
                name = station.get("name")
            msg = "Loaded climate data from the nearest Meteostat station."
            if name and dist is not None:
                msg = (
                    f"Loaded Meteostat station “{name}” "
                    f"({float(dist):.1f} km from the gage)."
                )
            elif name:
                msg = f"Loaded Meteostat station “{name}”."
            return {
                "data": station_climate,
                "source": "Meteostat",
                "message": msg,
                "station_name": name,
                "station_distance_km": dist,
            }
        return None

    # Prefer Daymet only when asked AND credentials exist (otherwise it is a slow 401).
    if prefer_daymet:
        daymet_result = _try_daymet()
        if daymet_result:
            return daymet_result

    meteo = _try_meteostat()
    if meteo:
        return meteo

    # Last resort Daymet (may still 401 without Earthdata)
    daymet_result = _try_daymet()
    if daymet_result:
        return daymet_result

    return {
        "data": None,
        "source": "Unavailable",
        "message": (
            "Could not load climate data. Daymet needs NASA Earthdata login; "
            "Meteostat had no usable series for this location/date range."
        ),
    }


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_climate_cached(lat: float, lon: float, start_str: str, end_str: str, site_id: str | None = None):
    """Fetch normalized climate data - cached."""
    return fetch_climate_cached_result(lat, lon, start_str, end_str, site_id)["data"]


def normalize_climate_columns(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Normalize supported climate providers to app column names."""
    if df is None or df.empty:
        return df

    climate = df.copy()
    rename_map = {
        'precip_mm': 'Precip_mm',
        'prcp': 'Precip_mm',
        'tmean_c': 'Temp_C',
        'tavg': 'Temp_C',
    }
    climate = climate.rename(columns={k: v for k, v in rename_map.items() if k in climate.columns})

    if 'Temp_C' not in climate.columns and {'tmin_c', 'tmax_c'}.issubset(climate.columns):
        climate['Temp_C'] = (climate['tmin_c'] + climate['tmax_c']) / 2

    keep_cols = [col for col in ['Temp_C', 'Precip_mm'] if col in climate.columns]
    if not keep_cols:
        return None

    climate = ensure_utc(climate[keep_cols])
    if 'Precip_mm' in climate.columns:
        climate['Precip_mm'] = climate['Precip_mm'].fillna(0)
    if 'Temp_C' in climate.columns:
        climate['Temp_C'] = climate['Temp_C'].ffill().bfill()
    return climate


# =============================================================================
# UI COMPONENTS
# =============================================================================

FIPS_TO_STATE = {
    '16': 'Idaho', '30': 'Montana', '32': 'Nevada',
    '41': 'Oregon', '53': 'Washington', '56': 'Wyoming',
}


def get_site_options(inventory_df):
    """Build site selection options list from inventory."""
    return [f"{row['site_id']} - {str(row.get('description', ''))[:40]}"
            for _, row in inventory_df.iterrows()]


def _filter_inventory(inventory_df, search_text="", state_filter="All States"):
    """Filter inventory by search text and state."""
    filtered = inventory_df.copy()

    if state_filter != "All States":
        state_cd = next((k for k, v in FIPS_TO_STATE.items() if v == state_filter), None)
        if state_cd:
            filtered = filtered[filtered['state_cd'].astype(str) == state_cd]

    if search_text:
        search = search_text.lower()
        mask = (
            filtered['site_id'].astype(str).str.contains(search, na=False) |
            filtered['description'].astype(str).str.lower().str.contains(search, na=False)
        )
        filtered = filtered[mask]

    return filtered


def site_picker(inventory_df, key="site", label="Select Site",
                location="sidebar", multi=False, max_selections=None,
                show_search=True):
    """
    Site picker with search filter and state grouping.
    Returns selected site_id string (or list for multi).
    """
    container = st.sidebar if location == "sidebar" else st

    # Search and filter controls
    if show_search:
        col_search, col_state = container.columns([2, 1])
        with col_search:
            search = st.text_input(
                "Search sites", placeholder="River name or site ID...",
                key=f"{key}_search", label_visibility="collapsed",
            )
        with col_state:
            states = ["All States"] + sorted(FIPS_TO_STATE.values())
            state_filter = st.selectbox(
                "State", states, key=f"{key}_state", label_visibility="collapsed",
            )
    else:
        search = ""
        state_filter = "All States"

    filtered = _filter_inventory(inventory_df, search, state_filter)

    if filtered.empty:
        container.warning("No sites match your search")
        return [] if multi else None

    site_options = [f"{row['site_id']} - {str(row.get('description', ''))[:60]}"
                    for _, row in filtered.iterrows()]

    if show_search and (search or state_filter != "All States"):
        container.caption(f"{len(site_options)} sites found")

    if multi:
        kwargs = {"max_selections": max_selections} if max_selections else {}
        selected = container.multiselect(label, site_options, key=f"{key}_multi", **kwargs)
        return [extract_site_id(s) for s in selected]
    else:
        # Determine default index: query params > global state > preferred site > 0
        default_index = 0
        last_used_key = f"{key}_last_site"
        preferred_site = "12422500"  # Spokane River at Spokane, WA

        # On fresh page load, clear stale widget key so global state wins
        page_init_key = f"{key}_page_initialized"
        widget_key = f"{key}_select"
        if page_init_key not in st.session_state:
            st.session_state[page_init_key] = True
            if widget_key in st.session_state:
                del st.session_state[widget_key]

        query_site = st.query_params.get("site")
        global_site = st.session_state.get("global_last_site")
        target_site = query_site or global_site or st.session_state.get(last_used_key) or preferred_site

        if target_site:
            for i, opt in enumerate(site_options):
                if opt.startswith(str(target_site)):
                    default_index = i
                    break

        selected = container.selectbox(label, site_options, index=default_index,
                                       key=f"{key}_select")
        site_id = extract_site_id(selected)

        # Persist selection to session_state, global state, and query params
        if site_id:
            st.session_state[last_used_key] = site_id
            st.session_state["global_last_site"] = site_id
            st.query_params["site"] = site_id

        return site_id


def sidebar_site_picker(inventory_df, key="site", label="Select Site", multi=False, max_selections=None):
    """Legacy wrapper - use site_picker() for new code."""
    return site_picker(inventory_df, key=key, label=label,
                       location="sidebar", multi=multi,
                       max_selections=max_selections, show_search=False)


def sidebar_date_range(key_prefix="global", default_start=None, default_end=None):
    """
    Render date range picker in sidebar with year sliders.
    Returns (start_date, end_date) tuple.
    """
    if default_start is None:
        default_start = date(2015, 1, 1)
    if default_end is None:
        default_end = date.today()

    min_year = 1900
    current_year = date.today().year
    today = date.today()

    start_year_key = f"{key_prefix}_start_year"
    end_year_key = f"{key_prefix}_end_year"

    if start_year_key not in st.session_state:
        st.session_state[start_year_key] = default_start.year
    if end_year_key not in st.session_state:
        st.session_state[end_year_key] = default_end.year

    st.sidebar.subheader("Date Range")

    start_year = st.sidebar.slider(
        "Start Year",
        min_value=min_year,
        max_value=current_year,
        value=st.session_state[start_year_key],
        key=f"{key_prefix}_start_slider"
    )

    end_year = st.sidebar.slider(
        "End Year",
        min_value=min_year,
        max_value=current_year,
        value=st.session_state[end_year_key],
        key=f"{key_prefix}_end_slider"
    )

    st.session_state[start_year_key] = start_year
    st.session_state[end_year_key] = end_year

    start = date(start_year, 1, 1)
    end = min(date(end_year, 12, 31), today)

    if start > end:
        start, end = end, start

    days_diff = (end - start).days
    st.sidebar.caption(f"{start} to {end} ({days_diff:,} days)")

    return start, end


def display_site_info(site_info: dict, show_check_button: bool = True):
    """Display site information and data availability in sidebar."""
    site_id = site_info.get('site_id', 'Unknown')
    lat = site_info.get('latitude')
    lon = site_info.get('longitude')
    desc = site_info.get('description', site_id)
    drain_area = site_info.get('drain_area_sq_mi')

    st.sidebar.markdown(f"**{desc}**")
    if lat and lon:
        st.sidebar.text(f"Lat: {float(lat):.4f}, Lon: {float(lon):.4f}")
    if drain_area:
        st.sidebar.text(f"Drainage Area: {drain_area} sq mi")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Availability")

    confirmed_key = f"confirmed_coverage_{site_id}"
    confirmed_coverage = st.session_state.get(confirmed_key)

    availability_text = []

    # Discharge
    if confirmed_coverage and confirmed_coverage.get('discharge'):
        dc = confirmed_coverage['discharge']
        windows_str = format_availability_windows(dc.get('windows', []))
        coverage_pct = dc.get('coverage_pct', 0)
        if coverage_pct < 90:
            availability_text.append(f"⚠️ **Discharge** ({windows_str}) - {coverage_pct:.0f}% coverage")
        else:
            availability_text.append(f"✅ **Discharge** ({windows_str})")
    else:
        discharge_windows = find_availability_windows(site_id, DEFAULT_PARAM_DISCHARGE)
        if discharge_windows:
            windows_str = format_availability_windows(discharge_windows)
            availability_text.append(f"✅ **Discharge** ({windows_str})")
        else:
            availability_text.append("❌ **Discharge** (not available)")

    # Gage Height
    if confirmed_coverage and confirmed_coverage.get('stage'):
        sc = confirmed_coverage['stage']
        windows_str = format_availability_windows(sc.get('windows', []))
        coverage_pct = sc.get('coverage_pct', 0)
        if coverage_pct < 90:
            availability_text.append(f"⚠️ **Gage Height** ({windows_str}) - {coverage_pct:.0f}% coverage")
        else:
            availability_text.append(f"✅ **Gage Height** ({windows_str})")
    elif confirmed_coverage:
        availability_text.append("❌ **Gage Height** (not in data)")
    else:
        stage_dv_windows = find_availability_windows(site_id, DEFAULT_PARAM_STAGE)
        if stage_dv_windows:
            windows_str = format_availability_windows(stage_dv_windows)
            availability_text.append(f"✅ **Gage Height** ({windows_str})")
        else:
            stage_iv_windows = find_availability_windows(site_id, DEFAULT_PARAM_STAGE, check_iv=True)
            if stage_iv_windows:
                windows_str = format_availability_windows(stage_iv_windows)
                availability_text.append(f"✅ **Gage Height** IV ({windows_str})")
            else:
                availability_text.append("❌ **Gage Height** (not available)")

    # Climate
    if lat and lon:
        station = get_weather_station_info(float(lat), float(lon))
        if station:
            dist = station.get('distance_km')
            name = station.get('name', 'Unknown')[:25]
            daily_start = station.get('daily_start')
            daily_end = station.get('daily_end')

            if daily_start and daily_end:
                coverage = f"{daily_start[:4]}-{daily_end[:4]}"
            elif daily_start:
                coverage = f"{daily_start[:4]}-present"
            else:
                coverage = "varies"

            if dist is not None:
                if dist < 20:
                    availability_text.append(f"✅ **Climate** ({name})")
                    availability_text.append(f"   ↳ {dist:.1f} km, {coverage}")
                elif dist < 50:
                    availability_text.append(f"⚠️ **Climate** ({name})")
                    availability_text.append(f"   ↳ {dist:.1f} km, {coverage}")
                else:
                    availability_text.append(f"⚠️ **Climate** ({name})")
                    availability_text.append(f"   ↳ {dist:.0f} km (distant), {coverage}")
            else:
                availability_text.append(f"⚠️ **Climate** ({name}, {coverage})")
        else:
            availability_text.append("❌ **Climate** (no station found)")
    else:
        availability_text.append("❌ **Climate** (no coordinates)")

    for text in availability_text:
        st.sidebar.markdown(text)


# =============================================================================
# PROCESSING FUNCTIONS
# =============================================================================

def process_site_data(site_id: str, lat: float, lon: float, start_str: str, end_str: str):
    """
    Fetch and process data for a single site.
    Returns dict with df_q, df_merged, analysis_results.
    """
    df_q = fetch_discharge_data(site_id, DEFAULT_DISCHARGE_CODE, start_str, end_str)
    if df_q is None or df_q.empty:
        return None

    try:
        df_stage = fetch_stage_data(site_id, start_str, end_str)
        if df_stage is not None and not df_stage.empty:
            df_stage = df_stage.rename(columns={'Stage_ft': 'Gage_Height_ft'})
            df_q = df_q.join(df_stage[['Gage_Height_ft']], how='left')
    except Exception as e:
        logger.info(f"Stage data not available for {site_id}: {e}")

    df_climate = fetch_climate_cached(lat, lon, start_str, end_str, site_id)

    df_merged = None
    analysis_results = None

    if df_q is not None and not df_q.empty and df_climate is not None and not df_climate.empty:
        df_q = ensure_utc(df_q)
        df_climate = ensure_utc(df_climate)

        df_merged = pd.merge(df_q, df_climate, left_index=True, right_index=True, how='inner')

        if df_merged.empty:
            logger.warning(f"Merge produced empty result. df_q: {len(df_q)} rows, df_climate: {len(df_climate)} rows")
            df_merged = None
        else:
            analysis_results = analyze_correlation(df_merged)

    discharge_coverage = analyze_data_coverage(df_q, start_str, end_str) if df_q is not None else None

    stage_coverage = None
    if df_q is not None and 'Gage_Height_ft' in df_q.columns:
        stage_df = df_q[['Gage_Height_ft']].dropna()
        stage_coverage = analyze_data_coverage(stage_df, start_str, end_str)

    return {
        'df_q': df_q,
        'df_merged': df_merged,
        'analysis_results': analysis_results,
        'discharge_count': len(df_q) if df_q is not None else 0,
        'climate_count': len(df_climate) if df_climate is not None else 0,
        'merged_count': len(df_merged) if df_merged is not None else 0,
        'discharge_coverage': discharge_coverage,
        'stage_coverage': stage_coverage
    }


def create_comparison_figure(plot_name, data_list, titles, nrows, ncols, dpi=150):
    """
    Create a comparison figure with multiple subplots.
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows), dpi=dpi, squeeze=False)

    plot_info = AVAILABLE_PLOTS.get(plot_name)
    plot_func = plot_info['function'] if plot_info else None

    for idx, (data, title) in enumerate(zip(data_list, titles)):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]

        if plot_func and data is not None:
            try:
                plot_requires = plot_info.get('requires', [])
                df_q = data.get('df_q')
                df_merged = data.get('df_merged')

                has_discharge = df_q is not None and not df_q.empty
                has_merged = df_merged is not None and not df_merged.empty
                has_gage = has_discharge and 'Gage_Height_ft' in df_q.columns and df_q['Gage_Height_ft'].notna().any()
                has_climate = has_merged and 'Precip_mm' in df_merged.columns

                missing = []
                if 'df_q' in plot_requires and not has_discharge:
                    missing.append("Discharge")
                if 'df_merged' in plot_requires and not has_merged:
                    if not has_discharge:
                        missing.append("Discharge")
                    if not has_climate:
                        missing.append("Climate")

                if missing:
                    msg = f"Missing: {', '.join(missing)}"
                    ax.text(0.5, 0.5, msg, ha='center', va='center',
                           transform=ax.transAxes, fontsize=11, color='red')
                    ax.text(0.5, 0.35, f"Q: {'✓' if has_discharge else '✗'}  Gage: {'✓' if has_gage else '✗'}  Climate: {'✓' if has_climate else '✗'}",
                           ha='center', va='center', transform=ax.transAxes, fontsize=9, color='gray')
                else:
                    plot_func(ax, **data, config={})
                ax.set_title(title, fontsize=10)
            except Exception as e:
                logger.error(f"Plot error for {title}: {e}")
                ax.text(0.5, 0.5, f"Plot Error:\n{str(e)[:50]}",
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title(title, fontsize=10)
        else:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=10)

    total_plots = len(data_list)
    for idx in range(total_plots, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].set_visible(False)

    fig.suptitle(f"Comparison: {plot_name}", fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig


def render_data_download(df: pd.DataFrame, filename_prefix: str = "discharge"):
    """Render a CSV download button for raw discharge data."""
    if df is None or df.empty:
        return
    csv_data = df.to_csv()
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name=f"{filename_prefix}_data.csv",
        mime="text/csv",
    )


def render_export_buttons(fig, filename_base: str, dpi: int):
    """Render export buttons for a figure."""
    st.subheader("Export")
    col1, col2 = st.columns(2)

    buf_png = io.BytesIO()
    fig.savefig(buf_png, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    buf_png.seek(0)
    col1.download_button("PNG (light)", buf_png, f"{filename_base}.png", "image/png")

    buf_pdf = io.BytesIO()
    fig.savefig(buf_pdf, format='pdf', dpi=dpi, bbox_inches='tight', facecolor='white')
    buf_pdf.seek(0)
    col2.download_button("PDF (light)", buf_pdf, f"{filename_base}.pdf", "application/pdf")


# Legacy compatibility - date_range_selector in main area (used by some modes)
def date_range_selector(key_prefix="", default_start=None, default_end=None):
    """
    Date range selector with year sliders and manual date inputs (main area version).
    Returns (start_date, end_date) tuple.
    """
    if default_start is None:
        default_start = date(2015, 1, 1)
    if default_end is None:
        default_end = date.today()

    min_year = 1900
    current_year = date.today().year

    start_year_key = f"{key_prefix}_start_year"
    end_year_key = f"{key_prefix}_end_year"

    if start_year_key not in st.session_state:
        st.session_state[start_year_key] = default_start.year
    if end_year_key not in st.session_state:
        st.session_state[end_year_key] = default_end.year

    start_key = f"{key_prefix}_start"
    end_key = f"{key_prefix}_end"
    today = date.today()

    col1, col2 = st.columns(2)
    with col1:
        start_year = st.slider(
            "Start Year",
            min_value=min_year,
            max_value=current_year,
            value=st.session_state[start_year_key],
            key=f"{key_prefix}_start_slider"
        )
        if st.session_state[start_year_key] != start_year:
            st.session_state[start_year_key] = start_year
            st.session_state[start_key] = date(start_year, 1, 1)
        start = date(start_year, 1, 1)

    with col2:
        end_year = st.slider(
            "End Year",
            min_value=min_year,
            max_value=current_year,
            value=st.session_state[end_year_key],
            key=f"{key_prefix}_end_slider"
        )
        if st.session_state[end_year_key] != end_year:
            st.session_state[end_year_key] = end_year
            st.session_state[end_key] = min(date(end_year, 12, 31), today)
        end = min(date(end_year, 12, 31), today)

    if start_key in st.session_state and st.session_state[start_key] > today:
        st.session_state[start_key] = today
    if end_key in st.session_state and st.session_state[end_key] > today:
        st.session_state[end_key] = today

    with st.expander("Fine-tune dates"):
        col3, col4 = st.columns(2)
        with col3:
            start_kwargs = dict(
                min_value=date(1900, 1, 1),
                max_value=today,
                key=start_key,
            )
            if start_key not in st.session_state:
                start_kwargs["value"] = start
            start = st.date_input("Start Date", **start_kwargs)
        with col4:
            end_kwargs = dict(
                min_value=date(1900, 1, 1),
                max_value=today,
                key=end_key,
            )
            if end_key not in st.session_state:
                end_kwargs["value"] = end
            end = st.date_input("End Date", **end_kwargs)

    if start > end:
        st.warning("⚠️ Start date was after end date - dates have been swapped.")
        start, end = end, start
        st.session_state[start_key] = start
        st.session_state[end_key] = end

    days_diff = (end - start).days
    if days_diff < 30:
        st.caption(f"Range: {start} → {end} ({days_diff} days) ⚠️ Short range")
    else:
        st.caption(f"Range: {start} → {end} ({days_diff:,} days)")

    return start, end


def plot_selector(available_plots=None):
    """Plot type selector. Returns list of selected plot names."""
    if available_plots is None:
        available_plots = AVAILABLE_PLOTS
    return multi_plot_selector(available_plots, key_prefix="")


def single_plot_selector_widget(key_suffix=""):
    """Select a single plot type for comparison mode."""
    return single_plot_selector(AVAILABLE_PLOTS, key_suffix=key_suffix)
