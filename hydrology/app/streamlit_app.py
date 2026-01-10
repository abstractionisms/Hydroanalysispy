"""
from hydrology.core import DEFAULT_DISCHARGE_CODE
Streamlit web application for hydrology analysis.

Features:
- Single site analysis with multiple plot types
- Data availability display with weather station distance
- Date range slider with visual selection
- Comparison modes: time periods, sites, and 2x2 grid
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import io
import sys

# Ensure the hydrology package is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hydrology.data.inventory import load_inventory, get_site_info
from hydrology.data.usgs import (
    fetch_waterml_data, parse_waterml, fetch_daily_values, fetch_stage_data,
    fetch_instantaneous_values, DEFAULT_PARAM_DISCHARGE, DEFAULT_PARAM_STAGE
)
from hydrology.data.climate import fetch_climate_data, fetch_nearest_station_info
from hydrology.data.nwm import NWMClient, compare_nwm_usgs, get_forecast_skill
from hydrology.analysis.alerts import (
    AlertMonitor, AlertThreshold, create_flood_alert, create_low_flow_alert
)
from hydrology.analysis.multisite import MultiSiteAnalyzer, quick_correlation_check
from hydrology.visualization import create_multi_plot, PlotLayout
from hydrology.visualization.plots import AVAILABLE_PLOTS
from hydrology.scripts.analyze_sites import analyze_correlation
from hydrology.core.logging_setup import get_logger
from hydrology.core.timezone import ensure_utc

logger = get_logger(__name__)
from hydrology.app.plot_config import (
    multi_plot_selector, single_plot_selector,
    CLIMATE_PLOTS, DISCHARGE_PLOTS, STAGE_PLOTS
)
from hydrology.app.styles import (
    apply_custom_css, render_site_header, render_availability_badges,
    render_metric_cards, render_footer
)

st.set_page_config(page_title="Hydrology Analysis", page_icon="💧", layout="wide")

# Apply custom styling
apply_custom_css()

# =============================================================================
# CACHED DATA FUNCTIONS
# =============================================================================

@st.cache_data(ttl=3600)
def get_inventory():
    """Load and cache inventory data."""
    return load_inventory()


@st.cache_data(ttl=3600)
def get_cached_site_info(site_id: str):
    """Get site info from inventory."""
    return get_site_info(site_id)


@st.cache_data(ttl=3600, show_spinner=False)
def get_weather_station_info(lat: float, lon: float):
    """Get nearest weather station info."""
    try:
        return fetch_nearest_station_info(lat, lon)
    except Exception:
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


@st.cache_data(ttl=3600, show_spinner=False)
def check_usgs_availability(site_id: str, param_cd: str):
    """
    Check USGS for actual data availability by sampling multiple decades.
    Returns dict with start_date, end_date, or None if not available.
    """
    try:
        # Check recent data first
        recent_end = datetime.now()
        recent_start = datetime(recent_end.year - 2, 1, 1)

        df_recent = fetch_daily_values(
            site_id, param_cd=param_cd,
            start_date=recent_start.strftime('%Y-%m-%d'),
            end_date=recent_end.strftime('%Y-%m-%d'),
            chunk_years=3
        )

        if df_recent is None or df_recent.empty:
            return None

        end_date = df_recent.index.max()

        # Check multiple decades to find the earliest data
        # Start from oldest and work forward to find first data
        decade_ranges = [
            ('1900-01-01', '1910-12-31'),
            ('1930-01-01', '1940-12-31'),
            ('1950-01-01', '1960-12-31'),
            ('1970-01-01', '1980-12-31'),
            ('1990-01-01', '2000-12-31'),
            ('2010-01-01', '2015-12-31'),
        ]

        start_date = df_recent.index.min()  # Default to recent if nothing older found

        for range_start, range_end in decade_ranges:
            try:
                df_historical = fetch_daily_values(
                    site_id, param_cd=param_cd,
                    start_date=range_start,
                    end_date=range_end,
                    chunk_years=15
                )
                if df_historical is not None and not df_historical.empty:
                    start_date = df_historical.index.min()
                    break  # Found earliest data, stop checking
            except Exception:
                continue

        return {
            'available': True,
            'start': start_date.strftime('%Y-%m-%d'),
            'end': end_date.strftime('%Y-%m-%d')
        }
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def quick_param_check(site_id: str, param_cd: str):
    """
    Quick check if a parameter has any recent data (last 2 years).
    Returns True/False for fast availability indicator.
    """
    try:
        recent_end = datetime.now()
        recent_start = datetime(recent_end.year - 2, 1, 1)

        df = fetch_daily_values(
            site_id, param_cd=param_cd,
            start_date=recent_start.strftime('%Y-%m-%d'),
            end_date=recent_end.strftime('%Y-%m-%d'),
            chunk_years=3
        )
        return df is not None and not df.empty
    except Exception:
        return False


@st.cache_data(ttl=3600, show_spinner=False)
def check_iv_availability(site_id: str, param_cd: str):
    """
    Check if instantaneous values (IV) are available for a parameter.
    Queries last 30 days of IV data from USGS.
    Returns dict with availability info or None.
    """
    import requests

    try:
        end_date = date.today()
        start_date = end_date - timedelta(days=30)

        url = "https://waterservices.usgs.gov/nwis/iv/"
        params = {
            "format": "json",
            "sites": site_id,
            "parameterCd": param_cd,
            "startDT": start_date.isoformat(),
            "endDT": end_date.isoformat(),
        }

        response = requests.get(url, params=params, timeout=15)
        if response.status_code != 200:
            return None

        json_data = response.json()

        # Check if there's actual data in the response
        time_series = json_data.get('value', {}).get('timeSeries', [])
        if not time_series:
            return None

        # Get values from first time series
        values = time_series[0].get('values', [{}])[0].get('value', [])
        if not values:
            return None

        return {
            'available': True,
            'type': 'instantaneous',
            'data_points': len(values)
        }
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_parameter_availability(site_id: str) -> dict:
    """
    Get exact availability dates for all parameters at a site using USGS series catalog.
    Returns dict mapping param_cd to {'begin_date': str, 'end_date': str, 'count': int}.
    Single API call gives accurate per-parameter dates.
    """
    import requests
    import io

    url = "https://waterservices.usgs.gov/nwis/site/"
    params = {
        "format": "rdb",
        "sites": site_id,
        "seriesCatalogOutput": "true",
        "outputDataTypeCd": "dv",  # Daily values
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        if response.status_code != 200:
            return {}

        # Parse RDB format (tab-separated with comment lines)
        lines = response.text.strip().split('\n')
        data_lines = [l for l in lines if not l.startswith('#') and l.strip()]

        if len(data_lines) < 2:
            return {}

        # First non-comment line is header, second is format spec, rest is data
        header = data_lines[0].split('\t')
        # Find column indices
        try:
            parm_idx = header.index('parm_cd')
            begin_idx = header.index('begin_date')
            end_idx = header.index('end_date')
            count_idx = header.index('count_nu')
        except ValueError:
            return {}

        result = {}
        for line in data_lines[2:]:  # Skip header and format spec
            cols = line.split('\t')
            if len(cols) > max(parm_idx, begin_idx, end_idx, count_idx):
                parm_cd = cols[parm_idx]
                begin_date = cols[begin_idx]
                end_date = cols[end_idx]
                count = cols[count_idx]

                # Store the parameter info (prefer longer record if duplicate)
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


@st.cache_data(ttl=3600, show_spinner=False)
def find_availability_windows(site_id: str, param_cd: str, check_iv: bool = False):
    """
    Find data availability for a parameter using USGS series catalog.
    Returns list of (start_year, end_year) tuples.

    Uses single API call to get exact per-parameter availability dates.
    """
    import requests
    from datetime import date

    current_year = date.today().year

    # Try to get exact availability from series catalog
    param_info = get_parameter_availability(site_id)

    if param_cd in param_info:
        info = param_info[param_cd]
        begin_date = info.get('begin_date', '')
        end_date = info.get('end_date', '')

        if begin_date:
            try:
                start_year = int(begin_date[:4])
                # Check if end_date is recent (within last year) = "present"
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

    # Fallback: check IV data if requested
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

    # Parse requested range
    req_start = pd.Timestamp(start_date)
    req_end = pd.Timestamp(end_date)
    requested_days = (req_end - req_start).days + 1

    # Actual data range
    actual_start = df.index.min()
    actual_end = df.index.max()
    actual_days = len(df)

    # Calculate coverage percentage
    coverage_pct = (actual_days / requested_days) * 100 if requested_days > 0 else 0

    # Detect gaps (periods > 30 days with no data)
    gaps = []
    if len(df) > 1:
        # Sort index and find gaps
        sorted_idx = df.index.sort_values()
        diffs = sorted_idx.to_series().diff()

        # Find gaps > 30 days
        gap_threshold = pd.Timedelta(days=30)
        large_gaps = diffs[diffs > gap_threshold]

        for gap_end, gap_size in large_gaps.items():
            gap_start = gap_end - gap_size
            gaps.append({
                'start': gap_start.strftime('%Y-%m-%d'),
                'end': gap_end.strftime('%Y-%m-%d'),
                'days': gap_size.days
            })

    # Build windows from actual data
    windows = []
    if len(df) > 0:
        sorted_idx = df.index.sort_values()
        window_start = sorted_idx[0]
        prev_date = sorted_idx[0]

        for curr_date in sorted_idx[1:]:
            gap = (curr_date - prev_date).days
            if gap > 365:  # Gap > 1 year = new window
                windows.append((window_start.year, prev_date.year))
                window_start = curr_date
            prev_date = curr_date

        # Close last window
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
    df = df[df['Discharge_cfs'] > 0]  # For log scale
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_climate_cached(lat: float, lon: float, start_str: str, end_str: str):
    """Fetch climate data - cached."""
    start_dt = datetime.strptime(start_str, '%Y-%m-%d')
    end_dt = datetime.strptime(end_str, '%Y-%m-%d')

    return fetch_climate_data(
        lat, lon,
        pd.Timestamp(start_dt),
        pd.Timestamp(end_dt),
        include_temp=True,
        include_precip=True
    )


# =============================================================================
# UI COMPONENTS
# =============================================================================

def display_site_info(site_info: dict, show_check_button: bool = True):
    """Display site information and data availability in sidebar."""
    site_id = site_info.get('site_id', 'Unknown')
    lat = site_info.get('latitude')
    lon = site_info.get('longitude')
    desc = site_info.get('description', site_id)
    begin_date = site_info.get('begin_date')
    drain_area = site_info.get('drain_area_sq_mi')

    st.sidebar.markdown(f"**{desc}**")
    if lat and lon:
        st.sidebar.text(f"Lat: {float(lat):.4f}, Lon: {float(lon):.4f}")
    if drain_area:
        st.sidebar.text(f"Drainage Area: {drain_area} sq mi")

    # Data availability section with emoji indicators
    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Availability")

    # Check if we have confirmed coverage from a previous data fetch
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
        # Get exact availability from USGS series catalog
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
        # We fetched data but no stage was found
        availability_text.append("❌ **Gage Height** (not in data)")
    else:
        # Get exact availability from USGS series catalog
        stage_dv_windows = find_availability_windows(site_id, DEFAULT_PARAM_STAGE)
        if stage_dv_windows:
            windows_str = format_availability_windows(stage_dv_windows)
            availability_text.append(f"✅ **Gage Height** ({windows_str})")
        else:
            # Try instantaneous values as fallback
            stage_iv_windows = find_availability_windows(site_id, DEFAULT_PARAM_STAGE, check_iv=True)
            if stage_iv_windows:
                windows_str = format_availability_windows(stage_iv_windows)
                availability_text.append(f"✅ **Gage Height** IV ({windows_str})")
            else:
                availability_text.append("❌ **Gage Height** (not available)")

    # Climate - based on weather station distance and data coverage
    if lat and lon:
        station = get_weather_station_info(float(lat), float(lon))
        if station:
            dist = station.get('distance_km')
            name = station.get('name', 'Unknown')[:25]  # Truncate long names
            daily_start = station.get('daily_start')
            daily_end = station.get('daily_end')

            # Build coverage string
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

    # Display all availability indicators
    for text in availability_text:
        st.sidebar.markdown(text)


def date_range_selector(key_prefix: str = "", default_start: date = None, default_end: date = None):
    """
    Date range selector with year sliders and manual date inputs.
    Returns (start_date, end_date) tuple.
    """
    if default_start is None:
        default_start = date(2015, 1, 1)
    if default_end is None:
        default_end = date.today()

    min_year = 1900
    current_year = date.today().year

    # Initialize session state for years
    start_year_key = f"{key_prefix}_start_year"
    end_year_key = f"{key_prefix}_end_year"

    if start_year_key not in st.session_state:
        st.session_state[start_year_key] = default_start.year
    if end_year_key not in st.session_state:
        st.session_state[end_year_key] = default_end.year

    # Date input keys
    start_key = f"{key_prefix}_start"
    end_key = f"{key_prefix}_end"
    today = date.today()

    # Year sliders
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.slider(
            "Start Year",
            min_value=min_year,
            max_value=current_year,
            value=st.session_state[start_year_key],
            key=f"{key_prefix}_start_slider"
        )
        # Sync date input when slider year changes
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
        # Sync date input when slider year changes
        if st.session_state[end_year_key] != end_year:
            st.session_state[end_year_key] = end_year
            st.session_state[end_key] = min(date(end_year, 12, 31), today)
        end = min(date(end_year, 12, 31), today)

    # Sanitize session state to prevent cached dates exceeding max_value
    if start_key in st.session_state and st.session_state[start_key] > today:
        st.session_state[start_key] = today
    if end_key in st.session_state and st.session_state[end_key] > today:
        st.session_state[end_key] = today

    # Fine-tune with date inputs (collapsed by default)
    with st.expander("Fine-tune dates"):
        col3, col4 = st.columns(2)
        with col3:
            start = st.date_input(
                "Start Date",
                value=start,
                min_value=date(1900, 1, 1),
                max_value=today,
                key=start_key
            )
        with col4:
            end = st.date_input(
                "End Date",
                value=end,
                min_value=date(1900, 1, 1),
                max_value=today,
                key=end_key
            )

    # Validate date order - swap if needed
    if start > end:
        st.warning("⚠️ Start date was after end date - dates have been swapped.")
        start, end = end, start
        # Update session state to reflect the swap
        st.session_state[start_key] = start
        st.session_state[end_key] = end

    # Show date range with validation
    days_diff = (end - start).days
    if days_diff < 30:
        st.caption(f"Range: {start} → {end} ({days_diff} days) ⚠️ Short range")
    else:
        st.caption(f"Range: {start} → {end} ({days_diff:,} days)")

    return start, end


def plot_selector():
    """Plot type selector. Returns list of selected plot names."""
    return multi_plot_selector(AVAILABLE_PLOTS, key_prefix="")


def single_plot_selector_widget(key_suffix: str = ""):
    """Select a single plot type for comparison mode."""
    return single_plot_selector(AVAILABLE_PLOTS, key_suffix=key_suffix)


# =============================================================================
# PROCESSING FUNCTIONS
# =============================================================================

def process_site_data(site_id: str, lat: float, lon: float, start_str: str, end_str: str):
    """
    Fetch and process data for a single site.
    Returns dict with df_q, df_merged, analysis_results.
    """
    # Fetch discharge
    df_q = fetch_discharge_data(site_id, DEFAULT_DISCHARGE_CODE, start_str, end_str)
    if df_q is None or df_q.empty:
        return None

    # Fetch gage height (stage) data and merge with df_q
    try:
        df_stage = fetch_stage_data(site_id, start_str, end_str)
        if df_stage is not None and not df_stage.empty:
            # Rename to Gage_Height_ft for consistency with plot expectations
            df_stage = df_stage.rename(columns={'Stage_ft': 'Gage_Height_ft'})
            # Merge with df_q on index
            df_q = df_q.join(df_stage[['Gage_Height_ft']], how='left')
    except Exception as e:
        pass  # Stage data is optional, continue without it

    # Fetch climate
    df_climate = fetch_climate_cached(lat, lon, start_str, end_str)

    # Merge data
    df_merged = None
    analysis_results = None

    if df_q is not None and not df_q.empty and df_climate is not None and not df_climate.empty:
        # Normalize timezones to UTC
        df_q = ensure_utc(df_q)
        df_climate = ensure_utc(df_climate)

        df_merged = pd.merge(df_q, df_climate, left_index=True, right_index=True, how='inner')

        # If merge resulted in empty DataFrame, set to None so fallback logic works
        if df_merged.empty:
            logger.warning(f"Merge produced empty result. df_q: {len(df_q)} rows, df_climate: {len(df_climate)} rows")
            df_merged = None
        else:
            analysis_results = analyze_correlation(df_merged)

    # Analyze coverage for discharge and stage
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


def create_single_plot(plot_name: str, data: dict, site_id: str, title: str, dpi: int = 150):
    """Create a single plot figure."""
    from hydrology.visualization.plots import AVAILABLE_PLOTS

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=dpi)

    plot_func = AVAILABLE_PLOTS.get(plot_name)
    if plot_func:
        plot_func(ax, **data)

    ax.set_title(f"{title}\n{plot_name}")
    fig.tight_layout()
    return fig


def create_comparison_figure(plot_name: str, data_list: list, titles: list,
                             nrows: int, ncols: int, dpi: int = 150):
    """
    Create a comparison figure with multiple subplots.

    Args:
        plot_name: Name of the plot type
        data_list: List of data dicts for each subplot
        titles: List of titles for each subplot
        nrows, ncols: Grid dimensions
        dpi: Resolution
    """
    from hydrology.visualization.plots import AVAILABLE_PLOTS

    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows), dpi=dpi, squeeze=False)

    # Get the actual plot function from the dict
    plot_info = AVAILABLE_PLOTS.get(plot_name)
    plot_func = plot_info['function'] if plot_info else None

    for idx, (data, title) in enumerate(zip(data_list, titles)):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]

        if plot_func and data is not None:
            try:
                # Check data availability in detail
                plot_requires = plot_info.get('requires', [])
                df_q = data.get('df_q')
                df_merged = data.get('df_merged')

                has_discharge = df_q is not None and not df_q.empty
                has_merged = df_merged is not None and not df_merged.empty
                has_gage = has_discharge and 'Gage_Height_ft' in df_q.columns and df_q['Gage_Height_ft'].notna().any()
                has_climate = has_merged and 'Precip_mm' in df_merged.columns

                # Build detailed availability message
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
                    # Let plot functions handle missing data gracefully with fallbacks
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

    # Hide unused subplots
    total_plots = len(data_list)
    for idx in range(total_plots, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].set_visible(False)

    fig.suptitle(f"Comparison: {plot_name}", fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig


# =============================================================================
# MAIN MODES
# =============================================================================

def single_analysis_mode(inventory_df):
    """Standard single site analysis mode."""
    # === SIDEBAR: Site Selection Only ===
    st.sidebar.header("Select Site")

    site_options = [f"{row['site_id']} - {str(row.get('description', ''))[:40]}"
                    for _, row in inventory_df.iterrows()]
    selected = st.sidebar.selectbox("Select Site", site_options, key="single_site")
    site_id = extract_site_id(selected)

    site_info = get_cached_site_info(site_id)
    if not site_info:
        st.error(f"Site {site_id} not found")
        return

    lat = site_info.get('latitude')
    lon = site_info.get('longitude')
    desc = site_info.get('description', site_id)

    display_site_info(site_info)

    # === MAIN AREA: Configuration & Results ===
    st.header("Single Site Analysis")
    st.caption(f"Site: {desc}")

    # Date range section
    st.subheader("Date Range")
    start_date, end_date = date_range_selector("single")

    st.markdown("---")

    # Plot selection section
    st.subheader("Plot Selection")
    selected_plots = plot_selector()

    st.markdown("---")

    # Layout and Generate row
    col_layout, col_dpi, col_btn = st.columns([2, 1, 2])

    layout_options = {
        'Auto': PlotLayout.AUTO,
        'Vertical': PlotLayout.VERTICAL,
        'Quad (2x2)': PlotLayout.QUAD,
        'Grid 2x3': PlotLayout.GRID_2x3,
        'Grid 3x2': PlotLayout.GRID_3x2,
        'Grid 2x5': PlotLayout.GRID_2x5,
    }

    with col_layout:
        layout = st.selectbox("Layout", list(layout_options.keys()))

    with col_dpi:
        dpi = st.number_input("DPI", min_value=72, max_value=300, value=150)

    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)  # Align button with inputs
        generate = st.button("🔍 Generate Plots", type="primary", width='stretch')

    st.markdown("---")

    # Generate plots
    if generate:
        if not selected_plots:
            st.warning("Select at least one plot type")
            return

        if not lat or not lon:
            st.error("Site missing coordinates")
            return

        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        with st.spinner("Processing..."):
            data = process_site_data(site_id, float(lat), float(lon), start_str, end_str)

        if data is None:
            st.error("No discharge data available")
            return

        # Store confirmed coverage in session state for sidebar update
        confirmed_key = f"confirmed_coverage_{site_id}"
        st.session_state[confirmed_key] = {
            'discharge': data.get('discharge_coverage'),
            'stage': data.get('stage_coverage')
        }

        # Render site header in main area
        render_site_header(site_id, desc, float(lat) if lat else None, float(lon) if lon else None)

        # Show data availability badges
        has_stage = 'Gage_Height_ft' in data['df_q'].columns if data['df_q'] is not None else False
        climate_info = get_weather_station_info(float(lat), float(lon)) if lat and lon else None
        render_availability_badges(True, has_stage, climate_info)

        # Show metric cards
        render_metric_cards(data['df_q'], data['df_merged'])

        st.markdown('---')

        plot_data = {
            'df_q': data['df_q'],
            'df_merged': data['df_merged'],
            'analysis_results': data['analysis_results']
        }

        with st.spinner("Generating plots..."):
            fig = create_multi_plot(
                plots=selected_plots,
                layout=layout_options[layout],
                data=plot_data,
                site_id=site_id,
                title=desc,
                dpi=dpi
            )

        if fig:
            st.pyplot(fig)
            render_export_buttons(fig, site_id, dpi)
            plt.close(fig)
    else:
        st.info("👈 Select plots above, then click 'Generate Plots'")


def compare_time_periods_mode(inventory_df):
    """Compare same site across two equal-length time periods."""
    # === SIDEBAR: Site Selection Only ===
    st.sidebar.header("Select Site")

    site_options = [f"{row['site_id']} - {str(row.get('description', ''))[:40]}"
                    for _, row in inventory_df.iterrows()]
    selected = st.sidebar.selectbox("Choose site", site_options, key="compare_time_site")
    site_id = extract_site_id(selected)

    site_info = get_cached_site_info(site_id)
    if not site_info:
        st.error(f"Site {site_id} not found")
        return

    lat = site_info.get('latitude')
    lon = site_info.get('longitude')
    desc = site_info.get('description', site_id)

    display_site_info(site_info, show_check_button=False)

    # === MAIN AREA: Configuration & Results ===
    st.header("Compare Time Periods")
    st.caption(f"Site: {desc}")

    # Helper functions for water year
    def get_water_year_start(d: date) -> date:
        if d.month >= 10:
            return date(d.year, 10, 1)
        else:
            return date(d.year - 1, 10, 1)

    def get_water_year_end(start: date) -> date:
        return date(start.year + 1, 9, 30)

    # Period length and dates in main area
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Period Length")
        period_lengths = {
            "1 Year": 365,
            "2 Years": 730,
            "5 Years": 1825,
            "10 Years": 3650,
            "Water Year": "water_year",
        }
        period_choice = st.selectbox("Duration", list(period_lengths.keys()), key="period_length")
        is_water_year = period_choice == "Water Year"
        period_days = 365 if is_water_year else period_lengths[period_choice]
        if is_water_year:
            st.caption("Oct 1 → Sep 30")
        else:
            st.caption(f"{period_days} days")

    # Initialize session state for years
    if 'period_a_year' not in st.session_state:
        st.session_state.period_a_year = 2010
    if 'period_b_year' not in st.session_state:
        st.session_state.period_b_year = 2020

    current_year = date.today().year
    today = date.today()

    # Sanitize cached date inputs that exceed today
    if 'start_a_input' in st.session_state and st.session_state.start_a_input > today:
        st.session_state.start_a_input = today
    if 'start_b_input' in st.session_state and st.session_state.start_b_input > today:
        st.session_state.start_b_input = today

    with col2:
        st.subheader("Period A")
        # Year slider
        year_a = st.slider(
            "Year",
            min_value=1900,
            max_value=current_year,
            value=st.session_state.period_a_year,
            key="slider_a",
            label_visibility="collapsed"
        )

        # Compute start date from year
        if is_water_year:
            start_a = date(year_a, 10, 1)
        else:
            start_a = date(year_a, 1, 1)

        # Sync date input when slider year changes
        if st.session_state.period_a_year != year_a:
            st.session_state.period_a_year = year_a
            st.session_state.start_a_input = start_a

        # Show date input for fine-tuning (optional)
        with st.expander("Fine-tune date"):
            start_a = st.date_input(
                "Start date",
                value=start_a,
                min_value=date(1900, 1, 1),
                max_value=today,
                key="start_a_input"
            )
            if is_water_year:
                start_a = get_water_year_start(start_a)

        end_a = get_water_year_end(start_a) if is_water_year else start_a + timedelta(days=period_days)
        st.caption(f"{start_a} → {end_a}")

    with col3:
        st.subheader("Period B")
        # Year slider
        year_b = st.slider(
            "Year",
            min_value=1900,
            max_value=current_year,
            value=st.session_state.period_b_year,
            key="slider_b",
            label_visibility="collapsed"
        )

        # Compute start date from year
        if is_water_year:
            start_b = date(year_b, 10, 1)
        else:
            start_b = date(year_b, 1, 1)

        # Sync date input when slider year changes
        if st.session_state.period_b_year != year_b:
            st.session_state.period_b_year = year_b
            st.session_state.start_b_input = start_b

        # Show date input for fine-tuning (optional)
        with st.expander("Fine-tune date"):
            start_b = st.date_input(
                "Start date",
                value=start_b,
                min_value=date(1900, 1, 1),
                max_value=today,
                key="start_b_input"
            )
            if is_water_year:
                start_b = get_water_year_start(start_b)

        end_b = get_water_year_end(start_b) if is_water_year else start_b + timedelta(days=period_days)
        st.caption(f"{start_b} → {end_b}")

    # Plot selection row
    col_plot, col_dpi, col_btn = st.columns([3, 1, 2])

    with col_plot:
        st.subheader("Plot Type")
        plot_name = single_plot_selector_widget("_compare")

    with col_dpi:
        st.subheader("Quality")
        dpi = st.selectbox("DPI", [100, 150, 200], index=1, key="compare_time_dpi", label_visibility="collapsed")

    with col_btn:
        st.subheader(" ")
        generate = st.button("🔍 Compare Periods", type="primary", width='stretch')

    # Generate comparison
    if generate:
        if not lat or not lon:
            st.error("Site missing coordinates")
            return

        # Render styled site header
        render_site_header(site_id, desc, float(lat) if lat else None, float(lon) if lon else None)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"Period A: {start_a} to {end_a}")
            with st.spinner("Processing Period A..."):
                data_a = process_site_data(site_id, float(lat), float(lon),
                                           start_a.strftime('%Y-%m-%d'), end_a.strftime('%Y-%m-%d'))
            if data_a:
                st.caption(f"Discharge: {data_a['discharge_count']:,} | Merged: {data_a['merged_count']:,}")

        with col2:
            st.subheader(f"Period B: {start_b} to {end_b}")
            with st.spinner("Processing Period B..."):
                data_b = process_site_data(site_id, float(lat), float(lon),
                                           start_b.strftime('%Y-%m-%d'), end_b.strftime('%Y-%m-%d'))
            if data_b:
                st.caption(f"Discharge: {data_b['discharge_count']:,} | Merged: {data_b['merged_count']:,}")

        # Create comparison figure - always show both periods
        data_list = []
        titles = []

        # Period A - always add (None if no data)
        if data_a:
            data_list.append({'df_q': data_a['df_q'], 'df_merged': data_a['df_merged'],
                             'analysis_results': data_a['analysis_results']})
            titles.append(f"{desc[:25]}\n{start_a} to {end_a}\n({data_a['discharge_count']:,} Q, {data_a['merged_count']:,} merged)")
        else:
            data_list.append(None)
            st.warning(f"No data available for Period A ({start_a} to {end_a})")
            titles.append(f"{desc[:25]}\n{start_a} to {end_a}\n(No data)")

        # Period B - always add (None if no data)
        if data_b:
            data_list.append({'df_q': data_b['df_q'], 'df_merged': data_b['df_merged'],
                             'analysis_results': data_b['analysis_results']})
            titles.append(f"{desc[:25]}\n{start_b} to {end_b}\n({data_b['discharge_count']:,} Q, {data_b['merged_count']:,} merged)")
        else:
            data_list.append(None)
            st.warning(f"No data available for Period B ({start_b} to {end_b})")
            titles.append(f"{desc[:25]}\n{start_b} to {end_b}\n(No data)")

        # Always create 2-column comparison (even with missing data)
        if data_a or data_b:  # At least one period has data
            fig = create_comparison_figure(plot_name, data_list, titles, 1, 2, dpi)
            st.pyplot(fig)
            render_export_buttons(fig, f"{site_id}_comparison", dpi)
            plt.close(fig)
        else:
            st.error("No data available for either period")
    else:
        st.info("Select a site, period length, and start dates to compare equal-length periods")


def compare_sites_mode(inventory_df):
    """Compare multiple sites for the same time period."""
    # === SIDEBAR: Site Selection Only ===
    st.sidebar.header("Select Sites")

    site_options = [f"{row['site_id']} - {str(row.get('description', ''))[:40]}"
                    for _, row in inventory_df.iterrows()]
    selected_sites = st.sidebar.multiselect(
        "Choose 2-4 sites",
        site_options,
        max_selections=4,
        key="compare_sites_multi"
    )

    # Show selected site info and data availability in sidebar
    if selected_sites:
        st.sidebar.markdown("---")
        for site_str in selected_sites:
            site_id = extract_site_id(site_str)
            site_info = get_cached_site_info(site_id)
            if site_info:
                display_site_info(site_info, show_check_button=False)

    # === MAIN AREA: Configuration & Results ===
    st.header("Compare Sites")

    if len(selected_sites) < 2:
        st.info("👈 Select 2-4 sites from the sidebar to compare")
        return

    # Date range using year sliders (like single analysis mode)
    st.subheader("Date Range")
    start_date, end_date = date_range_selector("compare_sites", default_start=date(2015, 1, 1))

    st.markdown("---")

    # Configuration in main area with columns
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        pass  # Date range moved above

    with col2:
        st.subheader("Plot Type")
        plot_name = single_plot_selector_widget("_sites")

    with col3:
        st.subheader("Options")
        dpi = st.selectbox("Quality", [100, 150, 200], index=1, key="cs_dpi")

    # Generate button in main area
    st.markdown("---")
    if st.button("🔍 Compare Sites", type="primary", width='stretch'):
        # Show comparison header (using safe Streamlit components)
        st.header("Multi-Site Comparison")
        st.caption(f"{len(selected_sites)} sites | {start_date} to {end_date}")

        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        data_list = []
        titles = []

        progress = st.progress(0, text="Loading sites...")
        for i, site_str in enumerate(selected_sites):
            site_id = extract_site_id(site_str)
            site_info = get_cached_site_info(site_id)
            desc = site_info.get('description', site_id) if site_info else site_id

            if not site_info:
                data_list.append(None)
                titles.append(f"{desc[:30]}\n(Site not found)")
                continue

            lat = site_info.get('latitude')
            lon = site_info.get('longitude')

            if not lat or not lon:
                data_list.append(None)
                titles.append(f"{desc[:30]}\n(No coordinates)")
                continue

            progress.progress((i + 1) / len(selected_sites), text=f"Processing {site_id}...")
            data = process_site_data(site_id, float(lat), float(lon), start_str, end_str)

            if data:
                data_list.append({
                    'df_q': data['df_q'],
                    'df_merged': data['df_merged'],
                    'analysis_results': data['analysis_results']
                })
                # Include date range and data count in title
                q_count = data['discharge_count']
                m_count = data['merged_count']
                titles.append(f"{desc[:30]}\n{start_str} to {end_str} ({q_count:,} Q, {m_count:,} merged)")
            else:
                data_list.append(None)
                titles.append(f"{desc[:30]}\n{start_str} to {end_str} (No data)")

        progress.empty()

        # Always show grid for all selected sites
        n = len(selected_sites)
        if n <= 2:
            nrows, ncols = 1, n
        elif n <= 4:
            nrows, ncols = 2, 2
        else:
            nrows, ncols = 2, 3

        if any(d is not None for d in data_list):
            fig = create_comparison_figure(plot_name, data_list, titles, nrows, ncols, dpi)
            st.pyplot(fig)
            render_export_buttons(fig, "multi_site_comparison", dpi)
            plt.close(fig)
        else:
            st.error("No data available for any of the selected sites")


def quad_comparison_mode(inventory_df):
    """2x2 comparison: 2 sites x 2 equal-length time periods."""
    # === SIDEBAR: Site Selection Only ===
    st.sidebar.header("Select Sites")

    site_options = [f"{row['site_id']} - {str(row.get('description', ''))[:40]}"
                    for _, row in inventory_df.iterrows()]

    site_a = st.sidebar.selectbox("Site A", site_options, key="quad_site_a")
    site_b = st.sidebar.selectbox("Site B", site_options, key="quad_site_b", index=min(1, len(site_options)-1))

    # Show selected site info and data availability
    st.sidebar.markdown("---")
    for label, site_str in [("Site A", site_a), ("Site B", site_b)]:
        site_id = extract_site_id(site_str)
        site_info = get_cached_site_info(site_id)
        if site_info:
            st.sidebar.caption(f"**{label}:**")
            display_site_info(site_info, show_check_button=False)

    # === MAIN AREA: Configuration & Results ===
    st.header("2×2 Comparison")
    st.caption("Compare 2 sites across 2 time periods")

    # Period length and dates
    col1, col2, col3 = st.columns(3)

    today = date.today()

    # Sanitize cached dates that exceed today
    if 'quad_start_1' in st.session_state and st.session_state.quad_start_1 > today:
        st.session_state.quad_start_1 = today
    if 'quad_start_2' in st.session_state and st.session_state.quad_start_2 > today:
        st.session_state.quad_start_2 = today

    with col1:
        st.subheader("Period Length")
        period_lengths = {
            "1 Year": 365,
            "2 Years": 730,
            "5 Years": 1825,
            "10 Years": 3650,
        }
        period_choice = st.selectbox("Duration", list(period_lengths.keys()), key="quad_period_length")
        period_days = period_lengths[period_choice]
        st.caption(f"{period_days} days")

    with col2:
        st.subheader("Period 1")
        if 'quad_p1_start' not in st.session_state:
            st.session_state.quad_p1_start = date(2010, 1, 1)

        col_p1a, col_p1b, col_p1c = st.columns([1, 3, 1])
        with col_p1a:
            if st.button("◀", key="shift_p1_back"):
                st.session_state.quad_p1_start = st.session_state.quad_p1_start - timedelta(days=period_days)
        with col_p1b:
            start_1 = st.date_input("Start", st.session_state.quad_p1_start, key="quad_start_1",
                                    min_value=date(1900, 1, 1), max_value=today, label_visibility="collapsed")
            st.session_state.quad_p1_start = start_1
        with col_p1c:
            if st.button("▶", key="shift_p1_fwd"):
                st.session_state.quad_p1_start = st.session_state.quad_p1_start + timedelta(days=period_days)

        end_1 = start_1 + timedelta(days=period_days)
        st.caption(f"→ {end_1}")

    with col3:
        st.subheader("Period 2")
        if 'quad_p2_start' not in st.session_state:
            st.session_state.quad_p2_start = date(2020, 1, 1)

        col_p2a, col_p2b, col_p2c = st.columns([1, 3, 1])
        with col_p2a:
            if st.button("◀", key="shift_p2_back"):
                st.session_state.quad_p2_start = st.session_state.quad_p2_start - timedelta(days=period_days)
        with col_p2b:
            start_2 = st.date_input("Start", st.session_state.quad_p2_start, key="quad_start_2",
                                    min_value=date(1900, 1, 1), max_value=today, label_visibility="collapsed")
            st.session_state.quad_p2_start = start_2
        with col_p2c:
            if st.button("▶", key="shift_p2_fwd"):
                st.session_state.quad_p2_start = st.session_state.quad_p2_start + timedelta(days=period_days)

        end_2 = start_2 + timedelta(days=period_days)
        st.caption(f"→ {end_2}")

    # Plot selection row
    col_plot, col_dpi, col_btn = st.columns([3, 1, 2])

    with col_plot:
        st.subheader("Plot Type")
        plot_name = single_plot_selector_widget("_quad")

    with col_dpi:
        st.subheader("Quality")
        dpi = st.selectbox("DPI", [100, 150, 200], index=1, key="quad_dpi", label_visibility="collapsed")

    with col_btn:
        st.subheader(" ")  # Spacer
        generate = st.button("🔍 Generate 2×2", type="primary", width='stretch')

    # Generate comparison
    if generate:
        site_id_a = extract_site_id(site_a)
        site_id_b = extract_site_id(site_b)

        site_info_a = get_cached_site_info(site_id_a)
        site_info_b = get_cached_site_info(site_id_b)

        if not site_info_a or not site_info_b:
            st.error("Could not load site info")
            return

        st.markdown("---")

        # Grid layout:
        #         Period 1    Period 2
        # Site A  [0,0]       [0,1]
        # Site B  [1,0]       [1,1]

        configs = [
            (site_id_a, site_info_a, start_1, end_1, f"{site_info_a['description'][:20]}\n{start_1} to {end_1}"),
            (site_id_a, site_info_a, start_2, end_2, f"{site_info_a['description'][:20]}\n{start_2} to {end_2}"),
            (site_id_b, site_info_b, start_1, end_1, f"{site_info_b['description'][:20]}\n{start_1} to {end_1}"),
            (site_id_b, site_info_b, start_2, end_2, f"{site_info_b['description'][:20]}\n{start_2} to {end_2}"),
        ]

        data_list = []
        titles = []

        progress = st.progress(0, text="Loading data...")
        for i, (site_id, site_info, start_d, end_d, title) in enumerate(configs):
            lat = site_info.get('latitude')
            lon = site_info.get('longitude')

            if not lat or not lon:
                data_list.append(None)
                titles.append(title + "\n(No coords)")
                continue

            progress.progress((i + 1) / 4, text=f"Processing {site_id}...")
            data = process_site_data(site_id, float(lat), float(lon),
                                    start_d.strftime('%Y-%m-%d'), end_d.strftime('%Y-%m-%d'))

            if data:
                data_list.append({
                    'df_q': data['df_q'],
                    'df_merged': data['df_merged'],
                    'analysis_results': data['analysis_results']
                })
                # Add data counts to title
                q_count = data['discharge_count']
                m_count = data['merged_count']
                titles.append(f"{title}\n({q_count:,} Q, {m_count:,} merged)")
            else:
                data_list.append(None)
                titles.append(title + "\n(No data)")

        progress.empty()

        fig = create_comparison_figure(plot_name, data_list, titles, 2, 2, dpi)

        # Add row/col labels
        fig.text(0.02, 0.75, 'Site A', rotation=90, fontsize=12, fontweight='bold', va='center')
        fig.text(0.02, 0.25, 'Site B', rotation=90, fontsize=12, fontweight='bold', va='center')
        fig.text(0.3, 0.98, 'Period 1', fontsize=12, fontweight='bold', ha='center')
        fig.text(0.7, 0.98, 'Period 2', fontsize=12, fontweight='bold', ha='center')

        st.pyplot(fig)
        render_export_buttons(fig, "2x2_comparison", dpi)
        plt.close(fig)


def render_export_buttons(fig, filename_base: str, dpi: int):
    """Render export buttons for a figure."""
    st.subheader("Export")
    col1, col2 = st.columns(2)

    # Light background version for print/documents
    buf_png = io.BytesIO()
    fig.savefig(buf_png, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    buf_png.seek(0)
    col1.download_button("PNG (light)", buf_png, f"{filename_base}.png", "image/png")

    buf_pdf = io.BytesIO()
    fig.savefig(buf_pdf, format='pdf', dpi=dpi, bbox_inches='tight', facecolor='white')
    buf_pdf.seek(0)
    col2.download_button("PDF (light)", buf_pdf, f"{filename_base}.pdf", "application/pdf")


# =============================================================================
# SITE MAP MODE
# =============================================================================

def site_map_mode(inventory_df):
    """Display all sites on an interactive Folium map with tooltips and HUC boundaries."""
    st.header("Site Map")

    # Prepare map data
    map_data = inventory_df[['latitude', 'longitude', 'site_id', 'description', 'begin_date']].copy()
    map_data = map_data.dropna(subset=['latitude', 'longitude'])
    map_data['latitude'] = map_data['latitude'].astype(float)
    map_data['longitude'] = map_data['longitude'].astype(float)

    # Map options in sidebar
    st.sidebar.header("Map Options")

    # Quick jump to watershed - with bounding boxes for filtering
    # bounds = [min_lat, max_lat, min_lon, max_lon]
    watersheds = {
        "All Sites (Pacific NW)": {"center": [46.5, -120.5], "zoom": 6, "bounds": None},
        "─── Washington ───": None,
        "Puget Sound": {"center": [47.5, -122.3], "zoom": 8, "bounds": [46.8, 49.0, -124.0, -121.0]},
        "Upper Columbia": {"center": [48.0, -118.0], "zoom": 7, "bounds": [47.0, 49.5, -121.0, -115.0]},
        "Spokane River": {"center": [47.7, -117.4], "zoom": 9, "bounds": [47.3, 48.2, -118.5, -116.5]},
        "Yakima": {"center": [46.6, -120.5], "zoom": 8, "bounds": [45.8, 47.4, -121.8, -119.0]},
        "Lower Columbia": {"center": [46.2, -123.0], "zoom": 8, "bounds": [45.5, 47.0, -124.5, -121.5]},
        "─── Oregon ───": None,
        "Willamette": {"center": [44.5, -122.8], "zoom": 8, "bounds": [43.0, 46.0, -124.0, -121.5]},
        "Oregon Coast": {"center": [44.0, -124.0], "zoom": 8, "bounds": [42.0, 46.5, -125.0, -123.0]},
        "Deschutes": {"center": [44.0, -121.2], "zoom": 8, "bounds": [43.0, 45.5, -122.5, -120.0]},
        "─── Idaho ───": None,
        "Snake River": {"center": [43.5, -115.5], "zoom": 7, "bounds": [41.5, 45.5, -118.0, -111.0]},
        "Clearwater": {"center": [46.5, -115.5], "zoom": 8, "bounds": [45.5, 47.5, -117.0, -114.0]},
        "Salmon River": {"center": [45.0, -114.5], "zoom": 8, "bounds": [44.0, 46.0, -116.5, -113.0]},
    }

    selected_watershed = st.sidebar.selectbox(
        "Jump to Watershed",
        list(watersheds.keys()),
        index=0,
        help="Quick zoom to a specific watershed and filter site list"
    )

    # Get map center and zoom from selection
    ws_config = watersheds.get(selected_watershed)
    if ws_config:
        center_lat, center_lon = ws_config["center"]
        zoom_start = ws_config["zoom"]
        ws_bounds = ws_config.get("bounds")
    else:
        # Separator selected, use default
        center_lat = map_data['latitude'].mean()
        center_lon = map_data['longitude'].mean()
        zoom_start = 6
        ws_bounds = None

    # Filter sites by watershed bounds
    if ws_bounds:
        min_lat, max_lat, min_lon, max_lon = ws_bounds
        filtered_sites = map_data[
            (map_data['latitude'] >= min_lat) & (map_data['latitude'] <= max_lat) &
            (map_data['longitude'] >= min_lon) & (map_data['longitude'] <= max_lon)
        ].copy()
    else:
        filtered_sites = map_data.copy()

    # Site selector to zoom to specific gage
    if not filtered_sites.empty:
        site_options = ["(Select a site to zoom)"] + [
            f"{row['site_id']} - {str(row['description'])[:40]}"
            for _, row in filtered_sites.iterrows()
        ]
        selected_site = st.sidebar.selectbox(
            f"Sites in {selected_watershed.replace('─', '').strip() if '─' not in selected_watershed else 'Region'} ({len(filtered_sites)})",
            site_options,
            key="site_zoom_select"
        )

        # If a site is selected from dropdown, override center/zoom
        if selected_site != "(Select a site to zoom)":
            site_id = selected_site.split(" - ")[0]
            site_row = filtered_sites[filtered_sites['site_id'] == site_id]
            if not site_row.empty:
                center_lat = site_row.iloc[0]['latitude']
                center_lon = site_row.iloc[0]['longitude']
                zoom_start = 12  # Zoom in close to the site

    # Check if a site was selected from the table (overrides dropdown)
    if 'table_selected_site' in st.session_state:
        table_site = st.session_state['table_selected_site']
        center_lat = table_site['lat']
        center_lon = table_site['lon']
        zoom_start = 12
        st.sidebar.success(f"📍 {table_site['site_id']}")
        st.sidebar.caption(table_site['description'][:50])
        if st.sidebar.button("Clear Selection", key="clear_table_selection"):
            del st.session_state['table_selected_site']
            st.rerun()

    st.sidebar.markdown("---")

    show_huc = st.sidebar.checkbox("Show Watershed Boundaries", value=True)

    # User-friendly watershed boundary options
    huc_options = {
        "Major Regions (HUC2)": 2,
        "Subregions (HUC4)": 4,
        "Basins (HUC6)": 6,
        "Subbasins - Detailed (HUC8)": 8
    }
    huc_choice = st.sidebar.selectbox(
        "Boundary Detail",
        list(huc_options.keys()),
        index=1,  # Default to Subregions
        help="Choose how detailed the watershed boundaries should be"
    )
    huc_level = huc_options[huc_choice]

    use_clustering = st.sidebar.checkbox("Cluster Nearby Sites", value=True,
                                         help="Group nearby sites into clusters")

    # Create Folium map with dark tiles
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles='CartoDB dark_matter',
        control_scale=True
    )

    # Add HUC watershed boundaries via WMS (static image, no hover effects)
    if show_huc:
        # USGS National Map WMS for Watershed Boundary Dataset
        wms_url = "https://hydro.nationalmap.gov/arcgis/services/wbd/MapServer/WMSServer"

        # HUC layer mapping (WMS layer indices)
        huc_layers = {2: "1", 4: "2", 6: "3", 8: "4"}

        folium.raster_layers.WmsTileLayer(
            url=wms_url,
            name=f"Watershed Boundaries",
            layers=huc_layers.get(huc_level, "2"),
            fmt="image/png",
            transparent=True,
            opacity=0.4,
            overlay=True,
            control=True,
            version="1.3.0"
        ).add_to(m)

    # Create marker group (clustered or regular)
    if use_clustering:
        marker_group = MarkerCluster(name="USGS Sites")
    else:
        marker_group = folium.FeatureGroup(name="USGS Sites")

    # Add site markers with tooltips and popups
    for _, row in map_data.iterrows():
        site_id = row['site_id']
        desc = str(row.get('description', ''))[:50]
        lat = row['latitude']
        lon = row['longitude']
        begin = row.get('begin_date', 'Unknown')

        # Tooltip (shown on hover)
        tooltip = f"<b>{site_id}</b><br>{desc}"

        # Popup (shown on click) with more detail
        popup_html = f"""
        <div style="width:200px">
            <b>{site_id}</b><br>
            <span style="font-size:11px">{desc}</span><br>
            <hr style="margin:5px 0">
            <b>Coords:</b> {lat:.4f}, {lon:.4f}<br>
            <b>Record Start:</b> {begin}<br>
        </div>
        """

        # Color based on record length (older = more blue, newer = more orange)
        try:
            start_year = int(str(begin)[:4]) if begin and begin != 'Unknown' else 2020
            if start_year < 1950:
                color = 'darkblue'
            elif start_year < 1980:
                color = 'blue'
            elif start_year < 2000:
                color = 'cadetblue'
            elif start_year < 2010:
                color = 'orange'
            else:
                color = 'red'
        except (ValueError, TypeError):
            color = 'gray'

        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=tooltip,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.8,
            weight=1
        ).add_to(marker_group)

    marker_group.add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Display info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Sites", len(map_data))
    with col2:
        # Find oldest discharge record
        oldest_display = 'N/A'
        try:
            if 'begin_date' in map_data.columns:
                # Get dates, convert to string, filter to only valid dates (start with digit)
                dates = map_data['begin_date'].dropna().astype(str)
                # Filter: must start with a digit (valid date like "1929-05-10")
                valid_dates = dates[dates.str.match(r'^\d')]
                if len(valid_dates) > 0:
                    # Sort and get oldest (earliest date)
                    sorted_dates = sorted(valid_dates.tolist())
                    oldest_display = sorted_dates[0][:4]  # Just the year
        except Exception as e:
            oldest_display = 'Error'
        st.metric("Oldest Record", oldest_display)
    with col3:
        st.metric("Region", "Pacific Northwest")

    # Color legend
    st.caption("🔵 Pre-1950 | 🔷 1950-1979 | 🔹 1980-1999 | 🟠 2000-2009 | 🔴 2010+")

    # Render the Folium map
    map_output = st_folium(m, width=None, height=600, returned_objects=["last_object_clicked"])

    # Handle click events - show site info
    if map_output and map_output.get("last_object_clicked"):
        clicked = map_output["last_object_clicked"]
        clicked_lat = clicked.get("lat")
        clicked_lng = clicked.get("lng")

        if clicked_lat and clicked_lng:
            # Find the clicked site (within small tolerance)
            tolerance = 0.01
            matches = map_data[
                (abs(map_data['latitude'] - clicked_lat) < tolerance) &
                (abs(map_data['longitude'] - clicked_lng) < tolerance)
            ]

            if not matches.empty:
                site = matches.iloc[0]
                st.success(f"Selected: **{site['site_id']}** - {site['description']}")
                st.caption("Switch to 'Single Analysis' mode to analyze this site")

    # Collapsible site list - shows filtered sites based on watershed selection
    sites_label = f"View Site List ({len(filtered_sites)} sites)"
    if ws_bounds:
        sites_label = f"Sites in {selected_watershed} ({len(filtered_sites)})"

    with st.expander(sites_label, expanded=False):
        if filtered_sites.empty:
            st.info("No sites found in this watershed area")
        else:
            display_df = filtered_sites[['site_id', 'description', 'latitude', 'longitude', 'begin_date']].copy()
            display_df = display_df.reset_index(drop=True)

            st.caption("👆 Click a row to zoom to that site")

            # Use dataframe with row selection
            selection = st.dataframe(
                display_df,
                width='stretch',
                hide_index=True,
                selection_mode="single-row",
                on_select="rerun"
            )

            # Handle row selection - store in session state (on_select="rerun" handles the rerun)
            if selection and selection.selection and selection.selection.rows:
                selected_row_idx = selection.selection.rows[0]
                if selected_row_idx < len(display_df):
                    selected_row = display_df.iloc[selected_row_idx]
                    new_site_id = selected_row['site_id']
                    # Only update if it's a different site (prevents rerun loop)
                    current = st.session_state.get('table_selected_site', {})
                    if current.get('site_id') != new_site_id:
                        st.session_state['table_selected_site'] = {
                            'site_id': new_site_id,
                            'lat': selected_row['latitude'],
                            'lon': selected_row['longitude'],
                            'description': selected_row['description']
                        }


# =============================================================================
# ALERT MONITOR MODE
# =============================================================================

def alert_monitor_mode(inventory_df):
    """Real-time alert monitoring for threshold exceedances."""
    st.sidebar.header("Alert Configuration")

    # Site selection
    site_options = [f"{row['site_id']} - {str(row.get('description', ''))[:40]}"
                    for _, row in inventory_df.iterrows()]
    selected = st.sidebar.selectbox("Monitor Site", site_options, key="alert_site")
    site_id = extract_site_id(selected)

    site_info = get_cached_site_info(site_id)
    if not site_info:
        st.error(f"Site {site_id} not found")
        return

    desc = site_info.get('description', site_id)

    # Main content
    st.header("🚨 Alert Monitor")
    st.caption(f"Real-time threshold monitoring for {desc}")

    # Alert configuration
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Flood Alerts")
        flood_enabled = st.checkbox("Enable flood alerts", value=True)
        if flood_enabled:
            action_stage = st.number_input("Action Stage (ft)", value=10.0, step=0.5,
                                          help="Minor flooding threshold")
            flood_stage = st.number_input("Flood Stage (ft)", value=12.0, step=0.5,
                                         help="Moderate flooding threshold")
            major_flood = st.number_input("Major Flood Stage (ft)", value=15.0, step=0.5,
                                         help="Major flooding threshold")

    with col2:
        st.subheader("Low Flow Alerts")
        low_flow_enabled = st.checkbox("Enable low flow alerts", value=False)
        if low_flow_enabled:
            low_flow_threshold = st.number_input("Low Flow (cfs)", value=100.0, step=10.0,
                                                help="Low flow warning threshold")
            critical_flow = st.number_input("Critical Flow (cfs)", value=50.0, step=10.0,
                                           help="Critical low flow threshold")

    st.markdown("---")

    # Check current conditions
    if st.button("🔍 Check Current Conditions", type="primary"):
        with st.spinner("Fetching current data..."):
            # Create alert monitor with configured thresholds
            monitor = AlertMonitor()

            if flood_enabled:
                thresholds = create_flood_alert(site_id, flood_stage, action_stage, major_flood)
                for t in thresholds:
                    monitor.add_threshold(t)

            if low_flow_enabled:
                thresholds = create_low_flow_alert(site_id, low_flow_threshold, critical_flow)
                for t in thresholds:
                    monitor.add_threshold(t)

            # Check for alerts
            alerts = monitor.check_site(site_id, use_instantaneous=True)

            # Display current conditions
            st.subheader("Current Conditions")

            # Try to get latest reading
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=1)

                df_instant = fetch_instantaneous_values(
                    site_id, param_cd=DEFAULT_PARAM_DISCHARGE,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                )

                if df_instant is not None and not df_instant.empty:
                    latest = df_instant.iloc[-1]
                    latest_time = df_instant.index[-1]

                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Latest Discharge", f"{latest['value']:.1f} cfs")
                    with col_b:
                        st.metric("Reading Time", latest_time.strftime('%Y-%m-%d %H:%M'))
                    with col_c:
                        if alerts:
                            st.metric("Active Alerts", len(alerts))
                        else:
                            st.metric("Status", "✅ Normal")
                else:
                    st.warning("Could not fetch current instantaneous data")

            except Exception as e:
                st.error(f"Error fetching data: {e}")

            # Display any alerts
            if alerts:
                st.subheader("⚠️ Active Alerts")
                for alert in alerts:
                    severity_colors = {
                        'critical': '🔴',
                        'warning': '🟡',
                        'info': '🔵'
                    }
                    icon = severity_colors.get(alert.severity, '⚪')
                    st.error(f"{icon} **{alert.severity.upper()}**: {alert.message}")
            else:
                st.success("✅ No alerts - all conditions normal")

    # Show recent history with context
    st.markdown("---")
    st.subheader("Recent Discharge History")

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)  # Get 7 days for better context

        df_recent = fetch_instantaneous_values(
            site_id, param_cd=DEFAULT_PARAM_DISCHARGE,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )

        if df_recent is not None and not df_recent.empty:
            import plotly.graph_objects as go

            # Create log-scale chart with dynamic y-axis
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=df_recent.index,
                y=df_recent['value'],
                mode='lines',
                name='Discharge',
                line=dict(color='#1f77b4', width=2),
                fill='tozeroy',
                fillcolor='rgba(31, 119, 180, 0.2)'
            ))

            # Get current value for annotation
            current_val = df_recent['value'].iloc[-1] if len(df_recent) > 0 else None

            fig.update_layout(
                yaxis_type="log",
                yaxis_title="Discharge (cfs)",
                xaxis_title="",
                height=300,
                margin=dict(l=60, r=20, t=30, b=40),
                showlegend=False,
                hovermode='x unified'
            )

            # Dynamic y-axis range based on data
            y_min = df_recent['value'].min() * 0.8
            y_max = df_recent['value'].max() * 1.2
            fig.update_yaxes(range=[np.log10(max(y_min, 0.1)), np.log10(y_max)])

            st.plotly_chart(fig, use_container_width=True)

            # Historical context and precipitation influence
            col1, col2 = st.columns(2)

            with col1:
                # Calculate historical percentile for this time of year
                try:
                    # Get historical data for same day of year (past 10 years)
                    current_doy = end_date.timetuple().tm_yday
                    hist_start = (end_date - timedelta(days=365*10)).strftime('%Y-%m-%d')
                    hist_end = end_date.strftime('%Y-%m-%d')

                    df_hist = fetch_daily_values(
                        site_id, param_cd=DEFAULT_PARAM_DISCHARGE,
                        start_date=hist_start, end_date=hist_end
                    )

                    if df_hist is not None and not df_hist.empty:
                        # Filter to same season (±15 days from current day of year)
                        df_hist['doy'] = df_hist.index.dayofyear
                        seasonal_data = df_hist[
                            (df_hist['doy'] >= current_doy - 15) &
                            (df_hist['doy'] <= current_doy + 15)
                        ]['value']

                        if len(seasonal_data) > 10 and current_val:
                            percentile = (seasonal_data < current_val).mean() * 100

                            # Color based on percentile
                            if percentile > 90:
                                pct_color = "🔴"
                                pct_status = "Very High"
                            elif percentile > 75:
                                pct_color = "🟠"
                                pct_status = "Above Normal"
                            elif percentile > 25:
                                pct_color = "🟢"
                                pct_status = "Normal"
                            elif percentile > 10:
                                pct_color = "🟡"
                                pct_status = "Below Normal"
                            else:
                                pct_color = "🔵"
                                pct_status = "Very Low"

                            st.metric(
                                "Seasonal Percentile",
                                f"{percentile:.0f}%",
                                delta=pct_status,
                                delta_color="off"
                            )
                            st.caption(f"{pct_color} vs. historical {current_doy-15}-{current_doy+15} day-of-year")
                        else:
                            st.metric("Seasonal Percentile", "N/A")
                            st.caption("Insufficient historical data")
                    else:
                        st.metric("Seasonal Percentile", "N/A")
                except Exception:
                    st.metric("Seasonal Percentile", "N/A")

            with col2:
                # Recent precipitation influence
                try:
                    lat = site_info.get('latitude')
                    lon = site_info.get('longitude')
                    if lat and lon:
                        from hydrology.data.climate import fetch_climate_data
                        precip_start = (end_date - timedelta(days=7)).strftime('%Y-%m-%d')
                        precip_end = end_date.strftime('%Y-%m-%d')

                        climate_df = fetch_climate_data(
                            float(lat), float(lon),
                            precip_start, precip_end
                        )

                        if climate_df is not None and 'prcp' in climate_df.columns:
                            total_precip_mm = climate_df['prcp'].sum()
                            total_precip_in = total_precip_mm / 25.4

                            # Precipitation influence indicator
                            if total_precip_in > 2:
                                precip_icon = "🌧️"
                                precip_status = "High influence"
                            elif total_precip_in > 0.5:
                                precip_icon = "🌦️"
                                precip_status = "Moderate"
                            elif total_precip_in > 0.1:
                                precip_icon = "☁️"
                                precip_status = "Low"
                            else:
                                precip_icon = "☀️"
                                precip_status = "Minimal"

                            st.metric(
                                "7-Day Precipitation",
                                f"{total_precip_in:.2f} in",
                                delta=precip_status,
                                delta_color="off"
                            )
                            st.caption(f"{precip_icon} Recent weather influence on flow")
                        else:
                            st.metric("7-Day Precipitation", "N/A")
                    else:
                        st.metric("7-Day Precipitation", "N/A")
                except Exception:
                    st.metric("7-Day Precipitation", "N/A")

        else:
            st.info("No recent instantaneous data available")
    except Exception as e:
        st.warning(f"Could not load recent history: {e}")


# =============================================================================
# MULTI-SITE ANALYSIS MODE
# =============================================================================

def multisite_analysis_mode(inventory_df):
    """Analyze correlations and relationships between multiple sites."""
    st.sidebar.header("Site Selection")

    # Multi-site selection
    site_options = [f"{row['site_id']} - {str(row.get('description', ''))[:40]}"
                    for _, row in inventory_df.iterrows()]

    selected_sites = st.sidebar.multiselect(
        "Select Sites (2-6)",
        site_options,
        max_selections=6,
        key="multisite_selection"
    )

    if len(selected_sites) < 2:
        st.sidebar.warning("Select at least 2 sites")

    st.sidebar.markdown("---")

    # Date range
    st.sidebar.subheader("Analysis Period")
    years_back = st.sidebar.slider("Years of data", 1, 10, 3)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_back * 365)

    # Main content
    st.header("🔗 Multi-Site Analysis")
    st.caption("Analyze correlations and upstream/downstream relationships between monitoring sites")

    if len(selected_sites) < 2:
        st.info("👈 Select 2-6 sites from the sidebar to analyze their relationships")

        # Show example use cases with better formatting
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            #### 📊 Correlation Analysis
            - Compare discharge patterns between sites
            - Identify sites that respond similarly to events
            - Find correlated tributaries
            """)
        with col2:
            st.markdown("""
            #### ⏱️ Lag & Travel Time
            - Estimate water travel time between sites
            - Identify upstream/downstream relationships
            - Detect flow routing patterns
            """)
        return

    # Extract site IDs
    site_ids = [extract_site_id(s) for s in selected_sites]

    # Display selected sites in a nice card layout
    st.subheader("📍 Selected Sites")
    site_cols = st.columns(min(len(site_ids), 3))
    for i, sid in enumerate(site_ids):
        info = get_cached_site_info(sid)
        with site_cols[i % 3]:
            with st.container():
                st.markdown(f"**`{sid}`**")
                if info:
                    st.caption(info.get('description', '')[:50])

    st.markdown("---")

    if st.button("🔍 Analyze Relationships", type="primary", width='stretch'):
        # Create analyzer and fetch data
        analyzer = MultiSiteAnalyzer()

        for sid in site_ids:
            info = get_cached_site_info(sid)
            name = info.get('description', sid) if info else sid
            lat = info.get('latitude') if info else None
            lon = info.get('longitude') if info else None
            analyzer.add_site(sid, name=name[:40], latitude=lat, longitude=lon)

        # Fetch data
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        from hydrology.data.usgs import fetch_daily_values

        # Progress tracking
        progress_bar = st.progress(0, text="Fetching site data...")
        fetch_results = []

        for i, sid in enumerate(site_ids):
            progress_bar.progress((i + 1) / len(site_ids), text=f"Fetching {sid}...")
            try:
                df = fetch_daily_values(
                    sid, param_cd='00060',
                    start_date=start_str, end_date=end_str
                )
                if df is not None and not df.empty:
                    analyzer.data[sid] = df
                    fetch_results.append((sid, len(df), "success"))
                else:
                    fetch_results.append((sid, 0, "empty"))
            except Exception as e:
                fetch_results.append((sid, 0, f"error: {e}"))

        progress_bar.empty()

        # Show fetch results in expander (cleaner UI)
        with st.expander("📋 Data Fetch Details", expanded=False):
            for sid, count, status in fetch_results:
                if status == "success":
                    st.success(f"✅ {sid}: {count:,} records")
                elif status == "empty":
                    st.warning(f"⚠️ {sid}: No data")
                else:
                    st.error(f"❌ {sid}: {status}")

        # Get synchronized data
        synced_data = analyzer.get_synchronized_data()

        if synced_data.empty:
            st.error("❌ Could not synchronize data for selected sites")
            st.warning("Sites may have no overlapping data in the selected time range. Try extending the analysis period.")
            return

        # Summary metrics
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📅 Overlapping Days", f"{len(synced_data):,}")
        with col2:
            st.metric("📆 Date Range", f"{synced_data.index.min().strftime('%Y-%m-%d')}")
        with col3:
            st.metric("📆 To", f"{synced_data.index.max().strftime('%Y-%m-%d')}")

        # Time series visualization
        st.markdown("---")
        st.subheader("📈 Synchronized Time Series")

        # Create site name mapping for cleaner labels
        site_names = {}
        for sid in site_ids:
            info = get_cached_site_info(sid)
            if info:
                name = info.get('description', sid)[:25]
                site_names[sid] = f"{sid} - {name}"
            else:
                site_names[sid] = sid

        # Rename columns for display
        plot_data = synced_data.rename(columns=site_names)
        st.line_chart(plot_data, width='stretch')

        # Correlation Matrix with improved visualization
        st.markdown("---")
        st.subheader("📊 Correlation Matrix")

        corr_matrix = analyzer.get_correlation_matrix()

        if not corr_matrix.empty:
            # Create a nicer heatmap display
            import plotly.express as px
            import plotly.graph_objects as go

            # Create heatmap with plotly for better interactivity
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdYlGn',
                zmin=0,
                zmax=1,
                text=[[f"{val:.3f}" for val in row] for row in corr_matrix.values],
                texttemplate="%{text}",
                textfont={"size": 12},
                hovertemplate="Site A: %{y}<br>Site B: %{x}<br>Correlation: %{z:.3f}<extra></extra>"
            ))

            fig.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=30, b=20),
                xaxis_title="",
                yaxis_title="",
            )

            st.plotly_chart(fig, width='stretch')

            # Correlation interpretation
            avg_corr = corr_matrix.values[~np.eye(len(corr_matrix), dtype=bool)].mean()
            if avg_corr > 0.8:
                st.success(f"🔗 **High correlation** (avg: {avg_corr:.2f}) - Sites respond similarly to hydrologic events")
            elif avg_corr > 0.5:
                st.info(f"🔗 **Moderate correlation** (avg: {avg_corr:.2f}) - Sites show related but distinct patterns")
            else:
                st.warning(f"🔗 **Low correlation** (avg: {avg_corr:.2f}) - Sites may be in different drainage areas or have different drivers")

        # Pairwise Analysis
        st.markdown("---")
        st.subheader("🔄 Pairwise Relationships")

        results = analyzer.analyze_all_pairs()

        if results:
            # Create summary table
            summary_data = []
            for result in results:
                site_a_info = get_cached_site_info(result.site_a)
                site_b_info = get_cached_site_info(result.site_b)
                name_a = site_a_info.get('description', '')[:20] if site_a_info else ''
                name_b = site_b_info.get('description', '')[:20] if site_b_info else ''

                relationship_display = {
                    'upstream': '⬆️ A upstream of B',
                    'downstream': '⬇️ A downstream of B',
                    'parallel': '↔️ Parallel/Same timing',
                    'unknown': '❓ Undetermined'
                }.get(result.relationship, result.relationship)

                summary_data.append({
                    'Site A': f"{result.site_a}",
                    'Site B': f"{result.site_b}",
                    'Correlation': f"{result.correlation:.3f}",
                    'Lag (days)': result.lag_days,
                    'Relationship': relationship_display,
                    'Observations': f"{result.n_observations:,}"
                })

            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, width='stretch', hide_index=True)

            # Detailed expandable sections
            st.markdown("##### Detailed Analysis")
            for result in results:
                site_a_info = get_cached_site_info(result.site_a)
                site_b_info = get_cached_site_info(result.site_b)
                name_a = site_a_info.get('description', '')[:30] if site_a_info else result.site_a
                name_b = site_b_info.get('description', '')[:30] if site_b_info else result.site_b

                with st.expander(f"**{result.site_a}** ↔ **{result.site_b}**"):
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        corr_color = "normal" if result.correlation > 0.7 else ("off" if result.correlation < 0.3 else "normal")
                        st.metric("Correlation", f"{result.correlation:.3f}")

                    with col2:
                        st.metric("Optimal Lag", f"{result.lag_days} days")

                    with col3:
                        if result.lag_days != 0:
                            travel_time = abs(result.lag_days * 24)
                            st.metric("Travel Time", f"~{travel_time}h")
                        else:
                            st.metric("Travel Time", "Same day")

                    with col4:
                        relationship_icons = {
                            'upstream': '⬆️ Upstream',
                            'downstream': '⬇️ Downstream',
                            'parallel': '↔️ Parallel',
                            'unknown': '❓ Unknown'
                        }
                        st.metric("Relationship", relationship_icons.get(result.relationship, result.relationship))

                    st.caption(f"Based on {result.n_observations:,} observations | p-value: {result.p_value:.2e}")

                    # Interpretation
                    if result.relationship == 'upstream':
                        st.info(f"📍 {result.site_a} appears to be **upstream** of {result.site_b} with ~{result.lag_days} day travel time")
                    elif result.relationship == 'downstream':
                        st.info(f"📍 {result.site_a} appears to be **downstream** of {result.site_b}")
                    elif result.relationship == 'parallel':
                        st.info("📍 Sites respond at the **same time** - likely parallel tributaries or very close together")
                    else:
                        st.info("📍 Relationship unclear - sites may be in different watersheds or have weak connection")

        # Relationship Summary
        st.markdown("---")
        st.subheader("🌊 Relationship Summary")

        relationships = analyzer.identify_upstream_downstream()

        # Count relationship types
        has_upstream_downstream = False
        parallel_pairs = []
        unknown_pairs = []

        for result in results:
            if result.relationship in ['upstream', 'downstream']:
                has_upstream_downstream = True
            elif result.relationship == 'parallel':
                parallel_pairs.append((result.site_a, result.site_b))
            else:
                unknown_pairs.append((result.site_a, result.site_b))

        if has_upstream_downstream:
            # Show upstream/downstream relationships
            st.markdown("##### 🔀 Flow Direction")
            for sid, rels in relationships.items():
                if rels['upstream'] or rels['downstream']:
                    info = get_cached_site_info(sid)
                    name = info.get('description', sid)[:30] if info else sid

                    with st.container():
                        st.markdown(f"**{sid}** ({name})")
                        if rels['downstream']:
                            st.markdown(f"  └─ ⬆️ Flows to: {', '.join(rels['downstream'])}")
                        if rels['upstream']:
                            st.markdown(f"  └─ ⬇️ Receives from: {', '.join(rels['upstream'])}")
        else:
            st.info("ℹ️ **No clear upstream/downstream relationships detected**")
            st.markdown("""
            This typically means:
            - Sites respond to events on the **same day** (too close together for daily data to detect lag)
            - Sites may be **parallel tributaries** rather than on the same flow path
            - Sites could be in **different watersheds** with independent hydrology
            """)

        if parallel_pairs:
            st.markdown("##### ↔️ Parallel/Same-Timing Sites")
            for a, b in parallel_pairs:
                st.markdown(f"- {a} ↔ {b}")

        if unknown_pairs and not parallel_pairs and not has_upstream_downstream:
            st.markdown("##### ❓ Weakly Connected Sites")
            for a, b in unknown_pairs:
                st.markdown(f"- {a} ↔ {b}")

        # Tips
        with st.expander("💡 Tips for Better Results"):
            st.markdown("""
            - **Short lag times**: If sites are close together, travel time may be less than 1 day. Try using instantaneous data for better resolution.
            - **Low correlation**: Sites may be in different sub-watersheds or have different water sources (groundwater vs surface).
            - **Parallel relationships**: Common for tributary sites that both feed into a main stem.
            - **More data**: Longer time series (3+ years) generally produce more reliable relationship detection.
            """)


# =============================================================================
# NWM COMPARISON MODE
# =============================================================================

def nwm_comparison_mode(inventory_df):
    """Compare USGS observations with National Water Model forecasts."""
    st.sidebar.header("Site Selection")

    # Site selection
    site_options = [f"{row['site_id']} - {str(row.get('description', ''))[:40]}"
                    for _, row in inventory_df.iterrows()]
    selected = st.sidebar.selectbox("Select Site", site_options, key="nwm_site")
    site_id = extract_site_id(selected)

    site_info = get_cached_site_info(site_id)
    if not site_info:
        st.error(f"Site {site_id} not found")
        return

    desc = site_info.get('description', site_id)

    st.sidebar.markdown("---")

    # Comparison period
    st.sidebar.subheader("Comparison Period")
    days_back = st.sidebar.slider("Days to compare", 7, 90, 30)

    # Main content
    st.header("🌊 NWM Comparison")
    st.caption(f"Compare USGS observations with National Water Model for {desc}")

    st.markdown("""
    The **National Water Model (NWM)** produces streamflow forecasts for the entire US river network.
    This tool compares NWM analysis data with actual USGS observations to assess model accuracy.
    """)

    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        if st.button("🔍 Compare with NWM", type="primary"):
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            # Debug info
            st.info(f"🔍 Site: {site_id} | Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

            with st.spinner("Fetching NWM reach ID..."):
                client = NWMClient()

                # Show the URL being used
                api_url = f"https://api.water.usgs.gov/nldi/linked-data/nwissite/USGS-{site_id}"
                st.code(f"NLDI API: {api_url}", language=None)

                # Direct debug - make a request ourselves to see what's happening
                import requests as req
                try:
                    debug_resp = req.get(api_url, headers={'Accept': 'application/json'}, timeout=30)
                    with st.expander("🔧 Debug: Raw API Response", expanded=False):
                        st.write(f"Status: {debug_resp.status_code}")
                        if debug_resp.status_code == 200:
                            debug_data = debug_resp.json()
                            st.write(f"Features count: {len(debug_data.get('features', []))}")
                            if debug_data.get('features'):
                                props = debug_data['features'][0].get('properties', {})
                                st.write(f"Property keys: {list(props.keys())}")
                                st.write(f"comid value: {props.get('comid', 'NOT FOUND')}")
                        else:
                            st.write(f"Response: {debug_resp.text[:500]}")
                except Exception as debug_e:
                    st.warning(f"Debug request failed: {debug_e}")

                try:
                    reach_id = client.get_reach_id(site_id)
                except Exception as e:
                    st.error(f"Error calling NLDI API: {e}")
                    reach_id = None

                if reach_id:
                    st.success(f"✅ NWM Reach ID: {reach_id}")
                else:
                    st.error("❌ Could not find NWM reach ID for this site.")
                    st.warning("The NLDI API may have returned empty results or an error.")

            with st.spinner("Comparing NWM with USGS observations..."):
                comparison = compare_nwm_usgs(
                    site_id,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )

                if comparison is None:
                    st.error("Could not complete comparison. This may be due to:")
                    st.markdown("""
                    - Site not in NWM network
                    - NWM API unavailable
                    - Insufficient overlapping data
                    """)
                    return

                # Display metrics
                st.subheader("📊 Model Performance Metrics")

                col_a, col_b, col_c, col_d = st.columns(4)

                with col_a:
                    nse = comparison.nash_sutcliffe
                    nse_color = "normal" if nse > 0.5 else "off"
                    st.metric("Nash-Sutcliffe", f"{nse:.3f}",
                             help="NSE > 0.5 is generally considered good")

                with col_b:
                    st.metric("Correlation", f"{comparison.correlation:.3f}")

                with col_c:
                    st.metric("RMSE", f"{comparison.rmse:.1f} cfs")

                with col_d:
                    st.metric("Bias", f"{comparison.bias:+.1f} cfs",
                             help="Positive = NWM overestimates")

                # Additional metrics
                st.markdown("---")
                col_e, col_f, col_g = st.columns(3)

                with col_e:
                    st.metric("MAE", f"{comparison.mae:.1f} cfs")

                with col_f:
                    st.metric("Percent Bias", f"{comparison.percent_bias:+.1f}%")

                with col_g:
                    st.metric("N Observations", comparison.n_observations)

                # Skill rating
                st.markdown("---")
                st.subheader("🎯 Forecast Skill Rating")

                skill_result = get_forecast_skill(site_id, n_days=days_back)

                if 'rating' in skill_result:
                    rating = skill_result['rating']
                    rating_colors = {
                        'Excellent': '🟢',
                        'Good': '🟡',
                        'Fair': '🟠',
                        'Poor': '🔴',
                        'Very Poor': '⚫'
                    }
                    icon = rating_colors.get(rating, '⚪')
                    st.markdown(f"### {icon} {rating}")

                    st.markdown("""
                    **Rating Criteria:**
                    - 🟢 Excellent: NSE > 0.75, Correlation > 0.9
                    - 🟡 Good: NSE > 0.5, Correlation > 0.8
                    - 🟠 Fair: NSE > 0.25, Correlation > 0.6
                    - 🔴 Poor: NSE > 0, Correlation < 0.6
                    - ⚫ Very Poor: NSE < 0
                    """)

    with col2:
        st.markdown("### About NWM")
        st.markdown("""
        **Data Sources:**
        - Analysis (hindcast)
        - Short-range (0-18h)
        - Medium-range (0-10 days)
        - Long-range (0-30 days)

        **Resolution:**
        - 2.7 million river reaches
        - Hourly to 6-hourly output
        """)


# =============================================================================
# MAIN
# =============================================================================

def main():
    st.title("Hydrology Analysis Dashboard")

    # Load inventory
    inventory_df = get_inventory()
    if inventory_df.empty:
        st.error("Could not load site inventory")
        return

    # Mode selection
    mode = st.sidebar.radio(
        "Analysis Mode",
        [
            "Site Map",
            "Single Analysis",
            "Compare Time Periods",
            "Compare Sites",
            "2x2 Comparison",
            "🚨 Alert Monitor",
            "🔗 Multi-Site Analysis",
            "🌊 NWM Comparison"
        ],
        key="mode"
    )

    st.sidebar.markdown("---")

    if mode == "Site Map":
        site_map_mode(inventory_df)
    elif mode == "Single Analysis":
        single_analysis_mode(inventory_df)
    elif mode == "Compare Time Periods":
        compare_time_periods_mode(inventory_df)
    elif mode == "Compare Sites":
        compare_sites_mode(inventory_df)
    elif mode == "2x2 Comparison":
        quad_comparison_mode(inventory_df)
    elif mode == "🚨 Alert Monitor":
        alert_monitor_mode(inventory_df)
    elif mode == "🔗 Multi-Site Analysis":
        multisite_analysis_mode(inventory_df)
    elif mode == "🌊 NWM Comparison":
        nwm_comparison_mode(inventory_df)

    # Footer in sidebar
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Sites: {len(inventory_df)} | Plots: {len(AVAILABLE_PLOTS)}")

    # Styled footer in main area
    render_footer()


if __name__ == "__main__":
    main()
