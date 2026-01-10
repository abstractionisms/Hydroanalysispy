"""
Streamlit web application for hydrology analysis.

Features:
- Single site analysis with multiple plot types
- Data availability display with weather station distance
- Date range slider with visual selection
- Comparison modes: time periods, sites, and 2x2 grid
"""

import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import io
import sys

# Ensure the hydrology package is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hydrology.data.inventory import load_inventory, get_site_info
from hydrology.data.usgs import (
    fetch_waterml_data, parse_waterml, fetch_daily_values, fetch_stage_data,
    DEFAULT_PARAM_DISCHARGE, DEFAULT_PARAM_STAGE
)
from hydrology.data.climate import fetch_climate_data, fetch_nearest_station_info
from hydrology.visualization import create_multi_plot, PlotLayout
from hydrology.visualization.plots import AVAILABLE_PLOTS
from hydrology.scripts.analyze_sites import analyze_correlation
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
def find_availability_windows(site_id: str, param_cd: str, check_iv: bool = False, hint_start_year: int = None):
    """
    Fast availability check - only makes 2-3 API calls instead of 25+.
    Returns list of (start_year, end_year) tuples.

    Strategy:
    1. Check recent data (last 2 years)
    2. Check historical data (around hint_start_year or 1950)
    3. If both exist, return single window. If gap detected, note it.
    """
    import requests

    current_year = date.today().year
    has_recent = False
    has_historical = False
    earliest_year = None

    # Check 1: Recent data (last 2 years) - single call
    try:
        recent_start = f"{current_year - 2}-01-01"
        recent_end = f"{current_year}-12-31"
        df = fetch_daily_values(
            site_id, param_cd=param_cd,
            start_date=recent_start, end_date=recent_end,
            chunk_years=3
        )
        if df is not None and not df.empty:
            has_recent = True
    except Exception:
        pass

    # Check 2: Historical data - check around hint year or default to 1960
    check_year = hint_start_year if hint_start_year else 1960
    try:
        hist_start = f"{check_year}-01-01"
        hist_end = f"{check_year + 5}-12-31"
        df = fetch_daily_values(
            site_id, param_cd=param_cd,
            start_date=hist_start, end_date=hist_end,
            chunk_years=6
        )
        if df is not None and not df.empty:
            has_historical = True
            earliest_year = df.index.min().year
    except Exception:
        pass

    # If no DV data found and check_iv requested, try IV for recent
    if not has_recent and not has_historical and check_iv:
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
                        # IV data found
                        return [(current_year - 1, "present")]
        except Exception:
            pass

    # Build result
    if not has_recent and not has_historical:
        return None

    if has_recent and has_historical:
        # Both exist - return single window from earliest to present
        start_year = earliest_year if earliest_year else check_year
        return [(start_year, "present")]
    elif has_recent:
        return [(current_year - 2, "present")]
    else:  # has_historical only
        return [(earliest_year if earliest_year else check_year, check_year + 5)]


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
        # Fall back to quick estimate
        hint_year = None
        if begin_date:
            try:
                hint_year = int(str(begin_date)[:4])
            except (ValueError, TypeError):
                pass

        discharge_windows = find_availability_windows(site_id, DEFAULT_PARAM_DISCHARGE, check_iv=False, hint_start_year=hint_year)
        if discharge_windows:
            windows_str = format_availability_windows(discharge_windows)
            availability_text.append(f"✅ **Discharge** ({windows_str}) ~est")
        elif begin_date:
            availability_text.append(f"✅ **Discharge** (from {begin_date}) ~est")
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
        # Fall back to quick estimate
        hint_year = None
        if begin_date:
            try:
                hint_year = int(str(begin_date)[:4])
            except (ValueError, TypeError):
                pass

        stage_dv_windows = find_availability_windows(site_id, DEFAULT_PARAM_STAGE, check_iv=False, hint_start_year=hint_year)
        if stage_dv_windows:
            windows_str = format_availability_windows(stage_dv_windows)
            availability_text.append(f"✅ **Gage Height** ({windows_str}) ~est")
        else:
            stage_iv_windows = find_availability_windows(site_id, DEFAULT_PARAM_STAGE, check_iv=True, hint_start_year=hint_year)
            if stage_iv_windows:
                windows_str = format_availability_windows(stage_iv_windows)
                availability_text.append(f"✅ **Gage Height** IV ({windows_str}) ~est")
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

    st.caption(f"Range: {start} → {end}")

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
    df_q = fetch_discharge_data(site_id, "00060", start_str, end_str)
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
        # Timezone handling
        if df_q.index.tz is None:
            df_q.index = df_q.index.tz_localize('UTC')
        else:
            df_q.index = df_q.index.tz_convert('UTC')

        if df_climate.index.tz is None:
            df_climate.index = df_climate.index.tz_localize('UTC')
        else:
            df_climate.index = df_climate.index.tz_convert('UTC')

        df_merged = pd.merge(df_q, df_climate, left_index=True, right_index=True, how='inner')

        if not df_merged.empty:
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
                # Check data availability
                plot_requires = plot_info.get('requires', [])
                has_merged = data.get('df_merged') is not None and not data['df_merged'].empty if 'df_merged' in data else False
                has_discharge = data.get('df_q') is not None and not data['df_q'].empty if 'df_q' in data else False

                # Only block if NO data at all is available
                if 'df_q' in plot_requires and not has_discharge and not has_merged:
                    ax.text(0.5, 0.5, "No Discharge Data\nfor this period",
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
                else:
                    # Let plot functions handle missing data gracefully with fallbacks
                    plot_func(ax, **data, config={})
                ax.set_title(title, fontsize=10)
            except Exception as e:
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
    site_id = selected.split(" - ")[0]

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
        generate = st.button("🔍 Generate Plots", type="primary", use_container_width=True)

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
    site_id = selected.split(" - ")[0]

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
        generate = st.button("🔍 Compare Periods", type="primary", use_container_width=True)

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
        else:
            data_list.append(None)
            st.warning(f"No data available for Period A ({start_a} to {end_a})")
        titles.append(f"{desc}\n{start_a} to {end_a}")

        # Period B - always add (None if no data)
        if data_b:
            data_list.append({'df_q': data_b['df_q'], 'df_merged': data_b['df_merged'],
                             'analysis_results': data_b['analysis_results']})
        else:
            data_list.append(None)
            st.warning(f"No data available for Period B ({start_b} to {end_b})")
        titles.append(f"{desc}\n{start_b} to {end_b}")

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
            site_id = site_str.split(" - ")[0]
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
    if st.button("🔍 Compare Sites", type="primary", use_container_width=True):
        # Show comparison header
        st.markdown(f"""
        <div class="site-header">
            <h1>Multi-Site Comparison</h1>
            <p>{len(selected_sites)} sites | {start_date} to {end_date}</p>
        </div>
        """, unsafe_allow_html=True)

        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        data_list = []
        titles = []

        progress = st.progress(0, text="Loading sites...")
        for i, site_str in enumerate(selected_sites):
            site_id = site_str.split(" - ")[0]
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
                titles.append(f"{desc[:30]}\n({data['merged_count']:,} merged)")
            else:
                data_list.append(None)
                titles.append(f"{desc[:30]}\n(No data)")

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
        site_id = site_str.split(" - ")[0]
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
        generate = st.button("🔍 Generate 2×2", type="primary", use_container_width=True)

    # Generate comparison
    if generate:
        site_id_a = site_a.split(" - ")[0]
        site_id_b = site_b.split(" - ")[0]

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
                titles.append(title)
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
    """Display all sites on an interactive map."""
    st.header("Site Locations")

    # Prepare map data
    map_data = inventory_df[['latitude', 'longitude', 'site_id', 'description']].copy()
    map_data = map_data.dropna(subset=['latitude', 'longitude'])
    map_data['latitude'] = map_data['latitude'].astype(float)
    map_data['longitude'] = map_data['longitude'].astype(float)

    # Rename for st.map compatibility
    map_data = map_data.rename(columns={'latitude': 'lat', 'longitude': 'lon'})

    st.caption(f"Showing {len(map_data)} sites with coordinates")

    # Display map - uses dark theme automatically
    st.map(map_data, width='stretch')

    # Site list below map
    st.subheader("Site List")
    display_df = inventory_df[['site_id', 'description', 'latitude', 'longitude', 'begin_date']].copy()
    display_df = display_df.dropna(subset=['latitude', 'longitude'])
    st.dataframe(display_df, width='stretch', hide_index=True)


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
        ["Site Map", "Single Analysis", "Compare Time Periods", "Compare Sites", "2x2 Comparison"],
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

    # Footer in sidebar
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Sites: {len(inventory_df)} | Plots: {len(AVAILABLE_PLOTS)}")

    # Styled footer in main area
    render_footer()


if __name__ == "__main__":
    main()
