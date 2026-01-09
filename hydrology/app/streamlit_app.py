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

    # Auto-fetch date ranges for all parameters
    availability_text = []

    # Discharge - get actual date range
    discharge_info = check_usgs_availability(site_id, DEFAULT_PARAM_DISCHARGE)
    if discharge_info:
        availability_text.append(f"✅ **Discharge** ({discharge_info['start']} to {discharge_info['end']})")
    elif begin_date:
        availability_text.append(f"✅ **Discharge** (from {begin_date})")
    else:
        availability_text.append("✅ **Discharge**")

    # Gage Height - get actual date range
    stage_info = check_usgs_availability(site_id, DEFAULT_PARAM_STAGE)
    if stage_info:
        availability_text.append(f"✅ **Gage Height** ({stage_info['start']} to {stage_info['end']})")
    else:
        availability_text.append("❌ **Gage Height** (not available)")

    # Climate - based on weather station distance
    if lat and lon:
        station = get_weather_station_info(float(lat), float(lon))
        if station:
            dist = station.get('distance_km')
            name = station.get('name', 'Unknown')
            if dist is not None:
                if dist < 20:
                    availability_text.append(f"✅ **Climate** ({name}, {dist:.1f} km)")
                elif dist < 50:
                    availability_text.append(f"⚠️ **Climate** ({name}, {dist:.1f} km)")
                else:
                    availability_text.append(f"⚠️ **Climate** ({name}, {dist:.0f} km - distant)")
            else:
                availability_text.append(f"⚠️ **Climate** ({name})")
        else:
            availability_text.append("❌ **Climate** (no station found)")
    else:
        availability_text.append("❌ **Climate** (no coordinates)")

    # Display all availability indicators
    for text in availability_text:
        st.sidebar.markdown(text)


def date_range_selector(key_prefix: str = "", default_start: date = None, default_end: date = None):
    """
    Date range selector with synced slider and manual inputs.
    Returns (start_date, end_date) tuple.
    """
    if default_start is None:
        default_start = date(2015, 1, 1)
    if default_end is None:
        default_end = date.today()

    min_date = date(1900, 1, 1)
    max_date = date.today()

    # Initialize session state for this date range
    start_key = f"{key_prefix}_start_val"
    end_key = f"{key_prefix}_end_val"
    
    if start_key not in st.session_state:
        st.session_state[start_key] = default_start
    if end_key not in st.session_state:
        st.session_state[end_key] = default_end

    # Manual input boxes (primary controls)
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input(
            "Start Date",
            value=st.session_state[start_key],
            min_value=min_date,
            max_value=max_date,
            key=f"{key_prefix}_start"
        )
    with col2:
        end = st.date_input(
            "End Date",
            value=st.session_state[end_key],
            min_value=min_date,
            max_value=max_date,
            key=f"{key_prefix}_end"
        )

    # Update session state from inputs
    st.session_state[start_key] = start
    st.session_state[end_key] = end

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

    return {
        'df_q': df_q,
        'df_merged': df_merged,
        'analysis_results': analysis_results,
        'discharge_count': len(df_q) if df_q is not None else 0,
        'climate_count': len(df_climate) if df_climate is not None else 0,
        'merged_count': len(df_merged) if df_merged is not None else 0
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
                # Check if plot requires merged data but it's missing
                plot_requires = plot_info.get('requires', [])
                has_merged = data.get('df_merged') is not None and not data['df_merged'].empty if 'df_merged' in data else False
                has_discharge = data.get('df_q') is not None and not data['df_q'].empty if 'df_q' in data else False

                if 'df_merged' in plot_requires and not has_merged:
                    ax.text(0.5, 0.5, "No Climate Data\nfor this period",
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
                elif 'df_q' in plot_requires and not has_discharge:
                    ax.text(0.5, 0.5, "No Discharge Data\nfor this period",
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
                else:
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
    st.sidebar.header("Site Selection")

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

    # Date range
    st.sidebar.markdown("---")
    st.sidebar.header("Date Range")
    start_date, end_date = date_range_selector("single")

    # Plot selection
    st.sidebar.markdown("---")
    st.sidebar.header("Plot Selection")
    selected_plots = plot_selector()

    # Layout
    st.sidebar.markdown("---")
    st.sidebar.header("Layout")
    layout_options = {
        'Auto': PlotLayout.AUTO,
        'Vertical': PlotLayout.VERTICAL,
        'Quad (2x2)': PlotLayout.QUAD,
        'Grid 2x3': PlotLayout.GRID_2x3,
        'Grid 3x2': PlotLayout.GRID_3x2,
        'Grid 2x5': PlotLayout.GRID_2x5,
    }
    layout = st.sidebar.selectbox("Layout", list(layout_options.keys()))
    dpi = st.sidebar.slider("DPI", 72, 300, 150)

    # Generate
    st.sidebar.markdown("---")
    if st.sidebar.button("Generate Plots", type="primary", width='stretch'):
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
        st.info("Select a site and date range, then click 'Generate Plots'")


def compare_time_periods_mode(inventory_df):
    """Compare same site across two equal-length time periods."""
    st.sidebar.header("Site Selection")

    site_options = [f"{row['site_id']} - {str(row.get('description', ''))[:40]}"
                    for _, row in inventory_df.iterrows()]
    selected = st.sidebar.selectbox("Select Site", site_options, key="compare_time_site")
    site_id = selected.split(" - ")[0]

    site_info = get_cached_site_info(site_id)
    if not site_info:
        st.error(f"Site {site_id} not found")
        return

    lat = site_info.get('latitude')
    lon = site_info.get('longitude')
    desc = site_info.get('description', site_id)

    display_site_info(site_info, show_check_button=False)

    # Period length selection
    st.sidebar.markdown("---")
    st.sidebar.header("Period Length")
    period_lengths = {
        "1 Year": 365,
        "2 Years": 730,
        "5 Years": 1825,
        "10 Years": 3650,
        "Water Year (Oct-Sep)": 365,
        "Custom": None
    }
    period_choice = st.sidebar.selectbox("Select period length", list(period_lengths.keys()), key="period_length")

    if period_choice == "Custom":
        custom_days = st.sidebar.number_input("Days", min_value=30, max_value=7300, value=365, key="custom_days")
        period_days = custom_days
    else:
        period_days = period_lengths[period_choice]

    st.sidebar.caption(f"Each period: {period_days} days ({period_days/365:.1f} years)")

    # Period A - start date with shift controls
    st.sidebar.markdown("---")
    st.sidebar.header("Period A")

    # Initialize session state for Period A start
    if 'period_a_start' not in st.session_state:
        st.session_state.period_a_start = date(2010, 1, 1)

    col_a1, col_a2, col_a3 = st.sidebar.columns([1, 2, 1])
    with col_a1:
        if st.button("◀", key="shift_a_back", help=f"Shift back {period_days} days"):
            st.session_state.period_a_start = st.session_state.period_a_start - timedelta(days=period_days)
    with col_a2:
        start_a = st.date_input("Start A", st.session_state.period_a_start, key="start_a_input",
                                min_value=date(1900, 1, 1), max_value=date.today())
        st.session_state.period_a_start = start_a
    with col_a3:
        if st.button("▶", key="shift_a_fwd", help=f"Shift forward {period_days} days"):
            st.session_state.period_a_start = st.session_state.period_a_start + timedelta(days=period_days)

    end_a = start_a + timedelta(days=period_days)
    st.sidebar.caption(f"End A: {end_a}")

    # Period B - start date with shift controls
    st.sidebar.markdown("---")
    st.sidebar.header("Period B")

    # Initialize session state for Period B start
    if 'period_b_start' not in st.session_state:
        st.session_state.period_b_start = date(2020, 1, 1)

    col_b1, col_b2, col_b3 = st.sidebar.columns([1, 2, 1])
    with col_b1:
        if st.button("◀", key="shift_b_back", help=f"Shift back {period_days} days"):
            st.session_state.period_b_start = st.session_state.period_b_start - timedelta(days=period_days)
    with col_b2:
        start_b = st.date_input("Start B", st.session_state.period_b_start, key="start_b_input",
                                min_value=date(1900, 1, 1), max_value=date.today())
        st.session_state.period_b_start = start_b
    with col_b3:
        if st.button("▶", key="shift_b_fwd", help=f"Shift forward {period_days} days"):
            st.session_state.period_b_start = st.session_state.period_b_start + timedelta(days=period_days)

    end_b = start_b + timedelta(days=period_days)
    st.sidebar.caption(f"End B: {end_b}")

    # Plot selection
    st.sidebar.markdown("---")
    st.sidebar.header("Plot to Compare")
    plot_name = single_plot_selector_widget("_compare")

    dpi = st.sidebar.slider("DPI", 72, 300, 150, key="compare_time_dpi")

    # Generate
    st.sidebar.markdown("---")
    if st.sidebar.button("Compare Periods", type="primary", width='stretch'):
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

        # Create comparison figure
        data_list = []
        titles = []

        if data_a:
            data_list.append({'df_q': data_a['df_q'], 'df_merged': data_a['df_merged'],
                             'analysis_results': data_a['analysis_results']})
            titles.append(f"{desc}\n{start_a} to {end_a}")

        if data_b:
            data_list.append({'df_q': data_b['df_q'], 'df_merged': data_b['df_merged'],
                             'analysis_results': data_b['analysis_results']})
            titles.append(f"{desc}\n{start_b} to {end_b}")

        if data_list:
            fig = create_comparison_figure(plot_name, data_list, titles, 1, len(data_list), dpi)
            st.pyplot(fig)
            render_export_buttons(fig, f"{site_id}_comparison", dpi)
            plt.close(fig)
    else:
        st.info("Select a site, period length, and start dates to compare equal-length periods")


def compare_sites_mode(inventory_df):
    """Compare multiple sites for the same time period."""
    st.sidebar.header("Site Selection")

    site_options = [f"{row['site_id']} - {str(row.get('description', ''))[:40]}"
                    for _, row in inventory_df.iterrows()]
    selected_sites = st.sidebar.multiselect(
        "Select Sites (2-4)",
        site_options,
        max_selections=4,
        key="compare_sites_multi"
    )

    if len(selected_sites) < 2:
        st.sidebar.warning("Select 2-4 sites to compare")

    # Date range
    st.sidebar.markdown("---")
    st.sidebar.header("Date Range")
    start_date, end_date = date_range_selector("compare_sites")

    # Plot selection
    st.sidebar.markdown("---")
    st.sidebar.header("Plot to Compare")
    plot_name = single_plot_selector_widget("_sites")

    dpi = st.sidebar.slider("DPI", 72, 300, 150, key="compare_sites_dpi")

    # Generate
    st.sidebar.markdown("---")
    if st.sidebar.button("Compare Sites", type="primary", width='stretch'):
        if len(selected_sites) < 2:
            st.warning("Select at least 2 sites")
            return

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

        for site_str in selected_sites:
            site_id = site_str.split(" - ")[0]
            site_info = get_cached_site_info(site_id)

            if not site_info:
                continue

            lat = site_info.get('latitude')
            lon = site_info.get('longitude')
            desc = site_info.get('description', site_id)

            if not lat or not lon:
                continue

            with st.spinner(f"Processing {site_id}..."):
                data = process_site_data(site_id, float(lat), float(lon), start_str, end_str)

            if data:
                data_list.append({
                    'df_q': data['df_q'],
                    'df_merged': data['df_merged'],
                    'analysis_results': data['analysis_results']
                })
                titles.append(f"{desc[:30]}\n({data['merged_count']:,} merged)")

        if data_list:
            # Determine grid layout
            n = len(data_list)
            if n <= 2:
                nrows, ncols = 1, n
            elif n <= 4:
                nrows, ncols = 2, 2
            else:
                nrows, ncols = 2, 3

            fig = create_comparison_figure(plot_name, data_list, titles, nrows, ncols, dpi)
            st.pyplot(fig)
            render_export_buttons(fig, "multi_site_comparison", dpi)
            plt.close(fig)
    else:
        st.info("Select 2-4 sites, a date range, and a plot type to compare")


def quad_comparison_mode(inventory_df):
    """2x2 comparison: 2 sites x 2 equal-length time periods."""
    st.sidebar.header("Sites")

    site_options = [f"{row['site_id']} - {str(row.get('description', ''))[:40]}"
                    for _, row in inventory_df.iterrows()]

    site_a = st.sidebar.selectbox("Site A", site_options, key="quad_site_a")
    site_b = st.sidebar.selectbox("Site B", site_options, key="quad_site_b", index=min(1, len(site_options)-1))

    # Period length selection
    st.sidebar.markdown("---")
    st.sidebar.header("Period Length")
    period_lengths = {
        "1 Year": 365,
        "2 Years": 730,
        "5 Years": 1825,
        "10 Years": 3650,
        "Custom": None
    }
    period_choice = st.sidebar.selectbox("Select period length", list(period_lengths.keys()), key="quad_period_length")

    if period_choice == "Custom":
        custom_days = st.sidebar.number_input("Days", min_value=30, max_value=7300, value=365, key="quad_custom_days")
        period_days = custom_days
    else:
        period_days = period_lengths[period_choice]

    st.sidebar.caption(f"Each period: {period_days} days ({period_days/365:.1f} years)")

    # Period 1 - start date with shift controls
    st.sidebar.markdown("---")
    st.sidebar.header("Period 1")

    if 'quad_p1_start' not in st.session_state:
        st.session_state.quad_p1_start = date(2010, 1, 1)

    col_p1a, col_p1b, col_p1c = st.sidebar.columns([1, 2, 1])
    with col_p1a:
        if st.button("◀", key="shift_p1_back", help=f"Shift back {period_days} days"):
            st.session_state.quad_p1_start = st.session_state.quad_p1_start - timedelta(days=period_days)
    with col_p1b:
        start_1 = st.date_input("Start 1", st.session_state.quad_p1_start, key="quad_start_1",
                                min_value=date(1900, 1, 1), max_value=date.today())
        st.session_state.quad_p1_start = start_1
    with col_p1c:
        if st.button("▶", key="shift_p1_fwd", help=f"Shift forward {period_days} days"):
            st.session_state.quad_p1_start = st.session_state.quad_p1_start + timedelta(days=period_days)

    end_1 = start_1 + timedelta(days=period_days)
    st.sidebar.caption(f"End 1: {end_1}")

    # Period 2 - start date with shift controls
    st.sidebar.markdown("---")
    st.sidebar.header("Period 2")

    if 'quad_p2_start' not in st.session_state:
        st.session_state.quad_p2_start = date(2020, 1, 1)

    col_p2a, col_p2b, col_p2c = st.sidebar.columns([1, 2, 1])
    with col_p2a:
        if st.button("◀", key="shift_p2_back", help=f"Shift back {period_days} days"):
            st.session_state.quad_p2_start = st.session_state.quad_p2_start - timedelta(days=period_days)
    with col_p2b:
        start_2 = st.date_input("Start 2", st.session_state.quad_p2_start, key="quad_start_2",
                                min_value=date(1900, 1, 1), max_value=date.today())
        st.session_state.quad_p2_start = start_2
    with col_p2c:
        if st.button("▶", key="shift_p2_fwd", help=f"Shift forward {period_days} days"):
            st.session_state.quad_p2_start = st.session_state.quad_p2_start + timedelta(days=period_days)

    end_2 = start_2 + timedelta(days=period_days)
    st.sidebar.caption(f"End 2: {end_2}")

    # Plot selection
    st.sidebar.markdown("---")
    st.sidebar.header("Plot to Compare")
    plot_name = single_plot_selector_widget("_quad")

    dpi = st.sidebar.slider("DPI", 72, 300, 150, key="quad_dpi")

    # Generate
    st.sidebar.markdown("---")
    if st.sidebar.button("Generate 2x2 Comparison", type="primary", width='stretch'):
        site_id_a = site_a.split(" - ")[0]
        site_id_b = site_b.split(" - ")[0]

        site_info_a = get_cached_site_info(site_id_a)
        site_info_b = get_cached_site_info(site_id_b)

        if not site_info_a or not site_info_b:
            st.error("Could not load site info")
            return

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

        for site_id, site_info, start_d, end_d, title in configs:
            lat = site_info.get('latitude')
            lon = site_info.get('longitude')

            if not lat or not lon:
                data_list.append(None)
                titles.append(title + "\n(No coords)")
                continue

            with st.spinner(f"Processing {site_id} ({start_d} to {end_d})..."):
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

        fig = create_comparison_figure(plot_name, data_list, titles, 2, 2, dpi)

        # Add row/col labels
        fig.text(0.02, 0.75, 'Site A', rotation=90, fontsize=12, fontweight='bold', va='center')
        fig.text(0.02, 0.25, 'Site B', rotation=90, fontsize=12, fontweight='bold', va='center')
        fig.text(0.3, 0.98, 'Period 1', fontsize=12, fontweight='bold', ha='center')
        fig.text(0.7, 0.98, 'Period 2', fontsize=12, fontweight='bold', ha='center')

        st.pyplot(fig)
        render_export_buttons(fig, "2x2_comparison", dpi)
        plt.close(fig)
    else:
        st.info("Select 2 sites, 2 time periods, and a plot type for 2x2 comparison")


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
