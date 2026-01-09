"""
Custom styling and UI components for the Hydrology Dashboard.
"""

import streamlit as st
from typing import Dict, Any, Optional
import pandas as pd


def apply_custom_css():
    """Apply custom CSS for a polished dark theme."""
    st.markdown("""
    <style>
    /* Main container padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Card-style containers */
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #0d1b2a 100%);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #2d4a6f;
        margin-bottom: 0.5rem;
    }

    /* Site header styling */
    .site-header {
        background: linear-gradient(90deg, #1a5f7a 0%, #0d3b4d 100%);
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border-left: 4px solid #4ecdc4;
    }

    .site-header h1 {
        margin: 0;
        color: #ffffff;
        font-size: 1.5rem;
    }

    .site-header p {
        margin: 0.5rem 0 0 0;
        color: #a0d2db;
        font-size: 0.9rem;
    }

    /* Data availability badges */
    .badge-available {
        background-color: #2d5a3d;
        color: #4ade80;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.8rem;
        display: inline-block;
        margin: 0.2rem;
    }

    .badge-unavailable {
        background-color: #5a2d2d;
        color: #f87171;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.8rem;
        display: inline-block;
        margin: 0.2rem;
    }

    .badge-warning {
        background-color: #5a4d2d;
        color: #fbbf24;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.8rem;
        display: inline-block;
        margin: 0.2rem;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 0.95rem;
    }

    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.85rem;
    }

    /* Sidebar improvements */
    .css-1d391kg {
        padding-top: 1rem;
    }

    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }

    /* Plot container */
    .plot-container {
        background: #0e1117;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    /* Footer styling */
    .footer-info {
        text-align: center;
        color: #6b7280;
        font-size: 0.8rem;
        padding: 1rem;
        border-top: 1px solid #374151;
        margin-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)


def render_site_header(site_id: str, description: str, lat: float = None, lon: float = None):
    """Render a styled header for the selected site."""
    coords = f"({lat:.4f}, {lon:.4f})" if lat and lon else ""

    st.markdown(f"""
    <div class="site-header">
        <h1>{description}</h1>
        <p>USGS Site {site_id} {coords}</p>
    </div>
    """, unsafe_allow_html=True)


def render_availability_badges(has_discharge: bool, has_stage: bool, climate_info: Optional[Dict] = None):
    """Render horizontal availability badges."""
    badges = []

    if has_discharge:
        badges.append('<span class="badge-available">✓ Discharge</span>')
    else:
        badges.append('<span class="badge-unavailable">✗ Discharge</span>')

    if has_stage:
        badges.append('<span class="badge-available">✓ Gage Height</span>')
    else:
        badges.append('<span class="badge-unavailable">✗ Gage Height</span>')

    if climate_info:
        dist = climate_info.get('distance_km')
        name = climate_info.get('name', 'Unknown')
        if dist is not None:
            if dist < 20:
                badges.append(f'<span class="badge-available">✓ Climate ({dist:.0f}km)</span>')
            elif dist < 50:
                badges.append(f'<span class="badge-warning">⚠ Climate ({dist:.0f}km)</span>')
            else:
                badges.append(f'<span class="badge-warning">⚠ Climate ({dist:.0f}km - distant)</span>')
        else:
            badges.append('<span class="badge-warning">⚠ Climate</span>')
    else:
        badges.append('<span class="badge-unavailable">✗ Climate</span>')

    st.markdown(' '.join(badges), unsafe_allow_html=True)


def render_metric_cards(df_q: pd.DataFrame, df_merged: pd.DataFrame = None, discharge_col: str = 'Discharge_cfs'):
    """Render key statistics as metric cards."""
    if df_q is None or df_q.empty:
        return

    col1, col2, col3, col4 = st.columns(4)

    # Record length
    if hasattr(df_q.index, 'min') and hasattr(df_q.index, 'max'):
        years = (df_q.index.max() - df_q.index.min()).days / 365.25
        with col1:
            st.metric("Record Length", f"{years:.1f} yrs", help="Total years of data")

    # Data points
    with col2:
        st.metric("Data Points", f"{len(df_q):,}", help="Number of daily observations")

    # Mean discharge
    if discharge_col in df_q.columns:
        mean_q = df_q[discharge_col].mean()
        with col3:
            st.metric("Mean Flow", f"{mean_q:,.0f} cfs", help="Average daily discharge")

        # Max discharge
        max_q = df_q[discharge_col].max()
        with col4:
            st.metric("Peak Flow", f"{max_q:,.0f} cfs", help="Maximum recorded discharge")


def render_progress_bar(current: int, total: int, text: str = "Loading..."):
    """Render a progress bar for data loading."""
    progress = current / total if total > 0 else 0
    st.progress(progress, text=f"{text} ({current}/{total})")


def render_footer():
    """Render footer with app info."""
    st.markdown("""
    <div class="footer-info">
        <p>Hydrology Analysis Dashboard | Data: USGS NWIS & Meteostat |
        <a href="https://github.com/abstractionisms/Hydroanalysispy" target="_blank">GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)
