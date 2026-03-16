"""
Hydrology Analysis Dashboard - Multipage App Entry Point

Run with: streamlit run hydrology/app/app.py
"""

import sys
from pathlib import Path

# Ensure the hydrology package is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

st.set_page_config(page_title="Hydrology Analysis", page_icon="\U0001f4a7", layout="wide")

from hydrology.app.styles import apply_custom_css, render_footer
from hydrology.app.shared import get_inventory
from hydrology.visualization.plots import AVAILABLE_PLOTS

# Page imports
from hydrology.app.pages import overview, single_analysis, comparisons, reach_analysis, alerts, advanced, indicators

apply_custom_css()

# Define pages with grouping
pages = {
    "Dashboard": [
        st.Page(overview.show, title="Overview", icon="\U0001f4ca", default=True, url_path="overview"),
        st.Page(single_analysis.show, title="Single Analysis", icon="\U0001f4c8", url_path="single-analysis"),
    ],
    "Compare": [
        st.Page(comparisons.show, title="Comparisons", icon="\U0001f504", url_path="comparisons"),
        st.Page(reach_analysis.show, title="Reach Analysis", icon="\U0001f30a", url_path="reach-analysis"),
    ],
    "Monitor": [
        st.Page(alerts.show, title="Alerts", icon="\U0001f6a8", url_path="alerts"),
        st.Page(indicators.show, title="Indicators", icon="\U0001f321\ufe0f", url_path="indicators"),
        st.Page(advanced.show, title="Advanced", icon="\U0001f52c", url_path="advanced"),
    ],
}

pg = st.navigation(pages)

# Footer in sidebar
inventory_df = get_inventory()
st.sidebar.markdown("---")
site_count = len(inventory_df) if not inventory_df.empty else 0
st.sidebar.caption(f"Sites: {site_count} | Plots: {len(AVAILABLE_PLOTS)}")

pg.run()

render_footer()
