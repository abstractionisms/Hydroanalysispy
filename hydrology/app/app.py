"""
Hydrology Analysis Dashboard - Multipage App Entry Point

Run with: streamlit run hydrology/app/app.py
"""

import sys
from pathlib import Path

# Ensure the hydrology package is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

st.set_page_config(
    page_title="HydroPlot",
    page_icon="\U0001f4a7",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from hydrology.app.styles import (
    apply_custom_css,
    render_footer,
    render_dashboard_hero,
    render_dashboard_meta,
    render_main_nav,
)
from hydrology.app.shared import get_inventory
from hydrology.visualization.plots import AVAILABLE_PLOTS

# Page imports
from hydrology.app.page_modules import (
    overview,
    single_analysis,
    comparisons,
    reach_analysis,
    watershed,
)

apply_custom_css()
render_dashboard_hero(
    "HydroPlot",
    "Find gages, analyze one site, compare records, evaluate reaches, and inspect basin context — one refined workspace.",
)

# Define pages with grouping
_single_analysis_page = st.Page(
    single_analysis.show, title="Site Analysis", icon="📈", url_path="single-analysis"
)
_overview_page = st.Page(
    overview.show, title="Stations", icon="\U0001f4ca", default=True, url_path="overview"
)
_comparisons_page = st.Page(
    comparisons.show, title="Compare Sites", icon="\U0001f504", url_path="comparisons"
)
_reach_page = st.Page(
    reach_analysis.show, title="Reach Analysis", icon="\U0001f30a", url_path="reach-analysis"
)
_watershed_page = st.Page(
    watershed.show, title="Watershed", icon="\U0001f5fa\ufe0f", url_path="watershed"
)

pages = {
    "Workflows": [
        _overview_page,
        _single_analysis_page,
        _comparisons_page,
        _reach_page,
        _watershed_page,
    ],
}

# Hide Streamlit's default multipage chrome; custom main-nav is the product UI.
try:
    pg = st.navigation(pages, position="hidden")
except TypeError:
    # Older Streamlit without position= support
    pg = st.navigation(pages)

# Active pill highlight for custom main nav
try:
    active = getattr(pg, "url_path", None) or getattr(pg, "_url_path", None) or "overview"
    st.session_state["_hydro_active_page"] = str(active).strip("/") or "overview"
except Exception:
    st.session_state["_hydro_active_page"] = "overview"

render_main_nav(st.session_state.get("_hydro_active_page"))

# Store page refs for cross-page navigation
st.session_state["_page_single_analysis"] = _single_analysis_page

inventory_df = get_inventory()
site_count = len(inventory_df) if inventory_df is not None and not inventory_df.empty else 0
render_dashboard_meta(site_count=site_count, plot_count=len(AVAILABLE_PLOTS))

# Background cache warmup — pre-fetch data so first user doesn't wait
if "warmup_started" not in st.session_state:
    st.session_state["warmup_started"] = True
    import threading

    def _warmup():
        try:
            from hydrology.data.usgs import fetch_current_conditions, fetch_daily_values
            from hydrology.data.usgs import DEFAULT_PARAM_DISCHARGE
            from datetime import datetime, timedelta

            priority = ["12422500", "12424000", "12419000"]
            fetch_current_conditions(priority)
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            for sid in priority:
                try:
                    fetch_daily_values(
                        sid,
                        param_cd=DEFAULT_PARAM_DISCHARGE,
                        start_date=start,
                        end_date=end,
                    )
                except Exception:
                    pass
        except Exception:
            pass

    threading.Thread(target=_warmup, daemon=True).start()

pg.run()

render_footer()
