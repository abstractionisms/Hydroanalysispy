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

from hydrology.app.styles import apply_custom_css, render_footer, render_dashboard_hero, render_main_nav
from hydrology.app.shared import get_inventory
from hydrology.visualization.plots import AVAILABLE_PLOTS

# Page imports
from hydrology.app.page_modules import overview, single_analysis, comparisons, reach_analysis, watershed

apply_custom_css()
render_dashboard_hero(
    "HydroPlot analysis dashboard",
    "Find gages, analyze one site, compare records, evaluate reaches, and inspect basin context from one workspace.",
)
render_main_nav()

# Background cache warmup — pre-fetch data so first user doesn't wait
if "warmup_started" not in st.session_state:
    st.session_state["warmup_started"] = True
    import threading

    def _warmup():
        try:
            from hydrology.data.usgs import fetch_current_conditions, fetch_daily_values
            from hydrology.data.usgs import DEFAULT_PARAM_DISCHARGE

            # Warm priority sites
            from datetime import datetime, timedelta
            priority = ["12422500", "12424000", "12419000"]
            fetch_current_conditions(priority)
            end = datetime.now().strftime('%Y-%m-%d')
            start = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            for sid in priority:
                try:
                    fetch_daily_values(sid, param_cd=DEFAULT_PARAM_DISCHARGE,
                                       start_date=start, end_date=end)
                except Exception:
                    pass
        except Exception:
            pass  # Warmup is best-effort

    threading.Thread(target=_warmup, daemon=True).start()

# Define pages with grouping
_single_analysis_page = st.Page(single_analysis.show, title="Site Analysis", icon="📈", url_path="single-analysis")

pages = {
    "Workflows": [
        st.Page(overview.show, title="Stations", icon="\U0001f4ca", default=True, url_path="overview"),
        _single_analysis_page,
        st.Page(comparisons.show, title="Compare Sites", icon="\U0001f504", url_path="comparisons"),
        st.Page(reach_analysis.show, title="Reach Analysis", icon="\U0001f30a", url_path="reach-analysis"),
        st.Page(watershed.show, title="Watershed", icon="\U0001f5fa\ufe0f", url_path="watershed"),
    ],
}

pg = st.navigation(pages)

# Store page refs for cross-page navigation
st.session_state["_page_single_analysis"] = _single_analysis_page

# Footer in sidebar
inventory_df = get_inventory()
st.sidebar.markdown("---")
site_count = len(inventory_df) if not inventory_df.empty else 0
st.sidebar.caption(f"Sites: {site_count} | Plots: {len(AVAILABLE_PLOTS)}")

pg.run()

render_footer()
