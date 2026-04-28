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

# Background cache warmup — pre-fetch data so first user doesn't wait
if "warmup_started" not in st.session_state:
    st.session_state["warmup_started"] = True
    import threading

    def _warmup():
        try:
            from hydrology.app.shared import get_site_conditions
            from hydrology.data.usgs import fetch_current_conditions, fetch_daily_values
            from hydrology.data.usgs import DEFAULT_PARAM_DISCHARGE

            # Warm inventory (already called below, but ensures cache)
            inv = get_inventory()
            site_ids = inv['site_id'].tolist() if not inv.empty else []

            # Warm site conditions for map
            if site_ids:
                get_site_conditions(site_ids)

            # Warm priority sites
            from datetime import datetime, timedelta
            priority = ["12422500", "12424000", "12419000"]
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
_single_analysis_page = st.Page(single_analysis.show, title="Single Analysis", icon="📈", url_path="single-analysis")

pages = {
    "Dashboard": [
        st.Page(overview.show, title="Overview", icon="\U0001f4ca", default=True, url_path="overview"),
        _single_analysis_page,
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

# Store page refs for cross-page navigation
st.session_state["_page_single_analysis"] = _single_analysis_page

# Footer in sidebar
inventory_df = get_inventory()
st.sidebar.markdown("---")
site_count = len(inventory_df) if not inventory_df.empty else 0
st.sidebar.caption(f"Sites: {site_count} | Plots: {len(AVAILABLE_PLOTS)}")

pg.run()

render_footer()
