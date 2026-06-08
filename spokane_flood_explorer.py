"""
Spokane County Flood Risk Explorer - Interactive Historical Dashboard

A companion to the ArcGIS Experience Builder app that provides
historical streamflow analysis with time controls for Spokane County
USGS gages.

Run with: streamlit run spokane_flood_explorer.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

from hydrology.data.usgs import (
    fetch_daily_values, fetch_peak_streamflow,
    DEFAULT_PARAM_DISCHARGE, DEFAULT_PARAM_STAGE
)
from hydrology.core.logging_setup import get_logger

logger = get_logger(__name__)

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Spokane Flood Explorer",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# SPOKANE COUNTY GAGES
# =============================================================================
SPOKANE_GAGES = {
    "12422500": {
        "name": "Spokane River at Spokane",
        "lat": 47.6588,
        "lon": -117.4260,
        "river": "Spokane River",
        "flood_stage_cfs": 25000,
        "action_stage_cfs": 18000,
        "color": "#00d4ff",
    },
    "12424000": {
        "name": "Hangman Creek at Spokane",
        "lat": 47.6436,
        "lon": -117.4039,
        "river": "Hangman Creek",
        "flood_stage_cfs": 5000,
        "action_stage_cfs": 3000,
        "color": "#ff6b35",
    },
    "12422000": {
        "name": "Spokane River above Liberty Bridge",
        "lat": 47.6750,
        "lon": -117.4310,
        "river": "Spokane River",
        "flood_stage_cfs": 24000,
        "action_stage_cfs": 17000,
        "color": "#7b68ee",
    },
    "12419000": {
        "name": "Spokane River nr Post Falls",
        "lat": 47.7178,
        "lon": -116.9817,
        "river": "Spokane River",
        "flood_stage_cfs": 30000,
        "action_stage_cfs": 20000,
        "color": "#00e676",
    },
    "12431000": {
        "name": "Little Spokane River at Dartford",
        "lat": 47.7583,
        "lon": -117.3917,
        "river": "Little Spokane River",
        "flood_stage_cfs": 3000,
        "action_stage_cfs": 2000,
        "color": "#ffab40",
    },
}

# =============================================================================
# DARK THEME CSS
# =============================================================================
st.markdown("""
<style>
    /* Dark theme overrides */
    .stApp {
        background-color: #0a0a1a;
        color: #e0e0e0;
    }
    .stSidebar {
        background-color: #0d1117;
    }
    .stSidebar .stMarkdown h1, .stSidebar .stMarkdown h2, .stSidebar .stMarkdown h3 {
        color: #00d4ff;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #e0e0e0 !important;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 5px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4ff, #7b68ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 5px;
    }
    .status-normal { color: #00e676; }
    .status-action { color: #ffab40; }
    .status-flood { color: #ff5252; }
    .gage-header {
        background: linear-gradient(90deg, #1a1a2e, #16213e);
        border-left: 4px solid #00d4ff;
        padding: 10px 15px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 10px;
    }
    .flood-event-row {
        background: #1a1a2e;
        border-left: 3px solid #ff5252;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 0 6px 6px 0;
        font-family: monospace;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 10px;
        padding: 15px;
    }
    div[data-testid="stMetric"] label {
        color: #8892b0 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #00d4ff !important;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CACHED DATA FUNCTIONS
# =============================================================================
@st.cache_data(ttl=3600, show_spinner="Fetching discharge data...")
def load_discharge(site_id: str, start: str, end: str) -> pd.DataFrame:
    """Fetch and cache daily discharge data."""
    df = fetch_daily_values(
        site_id,
        param_cd=DEFAULT_PARAM_DISCHARGE,
        start_date=start,
        end_date=end,
        chunk_years=10
    )
    if df is not None and not df.empty:
        df = df.rename(columns={"value": "discharge_cfs"})
        df = df[df["discharge_cfs"] > 0]
    return df


@st.cache_data(ttl=86400, show_spinner="Fetching peak flood data...")
def load_peaks(site_id: str) -> pd.DataFrame:
    """Fetch and cache peak streamflow data."""
    return fetch_peak_streamflow(site_id)


# =============================================================================
# PLOTLY DARK THEME
# =============================================================================
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(10, 10, 26, 0)",
    plot_bgcolor="rgba(26, 26, 46, 0.8)",
    font=dict(family="Inter, sans-serif", color="#e0e0e0"),
    margin=dict(l=60, r=20, t=50, b=40),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", showgrid=True),
    yaxis=dict(gridcolor="rgba(255,255,255,0.08)", showgrid=True),
    legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(255,255,255,0.1)"),
)


def create_hydrograph(data: pd.DataFrame, gage_info: dict, site_id: str) -> go.Figure:
    """Create an interactive hydrograph with flood thresholds."""
    fig = go.Figure()

    # Main discharge trace
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data["discharge_cfs"],
        mode="lines",
        name="Discharge",
        line=dict(color=gage_info["color"], width=1.5),
        fill="tozeroy",
        fillcolor=f"rgba({int(gage_info['color'][1:3], 16)}, "
                  f"{int(gage_info['color'][3:5], 16)}, "
                  f"{int(gage_info['color'][5:7], 16)}, 0.15)",
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Discharge: %{y:,.0f} cfs<extra></extra>",
    ))

    # Flood stage line
    fig.add_hline(
        y=gage_info["flood_stage_cfs"],
        line_dash="dash",
        line_color="#ff5252",
        annotation_text="Flood Stage",
        annotation_position="top left",
        annotation_font_color="#ff5252",
    )

    # Action stage line
    fig.add_hline(
        y=gage_info["action_stage_cfs"],
        line_dash="dot",
        line_color="#ffab40",
        annotation_text="Action Stage",
        annotation_position="top left",
        annotation_font_color="#ffab40",
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text=f"<b>{gage_info['name']}</b> — USGS {site_id}",
            font=dict(size=16, color=gage_info["color"]),
        ),
        yaxis_title="Discharge (cfs)",
        xaxis_title="",
        height=400,
        xaxis_rangeslider_visible=True,
        xaxis_rangeslider_thickness=0.06,
    )

    return fig


def create_comparison_chart(all_data: dict) -> go.Figure:
    """Create a multi-gage comparison chart."""
    fig = go.Figure()

    for site_id, (df, info) in all_data.items():
        if df is not None and not df.empty:
            # Normalize to percentage of flood stage for comparison
            normalized = (df["discharge_cfs"] / info["flood_stage_cfs"]) * 100

            fig.add_trace(go.Scatter(
                x=df.index,
                y=normalized,
                mode="lines",
                name=f"{info['name']}",
                line=dict(color=info["color"], width=1.5),
                hovertemplate=(
                    f"<b>{info['name']}</b><br>"
                    "%{x|%b %d, %Y}<br>"
                    "%{y:.0f}% of flood stage<br>"
                    "<extra></extra>"
                ),
            ))

    # 100% flood threshold
    fig.add_hline(
        y=100,
        line_dash="dash",
        line_color="#ff5252",
        annotation_text="Flood Stage (100%)",
        annotation_font_color="#ff5252",
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="<b>Multi-Gage Comparison</b> — % of Flood Stage", font=dict(size=16)),
        yaxis_title="% of Flood Stage",
        height=450,
        xaxis_rangeslider_visible=True,
        xaxis_rangeslider_thickness=0.06,
    )

    return fig


def create_annual_peak_chart(peaks: pd.DataFrame, gage_info: dict, site_id: str) -> go.Figure:
    """Create an annual peak flood chart."""
    if peaks.empty or "peak_discharge_cfs" not in peaks.columns:
        return None

    peaks_sorted = peaks.sort_values("peak_date")

    fig = go.Figure()

    # Color bars by severity
    colors = []
    for _, row in peaks_sorted.iterrows():
        q = row["peak_discharge_cfs"]
        if q >= gage_info["flood_stage_cfs"]:
            colors.append("#ff5252")
        elif q >= gage_info["action_stage_cfs"]:
            colors.append("#ffab40")
        else:
            colors.append(gage_info["color"])

    fig.add_trace(go.Bar(
        x=peaks_sorted["peak_date"],
        y=peaks_sorted["peak_discharge_cfs"],
        marker_color=colors,
        hovertemplate=(
            "<b>%{x|%Y}</b><br>"
            "Peak: %{y:,.0f} cfs<extra></extra>"
        ),
    ))

    fig.add_hline(
        y=gage_info["flood_stage_cfs"],
        line_dash="dash",
        line_color="#ff5252",
        annotation_text="Flood Stage",
        annotation_font_color="#ff5252",
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text=f"<b>Annual Peak Floods</b> — {gage_info['name']}",
            font=dict(size=16, color=gage_info["color"]),
        ),
        yaxis_title="Peak Discharge (cfs)",
        height=350,
    )

    return fig


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## 🌊 Spokane Flood Explorer")
    st.markdown("---")

    # Gage selection
    st.markdown("### Select Gages")
    selected_gages = {}
    for site_id, info in SPOKANE_GAGES.items():
        if st.checkbox(
            f"{info['name']}",
            value=(site_id in ["12422500", "12424000"]),
            key=f"chk_{site_id}"
        ):
            selected_gages[site_id] = info

    st.markdown("---")

    # Date range
    st.markdown("### Time Range")
    preset = st.selectbox("Quick Select", [
        "Last 1 Year",
        "Last 5 Years",
        "Last 10 Years",
        "Last 30 Years",
        "Custom Range",
    ], index=0)

    today = date.today()
    if preset == "Last 1 Year":
        start_date = today - timedelta(days=365)
        end_date = today
    elif preset == "Last 5 Years":
        start_date = today - timedelta(days=5 * 365)
        end_date = today
    elif preset == "Last 10 Years":
        start_date = today - timedelta(days=10 * 365)
        end_date = today
    elif preset == "Last 30 Years":
        start_date = today - timedelta(days=30 * 365)
        end_date = today
    else:
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start", today - timedelta(days=365 * 2))
        with col2:
            end_date = st.date_input("End", today)

    st.markdown("---")

    # View mode
    view_mode = st.radio("View Mode", [
        "Individual Gages",
        "Multi-Gage Comparison",
        "Flood History",
    ], index=0)

    st.markdown("---")
    st.markdown(
        '<div style="text-align:center; color:#555; font-size:0.75rem;">'
        'Data: USGS NWIS<br>'
        'Companion to ArcGIS Flood Risk Explorer'
        '</div>',
        unsafe_allow_html=True
    )


# =============================================================================
# MAIN CONTENT
# =============================================================================
st.markdown(
    '<h1 style="text-align:center; background: linear-gradient(135deg, #00d4ff, #7b68ee); '
    '-webkit-background-clip: text; -webkit-text-fill-color: transparent; '
    'font-size: 2.5rem; margin-bottom: 0;">Spokane County Flood Explorer</h1>',
    unsafe_allow_html=True
)
st.markdown(
    '<p style="text-align:center; color:#8892b0; margin-bottom: 30px;">'
    'Historical streamflow analysis for Spokane County USGS gages  •  '
    f'Showing {start_date.strftime("%b %Y")} — {end_date.strftime("%b %Y")}'
    '</p>',
    unsafe_allow_html=True
)

if not selected_gages:
    st.warning("Select at least one gage from the sidebar to get started.")
    st.stop()

# Fetch data for all selected gages
all_gage_data = {}
for site_id, info in selected_gages.items():
    df = load_discharge(site_id, start_date.isoformat(), end_date.isoformat())
    all_gage_data[site_id] = (df, info)

# =============================================================================
# GAGE MAP
# =============================================================================
st.markdown("### 📍 Gage Locations")

m = folium.Map(
    location=[47.66, -117.35],
    zoom_start=10,
    tiles="CartoDB dark_matter",
)

for site_id, info in selected_gages.items():
    df, _ = all_gage_data.get(site_id, (None, None))
    latest = "N/A"
    if df is not None and not df.empty:
        latest = f"{df['discharge_cfs'].iloc[-1]:,.0f} cfs"

    folium.CircleMarker(
        location=[info["lat"], info["lon"]],
        radius=10,
        color=info["color"],
        fill=True,
        fill_color=info["color"],
        fill_opacity=0.7,
        popup=folium.Popup(
            f"<b>{info['name']}</b><br>"
            f"USGS {site_id}<br>"
            f"Latest: {latest}<br>"
            f"River: {info['river']}",
            max_width=250
        ),
        tooltip=info["name"],
    ).add_to(m)

st_folium(m, width=None, height=300, returned_objects=[])

# =============================================================================
# METRIC CARDS
# =============================================================================
cols = st.columns(len(selected_gages))
for i, (site_id, (df, info)) in enumerate(all_gage_data.items()):
    with cols[i]:
        if df is not None and not df.empty:
            latest = df["discharge_cfs"].iloc[-1]
            avg = df["discharge_cfs"].mean()
            max_val = df["discharge_cfs"].max()

            # Determine status
            if latest >= info["flood_stage_cfs"]:
                status = "🔴 FLOOD"
            elif latest >= info["action_stage_cfs"]:
                status = "🟡 ACTION"
            else:
                status = "🟢 NORMAL"

            st.metric(
                label=info["name"],
                value=f"{latest:,.0f} cfs",
                delta=f"{status}",
            )
            st.caption(f"Avg: {avg:,.0f} | Max: {max_val:,.0f} cfs")
        else:
            st.metric(label=info["name"], value="No Data")

st.markdown("---")

# =============================================================================
# CHARTS
# =============================================================================
if view_mode == "Individual Gages":
    for site_id, (df, info) in all_gage_data.items():
        if df is not None and not df.empty:
            fig = create_hydrograph(df, info, site_id)
            st.plotly_chart(fig, use_container_width=True)

            # Quick stats expander
            with st.expander(f"📊 Statistics for {info['name']}"):
                stat_cols = st.columns(5)
                stat_cols[0].metric("Records", f"{len(df):,}")
                stat_cols[1].metric("Mean", f"{df['discharge_cfs'].mean():,.0f} cfs")
                stat_cols[2].metric("Median", f"{df['discharge_cfs'].median():,.0f} cfs")
                stat_cols[3].metric("Max", f"{df['discharge_cfs'].max():,.0f} cfs")
                stat_cols[4].metric("Min", f"{df['discharge_cfs'].min():,.0f} cfs")

                # Days above flood/action stage
                days_flood = (df["discharge_cfs"] >= info["flood_stage_cfs"]).sum()
                days_action = (df["discharge_cfs"] >= info["action_stage_cfs"]).sum()
                st.markdown(
                    f"**Days above flood stage:** {days_flood} &nbsp;|&nbsp; "
                    f"**Days above action stage:** {days_action}"
                )
        else:
            st.warning(f"No data available for {info['name']} (USGS {site_id})")

elif view_mode == "Multi-Gage Comparison":
    fig = create_comparison_chart(all_gage_data)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "*Each gage is normalized to its flood stage threshold so you can "
        "compare relative flood risk across different rivers.*"
    )

elif view_mode == "Flood History":
    st.markdown("### 🏔️ Historical Flood Events")
    st.markdown(
        "Annual peak streamflow records from USGS — "
        "the single highest instantaneous discharge each water year."
    )

    for site_id, info in selected_gages.items():
        peaks = load_peaks(site_id)

        if peaks is not None and not peaks.empty:
            fig = create_annual_peak_chart(peaks, info, site_id)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            # Top 5 floods table
            with st.expander(f"🏆 Top 10 Floods — {info['name']}"):
                top = peaks.nlargest(10, "peak_discharge_cfs")[
                    ["peak_date", "peak_discharge_cfs", "peak_gage_height_ft"]
                ].copy()
                top["peak_date"] = top["peak_date"].dt.strftime("%b %d, %Y")
                top["peak_discharge_cfs"] = top["peak_discharge_cfs"].apply(
                    lambda x: f"{x:,.0f} cfs"
                )
                top["peak_gage_height_ft"] = top["peak_gage_height_ft"].apply(
                    lambda x: f"{x:.1f} ft" if pd.notna(x) else "—"
                )
                top.columns = ["Date", "Peak Discharge", "Gage Height"]
                top = top.reset_index(drop=True)
                top.index = top.index + 1
                st.dataframe(top, use_container_width=True)
        else:
            st.info(f"No peak streamflow records found for {info['name']}")
