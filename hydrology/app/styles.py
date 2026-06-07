"""
Custom styling and UI components for the Hydrology Dashboard.
"""

import streamlit as st
from typing import Dict, Any, Optional
import pandas as pd
from html import escape


def apply_custom_css():
    """Apply custom CSS for a polished dark theme."""
    st.markdown("""
    <style>
    :root {
        --hydro-bg: #08111f;
        --hydro-panel: #101c2e;
        --hydro-panel-2: #13283d;
        --hydro-border: rgba(118, 169, 192, 0.26);
        --hydro-text: #e7eef7;
        --hydro-muted: #8da2b8;
        --hydro-accent: #4ecdc4;
        --hydro-accent-2: #78a6ff;
        --hydro-warn: #fbbf24;
    }

    .stApp {
        background:
            radial-gradient(circle at 16% 8%, rgba(78, 205, 196, 0.10), transparent 26rem),
            radial-gradient(circle at 90% 12%, rgba(120, 166, 255, 0.10), transparent 30rem),
            linear-gradient(180deg, #08111f 0%, #0d1420 48%, #090f18 100%);
        color: var(--hydro-text);
    }

    /* Main container padding. Extra top clearance keeps Streamlit's deploy ribbon
       from covering the dashboard header on hosted and preview builds. */
    .main .block-container {
        padding-top: 4.25rem;
        padding-bottom: 2rem;
        max-width: 1480px;
    }

    /* Card-style containers */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 58, 95, 0.95) 0%, rgba(13, 27, 42, 0.95) 100%);
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid var(--hydro-border);
        margin-bottom: 0.5rem;
        box-shadow: 0 14px 34px rgba(0, 0, 0, 0.22);
    }

    .dashboard-hero {
        border: 1px solid var(--hydro-border);
        border-radius: 8px;
        padding: 1rem 1.1rem;
        margin: 0 0 1rem 0;
        background: linear-gradient(135deg, rgba(16, 28, 46, 0.92), rgba(12, 40, 55, 0.72));
        box-shadow: 0 16px 38px rgba(0, 0, 0, 0.28);
    }

    .dashboard-hero .eyebrow {
        color: var(--hydro-accent);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }

    .dashboard-hero h1 {
        font-size: 1.85rem;
        line-height: 1.15;
        margin: 0;
        color: var(--hydro-text);
    }

    .dashboard-hero p {
        margin: 0.4rem 0 0 0;
        color: var(--hydro-muted);
        max-width: 78ch;
    }

    .workflow-strip {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.55rem;
        margin: 0.75rem 0 1.1rem;
    }

    .workflow-step {
        border: 1px solid rgba(118, 169, 192, 0.20);
        background: rgba(12, 21, 34, 0.72);
        border-radius: 8px;
        padding: 0.65rem 0.75rem;
        min-height: 72px;
    }

    .workflow-step strong {
        display: block;
        color: var(--hydro-text);
        font-size: 0.86rem;
        margin-bottom: 0.15rem;
    }

    .workflow-step span {
        display: block;
        color: var(--hydro-muted);
        font-size: 0.76rem;
        line-height: 1.25;
    }

    .workspace-panel {
        border: 1px solid var(--hydro-border);
        border-radius: 8px;
        background: linear-gradient(180deg, rgba(10, 21, 36, 0.92), rgba(7, 17, 29, 0.88));
        padding: 0.95rem;
        box-shadow: 0 14px 34px rgba(0, 0, 0, 0.20);
    }

    .workspace-panel h3 {
        margin: 0 0 0.25rem 0;
        font-size: 1rem;
        color: var(--hydro-text);
    }

    .workspace-panel p {
        margin: 0;
        color: var(--hydro-muted);
        font-size: 0.82rem;
        line-height: 1.35;
    }

    .status-chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
        margin-top: 0.65rem;
    }

    .status-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        border-radius: 999px;
        padding: 0.22rem 0.55rem;
        font-size: 0.72rem;
        font-weight: 700;
        border: 1px solid rgba(118, 169, 192, 0.22);
        color: var(--hydro-text);
        background: rgba(255, 255, 255, 0.055);
    }

    .status-chip.ready {
        border-color: rgba(78, 205, 196, 0.48);
    }

    .status-chip.limited {
        border-color: rgba(255, 193, 7, 0.55);
    }

    .status-chip.blocked {
        border-color: rgba(255, 107, 107, 0.55);
    }

    .action-card-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 0.55rem;
        margin: 0.75rem 0;
    }

    .action-card {
        display: block;
        border: 1px solid rgba(118, 169, 192, 0.20);
        border-radius: 8px;
        padding: 0.75rem;
        background: rgba(12, 21, 34, 0.72);
        color: var(--hydro-text);
        text-decoration: none;
        min-height: 92px;
    }

    .action-card:hover {
        border-color: rgba(78, 205, 196, 0.62);
        background: rgba(78, 205, 196, 0.11);
        text-decoration: none;
        color: var(--hydro-text);
    }

    .action-card strong {
        display: block;
        font-size: 0.9rem;
        line-height: 1.2;
    }

    .action-card span {
        display: block;
        margin-top: 0.25rem;
        color: var(--hydro-muted);
        font-size: 0.76rem;
        line-height: 1.25;
    }

    .plot-board {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.65rem;
        margin: 0.75rem 0 1rem;
    }

    .plot-card {
        border: 1px solid rgba(118, 169, 192, 0.18);
        border-radius: 8px;
        padding: 0.72rem 0.78rem;
        background: rgba(12, 21, 34, 0.76);
        min-height: 112px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
    }

    .plot-card.ready {
        border-color: rgba(78, 205, 196, 0.35);
    }

    .plot-card.limited {
        border-color: rgba(251, 191, 36, 0.34);
    }

    .plot-card.blocked {
        border-color: rgba(248, 113, 113, 0.34);
        opacity: 0.84;
    }

    .plot-card strong {
        display: block;
        color: var(--hydro-text);
        font-size: 0.86rem;
        line-height: 1.2;
        margin-bottom: 0.22rem;
    }

    .plot-card span {
        color: var(--hydro-muted);
        font-size: 0.74rem;
        line-height: 1.25;
    }

    .plot-card .status {
        display: inline-block;
        margin-top: 0.55rem;
        padding: 0.12rem 0.45rem;
        border-radius: 999px;
        background: rgba(78, 205, 196, 0.12);
        color: var(--hydro-accent);
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }

    .plot-card.limited .status {
        background: rgba(251, 191, 36, 0.12);
        color: var(--hydro-warn);
    }

    .plot-card.blocked .status {
        background: rgba(248, 113, 113, 0.12);
        color: #f87171;
    }

    .insight-board {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.65rem;
        margin: 0.7rem 0 1.1rem;
    }

    .insight-card {
        background: rgba(15, 27, 44, 0.86);
        border: 1px solid rgba(118, 169, 192, 0.20);
        border-radius: 8px;
        padding: 0.8rem;
        min-height: 128px;
    }

    .insight-card.ready {
        border-top: 3px solid var(--hydro-accent);
    }

    .insight-card.limited {
        border-top: 3px solid var(--hydro-warn);
    }

    .insight-card.blocked {
        border-top: 3px solid #f87171;
    }

    .insight-card .label {
        color: var(--hydro-muted);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .insight-card .value {
        color: var(--hydro-text);
        font-size: 1.1rem;
        font-weight: 700;
        line-height: 1.2;
        margin-top: 0.22rem;
    }

    .insight-card .body {
        color: var(--hydro-muted);
        font-size: 0.78rem;
        line-height: 1.32;
        margin-top: 0.42rem;
    }

    /* Site header styling */
    .site-header {
        background: linear-gradient(90deg, rgba(78, 205, 196, 0.12), rgba(120, 166, 255, 0.06), transparent);
        padding: 0.85rem 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid var(--hydro-accent);
        border-top: 1px solid rgba(78, 205, 196, 0.18);
        border-bottom: 1px solid rgba(78, 205, 196, 0.10);
        border-radius: 8px;
    }

    .site-header h1 {
        margin: 0;
        color: var(--hydro-text);
        font-size: 1.42rem;
        font-weight: 600;
        letter-spacing: 0;
    }

    .site-header p {
        margin: 0.25rem 0 0 0;
        color: var(--hydro-muted);
        font-size: 0.85rem;
    }

    /* Data availability badges */
    .badge-available {
        background-color: #2d5a3d;
        color: #4ade80;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.8rem;
        display: inline-block;
        margin: 0.2rem;
    }

    .badge-unavailable {
        background-color: #5a2d2d;
        color: #f87171;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.8rem;
        display: inline-block;
        margin: 0.2rem;
    }

    .badge-warning {
        background-color: #5a4d2d;
        color: #fbbf24;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
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
        font-size: 1.75rem;
        color: var(--hydro-text);
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        color: var(--hydro-muted);
    }

    [data-testid="stMetric"] {
        background: rgba(16, 28, 46, 0.72);
        border: 1px solid rgba(118, 169, 192, 0.18);
        border-radius: 8px;
        padding: 0.75rem 0.85rem;
        min-height: 96px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
    }

    /* Sidebar is hidden so the app works from the main workspace instead of
       splitting navigation and controls into a secondary rail. */
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"] {
        display: none;
    }

    .main-nav {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        align-items: center;
        margin: 0.25rem 0 1rem 0;
        padding: 0.55rem;
        border: 1px solid var(--hydro-border);
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.035);
    }

    .main-nav a {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.42rem 0.7rem;
        border-radius: 7px;
        border: 1px solid rgba(118, 169, 192, 0.18);
        color: var(--hydro-text);
        text-decoration: none;
        font-size: 0.86rem;
        font-weight: 600;
        background: rgba(8, 18, 32, 0.55);
    }

    .main-nav a:hover {
        border-color: rgba(78, 205, 196, 0.65);
        background: rgba(78, 205, 196, 0.12);
        color: var(--hydro-text);
        text-decoration: none;
    }

    /*
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1524 0%, #07111d 100%);
        border-right: 1px solid rgba(118, 169, 192, 0.18);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {
        color: var(--hydro-text);
    }

    [data-testid="stSidebarNav"] {
        padding-top: 0.35rem;
    }

    [data-testid="stSidebarNav"] a {
        border-radius: 8px;
        margin: 0.12rem 0.35rem;
        color: var(--hydro-muted);
    }

    [data-testid="stSidebarNav"] a:hover {
        background: rgba(78, 205, 196, 0.10);
        color: var(--hydro-text);
    }
    */

    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
        border-color: rgba(78, 205, 196, 0.45);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }

    /* Plot container */
    .plot-container {
        background: rgba(14, 17, 23, 0.82);
        border: 1px solid rgba(118, 169, 192, 0.18);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    div[data-testid="stPlotlyChart"],
    iframe[title="streamlit_folium.st_folium"] {
        border-radius: 8px;
        overflow: hidden;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(118, 169, 192, 0.14);
        border-radius: 8px;
        overflow: hidden;
    }

    /* Footer styling */
    .footer-info {
        text-align: center;
        color: #6b7280;
        font-size: 0.8rem;
        padding: 1rem;
        border-top: 1px solid rgba(118, 169, 192, 0.18);
        margin-top: 2rem;
    }

    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main .block-container {
            padding-top: 3.5rem;
            padding-left: 0.5rem;
            padding-right: 0.5rem;
        }

        .workflow-strip {
            grid-template-columns: 1fr 1fr;
        }

        .plot-board {
            grid-template-columns: 1fr 1fr;
        }

        .insight-board {
            grid-template-columns: 1fr 1fr;
        }

        .dashboard-hero h1 {
            font-size: 1.35rem;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.2rem;
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.75rem;
        }

        .site-header h1 {
            font-size: 1.1rem;
        }

        .site-header p {
            font-size: 0.8rem;
        }

        .badge-available, .badge-unavailable, .badge-warning {
            font-size: 0.7rem;
            padding: 0.15rem 0.5rem;
        }
    }

    @media (max-width: 480px) {
        [data-testid="column"] {
            min-width: 100% !important;
        }

        [data-testid="stMetricValue"] {
            font-size: 1rem;
        }

        .site-header {
            padding: 0.3rem 0;
            padding-left: 0.75rem;
        }

        .workflow-strip {
            grid-template-columns: 1fr;
        }

        .plot-board {
            grid-template-columns: 1fr;
        }

        .insight-board {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """, unsafe_allow_html=True)


def render_dashboard_hero(title: str, subtitle: str):
    """Render a compact first-screen dashboard header."""
    st.markdown(f"""
    <div class="dashboard-hero">
        <div class="eyebrow">Pacific Northwest hydrology workspace</div>
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def render_main_nav():
    """Render main-page navigation so the sidebar is not the primary workflow."""
    st.markdown("""
    <nav class="main-nav" aria-label="Main workflow navigation">
        <a href="overview" target="_self">Stations</a>
        <a href="single-analysis" target="_self">Site Analysis</a>
        <a href="comparisons" target="_self">Compare Sites</a>
        <a href="reach-analysis" target="_self">Reach Analysis</a>
        <a href="watershed" target="_self">Watershed</a>
    </nav>
    """, unsafe_allow_html=True)


def render_workflow_strip():
    """Render visible navigation intent without replacing Streamlit navigation."""
    render_action_cards([
        {
            "title": "Stations",
            "href": "overview",
            "body": "Find gages, inspect map layers, live flow context, and record coverage.",
        },
        {
            "title": "Site Analysis",
            "href": "single-analysis",
            "body": "Run guided or fully custom plot sets for one selected gage.",
        },
        {
            "title": "Compare Sites",
            "href": "comparisons",
            "body": "Compare periods, compare gages, and inspect multi-site relationships.",
        },
        {
            "title": "Reach Analysis",
            "href": "reach-analysis",
            "body": "Evaluate paired gages, gain/loss patterns, and reach behavior.",
        },
        {
            "title": "Watershed",
            "href": "watershed",
            "body": "Inspect basin boundaries, dams, land cover, and HUC inventory context.",
        },
    ])


def render_workspace_panel(title: str, body: str, chips: list[dict] | None = None):
    """Render a reusable workspace panel."""
    chip_html = ""
    if chips:
        chip_html = '<div class="status-chip-row">' + "".join(
            f'<span class="status-chip {escape(str(chip.get("state", "ready")))}">{escape(str(chip["label"]))}</span>'
            for chip in chips
        ) + "</div>"
    st.markdown(
        f'<section class="workspace-panel"><h3>{escape(str(title))}</h3><p>{escape(str(body))}</p>{chip_html}</section>',
        unsafe_allow_html=True,
    )


def render_status_chips(chips: list[dict]):
    """Render standalone status chips."""
    if not chips:
        return
    st.markdown(
        '<div class="status-chip-row">' + "".join(
            f'<span class="status-chip {escape(str(chip.get("state", "ready")))}">{escape(str(chip["label"]))}</span>'
            for chip in chips
        ) + "</div>",
        unsafe_allow_html=True,
    )


def render_action_cards(cards: list[dict]):
    """Render action cards that link to app pages."""
    html = []
    for card in cards:
        href = escape(str(card["href"]), quote=True)
        title = escape(str(card["title"]))
        body = escape(str(card["body"]))
        html.append(
            f'<a class="action-card" href="{href}" target="_self">'
            f"<strong>{title}</strong><span>{body}</span></a>"
        )
    st.markdown('<div class="action-card-grid">' + "".join(html) + "</div>", unsafe_allow_html=True)


def render_plot_capability_board(cards: list[dict]):
    """Render compact plot capability cards."""
    if not cards:
        return

    html_cards = []
    for card in cards:
        state = card.get("state", "ready")
        title = card.get("title", "Plot")
        body = card.get("body", "")
        status = card.get("status", "Ready")
        html_cards.append(
            f'<div class="plot-card {state}">'
            f"<strong>{title}</strong>"
            f"<span>{body}</span>"
            f'<div class="status">{status}</div>'
            "</div>"
        )

    st.markdown(
        '<div class="plot-board">' + "".join(html_cards) + "</div>",
        unsafe_allow_html=True,
    )


def render_insight_board(cards):
    """Render data interpretation cards."""
    if not cards:
        return

    html_cards = []
    for card in cards:
        state = getattr(card, "state", "ready")
        title = getattr(card, "title", "")
        value = getattr(card, "value", "")
        body = getattr(card, "body", "")
        html_cards.append(
            f'<div class="insight-card {state}">'
            f'<div class="label">{title}</div>'
            f'<div class="value">{value}</div>'
            f'<div class="body">{body}</div>'
            "</div>"
        )

    st.markdown(
        '<div class="insight-board">' + "".join(html_cards) + "</div>",
        unsafe_allow_html=True,
    )


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
