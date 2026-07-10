"""
Custom styling and UI components for the Hydrology Dashboard.
"""

import streamlit as st
from typing import Dict, Any, Optional
import pandas as pd
from html import escape


def _inject_css(css: str) -> None:
    """Inject CSS without dumping rules as visible page text.

    Streamlit sometimes sanitizes bare <style>/<link> in st.markdown and can
    leave raw CSS visible. Prefer st.html / components.html (height=0).
    """
    payload = f"<style>\n{css}\n</style>"
    # Streamlit >= 1.33
    if hasattr(st, "html"):
        try:
            st.html(payload)
            return
        except Exception:
            pass
    try:
        import streamlit.components.v1 as components

        components.html(payload, height=0, scrolling=False)
        return
    except Exception:
        pass
    # Last resort — may still work on older hosts
    st.markdown(payload, unsafe_allow_html=True)


def apply_custom_css():
    """Apply custom CSS for a refined dark theme with premium fluid motion.

    Design language: clean, sleek, detailed — dense where useful, airy where
    maps/plots need space. Motion is short and deliberate so Streamlit reruns
    stay snappy (not gimmicky loops).

    Motion (subtle, performant, replay-safe under Streamlit reruns):
    - Short fade + lift entrances on cards/panels.
    - Hover lifts + accent intensification on actionable surfaces.
    - Button transitions + active feedback.
    - Status chip polish.
    - Respects prefers-reduced-motion.

    Inspiration: elevated animated "award-winning" web UI patterns (see
    https://x.com/twetsfyp/status/2065283731833651709 for the reference video
    of modern fluid website design that informed the polish direction).
    """
    css = """
    @import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap");

    :root {
        --hydro-bg: #070d16;
        --hydro-panel: #0c1624;
        --hydro-panel-2: #101f33;
        --hydro-border: rgba(125, 170, 200, 0.18);
        --hydro-text: #edf3fa;
        --hydro-muted: #8b9cb0;
        --hydro-accent: #3dd6c6;
        --hydro-accent-2: #6b9fff;
        --hydro-warn: #f0b429;
        --hydro-danger: #f07178;
        --hydro-radius: 12px;
        --hydro-radius-sm: 8px;
        --hydro-shadow: 0 12px 40px rgba(0, 0, 0, 0.35);
        --hydro-font: "Inter", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
        --hydro-mono: "JetBrains Mono", ui-monospace, "SF Mono", Consolas, monospace;
    }

    /* Premium fluid motion keyframes — award-winning feel, short durations for snappy Streamlit. */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes cardPop {
        from { opacity: 0; transform: translateY(6px) scale(0.99); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }
    @keyframes subtlePulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(61, 214, 198, 0.0); }
        50% { box-shadow: 0 0 0 6px rgba(61, 214, 198, 0.08); }
    }
    @keyframes shimmer {
        0% { transform: translateX(-120%); }
        100% { transform: translateX(220%); }
    }
    @keyframes accentShift {
        0%, 100% { border-color: rgba(125, 170, 200, 0.18); }
        50% { border-color: rgba(61, 214, 198, 0.35); }
    }

    html, body, [class*="css"] {
        font-family: var(--hydro-font);
    }

    .stApp {
        background:
            radial-gradient(ellipse 80% 50% at 10% -10%, rgba(61, 214, 198, 0.09), transparent 50%),
            radial-gradient(ellipse 60% 40% at 100% 0%, rgba(107, 159, 255, 0.08), transparent 45%),
            radial-gradient(ellipse 50% 30% at 50% 100%, rgba(61, 214, 198, 0.04), transparent 50%),
            linear-gradient(180deg, #070d16 0%, #0a121e 55%, #060b13 100%);
        color: var(--hydro-text);
        letter-spacing: -0.01em;
    }

    /* Hide Streamlit chrome that fights the product shell */
    #MainMenu { visibility: hidden; }
    header[data-testid="stHeader"] {
        background: transparent;
        border: none;
    }
    div[data-testid="stToolbar"] { display: none; }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: rgba(125, 170, 200, 0.28);
        border-radius: 999px;
    }
    ::-webkit-scrollbar-thumb:hover { background: rgba(61, 214, 198, 0.45); }

    /* Main container — top clearance for hosted deploy ribbon / toolbar ghost */
    .main .block-container {
        padding-top: 3.5rem;
        padding-bottom: 2.5rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        max-width: 1440px;
    }

    h1, h2, h3, h4 {
        font-family: var(--hydro-font);
        letter-spacing: -0.02em;
        font-weight: 600;
        color: var(--hydro-text);
    }

    p, label, .stMarkdown {
        color: var(--hydro-text);
    }

    code, pre, .stCode {
        font-family: var(--hydro-mono) !important;
    }

    /* Card-style containers */
    .metric-card {
        background: linear-gradient(160deg, rgba(16, 31, 51, 0.95) 0%, rgba(10, 18, 30, 0.98) 100%);
        border-radius: var(--hydro-radius);
        padding: 1rem 1.1rem;
        border: 1px solid var(--hydro-border);
        margin-bottom: 0.5rem;
        box-shadow: var(--hydro-shadow);
        backdrop-filter: blur(8px);
    }

    .dashboard-hero {
        position: relative;
        overflow: hidden;
        border: 1px solid var(--hydro-border);
        border-radius: var(--hydro-radius);
        padding: 1.15rem 1.35rem 1.2rem;
        margin: 0 0 0.85rem 0;
        background:
            linear-gradient(135deg, rgba(12, 24, 40, 0.96), rgba(8, 28, 42, 0.88));
        box-shadow: var(--hydro-shadow);
        animation: fadeInUp 0.28s cubic-bezier(0.2, 0.8, 0.2, 1) both;
        will-change: transform;
    }

    .dashboard-hero::before {
        content: "";
        position: absolute;
        inset: 0 auto 0 0;
        width: 3px;
        background: linear-gradient(180deg, var(--hydro-accent), var(--hydro-accent-2));
        border-radius: 3px 0 0 3px;
    }

    .dashboard-hero .eyebrow {
        color: var(--hydro-accent);
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
        opacity: 0.95;
    }

    .dashboard-hero h1 {
        font-size: clamp(1.45rem, 2.2vw, 1.85rem);
        line-height: 1.15;
        margin: 0;
        color: var(--hydro-text);
        font-weight: 700;
        letter-spacing: -0.03em;
    }

    .dashboard-hero p {
        margin: 0.45rem 0 0 0;
        color: var(--hydro-muted);
        max-width: 72ch;
        font-size: 0.92rem;
        line-height: 1.45;
        font-weight: 400;
    }

    .dashboard-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin: 0.15rem 0 0.9rem;
    }

    .dashboard-meta .meta-pill {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        color: var(--hydro-muted);
        border: 1px solid var(--hydro-border);
        background: rgba(255,255,255,0.03);
        border-radius: 999px;
        padding: 0.28rem 0.65rem;
        font-family: var(--hydro-mono);
        white-space: nowrap;
    }

    .dashboard-meta .meta-pill + .meta-pill {
        margin-left: 0; /* flex gap handles spacing; avoid sticky paste */
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
        border-radius: var(--hydro-radius);
        background: linear-gradient(180deg, rgba(12, 22, 36, 0.94), rgba(8, 15, 26, 0.92));
        padding: 1rem 1.1rem;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.22);
        animation: fadeInUp 0.22s ease-out both;
        will-change: transform;
        backdrop-filter: blur(6px);
    }

    .workspace-panel h3 {
        margin: 0 0 0.3rem 0;
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--hydro-text);
        letter-spacing: -0.015em;
    }

    .workspace-panel p {
        margin: 0;
        color: var(--hydro-muted);
        font-size: 0.82rem;
        line-height: 1.4;
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
        transition: transform 0.12s ease, border-color 0.12s ease, background 0.12s ease;
    }

    .status-chip.ready {
        border-color: rgba(78, 205, 196, 0.48);
    }

    .status-chip:hover {
        transform: scale(1.02);
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
        border: 1px solid rgba(125, 170, 200, 0.16);
        border-radius: var(--hydro-radius);
        padding: 0.85rem 0.9rem;
        background: rgba(10, 18, 30, 0.82);
        color: var(--hydro-text);
        text-decoration: none;
        min-height: 96px;
        transition: transform 0.18s cubic-bezier(0.2, 0.8, 0.2, 1),
                    box-shadow 0.18s ease,
                    border-color 0.18s ease,
                    background 0.18s ease;
        animation: cardPop 0.24s cubic-bezier(0.2, 0.8, 0.2, 1) both;
        will-change: transform;
    }

    .action-card:hover {
        border-color: rgba(61, 214, 198, 0.5);
        background: rgba(61, 214, 198, 0.08);
        transform: translateY(-2px);
        box-shadow: 0 14px 32px rgba(0, 0, 0, 0.32);
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
        border: 1px solid rgba(125, 170, 200, 0.14);
        border-radius: var(--hydro-radius-sm);
        padding: 0.78rem 0.85rem;
        background: rgba(10, 18, 30, 0.88);
        min-height: 112px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
        transition: transform 0.16s cubic-bezier(0.2, 0, 0, 1),
                    box-shadow 0.16s ease,
                    border-color 0.16s ease;
        animation: cardPop 0.22s ease-out both;
        will-change: transform;
    }

    .plot-card.ready {
        border-color: rgba(61, 214, 198, 0.32);
    }

    .plot-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.28), inset 0 1px 0 rgba(255,255,255,0.04);
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
        background: rgba(12, 22, 36, 0.92);
        border: 1px solid rgba(125, 170, 200, 0.14);
        border-radius: var(--hydro-radius);
        padding: 0.9rem;
        min-height: 128px;
        transition: transform 0.16s cubic-bezier(0.2, 0, 0, 1),
                    box-shadow 0.16s ease,
                    border-color 0.16s ease;
        animation: fadeInUp 0.22s ease-out both;
        will-change: transform;
    }

    .insight-card.ready {
        border-top: 2px solid var(--hydro-accent);
    }

    .insight-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.26);
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
        background: linear-gradient(100deg, rgba(61, 214, 198, 0.10), rgba(107, 159, 255, 0.05), transparent 70%);
        padding: 0.95rem 1.1rem;
        margin-bottom: 1rem;
        border-left: 3px solid var(--hydro-accent);
        border: 1px solid rgba(61, 214, 198, 0.14);
        border-left-width: 3px;
        border-radius: var(--hydro-radius);
        animation: fadeInUp 0.2s ease-out both;
    }

    .site-header h1 {
        margin: 0;
        color: var(--hydro-text);
        font-size: 1.35rem;
        font-weight: 600;
        letter-spacing: -0.02em;
    }

    .site-header p {
        margin: 0.3rem 0 0 0;
        color: var(--hydro-muted);
        font-size: 0.82rem;
        font-family: var(--hydro-mono);
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
        font-size: 1.55rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: var(--hydro-text);
        font-variant-numeric: tabular-nums;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: var(--hydro-muted);
    }

    [data-testid="stMetric"] {
        background: rgba(12, 22, 36, 0.88);
        border: 1px solid rgba(125, 170, 200, 0.14);
        border-radius: var(--hydro-radius);
        padding: 0.85rem 0.95rem;
        min-height: 92px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
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
        gap: 0.35rem;
        align-items: center;
        margin: 0 0 1rem 0;
        padding: 0.4rem;
        border: 1px solid var(--hydro-border);
        border-radius: 999px;
        background: rgba(8, 14, 24, 0.72);
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.2);
        animation: fadeInUp 0.2s ease-out both;
    }

    .main-nav a {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.48rem 0.9rem;
        border-radius: 999px;
        border: 1px solid transparent;
        color: var(--hydro-muted);
        text-decoration: none;
        font-size: 0.84rem;
        font-weight: 600;
        background: transparent;
        transition: transform 0.14s ease, border-color 0.14s ease, background 0.14s ease, color 0.14s ease;
    }

    .main-nav a:hover {
        border-color: rgba(61, 214, 198, 0.35);
        background: rgba(61, 214, 198, 0.10);
        transform: translateY(-1px);
        color: var(--hydro-text);
        text-decoration: none;
    }

    .main-nav a.active {
        color: var(--hydro-bg);
        background: linear-gradient(135deg, var(--hydro-accent), #2bb8ab);
        border-color: transparent;
        box-shadow: 0 4px 14px rgba(61, 214, 198, 0.25);
    }

    .main-nav a:active {
        transform: translateY(0) scale(0.985);
    }

    /* Streamlit widgets — refined controls */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    .stTextInput input,
    .stNumberInput input,
    .stDateInput input {
        background-color: rgba(8, 14, 24, 0.9) !important;
        border-color: rgba(125, 170, 200, 0.22) !important;
        border-radius: var(--hydro-radius-sm) !important;
        color: var(--hydro-text) !important;
    }

    .stSelectbox label, .stMultiSelect label, .stTextInput label,
    .stNumberInput label, .stDateInput label, .stSlider label {
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.03em !important;
        color: var(--hydro-muted) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem;
        background: rgba(8, 14, 24, 0.55);
        border-radius: 999px;
        padding: 0.3rem;
        border: 1px solid var(--hydro-border);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        padding: 0.4rem 0.9rem;
        color: var(--hydro-muted);
        font-weight: 600;
        font-size: 0.84rem;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(61, 214, 198, 0.14) !important;
        color: var(--hydro-text) !important;
    }

    .stExpander {
        border: 1px solid var(--hydro-border) !important;
        border-radius: var(--hydro-radius) !important;
        background: rgba(10, 18, 30, 0.55) !important;
    }

    div[data-testid="stAlert"] {
        border-radius: var(--hydro-radius-sm);
        border: 1px solid var(--hydro-border);
        background: rgba(12, 22, 36, 0.9);
    }

    .stProgress > div > div {
        background: linear-gradient(90deg, var(--hydro-accent), var(--hydro-accent-2)) !important;
    }

    .stSpinner > div {
        border-top-color: var(--hydro-accent) !important;
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

    /* Button styling — refined CTAs (subtlePulse kept for polish, not circus) */
    .stButton > button {
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.88rem;
        letter-spacing: -0.01em;
        transition: transform 0.15s cubic-bezier(0.2, 0, 0, 1),
                    box-shadow 0.15s ease,
                    border-color 0.15s ease,
                    background 0.15s ease;
        border-color: rgba(61, 214, 198, 0.35) !important;
        background: rgba(10, 18, 30, 0.9) !important;
        color: var(--hydro-text) !important;
        min-height: 2.5rem;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.32);
        border-color: rgba(61, 214, 198, 0.7) !important;
    }

    .stButton > button[kind="primary"],
    .stButton > button[type="primary"],
    .stButton > button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #3dd6c6 0%, #2bb8ab 55%, #2499d6 100%) !important;
        color: #061018 !important;
        border: none !important;
        box-shadow: 0 6px 18px rgba(61, 214, 198, 0.22);
        animation: subtlePulse 2.8s ease-in-out infinite, fadeInUp 0.25s ease-out;
        position: relative;
        overflow: hidden;
    }

    .stButton > button[kind="primary"]::after,
    .stButton > button[type="primary"]::after,
    .stButton > button[data-testid="baseButton-primary"]::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 40%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.22), transparent);
        animation: shimmer 2.4s ease-in-out infinite;
        pointer-events: none;
    }

    .stButton > button[kind="primary"]:hover,
    .stButton > button[type="primary"]:hover,
    .stButton > button[data-testid="baseButton-primary"]:hover {
        box-shadow: 0 10px 24px rgba(61, 214, 198, 0.32);
        transform: translateY(-2px);
        filter: brightness(1.04);
    }

    .stButton > button:active {
        transform: translateY(0) scale(0.98);
    }

    .stButton > button:focus-visible {
        outline: 2px solid rgba(61, 214, 198, 0.55);
        outline-offset: 2px;
    }

    /* Plot / map surfaces */
    .plot-container {
        background: rgba(8, 14, 24, 0.88);
        border: 1px solid rgba(125, 170, 200, 0.14);
        border-radius: var(--hydro-radius);
        padding: 1rem;
        margin-bottom: 1rem;
    }

    div[data-testid="stPlotlyChart"],
    iframe[title="streamlit_folium.st_folium"] {
        border-radius: var(--hydro-radius);
        overflow: hidden;
        border: 1px solid rgba(125, 170, 200, 0.12);
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(125, 170, 200, 0.12);
        border-radius: var(--hydro-radius);
        overflow: hidden;
    }

    /* Footer styling */
    .footer-info {
        text-align: center;
        color: #6b7a8c;
        font-size: 0.76rem;
        padding: 1.25rem 1rem 0.5rem;
        border-top: 1px solid rgba(125, 170, 200, 0.12);
        margin-top: 2.25rem;
        letter-spacing: 0.01em;
    }

    .footer-info a {
        color: var(--hydro-accent);
        text-decoration: none;
        font-weight: 600;
    }

    .footer-info a:hover {
        text-decoration: underline;
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

    /* Respect user motion preference and add subtle global polish */
    @media (prefers-reduced-motion: reduce) {
        .dashboard-hero,
        .workspace-panel,
        .action-card,
        .plot-card,
        .insight-card,
        .status-chip,
        .main-nav a,
        .stButton > button {
            animation: none !important;
            transition: none !important;
            transform: none !important;
        }
    }

    /* Extra container polish for Plotly/folium (map & chart surfaces feel premium) */
    div[data-testid="stPlotlyChart"],
    iframe[title="streamlit_folium.st_folium"] {
        transition: box-shadow 0.2s ease, border-color 0.2s ease;
    }

    div[data-testid="stPlotlyChart"]:hover,
    iframe[title="streamlit_folium.st_folium"]:hover {
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.22);
    }
    """
    _inject_css(css)


def render_dashboard_hero(title: str, subtitle: str):
    """Render a compact first-screen dashboard header."""
    st.markdown(f"""
    <div class="dashboard-hero">
        <div class="eyebrow">HydroPlot · Pacific Northwest</div>
        <h1>{escape(str(title))}</h1>
        <p>{escape(str(subtitle))}</p>
    </div>
    """, unsafe_allow_html=True)


def render_dashboard_meta(site_count: int = 0, plot_count: int = 0):
    """Compact inventory strip under the hero (snappier than sidebar captions)."""
    bits = []
    if site_count:
        bits.append(f'<span class="meta-pill">{int(site_count):,} gages</span>')
    if plot_count:
        bits.append(f'<span class="meta-pill">{int(plot_count)} plot types</span>')
    bits.append('<span class="meta-pill">USGS · Meteostat · NLDI</span>')
    st.markdown(
        '<div class="dashboard-meta">' + "".join(bits) + "</div>",
        unsafe_allow_html=True,
    )


def _nav_active_path() -> str:
    """Best-effort active page path for pill highlight."""
    try:
        path = (getattr(st, "context", None) and getattr(st.context, "url_path", None)) or ""
        path = str(path).strip("/")
        if path:
            return path.split("/")[-1]
    except Exception:
        pass
    # Fallback: Streamlit multipage often leaves last path fragment in session
    for key in ("_hydro_active_page", "page"):
        if key in st.session_state and st.session_state[key]:
            return str(st.session_state[key]).strip("/")
    return "overview"


def render_main_nav(active: str | None = None):
    """Render main-page navigation so the sidebar is not the primary workflow."""
    active = (active or _nav_active_path() or "overview").strip("/")
    items = [
        ("overview", "Stations"),
        ("single-analysis", "Site Analysis"),
        ("comparisons", "Compare Sites"),
        ("reach-analysis", "Reach Analysis"),
        ("watershed", "Watershed"),
    ]
    links = []
    for path, label in items:
        cls = "active" if active == path or active.endswith(path) else ""
        links.append(
            f'<a class="{cls}" href="{path}" target="_self">{escape(label)}</a>'
        )
    st.markdown(
        '<nav class="main-nav" aria-label="Main workflow navigation">'
        + "".join(links)
        + "</nav>",
        unsafe_allow_html=True,
    )


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
        <p><strong>HydroPlot</strong> · USGS NWIS · Meteostat · NLDI · NWM
        · <a href="https://github.com/abstractionisms/Hydroanalysispy" target="_blank">GitHub</a>
        · Built for clean gage-to-reach workflows</p>
    </div>
    """, unsafe_allow_html=True)
