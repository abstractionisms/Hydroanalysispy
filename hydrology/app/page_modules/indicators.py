"""
Indicators page - Standardized drought and climate indicators.

Displays SPI/SRI time series with drought severity bands,
current drought status, BFI trends, and seasonal anomalies.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hydrology.app.shared import (
    get_inventory, get_cached_site_info,
    site_picker, logger)
from hydrology.app.interpretation import InsightCard, describe_standardized_index
from hydrology.app.styles import render_insight_board
from hydrology.data.usgs import fetch_daily_values, DEFAULT_PARAM_DISCHARGE
from hydrology.analysis.indicators import (
    calculate_spi, calculate_sri, classify_drought,
    calculate_baseflow_index_timeseries, get_seasonal_anomaly)


def show():
    """Render the Indicators page."""
    st.header("Hydrological Indicators")
    st.caption("Standardized drought indices, baseflow trends, and seasonal anomalies")

    inventory_df = get_inventory()
    if inventory_df.empty:
        st.error("Could not load site inventory")
        return

    st.subheader("Indicator Workspace")
    site_id = site_picker(inventory_df, key="indicators", label="Site", location="main")

    site_info = get_cached_site_info(site_id)
    if not site_info:
        st.error(f"Site {site_id} not found")
        return

    desc = site_info.get('description', site_id)
    lat = site_info.get('latitude')
    lon = site_info.get('longitude')
    st.caption(f"Selected `{site_id}` - {desc}")

    years_back = st.slider("Years of data", 5, 30, 15, key="ind_years")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_back * 365)
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    # Fetch discharge data
    with st.spinner("Loading discharge data..."):
        df_q = fetch_daily_values(
            site_id, param_cd='00060',
            start_date=start_str, end_date=end_str
        )

    if df_q is None or df_q.empty:
        st.error("No discharge data available for this site/period")
        return

    q_col = 'value' if 'value' in df_q.columns else df_q.columns[0]

    tab_drought, tab_bfi, tab_anomaly = st.tabs([
        "Drought Indices (SRI)",
        "Baseflow Index",
        "Seasonal Anomaly",
    ])

    with tab_drought:
        _render_drought_tab(df_q, q_col, site_id, desc, lat, lon, start_str, end_str)

    with tab_bfi:
        _render_bfi_tab(df_q, q_col, desc)

    with tab_anomaly:
        _render_anomaly_tab(df_q, q_col, desc)


def _render_drought_tab(df_q, q_col, site_id, desc, lat, lon, start_str, end_str):
    """Render SRI drought indices with severity bands."""
    st.subheader("Standardized Runoff Index (SRI)")
    st.caption("Gamma-fitted streamflow anomalies. Negative = drier than normal.")

    # Calculate SRI
    sri_df = calculate_sri(df_q[q_col], windows=[1, 3, 6])

    if sri_df.empty:
        st.warning("Insufficient data for SRI calculation (need 10+ years of monthly data)")
        return

    # Current drought status
    _render_drought_status_cards(sri_df)
    _render_index_interpretation(sri_df, "SRI", "streamflow")

    # SRI time series with drought bands
    fig = _create_drought_timeseries(sri_df, title=f"{desc} - Standardized Runoff Index")
    st.plotly_chart(fig, width="stretch", key="sri_chart")

    # SPI if climate data available
    st.markdown("---")
    st.subheader("Standardized Precipitation Index (SPI)")
    show_spi = st.checkbox("Calculate SPI (requires climate data)", value=False, key="show_spi")

    if show_spi:
        with st.spinner("Fetching climate data..."):
            precip_data = _fetch_precip_data(site_id, lat, lon, start_str, end_str)

        if precip_data is not None and not precip_data.empty:
            spi_df = calculate_spi(precip_data, windows=[1, 3, 6, 12])
            if not spi_df.empty:
                _render_index_interpretation(spi_df, "SPI", "precipitation")
                fig_spi = _create_drought_timeseries(spi_df, title=f"{desc} - SPI")
                st.plotly_chart(fig_spi, width="stretch", key="spi_chart")
            else:
                st.warning("Insufficient precipitation data for SPI. Try a longer period or a site with stronger climate coverage.")
        else:
            st.warning(
                "Could not fetch precipitation data. SPI uses Daymet first and then the nearest Meteostat station from the selected site coordinates."
            )


def _render_drought_status_cards(sri_df: pd.DataFrame):
    """Show current drought status as metric cards."""
    cols = st.columns(len(sri_df.columns))

    for i, col_name in enumerate(sri_df.columns):
        latest = sri_df[col_name].dropna()
        if latest.empty:
            continue
        val = latest.iloc[-1]
        status = classify_drought(val)

        with cols[i]:
            window = col_name.split('_')[1]
            st.metric(
                f"{window}-Month SRI",
                f"{val:+.2f}",
                delta=status['label'],
                delta_color="off")


def _render_index_interpretation(index_df: pd.DataFrame, label: str, subject: str):
    """Show plain-language interpretation for latest standardized indices."""
    cards = []
    for col_name in index_df.columns:
        latest = index_df[col_name].dropna()
        if latest.empty:
            continue
        value = float(latest.iloc[-1])
        meaning, body = describe_standardized_index(value)
        window = col_name.split('_')[1] if '_' in col_name else col_name
        cards.append(
            InsightCard(
                f"{window}-Month {label}",
                f"{value:+.2f}",
                f"{meaning}. {body} This summarizes {subject} over the selected record.",
                "limited" if abs(value) >= 1.3 else "ready",
            )
        )
    if cards:
        st.caption(f"Interpretation of latest {label} values")
        render_insight_board(cards)


def _create_drought_timeseries(index_df: pd.DataFrame, title: str) -> go.Figure:
    """Create drought index timeseries with colored severity bands."""
    fig = go.Figure()

    # Drought severity bands (background)
    band_defs = [
        (-2.0, -1.6, 'rgba(230, 0, 0, 0.15)', 'D3-D4: Extreme/Exceptional'),
        (-1.6, -1.3, 'rgba(255, 170, 0, 0.15)', 'D2: Severe'),
        (-1.3, -0.8, 'rgba(252, 211, 127, 0.15)', 'D1: Moderate'),
        (-0.8, -0.5, 'rgba(255, 255, 0, 0.10)', 'D0: Abnormally Dry'),
    ]

    x_range = [index_df.index.min(), index_df.index.max()]

    for y0, y1, color, label in band_defs:
        fig.add_shape(
            type="rect",
            x0=x_range[0], x1=x_range[1],
            y0=y0, y1=y1,
            fillcolor=color,
            line=dict(width=0),
            layer="below")

    # Zero line
    fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1)
    fig.add_hline(y=-0.5, line_dash="dot", line_color="orange", line_width=0.5, opacity=0.5)
    fig.add_hline(y=-1.3, line_dash="dot", line_color="red", line_width=0.5, opacity=0.5)

    # Plot each window
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, col in enumerate(index_df.columns):
        series = index_df[col].dropna()
        if series.empty:
            continue

        fig.add_trace(go.Scatter(
            x=series.index, y=series.values,
            mode='lines', name=col,
            line=dict(color=colors[i % len(colors)], width=2),
            hovertemplate=f'{col}<br>%{{x|%Y-%m}}<br>Value: %{{y:+.2f}}<extra></extra>'))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Index Value",
        yaxis=dict(range=[-3, 3]),
        height=450,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=60, b=40))

    # Add drought severity labels on the right
    annotations = [
        dict(x=1.02, y=-2.3, text="D4", font=dict(color='#730000', size=9)),
        dict(x=1.02, y=-1.8, text="D3", font=dict(color='#E60000', size=9)),
        dict(x=1.02, y=-1.45, text="D2", font=dict(color='#FFAA00', size=9)),
        dict(x=1.02, y=-1.05, text="D1", font=dict(color='#FCD37F', size=9)),
        dict(x=1.02, y=-0.65, text="D0", font=dict(color='#CCCC00', size=9)),
    ]
    for ann in annotations:
        fig.add_annotation(
            xref="paper", yref="y",
            showarrow=False, **ann
        )

    return fig


def _render_bfi_tab(df_q, q_col, desc):
    """Render Baseflow Index trend analysis."""
    st.subheader("Baseflow Index (BFI) Trend")
    st.caption(
        "Ratio of baseflow to total flow. Declining BFI may indicate "
        "reduced groundwater contribution or aquifer depletion."
    )

    window = st.slider("Rolling window (days)", 30, 365, 90, step=30, key="bfi_window")

    bfi_df = calculate_baseflow_index_timeseries(df_q[q_col], window_days=window)

    if bfi_df.empty:
        st.warning("Insufficient data for BFI analysis")
        return

    # Current BFI
    current_bfi = bfi_df['bfi'].dropna().iloc[-1] if not bfi_df['bfi'].dropna().empty else None

    col1, col2, col3 = st.columns(3)
    with col1:
        if current_bfi is not None:
            st.metric("Current BFI", f"{current_bfi:.2f}",
                         help="Baseflow Index: fraction of flow from groundwater vs. surface runoff. 0-1, higher = more groundwater")
    with col2:
        mean_bfi = bfi_df['bfi'].mean()
        st.metric("Mean BFI", f"{mean_bfi:.2f}",
                         help="Long-term average Baseflow Index. Higher = greater groundwater contribution")
    with col3:
        if current_bfi is not None:
            delta = current_bfi - mean_bfi
            st.metric("vs Mean", f"{delta:+.2f}",
                     delta="Above" if delta > 0 else "Below",
                     delta_color="normal" if delta > 0 else "inverse")

    # BFI time series
    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.6, 0.4],
        shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=["Baseflow Separation", f"Rolling {window}-Day BFI"])

    # Baseflow + quickflow stacked area
    fig.add_trace(go.Scatter(
        x=bfi_df.index, y=bfi_df['baseflow'],
        mode='lines', name='Baseflow',
        line=dict(width=0), fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.4)',
        hovertemplate='%{x|%Y-%m-%d}<br>Baseflow: %{y:,.0f} cfs<extra></extra>'), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=bfi_df.index, y=bfi_df['total_flow'],
        mode='lines', name='Total Flow',
        line=dict(color='#1f77b4', width=1),
        fill='tonexty', fillcolor='rgba(255, 127, 14, 0.3)',
        hovertemplate='%{x|%Y-%m-%d}<br>Total: %{y:,.0f} cfs<extra></extra>'), row=1, col=1)

    # BFI line
    bfi_series = bfi_df['bfi'].dropna()
    fig.add_trace(go.Scatter(
        x=bfi_series.index, y=bfi_series.values,
        mode='lines', name='BFI',
        line=dict(color='#2ca02c', width=2),
        hovertemplate='%{x|%Y-%m-%d}<br>BFI: %{y:.2f}<extra></extra>'), row=2, col=1)

    # Mean BFI reference line
    fig.add_hline(y=mean_bfi, line_dash="dash", line_color="gray",
                  row=2, col=1, annotation_text=f"Mean: {mean_bfi:.2f}")

    fig.update_layout(
        title=f"{desc} - Baseflow Analysis",
        height=600,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=80, b=40))
    fig.update_yaxes(title_text="Discharge (cfs)", type="log", row=1, col=1)
    fig.update_yaxes(title_text="BFI", range=[0, 1], row=2, col=1)

    st.plotly_chart(fig, width="stretch", key="bfi_chart")


def _render_anomaly_tab(df_q, q_col, desc):
    """Render seasonal flow anomaly comparison."""
    st.subheader("Seasonal Anomaly")
    st.caption("Current year flow compared to historical norms by day-of-year")

    anomaly = get_seasonal_anomaly(df_q[q_col])

    if anomaly.empty:
        st.warning("Insufficient data for seasonal anomaly (need 3+ years)")
        return

    current_year = df_q.index.year.max()

    # Summary metrics
    recent = anomaly.tail(30)
    if not recent.empty:
        col1, col2, col3 = st.columns(3)
        with col1:
            mean_anomaly = recent['anomaly_pct'].mean()
            st.metric("30-Day Avg Anomaly", f"{mean_anomaly:+.0f}%")
        with col2:
            mean_pctile = recent['percentile'].mean()
            st.metric("30-Day Avg Percentile", f"{mean_pctile:.0f}th")
        with col3:
            if mean_pctile > 75:
                status = "Above Normal"
            elif mean_pctile > 25:
                status = "Normal"
            else:
                status = "Below Normal"
            st.metric("Status", status)

    # Anomaly plot
    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.6, 0.4],
        shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=[
            f"{current_year} Flow vs Historical Median",
            "Anomaly (% of Median)"
        ])

    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Top: current year vs historical median
    fig.add_trace(go.Scatter(
        x=anomaly['doy'], y=anomaly['historical_median'],
        mode='lines', name='Historical Median',
        line=dict(color='gray', width=1.5, dash='dash'),
        hovertemplate='Day %{x}<br>Median: %{y:,.0f} cfs<extra></extra>'), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=anomaly['doy'], y=anomaly['current_year_flow'],
        mode='lines', name=f'{current_year}',
        line=dict(color='#d62728', width=2),
        hovertemplate=f'{current_year} Day %{{x}}<br>%{{y:,.0f}} cfs<extra></extra>'), row=1, col=1)

    # Bottom: anomaly bars colored by sign
    positive = anomaly[anomaly['anomaly_pct'] >= 0]
    negative = anomaly[anomaly['anomaly_pct'] < 0]

    if not positive.empty:
        fig.add_trace(go.Bar(
            x=positive['doy'], y=positive['anomaly_pct'],
            name='Above Normal', marker_color='rgba(44, 160, 44, 0.6)',
            hovertemplate='Day %{x}<br>%{y:+.0f}%<extra></extra>'), row=2, col=1)

    if not negative.empty:
        fig.add_trace(go.Bar(
            x=negative['doy'], y=negative['anomaly_pct'],
            name='Below Normal', marker_color='rgba(214, 39, 40, 0.6)',
            hovertemplate='Day %{x}<br>%{y:+.0f}%<extra></extra>'), row=2, col=1)

    fig.add_hline(y=0, line_color="gray", line_width=1, row=2, col=1)

    fig.update_layout(
        title=f"{desc} - Seasonal Anomaly ({current_year})",
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=80, b=40))
    fig.update_xaxes(tickvals=month_starts, ticktext=month_labels, row=2, col=1)
    fig.update_yaxes(title_text="Discharge (cfs)", type="log", row=1, col=1)
    fig.update_yaxes(title_text="Anomaly (%)", row=2, col=1)

    st.plotly_chart(fig, width="stretch", key="anomaly_chart")


def _fetch_precip_data(site_id, lat, lon, start_str, end_str):
    """Try Daymet first, fall back to Meteostat for precipitation."""
    try:
        from hydrology.data.hyriver import get_daymet_climate
        daymet = get_daymet_climate(site_id, start_str, end_str, variables=['prcp'])
        if daymet is not None and 'precip_mm' in daymet.columns:
            return daymet['precip_mm']
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Daymet failed, trying Meteostat: {e}")

    # Meteostat fallback
    try:
        from hydrology.data.climate import fetch_climate_data
        if lat and lon:
            climate = fetch_climate_data(
                float(lat), float(lon),
                pd.Timestamp(start_str), pd.Timestamp(end_str),
                include_temp=False, include_precip=True)
            if climate is not None:
                if 'Precip_mm' in climate.columns:
                    return climate['Precip_mm']
                if 'precip_mm' in climate.columns:
                    return climate['precip_mm']
                if 'prcp' in climate.columns:
                    return climate['prcp']
    except Exception as e:
        logger.debug(f"Meteostat precip fetch failed: {e}")

    return None
