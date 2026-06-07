"""
Alerts page - real-time alert monitoring and recent discharge history.
Also includes Multi-Site Analysis, NWM Comparison, Flood Animation,
and Watershed View as sub-tabs under an "Advanced" section.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import plotly.graph_objects as go

from hydrology.app.shared import (
    get_inventory, get_cached_site_info, get_weather_station_info,
    extract_site_id, display_site_info, site_picker,
    logger)
from hydrology.data.usgs import (
    fetch_daily_values, fetch_instantaneous_values,
    DEFAULT_PARAM_DISCHARGE, DEFAULT_PARAM_STAGE
)
from hydrology.analysis.alerts import (
    AlertMonitor, create_flood_alert, create_low_flow_alert
)


def show():
    """Render the Alert Monitor page."""
    inventory_df = get_inventory()
    if inventory_df.empty:
        st.error("Could not load site inventory")
        return

    st.subheader("Monitor Site")
    site_id = site_picker(inventory_df, key="alert", label="Monitor Site", location="main")

    site_info = get_cached_site_info(site_id)
    if not site_info:
        st.error(f"Site {site_id} not found")
        return

    render_site_current_check(site_id, site_info, key_prefix="alert_page")


def render_site_current_check(site_id: str, site_info: dict, key_prefix: str = "current_check"):
    """Render a manual current threshold check for a selected site."""
    desc = site_info.get('description', site_id)

    st.subheader("Current Conditions Check")
    st.caption(
        f"Manual threshold check for {desc}. Streamlit does not run background monitoring "
        "unless it is connected to a scheduled job or notification service."
    )

    # Alert configuration
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Flood Alerts")
        flood_enabled = st.checkbox("Enable flood alerts", value=True, key=f"{key_prefix}_flood_enabled")
        if flood_enabled:
            action_stage = st.number_input("Action Stage (ft)", value=10.0, step=0.5, key=f"{key_prefix}_action_stage")
            flood_stage = st.number_input("Flood Stage (ft)", value=12.0, step=0.5, key=f"{key_prefix}_flood_stage")
            major_flood = st.number_input("Major Flood Stage (ft)", value=15.0, step=0.5, key=f"{key_prefix}_major_flood")

    with col2:
        st.subheader("Low Flow Alerts")
        low_flow_enabled = st.checkbox("Enable low flow alerts", value=False, key=f"{key_prefix}_low_flow_enabled")
        if low_flow_enabled:
            low_flow_threshold = st.number_input("Low Flow (cfs)", value=100.0, step=10.0, key=f"{key_prefix}_low_flow")
            critical_flow = st.number_input("Critical Flow (cfs)", value=50.0, step=10.0, key=f"{key_prefix}_critical_flow")

    st.markdown("---")

    st.info(
        "This page evaluates the latest USGS reading when you run a check. It does not "
        "send notifications or keep checking after you leave the page.",
        icon="ℹ️",
    )

    if st.button("Check Current Conditions", type="primary", key=f"{key_prefix}_run"):
        with st.spinner("Fetching current data..."):
            monitor = AlertMonitor()

            if flood_enabled:
                thresholds = create_flood_alert(site_id, flood_stage, action_stage, major_flood)
                for t in thresholds:
                    monitor.add_threshold(t)

            if low_flow_enabled:
                thresholds = create_low_flow_alert(site_id, low_flow_threshold, critical_flow)
                for t in thresholds:
                    monitor.add_threshold(t)

            alerts = monitor.check_site(site_id, use_instantaneous=True)

            st.subheader("Current Reading")

            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=1)

                df_instant = fetch_instantaneous_values(
                    site_id, param_cd=DEFAULT_PARAM_DISCHARGE,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                )

                if df_instant is not None and not df_instant.empty:
                    latest = df_instant.iloc[-1]
                    latest_time = df_instant.index[-1]

                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Latest Discharge", f"{latest['value']:.1f} cfs",
                         help="Most recent instantaneous discharge from the USGS gage")
                    with col_b:
                        st.metric("Reading Time", latest_time.strftime('%Y-%m-%d %H:%M'))
                    with col_c:
                        if alerts:
                            st.metric("Triggered Thresholds", len(alerts),
                         help="Number of configured flood or low-flow thresholds triggered by the latest reading")
                        else:
                            st.metric("Status", "Normal")
                else:
                    st.warning("Could not fetch current instantaneous data")

            except Exception as e:
                st.error(f"Error fetching data: {e}")

            if alerts:
                st.subheader("Triggered Thresholds")
                for alert in alerts:
                    severity_colors = {'critical': '🔴', 'warning': '🟡', 'info': '🔵'}
                    icon = severity_colors.get(alert.severity, '⚪')
                    st.error(f"{icon} **{alert.severity.upper()}**: {alert.message}")
            else:
                st.success("No configured thresholds were triggered by the latest reading")

    # Recent history
    st.markdown("---")
    st.subheader("Recent Discharge History")

    history_days = st.select_slider(
        "History length",
        options=[7, 14, 30, 60, 90],
        value=30,
        format_func=lambda x: f"{x} days"
    )

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=history_days)

        df_recent = fetch_instantaneous_values(
            site_id, param_cd=DEFAULT_PARAM_DISCHARGE,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )

        if df_recent is not None and not df_recent.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_recent.index, y=df_recent['value'],
                mode='lines', name='Discharge',
                line=dict(color='#1f77b4', width=2),
                fill='tozeroy', fillcolor='rgba(31, 119, 180, 0.2)'
            ))

            current_val = df_recent['value'].iloc[-1] if len(df_recent) > 0 else None

            fig.update_layout(
                yaxis_type="log", yaxis_title="Discharge (cfs)",
                xaxis_title="", height=300,
                margin=dict(l=60, r=20, t=30, b=40),
                showlegend=False, hovermode='x unified'
            )

            y_min = df_recent['value'].min() * 0.8
            y_max = df_recent['value'].max() * 1.2
            fig.update_yaxes(range=[np.log10(max(y_min, 0.1)), np.log10(y_max)])

            st.plotly_chart(fig, width="stretch")

            # Historical context
            col1, col2 = st.columns(2)
            with col1:
                try:
                    current_doy = end_date.timetuple().tm_yday
                    hist_start = (end_date - timedelta(days=365 * 10)).strftime('%Y-%m-%d')
                    hist_end = end_date.strftime('%Y-%m-%d')

                    df_hist = fetch_daily_values(
                        site_id, param_cd=DEFAULT_PARAM_DISCHARGE,
                        start_date=hist_start, end_date=hist_end
                    )

                    if df_hist is not None and not df_hist.empty:
                        df_hist['doy'] = df_hist.index.dayofyear
                        seasonal_data = df_hist[
                            (df_hist['doy'] >= current_doy - 15) &
                            (df_hist['doy'] <= current_doy + 15)
                        ]['value']

                        if len(seasonal_data) > 10 and current_val:
                            percentile = (seasonal_data < current_val).mean() * 100

                            if percentile > 90:
                                pct_status = "Very High"
                            elif percentile > 75:
                                pct_status = "Above Normal"
                            elif percentile > 25:
                                pct_status = "Normal"
                            elif percentile > 10:
                                pct_status = "Below Normal"
                            else:
                                pct_status = "Very Low"

                            st.metric("Seasonal Percentile", f"{percentile:.0f}%",
                                     delta=pct_status, delta_color="off",
                         help="Where current flow ranks compared to historical flows for this day of year. 50% = median, <10% = much below normal, >90% = much above normal")
                        else:
                            st.metric("Seasonal Percentile", "N/A",
                         help="Where current flow ranks compared to historical flows for this day of year. 50% = median, <10% = much below normal, >90% = much above normal")
                    else:
                        st.metric("Seasonal Percentile", "N/A",
                         help="Where current flow ranks compared to historical flows for this day of year. 50% = median, <10% = much below normal, >90% = much above normal")
                except Exception:
                    st.metric("Seasonal Percentile", "N/A",
                         help="Where current flow ranks compared to historical flows for this day of year. 50% = median, <10% = much below normal, >90% = much above normal")

            with col2:
                try:
                    lat = site_info.get('latitude')
                    lon = site_info.get('longitude')
                    if lat and lon:
                        from hydrology.data.climate import fetch_climate_data
                        precip_start = (end_date - timedelta(days=7)).strftime('%Y-%m-%d')
                        precip_end = end_date.strftime('%Y-%m-%d')

                        climate_df = fetch_climate_data(float(lat), float(lon), precip_start, precip_end)

                        if climate_df is not None and 'prcp' in climate_df.columns:
                            total_precip_mm = climate_df['prcp'].sum()
                            total_precip_in = total_precip_mm / 25.4

                            if total_precip_in > 2:
                                precip_status = "High influence"
                            elif total_precip_in > 0.5:
                                precip_status = "Moderate"
                            elif total_precip_in > 0.1:
                                precip_status = "Low"
                            else:
                                precip_status = "Minimal"

                            st.metric("7-Day Precipitation", f"{total_precip_in:.2f} in",
                                     delta=precip_status, delta_color="off",
                         help="Total precipitation in the last 7 days from nearest weather station")
                        else:
                            st.metric("7-Day Precipitation", "N/A",
                         help="Total precipitation in the last 7 days from nearest weather station")
                    else:
                        st.metric("7-Day Precipitation", "N/A",
                         help="Total precipitation in the last 7 days from nearest weather station")
                except Exception:
                    st.metric("7-Day Precipitation", "N/A",
                         help="Total precipitation in the last 7 days from nearest weather station")
        else:
            st.info("No recent instantaneous data available")
    except Exception as e:
        st.warning(f"Could not load recent history: {e}")
