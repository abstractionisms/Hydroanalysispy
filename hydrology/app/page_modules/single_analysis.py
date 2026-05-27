"""
Single Analysis page - analyze a single site with multiple plot types.
Interactive Plotly charts auto-load on station/date selection.
Static matplotlib grid available via Generate button.
"""

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date

from hydrology.app.shared import (
    get_inventory, get_cached_site_info, get_weather_station_info,
    extract_site_id, process_site_data,
    date_range_selector, plot_selector, render_export_buttons,
    render_data_download, site_picker, logger, AVAILABLE_PLOTS)
from hydrology.app.styles import (
    render_site_header, render_availability_badges, render_metric_cards,
    render_plot_capability_board, render_insight_board
)
from hydrology.app.interpretation import summarize_flow_context, summarize_recommendations
from hydrology.app.plot_config import SINGLE_SITE_PLOTS, resolve_generated_plots
from hydrology.visualization import create_multi_plot, PlotLayout
from hydrology.visualization.interactive import (
    interactive_hydrograph, interactive_fdc,
    raster_hydrograph, percentile_bands_hydrograph)
from hydrology.core import DEFAULT_DISCHARGE_CODE


@st.cache_data(ttl=3600, show_spinner=False)
def _load_site_data(site_id, lat, lon, start_str, end_str):
    """Cached wrapper around process_site_data."""
    return process_site_data(site_id, lat, lon, start_str, end_str)


def _render_analysis_readiness(data, has_stage: bool, climate_info):
    """Show visible plot readiness before users open deeper controls."""
    df_q = data.get('df_q') if data else None
    df_merged = data.get('df_merged') if data else None
    discharge_days = len(df_q) if df_q is not None else 0
    has_climate = df_merged is not None and not df_merged.empty and (
        'Precip_mm' in df_merged.columns or 'Temp_C' in df_merged.columns
    )

    climate_status = "Ready" if has_climate else "Limited"
    if climate_info and climate_info.get('distance_km') is not None and climate_info['distance_km'] > 50:
        climate_status = "Distant station"

    cards = [
        {
            "title": "Core Flow",
            "body": f"{discharge_days:,} daily observations for hydrographs, duration curves, anomalies, and seasonal context.",
            "status": "Ready" if discharge_days else "Unavailable",
            "state": "ready" if discharge_days else "blocked",
        },
        {
            "title": "Climate Links",
            "body": "Temperature and precipitation overlays, lag response, SPI, and correlation views.",
            "status": climate_status,
            "state": "ready" if has_climate else "limited",
        },
        {
            "title": "Stage + Rating",
            "body": "Dual-axis stage overlay and stage-discharge rating curve when gage height exists.",
            "status": "Ready" if has_stage else "Needs gage height",
            "state": "ready" if has_stage else "blocked",
        },
        {
            "title": "Frequency",
            "body": "Flood frequency runs on demand from annual peak records so it does not slow page load.",
            "status": "On demand",
            "state": "limited",
        },
    ]

    st.subheader("Analysis Readiness")
    st.caption("Plot availability is based on the selected site, date range, and linked weather station.")
    render_plot_capability_board(cards)


def _render_hydrologic_summary(data, has_stage: bool, climate_info):
    """Render automated interpretation cards for the selected record."""
    df_q = data.get('df_q') if data else None
    df_merged = data.get('df_merged') if data else None
    has_climate = df_merged is not None and not df_merged.empty and (
        'Precip_mm' in df_merged.columns or 'Temp_C' in df_merged.columns
    )
    record_years = 0
    if df_q is not None and not df_q.empty:
        record_years = (df_q.index.max() - df_q.index.min()).days / 365.25

    st.subheader("Hydrologic Summary")
    st.caption("Automated context from the selected site's historical daily record.")
    render_insight_board(summarize_flow_context(df_q))

    with st.expander("Recommended next views", expanded=False):
        render_insight_board(summarize_recommendations(has_stage, has_climate, record_years))
        if climate_info:
            station = climate_info.get('name', 'nearest weather station')
            distance = climate_info.get('distance_km')
            if distance is not None:
                st.caption(f"Climate-linked recommendations use {station}, about {distance:.1f} km from this gage.")
            else:
                st.caption(f"Climate-linked recommendations use {station}.")


def show():
    """Render the Single Analysis page."""
    inventory_df = get_inventory()
    if inventory_df.empty:
        st.error("Could not load site inventory")
        return

    st.subheader("Analysis Workspace")
    site_id = site_picker(inventory_df, key="single", label="Select Site", location="main")

    site_info = get_cached_site_info(site_id)
    if not site_info:
        st.error(f"Site {site_id} not found")
        return

    lat = site_info.get('latitude')
    lon = site_info.get('longitude')
    desc = site_info.get('description', site_id)

    # Main Area
    st.header("Single Site Analysis")
    st.caption(f"Site: {desc}")

    # Date range
    st.subheader("Date Range")
    start_date, end_date = date_range_selector("single")

    if not lat or not lon:
        st.error("Site missing coordinates")
        return

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    # Auto-load data whenever station or dates change
    with st.spinner("Loading discharge data..."):
        data = _load_site_data(site_id, float(lat), float(lon), start_str, end_str)

    if data is None:
        st.error("No discharge data available for this site/period")
        return

    # Site header and metrics
    render_site_header(site_id, desc, float(lat), float(lon))

    has_stage = 'Gage_Height_ft' in data['df_q'].columns if data['df_q'] is not None else False
    climate_info = get_weather_station_info(float(lat), float(lon)) if lat and lon else None
    render_availability_badges(True, has_stage, climate_info)
    render_metric_cards(data['df_q'], data['df_merged'])

    st.markdown("---")
    _render_hydrologic_summary(data, has_stage, climate_info)
    st.markdown("---")
    _render_analysis_readiness(data, has_stage, climate_info)

    # ── Interactive Charts (auto-loaded) ──
    st.subheader("Interactive Charts")
    st.caption("Hover, zoom, and pan. Click legend entries to toggle.")

    # Hydrograph
    col_agg, col_nwm, col_stage = st.columns(3)
    with col_agg:
        agg = st.radio("Aggregation", ["daily", "weekly", "monthly"],
                        horizontal=True, key="hydrograph_agg",
                        help="Changes the hydrograph time scale without refetching the selected site data.")
    with col_nwm:
        show_nwm = st.checkbox(
            "NWM forecast overlay",
            value=False,
            key="nwm_overlay",
            help="Adds NOAA National Water Model streamflow where a matching forecast record is available.",
        )
    with col_stage:
        show_stage = st.checkbox(
            "Show stage (dual axis)",
            value=False,
            disabled=not has_stage,
            key="stage_overlay",
            help="Requires gage-height observations at this site for the selected period.",
        )

    fig_hydro = interactive_hydrograph(
        data['df_q'], discharge_col='Discharge_cfs',
        title=f"{desc} - Hydrograph", aggregation=agg,
        show_percentile_bands=True)

    # NWM overlay
    if show_nwm:
        try:
            from hydrology.data.nwm import compare_nwm_usgs
            from datetime import timedelta
            import plotly.graph_objects as go

            end_dt = data['df_q'].index.max()
            start_dt = end_dt - timedelta(days=30)
            nwm_result = compare_nwm_usgs(
                site_id,
                start_dt.strftime('%Y-%m-%d'),
                end_dt.strftime('%Y-%m-%d')
            )
            if nwm_result and hasattr(nwm_result, 'nwm_data') and nwm_result.nwm_data is not None:
                fig_hydro.add_trace(go.Scatter(
                    x=nwm_result.nwm_data.index,
                    y=nwm_result.nwm_data['streamflow_cfs'],
                    mode='lines', name='NWM Forecast',
                    line=dict(color='#ff7f0e', width=2, dash='dot')))
                st.caption(f"NWM: NSE={nwm_result.nash_sutcliffe:.2f}, RMSE={nwm_result.rmse:.0f} cfs")
            else:
                st.caption("NWM data not available for this site/period")
        except Exception as e:
            st.caption(f"NWM overlay unavailable: {str(e)[:60]}")

    # Stage overlay (dual axis)
    if show_stage and has_stage:
        import plotly.graph_objects as go
        stage = data['df_q']['Gage_Height_ft'].dropna()
        if not stage.empty:
            fig_hydro.add_trace(go.Scatter(
                x=stage.index, y=stage.values,
                mode='lines', name='Gage Height (ft)',
                line=dict(color='#2ca02c', width=1.5),
                yaxis='y2',
                hovertemplate='%{x|%Y-%m-%d}<br>%{y:.2f} ft<extra>Stage</extra>'
            ))
            fig_hydro.update_layout(
                yaxis2=dict(
                    title="Gage Height (ft)",
                    overlaying='y', side='right',
                    showgrid=False)
            )

    st.plotly_chart(fig_hydro, width="stretch", key="plotly_hydro")

    # Flow Duration Curve
    koehler = st.checkbox(
        "Koehler (2025) dQ/dt coloring",
        value=True,
        key="fdc_koehler",
        help="Colors the flow-duration curve by rate-of-change to separate stable flow states from flashier transitions.",
    )
    fig_fdc = interactive_fdc(
        data['df_q'], discharge_col='Discharge_cfs',
        title=f"{desc} - Flow Duration Curve",
        color_by_dqdt=koehler)
    st.plotly_chart(fig_fdc, width="stretch", key="plotly_fdc")

    # CSV download
    render_data_download(data['df_q'], filename_prefix=site_id)

    # Advanced visualizations (opt-in)
    st.markdown("---")
    st.subheader("Advanced Visualizations")
    col_raster, col_pctile = st.columns(2)
    with col_raster:
        show_raster = st.checkbox("Raster Hydrograph", value=False,
                                   key="show_raster")
    with col_pctile:
        show_pctile = st.checkbox("Percentile Bands", value=False,
                                   key="show_pctile")

    if show_raster:
        fig_raster = raster_hydrograph(
            data['df_q'], discharge_col='Discharge_cfs',
            title=f"{desc} - Raster Hydrograph")
        st.plotly_chart(fig_raster, width="stretch", key="plotly_raster")

    if show_pctile:
        fig_pctile = percentile_bands_hydrograph(
            data['df_q'], discharge_col='Discharge_cfs',
            title=f"{desc} - Percentile Bands")
        st.plotly_chart(fig_pctile, width="stretch", key="plotly_pctile")

    # ── Frequency Analysis (on demand) ──
    st.markdown("---")
    with st.expander("Frequency Analysis", expanded=False):
        st.caption("Multi-distribution flood frequency with return period estimation")

        if st.button("Run Frequency Analysis", type="primary", key="gen_freq"):
            from hydrology.data.usgs import fetch_peak_streamflow
            from hydrology.analysis.frequency import (
                fit_flood_frequency, estimate_return_periods, get_plotting_positions)
            from hydrology.visualization.interactive import interactive_return_period

            with st.spinner("Fetching peak streamflow data..."):
                peaks_df = fetch_peak_streamflow(site_id)

            if peaks_df is None or peaks_df.empty:
                st.warning("No peak streamflow data available for this site")
            else:
                peak_col = 'peak_va' if 'peak_va' in peaks_df.columns else peaks_df.columns[0]
                peak_values = peaks_df[peak_col].dropna().values.astype(float)
                peak_values = peak_values[peak_values > 0]

                if len(peak_values) < 10:
                    st.warning(f"Insufficient peaks for analysis ({len(peak_values)} < 10)")
                else:
                    st.metric("Peak Records", len(peak_values))

                    # Fit distributions
                    fits = fit_flood_frequency(peak_values)
                    observed = get_plotting_positions(peak_values)
                    rp_table = estimate_return_periods(peak_values)

                    if fits:
                        # Distribution comparison chart
                        fig_rp = interactive_return_period(
                            observed, fits, rp_table,
                            title=f"{desc} - Flood Frequency Analysis")
                        st.plotly_chart(fig_rp, width="stretch", key="plotly_rp")

                        # Return period table
                        if not rp_table.empty:
                            st.subheader("Return Period Estimates")
                            display_rp = rp_table.copy()
                            for col in ['flow_cfs', 'lower_ci', 'upper_ci']:
                                if col in display_rp.columns:
                                    display_rp[col] = display_rp[col].apply(
                                        lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A"
                                    )
                            st.dataframe(display_rp, width="stretch", hide_index=True)

                        # Model comparison table
                        st.subheader("Distribution Comparison")
                        model_rows = []
                        for name, fit in fits.items():
                            model_rows.append({
                                'Distribution': fit.display_name,
                                'AIC': f"{fit.aic:.1f}",
                                'BIC': f"{fit.bic:.1f}",
                                'KS Statistic': f"{fit.ks_statistic:.4f}",
                                'KS p-value': f"{fit.ks_pvalue:.4f}",
                            })
                        st.dataframe(
                            pd.DataFrame(model_rows),
                            width="stretch", hide_index=True
                        )
                        st.caption("Lower AIC/BIC = better fit. Best distribution listed first.")
                    else:
                        st.error("Could not fit any distributions")

    # ── Static Matplotlib Plots (on demand) ──
    st.markdown("---")
    st.subheader("Guided Plot Builder")
    st.caption("Use presets for common workflows or choose any plot manually. Static figures are generated on demand for export.")
    single_site_available = {
        plot: info
        for plot, info in AVAILABLE_PLOTS.items()
        if plot in SINGLE_SITE_PLOTS
    }
    selected_plots = plot_selector(single_site_available)

    static_plots = resolve_generated_plots(selected_plots)

    layout_options = {
        'Auto': PlotLayout.AUTO,
        'Vertical': PlotLayout.VERTICAL,
        'Quad (2x2)': PlotLayout.QUAD,
        'Grid 2x3': PlotLayout.GRID_2x3,
        'Grid 3x2': PlotLayout.GRID_3x2,
        'Grid 2x5': PlotLayout.GRID_2x5,
    }

    col_layout, col_dpi, col_btn = st.columns([2, 1, 2])
    with col_layout:
        layout = st.selectbox("Layout", list(layout_options.keys()))
    with col_dpi:
        dpi = st.number_input("DPI", min_value=72, max_value=300, value=150)
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        generate = st.button("Generate Plots", type="primary", width="stretch")

    if generate:
        if not static_plots:
            st.warning("Select at least one plot type")
        else:
            plot_data = {
                'df_q': data['df_q'],
                'df_merged': data['df_merged'],
                'analysis_results': data['analysis_results']
            }

            with st.spinner("Generating plots..."):
                fig = create_multi_plot(
                    plots=static_plots,
                    layout=layout_options[layout],
                    data=plot_data,
                    site_id=site_id,
                    title=desc,
                    dpi=dpi
                )

            if fig:
                st.pyplot(fig)
                render_export_buttons(fig, site_id, dpi)
                plt.close(fig)
