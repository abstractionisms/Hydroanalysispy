"""
Single Analysis page - analyze a single site with multiple plot types.
Interactive Plotly charts auto-load on station/date selection.
Static matplotlib grid available via Generate button.
"""

import streamlit as st
import matplotlib.pyplot as plt
from datetime import date

from hydrology.app.shared import (
    get_inventory, get_cached_site_info, get_weather_station_info,
    extract_site_id, display_site_info, process_site_data,
    date_range_selector, plot_selector, render_export_buttons,
    site_picker, logger, AVAILABLE_PLOTS,
)
from hydrology.app.styles import (
    render_site_header, render_availability_badges, render_metric_cards
)
from hydrology.visualization import create_multi_plot, PlotLayout
from hydrology.visualization.interactive import (
    interactive_hydrograph, interactive_fdc,
    raster_hydrograph, percentile_bands_hydrograph,
)
from hydrology.core import DEFAULT_DISCHARGE_CODE


@st.cache_data(ttl=3600, show_spinner=False)
def _load_site_data(site_id, lat, lon, start_str, end_str):
    """Cached wrapper around process_site_data."""
    return process_site_data(site_id, lat, lon, start_str, end_str)


def show():
    """Render the Single Analysis page."""
    inventory_df = get_inventory()
    if inventory_df.empty:
        st.error("Could not load site inventory")
        return

    # Sidebar: Site Selection with search
    st.sidebar.header("Select Site")
    site_id = site_picker(inventory_df, key="single", label="Select Site", location="sidebar")

    site_info = get_cached_site_info(site_id)
    if not site_info:
        st.error(f"Site {site_id} not found")
        return

    lat = site_info.get('latitude')
    lon = site_info.get('longitude')
    desc = site_info.get('description', site_id)

    display_site_info(site_info)

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

    # ── Interactive Charts (auto-loaded) ──
    st.subheader("Interactive Charts")
    st.caption("Hover, zoom, and pan. Click legend entries to toggle.")

    # Hydrograph
    col_agg, col_nwm, col_stage = st.columns(3)
    with col_agg:
        agg = st.radio("Aggregation", ["daily", "weekly", "monthly"],
                        horizontal=True, key="hydrograph_agg")
    with col_nwm:
        show_nwm = st.checkbox("NWM forecast overlay", value=False, key="nwm_overlay")
    with col_stage:
        show_stage = st.checkbox("Show stage (dual axis)", value=False,
                                 disabled=not has_stage, key="stage_overlay")

    fig_hydro = interactive_hydrograph(
        data['df_q'], discharge_col='Discharge_cfs',
        title=f"{desc} - Hydrograph", aggregation=agg,
        show_percentile_bands=True,
    )

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
                    line=dict(color='#ff7f0e', width=2, dash='dot'),
                ))
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
                    showgrid=False,
                )
            )

    st.plotly_chart(fig_hydro, use_container_width=True, key="plotly_hydro")

    # Flow Duration Curve
    koehler = st.checkbox("Koehler (2025) dQ/dt coloring", value=True, key="fdc_koehler")
    fig_fdc = interactive_fdc(
        data['df_q'], discharge_col='Discharge_cfs',
        title=f"{desc} - Flow Duration Curve",
        color_by_dqdt=koehler,
    )
    st.plotly_chart(fig_fdc, use_container_width=True, key="plotly_fdc")

    # Advanced visualizations (opt-in)
    st.markdown("---")
    st.subheader("Advanced Visualizations")
    col_raster, col_pctile = st.columns(2)
    with col_raster:
        show_raster = st.checkbox("Raster Hydrograph", value=False,
                                   key="show_raster",
                                   help="Year x Day-of-Year heatmap of entire flow record")
    with col_pctile:
        show_pctile = st.checkbox("Percentile Bands", value=False,
                                   key="show_pctile",
                                   help="Current year flow vs historical percentile envelopes")

    if show_raster:
        fig_raster = raster_hydrograph(
            data['df_q'], discharge_col='Discharge_cfs',
            title=f"{desc} - Raster Hydrograph",
        )
        st.plotly_chart(fig_raster, use_container_width=True, key="plotly_raster")

    if show_pctile:
        fig_pctile = percentile_bands_hydrograph(
            data['df_q'], discharge_col='Discharge_cfs',
            title=f"{desc} - Percentile Bands",
        )
        st.plotly_chart(fig_pctile, use_container_width=True, key="plotly_pctile")

    # ── Static Matplotlib Plots (on demand) ──
    st.markdown("---")
    with st.expander("Static Plot Grid (matplotlib)", expanded=False):
        st.caption("Select plots and generate a static multi-plot figure for export.")
        selected_plots = plot_selector()

        # Filter out plots already shown interactively
        static_plots = [p for p in selected_plots if p not in {'timeseries', 'flow_duration'}]

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
            generate = st.button("Generate Plots", type="primary", use_container_width=True)

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
