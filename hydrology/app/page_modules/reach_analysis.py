"""
Reach Analysis page - surfaces the 9 reach analysis plot functions
that exist in plots.py but were previously unreachable from the UI.
"""

import streamlit as st
import matplotlib.pyplot as plt
from datetime import date

from hydrology.app.shared import (
    get_inventory, get_cached_site_info,
    extract_site_id, fetch_discharge_data,
    date_range_selector, render_export_buttons,
    _filter_inventory, FIPS_TO_STATE,
    logger)
from hydrology.app.styles import render_site_header
from hydrology.app.plot_config import REACH_PLOTS, get_display_name
from hydrology.visualization.plots import AVAILABLE_PLOTS
from hydrology.core import DEFAULT_DISCHARGE_CODE
from hydrology.core.timezone import ensure_utc
from hydrology.data.climate import fetch_climate_data
from hydrology.app.shared import fetch_climate_cached
from hydrology.visualization.interactive import baseflow_waterfall

import pandas as pd


# Default Spokane River reach stations
DEFAULT_UPSTREAM = "12419000"   # Post Falls
DEFAULT_DOWNSTREAM = "12422500"  # Spokane at Spokane (Greene St)


def show():
    """Render the Reach Analysis page."""
    inventory_df = get_inventory()
    if inventory_df.empty:
        st.error("Could not load site inventory")
        return

    st.header("Reach Analysis")
    st.caption("Compare upstream and downstream stations to analyze gains, losses, and aquifer contributions")

    # Each station gets its own independent search/filter
    all_options = [f"{row['site_id']} - {str(row.get('description', ''))[:60]}"
                   for _, row in inventory_df.iterrows()]
    states = ["All States"] + sorted(FIPS_TO_STATE.values())

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Upstream Station")
        up_search = st.text_input("Search", placeholder="River name or site ID...",
                                  key="reach_up_search", label_visibility="collapsed")
        up_state = st.selectbox("State", states, key="reach_up_state", label_visibility="collapsed")
        up_filtered = _filter_inventory(inventory_df, up_search, up_state)
        up_options = [f"{row['site_id']} - {str(row.get('description', ''))[:60]}"
                      for _, row in up_filtered.iterrows()]
        if up_search or up_state != "All States":
            st.caption(f"{len(up_options)} sites")
        default_up_idx = 0
        for i, opt in enumerate(up_options):
            if opt.startswith(DEFAULT_UPSTREAM):
                default_up_idx = i
        if up_options:
            upstream_sel = st.selectbox("Upstream", up_options, index=default_up_idx, key="reach_upstream")
            upstream_id = extract_site_id(upstream_sel)
            up_info = get_cached_site_info(upstream_id)
            if up_info:
                st.caption(f"{up_info.get('description', '')}")
        else:
            st.warning("No sites match")
            return

    with col2:
        st.subheader("Downstream Station")
        dn_search = st.text_input("Search", placeholder="River name or site ID...",
                                  key="reach_dn_search", label_visibility="collapsed")
        dn_state = st.selectbox("State", states, key="reach_dn_state", label_visibility="collapsed")
        dn_filtered = _filter_inventory(inventory_df, dn_search, dn_state)
        dn_options = [f"{row['site_id']} - {str(row.get('description', ''))[:60]}"
                      for _, row in dn_filtered.iterrows()]
        if dn_search or dn_state != "All States":
            st.caption(f"{len(dn_options)} sites")
        default_dn_idx = 0
        for i, opt in enumerate(dn_options):
            if opt.startswith(DEFAULT_DOWNSTREAM):
                default_dn_idx = i
        if dn_options:
            downstream_sel = st.selectbox("Downstream", dn_options, index=default_dn_idx, key="reach_downstream")
            downstream_id = extract_site_id(downstream_sel)
            dn_info = get_cached_site_info(downstream_id)
            if dn_info:
                st.caption(f"{dn_info.get('description', '')}")
        else:
            st.warning("No sites match")
            return

    st.markdown("---")

    # Date range
    st.subheader("Date Range")
    start_date, end_date = date_range_selector("reach", default_start=date(2000, 1, 1))

    st.markdown("---")

    # Plot selection - default to showing key reach plots
    st.subheader("Reach Analysis Plots")

    reach_plot_options = {get_display_name(p): p for p in REACH_PLOTS if p in AVAILABLE_PLOTS}
    default_plots = ['reach_comparison', 'reach_index', 'seasonal_gain_loss']
    default_display = [get_display_name(p) for p in default_plots if p in reach_plot_options.values()]

    selected_display = st.multiselect(
        "Select plots",
        list(reach_plot_options.keys()),
        default=default_display,
        key="reach_plot_select"
    )
    selected_plots = [reach_plot_options[d] for d in selected_display]

    # Show descriptions
    with st.expander("Plot descriptions"):
        for display_name, plot_key in reach_plot_options.items():
            info = AVAILABLE_PLOTS.get(plot_key, {})
            desc = info.get('description', '') if isinstance(info, dict) else ''
            st.markdown(f"**{display_name}**: {desc}")

    st.markdown("---")

    # Reach map - show upstream/downstream stations
    if up_info and dn_info and up_info.get('latitude') and dn_info.get('latitude'):
        with st.expander("Reach Map", expanded=False):
            import folium
            from streamlit_folium import st_folium

            up_lat, up_lon = float(up_info['latitude']), float(up_info['longitude'])
            dn_lat, dn_lon = float(dn_info['latitude']), float(dn_info['longitude'])
            center_lat = (up_lat + dn_lat) / 2
            center_lon = (up_lon + dn_lon) / 2

            m = folium.Map(location=[center_lat, center_lon], zoom_start=10,
                          tiles='CartoDB dark_matter')

            # Upstream marker (blue)
            folium.CircleMarker(
                [up_lat, up_lon], radius=10, color='#2196F3', fill=True,
                fillColor='#2196F3', fillOpacity=0.8,
                tooltip=f"Upstream: {up_info.get('description', upstream_id)}"
            ).add_to(m)

            # Downstream marker (orange)
            folium.CircleMarker(
                [dn_lat, dn_lon], radius=10, color='#FF9800', fill=True,
                fillColor='#FF9800', fillOpacity=0.8,
                tooltip=f"Downstream: {dn_info.get('description', downstream_id)}"
            ).add_to(m)

            # River line connecting them
            folium.PolyLine(
                [[up_lat, up_lon], [dn_lat, dn_lon]],
                color='#4FC3F7', weight=3, opacity=0.7, dash_array='10'
            ).add_to(m)

            st_folium(m, width=None, height=300, returned_objects=[])
            st.caption("Blue = upstream, Orange = downstream")

    st.markdown("---")

    # Layout and generate
    col_layout, col_dpi, col_btn = st.columns([2, 1, 2])
    with col_layout:
        layout_choice = st.selectbox("Layout", ["Auto", "Vertical", "Grid 2x3"], key="reach_layout")
    with col_dpi:
        dpi = st.number_input("DPI", min_value=72, max_value=300, value=150, key="reach_dpi")
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        generate = st.button("Generate Reach Analysis", type="primary", use_container_width=True, key="gen_reach")

    if generate:
        if not selected_plots:
            st.warning("Select at least one plot")
            return

        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        # Fetch data for both stations
        with st.spinner("Fetching upstream data..."):
            df_upstream = fetch_discharge_data(upstream_id, DEFAULT_DISCHARGE_CODE, start_str, end_str)

        with st.spinner("Fetching downstream data..."):
            df_downstream = fetch_discharge_data(downstream_id, DEFAULT_DISCHARGE_CODE, start_str, end_str)

        if df_upstream is None or df_upstream.empty:
            st.error(f"No discharge data for upstream station {upstream_id}")
            return
        if df_downstream is None or df_downstream.empty:
            st.error(f"No discharge data for downstream station {downstream_id}")
            return

        # Fetch climate data if needed for precip_response_comparison or summer_climate_context
        df_climate = None
        climate_needed = any(p in selected_plots for p in ['precip_response_comparison', 'summer_climate_context'])
        if climate_needed:
            # Use downstream station coords for climate
            if dn_info and dn_info.get('latitude') and dn_info.get('longitude'):
                with st.spinner("Fetching climate data..."):
                    df_climate = fetch_climate_cached(
                        float(dn_info['latitude']), float(dn_info['longitude']),
                        start_str, end_str
                    )

        # Show data summary
        up_desc = up_info.get('description', upstream_id) if up_info else upstream_id
        dn_desc = dn_info.get('description', downstream_id) if dn_info else downstream_id

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Upstream", f"{len(df_upstream):,} days", delta=up_desc, delta_color="off")
        with col2:
            st.metric("Downstream", f"{len(df_downstream):,} days", delta=dn_desc, delta_color="off")

        st.markdown("---")

        # Generate plots
        n_plots = len(selected_plots)
        if layout_choice == "Vertical" or n_plots <= 2:
            ncols = 1
            nrows = n_plots
        elif layout_choice == "Grid 2x3" or n_plots > 3:
            ncols = 2
            nrows = (n_plots + 1) // 2
        else:
            ncols = min(n_plots, 3)
            nrows = (n_plots + ncols - 1) // ncols

        fig_height = min(max(5 * nrows, 6), 40)  # Cap to prevent image size errors
        fig_width = min(8 * ncols, 20)

        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), dpi=dpi, squeeze=False)

        with st.spinner("Generating reach analysis plots..."):
            for idx, plot_name in enumerate(selected_plots):
                row = idx // ncols
                col = idx % ncols
                ax = axes[row, col]

                plot_info = AVAILABLE_PLOTS.get(plot_name)
                if plot_info and isinstance(plot_info, dict):
                    plot_func = plot_info.get('function')
                    if plot_func:
                        try:
                            kwargs = {
                                'ax': ax,
                                'df_upstream': df_upstream,
                                'df_downstream': df_downstream,
                                'df_q': df_downstream,  # Some plots use df_q as primary
                                'config': {
                                    'upstream_name': up_desc[:40],
                                    'downstream_name': dn_desc[:40],
                                },
                            }
                            if df_climate is not None:
                                kwargs['df_climate'] = df_climate
                            plot_func(**kwargs)
                        except Exception as e:
                            ax.text(0.5, 0.5, f"Error: {str(e)[:60]}",
                                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
                            logger.error(f"Reach plot error for {plot_name}: {e}")

            # Hide unused axes
            for idx in range(n_plots, nrows * ncols):
                axes[idx // ncols, idx % ncols].set_visible(False)

        fig.suptitle(f"Reach Analysis: {up_desc[:45]} \u2192 {dn_desc[:45]}", fontsize=14, fontweight='bold')
        fig.tight_layout()

        st.pyplot(fig)
        render_export_buttons(fig, f"reach_{upstream_id}_{downstream_id}", dpi)
        plt.close(fig)

        # Baseflow separation waterfall (interactive Plotly)
        st.markdown("---")
        st.subheader("Baseflow Separation")
        show_waterfall = st.checkbox(
            "Show Baseflow Waterfall", value=True, key="show_bf_waterfall"
        )
        if show_waterfall:
            with st.spinner("Computing baseflow separation..."):
                fig_wf = baseflow_waterfall(
                    df_upstream, df_downstream,
                    upstream_name=up_desc[:40],
                    downstream_name=dn_desc[:40],
                    title=f"Baseflow Waterfall: {up_desc[:35]} → {dn_desc[:35]}")
            st.plotly_chart(fig_wf, use_container_width=True, key="plotly_bf_waterfall")
    else:
        st.info("Select upstream and downstream stations, choose plots, then click 'Generate Reach Analysis'")
