"""
Comparisons page - merges Time Periods, Sites, and 2x2 comparison modes
into one page with sub-tabs.
"""

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date, timedelta

from hydrology.app.shared import (
    get_inventory, get_cached_site_info,
    extract_site_id, display_site_info, process_site_data,
    date_range_selector, single_plot_selector_widget,
    create_comparison_figure, render_export_buttons,
    site_picker, logger,
)
from hydrology.app.styles import render_site_header
from hydrology.visualization.plots import AVAILABLE_PLOTS
from hydrology.visualization.interactive import interactive_comparison


def show():
    """Render the Comparisons page with sub-tabs."""
    inventory_df = get_inventory()
    if inventory_df.empty:
        st.error("Could not load site inventory")
        return

    st.header("Comparisons")

    tab_time, tab_sites, tab_quad = st.tabs([
        "Compare Time Periods",
        "Compare Sites",
        "2x2 Comparison"
    ])

    with tab_time:
        _compare_time_periods(inventory_df)

    with tab_sites:
        _compare_sites(inventory_df)

    with tab_quad:
        _quad_comparison(inventory_df)


def _compare_time_periods(inventory_df):
    """Compare same site across two equal-length time periods."""
    site_id = site_picker(inventory_df, key="compare_time", label="Choose site",
                          location="main", show_search=True)

    site_info = get_cached_site_info(site_id)
    if not site_info:
        st.error(f"Site {site_id} not found")
        return

    lat = site_info.get('latitude')
    lon = site_info.get('longitude')
    desc = site_info.get('description', site_id)

    # Water year helpers
    def get_water_year_start(d: date) -> date:
        return date(d.year, 10, 1) if d.month >= 10 else date(d.year - 1, 10, 1)

    def get_water_year_end(start: date) -> date:
        return date(start.year + 1, 9, 30)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Period Length")
        period_lengths = {
            "1 Year": 365, "2 Years": 730, "5 Years": 1825,
            "10 Years": 3650, "Water Year": "water_year",
        }
        period_choice = st.selectbox("Duration", list(period_lengths.keys()), key="period_length")
        is_water_year = period_choice == "Water Year"
        period_days = 365 if is_water_year else period_lengths[period_choice]
        st.caption("Oct 1 → Sep 30" if is_water_year else f"{period_days} days")

    if 'period_a_year' not in st.session_state:
        st.session_state.period_a_year = 2010
    if 'period_b_year' not in st.session_state:
        st.session_state.period_b_year = 2020

    current_year = date.today().year
    today = date.today()

    if 'start_a_input' in st.session_state and st.session_state.start_a_input > today:
        st.session_state.start_a_input = today
    if 'start_b_input' in st.session_state and st.session_state.start_b_input > today:
        st.session_state.start_b_input = today

    with col2:
        st.subheader("Period A")
        year_a = st.slider("Year", min_value=1900, max_value=current_year,
                           value=st.session_state.period_a_year, key="slider_a",
                           label_visibility="collapsed")
        start_a = date(year_a, 10, 1) if is_water_year else date(year_a, 1, 1)
        if st.session_state.period_a_year != year_a:
            st.session_state.period_a_year = year_a
            st.session_state.start_a_input = start_a

        with st.expander("Fine-tune date"):
            start_a = st.date_input("Start date", value=start_a, min_value=date(1900, 1, 1),
                                    max_value=today, key="start_a_input")
            if is_water_year:
                start_a = get_water_year_start(start_a)

        end_a = get_water_year_end(start_a) if is_water_year else start_a + timedelta(days=period_days)
        st.caption(f"{start_a} → {end_a}")

    with col3:
        st.subheader("Period B")
        year_b = st.slider("Year", min_value=1900, max_value=current_year,
                           value=st.session_state.period_b_year, key="slider_b",
                           label_visibility="collapsed")
        start_b = date(year_b, 10, 1) if is_water_year else date(year_b, 1, 1)
        if st.session_state.period_b_year != year_b:
            st.session_state.period_b_year = year_b
            st.session_state.start_b_input = start_b

        with st.expander("Fine-tune date"):
            start_b = st.date_input("Start date", value=start_b, min_value=date(1900, 1, 1),
                                    max_value=today, key="start_b_input")
            if is_water_year:
                start_b = get_water_year_start(start_b)

        end_b = get_water_year_end(start_b) if is_water_year else start_b + timedelta(days=period_days)
        st.caption(f"{start_b} → {end_b}")

    col_plot, col_dpi, col_btn = st.columns([3, 1, 2])
    with col_plot:
        st.subheader("Plot Type")
        plot_name = single_plot_selector_widget("_compare")
    with col_dpi:
        st.subheader("Quality")
        dpi = st.selectbox("DPI", [100, 150, 200], index=1, key="compare_time_dpi", label_visibility="collapsed")
    with col_btn:
        st.subheader(" ")
        generate = st.button("Compare Periods", type="primary", use_container_width=True, key="gen_time")

    if generate:
        if not lat or not lon:
            st.error("Site missing coordinates")
            return

        render_site_header(site_id, desc, float(lat) if lat else None, float(lon) if lon else None)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Period A: {start_a} to {end_a}")
            with st.spinner("Processing Period A..."):
                data_a = process_site_data(site_id, float(lat), float(lon),
                                           start_a.strftime('%Y-%m-%d'), end_a.strftime('%Y-%m-%d'))
            if data_a:
                st.caption(f"Discharge: {data_a['discharge_count']:,} | Merged: {data_a['merged_count']:,}")

        with col2:
            st.subheader(f"Period B: {start_b} to {end_b}")
            with st.spinner("Processing Period B..."):
                data_b = process_site_data(site_id, float(lat), float(lon),
                                           start_b.strftime('%Y-%m-%d'), end_b.strftime('%Y-%m-%d'))
            if data_b:
                st.caption(f"Discharge: {data_b['discharge_count']:,} | Merged: {data_b['merged_count']:,}")

        data_list = []
        titles = []

        if data_a:
            data_list.append({'df_q': data_a['df_q'], 'df_merged': data_a['df_merged'],
                             'analysis_results': data_a['analysis_results']})
            titles.append(f"{desc[:25]}\n{start_a} to {end_a}\n({data_a['discharge_count']:,} Q, {data_a['merged_count']:,} merged)")
        else:
            data_list.append(None)
            st.warning(f"No data available for Period A ({start_a} to {end_a})")
            titles.append(f"{desc[:25]}\n{start_a} to {end_a}\n(No data)")

        if data_b:
            data_list.append({'df_q': data_b['df_q'], 'df_merged': data_b['df_merged'],
                             'analysis_results': data_b['analysis_results']})
            titles.append(f"{desc[:25]}\n{start_b} to {end_b}\n({data_b['discharge_count']:,} Q, {data_b['merged_count']:,} merged)")
        else:
            data_list.append(None)
            st.warning(f"No data available for Period B ({start_b} to {end_b})")
            titles.append(f"{desc[:25]}\n{start_b} to {end_b}\n(No data)")

        if data_a or data_b:
            fig = create_comparison_figure(plot_name, data_list, titles, 1, 2, dpi)
            st.pyplot(fig)
            render_export_buttons(fig, f"{site_id}_comparison", dpi)
            plt.close(fig)
        else:
            st.error("No data available for either period")


def _compare_sites(inventory_df):
    """Compare multiple sites for the same time period."""
    selected_sites = site_picker(inventory_df, key="compare_sites", label="Choose 2-4 sites",
                                 location="main", multi=True, max_selections=4)

    if len(selected_sites) < 2:
        st.info("Select 2-4 sites to compare")
        return

    st.subheader("Date Range")
    start_date, end_date = date_range_selector("compare_sites", default_start=date(2015, 1, 1))

    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Plot Type")
        plot_name = single_plot_selector_widget("_sites")
    with col2:
        st.subheader("Quality")
        dpi = st.selectbox("Quality", [100, 150, 200], index=1, key="cs_dpi")

    st.markdown("---")
    if st.button("Compare Sites", type="primary", use_container_width=True, key="gen_sites"):
        st.header("Multi-Site Comparison")
        st.caption(f"{len(selected_sites)} sites | {start_date} to {end_date}")

        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        data_list = []
        titles = []

        progress = st.progress(0, text="Loading sites...")
        for i, site_id in enumerate(selected_sites):
            site_info = get_cached_site_info(site_id)
            desc = site_info.get('description', site_id) if site_info else site_id

            if not site_info:
                data_list.append(None)
                titles.append(f"{desc[:30]}\n(Site not found)")
                continue

            lat = site_info.get('latitude')
            lon = site_info.get('longitude')

            if not lat or not lon:
                data_list.append(None)
                titles.append(f"{desc[:30]}\n(No coordinates)")
                continue

            progress.progress((i + 1) / len(selected_sites), text=f"Processing {site_id}...")
            data = process_site_data(site_id, float(lat), float(lon), start_str, end_str)

            if data:
                data_list.append({
                    'df_q': data['df_q'], 'df_merged': data['df_merged'],
                    'analysis_results': data['analysis_results']
                })
                q_count = data['discharge_count']
                m_count = data['merged_count']
                titles.append(f"{desc[:30]}\n{start_str} to {end_str} ({q_count:,} Q, {m_count:,} merged)")
            else:
                data_list.append(None)
                titles.append(f"{desc[:30]}\n{start_str} to {end_str} (No data)")

        progress.empty()

        n = len(selected_sites)
        nrows, ncols = (1, n) if n <= 2 else (2, 2) if n <= 4 else (2, 3)

        if any(d is not None for d in data_list):
            # Interactive overlay for timeseries
            if plot_name == 'timeseries':
                site_dict = {}
                for site_str, data_item in zip(selected_sites, data_list):
                    if data_item and data_item.get('df_q') is not None:
                        sid = extract_site_id(site_str)
                        info = get_cached_site_info(sid)
                        label = info.get('description', sid)[:30] if info else sid
                        site_dict[label] = data_item['df_q']

                if site_dict:
                    st.subheader("Interactive Overlay")
                    fig_interactive = interactive_comparison(
                        site_dict, title="Multi-Site Discharge Comparison"
                    )
                    st.plotly_chart(fig_interactive, use_container_width=True)
                    st.markdown("---")

            # Static comparison grid
            fig = create_comparison_figure(plot_name, data_list, titles, nrows, ncols, dpi)
            st.pyplot(fig)
            render_export_buttons(fig, "multi_site_comparison", dpi)
            plt.close(fig)
        else:
            st.error("No data available for any of the selected sites")


def _quad_comparison(inventory_df):
    """2x2 comparison: 2 sites x 2 equal-length time periods."""
    col_sa, col_sb = st.columns(2)
    with col_sa:
        site_a = site_picker(inventory_df, key="quad_a", label="Site A",
                             location="main", show_search=True)
    with col_sb:
        site_b = site_picker(inventory_df, key="quad_b", label="Site B",
                             location="main", show_search=True)

    today = date.today()

    if 'quad_start_1' in st.session_state and st.session_state.quad_start_1 > today:
        st.session_state.quad_start_1 = today
    if 'quad_start_2' in st.session_state and st.session_state.quad_start_2 > today:
        st.session_state.quad_start_2 = today

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Period Length")
        period_lengths = {"1 Year": 365, "2 Years": 730, "5 Years": 1825, "10 Years": 3650}
        period_choice = st.selectbox("Duration", list(period_lengths.keys()), key="quad_period_length")
        period_days = period_lengths[period_choice]

    with col2:
        st.subheader("Period 1")
        if 'quad_p1_start' not in st.session_state:
            st.session_state.quad_p1_start = date(2010, 1, 1)

        col_p1a, col_p1b, col_p1c = st.columns([1, 3, 1])
        with col_p1a:
            if st.button("◀", key="shift_p1_back"):
                st.session_state.quad_p1_start -= timedelta(days=period_days)
        with col_p1b:
            start_1 = st.date_input("Start", st.session_state.quad_p1_start, key="quad_start_1",
                                    min_value=date(1900, 1, 1), max_value=today, label_visibility="collapsed")
            st.session_state.quad_p1_start = start_1
        with col_p1c:
            if st.button("▶", key="shift_p1_fwd"):
                st.session_state.quad_p1_start += timedelta(days=period_days)
        end_1 = start_1 + timedelta(days=period_days)
        st.caption(f"→ {end_1}")

    with col3:
        st.subheader("Period 2")
        if 'quad_p2_start' not in st.session_state:
            st.session_state.quad_p2_start = date(2020, 1, 1)

        col_p2a, col_p2b, col_p2c = st.columns([1, 3, 1])
        with col_p2a:
            if st.button("◀", key="shift_p2_back"):
                st.session_state.quad_p2_start -= timedelta(days=period_days)
        with col_p2b:
            start_2 = st.date_input("Start", st.session_state.quad_p2_start, key="quad_start_2",
                                    min_value=date(1900, 1, 1), max_value=today, label_visibility="collapsed")
            st.session_state.quad_p2_start = start_2
        with col_p2c:
            if st.button("▶", key="shift_p2_fwd"):
                st.session_state.quad_p2_start += timedelta(days=period_days)
        end_2 = start_2 + timedelta(days=period_days)
        st.caption(f"→ {end_2}")

    col_plot, col_dpi, col_btn = st.columns([3, 1, 2])
    with col_plot:
        st.subheader("Plot Type")
        plot_name = single_plot_selector_widget("_quad")
    with col_dpi:
        st.subheader("Quality")
        dpi = st.selectbox("DPI", [100, 150, 200], index=1, key="quad_dpi", label_visibility="collapsed")
    with col_btn:
        st.subheader(" ")
        generate = st.button("Generate 2x2", type="primary", use_container_width=True, key="gen_quad")

    if generate:
        site_id_a = site_a
        site_id_b = site_b
        site_info_a = get_cached_site_info(site_id_a)
        site_info_b = get_cached_site_info(site_id_b)

        if not site_info_a or not site_info_b:
            st.error("Could not load site info")
            return

        st.markdown("---")

        configs = [
            (site_id_a, site_info_a, start_1, end_1, f"{site_info_a['description'][:20]}\n{start_1} to {end_1}"),
            (site_id_a, site_info_a, start_2, end_2, f"{site_info_a['description'][:20]}\n{start_2} to {end_2}"),
            (site_id_b, site_info_b, start_1, end_1, f"{site_info_b['description'][:20]}\n{start_1} to {end_1}"),
            (site_id_b, site_info_b, start_2, end_2, f"{site_info_b['description'][:20]}\n{start_2} to {end_2}"),
        ]

        data_list = []
        titles = []

        progress = st.progress(0, text="Loading data...")
        for i, (sid, sinfo, start_d, end_d, title) in enumerate(configs):
            lat = sinfo.get('latitude')
            lon = sinfo.get('longitude')

            if not lat or not lon:
                data_list.append(None)
                titles.append(title + "\n(No coords)")
                continue

            progress.progress((i + 1) / 4, text=f"Processing {sid}...")
            data = process_site_data(sid, float(lat), float(lon),
                                    start_d.strftime('%Y-%m-%d'), end_d.strftime('%Y-%m-%d'))

            if data:
                data_list.append({
                    'df_q': data['df_q'], 'df_merged': data['df_merged'],
                    'analysis_results': data['analysis_results']
                })
                titles.append(f"{title}\n({data['discharge_count']:,} Q, {data['merged_count']:,} merged)")
            else:
                data_list.append(None)
                titles.append(title + "\n(No data)")

        progress.empty()

        fig = create_comparison_figure(plot_name, data_list, titles, 2, 2, dpi)
        fig.text(0.02, 0.75, 'Site A', rotation=90, fontsize=12, fontweight='bold', va='center')
        fig.text(0.02, 0.25, 'Site B', rotation=90, fontsize=12, fontweight='bold', va='center')
        fig.text(0.3, 0.98, 'Period 1', fontsize=12, fontweight='bold', ha='center')
        fig.text(0.7, 0.98, 'Period 2', fontsize=12, fontweight='bold', ha='center')

        st.pyplot(fig)
        render_export_buttons(fig, "2x2_comparison", dpi)
        plt.close(fig)
