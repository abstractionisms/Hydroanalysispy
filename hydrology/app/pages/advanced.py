"""
Advanced page - houses Multi-Site Analysis, NWM Comparison,
Flood Animation, and Watershed View as sub-tabs.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import plotly.graph_objects as go
import plotly.express as px
# folium/streamlit_folium imported locally where used (can't import outside Streamlit runtime)

from hydrology.app.shared import (
    get_inventory, get_cached_site_info,
    extract_site_id, site_picker, logger)
from hydrology.data.usgs import fetch_daily_values, DEFAULT_PARAM_DISCHARGE
from hydrology.data.nwm import NWMClient, compare_nwm_usgs, get_forecast_skill
from hydrology.analysis.multisite import MultiSiteAnalyzer
from hydrology.analysis.flood_events import FloodEventAnalyzer, calculate_event_statistics
from hydrology.data.national_inventory import get_national_inventory, get_region_inventory
from hydrology.core.huc_regions import HUC2_REGIONS, get_region_name, get_region_center, US_CENTER


def show():
    """Render the Advanced page with sub-tabs."""
    st.header("Advanced Analysis")

    tab_multi, tab_nwm, tab_flood, tab_watershed = st.tabs([
        "Multi-Site Analysis",
        "NWM Comparison",
        "Flood Animation",
        "Watershed View"
    ])

    inventory_df = get_inventory()
    if inventory_df.empty:
        st.error("Could not load site inventory")
        return

    with tab_multi:
        _multisite_analysis(inventory_df)

    with tab_nwm:
        _nwm_comparison(inventory_df)

    with tab_flood:
        _flood_animation(inventory_df)

    with tab_watershed:
        _watershed_view(inventory_df)


def _multisite_analysis(inventory_df):
    """Analyze correlations and relationships between multiple sites."""
    selected_sites = site_picker(inventory_df, key="multisite", label="Select Sites (2-6)",
                                 location="main", multi=True, max_selections=6)

    years_back = st.slider("Years of data", 1, 10, 3, key="multisite_years")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_back * 365)

    if len(selected_sites) < 2:
        st.info("Select 2-6 sites to analyze their relationships")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Correlation Analysis** - Compare discharge patterns")
        with col2:
            st.markdown("**Lag & Travel Time** - Estimate flow routing")
        return

    site_ids = selected_sites

    st.subheader("Selected Sites")
    site_cols = st.columns(min(len(site_ids), 3))
    for i, sid in enumerate(site_ids):
        info = get_cached_site_info(sid)
        with site_cols[i % 3]:
            st.markdown(f"**`{sid}`**")
            if info:
                st.caption(info.get('description', '')[:50])

    if st.button("Analyze Relationships", type="primary", use_container_width=True, key="gen_multi"):
        analyzer = MultiSiteAnalyzer()
        for sid in site_ids:
            info = get_cached_site_info(sid)
            name = info.get('description', sid) if info else sid
            lat = info.get('latitude') if info else None
            lon = info.get('longitude') if info else None
            analyzer.add_site(sid, name=name[:40], latitude=lat, longitude=lon)

        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        progress_bar = st.progress(0, text="Fetching site data...")
        for i, sid in enumerate(site_ids):
            progress_bar.progress((i + 1) / len(site_ids), text=f"Fetching {sid}...")
            try:
                df = fetch_daily_values(sid, param_cd='00060',
                                       start_date=start_str, end_date=end_str)
                if df is not None and not df.empty:
                    analyzer.data[sid] = df
            except Exception as e:
                st.warning(f"Failed to fetch {sid}: {e}")
        progress_bar.empty()

        synced_data = analyzer.get_synchronized_data()
        if synced_data.empty:
            st.error("Could not synchronize data for selected sites")
            return

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overlapping Days", f"{len(synced_data):,}")
        with col2:
            st.metric("From", synced_data.index.min().strftime('%Y-%m-%d'))
        with col3:
            st.metric("To", synced_data.index.max().strftime('%Y-%m-%d'))

        # Time series
        st.subheader("Synchronized Time Series")
        site_names = {}
        for sid in site_ids:
            info = get_cached_site_info(sid)
            name = info.get('description', sid)[:25] if info else sid
            site_names[sid] = f"{sid} - {name}"
        plot_data = synced_data.rename(columns=site_names)
        st.line_chart(plot_data, use_container_width=True)

        # Correlation matrix
        st.subheader("Correlation Matrix")
        corr_matrix = analyzer.get_correlation_matrix()
        if not corr_matrix.empty:
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index,
                colorscale='Viridis', zmin=0, zmax=1,
                text=[[f"{val:.3f}" for val in row] for row in corr_matrix.values],
                texttemplate="%{text}", textfont={"size": 12}))
            fig.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)

            avg_corr = corr_matrix.values[~np.eye(len(corr_matrix), dtype=bool)].mean()
            if avg_corr > 0.8:
                st.success(f"High correlation (avg: {avg_corr:.2f})")
            elif avg_corr > 0.5:
                st.info(f"Moderate correlation (avg: {avg_corr:.2f})")
            else:
                st.warning(f"Low correlation (avg: {avg_corr:.2f})")

        # Pairwise analysis
        st.subheader("Pairwise Relationships")
        results = analyzer.analyze_all_pairs()
        if results:
            summary_data = []
            for result in results:
                relationship_display = {
                    'upstream': 'A upstream of B', 'downstream': 'A downstream of B',
                    'parallel': 'Parallel/Same timing', 'unknown': 'Undetermined'
                }.get(result.relationship, result.relationship)

                summary_data.append({
                    'Site A': result.site_a, 'Site B': result.site_b,
                    'Correlation': f"{result.correlation:.3f}",
                    'Lag (days)': result.lag_days,
                    'Relationship': relationship_display,
                    'Observations': f"{result.n_observations:,}"
                })

            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)


def _nwm_comparison(inventory_df):
    """Compare USGS observations with National Water Model forecasts."""
    site_id = site_picker(inventory_df, key="nwm", label="Select Site",
                          location="main", show_search=True)

    site_info = get_cached_site_info(site_id)
    if not site_info:
        st.error(f"Site {site_id} not found")
        return

    desc = site_info.get('description', site_id)

    st.markdown("The **National Water Model (NWM)** produces streamflow forecasts for the entire US river network.")

    mode = st.radio("Comparison Mode", ["Recent (API)", "Retrospective (S3)"],
                    horizontal=True, key="nwm_mode")

    if mode == "Recent (API)":
        days_back = st.slider("Days to compare", 7, 90, 30, key="nwm_days")

        if st.button("Compare with NWM", type="primary", key="gen_nwm"):
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            with st.spinner("Fetching NWM data..."):
                comparison = compare_nwm_usgs(
                    site_id, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
                )

            if comparison is None:
                st.error("Could not complete comparison. Site may not be in NWM network.")
                return

            st.subheader("Model Performance")
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("Nash-Sutcliffe", f"{comparison.nash_sutcliffe:.3f}",
                         help="NSE: Model efficiency. 1.0 = perfect, 0 = no better than using the mean, <0 = worse than the mean. >0.5 is generally acceptable")
            with col_b:
                st.metric("Correlation", f"{comparison.correlation:.3f}",
                         help="Pearson correlation (r): how well two time series track each other. 1.0 = perfect, 0 = none")
            with col_c:
                st.metric("RMSE", f"{comparison.rmse:.1f} cfs",
                         help="Root Mean Square Error: average magnitude of errors in cfs. Lower is better. Sensitive to outliers")
            with col_d:
                st.metric("Bias", f"{comparison.bias:+.1f} cfs",
                         help="Mean difference between modeled and observed. Positive = over-prediction, negative = under-prediction")

            col_e, col_f, col_g = st.columns(3)
            with col_e:
                st.metric("MAE", f"{comparison.mae:.1f} cfs",
                         help="Mean Absolute Error: average error size in cfs, regardless of direction. Less sensitive to outliers than RMSE")
            with col_f:
                st.metric("Percent Bias", f"{comparison.percent_bias:+.1f}%",
                         help="Systematic over/under-prediction as %. 0% = no bias, positive = over-prediction")
            with col_g:
                st.metric("N Observations", comparison.n_observations)

            skill_result = get_forecast_skill(site_id, n_days=days_back)
            if 'rating' in skill_result:
                rating = skill_result['rating']
                icons = {'Excellent': '🟢', 'Good': '🟡', 'Fair': '🟠', 'Poor': '🔴', 'Very Poor': '⚫'}
                st.markdown(f"### {icons.get(rating, '⚪')} Forecast Skill: {rating}")

    else:
        # Retrospective mode
        col1, col2 = st.columns(2)
        with col1:
            retro_start = st.date_input("Start date", value=date(2010, 1, 1), key="retro_start")
        with col2:
            retro_end = st.date_input("End date", value=date(2020, 12, 31), key="retro_end")

        if st.button("Evaluate Retrospective Skill", type="primary", key="gen_retro"):
            client = NWMClient()

            with st.spinner("Fetching NWM retrospective data from S3 (this may take a moment)..."):
                skill = client.evaluate_model_skill(
                    site_id,
                    retro_start.strftime('%Y-%m-%d'),
                    retro_end.strftime('%Y-%m-%d'))

            if skill is None:
                st.error("Could not evaluate model skill. Site may not be in NWM network, "
                        "or xarray/s3fs may not be installed.")
                return

            st.subheader("Retrospective Model Skill")

            # Rating banner
            rating = skill.get('rating', 'Unknown')
            icons = {'Excellent': '🟢', 'Good': '🟡', 'Fair': '🟠', 'Poor': '🔴', 'Very Poor': '⚫'}
            st.markdown(f"### {icons.get(rating, '⚪')} Model Skill: {rating}")

            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("NSE", f"{skill['nse']:.3f}",
                         help="Nash-Sutcliffe Efficiency: 1.0 = perfect match, 0 = as good as the mean, <0 = worse than the mean")
            with col_b:
                st.metric("KGE", f"{skill['kge']:.3f}",
                         help="Kling-Gupta Efficiency: combines correlation, variability bias, and mean bias. 1.0 = perfect, >0.5 is generally good")
            with col_c:
                st.metric("RMSE", f"{skill['rmse']:.1f} cfs",
                         help="Root Mean Square Error: average magnitude of errors in cfs. Lower is better. Sensitive to outliers")
            with col_d:
                st.metric("Correlation", f"{skill['correlation']:.3f}",
                         help="Pearson correlation (r): how well two time series track each other. 1.0 = perfect, 0 = none")

            col_e, col_f, col_g, col_h = st.columns(4)
            with col_e:
                st.metric("Percent Bias", f"{skill['percent_bias']:+.1f}%",
                         help="Systematic over/under-prediction as %. 0% = no bias, positive = over-prediction")
            with col_f:
                st.metric("MAE", f"{skill['mae']:.1f} cfs",
                         help="Mean Absolute Error: average error size in cfs, regardless of direction. Less sensitive to outliers than RMSE")
            with col_g:
                st.metric("N Days", skill['n_observations'])
            with col_h:
                st.metric("Bias", f"{skill['bias']:+.1f} cfs",
                         help="Mean difference between modeled and observed. Positive = over-prediction, negative = under-prediction")

            # KGE components
            st.caption(
                f"KGE components: r={skill['correlation']:.3f}, "
                f"alpha={skill['alpha']:.3f} (variability ratio), "
                f"beta={skill['beta']:.3f} (bias ratio)"
            )

            # Fetch data for overlay plot
            with st.spinner("Generating comparison plot..."):
                nwm_retro = client.get_retrospective_streamflow(
                    site_id,
                    retro_start.strftime('%Y-%m-%d'),
                    retro_end.strftime('%Y-%m-%d'))
                usgs_data = fetch_daily_values(
                    site_id, param_cd='00060',
                    start_date=retro_start.strftime('%Y-%m-%d'),
                    end_date=retro_end.strftime('%Y-%m-%d'))

            if nwm_retro is not None and usgs_data is not None:
                # Time series overlay
                nwm_daily = nwm_retro['streamflow_cfs'].resample('D').mean()

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=usgs_data.index, y=usgs_data['value'],
                    mode='lines', name='USGS Observed',
                    line=dict(color='#1f77b4', width=1.5),
                    hovertemplate='%{x|%Y-%m-%d}<br>Observed: %{y:,.0f} cfs<extra></extra>'))
                fig.add_trace(go.Scatter(
                    x=nwm_daily.index, y=nwm_daily.values,
                    mode='lines', name='NWM Retrospective',
                    line=dict(color='#ff7f0e', width=1.5, dash='dot'),
                    hovertemplate='%{x|%Y-%m-%d}<br>NWM: %{y:,.0f} cfs<extra></extra>'))
                fig.update_layout(
                    title=f"{desc} - NWM Retrospective vs USGS",
                    xaxis_title="Date", yaxis_title="Discharge (cfs)",
                    yaxis_type="log", height=450,
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig, use_container_width=True, key="retro_overlay")

                # Residual analysis
                usgs_daily = usgs_data.copy()
                if usgs_daily.index.tz is not None:
                    usgs_daily.index = usgs_daily.index.tz_localize(None)
                usgs_daily.index = usgs_daily.index.normalize()

                merged = pd.merge(
                    usgs_daily[['value']].rename(columns={'value': 'observed'}),
                    nwm_daily.rename('simulated').to_frame(),
                    left_index=True, right_index=True, how='inner'
                ).dropna()

                if len(merged) > 30:
                    merged['residual'] = merged['simulated'] - merged['observed']
                    merged['residual_pct'] = merged['residual'] / merged['observed'] * 100

                    fig_resid = go.Figure()
                    fig_resid.add_trace(go.Scatter(
                        x=merged.index, y=merged['residual_pct'],
                        mode='markers', name='Residual',
                        marker=dict(size=3, color='#9467bd', opacity=0.5),
                        hovertemplate='%{x|%Y-%m-%d}<br>%{y:+.1f}%<extra></extra>'))
                    fig_resid.add_hline(y=0, line_color="gray", line_width=1)
                    fig_resid.update_layout(
                        title="Residual Analysis (NWM - Observed)",
                        xaxis_title="Date",
                        yaxis_title="Residual (% of observed)",
                        yaxis=dict(range=[-200, 200]),
                        height=300,
                        margin=dict(l=60, r=20, t=40, b=40))
                    st.plotly_chart(fig_resid, use_container_width=True, key="retro_residual")


def _flood_animation(inventory_df):
    """Animated replay of historical flood events."""
    site_id = site_picker(inventory_df, key="flood", label="Select Primary Site",
                          location="main", show_search=True)

    site_info = get_cached_site_info(site_id)
    if not site_info:
        st.error(f"Site {site_id} not found")
        return

    desc = site_info.get('description', site_id)

    col1, col2, col3 = st.columns(3)
    with col1:
        distance_km = st.slider("Search Distance (km)", 25, 200, 100, key="flood_dist")
    with col2:
        days_before = st.slider("Days Before Peak", 2, 10, 5, key="flood_before")
    with col3:
        days_after = st.slider("Days After Peak", 5, 20, 10, key="flood_after")

    analyzer = FloodEventAnalyzer(site_id)

    with st.spinner("Fetching peak streamflow data..."):
        events = analyzer.get_top_events(n=10, min_year=1980)

    if not events:
        st.warning("No peak streamflow data found for this site.")
        return

    event_options = []
    for e in events:
        date_str = e.peak_date.strftime('%Y-%m-%d') if e.peak_date else 'Unknown'
        event_options.append(f"{date_str} - {e.peak_discharge_cfs:,.0f} cfs (Rank #{e.rank})")

    selected_idx = st.selectbox("Select Flood Event", range(len(event_options)),
                                format_func=lambda i: event_options[i], key="flood_event")
    selected_event = events[selected_idx]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Peak Discharge", f"{selected_event.peak_discharge_cfs:,.0f} cfs")
    with col2:
        st.metric("Date", selected_event.peak_date.strftime('%Y-%m-%d') if selected_event.peak_date else "Unknown")
    with col3:
        st.metric("Rank", f"#{selected_event.rank} of all time")

    if st.button("Generate Animation", type="primary", key="gen_flood"):
        with st.spinner("Fetching data for all sites..."):
            animation = analyzer.prepare_animation(
                selected_event, days_before=days_before, days_after=days_after,
                distance_km=distance_km, frame_interval_minutes=180, max_sites=8
            )

        if not animation.sites:
            st.error("Could not fetch data for animation.")
            return

        stats = calculate_event_statistics(animation)
        st.write(f"**Sites with data:** {stats['sites_with_data']} of {stats['total_sites']}")

        fig = go.Figure()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        for i, site in enumerate(animation.sites):
            if site.data is not None and not site.data.empty:
                color = colors[i % len(colors)]
                label = f"{site.site_id}"
                if site.direction and site.direction != 'origin':
                    label += f" ({site.direction[:2].upper()})"

                fig.add_trace(go.Scatter(
                    x=site.data.index, y=site.data['value'],
                    mode='lines', name=label,
                    line=dict(color=color, width=2)))

                if site.peak_time and site.peak_value:
                    fig.add_trace(go.Scatter(
                        x=[site.peak_time], y=[site.peak_value],
                        mode='markers', marker=dict(size=12, color=color, symbol='star'),
                        name=f"{site.site_id} Peak", showlegend=False
                    ))

        if selected_event.peak_date:
            peak_str = selected_event.peak_date.strftime('%Y-%m-%d %H:%M:%S')
            fig.add_shape(type="line", x0=peak_str, x1=peak_str, y0=0, y1=1,
                         yref="paper", line=dict(color="red", width=2, dash="dash"))

        fig.update_layout(
            title=f"Flood Event: {selected_event.peak_date.strftime('%Y-%m-%d') if selected_event.peak_date else 'Unknown'}",
            xaxis_title="Date/Time", yaxis_title="Discharge (cfs)",
            yaxis_type="log", height=500, hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)


def _watershed_view(inventory_df):
    """National watershed view with all US USGS gages, plus HyRiver basin boundary."""
    # Basin boundary section (HyRiver)
    with st.expander("Basin Boundary & Characteristics (HyRiver)", expanded=False):
        basin_site = site_picker(inventory_df, key="basin_hr", label="Select Site for Basin",
                                 location="main", show_search=True)

        if st.button("Load Basin Data", type="primary", key="load_basin"):
            try:
                from hydrology.data.hyriver import get_watershed_boundary, get_basin_characteristics, get_nid_dams

                with st.spinner("Fetching watershed boundary..."):
                    boundary = get_watershed_boundary(basin_site)
                    chars = get_basin_characteristics(basin_site)
                    dams = get_nid_dams(basin_site, distance_km=50)

                if boundary is not None and not boundary.empty:
                    # Basin characteristics cards
                    if chars:
                        char_cols = st.columns(4)
                        with char_cols[0]:
                            area = chars.get('drainage_area_sq_km', 0)
                            st.metric("Drainage Area", f"{area:,.1f} km2")
                        with char_cols[1]:
                            elev = chars.get('elevation_mean_m')
                            st.metric("Mean Elevation", f"{elev:,.0f} m" if elev else "N/A")
                        with char_cols[2]:
                            elev_min = chars.get('elevation_min_m')
                            elev_max = chars.get('elevation_max_m')
                            if elev_min is not None and elev_max is not None:
                                st.metric("Elevation Range", f"{elev_max - elev_min:,.0f} m")
                            else:
                                st.metric("Elevation Range", "N/A")
                        with char_cols[3]:
                            n_dams = len(dams) if dams is not None else 0
                            st.metric("Nearby Dams", n_dams)

                    # Render basin boundary on folium map
                    import folium
                    from streamlit_folium import st_folium

                    centroid = boundary.geometry.centroid.iloc[0]
                    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=10,
                                  tiles='CartoDB dark_matter')

                    # Add boundary polygon
                    folium.GeoJson(
                        boundary.to_json(),
                        name="Watershed Boundary",
                        style_function=lambda x: {
                            'fillColor': '#3388ff',
                            'color': '#3388ff',
                            'weight': 2,
                            'fillOpacity': 0.15,
                        }).add_to(m)

                    # Add dams if available
                    if dams is not None and not dams.empty:
                        for _, dam in dams.head(50).iterrows():
                            dam_lat = dam.get('latitude') or dam.geometry.y if hasattr(dam, 'geometry') else None
                            dam_lon = dam.get('longitude') or dam.geometry.x if hasattr(dam, 'geometry') else None
                            if dam_lat and dam_lon:
                                dam_name = dam.get('dam_name', dam.get('name', 'Unknown'))
                                folium.CircleMarker(
                                    location=[dam_lat, dam_lon],
                                    radius=6, color='red', fill=True, fillOpacity=0.8,
                                    tooltip=str(dam_name)).add_to(m)

                    # Add site marker
                    site_info = get_cached_site_info(basin_site)
                    if site_info and site_info.get('latitude'):
                        folium.Marker(
                            location=[float(site_info['latitude']), float(site_info['longitude'])],
                            tooltip=f"USGS {basin_site}",
                            icon=folium.Icon(color='green', icon='tint', prefix='fa')).add_to(m)

                    folium.LayerControl().add_to(m)
                    st_folium(m, width=None, height=450)

                    # Land cover breakdown
                    if chars and 'land_cover' in chars:
                        st.subheader("Land Cover")
                        lc = chars['land_cover']
                        if isinstance(lc, dict):
                            lc_df = pd.DataFrame(list(lc.items()), columns=['Class', 'Percentage'])
                            lc_df = lc_df.sort_values('Percentage', ascending=False).head(10)
                            st.dataframe(lc_df, use_container_width=True, hide_index=True)
                else:
                    st.warning("Could not retrieve watershed boundary. pynhd may not be installed.")

            except ImportError:
                st.error("HyRiver packages not installed. Install with: "
                        "conda install -c conda-forge pygeohydro pynhd py3dep")
            except Exception as e:
                st.error(f"Error loading basin data: {e}")

    st.markdown("---")
    st.subheader("National Site Inventory")

    view_level = st.radio("View Level", ["National Overview", "By Region", "By State"],
                          horizontal=True, key="watershed_view_level")

    selected_huc2 = None
    selected_state = None

    if view_level == "By Region":
        region_options = {f"{huc2} - {info['name']}": huc2
                         for huc2, info in sorted(HUC2_REGIONS.items())}
        selected_region = st.selectbox("Select HUC-2 Region", list(region_options.keys()), key="watershed_region")
        selected_huc2 = region_options[selected_region]

    elif view_level == "By State":
        all_states = set()
        for info in HUC2_REGIONS.values():
            all_states.update(info.get('states', []))
        selected_state = st.selectbox("Select State", sorted(all_states), key="watershed_state")

    col1, col2 = st.columns([2, 1])
    with col1:
        load_button = st.button("Load Site Inventory", type="primary", key="load_watershed")
    with col2:
        force_refresh = st.checkbox("Force refresh", value=False)

    if load_button or 'watershed_data' in st.session_state:
        with st.spinner("Loading inventory..."):
            try:
                if view_level == "National Overview":
                    sites_df = get_national_inventory(force_refresh=force_refresh)
                elif view_level == "By Region" and selected_huc2:
                    sites_df = get_region_inventory(selected_huc2, force_refresh=force_refresh)
                elif view_level == "By State" and selected_state:
                    full_df = get_national_inventory(force_refresh=force_refresh)
                    sites_df = full_df[full_df['state_cd'] == selected_state] if not full_df.empty else pd.DataFrame()
                else:
                    sites_df = pd.DataFrame()

                st.session_state['watershed_data'] = sites_df
            except Exception as e:
                st.error(f"Failed to load inventory: {e}")
                return

        if 'watershed_data' not in st.session_state or st.session_state['watershed_data'].empty:
            st.warning("No site data available.")
            return

        sites_df = st.session_state['watershed_data']

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Sites", f"{len(sites_df):,}")
        with col2:
            if 'huc2' in sites_df.columns:
                st.metric("Regions", sites_df['huc2'].nunique())
        with col3:
            if 'state_cd' in sites_df.columns:
                st.metric("States", sites_df['state_cd'].nunique())

        # Map
        max_map_sites = st.slider("Max sites to display", 100, 5000, 1000, step=100, key="ws_max_sites")
        map_df = sites_df.sample(n=min(max_map_sites, len(sites_df)), random_state=42) if len(sites_df) > max_map_sites else sites_df
        map_df = map_df.dropna(subset=['latitude', 'longitude'])

        if not map_df.empty:
            if view_level == "By Region" and selected_huc2:
                center = get_region_center(selected_huc2)
                zoom = HUC2_REGIONS.get(selected_huc2, {}).get('zoom', 6)
            elif view_level == "By State":
                center = [map_df['latitude'].mean(), map_df['longitude'].mean()]
                zoom = 6
            else:
                center = US_CENTER
                zoom = 4

            import folium
            from folium.plugins import MarkerCluster
            from streamlit_folium import st_folium

            m = folium.Map(location=center, zoom_start=zoom, tiles='CartoDB dark_matter')
            marker_cluster = MarkerCluster(name="USGS Sites")
            for _, row in map_df.iterrows():
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=5, color='#1f77b4', fill=True, fill_opacity=0.7,
                    tooltip=str(row.get('site_id', ''))
                ).add_to(marker_cluster)
            marker_cluster.add_to(m)
            st_folium(m, width=None, height=500)

        # Site table
        st.subheader("Site List")
        search = st.text_input("Search", placeholder="Enter site ID or name...", key="ws_search")
        display_df = sites_df.copy()
        if search:
            mask = (
                display_df['site_id'].astype(str).str.contains(search, case=False, na=False) |
                display_df['site_name'].astype(str).str.contains(search, case=False, na=False)
            )
            display_df = display_df[mask]

        display_cols = [c for c in ['site_id', 'site_name', 'state_cd', 'huc2', 'drainage_area', 'begin_date']
                       if c in display_df.columns]
        st.dataframe(display_df[display_cols].head(500), use_container_width=True, hide_index=True)
        if len(display_df) > 500:
            st.caption(f"Showing first 500 of {len(display_df):,} sites")
