"""Shared site-relationship and watershed views used by the live pages."""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
# folium/streamlit_folium imported locally where used (can't import outside Streamlit runtime)

from hydrology.app.shared import get_cached_site_info, site_picker
from hydrology.data.usgs import fetch_daily_values
from hydrology.analysis.multisite import MultiSiteAnalyzer
from hydrology.data.national_inventory import get_national_inventory, get_region_inventory
from hydrology.core.huc_regions import HUC2_REGIONS, get_region_center, US_CENTER
from hydrology.app.styles import mark_folium_slot
from hydrology.visualization.interactive import apply_hydro_theme


def _multisite_analysis(inventory_df):
    """Analyze correlations and relationships between multiple sites."""
    selected_sites = site_picker(inventory_df, key="multisite", label="Select Sites (2-6)",
                                 location="main", multi=True, max_selections=6, show_search=False)

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

    if st.button("Analyze Relationships", type="primary", width="stretch", key="gen_multi"):
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
        fig = go.Figure()
        for col in plot_data.columns:
            fig.add_trace(go.Scatter(
                x=plot_data.index, y=plot_data[col],
                mode="lines", name=str(col),
            ))
        fig.update_layout(
            title="Synchronized time series",
            xaxis_title="Date",
            yaxis_title="Discharge (cfs)",
            height=420,
            hovermode="x unified",
        )
        st.plotly_chart(apply_hydro_theme(fig), width="stretch")

        # Correlation matrix
        st.subheader("Correlation Matrix")
        corr_matrix = analyzer.get_correlation_matrix()
        if not corr_matrix.empty:
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index,
                colorscale='Viridis', zmin=0, zmax=1,
                text=[[f"{val:.3f}" for val in row] for row in corr_matrix.values],
                texttemplate="%{text}", textfont={"size": 12, "color": "#e7eef7"}))
            fig.update_layout(height=350)
            st.plotly_chart(apply_hydro_theme(fig), width="stretch")

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

            st.dataframe(pd.DataFrame(summary_data), width="stretch", hide_index=True)


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

                # Store in session_state so results persist across reruns
                st.session_state["basin_data"] = {
                    "site": basin_site, "boundary": boundary,
                    "chars": chars, "dams": dams,
                }

            except ImportError:
                st.error("HyRiver packages not installed. Install with: "
                        "conda install -c conda-forge pygeohydro pynhd py3dep")
            except Exception as e:
                st.error(f"Error loading basin data: {e}")

        # Render from session_state (persists across reruns)
        basin_data = st.session_state.get("basin_data")
        if basin_data and basin_data.get("site") == basin_site:
            boundary = basin_data["boundary"]
            chars = basin_data["chars"]
            dams = basin_data["dams"]

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
                    from hydrology.visualization.map_utils import add_basemap

                    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=10, tiles=None)
                    add_basemap(m)

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
                    mark_folium_slot()
                    st_folium(m, width=None, height=450)

                    # Land cover breakdown
                    if chars and 'land_cover' in chars:
                        st.subheader("Land Cover")
                        lc = chars['land_cover']
                        if isinstance(lc, dict):
                            lc_df = pd.DataFrame(list(lc.items()), columns=['Class', 'Percentage'])
                            lc_df = lc_df.sort_values('Percentage', ascending=False).head(10)
                            st.dataframe(lc_df, width="stretch", hide_index=True)
            else:
                    st.warning("Could not retrieve watershed boundary. pynhd may not be installed.")

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

            from hydrology.visualization.map_utils import add_basemap

            m = folium.Map(location=center, zoom_start=zoom, tiles=None)
            add_basemap(m)
            marker_cluster = MarkerCluster(name="USGS Sites")
            for _, row in map_df.iterrows():
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=5, color='#1f77b4', fill=True, fill_opacity=0.7,
                    tooltip=str(row.get('site_id', ''))
                ).add_to(marker_cluster)
            marker_cluster.add_to(m)
            mark_folium_slot()
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
        st.dataframe(display_df[display_cols].head(500), width="stretch", hide_index=True)
        if len(display_df) > 500:
            st.caption(f"Showing first 500 of {len(display_df):,} sites")
