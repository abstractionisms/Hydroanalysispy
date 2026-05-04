"""
Overview page - KPI cards, station map, and condition summary.
"""

import streamlit as st
import pandas as pd
import numpy as np
import importlib.util
from datetime import datetime, timedelta, date
import plotly.graph_objects as go
# Deferred imports: folium/streamlit_folium can't be imported outside Streamlit runtime

from hydrology.app.shared import (
    get_inventory, get_cached_site_info, get_weather_station_info,
    extract_site_id, site_picker,
    fetch_discharge_data, process_site_data,
    find_availability_windows, format_availability_windows,
    render_export_buttons, get_site_conditions,
    logger)
from hydrology.app.styles import (
    render_site_header, render_availability_badges, render_metric_cards, render_insight_board
)
from hydrology.app.interpretation import summarize_flow_context
from hydrology.data.usgs import (
    fetch_daily_values, fetch_instantaneous_values,
    DEFAULT_PARAM_DISCHARGE, DEFAULT_PARAM_STAGE
)
from hydrology.core import DEFAULT_DISCHARGE_CODE

def _mini_sparkline(series, height=50):
    """Create a tiny sparkline Plotly figure."""
    if series is None or len(series) < 2:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=series.values,
        mode='lines',
        line=dict(color='#4ecdc4', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(78, 205, 196, 0.1)',
        hoverinfo='skip'))
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False)
    return fig



# Priority sites for regional summary
LOCAL_SITES = {
    "12422500": "Spokane River at Spokane",
    "12424000": "Hangman Creek at Spokane",
    "12419000": "Spokane River nr Post Falls",
    "12422000": "Spokane River bl N Greene St",
    "12431000": "Little Spokane River at Dartford",
}


def _render_regional_summary():
    """Show a compact conditions table for local/priority sites."""
    from hydrology.data.usgs import fetch_current_conditions, fetch_daily_percentiles, classify_condition
    from hydrology.visualization.map_utils import get_condition_color, get_condition_label

    site_ids = list(LOCAL_SITES.keys())
    current = fetch_current_conditions(site_ids)
    percentiles = fetch_daily_percentiles(site_ids)

    rows = []
    for sid, name in LOCAL_SITES.items():
        flow = current.get(sid)
        pcts = percentiles.get(sid)
        pctile = classify_condition(flow, pcts) if flow and pcts else None
        label = get_condition_label(pctile) if pctile is not None else "N/A"
        color = get_condition_color(pctile) if pctile is not None else "#808080"

        rows.append({
            "Site": name,
            "Flow (cfs)": f"{flow:,.0f}" if flow else "N/A",
            "Condition": label,
            "_color": color,
            "_site_id": sid,
        })

    if not rows:
        return

    # Render as styled cards in columns
    cols = st.columns(len(rows))
    for col, row in zip(cols, rows):
        with col:
            st.markdown(
                f'<div style="border-left: 3px solid {row["_color"]}; '
                f'padding-left: 0.5rem; margin-bottom: 0.5rem;">'
                f'<div style="font-size: 0.75rem; color: #8899a6;">{row["Site"]}</div>'
                f'<div style="font-size: 1.1rem; font-weight: 600; color: #e0e0e0;">{row["Flow (cfs)"]}</div>'
                f'<div style="font-size: 0.7rem; color: {row["_color"]};">{row["Condition"]}</div>'
                f'</div>',
                unsafe_allow_html=True
            )




def show():
    """Render the Overview page."""
    inventory_df = get_inventory()
    if inventory_df.empty:
        st.error("Could not load site inventory")
        return

    # Sidebar: site picker with search
    st.sidebar.header("Select Site")
    site_id = site_picker(inventory_df, key="overview", label="Site", location="sidebar")

    site_info = get_cached_site_info(site_id)
    if not site_info:
        st.error(f"Site {site_id} not found")
        return

    lat = site_info.get('latitude')
    lon = site_info.get('longitude')
    desc = site_info.get('description', site_id)

    st.sidebar.caption(f"Selected: `{site_id}`")
    st.sidebar.caption(desc)

    _render_site_workspace(site_id, desc, float(lat) if lat else None, float(lon) if lon else None)

    st.markdown("---")

    # Main area
    render_site_header(site_id, desc, float(lat) if lat else None, float(lon) if lon else None)

    # Deep-link to full analysis
    if st.button("Full Analysis →", type="secondary", key="deep_link_analysis"):
        st.query_params["site"] = site_id
        st.switch_page(st.session_state["_page_single_analysis"])

    # KPI row - current conditions
    df_hist = _render_kpi_row(site_id, site_info)
    if df_hist is not None and not df_hist.empty:
        st.subheader("Current Interpretation")
        st.caption("Fast context from the last 10 years of daily discharge.")
        render_insight_board(summarize_flow_context(df_hist))

    # Quick Stats summary table
    _render_quick_stats(df_hist)

    st.markdown("---")

    # Station map
    _render_station_map(inventory_df, site_id)


def _render_site_workspace(site_id: str, desc: str, lat: float | None, lon: float | None):
    """Render a visible main-page site workspace instead of burying context in the sidebar."""
    st.subheader("Site Workspace")
    st.caption("Search in the sidebar, then use these actions to move through the analysis workflow.")

    col_site, col_actions = st.columns([2, 1])
    with col_site:
        coord_text = f"{lat:.4f}, {lon:.4f}" if lat is not None and lon is not None else "coordinates unavailable"
        st.markdown(f"**{desc}**")
        st.caption(f"USGS `{site_id}` | {coord_text}")
    with col_actions:
        if st.button("Open Single Analysis", type="primary", use_container_width=True, key="overview_open_single"):
            st.query_params["site"] = site_id
            st.switch_page(st.session_state["_page_single_analysis"])


def _render_kpi_row(site_id: str, site_info: dict) -> "pd.DataFrame | None":
    """Render KPI cards with current conditions and sparklines.

    Returns the historical daily-values DataFrame (or None) so callers
    can compute additional statistics without a second fetch.
    """
    lat = site_info.get('latitude')
    lon = site_info.get('longitude')

    # Parallel fetch: IV (7-day) and DV (10-year) simultaneously
    from concurrent.futures import ThreadPoolExecutor

    end_date = datetime.now()
    iv_start = (end_date - timedelta(days=7)).strftime('%Y-%m-%d')
    hist_start = (end_date - timedelta(days=365 * 10)).strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    # Check session_state for stale data to show immediately
    stale_key = f"kpi_stale_{site_id}"
    if stale_key in st.session_state:
        df_iv, df_hist = st.session_state[stale_key]
        st.caption("Refreshing data...")
    else:
        df_iv, df_hist = None, None

    with st.spinner("Loading current conditions..."):
        def _fetch_iv():
            try:
                return fetch_instantaneous_values(
                    site_id, param_cd=DEFAULT_PARAM_DISCHARGE,
                    start_date=iv_start, end_date=end_str)
            except Exception:
                return None

        def _fetch_hist():
            try:
                return fetch_daily_values(
                    site_id, param_cd=DEFAULT_PARAM_DISCHARGE,
                    start_date=hist_start, end_date=end_str)
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=2) as executor:
            fut_iv = executor.submit(_fetch_iv)
            fut_hist = executor.submit(_fetch_hist)
            df_iv = fut_iv.result()
            df_hist = fut_hist.result()

        # Cache for stale-while-revalidate on next page visit
        st.session_state[stale_key] = (df_iv, df_hist)

    # Build KPI columns
    col1, col2, col3, col4 = st.columns(4)

    # Current flow
    current_val = None
    with col1:
        if df_iv is not None and not df_iv.empty:
            current_val = df_iv['value'].iloc[-1]
            latest_time = df_iv.index[-1]

            # 24h change
            if len(df_iv) > 96:  # ~24h of 15-min data
                val_24h_ago = df_iv['value'].iloc[-96]
                delta = current_val - val_24h_ago
                delta_pct = (delta / val_24h_ago * 100) if val_24h_ago > 0 else 0
                st.metric("Current Flow", f"{current_val:,.0f} cfs",
                         delta=f"{delta_pct:+.1f}% (24h)",
                         delta_color="off",
                         help="Latest instantaneous discharge reading from the USGS gage, updated every 15 minutes")
            else:
                st.metric("Current Flow", f"{current_val:,.0f} cfs",
                         help="Latest instantaneous discharge reading from the USGS gage, updated every 15 minutes")
            st.caption(f"Updated {latest_time.strftime('%H:%M %b %d')}")
        elif df_hist is not None and not df_hist.empty:
            # Fallback: use most recent daily value
            current_val = df_hist['value'].iloc[-1]
            latest_time = df_hist.index[-1]
            st.metric("Current Flow", f"{current_val:,.0f} cfs",
                         help="Latest instantaneous discharge reading from the USGS gage, updated every 15 minutes")
            st.caption(f"Daily value: {latest_time.strftime('%b %d, %Y')}")
        else:
            st.metric("Current Flow", "N/A",
                         help="Latest instantaneous discharge reading from the USGS gage, updated every 15 minutes")

    # Seasonal percentile
    with col2:
        if df_hist is not None and not df_hist.empty and current_val is not None:
            current_doy = end_date.timetuple().tm_yday
            df_hist_copy = df_hist.copy()
            df_hist_copy['doy'] = df_hist_copy.index.dayofyear
            seasonal_data = df_hist_copy[
                (df_hist_copy['doy'] >= current_doy - 15) &
                (df_hist_copy['doy'] <= current_doy + 15)
            ]['value']

            if len(seasonal_data) > 10:
                percentile = (seasonal_data < current_val).mean() * 100

                if percentile > 90:
                    status = "Much Above Normal"
                    color = "inverse"
                elif percentile > 75:
                    status = "Above Normal"
                    color = "off"
                elif percentile > 25:
                    status = "Normal"
                    color = "normal"
                elif percentile > 10:
                    status = "Below Normal"
                    color = "off"
                else:
                    status = "Much Below Normal"
                    color = "inverse"

                st.metric("Seasonal Percentile", f"{percentile:.0f}%",
                         delta=status, delta_color=color,
                         help="Where current flow ranks compared to historical flows for this day of year. 50% = median, <10% = much below normal, >90% = much above normal")
            else:
                st.metric("Seasonal Percentile", "N/A",
                         help="Where current flow ranks compared to historical flows for this day of year. 50% = median, <10% = much below normal, >90% = much above normal")
        else:
            st.metric("Seasonal Percentile", "N/A",
                         help="Where current flow ranks compared to historical flows for this day of year. 50% = median, <10% = much below normal, >90% = much above normal")

    # Period of record
    with col3:
        avail = find_availability_windows(site_id, DEFAULT_PARAM_DISCHARGE)
        if avail:
            start_year = avail[0][0]
            end_label = avail[0][1]
            years = (date.today().year - start_year) if end_label == "present" else (end_label - start_year)
            st.metric("Period of Record", f"{years} years",
                     delta=f"Since {start_year}", delta_color="off",
                         help="Total years of continuous discharge data available for this site")
        else:
            st.metric("Period of Record", "N/A",
                         help="Total years of continuous discharge data available for this site")

    # Median comparison
    with col4:
        if df_hist is not None and not df_hist.empty and current_val is not None:
            current_doy = end_date.timetuple().tm_yday
            df_hist_copy = df_hist.copy()
            df_hist_copy['doy'] = df_hist_copy.index.dayofyear
            seasonal = df_hist_copy[
                (df_hist_copy['doy'] >= current_doy - 15) &
                (df_hist_copy['doy'] <= current_doy + 15)
            ]['value']
            if len(seasonal) > 10:
                median_val = seasonal.median()
                pct_of_median = (current_val / median_val * 100) if median_val > 0 else 0
                st.metric("% of Median", f"{pct_of_median:.0f}%",
                         delta=f"Median: {median_val:,.0f} cfs", delta_color="off",
                         help="Current flow as a percentage of the historical median. 100% = exactly normal, <50% = well below normal, >200% = well above normal")
            else:
                st.metric("% of Median", "N/A",
                         help="Current flow as a percentage of the historical median. 100% = exactly normal, <50% = well below normal, >200% = well above normal")
        else:
            st.metric("% of Median", "N/A",
                         help="Current flow as a percentage of the historical median. 100% = exactly normal, <50% = well below normal, >200% = well above normal")

    return df_hist


def _render_quick_stats(df_hist: "pd.DataFrame | None"):
    """Show a compact summary table of key discharge statistics."""
    if df_hist is None or df_hist.empty:
        return

    values = df_hist['value'].dropna()
    if values.empty:
        return

    mean_flow = values.mean()
    median_flow = values.median()
    min_flow = values.min()
    max_flow = values.max()
    std_flow = values.std()
    cv = (std_flow / mean_flow * 100) if mean_flow > 0 else np.nan

    stats = pd.DataFrame({
        "Metric": [
            "Mean Flow", "Median Flow", "Min Flow",
            "Max Flow", "Std Dev", "CV"
        ],
        "Value": [
            f"{mean_flow:,.1f} cfs",
            f"{median_flow:,.1f} cfs",
            f"{min_flow:,.1f} cfs",
            f"{max_flow:,.1f} cfs",
            f"{std_flow:,.1f} cfs",
            f"{cv:,.1f}%",
        ],
    })

    st.subheader("Quick Stats")
    st.caption("Based on the last 10 years of daily discharge data")
    st.dataframe(stats, use_container_width=True, hide_index=True)


def _render_station_map(inventory_df, selected_site_id):
    """Render station map with condition-colored markers and optional watershed overlay."""
    st.subheader("Station Map")
    st.caption("Fast station map loads by default. Basin boundaries, flowlines, dams, and live condition coloring are optional because they call slower external geospatial services.")

    map_data = inventory_df[['latitude', 'longitude', 'site_id', 'description', 'begin_date']].copy()
    map_data = map_data.dropna(subset=['latitude', 'longitude'])
    map_data['latitude'] = map_data['latitude'].astype(float)
    map_data['longitude'] = map_data['longitude'].astype(float)

    show_boundary = False
    show_flowlines = False
    show_dams = False
    color_by_conditions = False
    has_pynhd = importlib.util.find_spec("pynhd") is not None
    has_pygeohydro = importlib.util.find_spec("pygeohydro") is not None

    with st.expander("Optional slow map layers", expanded=False):
        st.caption("These layers may take a while or fail if HyRiver/NLDI services are unavailable.")
        col_opt1, col_opt2, col_opt3, col_opt4 = st.columns(4)
        with col_opt1:
            color_by_conditions = st.checkbox(
                "Color by live flow",
                value=False,
                key="ov_conditions",
                help="Fetches current conditions for all inventory sites. Leave off for fastest map loading.",
            )
        with col_opt2:
            show_boundary = st.checkbox(
                "Watershed boundary",
                value=False,
                key="ov_boundary",
                disabled=not has_pynhd,
                help="Requires HyRiver basin delineation and can be slow.",
            )
            if not has_pynhd:
                st.caption("Unavailable: install `pynhd`.")
        with col_opt3:
            show_flowlines = st.checkbox(
                "Flowlines",
                value=False,
                key="ov_flowlines",
                disabled=not has_pynhd,
                help="Requires NHD/NLDI geospatial services and can be slow.",
            )
            if not has_pynhd:
                st.caption("Unavailable: install `pynhd`.")
        with col_opt4:
            show_dams = st.checkbox(
                "Nearby dams",
                value=False,
                key="ov_dams",
                disabled=not has_pygeohydro,
                help="Requires National Inventory of Dams geospatial lookup.",
            )
            if not has_pygeohydro:
                st.caption("Unavailable: install `pygeohydro`.")

    # Center on selected site if possible
    selected_info = get_cached_site_info(selected_site_id)
    if selected_info and selected_info.get('latitude') and selected_info.get('longitude'):
        center = [float(selected_info['latitude']), float(selected_info['longitude'])]
        zoom = 9
    else:
        center = [map_data['latitude'].mean(), map_data['longitude'].mean()]
        zoom = 6

    # Try enhanced map if any geospatial features requested
    if show_boundary or show_flowlines or show_dams:
        try:
            from hydrology.visualization.map_utils import create_watershed_map, add_condition_legend
            from streamlit_folium import st_folium

            all_site_ids = map_data['site_id'].tolist()
            conditions = get_site_conditions(all_site_ids)

            additional_sites = []
            for _, row in map_data.iterrows():
                if row['site_id'] != selected_site_id:
                    additional_sites.append({
                        'site_id': row['site_id'],
                        'latitude': row['latitude'],
                        'longitude': row['longitude'],
                        'description': str(row.get('description', '')),
                        'percentile': conditions.get(row['site_id']),
                    })

            m = create_watershed_map(
                selected_site_id,
                show_boundary=show_boundary,
                show_flowlines=show_flowlines,
                show_dams=show_dams,
                site_info=selected_info,
                additional_sites=additional_sites)

            if m is not None:
                add_condition_legend(m)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Sites", len(map_data))
                with col2:
                    oldest = 'N/A'
                    try:
                        if 'begin_date' in map_data.columns:
                            dates = map_data['begin_date'].dropna().astype(str)
                            valid = dates[dates.str.match(r'^\d')]
                            if len(valid) > 0:
                                oldest = sorted(valid.tolist())[0][:4]
                    except Exception:
                        pass
                    st.metric("Oldest Record", oldest)
                with col3:
                    st.metric("Region", "Pacific Northwest")

                st_folium(m, width=None, height=500)
                return

        except ImportError:
            st.caption("HyRiver not installed — showing standard map")
        except Exception as e:
            st.caption(f"Enhanced map unavailable: {str(e)[:60]}")

    # Standard map
    import folium
    conditions = {}
    if color_by_conditions:
        all_site_ids = map_data['site_id'].tolist()
        with st.spinner("Loading live flow conditions for map markers..."):
            conditions = get_site_conditions(all_site_ids)
        if conditions:
            st.caption(
                "Marker colors use USGS seasonal percentiles when available; if not, they use relative live-flow rank among mapped sites."
            )
        else:
            st.warning("Live flow coloring is unavailable for these sites right now; showing default station markers.")
    from folium.plugins import MarkerCluster
    from streamlit_folium import st_folium

    m = folium.Map(
        location=center, zoom_start=zoom,
        tiles='CartoDB dark_matter', control_scale=True
    )

    marker_group = MarkerCluster(name="USGS Sites")

    for _, row in map_data.iterrows():
        sid = row['site_id']
        desc = str(row.get('description', ''))[:50]
        lat = row['latitude']
        lon = row['longitude']

        is_selected = (sid == selected_site_id)
        if is_selected:
            color = 'red'
            radius = 8
        else:
            if color_by_conditions:
                from hydrology.visualization.map_utils import get_condition_color
                pctile = conditions.get(sid)
                color = get_condition_color(pctile) if pctile is not None else '#4488cc'
            else:
                color = '#4488cc'
            radius = 5

        tooltip = f"<b>{sid}</b><br>{desc}"
        popup_html = f"""
        <div style="width:200px">
            <b>{sid}</b><br>
            <span style="font-size:11px">{desc}</span><br>
        </div>
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=tooltip,
            color=color, fill=True, fillColor=color,
            fillOpacity=0.8, weight=1 if not is_selected else 3
        ).add_to(marker_group)

    marker_group.add_to(m)
    folium.LayerControl().add_to(m)

    if color_by_conditions:
        from hydrology.visualization.map_utils import add_condition_legend
        add_condition_legend(m)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Sites", len(map_data))
    with col2:
        oldest = 'N/A'
        try:
            if 'begin_date' in map_data.columns:
                dates = map_data['begin_date'].dropna().astype(str)
                valid = dates[dates.str.match(r'^\d')]
                if len(valid) > 0:
                    oldest = sorted(valid.tolist())[0][:4]
        except Exception:
            pass
        st.metric("Oldest Record", oldest)
    with col3:
        st.metric("Region", "Pacific Northwest")

    map_output = st_folium(m, width=None, height=500, returned_objects=["last_object_clicked"])

    # Handle map click — show site info and suggest navigation
    if map_output and map_output.get("last_object_clicked"):
        clicked = map_output["last_object_clicked"]
        clicked_lat = clicked.get("lat")
        clicked_lng = clicked.get("lng")

        if clicked_lat and clicked_lng:
            tolerance = 0.01
            matches = map_data[
                (abs(map_data['latitude'] - clicked_lat) < tolerance) &
                (abs(map_data['longitude'] - clicked_lng) < tolerance)
            ]

            if not matches.empty:
                matches["_click_distance"] = (
                    (matches["latitude"] - clicked_lat) ** 2 +
                    (matches["longitude"] - clicked_lng) ** 2
                )
                site = matches.sort_values("_click_distance").iloc[0]
                clicked_id = str(site["site_id"])
                st.query_params["site"] = clicked_id
                st.success(f"Selected: **{clicked_id}** - {site['description']}")
                if st.button("Open analysis for selected site", type="primary", key="map_open_analysis"):
                    st.switch_page(st.session_state["_page_single_analysis"])
