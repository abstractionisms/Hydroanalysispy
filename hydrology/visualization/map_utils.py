"""
Enhanced watershed map utilities for the Hydrology Dashboard.

Provides folium maps with watershed boundary polygons, flowline traces,
dam overlays, and condition-based site coloring (percentile-based).

Example:
    >>> from hydrology.visualization.map_utils import create_watershed_map
    >>> m = create_watershed_map('12422500', show_boundary=True, show_flowlines=True)
"""

from typing import Dict, List, Optional, Any
import numpy as np

from ..core.logging_setup import get_logger

logger = get_logger(__name__)


def get_condition_color(percentile: float) -> str:
    """
    Map a seasonal percentile to a condition color.

    Uses USGS-style streamflow condition colors.

    Args:
        percentile: Flow percentile (0-100)

    Returns:
        Hex color string
    """
    if percentile is None or np.isnan(percentile):
        return '#808080'  # gray for unknown

    if percentile >= 90:
        return '#00008B'   # dark blue: much above normal
    elif percentile >= 75:
        return '#00BFFF'   # light blue: above normal
    elif percentile >= 25:
        return '#00CC00'   # green: normal
    elif percentile >= 10:
        return '#FF8C00'   # orange: below normal
    else:
        return '#FF0000'   # red: much below normal


def get_condition_label(percentile: float) -> str:
    """Get condition label for a percentile value."""
    if percentile is None or np.isnan(percentile):
        return 'Unknown'
    if percentile >= 90:
        return 'Much Above Normal'
    elif percentile >= 75:
        return 'Above Normal'
    elif percentile >= 25:
        return 'Normal'
    elif percentile >= 10:
        return 'Below Normal'
    else:
        return 'Much Below Normal'


def create_watershed_map(
    site_id: str,
    show_boundary: bool = True,
    show_flowlines: bool = False,
    show_dams: bool = False,
    site_info: Optional[Dict] = None,
    additional_sites: Optional[List[Dict]] = None,
) -> Optional[Any]:
    """
    Create an enhanced folium map with watershed features.

    Args:
        site_id: Primary USGS site identifier
        show_boundary: Overlay watershed boundary polygon
        show_flowlines: Show upstream flowline traces
        show_dams: Show nearby dams from NID
        site_info: Dict with latitude/longitude for the primary site
        additional_sites: List of dicts with site_id, latitude, longitude,
            and optionally 'percentile' for condition coloring

    Returns:
        folium.Map object, or None if folium not available
    """
    try:
        import folium
        from folium.plugins import MarkerCluster
    except ImportError:
        logger.error("folium not installed")
        return None

    # Determine map center
    center = [47.6, -117.4]  # Default: Spokane area
    zoom = 9

    if site_info and site_info.get('latitude') and site_info.get('longitude'):
        center = [float(site_info['latitude']), float(site_info['longitude'])]

    m = folium.Map(location=center, zoom_start=zoom, tiles='CartoDB positron', control_scale=True)

    # Watershed boundary
    if show_boundary:
        try:
            from ..data.hyriver import get_watershed_boundary
            boundary = get_watershed_boundary(site_id)
            if boundary is not None and not boundary.empty:
                folium.GeoJson(
                    boundary.to_json(),
                    name="Watershed Boundary",
                    style_function=lambda x: {
                        'fillColor': '#3388ff',
                        'color': '#3388ff',
                        'weight': 2,
                        'fillOpacity': 0.12,
                    },
                    tooltip=f"Watershed: USGS {site_id}",
                ).add_to(m)

                # Fit map to boundary bounds
                bounds = boundary.total_bounds  # [minx, miny, maxx, maxy]
                m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

        except ImportError:
            logger.debug("HyRiver not available for boundary")
        except Exception as e:
            logger.warning(f"Could not add boundary: {e}")

    # Flowlines
    if show_flowlines:
        try:
            from ..data.hyriver import get_flowlines
            flowlines = get_flowlines(site_id, distance_km=50)
            if flowlines is not None and not flowlines.empty:
                folium.GeoJson(
                    flowlines.to_json(),
                    name="Flowlines",
                    style_function=lambda x: {
                        'color': '#4488cc',
                        'weight': 2,
                        'opacity': 0.6,
                    },
                ).add_to(m)

        except ImportError:
            logger.debug("HyRiver not available for flowlines")
        except Exception as e:
            logger.warning(f"Could not add flowlines: {e}")

    # Dams
    if show_dams:
        try:
            from ..data.hyriver import get_nid_dams
            dams = get_nid_dams(site_id, distance_km=50)
            if dams is not None and not dams.empty:
                dam_group = folium.FeatureGroup(name="Dams")
                for _, dam in dams.head(50).iterrows():
                    dam_lat = dam.get('latitude') or (dam.geometry.y if hasattr(dam, 'geometry') else None)
                    dam_lon = dam.get('longitude') or (dam.geometry.x if hasattr(dam, 'geometry') else None)
                    if dam_lat and dam_lon:
                        dam_name = str(dam.get('dam_name', dam.get('name', 'Dam')))
                        folium.CircleMarker(
                            location=[float(dam_lat), float(dam_lon)],
                            radius=5, color='#d62728', fill=True, fillOpacity=0.7,
                            tooltip=dam_name,
                        ).add_to(dam_group)
                dam_group.add_to(m)

        except ImportError:
            logger.debug("HyRiver not available for dams")
        except Exception as e:
            logger.warning(f"Could not add dams: {e}")

    # Primary site marker
    if site_info and site_info.get('latitude'):
        folium.Marker(
            location=[float(site_info['latitude']), float(site_info['longitude'])],
            tooltip=f"USGS {site_id} (selected)",
            icon=folium.Icon(color='red', icon='tint', prefix='fa'),
        ).add_to(m)

    # Additional sites with condition coloring
    if additional_sites:
        site_group = folium.FeatureGroup(name="Monitoring Sites")
        for site in additional_sites:
            lat = site.get('latitude')
            lon = site.get('longitude')
            if not lat or not lon:
                continue

            pctile = site.get('percentile')
            color = get_condition_color(pctile)
            label = get_condition_label(pctile)
            sid = site.get('site_id', '')
            desc = site.get('description', '')[:40]

            tooltip = f"<b>{sid}</b><br>{desc}<br>{label}"
            if pctile is not None and not np.isnan(pctile):
                tooltip += f" ({pctile:.0f}th percentile)"

            folium.CircleMarker(
                location=[float(lat), float(lon)],
                radius=6, color=color, fill=True, fillColor=color, fillOpacity=0.8,
                tooltip=tooltip, weight=1,
            ).add_to(site_group)
        site_group.add_to(m)

    folium.LayerControl().add_to(m)
    return m


def add_condition_legend(m) -> None:
    """Add a condition color legend to a folium map."""
    try:
        import folium
    except ImportError:
        return

    legend_html = """
    <div style="position: fixed; bottom: 30px; right: 10px; z-index: 1000;
                background-color: white; border: 2px solid grey; border-radius: 5px;
                padding: 8px; font-size: 11px; opacity: 0.9;">
        <b>Streamflow Condition</b><br>
        <span style="color: #00008B;">&#9679;</span> Much Above Normal (>90th)<br>
        <span style="color: #00BFFF;">&#9679;</span> Above Normal (75-90th)<br>
        <span style="color: #00CC00;">&#9679;</span> Normal (25-75th)<br>
        <span style="color: #FF8C00;">&#9679;</span> Below Normal (10-25th)<br>
        <span style="color: #FF0000;">&#9679;</span> Much Below Normal (<10th)<br>
        <span style="color: #808080;">&#9679;</span> Unknown
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
