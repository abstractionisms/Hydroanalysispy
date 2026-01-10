"""
NLDI (Hydro Network-Linked Data Index) integration for river network navigation.

Provides tools for discovering related USGS sites along river networks,
finding upstream/downstream monitoring points, and navigating the hydrologic
network.

API Documentation: https://labs.waterdata.usgs.gov/docs/nldi/about-nldi/index.html

Example:
    >>> from hydrology.data.nldi import discover_related_sites
    >>> sites = discover_related_sites('12422500', direction='both', distance_km=100)
    >>> for site in sites:
    ...     print(f"{site['site_id']}: {site['name']} ({site['direction']}, {site['distance_km']:.1f} km)")
"""

import time
import random
from typing import Dict, List, Optional, Any
import requests
import pandas as pd

from ..core.logging_setup import get_logger

logger = get_logger(__name__)

# NLDI API base URL
NLDI_BASE = "https://api.water.usgs.gov/nldi/linked-data"

# Navigation modes
NAVIGATION_MODES = {
    'upstream_main': 'UM',      # Upstream along main stem
    'upstream_trib': 'UT',      # Upstream including tributaries
    'downstream_main': 'DM',    # Downstream along main stem
    'downstream_div': 'DD',     # Downstream including diversions
}


def http_get_json(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    retries: int = 4,
    base_sleep: float = 0.5,
    timeout: int = 30
) -> Optional[Dict]:
    """
    Fetch JSON data with retry logic.

    Args:
        url: URL to fetch
        params: Query parameters
        retries: Number of retry attempts
        base_sleep: Base sleep time between retries
        timeout: Request timeout in seconds

    Returns:
        Parsed JSON data or None on failure
    """
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 404:
                logger.warning(f"Resource not found: {url}")
                return None
            logger.warning(f"HTTP {r.status_code} for {url}")
            last_err = requests.HTTPError(f"{r.status_code}")
        except requests.RequestException as e:
            last_err = e
            logger.warning(f"HTTP error ({i+1}/{retries}): {e}")

        if i < retries - 1:
            sleep_time = base_sleep * (2 ** i) + random.random() * 0.2
            time.sleep(sleep_time)

    logger.error(f"Failed to fetch {url}: {last_err}")
    return None


def get_site_info(site_id: str) -> Optional[Dict[str, Any]]:
    """
    Get NLDI feature info for a USGS site.

    Args:
        site_id: USGS site identifier (e.g., '12422500')

    Returns:
        Dict with site info including COMID (NWM reach ID), coordinates, etc.
    """
    url = f"{NLDI_BASE}/nwissite/USGS-{site_id}"
    data = http_get_json(url)

    if not data:
        return None

    try:
        features = data.get('features', [])
        if features:
            props = features[0].get('properties', {})
            geom = features[0].get('geometry', {})
            coords = geom.get('coordinates', [None, None])

            return {
                'site_id': site_id,
                'comid': props.get('comid'),
                'name': props.get('name', ''),
                'uri': props.get('uri', ''),
                'longitude': coords[0] if coords else None,
                'latitude': coords[1] if coords else None,
            }
    except (KeyError, IndexError, TypeError) as e:
        logger.error(f"Error parsing NLDI response for {site_id}: {e}")

    return None


def navigate_network(
    site_id: str,
    direction: str = 'downstream_main',
    distance_km: float = 100,
    data_source: str = 'nwissite'
) -> List[Dict[str, Any]]:
    """
    Navigate the hydrologic network from a starting site.

    Args:
        site_id: Starting USGS site ID
        direction: Navigation direction (upstream_main, upstream_trib,
                   downstream_main, downstream_div)
        distance_km: Maximum navigation distance in kilometers
        data_source: Type of features to return ('nwissite' for USGS sites)

    Returns:
        List of discovered features with site info
    """
    nav_code = NAVIGATION_MODES.get(direction, direction)
    url = f"{NLDI_BASE}/nwissite/USGS-{site_id}/navigation/{nav_code}/{data_source}"
    params = {'distance': distance_km}

    data = http_get_json(url, params)

    if not data:
        return []

    results = []
    try:
        features = data.get('features', [])
        for feature in features:
            props = feature.get('properties', {})
            geom = feature.get('geometry', {})
            coords = geom.get('coordinates', [None, None])

            # Extract site ID from identifier (format: USGS-12345678)
            identifier = props.get('identifier', '')
            extracted_id = identifier.replace('USGS-', '') if identifier.startswith('USGS-') else identifier

            # Skip the origin site
            if extracted_id == site_id:
                continue

            results.append({
                'site_id': extracted_id,
                'name': props.get('name', ''),
                'uri': props.get('uri', ''),
                'comid': props.get('comid'),
                'longitude': coords[0] if coords else None,
                'latitude': coords[1] if coords else None,
                'direction': 'upstream' if 'upstream' in direction else 'downstream',
                'navigation_mode': direction,
            })
    except (KeyError, TypeError) as e:
        logger.error(f"Error parsing navigation response: {e}")

    return results


def discover_related_sites(
    site_id: str,
    direction: str = 'both',
    distance_km: float = 100,
    include_tributaries: bool = False,
    max_sites: int = 10
) -> List[Dict[str, Any]]:
    """
    Discover USGS monitoring sites along the same river network.

    Args:
        site_id: Origin USGS site ID
        direction: 'upstream', 'downstream', or 'both'
        distance_km: Maximum distance to search (km)
        include_tributaries: If True, include tributary sites
        max_sites: Maximum number of sites to return

    Returns:
        List of site dictionaries ordered by distance, with fields:
        - site_id: USGS site identifier
        - name: Site name
        - latitude, longitude: Coordinates
        - direction: 'upstream' or 'downstream'
        - distance_km: Approximate distance from origin

    Example:
        >>> sites = discover_related_sites('12422500', direction='both', distance_km=50)
        >>> print(f"Found {len(sites)} related sites")
    """
    all_sites = []

    # Get origin site info for distance calculation
    origin_info = get_site_info(site_id)
    origin_coords = None
    if origin_info and origin_info.get('latitude') and origin_info.get('longitude'):
        origin_coords = (origin_info['latitude'], origin_info['longitude'])

    # Navigate upstream
    if direction in ('upstream', 'both'):
        mode = 'upstream_trib' if include_tributaries else 'upstream_main'
        upstream = navigate_network(site_id, mode, distance_km)
        all_sites.extend(upstream)

    # Navigate downstream
    if direction in ('downstream', 'both'):
        mode = 'downstream_div' if include_tributaries else 'downstream_main'
        downstream = navigate_network(site_id, mode, distance_km)
        all_sites.extend(downstream)

    # Calculate approximate distances
    for site in all_sites:
        if origin_coords and site.get('latitude') and site.get('longitude'):
            site['distance_km'] = _haversine_distance(
                origin_coords[0], origin_coords[1],
                site['latitude'], site['longitude']
            )
        else:
            site['distance_km'] = None

    # Sort by distance and limit
    all_sites.sort(key=lambda s: s.get('distance_km') or 999)

    # Remove duplicates (same site might appear in both directions)
    seen = set()
    unique_sites = []
    for site in all_sites:
        if site['site_id'] not in seen:
            seen.add(site['site_id'])
            unique_sites.append(site)

    return unique_sites[:max_sites]


def get_basin_sites(
    site_id: str,
    distance_km: float = 200
) -> Dict[str, List[Dict]]:
    """
    Get all sites in the basin for a given site (upstream and downstream).

    Args:
        site_id: Origin USGS site ID
        distance_km: Maximum distance to search

    Returns:
        Dict with 'upstream' and 'downstream' site lists
    """
    return {
        'upstream': navigate_network(site_id, 'upstream_main', distance_km),
        'downstream': navigate_network(site_id, 'downstream_main', distance_km),
    }


def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points in kilometers.
    """
    from math import radians, cos, sin, asin, sqrt

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))

    # Earth radius in km
    r = 6371
    return c * r


def order_sites_by_flow(sites: List[Dict], origin_site_id: str) -> List[Dict]:
    """
    Order sites from upstream to downstream along the flow path.

    Args:
        sites: List of site dictionaries (must include 'direction')
        origin_site_id: The origin site ID for reference

    Returns:
        Sites ordered from most upstream to most downstream
    """
    # Split into upstream and downstream
    upstream = [s for s in sites if s.get('direction') == 'upstream']
    downstream = [s for s in sites if s.get('direction') == 'downstream']

    # Sort upstream by distance (furthest first)
    upstream.sort(key=lambda s: s.get('distance_km', 0), reverse=True)

    # Sort downstream by distance (closest first)
    downstream.sort(key=lambda s: s.get('distance_km', 0))

    # Combine: upstream (far to near) + downstream (near to far)
    return upstream + downstream
