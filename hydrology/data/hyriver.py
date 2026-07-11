"""
HyRiver integration for watershed-scale geospatial data access.

Provides watershed boundaries, basin characteristics, gridded climate data,
elevation profiles, and dam locations through the HyRiver suite:
- pynhd: NHDPlus watershed boundaries and flowlines
- pygeohydro: NLCD land cover, NID dams
- pydaymet: Gridded temperature and precipitation
- py3dep: 3DEP elevation data

These complement the existing point-based USGS analysis with watershed context.

Example:
    >>> from hydrology.data.hyriver import get_watershed_boundary, get_basin_characteristics
    >>> boundary = get_watershed_boundary('12422500')
    >>> chars = get_basin_characteristics('12422500')
    >>> print(f"Drainage area: {chars['drainage_area_sq_km']:.1f} km2")
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

import requests

from ..core.logging_setup import get_logger
from ..core.paths import CACHE_DIR, ensure_dir

logger = get_logger(__name__)

# Cache for expensive geospatial lookups
_boundary_cache: Dict[str, Any] = {}
_characteristics_cache: Dict[str, Dict] = {}
_flowlines_cache: Dict[Tuple[str, float], Any] = {}
_navigation_flowlines_cache: Dict[Tuple[str, str, float], Any] = {}
_dams_cache: Dict[Tuple[str, float], Any] = {}


def _get_site_coords(site_id: str) -> Optional[Tuple[float, float]]:
    """Get lat/lon for a USGS site via NLDI."""
    from .nldi import get_site_info
    info = get_site_info(site_id)
    if info and info.get('latitude') and info.get('longitude'):
        return (info['latitude'], info['longitude'])
    return None


def get_watershed_boundary(site_id: str) -> Optional[Any]:
    """
    Get the contributing area (watershed boundary) polygon for a USGS site.

    Uses pynhd NLDI to delineate the upstream drainage basin.

    Args:
        site_id: USGS site identifier (e.g., '12422500')

    Returns:
        geopandas GeoDataFrame with watershed boundary polygon, or None
    """
    if site_id in _boundary_cache:
        return _boundary_cache[site_id]

    try:
        from pynhd import NLDI

        nldi = NLDI()
        basin = nldi.get_basins(f"USGS-{site_id}")

        if basin is not None and not basin.empty:
            _boundary_cache[site_id] = basin
            logger.info(f"Retrieved watershed boundary for {site_id}")
            return basin

        logger.warning(f"No basin boundary found for {site_id}")
        return None

    except ImportError:
        logger.error("pynhd not installed. Install with: conda install -c conda-forge pynhd")
        return None
    except Exception as e:
        logger.error(f"Error fetching watershed boundary for {site_id}: {e}")
        return None


def get_flowlines(site_id: str, distance_km: float = 50) -> Optional[Any]:
    """
    Get NHDPlus flowlines upstream of a USGS site.

    Args:
        site_id: USGS site identifier
        distance_km: Distance upstream to retrieve flowlines

    Returns:
        geopandas GeoDataFrame with flowline geometries, or None
    """
    cache_key = (site_id, float(distance_km))
    if cache_key in _flowlines_cache:
        return _flowlines_cache[cache_key]

    try:
        from pynhd import NLDI

        nldi = NLDI()
        flowlines = nldi.navigate_byid(
            fsource="nwissite",
            fid=f"USGS-{site_id}",
            navigation="upstreamTributaries",
            source="flowlines",
            distance=distance_km,
        )

        if flowlines is not None and not flowlines.empty:
            _flowlines_cache[cache_key] = flowlines
            logger.info(f"Retrieved {len(flowlines)} flowlines for {site_id}")
            return flowlines

        _flowlines_cache[cache_key] = None
        return None

    except ImportError:
        logger.error("pynhd not installed")
        return None
    except Exception as e:
        logger.error(f"Error fetching flowlines for {site_id}: {e}")
        return None


def get_navigation_flowlines(
    site_id: str,
    navigation: str = "upstreamMain",
    distance_km: float = 50,
) -> Optional[Any]:
    """
    Get NHDPlus flowlines for a specific NLDI navigation mode.

    Args:
        site_id: USGS site identifier
        navigation: NLDI navigation mode, such as upstreamMain or upstreamTributaries
        distance_km: Navigation distance to retrieve

    Returns:
        geopandas GeoDataFrame with flowline geometries, or None
    """
    cache_key = (site_id, navigation, float(distance_km))
    if cache_key in _navigation_flowlines_cache:
        return _navigation_flowlines_cache[cache_key]

    try:
        from pynhd import NLDI

        nldi = NLDI()
        flowlines = nldi.navigate_byid(
            fsource="nwissite",
            fid=f"USGS-{site_id}",
            navigation=navigation,
            source="flowlines",
            distance=distance_km,
        )

        if flowlines is not None and not flowlines.empty:
            _navigation_flowlines_cache[cache_key] = flowlines
            logger.info(f"Retrieved {len(flowlines)} {navigation} flowlines for {site_id}")
            return flowlines

        _navigation_flowlines_cache[cache_key] = None
        return None

    except ImportError:
        logger.error("pynhd not installed")
        return None
    except Exception as e:
        logger.error(f"Error fetching {navigation} flowlines for {site_id}: {e}")
        return None


def get_basin_characteristics(site_id: str) -> Optional[Dict[str, Any]]:
    """
    Get basin characteristics including drainage area, land cover, and soil.

    Combines NLDI basin info with NLCD land cover classification.

    Args:
        site_id: USGS site identifier

    Returns:
        Dict with basin characteristics:
        - drainage_area_sq_km: Basin area in square kilometers
        - centroid_lat, centroid_lon: Basin centroid coordinates
        - land_cover: Dict of NLCD land cover percentages
        - elevation_mean_m: Mean basin elevation (if available)
    """
    if site_id in _characteristics_cache:
        return _characteristics_cache[site_id]

    result = {}

    # Get watershed boundary first
    basin = get_watershed_boundary(site_id)
    if basin is None:
        return None

    try:
        import geopandas as gpd

        # Calculate area (reproject to equal-area for accuracy)
        basin_ea = basin.to_crs(epsg=5070)  # CONUS Albers Equal Area
        area_sq_m = basin_ea.geometry.area.iloc[0]
        result['drainage_area_sq_km'] = area_sq_m / 1e6

        # Centroid
        centroid = basin.geometry.centroid.iloc[0]
        result['centroid_lon'] = centroid.x
        result['centroid_lat'] = centroid.y

        # Bounding box
        bounds = basin.total_bounds  # [minx, miny, maxx, maxy]
        result['bbox'] = {
            'west': bounds[0], 'south': bounds[1],
            'east': bounds[2], 'north': bounds[3]
        }

    except Exception as e:
        logger.warning(f"Error computing basic basin stats: {e}")

    # NLCD land cover
    try:
        from pygeohydro import NLCD

        nlcd = NLCD()
        land_cover = nlcd.get_coverage(basin.geometry.iloc[0], resolution=30)

        if land_cover is not None:
            result['land_cover'] = land_cover
            logger.info(f"Retrieved NLCD land cover for {site_id}")

    except ImportError:
        logger.debug("pygeohydro not available for NLCD")
    except Exception as e:
        logger.debug(f"NLCD unavailable for {site_id}: {e}")

    # Mean elevation via py3dep
    try:
        from py3dep import get_map

        dem = get_map("DEM", basin.geometry.iloc[0], resolution=30, crs=basin.crs)
        if dem is not None:
            result['elevation_mean_m'] = float(np.nanmean(dem.values))
            result['elevation_min_m'] = float(np.nanmin(dem.values))
            result['elevation_max_m'] = float(np.nanmax(dem.values))

    except ImportError:
        logger.debug("py3dep not available for elevation")
    except Exception as e:
        logger.debug(f"Elevation data unavailable: {e}")

    if result:
        _characteristics_cache[site_id] = result

    return result if result else None


def _daymet_credentials_available() -> bool:
    """Daymet/ORNL THREDDS requires NASA Earthdata login (401 without it)."""
    import os
    from pathlib import Path

    if os.environ.get("EARTHDATA_USERNAME") or os.environ.get("EARTHDATA_USER"):
        return True
    if os.environ.get("EARTHDATA_PASSWORD") or os.environ.get("EARTHDATA_TOKEN"):
        return True
    # netrc is the usual Earthdata CLI auth location
    for candidate in (Path.home() / ".netrc", Path.home() / "_netrc"):
        try:
            if candidate.is_file() and "urs.earthdata.nasa.gov" in candidate.read_text(
                encoding="utf-8", errors="ignore"
            ):
                return True
        except Exception:
            pass
    return False


def get_daymet_climate(
    site_id: str,
    start_date: str,
    end_date: str,
    variables: List[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Get Daymet gridded climate data averaged over the watershed.

    Daymet provides daily gridded temperature and precipitation at 1km resolution,
    which gives a more representative watershed average than single-station Meteostat.

    Args:
        site_id: USGS site identifier
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        variables: Daymet variables to fetch. Default: ['prcp', 'tmin', 'tmax']
            Available: prcp, tmin, tmax, srad, vp, swe, dayl

    Returns:
        DataFrame with daily climate data indexed by date, or None
    """
    if variables is None:
        variables = ['prcp', 'tmin', 'tmax']

    # Avoid slow watershed + 401 storm when Earthdata is not configured.
    if not _daymet_credentials_available():
        logger.info(
            "Skipping Daymet for %s: NASA Earthdata credentials not configured "
            "(set EARTHDATA_USERNAME/PASSWORD or ~/.netrc for urs.earthdata.nasa.gov)",
            site_id,
        )
        return None

    # Get watershed boundary for spatial averaging
    basin = get_watershed_boundary(site_id)
    if basin is None:
        # Fall back to point-based query
        coords = _get_site_coords(site_id)
        if coords is None:
            return None
        return _get_daymet_point(coords[0], coords[1], start_date, end_date, variables)

    try:
        from pydaymet import get_bygeom

        dates = (start_date, end_date)
        ds = get_bygeom(
            basin.geometry.iloc[0],
            dates,
            crs=basin.crs,
            variables=variables,
        )

        if ds is None:
            return None

        # Spatial average across the watershed
        df = ds.mean(dim=['x', 'y']).to_dataframe()
        df.index.name = 'date'

        # Rename for consistency
        rename_map = {
            'prcp': 'precip_mm',
            'tmin': 'tmin_c',
            'tmax': 'tmax_c',
            'srad': 'solar_rad_wm2',
            'vp': 'vapor_pressure_pa',
            'swe': 'snow_water_eq_mm',
            'dayl': 'day_length_s',
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        # Add mean temperature
        if 'tmin_c' in df.columns and 'tmax_c' in df.columns:
            df['tmean_c'] = (df['tmin_c'] + df['tmax_c']) / 2

        logger.info(f"Retrieved Daymet climate data for {site_id}: {len(df)} days")
        return df

    except ImportError:
        logger.error("pydaymet not installed. Install with: conda install -c conda-forge pydaymet")
        return None
    except Exception as e:
        logger.error(f"Error fetching Daymet data for {site_id}: {e}")
        return None


def _get_daymet_point(
    lat: float, lon: float,
    start_date: str, end_date: str,
    variables: List[str],
) -> Optional[pd.DataFrame]:
    """Fallback: get Daymet data for a single point."""
    try:
        from pydaymet import get_bycoords

        dates = (start_date, end_date)
        df = get_bycoords((lon, lat), dates, variables=variables)

        if df is not None and not df.empty:
            rename_map = {
                'prcp (mm/day)': 'precip_mm',
                'tmin (degrees C)': 'tmin_c',
                'tmax (degrees C)': 'tmax_c',
            }
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
            if 'tmin_c' in df.columns and 'tmax_c' in df.columns:
                df['tmean_c'] = (df['tmin_c'] + df['tmax_c']) / 2
            return df

    except Exception as e:
        logger.error(f"Daymet point query failed: {e}")

    return None


def get_elevation_profile(
    site_id: str,
    n_points: int = 100,
) -> Optional[pd.DataFrame]:
    """
    Get elevation profile along the main flowline upstream of a site.

    Args:
        site_id: USGS site identifier
        n_points: Number of sample points along the flowline

    Returns:
        DataFrame with columns: distance_km, elevation_m
    """
    try:
        from py3dep import get_map
        from pynhd import NLDI
        from shapely.geometry import LineString, Point

        nldi = NLDI()
        flowlines = nldi.navigate_byid(
            fsource="nwissite",
            fid=f"USGS-{site_id}",
            navigation="upstreamMain",
            source="flowlines",
            distance=200,
        )

        if flowlines is None or flowlines.empty:
            return None

        # Merge flowlines into a single line
        from shapely.ops import linemerge
        merged = linemerge(flowlines.geometry.tolist())
        if merged.geom_type == 'MultiLineString':
            # Take the longest linestring
            merged = max(merged.geoms, key=lambda g: g.length)

        # Sample points along the line
        total_length = merged.length
        distances = np.linspace(0, total_length, n_points)
        points = [merged.interpolate(d) for d in distances]

        # Get elevation at each point
        elevations = []
        for pt in points:
            try:
                dem = get_map("DEM", pt.buffer(0.001), resolution=10, crs="EPSG:4326")
                if dem is not None:
                    elevations.append(float(np.nanmean(dem.values)))
                else:
                    elevations.append(np.nan)
            except Exception:
                elevations.append(np.nan)

        # Convert distances to km (approximate)
        distance_km = distances * 111  # rough degree-to-km conversion

        df = pd.DataFrame({
            'distance_km': distance_km,
            'elevation_m': elevations,
        })

        return df.dropna()

    except ImportError:
        logger.error("py3dep/pynhd not installed")
        return None
    except Exception as e:
        logger.error(f"Error computing elevation profile for {site_id}: {e}")
        return None


def get_nid_dams(
    site_id: str,
    distance_km: float = 50,
) -> Optional[Any]:
    """
    Get dams from the National Inventory of Dams near a USGS site.

    Uses the NID REST API directly instead of pygeohydro (which has
    a schema mismatch bug with the current NID data).

    Args:
        site_id: USGS site identifier
        distance_km: Search radius in kilometers

    Returns:
        geopandas GeoDataFrame with dam locations and attributes, or None
    """
    cache_key = (site_id, float(distance_km))
    if cache_key in _dams_cache:
        return _dams_cache[cache_key]

    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError:
        logger.error("geopandas not installed")
        return None

    coords = _get_site_coords(site_id)
    if coords is None:
        logger.warning(f"Could not get coordinates for {site_id}")
        _dams_cache[cache_key] = None
        return None

    lat, lon = coords

    # Query NID REST API with bounding box
    deg_offset = distance_km / 111.0
    bbox = f"{lon - deg_offset},{lat - deg_offset},{lon + deg_offset},{lat + deg_offset}"

    url = "https://nid.sec.usace.army.mil/api/nation/dams"
    params = {
        "bbox": bbox,
        "limit": 100,
    }

    try:
        resp = requests.get(url, params=params, timeout=12, headers={
            "Accept": "application/json",
        })

        # If bbox param doesn't work, fall back to downloading nearby via offset query
        if resp.status_code != 200:
            # Try alternative: query by state and filter locally
            logger.debug(f"NID bbox query returned {resp.status_code}, trying spatial filter")
            dams = _nid_fallback_spatial(lat, lon, distance_km)
            _dams_cache[cache_key] = dams
            return dams

        data = resp.json()
        if not data:
            _dams_cache[cache_key] = None
            return None

        records = []
        for dam in data:
            dam_lat = dam.get('latitude')
            dam_lon = dam.get('longitude')
            if dam_lat is None or dam_lon is None:
                continue
            records.append({
                'dam_name': dam.get('name', dam.get('damName', 'Unknown')),
                'latitude': float(dam_lat),
                'longitude': float(dam_lon),
                'height_ft': dam.get('nidHeight', dam.get('height')),
                'storage_acre_ft': dam.get('maxStorage'),
                'year_completed': dam.get('yearCompleted'),
                'hazard': dam.get('hazardPotentialClassification', dam.get('hazard')),
                'purposes': dam.get('purposes'),
                'owner_type': dam.get('primaryOwnerType', dam.get('ownerType')),
            })

        if not records:
            _dams_cache[cache_key] = None
            return None

        import pandas as pd
        df = pd.DataFrame(records)
        geometry = [Point(row['longitude'], row['latitude']) for _, row in df.iterrows()]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

        logger.info(f"Found {len(gdf)} dams near {site_id} via NID API")
        _dams_cache[cache_key] = gdf
        return gdf

    except Exception as e:
        logger.error(f"NID API error for {site_id}: {e}")
        dams = _nid_fallback_spatial(lat, lon, distance_km)
        _dams_cache[cache_key] = dams
        return dams


def _nid_fallback_spatial(lat: float, lon: float, distance_km: float) -> Optional[Any]:
    """Fallback: try pygeohydro NID if REST API fails."""
    try:
        from pygeohydro import NID
        nid = NID()
        deg_offset = distance_km / 111.0
        bbox = (lon - deg_offset, lat - deg_offset, lon + deg_offset, lat + deg_offset)
        dams = nid.get_bygeom(bbox, geo_crs="EPSG:4326")
        if dams is not None and not dams.empty:
            return dams
    except Exception as e:
        logger.debug(f"pygeohydro NID fallback also failed: {e}")
    return None
