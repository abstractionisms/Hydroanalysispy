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

from ..core.logging_setup import get_logger
from ..core.paths import CACHE_DIR, ensure_dir

logger = get_logger(__name__)

# Cache for expensive geospatial lookups
_boundary_cache: Dict[str, Any] = {}
_characteristics_cache: Dict[str, Dict] = {}


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
            logger.info(f"Retrieved {len(flowlines)} flowlines for {site_id}")
            return flowlines

        return None

    except ImportError:
        logger.error("pynhd not installed")
        return None
    except Exception as e:
        logger.error(f"Error fetching flowlines for {site_id}: {e}")
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

    Args:
        site_id: USGS site identifier
        distance_km: Search radius in kilometers

    Returns:
        geopandas GeoDataFrame with dam locations and attributes, or None
    """
    try:
        from pygeohydro import NID

        coords = _get_site_coords(site_id)
        if coords is None:
            return None

        lat, lon = coords
        nid = NID()

        # Query by bounding box around the site
        # Approximate degree offset for distance
        deg_offset = distance_km / 111.0
        bbox = (
            lon - deg_offset,
            lat - deg_offset,
            lon + deg_offset,
            lat + deg_offset,
        )

        dams = nid.get_bygeom(bbox, geo_crs="EPSG:4326")

        if dams is not None and not dams.empty:
            logger.info(f"Found {len(dams)} dams near {site_id}")
            return dams

        return None

    except ImportError:
        logger.error("pygeohydro not installed")
        return None
    except Exception as e:
        logger.error(f"Error fetching dams for {site_id}: {e}")
        return None
