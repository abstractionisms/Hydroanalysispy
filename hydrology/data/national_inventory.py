"""
National USGS Site Inventory for Watershed View.

Provides functions to download, cache, and query the full inventory of
active USGS streamflow monitoring sites across the United States.

The inventory is organized by HUC-2 regions (21 major water resource regions)
and includes site metadata, coordinates, and data availability.

Example:
    >>> from hydrology.data.national_inventory import get_national_inventory
    >>> sites_df = get_national_inventory()
    >>> print(f"Total active sites: {len(sites_df)}")
    >>> pnw_sites = sites_df[sites_df['huc2'] == '17']
"""

import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests

from ..core.logging_setup import get_logger
from ..core.paths import CACHE_DIR, ensure_dir
from ..core.huc_regions import HUC2_REGIONS, CONUS_REGIONS

logger = get_logger(__name__)

# Cache settings
NATIONAL_CACHE_DIR = CACHE_DIR / 'national_inventory'
INVENTORY_CACHE_FILE = NATIONAL_CACHE_DIR / 'all_sites.parquet'
METADATA_FILE = NATIONAL_CACHE_DIR / 'inventory_metadata.json'
CACHE_EXPIRY_DAYS = 7

# USGS Site Service
SITE_SERVICE_URL = "https://waterservices.usgs.gov/nwis/site/"


def get_national_inventory(force_refresh: bool = False) -> pd.DataFrame:
    """
    Get the full national inventory of active USGS streamflow sites.

    Downloads and caches site data for all HUC-2 regions. The cache is
    refreshed automatically after CACHE_EXPIRY_DAYS.

    Args:
        force_refresh: If True, re-download even if cache is valid

    Returns:
        DataFrame with columns:
        - site_id: USGS site identifier
        - site_name: Site description
        - latitude, longitude: Coordinates
        - huc_cd: Full HUC code (8 digits)
        - huc2: HUC-2 region code
        - state_cd: State code
        - drainage_area: Drainage area in sq mi
        - begin_date: Start of record
        - data_type_cd: Available data types

    Example:
        >>> df = get_national_inventory()
        >>> print(f"Sites: {len(df)}, Regions: {df['huc2'].nunique()}")
    """
    ensure_dir(NATIONAL_CACHE_DIR)

    if not force_refresh and _is_cache_valid():
        logger.info("Loading national inventory from cache")
        return _load_cached_inventory()

    logger.info("Downloading fresh national inventory...")
    df = _download_all_regions()

    if not df.empty:
        _save_inventory_cache(df)

    return df


def get_region_inventory(huc2: str, force_refresh: bool = False) -> pd.DataFrame:
    """
    Get inventory for a specific HUC-2 region.

    Args:
        huc2: HUC-2 region code (e.g., '17' for Pacific Northwest)
        force_refresh: Force re-download

    Returns:
        DataFrame with sites in the specified region
    """
    # Try loading from full cache first
    full_df = get_national_inventory(force_refresh=force_refresh)

    if not full_df.empty:
        return full_df[full_df['huc2'] == huc2.zfill(2)]

    # Fallback: fetch just this region
    return _fetch_region_sites(huc2)


def get_inventory_summary() -> Dict[str, Any]:
    """
    Get summary statistics about the national inventory.

    Returns:
        Dict with total counts, per-region counts, last update, etc.
    """
    df = get_national_inventory()

    if df.empty:
        return {'total_sites': 0, 'regions': {}, 'last_update': None}

    summary = {
        'total_sites': len(df),
        'total_states': df['state_cd'].nunique() if 'state_cd' in df.columns else 0,
        'regions': {},
        'last_update': None,
    }

    # Per-region counts
    if 'huc2' in df.columns:
        region_counts = df.groupby('huc2').size().to_dict()
        for huc2, count in region_counts.items():
            region_name = HUC2_REGIONS.get(huc2, {}).get('name', huc2)
            summary['regions'][huc2] = {'name': region_name, 'count': count}

    # Load metadata for last update time
    if METADATA_FILE.exists():
        with open(METADATA_FILE) as f:
            metadata = json.load(f)
            summary['last_update'] = metadata.get('download_date')

    return summary


def _is_cache_valid() -> bool:
    """Check if the cached inventory is still valid."""
    if not INVENTORY_CACHE_FILE.exists():
        return False

    if not METADATA_FILE.exists():
        return False

    try:
        with open(METADATA_FILE) as f:
            metadata = json.load(f)

        download_date = datetime.fromisoformat(metadata.get('download_date', ''))
        expiry = download_date + timedelta(days=CACHE_EXPIRY_DAYS)

        return datetime.now() < expiry
    except (json.JSONDecodeError, ValueError, KeyError):
        return False


def _load_cached_inventory() -> pd.DataFrame:
    """Load inventory from cache file."""
    try:
        return pd.read_parquet(INVENTORY_CACHE_FILE)
    except Exception as e:
        logger.error(f"Failed to load cached inventory: {e}")
        return pd.DataFrame()


def _save_inventory_cache(df: pd.DataFrame):
    """Save inventory to cache with metadata."""
    try:
        df.to_parquet(INVENTORY_CACHE_FILE, index=False)

        metadata = {
            'download_date': datetime.now().isoformat(),
            'total_sites': len(df),
            'regions_included': df['huc2'].unique().tolist() if 'huc2' in df.columns else [],
        }

        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved {len(df)} sites to cache")
    except Exception as e:
        logger.error(f"Failed to save inventory cache: {e}")


def _download_all_regions(max_workers: int = 4) -> pd.DataFrame:
    """Download site inventory for all HUC-2 regions."""
    all_sites = []

    # Download CONUS regions in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_region_sites, huc2): huc2
            for huc2 in CONUS_REGIONS
        }

        for future in as_completed(futures):
            huc2 = futures[future]
            try:
                df = future.result()
                if not df.empty:
                    all_sites.append(df)
                    logger.info(f"HUC-2 {huc2}: {len(df)} sites")
            except Exception as e:
                logger.error(f"Failed to fetch HUC-2 {huc2}: {e}")

    # Also fetch non-CONUS regions (Alaska, Hawaii, Caribbean)
    for huc2 in ['19', '20', '21']:
        try:
            df = _fetch_region_sites(huc2)
            if not df.empty:
                all_sites.append(df)
                logger.info(f"HUC-2 {huc2}: {len(df)} sites")
        except Exception as e:
            logger.error(f"Failed to fetch HUC-2 {huc2}: {e}")

    if not all_sites:
        return pd.DataFrame()

    combined = pd.concat(all_sites, ignore_index=True)
    logger.info(f"Total: {len(combined)} active sites across {combined['huc2'].nunique()} regions")

    return combined


def _fetch_region_sites(huc2: str) -> pd.DataFrame:
    """
    Fetch site inventory for a single HUC-2 region.

    Args:
        huc2: HUC-2 region code

    Returns:
        DataFrame with site data for the region
    """
    params = {
        'format': 'rdb',
        'huc': huc2.zfill(2),
        'siteType': 'ST',  # Stream sites
        'siteStatus': 'active',
        'hasDataTypeCd': 'dv',  # Has daily values
        'parameterCd': '00060',  # Discharge
    }

    try:
        response = requests.get(SITE_SERVICE_URL, params=params, timeout=60)
        response.raise_for_status()

        df = _parse_site_rdb(response.text)

        if not df.empty:
            df['huc2'] = huc2.zfill(2)

        return df

    except requests.RequestException as e:
        logger.error(f"HTTP error fetching HUC-2 {huc2}: {e}")
        return pd.DataFrame()


def _parse_site_rdb(rdb_text: str) -> pd.DataFrame:
    """Parse USGS RDB format for site inventory."""
    lines = rdb_text.strip().split('\n')

    # Find header line (first non-comment line)
    headers = []
    data_start = 0

    for i, line in enumerate(lines):
        if line.startswith('#'):
            continue
        if not headers:
            headers = [h.strip() for h in line.split('\t')]
            continue
        # Skip format line
        if any(c in line for c in ['s', 'd', 'n']) and '\t' in line:
            if all(part.strip().replace('s', '').replace('d', '').replace('n', '').isdigit()
                   for part in line.split('\t') if part.strip()):
                continue
        data_start = i
        break

    if not headers:
        return pd.DataFrame()

    # Parse data
    records = []
    for line in lines[data_start:]:
        if not line or line.startswith('#'):
            continue
        values = line.split('\t')
        if len(values) >= len(headers):
            records.append(dict(zip(headers, values[:len(headers)])))

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Standardize column names
    column_map = {
        'site_no': 'site_id',
        'station_nm': 'site_name',
        'dec_lat_va': 'latitude',
        'dec_long_va': 'longitude',
        'huc_cd': 'huc_cd',
        'state_cd': 'state_cd',
        'drain_area_va': 'drainage_area',
        'begin_date': 'begin_date',
        'end_date': 'end_date',
        'data_type_cd': 'data_type_cd',
    }

    result = pd.DataFrame()
    for old_col, new_col in column_map.items():
        if old_col in df.columns:
            result[new_col] = df[old_col]

    # Convert numeric columns
    for col in ['latitude', 'longitude', 'drainage_area']:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors='coerce')

    return result


def search_sites(
    query: Optional[str] = None,
    huc2: Optional[str] = None,
    state: Optional[str] = None,
    min_drainage_area: Optional[float] = None,
    max_drainage_area: Optional[float] = None
) -> pd.DataFrame:
    """
    Search the national inventory with filters.

    Args:
        query: Text search in site name
        huc2: Filter by HUC-2 region
        state: Filter by state code
        min_drainage_area: Minimum drainage area (sq mi)
        max_drainage_area: Maximum drainage area (sq mi)

    Returns:
        Filtered DataFrame
    """
    df = get_national_inventory()

    if df.empty:
        return df

    mask = pd.Series([True] * len(df))

    if query:
        mask &= df['site_name'].str.contains(query, case=False, na=False)

    if huc2:
        mask &= df['huc2'] == huc2.zfill(2)

    if state:
        mask &= df['state_cd'] == state.upper()

    if min_drainage_area is not None and 'drainage_area' in df.columns:
        mask &= df['drainage_area'] >= min_drainage_area

    if max_drainage_area is not None and 'drainage_area' in df.columns:
        mask &= df['drainage_area'] <= max_drainage_area

    return df[mask]
