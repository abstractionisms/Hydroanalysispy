"""
NWIS inventory file parsing for hydrology package.

Functions to load site information from the NWIS inventory text files.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

from ..core.paths import PROJECT_ROOT
from ..core.logging_setup import get_logger

logger = get_logger(__name__)

# Default inventory file location
DEFAULT_INVENTORY = PROJECT_ROOT / "nwis_inventory_filtered_data_only.txt"


def load_inventory(inventory_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load NWIS inventory file into DataFrame.

    Args:
        inventory_path: Path to inventory file (uses default if None)

    Returns:
        DataFrame with site information

    Columns:
        - site_id: USGS site identifier
        - description: Site name/description
        - latitude: Decimal degrees
        - longitude: Decimal degrees
        - (additional columns from inventory file)
    """
    if inventory_path is None:
        inventory_path = DEFAULT_INVENTORY

    if not inventory_path.exists():
        logger.error(f"Inventory file not found: {inventory_path}")
        return pd.DataFrame()

    logger.info(f"Loading inventory from: {inventory_path}")

    try:
        # Read tab-delimited file
        # Format: site_id, description, lat, lon, ...
        df = pd.read_csv(
            inventory_path,
            sep='\t',
            header=None,
            names=[
                'site_id', 'description', 'latitude', 'longitude',
                'coord_accuracy', 'datum', 'state_cd', 'county_cd',
                'district_cd', 'alt', 'alt_accuracy', 'alt_datum',
                'huc_cd', 'basin_cd', 'drain_area_sq_mi', 'contrib_drain_area_sq_mi',
                'begin_date', 'count_nu'
            ],
            dtype={'site_id': str}
        )

        logger.info(f"Loaded {len(df)} sites from inventory")
        return df

    except Exception as e:
        logger.error(f"Error loading inventory: {e}")
        return pd.DataFrame()


def get_site_info(site_id: str, inventory_path: Optional[Path] = None) -> Optional[Dict]:
    """
    Get information for a specific site from inventory.

    Args:
        site_id: USGS site identifier (e.g., '12422500')
        inventory_path: Path to inventory file (uses default if None)

    Returns:
        Dictionary with site information or None if not found

    Example:
        >>> site = get_site_info('12422500')
        >>> print(site['description'])
        'Spokane River at Spokane, WA'
        >>> print(site['latitude'], site['longitude'])
        47.65933540 -117.44910290
    """
    df = load_inventory(inventory_path)

    if df.empty:
        return None

    # Find site
    site = df[df['site_id'] == str(site_id)]

    if site.empty:
        logger.warning(f"Site not found in inventory: {site_id}")
        return None

    # Convert to dict
    site_dict = site.iloc[0].to_dict()

    logger.info(f"Found site: {site_id} - {site_dict['description']}")
    return site_dict


def get_multiple_sites(site_ids: List[str], inventory_path: Optional[Path] = None) -> List[Dict]:
    """
    Get information for multiple sites from inventory.

    Args:
        site_ids: List of USGS site identifiers
        inventory_path: Path to inventory file (uses default if None)

    Returns:
        List of site dictionaries

    Example:
        >>> sites = get_multiple_sites(['12422500', '12424000'])
        >>> for site in sites:
        ...     print(site['site_id'], site['description'])
        12422500 Spokane River at Spokane, WA
        12424000 Hangman Creek at Spokane, WA
    """
    df = load_inventory(inventory_path)

    if df.empty:
        return []

    # Find all sites
    site_ids_str = [str(sid) for sid in site_ids]
    sites_df = df[df['site_id'].isin(site_ids_str)]

    if sites_df.empty:
        logger.warning(f"No sites found in inventory for: {site_ids}")
        return []

    # Convert to list of dicts
    sites = sites_df.to_dict('records')

    logger.info(f"Found {len(sites)} of {len(site_ids)} requested sites")
    return sites


def search_sites(
    state: Optional[str] = None,
    description_contains: Optional[str] = None,
    huc: Optional[str] = None,
    min_drainage_area: Optional[float] = None,
    max_drainage_area: Optional[float] = None,
    inventory_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Search for sites matching criteria.

    Args:
        state: State code (e.g., '53' for Washington)
        description_contains: Search description text (case-insensitive)
        huc: HUC code or prefix
        min_drainage_area: Minimum drainage area in square miles
        max_drainage_area: Maximum drainage area in square miles
        inventory_path: Path to inventory file (uses default if None)

    Returns:
        DataFrame with matching sites

    Example:
        >>> # Find all sites in Washington with "Spokane" in name
        >>> sites = search_sites(state='53', description_contains='Spokane')
        >>> print(sites[['site_id', 'description']])
    """
    df = load_inventory(inventory_path)

    if df.empty:
        return df

    # Apply filters
    mask = pd.Series([True] * len(df))

    if state is not None:
        mask &= (df['state_cd'] == str(state))

    if description_contains is not None:
        mask &= df['description'].str.contains(description_contains, case=False, na=False)

    if huc is not None:
        mask &= df['huc_cd'].str.startswith(str(huc), na=False)

    if min_drainage_area is not None:
        mask &= (pd.to_numeric(df['drain_area_sq_mi'], errors='coerce') >= min_drainage_area)

    if max_drainage_area is not None:
        mask &= (pd.to_numeric(df['drain_area_sq_mi'], errors='coerce') <= max_drainage_area)

    result = df[mask]

    logger.info(f"Search found {len(result)} sites")
    return result
