"""
Climate data fetching using Meteostat for hydrology package.

Provides functions to fetch temperature and precipitation data
from the Meteostat API for hydrological analysis.
"""

import pandas as pd
from typing import Optional, Tuple, Dict
from meteostat import Point, Daily

from ..core.logging_setup import get_logger
from ..core.timezone import ensure_utc

logger = get_logger(__name__)


def fetch_climate_data(
    latitude: float,
    longitude: float,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    include_temp: bool = True,
    include_precip: bool = True
) -> Optional[pd.DataFrame]:
    """
    Fetch daily climate data using Meteostat.

    Fetches temperature and/or precipitation data for a location.
    Fills missing precipitation with zeros and forwards/backwards fills
    missing temperature values.

    Args:
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees
        start_date: Start datetime
        end_date: End datetime
        include_temp: Include temperature data (column: 'Temp_C')
        include_precip: Include precipitation data (column: 'Precip_mm')

    Returns:
        DataFrame with UTC datetime index and climate columns, or None if failed

    Example:
        >>> climate = fetch_climate_data(47.6593, -117.4491,
        ...                              pd.Timestamp('2020-01-01'),
        ...                              pd.Timestamp('2023-12-31'))
        >>> print(climate[['Temp_C', 'Precip_mm']].describe())
    """
    # Validate date order
    if start_date > end_date:
        logger.warning(f"Start date ({start_date.date()}) after end date ({end_date.date()}) - swapping")
        start_date, end_date = end_date, start_date

    logger.info(f"Fetching climate data: Lat={latitude:.4f}, Lon={longitude:.4f}, "
                f"{start_date.date()} to {end_date.date()}")

    try:
        # Create location point and fetch data
        location = Point(latitude, longitude)
        data = Daily(location, start_date, end_date)
        data = data.fetch()

        if data is None or data.empty:
            logger.warning("Meteostat returned no data for this location/period")
            return None

        # Select and rename columns
        cols_to_keep = []
        rename_map = {}

        if include_temp and 'tavg' in data.columns:
            cols_to_keep.append('tavg')
            rename_map['tavg'] = 'Temp_C'
        elif include_temp:
            logger.warning("Temperature requested but 'tavg' column not in Meteostat data")

        if include_precip and 'prcp' in data.columns:
            cols_to_keep.append('prcp')
            rename_map['prcp'] = 'Precip_mm'
        elif include_precip:
            logger.warning("Precipitation requested but 'prcp' column not in Meteostat data")

        if not cols_to_keep:
            logger.warning("No requested climate columns available in Meteostat data")
            return None

        # Create climate dataframe
        climate_df = data[cols_to_keep].copy()
        climate_df = climate_df.rename(columns=rename_map)

        # Ensure UTC index
        climate_df = ensure_utc(climate_df)

        # Fill missing values
        if 'Precip_mm' in climate_df.columns:
            # Fill missing precipitation with 0 (assume no rain on missing days)
            missing_precip = climate_df['Precip_mm'].isna().sum()
            climate_df['Precip_mm'] = climate_df['Precip_mm'].fillna(0)
            if missing_precip > 0:
                logger.info(f"Filled {missing_precip} missing precipitation values with 0")

        if 'Temp_C' in climate_df.columns:
            # Forward/backward fill missing temperature
            missing_temp_before = climate_df['Temp_C'].isna().sum()
            climate_df['Temp_C'] = climate_df['Temp_C'].ffill().bfill()
            missing_temp_after = climate_df['Temp_C'].isna().sum()
            if missing_temp_before > 0:
                logger.info(f"Filled {missing_temp_before - missing_temp_after} "
                           f"missing temperature values via forward/backward fill")

        logger.info(f"Climate data fetched successfully: {len(climate_df)} rows")
        return climate_df

    except Exception as e:
        logger.error(f"Error fetching/processing climate data: {e}")
        return None


def fetch_nearest_station_info(
    latitude: float,
    longitude: float,
    max_distance_km: float = 100.0
) -> Optional[Dict]:
    """
    Get information about the nearest weather station.

    Useful for understanding data quality and station distance.

    Args:
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees
        max_distance_km: Maximum search radius in kilometers

    Returns:
        Dictionary with station information or None

    Example:
        >>> station = fetch_nearest_station_info(47.6593, -117.4491)
        >>> print(f"Nearest station: {station['name']} ({station['distance_km']:.1f} km)")
    """
    try:
        from meteostat import Stations

        location = Point(latitude, longitude)
        stations = Stations()
        stations = stations.nearby(latitude, longitude)
        stations = stations.fetch(1)  # Get closest station

        if stations.empty:
            logger.warning(f"No weather stations found within {max_distance_km} km")
            return None

        station = stations.iloc[0]
        station_dict = {
            'id': station.name,  # Station ID
            'name': station.get('name', 'Unknown'),
            'latitude': station.get('latitude', None),
            'longitude': station.get('longitude', None),
            'elevation_m': station.get('elevation', None),
            'distance_km': station.get('distance', 0) / 1000,  # Meteostat returns meters
        }

        # Include data coverage dates if available
        daily_start = station.get('daily_start', None)
        daily_end = station.get('daily_end', None)
        if daily_start is not None:
            station_dict['daily_start'] = str(daily_start)[:10] if daily_start else None
        if daily_end is not None:
            station_dict['daily_end'] = str(daily_end)[:10] if daily_end else None

        logger.info(f"Nearest station: {station_dict['name']} "
                   f"({station_dict.get('distance_km', '?')} km away)")

        return station_dict

    except Exception as e:
        logger.error(f"Error fetching station info: {e}")
        return None


def merge_discharge_climate(
    discharge_df: pd.DataFrame,
    climate_df: pd.DataFrame,
    discharge_col: str = 'Discharge_cfs'
) -> pd.DataFrame:
    """
    Merge discharge and climate data on datetime index.

    Performs inner join to keep only dates with both discharge and climate data.

    Args:
        discharge_df: Discharge data with datetime index
        climate_df: Climate data with datetime index
        discharge_col: Name of discharge column

    Returns:
        Merged DataFrame with all columns

    Example:
        >>> discharge = fetch_discharge_data('12422500', '2020-01-01', '2023-12-31')
        >>> climate = fetch_climate_data(47.6593, -117.4491,
        ...                              pd.Timestamp('2020-01-01'),
        ...                              pd.Timestamp('2023-12-31'))
        >>> merged = merge_discharge_climate(discharge, climate)
        >>> print(merged.columns)
        Index(['Discharge_cfs', 'Temp_C', 'Precip_mm'], dtype='object')
    """
    logger.info("Merging discharge and climate data...")

    if discharge_df is None or discharge_df.empty:
        logger.error("Discharge data is empty")
        return pd.DataFrame()

    if climate_df is None or climate_df.empty:
        logger.error("Climate data is empty")
        return pd.DataFrame()

    # Ensure both have datetime index
    if not isinstance(discharge_df.index, pd.DatetimeIndex):
        logger.error("Discharge data must have datetime index")
        return pd.DataFrame()

    if not isinstance(climate_df.index, pd.DatetimeIndex):
        logger.error("Climate data must have datetime index")
        return pd.DataFrame()

    # Inner join on datetime index
    merged = discharge_df.join(climate_df, how='inner')

    logger.info(f"Merged data: {len(merged)} rows "
                f"(discharge: {len(discharge_df)}, climate: {len(climate_df)})")

    if merged.empty:
        logger.warning("Merged data is empty - check date ranges overlap")

    return merged
