"""
Timezone handling utilities for hydrology package.

Centralizes timezone normalization to ensure consistent datetime handling
across all data sources (USGS, Meteostat, etc.).
"""

import pandas as pd
from typing import Union

# Standard timezone for all hydrology data
DEFAULT_TIMEZONE = 'UTC'


def normalize_index_timezone(df: pd.DataFrame, target_tz: str = DEFAULT_TIMEZONE) -> pd.DataFrame:
    """
    Normalize DataFrame index to target timezone.

    Args:
        df: DataFrame with DatetimeIndex
        target_tz: Target timezone (default: UTC)

    Returns:
        DataFrame with timezone-normalized index

    Example:
        >>> df = normalize_index_timezone(df_q)
        >>> df.index.tz
        <UTC>
    """
    if df is None or df.empty:
        return df

    if not isinstance(df.index, pd.DatetimeIndex):
        return df

    if df.index.tz is None:
        df = df.copy()
        df.index = df.index.tz_localize(target_tz)
    elif str(df.index.tz) != target_tz:
        df = df.copy()
        df.index = df.index.tz_convert(target_tz)

    return df


def ensure_utc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to ensure DataFrame index is UTC.

    Args:
        df: DataFrame with DatetimeIndex

    Returns:
        DataFrame with UTC-normalized index
    """
    return normalize_index_timezone(df, 'UTC')


def remove_timezone(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove timezone info from DataFrame index (make timezone-naive).

    Useful for operations that don't support timezone-aware datetimes.

    Args:
        df: DataFrame with DatetimeIndex

    Returns:
        DataFrame with timezone-naive index
    """
    if df is None or df.empty:
        return df

    if not isinstance(df.index, pd.DatetimeIndex):
        return df

    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)

    return df
