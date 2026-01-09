"""
USGS NWIS data fetching and parsing for hydrology package.

Provides robust data fetching from USGS National Water Information System
with retry logic, chunked requests to avoid throttling, and support for
both daily values (DV) and instantaneous values (IV).
"""

import time
import random
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import requests

from ..core.logging_setup import get_logger
from ..core.paths import CACHE_DIR, ensure_dir

logger = get_logger(__name__)

# API endpoints
BASE_URL_DV = "https://waterservices.usgs.gov/nwis/dv/"
BASE_URL_IV = "https://waterservices.usgs.gov/nwis/iv/"

# Parameter codes
DEFAULT_PARAM_DISCHARGE = "00060"  # Discharge, cubic feet per second
DEFAULT_PARAM_STAGE = "00065"      # Gage height, feet
DEFAULT_PARAM_TEMP = "00010"       # Water temperature, degrees Celsius

# Statistic codes for daily values
STAT_MEAN = "00003"    # Mean
STAT_MIN = "00002"     # Minimum
STAT_MAX = "00001"     # Maximum


def http_get_text(
    url: str,
    params: Dict[str, Any],
    retries: int = 6,
    base_sleep: float = 0.8,
    timeout: int = 25
) -> str:
    """
    Robust HTTP GET with exponential backoff retry logic.

    Args:
        url: URL to fetch
        params: Query parameters
        retries: Number of retry attempts
        base_sleep: Base sleep time between retries (exponential backoff)
        timeout: Request timeout in seconds

    Returns:
        Response text

    Raises:
        requests.RequestException: If all retries fail
    """
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.text
            logger.warning(f"HTTP {r.status_code} for {r.url}")
            last_err = requests.HTTPError(f"{r.status_code} for {r.url}")
        except requests.RequestException as e:
            last_err = e
            logger.warning(f"HTTP error ({i+1}/{retries}): {e}")

        # Exponential backoff with jitter
        if i < retries - 1:  # Don't sleep on last retry
            sleep_time = base_sleep * (2 ** i) + random.random() * 0.25
            time.sleep(sleep_time)

    if last_err:
        raise last_err
    return ""


def parse_waterml(xml_text: str) -> pd.DataFrame:
    """
    Parse WaterML XML format into pandas DataFrame.

    Args:
        xml_text: WaterML XML string

    Returns:
        DataFrame with datetime index and 'value' column

    Example:
        >>> xml = fetch_waterml_data('12422500', '00060', '2020-01-01', '2020-12-31')
        >>> df = parse_waterml(xml)
    """
    if not xml_text:
        return pd.DataFrame(columns=["value"])

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.error(f"XML parse error: {e}")
        return pd.DataFrame(columns=["value"])

    timestamps, values = [], []

    for value_elem in root.findall(".//{*}value"):
        datetime_str = value_elem.attrib.get("dateTime")
        value_str = (value_elem.text or "").strip()

        if not datetime_str or value_str == "":
            continue

        timestamps.append(datetime_str)
        values.append(value_str)

    if not timestamps:
        return pd.DataFrame(columns=["value"])

    # Parse to datetime and numeric
    idx = pd.to_datetime(timestamps, errors="coerce")
    vals = pd.to_numeric(values, errors="coerce")

    # Create series and clean
    series = pd.Series(vals, index=idx).dropna()

    if series.empty:
        return pd.DataFrame(columns=["value"])

    # Normalize to daily (midnight) timestamps
    series.index = pd.to_datetime(series.index.date)

    return series.sort_index().to_frame("value")


def parse_json_series(json_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Parse USGS JSON format into pandas DataFrame.

    Args:
        json_data: Parsed JSON response from USGS API

    Returns:
        DataFrame with datetime index and 'value' column
    """
    try:
        series = json_data["value"]["timeSeries"][0]["values"][0]["value"]
    except (KeyError, IndexError, TypeError):
        return pd.DataFrame(columns=["value"])

    dates = pd.to_datetime([x.get("dateTime") for x in series], errors="coerce")
    vals = pd.to_numeric([x.get("value") for x in series], errors="coerce")

    df = pd.DataFrame({"value": vals}, index=dates)
    df = df[~df.index.isna()]
    df.index = pd.to_datetime(df.index.date)

    return df.sort_index()


def fetch_daily_values(
    site_id: str,
    param_cd: str = DEFAULT_PARAM_DISCHARGE,
    stat_cd: str = STAT_MEAN,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    chunk_years: int = 5
) -> pd.DataFrame:
    """
    Fetch daily values from USGS NWIS with chunking to avoid throttling.

    Uses JSON format first, falls back to WaterML if JSON fails.
    Splits large date ranges into chunks to avoid 503 errors.

    Args:
        site_id: USGS site identifier (e.g., '12422500')
        param_cd: Parameter code (default: discharge)
        stat_cd: Statistic code (default: mean)
        start_date: Start date (YYYY-MM-DD or 'today')
        end_date: End date (YYYY-MM-DD or 'today')
        chunk_years: Years per chunk (smaller = more robust but slower)

    Returns:
        DataFrame with datetime index and 'value' column

    Example:
        >>> df = fetch_daily_values('12422500', start_date='2020-01-01', end_date='2023-12-31')
        >>> print(df['value'].mean())
    """
    # Parse dates
    if end_date is None or end_date == 'today':
        end = pd.Timestamp.today().normalize()
    else:
        end = pd.Timestamp(end_date).normalize()

    if start_date is None:
        start = end - pd.Timedelta(days=40 * 365)  # 40 years default
    else:
        start = pd.Timestamp(start_date).normalize()

    logger.info(f"Fetching daily values: site={site_id}, param={param_cd}, "
                f"{start.date()} to {end.date()}")

    # Create date chunks
    chunks: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    current = start
    while current <= end:
        chunk_end = min(
            current + pd.DateOffset(years=chunk_years) - pd.Timedelta(days=1),
            end
        )
        chunks.append((current, chunk_end))
        current = chunk_end + pd.Timedelta(days=1)

    # Fetch each chunk
    parts: List[pd.DataFrame] = []
    for i, (chunk_start, chunk_end) in enumerate(chunks, 1):
        # Try JSON first
        try:
            params_json = {
                "format": "json",
                "sites": site_id,
                "parameterCd": param_cd,
                "statCd": stat_cd,
                "startDT": chunk_start.date().isoformat(),
                "endDT": chunk_end.date().isoformat(),
            }
            text = http_get_text(BASE_URL_DV, params_json, retries=5)
            json_data = requests.models.complexjson.loads(text) if text else {}
            df_json = parse_json_series(json_data)

            if not df_json.empty:
                logger.info(f"Chunk {i}/{len(chunks)} (JSON): "
                           f"{chunk_start.date()}→{chunk_end.date()}, {len(df_json)} rows")
                parts.append(df_json)
                continue

        except Exception as e:
            logger.warning(f"JSON failed for chunk {i}: {e}")

        # Fallback to WaterML
        try:
            params_xml = {
                "format": "waterml,1.1",
                "sites": site_id,
                "parameterCd": param_cd,
                "statCd": stat_cd,
                "startDT": chunk_start.date().isoformat(),
                "endDT": chunk_end.date().isoformat(),
            }
            xml = http_get_text(BASE_URL_DV, params_xml, retries=5)
            df_xml = parse_waterml(xml)

            if not df_xml.empty:
                logger.info(f"Chunk {i}/{len(chunks)} (XML): "
                           f"{chunk_start.date()}→{chunk_end.date()}, {len(df_xml)} rows")
                parts.append(df_xml)
            else:
                logger.warning(f"Chunk {i} returned no data")

        except Exception as e:
            logger.error(f"WaterML also failed for chunk {i}: {e}")

    # Combine and clean
    if not parts:
        logger.warning(f"No data retrieved for site {site_id}")
        return pd.DataFrame(columns=["value"])

    result = pd.concat(parts).sort_index()
    result = result[~result.index.duplicated(keep="last")]  # Remove duplicates

    logger.info(f"Total rows retrieved: {len(result)}")
    return result


def fetch_instantaneous_values(
    site_id: str,
    param_cd: str = DEFAULT_PARAM_DISCHARGE,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    aggregate_to_daily: bool = True,
    chunk_days: int = 120
) -> pd.DataFrame:
    """
    Fetch instantaneous values (IV) from USGS NWIS and optionally aggregate to daily.

    IV data is typically recorded at 15-minute intervals. This function fetches
    the raw IV data and can aggregate it to daily means for compatibility with
    daily value workflows.

    Args:
        site_id: USGS site identifier (e.g., '12422500')
        param_cd: Parameter code (default: discharge)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD or 'today')
        aggregate_to_daily: If True, aggregate to daily means (default: True)
        chunk_days: Days per chunk to avoid API limits (default: 120)

    Returns:
        DataFrame with datetime index and 'value' column
    """
    # Parse dates
    if end_date is None or end_date == 'today':
        end = pd.Timestamp.today().normalize()
    else:
        end = pd.Timestamp(end_date).normalize()

    if start_date is None:
        start = end - pd.Timedelta(days=365)  # Default 1 year for IV
    else:
        start = pd.Timestamp(start_date).normalize()

    logger.info(f"Fetching instantaneous values: site={site_id}, param={param_cd}, "
                f"{start.date()} to {end.date()}")

    # Create date chunks (IV has stricter limits than DV)
    chunks: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    current = start
    while current <= end:
        chunk_end = min(current + pd.Timedelta(days=chunk_days), end)
        chunks.append((current, chunk_end))
        current = chunk_end + pd.Timedelta(days=1)

    # Fetch each chunk
    parts: List[pd.DataFrame] = []
    for i, (chunk_start, chunk_end) in enumerate(chunks, 1):
        try:
            params = {
                "format": "json",
                "sites": site_id,
                "parameterCd": param_cd,
                "startDT": chunk_start.date().isoformat(),
                "endDT": chunk_end.date().isoformat(),
            }
            text = http_get_text(BASE_URL_IV, params, retries=4, timeout=30)

            if text:
                import json
                json_data = json.loads(text)
                df_chunk = parse_json_series(json_data)

                if not df_chunk.empty:
                    logger.info(f"IV Chunk {i}/{len(chunks)}: "
                               f"{chunk_start.date()}→{chunk_end.date()}, {len(df_chunk)} rows")
                    parts.append(df_chunk)
        except Exception as e:
            logger.warning(f"IV fetch failed for chunk {i}: {e}")

    if not parts:
        logger.warning(f"No IV data retrieved for site {site_id}")
        return pd.DataFrame(columns=["value"])

    result = pd.concat(parts).sort_index()
    result = result[~result.index.duplicated(keep="last")]

    # Aggregate to daily means if requested
    if aggregate_to_daily and not result.empty:
        result = result.resample('D').mean().dropna()
        logger.info(f"Aggregated to {len(result)} daily values")

    return result


def check_iv_availability(site_id: str, param_cd: str) -> Optional[Dict[str, Any]]:
    """
    Quick check if instantaneous values are available for a parameter.
    Checks last 30 days of IV data.

    Returns dict with 'available', 'start', 'end' or None if not available.
    """
    try:
        end = pd.Timestamp.today()
        start = end - pd.Timedelta(days=30)

        params = {
            "format": "json",
            "sites": site_id,
            "parameterCd": param_cd,
            "startDT": start.date().isoformat(),
            "endDT": end.date().isoformat(),
        }
        text = http_get_text(BASE_URL_IV, params, retries=2, timeout=15)

        if text:
            import json
            json_data = json.loads(text)
            df = parse_json_series(json_data)

            if not df.empty:
                return {
                    'available': True,
                    'type': 'instantaneous',
                    'recent_start': df.index.min().strftime('%Y-%m-%d'),
                    'recent_end': df.index.max().strftime('%Y-%m-%d')
                }
    except Exception:
        pass

    return None


def fetch_discharge_data(
    site_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    positive_only: bool = True
) -> Optional[pd.DataFrame]:
    """
    Convenience function to fetch discharge data.

    Args:
        site_id: USGS site identifier
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD or 'today')
        positive_only: Filter out non-positive values

    Returns:
        DataFrame with 'Discharge_cfs' column or None if fetch failed

    Example:
        >>> df = fetch_discharge_data('12422500', '2020-01-01', '2023-12-31')
        >>> print(df['Discharge_cfs'].describe())
    """
    df = fetch_daily_values(
        site_id,
        param_cd=DEFAULT_PARAM_DISCHARGE,
        stat_cd=STAT_MEAN,
        start_date=start_date,
        end_date=end_date
    )

    if df.empty:
        return None

    df = df.rename(columns={"value": "Discharge_cfs"})

    if positive_only:
        df = df[df["Discharge_cfs"] > 0]

    return df


def fetch_stage_data(
    site_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    try_iv_fallback: bool = True
) -> Optional[pd.DataFrame]:
    """
    Fetch gage height (stage) data.

    First tries daily values (DV). If DV is not available, falls back to
    instantaneous values (IV) aggregated to daily means.

    Args:
        site_id: USGS site identifier
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD or 'today')
        try_iv_fallback: If True, try IV endpoint when DV is empty (default: True)

    Returns:
        DataFrame with 'Stage_ft' column or None if fetch failed
    """
    # Try daily values first
    df = fetch_daily_values(
        site_id,
        param_cd=DEFAULT_PARAM_STAGE,
        stat_cd=STAT_MEAN,
        start_date=start_date,
        end_date=end_date
    )

    if not df.empty:
        logger.info(f"Stage data from daily values: {len(df)} rows")
        df = df.rename(columns={"value": "Stage_ft"})
        return df

    # Fall back to instantaneous values if DV is empty
    if try_iv_fallback:
        logger.info(f"No daily stage data, trying instantaneous values for {site_id}")
        df = fetch_instantaneous_values(
            site_id,
            param_cd=DEFAULT_PARAM_STAGE,
            start_date=start_date,
            end_date=end_date,
            aggregate_to_daily=True
        )

        if not df.empty:
            logger.info(f"Stage data from IV (aggregated): {len(df)} rows")
            df = df.rename(columns={"value": "Stage_ft"})
            return df

    return None


# Legacy compatibility functions
def fetch_waterml_data(
    site_id: str,
    param_cd: str = DEFAULT_PARAM_DISCHARGE,
    start_date: str = "2000-01-01",
    end_date: str = "today",
    timeout: int = 90
) -> Optional[str]:
    """
    Legacy function: Fetch WaterML XML data.

    NOTE: Consider using fetch_daily_values() instead for better error handling.

    Returns:
        WaterML XML string or None
    """
    logger.info(f"Fetching WaterML: {site_id}, {param_cd}, {start_date} to {end_date}")

    params = {
        'format': 'waterml,1.1',
        'sites': site_id,
        'parameterCd': param_cd,
        'startDT': start_date,
        'endDT': end_date if end_date != 'today' else pd.Timestamp.today().date().isoformat()
    }

    try:
        response = requests.get(BASE_URL_DV, params=params, timeout=timeout)
        response.raise_for_status()
        logger.info(f"WaterML fetch successful: {site_id}")
        return response.text
    except requests.exceptions.Timeout:
        logger.error(f"WaterML fetch timeout: {site_id}")
    except requests.exceptions.RequestException as e:
        logger.error(f"WaterML fetch failed: {site_id}: {e}")

    return None
