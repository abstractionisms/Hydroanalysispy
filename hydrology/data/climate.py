"""
Climate data fetching for the hydrology package.

Providers:
  - Open-Meteo Historical archive (no key, gridded, excellent modern coverage)
  - Meteostat station daily (station-based; may pick dead nearest stations)
  - (Daymet is optional via hyriver and needs NASA Earthdata)

Meteostat caveat: Point(lat, lon) picks the *nearest* station even if that
station has no data in the requested window (e.g. Walla Walla 72788 ends 1988).
We therefore try several nearby stations with coverage overlap before failing.
"""

from __future__ import annotations

from typing import Optional, Dict, List, Any
import pandas as pd

from ..core.logging_setup import get_logger
from ..core.timezone import ensure_utc

logger = get_logger(__name__)


def _get_meteostat():
    try:
        from meteostat import Point, Daily, Stations
        return Point, Daily, Stations
    except ImportError:
        return None, None, None


def _normalize_ts(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def _station_covers_window(station_row, start_date: pd.Timestamp, end_date: pd.Timestamp) -> bool:
    """True if station daily inventory overlaps the requested window at all."""
    start = _normalize_ts(start_date)
    end = _normalize_ts(end_date)
    raw_start = station_row.get("daily_start", None) if hasattr(station_row, "get") else station_row["daily_start"] if "daily_start" in station_row.index else None
    raw_end = station_row.get("daily_end", None) if hasattr(station_row, "get") else station_row["daily_end"] if "daily_end" in station_row.index else None
    try:
        if pd.isna(raw_start) or pd.isna(raw_end):
            # Unknown inventory — still try fetching
            return True
        s0 = _normalize_ts(raw_start)
        s1 = _normalize_ts(raw_end)
    except Exception:
        return True
    # Overlap if station ends after request start and starts before request end
    return s1 >= start and s0 <= end


def _process_meteostat_daily(
    data: pd.DataFrame,
    include_temp: bool,
    include_precip: bool,
) -> Optional[pd.DataFrame]:
    if data is None or data.empty:
        return None

    cols_to_keep = []
    rename_map = {}
    if include_temp and "tavg" in data.columns:
        cols_to_keep.append("tavg")
        rename_map["tavg"] = "Temp_C"
    if include_precip and "prcp" in data.columns:
        cols_to_keep.append("prcp")
        rename_map["prcp"] = "Precip_mm"
    if not cols_to_keep:
        return None

    climate_df = data[cols_to_keep].copy().rename(columns=rename_map)
    climate_df = ensure_utc(climate_df)

    if "Precip_mm" in climate_df.columns:
        climate_df["Precip_mm"] = climate_df["Precip_mm"].fillna(0)
    if "Temp_C" in climate_df.columns:
        climate_df["Temp_C"] = climate_df["Temp_C"].ffill().bfill()

    # Reject nearly-empty series
    usable = 0
    if "Precip_mm" in climate_df.columns:
        usable = max(usable, int(climate_df["Precip_mm"].notna().sum()))
    if "Temp_C" in climate_df.columns:
        usable = max(usable, int(climate_df["Temp_C"].notna().sum()))
    if usable < 30:
        return None
    return climate_df


def fetch_climate_data(
    latitude: float,
    longitude: float,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    include_temp: bool = True,
    include_precip: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Fetch daily climate data using Meteostat with multi-station fallback.

    Tries nearby stations that actually inventory-cover the request window,
    instead of trusting Point() which may select a dead nearest station.
    """
    if start_date > end_date:
        logger.warning(
            f"Start date ({start_date.date()}) after end date ({end_date.date()}) - swapping"
        )
        start_date, end_date = end_date, start_date

    logger.info(
        f"Fetching climate data: Lat={latitude:.4f}, Lon={longitude:.4f}, "
        f"{start_date.date()} to {end_date.date()}"
    )

    Point, Daily, Stations = _get_meteostat()
    if Point is None or Daily is None:
        logger.warning("meteostat not installed - skipping Meteostat source")
        return None

    start_date = _normalize_ts(start_date)
    end_date = _normalize_ts(end_date)

    try:
        # 1) Multi-station search with coverage filter
        if Stations is not None:
            nearby = Stations().nearby(latitude, longitude).fetch(15)
            if nearby is not None and not nearby.empty:
                # Prefer stations with inventory overlap, ordered by distance
                if "distance" in nearby.columns:
                    nearby = nearby.sort_values("distance")
                candidates = []
                for sid, row in nearby.iterrows():
                    if _station_covers_window(row, start_date, end_date):
                        candidates.append(sid)
                # Also keep first few regardless as last-chance tries
                for sid in list(nearby.index)[:5]:
                    if sid not in candidates:
                        candidates.append(sid)

                for sid in candidates[:10]:
                    try:
                        data = Daily(str(sid), start_date, end_date).fetch()
                        climate_df = _process_meteostat_daily(
                            data, include_temp, include_precip
                        )
                        if climate_df is not None and not climate_df.empty:
                            name = nearby.loc[sid].get("name", sid) if sid in nearby.index else sid
                            logger.info(
                                f"Meteostat station {sid} ({name}): {len(climate_df)} rows"
                            )
                            return climate_df
                    except Exception as e:
                        logger.debug(f"Meteostat station {sid} failed: {e}")
                        continue

        # 2) Legacy Point() interpolation as last Meteostat attempt
        location = Point(latitude, longitude)
        data = Daily(location, start_date, end_date).fetch()
        climate_df = _process_meteostat_daily(data, include_temp, include_precip)
        if climate_df is not None and not climate_df.empty:
            logger.info(f"Climate data via Point(): {len(climate_df)} rows")
            return climate_df

        logger.warning("Meteostat returned no usable data for this location/period")
        return None

    except Exception as e:
        logger.error(f"Error fetching/processing climate data: {e}")
        return None


def fetch_nearest_station_info(
    latitude: float,
    longitude: float,
    max_distance_km: float = 150.0,
    prefer_recent: bool = True,
) -> Optional[Dict]:
    """
    Get information about a nearby weather station.

    When prefer_recent=True, skip stations whose daily inventory ended before
    ~2 years ago so the UI does not claim a dead station is the climate source.
    """
    try:
        Point, _, Stations = _get_meteostat()
        if Stations is None:
            logger.warning("meteostat not installed - skipping station lookup")
            return None

        stations = Stations().nearby(latitude, longitude).fetch(15)
        if stations is None or stations.empty:
            logger.warning(f"No weather stations found near ({latitude}, {longitude})")
            return None

        if "distance" in stations.columns:
            stations = stations.sort_values("distance")

        cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=365 * 2)
        chosen = None
        for sid, row in stations.iterrows():
            dist_m = row.get("distance", None)
            dist_km = (float(dist_m) / 1000.0) if dist_m is not None and pd.notna(dist_m) else None
            if dist_km is not None and dist_km > max_distance_km:
                continue
            if prefer_recent:
                raw_end = row.get("daily_end", None)
                try:
                    if raw_end is not None and not pd.isna(raw_end):
                        if _normalize_ts(raw_end) < cutoff:
                            continue
                except Exception:
                    pass
            chosen = (sid, row, dist_km)
            break

        # Fallback: absolute nearest even if historical-only
        if chosen is None:
            sid = stations.index[0]
            row = stations.iloc[0]
            dist_m = row.get("distance", 0)
            dist_km = float(dist_m) / 1000.0 if dist_m is not None else None
            chosen = (sid, row, dist_km)

        sid, station, dist_km = chosen
        station_dict = {
            "id": sid,
            "name": station.get("name", "Unknown"),
            "latitude": station.get("latitude", None),
            "longitude": station.get("longitude", None),
            "elevation_m": station.get("elevation", None),
            "distance_km": dist_km if dist_km is not None else 0.0,
        }
        daily_start = station.get("daily_start", None)
        daily_end = station.get("daily_end", None)
        if daily_start is not None and not pd.isna(daily_start):
            station_dict["daily_start"] = str(daily_start)[:10]
        if daily_end is not None and not pd.isna(daily_end):
            station_dict["daily_end"] = str(daily_end)[:10]

        logger.info(
            f"Nearest station: {station_dict['name']} "
            f"({station_dict.get('distance_km', '?')} km away)"
        )
        return station_dict

    except Exception as e:
        logger.error(f"Error fetching station info: {e}")
        return None


def fetch_open_meteo_climate(
    latitude: float,
    longitude: float,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> Optional[pd.DataFrame]:
    """
    Fetch historical daily climate from Open-Meteo Archive API (no API key).

    Robust primary source when Meteostat's nearest station has no modern data.
    Returns Temp_C / Precip_mm with UTC DatetimeIndex.
    """
    try:
        import requests

        if isinstance(start_date, pd.Timestamp):
            start_date = start_date.strftime("%Y-%m-%d")
        if isinstance(end_date, pd.Timestamp):
            end_date = end_date.strftime("%Y-%m-%d")

        logger.info(
            f"Fetching Open-Meteo historical: {latitude:.4f}, {longitude:.4f} "
            f"{start_date} to {end_date}"
        )

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "temperature_2m_mean,precipitation_sum",
            "timezone": "UTC",
        }
        resp = requests.get(url, params=params, timeout=45)
        if resp.status_code != 200:
            logger.warning(f"Open-Meteo returned {resp.status_code}: {resp.text[:200]}")
            return None

        payload = resp.json()
        daily = payload.get("daily") or {}
        if "time" not in daily:
            logger.warning("Open-Meteo returned no daily time axis")
            return None

        df = pd.DataFrame(
            {
                "date": pd.to_datetime(daily["time"]),
                "Temp_C": daily.get("temperature_2m_mean"),
                "Precip_mm": daily.get("precipitation_sum"),
            }
        ).set_index("date")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        if "Precip_mm" in df.columns:
            df["Precip_mm"] = df["Precip_mm"].fillna(0)
        if "Temp_C" in df.columns:
            df["Temp_C"] = df["Temp_C"].ffill().bfill()
        df = df.dropna(how="all")
        if df.empty:
            return None

        logger.info(f"Open-Meteo historical fetched: {len(df)} days")
        return df

    except Exception as e:
        logger.error(f"Open-Meteo historical fetch failed: {e}")
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
