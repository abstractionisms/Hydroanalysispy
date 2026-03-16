"""
National Water Model (NWM) data integration.

Provides tools for fetching NWM forecast and analysis data and comparing
with observed USGS measurements.

The National Water Model produces:
- Analysis: Best estimate of current conditions (hourly)
- Short-range forecast: 0-18 hours ahead (hourly)
- Medium-range forecast: 0-10 days ahead (3-hourly)
- Long-range forecast: 0-30 days ahead (6-hourly)

Data sources:
- NOAA NOMADS: https://nomads.ncep.noaa.gov/
- Google Cloud: gs://national-water-model
- AWS S3: s3://noaa-nwm-pds

Example:
    >>> from hydrology.data.nwm import NWMClient, compare_nwm_usgs
    >>> client = NWMClient()
    >>> forecast = client.get_streamflow_forecast('12422500')
    >>> comparison = compare_nwm_usgs('12422500', '2024-01-01', '2024-01-07')
"""

import json
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import requests

from ..core.logging_setup import get_logger
from ..core.paths import CACHE_DIR, ensure_dir
from .usgs import fetch_daily_values, fetch_instantaneous_values

logger = get_logger(__name__)

# NWM v2.1 retrospective Zarr store on S3 (NOAA Open Data)
NWM_RETRO_ZARR = "s3://noaa-nwm-retrospective-2-1-zarr-pds/chrtout.zarr"


# NWM reach ID mapping service (USGS site to NWM reach)
# Updated to use the production NLDI API endpoint
NLDI_API_BASE = "https://api.water.usgs.gov/nldi/linked-data/nwissite/USGS-{site_id}"

# NOAA Water Prediction Service API (preferred method)
NOAA_API_BASE = "https://api.water.noaa.gov/nwps/v1"


@dataclass
class NWMForecast:
    """
    NWM forecast data container.

    Attributes:
        site_id: USGS site ID (or NWM reach ID)
        reach_id: NWM reach ID (COMID)
        forecast_type: Type of forecast (analysis, short_range, medium_range, long_range)
        reference_time: Model initialization time
        valid_times: List of forecast valid times
        streamflow: List of streamflow values (cms)
        units: Flow units
    """
    site_id: str
    reach_id: Optional[str]
    forecast_type: str
    reference_time: datetime
    valid_times: List[datetime]
    streamflow: List[float]
    units: str = "cms"

    def to_dataframe(self) -> pd.DataFrame:
        """Convert forecast to DataFrame."""
        return pd.DataFrame({
            'valid_time': self.valid_times,
            'streamflow': self.streamflow,
            'forecast_type': self.forecast_type,
            'reference_time': self.reference_time
        }).set_index('valid_time')

    def to_cfs(self) -> 'NWMForecast':
        """Convert streamflow to cubic feet per second."""
        if self.units == 'cms':
            # 1 cms = 35.3147 cfs
            return NWMForecast(
                site_id=self.site_id,
                reach_id=self.reach_id,
                forecast_type=self.forecast_type,
                reference_time=self.reference_time,
                valid_times=self.valid_times,
                streamflow=[q * 35.3147 for q in self.streamflow],
                units='cfs'
            )
        return self


@dataclass
class NWMUSGSComparison:
    """
    Results from comparing NWM forecasts with USGS observations.

    Attributes:
        site_id: USGS site ID
        start_date: Comparison period start
        end_date: Comparison period end
        n_observations: Number of comparison points
        bias: Mean bias (NWM - USGS)
        mae: Mean absolute error
        rmse: Root mean square error
        correlation: Correlation coefficient
        nash_sutcliffe: Nash-Sutcliffe efficiency
        percent_bias: Percent bias
    """
    site_id: str
    start_date: datetime
    end_date: datetime
    n_observations: int
    bias: float
    mae: float
    rmse: float
    correlation: float
    nash_sutcliffe: float
    percent_bias: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'site_id': self.site_id,
            'start_date': self.start_date.strftime('%Y-%m-%d'),
            'end_date': self.end_date.strftime('%Y-%m-%d'),
            'n_observations': self.n_observations,
            'bias': self.bias,
            'mae': self.mae,
            'rmse': self.rmse,
            'correlation': self.correlation,
            'nash_sutcliffe': self.nash_sutcliffe,
            'percent_bias': self.percent_bias
        }


class NWMClient:
    """
    Client for fetching National Water Model data.

    Supports multiple data sources:
    - NOAA Water Prediction Service API (primary)
    - Direct NOMADS/cloud access (fallback)

    Example:
        >>> client = NWMClient()
        >>> forecast = client.get_streamflow_forecast('12422500')
        >>> print(f"Forecast for next 18 hours: {forecast.streamflow[:18]}")
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize NWM client.

        Args:
            cache_dir: Directory for caching NWM data
        """
        self.cache_dir = cache_dir or (CACHE_DIR / 'nwm')
        ensure_dir(self.cache_dir)
        self.reach_cache: Dict[str, str] = {}  # site_id -> reach_id

    def get_reach_id(self, site_id: str) -> Optional[str]:
        """
        Get NWM reach ID (COMID) for a USGS site.

        Args:
            site_id: USGS site ID

        Returns:
            NWM reach ID or None if not found
        """
        # Check cache
        if site_id in self.reach_cache:
            return self.reach_cache[site_id]

        try:
            # Use NLDI service to find associated NWM reach
            url = NLDI_API_BASE.format(site_id=site_id)
            logger.info(f"Fetching NLDI data from: {url}")

            headers = {'Accept': 'application/json'}
            response = requests.get(url, headers=headers, timeout=30)

            logger.info(f"NLDI response status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                logger.info(f"NLDI response has {len(data.get('features', []))} features")

                if data.get('features'):
                    # Get the comid from the feature properties
                    props = data['features'][0].get('properties', {})
                    logger.info(f"Feature properties keys: {list(props.keys())}")

                    reach_id = str(props.get('comid', ''))
                    if reach_id:
                        self.reach_cache[site_id] = reach_id
                        logger.info(f"Found NWM reach {reach_id} for USGS site {site_id}")
                        return reach_id
                    else:
                        logger.warning(f"comid field empty or missing in properties: {props}")
            else:
                logger.warning(f"NLDI returned status {response.status_code}: {response.text[:200]}")

            logger.warning(f"Could not find NWM reach for USGS site {site_id}")
            return None

        except Exception as e:
            logger.error(f"Error looking up reach ID for {site_id}: {e}")
            return None

    def get_streamflow_forecast(
        self,
        site_id: str,
        forecast_type: str = 'short_range'
    ) -> Optional[NWMForecast]:
        """
        Get NWM streamflow forecast for a site.

        Args:
            site_id: USGS site ID
            forecast_type: Type of forecast (short_range, medium_range, long_range)

        Returns:
            NWMForecast object or None if unavailable
        """
        reach_id = self.get_reach_id(site_id)

        if not reach_id:
            logger.warning(f"Cannot get forecast without reach ID for {site_id}")
            return None

        try:
            # Use NOAA Water Prediction Service API
            url = f"{NOAA_API_BASE}/reaches/{reach_id}/streamflow"
            params = {
                'series': forecast_type
            }

            response = requests.get(url, params=params, timeout=60)

            if response.status_code == 200:
                data = response.json()
                return self._parse_forecast_response(site_id, reach_id, forecast_type, data)
            else:
                logger.warning(f"NWM API returned {response.status_code} for {site_id}")
                return None

        except Exception as e:
            logger.error(f"Error fetching NWM forecast for {site_id}: {e}")
            return None

    def get_analysis(
        self,
        site_id: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        Get NWM analysis (retrospective) data for a site.

        Note: The NOAA API returns the last ~5 days of analysis_assimilation data.
        Date filtering is done client-side after fetching.

        Args:
            site_id: USGS site ID
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with analysis streamflow or None
        """
        reach_id = self.get_reach_id(site_id)

        if not reach_id:
            return None

        try:
            # The API returns recent analysis_assimilation data
            url = f"{NOAA_API_BASE}/reaches/{reach_id}/streamflow"

            response = requests.get(url, timeout=60)

            if response.status_code == 200:
                data = response.json()
                df = self._parse_analysis_response(data)

                if df is not None and not df.empty:
                    # Filter to requested date range
                    start_dt = pd.to_datetime(start_date)
                    end_dt = pd.to_datetime(end_date)
                    # Make timezone-aware if needed
                    if df.index.tz is not None:
                        start_dt = start_dt.tz_localize(df.index.tz)
                        end_dt = end_dt.tz_localize(df.index.tz)
                    df = df[(df.index >= start_dt) & (df.index <= end_dt)]

                return df
            else:
                logger.warning(f"NWM analysis API returned {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Error fetching NWM analysis for {site_id}: {e}")
            return None

    def _parse_forecast_response(
        self,
        site_id: str,
        reach_id: str,
        forecast_type: str,
        data: Dict
    ) -> Optional[NWMForecast]:
        """Parse forecast API response."""
        try:
            # Extract time series data
            # Note: Actual API response structure may vary
            if 'data' not in data:
                return None

            values = data['data']

            # Validate and extract data with error handling for missing keys
            valid_times = []
            streamflow = []
            for v in values:
                if 'validTime' in v and 'value' in v:
                    try:
                        valid_times.append(
                            datetime.fromisoformat(v['validTime'].replace('Z', '+00:00'))
                        )
                        streamflow.append(float(v['value']))
                    except (ValueError, TypeError):
                        continue  # Skip malformed entries

            if not valid_times:
                return None

            ref_time = datetime.fromisoformat(
                data.get('referenceTime', datetime.now().isoformat()).replace('Z', '+00:00')
            )

            return NWMForecast(
                site_id=site_id,
                reach_id=reach_id,
                forecast_type=forecast_type,
                reference_time=ref_time,
                valid_times=valid_times,
                streamflow=streamflow,
                units='cms'
            )

        except Exception as e:
            logger.error(f"Error parsing forecast response: {e}")
            return None

    def _parse_analysis_response(self, data: Dict) -> Optional[pd.DataFrame]:
        """Parse analysis API response from NOAA NWM API."""
        try:
            # The API returns analysisAssimilation.series.data
            analysis_data = data.get('analysisAssimilation', {})
            series = analysis_data.get('series', {})
            values = series.get('data', [])

            if not values:
                logger.warning("No analysis data in response")
                return None

            # Units are already in cfs (ft³/s) from this API
            # Validate each entry before processing
            parsed_data = []
            for v in values:
                if 'validTime' in v and 'flow' in v:
                    try:
                        parsed_data.append({
                            'datetime': datetime.fromisoformat(v['validTime'].replace('Z', '+00:00')),
                            'streamflow_cfs': float(v['flow'])
                        })
                    except (ValueError, TypeError):
                        continue  # Skip malformed entries

            if not parsed_data:
                logger.warning("No valid analysis data after parsing")
                return None

            df = pd.DataFrame(parsed_data)

            df.set_index('datetime', inplace=True)
            # Also provide cms for compatibility
            df['streamflow_cms'] = df['streamflow_cfs'] / 35.3147

            return df

        except Exception as e:
            logger.error(f"Error parsing analysis response: {e}")
            return None

    def get_retrospective_streamflow(
        self,
        site_id: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """
        Get NWM v2.1 retrospective streamflow from the Zarr store on S3.

        The retrospective dataset covers 1979-2020 and provides hourly
        streamflow for all NWM reaches. This enables proper model evaluation
        over historical periods (not just the last ~5 days from the API).

        Args:
            site_id: USGS site ID
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with streamflow_cfs column, or None
        """
        reach_id = self.get_reach_id(site_id)
        if not reach_id:
            logger.warning(f"Cannot get retrospective data without reach ID for {site_id}")
            return None

        try:
            import xarray as xr
            import s3fs

            fs = s3fs.S3FileSystem(anon=True)
            store = s3fs.S3Map(root=NWM_RETRO_ZARR, s3=fs)

            ds = xr.open_zarr(store)

            # Find the feature_id index for this reach
            reach_int = int(reach_id)
            feature_ids = ds['feature_id'].values
            idx = np.where(feature_ids == reach_int)[0]

            if len(idx) == 0:
                logger.warning(f"Reach {reach_id} not found in NWM retrospective dataset")
                return None

            idx = idx[0]

            # Select time range and feature
            streamflow = ds['streamflow'].sel(
                time=slice(start_date, end_date),
            ).isel(feature_id=idx)

            df = streamflow.to_dataframe().reset_index()
            df = df.rename(columns={'time': 'datetime', 'streamflow': 'streamflow_cms'})
            df = df.set_index('datetime')

            # Convert cms to cfs
            df['streamflow_cfs'] = df['streamflow_cms'] * 35.3147

            # Keep only relevant columns
            df = df[['streamflow_cfs', 'streamflow_cms']]

            logger.info(f"Retrieved {len(df)} retrospective NWM records for {site_id}")
            return df

        except ImportError:
            logger.error("xarray and s3fs required for retrospective NWM. "
                        "Install with: conda install -c conda-forge xarray s3fs zarr")
            return None
        except Exception as e:
            logger.error(f"Error fetching retrospective NWM data for {site_id}: {e}")
            return None

    def evaluate_model_skill(
        self,
        site_id: str,
        start_date: str,
        end_date: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate NWM retrospective model skill against USGS observations.

        Computes NSE, KGE, RMSE, PBIAS, and correlation over a historical
        period using the retrospective Zarr dataset.

        Args:
            site_id: USGS site ID
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Dict with skill metrics and rating, or None
        """
        # Get NWM retrospective
        nwm_df = self.get_retrospective_streamflow(site_id, start_date, end_date)
        if nwm_df is None or nwm_df.empty:
            return None

        # Get USGS observations
        usgs_df = fetch_daily_values(
            site_id, param_cd='00060',
            start_date=start_date, end_date=end_date
        )
        if usgs_df is None or usgs_df.empty:
            return None

        # Resample NWM to daily mean
        nwm_daily = nwm_df[['streamflow_cfs']].resample('D').mean()

        # Normalize USGS index
        usgs_daily = usgs_df.copy()
        if usgs_daily.index.tz is not None:
            usgs_daily.index = usgs_daily.index.tz_localize(None)
        usgs_daily.index = usgs_daily.index.normalize()

        # Merge on common dates
        merged = pd.merge(
            usgs_daily[['value']].rename(columns={'value': 'observed'}),
            nwm_daily.rename(columns={'streamflow_cfs': 'simulated'}),
            left_index=True, right_index=True,
            how='inner'
        ).dropna()

        if len(merged) < 30:
            logger.warning(f"Insufficient overlap ({len(merged)} days)")
            return None

        obs = merged['observed'].values
        sim = merged['simulated'].values

        # Standard metrics
        bias = float(np.mean(sim - obs))
        mae = float(np.mean(np.abs(sim - obs)))
        rmse = float(np.sqrt(np.mean((sim - obs) ** 2)))
        correlation = float(np.corrcoef(obs, sim)[0, 1])

        # NSE
        ss_res = np.sum((obs - sim) ** 2)
        ss_tot = np.sum((obs - np.mean(obs)) ** 2)
        nse = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

        # KGE (Kling-Gupta Efficiency) — modern alternative to NSE
        r = correlation
        alpha = float(np.std(sim) / np.std(obs)) if np.std(obs) > 0 else np.nan
        beta = float(np.mean(sim) / np.mean(obs)) if np.mean(obs) > 0 else np.nan
        kge = float(1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2))

        # Percent bias
        sum_obs = np.sum(obs)
        pbias = float(100 * np.sum(sim - obs) / sum_obs) if sum_obs > 0 else np.nan

        return {
            'site_id': site_id,
            'start_date': start_date,
            'end_date': end_date,
            'n_observations': len(merged),
            'nse': nse,
            'kge': kge,
            'rmse': rmse,
            'mae': mae,
            'bias': bias,
            'correlation': correlation,
            'percent_bias': pbias,
            'alpha': alpha,  # KGE variability ratio
            'beta': beta,    # KGE bias ratio
            'rating': _rate_model_skill(nse, kge, correlation),
        }


def _rate_model_skill(nse: float, kge: float, correlation: float) -> str:
    """Rate model skill based on NSE, KGE, and correlation."""
    if nse > 0.75 and kge > 0.75:
        return 'Excellent'
    elif nse > 0.5 and kge > 0.5:
        return 'Good'
    elif nse > 0.25 and kge > 0.25:
        return 'Fair'
    elif nse > 0:
        return 'Poor'
    else:
        return 'Very Poor'


def compare_nwm_usgs(
    site_id: str,
    start_date: str,
    end_date: str,
    use_instantaneous: bool = False
) -> Optional[NWMUSGSComparison]:
    """
    Compare NWM analysis with USGS observations.

    Args:
        site_id: USGS site ID
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        use_instantaneous: Use instantaneous values instead of daily

    Returns:
        NWMUSGSComparison with error metrics or None if insufficient data

    Example:
        >>> result = compare_nwm_usgs('12422500', '2024-01-01', '2024-01-31')
        >>> print(f"Nash-Sutcliffe: {result.nash_sutcliffe:.3f}")
        >>> print(f"RMSE: {result.rmse:.1f} cfs")
    """
    client = NWMClient()

    # Fetch NWM analysis
    nwm_data = client.get_analysis(site_id, start_date, end_date)

    if nwm_data is None or nwm_data.empty:
        logger.warning(f"No NWM data available for {site_id}")
        return None

    # Fetch USGS observations
    if use_instantaneous:
        usgs_data = fetch_instantaneous_values(
            site_id, param_cd='00060',
            start_date=start_date, end_date=end_date
        )
    else:
        usgs_data = fetch_daily_values(
            site_id, param_cd='00060',
            start_date=start_date, end_date=end_date
        )

    if usgs_data is None or usgs_data.empty:
        logger.warning(f"No USGS data available for {site_id}")
        return None

    # Align data
    usgs_data = usgs_data.rename(columns={'value': 'usgs_cfs'})

    # Normalize timezones - convert both to timezone-naive for merging
    if nwm_data.index.tz is not None:
        nwm_data = nwm_data.copy()
        nwm_data.index = nwm_data.index.tz_convert('UTC').tz_localize(None)
    if usgs_data.index.tz is not None:
        usgs_data = usgs_data.copy()
        usgs_data.index = usgs_data.index.tz_convert('UTC').tz_localize(None)

    if use_instantaneous:
        # Resample NWM to match USGS instantaneous frequency
        merged = pd.merge_asof(
            usgs_data.sort_index(),
            nwm_data[['streamflow_cfs']].sort_index(),
            left_index=True,
            right_index=True,
            direction='nearest',
            tolerance=pd.Timedelta('1h')
        )
    else:
        # For daily values, resample NWM to daily mean
        nwm_daily = nwm_data[['streamflow_cfs']].resample('D').mean()
        # Normalize USGS daily data to just date (remove time component)
        usgs_daily = usgs_data.copy()
        usgs_daily.index = usgs_daily.index.normalize()
        merged = pd.merge(
            usgs_daily,
            nwm_daily,
            left_index=True,
            right_index=True,
            how='inner'
        )

    merged = merged.dropna()

    if len(merged) < 3:
        logger.warning(f"Insufficient overlap for comparison ({len(merged)} points)")
        return None

    # Calculate metrics
    obs = merged['usgs_cfs'].values
    sim = merged['streamflow_cfs'].values

    bias = np.mean(sim - obs)
    mae = np.mean(np.abs(sim - obs))
    rmse = np.sqrt(np.mean((sim - obs) ** 2))
    correlation = np.corrcoef(obs, sim)[0, 1]

    # Nash-Sutcliffe efficiency
    ss_res = np.sum((obs - sim) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    nse = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Percent bias
    sum_obs = np.sum(obs)
    pbias = 100 * np.sum(sim - obs) / sum_obs if sum_obs > 0 else np.nan

    return NWMUSGSComparison(
        site_id=site_id,
        start_date=datetime.strptime(start_date, '%Y-%m-%d'),
        end_date=datetime.strptime(end_date, '%Y-%m-%d'),
        n_observations=len(merged),
        bias=bias,
        mae=mae,
        rmse=rmse,
        correlation=correlation,
        nash_sutcliffe=nse,
        percent_bias=pbias
    )


def get_forecast_skill(
    site_id: str,
    n_days: int = 30
) -> Dict[str, Any]:
    """
    Evaluate NWM forecast skill over a historical period.

    Compares forecasts with observations to assess forecast quality
    at different lead times.

    Args:
        site_id: USGS site ID
        n_days: Number of days to evaluate

    Returns:
        Dict with forecast skill metrics by lead time
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=n_days)

    # Get comparison metrics
    comparison = compare_nwm_usgs(
        site_id,
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )

    if comparison is None:
        return {'error': 'Insufficient data for skill assessment'}

    return {
        'site_id': site_id,
        'evaluation_period_days': n_days,
        'metrics': comparison.to_dict(),
        'rating': _rate_forecast_skill(comparison)
    }


def _rate_forecast_skill(comparison: NWMUSGSComparison) -> str:
    """Rate forecast skill based on NSE and correlation."""
    nse = comparison.nash_sutcliffe
    corr = comparison.correlation

    if nse > 0.75 and corr > 0.9:
        return 'Excellent'
    elif nse > 0.5 and corr > 0.8:
        return 'Good'
    elif nse > 0.25 and corr > 0.6:
        return 'Fair'
    elif nse > 0:
        return 'Poor'
    else:
        return 'Very Poor'


# Convenience function for quick NWM vs USGS plot data
def get_comparison_data(
    site_id: str,
    start_date: str,
    end_date: str
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Get synchronized NWM and USGS data for plotting.

    Args:
        site_id: USGS site ID
        start_date: Start date
        end_date: End date

    Returns:
        Tuple of (NWM DataFrame, USGS DataFrame) or (None, None)
    """
    client = NWMClient()

    nwm_data = client.get_analysis(site_id, start_date, end_date)
    usgs_data = fetch_daily_values(site_id, param_cd='00060', start_date=start_date, end_date=end_date)

    return nwm_data, usgs_data
