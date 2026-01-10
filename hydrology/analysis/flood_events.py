"""
Flood event detection and animation processing.

Provides tools for identifying major flood events, preparing multi-site
animation data, and calculating event characteristics like timing and
propagation.

Example:
    >>> from hydrology.analysis.flood_events import FloodEventAnalyzer
    >>> analyzer = FloodEventAnalyzer('12422500')
    >>> events = analyzer.get_top_events(n=5)
    >>> animation = analyzer.prepare_animation(events[0])
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..core.logging_setup import get_logger
from ..data.usgs import fetch_peak_streamflow, fetch_instantaneous_values
from ..data.nldi import discover_related_sites, order_sites_by_flow

logger = get_logger(__name__)


@dataclass
class FloodEvent:
    """
    A single flood event with metadata.

    Attributes:
        site_id: Origin USGS site ID
        peak_date: Date/time of peak discharge
        peak_discharge_cfs: Peak discharge value
        water_year: Water year of the event
        rank: Rank among all peaks (1 = largest)
    """
    site_id: str
    peak_date: datetime
    peak_discharge_cfs: float
    water_year: Optional[int] = None
    rank: Optional[int] = None
    peak_gage_height_ft: Optional[float] = None

    def __str__(self) -> str:
        date_str = self.peak_date.strftime('%Y-%m-%d') if self.peak_date else 'Unknown'
        return f"Flood Event: {date_str} - {self.peak_discharge_cfs:,.0f} cfs"


@dataclass
class SiteTimeSeries:
    """
    Time series data for a single site during an event.

    Attributes:
        site_id: USGS site identifier
        site_name: Human-readable site name
        data: DataFrame with datetime index and 'value' column
        peak_value: Maximum discharge during the event
        peak_time: Time of peak discharge
        lag_hours: Hours after origin site peak (negative = before)
    """
    site_id: str
    site_name: str
    data: pd.DataFrame
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    distance_km: Optional[float] = None
    direction: Optional[str] = None
    peak_value: Optional[float] = None
    peak_time: Optional[datetime] = None
    lag_hours: Optional[float] = None


@dataclass
class FloodEventAnimation:
    """
    Complete animation data for a flood event across multiple sites.

    Attributes:
        event: The flood event being animated
        origin_site_id: The site where the event was detected
        sites: List of SiteTimeSeries in flow order (upstream to downstream)
        time_range: (start, end) datetime tuple
        frame_timestamps: List of timestamps for animation frames
        frame_interval_minutes: Time between frames
    """
    event: FloodEvent
    origin_site_id: str
    sites: List[SiteTimeSeries] = field(default_factory=list)
    time_range: Optional[Tuple[datetime, datetime]] = None
    frame_timestamps: List[datetime] = field(default_factory=list)
    frame_interval_minutes: int = 60

    @property
    def n_frames(self) -> int:
        return len(self.frame_timestamps)

    @property
    def site_ids(self) -> List[str]:
        return [s.site_id for s in self.sites]

    def get_frame_data(self, frame_idx: int) -> Dict[str, float]:
        """Get discharge values for all sites at a specific frame."""
        if frame_idx < 0 or frame_idx >= len(self.frame_timestamps):
            return {}

        timestamp = self.frame_timestamps[frame_idx]
        values = {}

        for site in self.sites:
            if site.data is not None and not site.data.empty:
                # Find closest timestamp
                idx = site.data.index.get_indexer([timestamp], method='nearest')[0]
                if idx >= 0 and idx < len(site.data):
                    values[site.site_id] = site.data['value'].iloc[idx]

        return values

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all site data to a single DataFrame with sites as columns."""
        if not self.sites:
            return pd.DataFrame()

        dfs = {}
        for site in self.sites:
            if site.data is not None and not site.data.empty:
                dfs[site.site_id] = site.data['value']

        if not dfs:
            return pd.DataFrame()

        return pd.DataFrame(dfs)


class FloodEventAnalyzer:
    """
    Analyzer for flood events at a USGS site.

    Provides methods to identify major flood events, discover related
    monitoring sites, and prepare animation data.

    Example:
        >>> analyzer = FloodEventAnalyzer('12422500')
        >>> events = analyzer.get_top_events(n=10)
        >>> print(f"Top event: {events[0]}")
    """

    def __init__(self, site_id: str):
        """
        Initialize analyzer for a specific site.

        Args:
            site_id: USGS site identifier
        """
        self.site_id = site_id
        self._peak_data: Optional[pd.DataFrame] = None
        self._related_sites: Optional[List[Dict]] = None

    def get_peak_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """Get cached peak streamflow data."""
        if self._peak_data is None or force_refresh:
            self._peak_data = fetch_peak_streamflow(self.site_id)
        return self._peak_data

    def get_top_events(
        self,
        n: int = 10,
        min_year: Optional[int] = None
    ) -> List[FloodEvent]:
        """
        Get the top N flood events by peak discharge.

        Args:
            n: Number of events to return
            min_year: Only include events from this year onward

        Returns:
            List of FloodEvent objects, sorted by discharge (highest first)
        """
        df = self.get_peak_data()

        if df.empty:
            return []

        # Filter by year if specified
        if min_year and 'water_year' in df.columns:
            df = df[df['water_year'] >= min_year]

        # Sort and rank
        df = df.sort_values('peak_discharge_cfs', ascending=False).head(n)

        events = []
        for rank, (_, row) in enumerate(df.iterrows(), 1):
            events.append(FloodEvent(
                site_id=self.site_id,
                peak_date=row.get('peak_date'),
                peak_discharge_cfs=row.get('peak_discharge_cfs'),
                water_year=row.get('water_year'),
                peak_gage_height_ft=row.get('peak_gage_height_ft'),
                rank=rank
            ))

        return events

    def discover_related_sites(
        self,
        distance_km: float = 100,
        include_tributaries: bool = False,
        max_sites: int = 8,
        force_refresh: bool = False
    ) -> List[Dict]:
        """
        Discover monitoring sites along the same river network.

        Args:
            distance_km: Maximum distance to search
            include_tributaries: Include tributary sites
            max_sites: Maximum number of sites
            force_refresh: Force re-fetch from API

        Returns:
            List of site info dictionaries
        """
        if self._related_sites is None or force_refresh:
            self._related_sites = discover_related_sites(
                self.site_id,
                direction='both',
                distance_km=distance_km,
                include_tributaries=include_tributaries,
                max_sites=max_sites
            )
        return self._related_sites

    def prepare_animation(
        self,
        event: FloodEvent,
        days_before: int = 5,
        days_after: int = 10,
        distance_km: float = 100,
        frame_interval_minutes: int = 60,
        max_sites: int = 8
    ) -> FloodEventAnimation:
        """
        Prepare complete animation data for a flood event.

        Fetches instantaneous data for the origin site and all discovered
        related sites, calculates peak times and lags, and generates
        animation frame timestamps.

        Args:
            event: The flood event to animate
            days_before: Days before peak to include
            days_after: Days after peak to include
            distance_km: Distance for site discovery
            frame_interval_minutes: Minutes between animation frames
            max_sites: Maximum number of sites to include

        Returns:
            FloodEventAnimation with all site data and frame info
        """
        if not event.peak_date:
            logger.error("Event has no peak date")
            return FloodEventAnimation(event=event, origin_site_id=self.site_id)

        # Calculate time window
        start_time = event.peak_date - timedelta(days=days_before)
        end_time = event.peak_date + timedelta(days=days_after)

        logger.info(f"Preparing animation for {event}, window: {start_time} to {end_time}")

        # Discover related sites
        related_sites = self.discover_related_sites(
            distance_km=distance_km,
            max_sites=max_sites - 1  # Leave room for origin site
        )

        # Build site list (origin + related)
        all_site_info = [
            {
                'site_id': self.site_id,
                'name': 'Origin Site',
                'direction': 'origin',
                'distance_km': 0,
                'latitude': None,
                'longitude': None,
            }
        ] + related_sites

        # Fetch IV data for all sites (parallel)
        site_data = self._fetch_multi_site_iv(
            [s['site_id'] for s in all_site_info],
            start_time,
            end_time
        )

        # Build SiteTimeSeries objects
        sites = []
        for info in all_site_info:
            sid = info['site_id']
            df = site_data.get(sid, pd.DataFrame())

            # Calculate peak
            peak_value = None
            peak_time = None
            lag_hours = None

            if not df.empty and 'value' in df.columns:
                peak_idx = df['value'].idxmax()
                peak_value = df['value'].max()
                peak_time = peak_idx
                if event.peak_date and peak_time:
                    lag_hours = (peak_time - event.peak_date).total_seconds() / 3600

            sites.append(SiteTimeSeries(
                site_id=sid,
                site_name=info.get('name', sid),
                data=df,
                latitude=info.get('latitude'),
                longitude=info.get('longitude'),
                distance_km=info.get('distance_km', 0),
                direction=info.get('direction'),
                peak_value=peak_value,
                peak_time=peak_time,
                lag_hours=lag_hours
            ))

        # Order sites by flow (upstream to downstream)
        sites = self._order_sites_by_flow(sites)

        # Generate frame timestamps
        frame_timestamps = self._generate_frame_timestamps(
            start_time, end_time, frame_interval_minutes
        )

        return FloodEventAnimation(
            event=event,
            origin_site_id=self.site_id,
            sites=sites,
            time_range=(start_time, end_time),
            frame_timestamps=frame_timestamps,
            frame_interval_minutes=frame_interval_minutes
        )

    def _fetch_multi_site_iv(
        self,
        site_ids: List[str],
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, pd.DataFrame]:
        """Fetch instantaneous values for multiple sites in parallel."""
        results = {}
        start_str = start_time.strftime('%Y-%m-%d')
        end_str = end_time.strftime('%Y-%m-%d')

        def fetch_site(site_id: str) -> Tuple[str, pd.DataFrame]:
            try:
                df = fetch_instantaneous_values(
                    site_id,
                    start_date=start_str,
                    end_date=end_str,
                    aggregate_to_daily=False
                )
                return site_id, df if df is not None else pd.DataFrame()
            except Exception as e:
                logger.warning(f"Failed to fetch IV for {site_id}: {e}")
                return site_id, pd.DataFrame()

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(fetch_site, sid): sid for sid in site_ids}
            for future in as_completed(futures):
                site_id, df = future.result()
                results[site_id] = df
                if not df.empty:
                    logger.info(f"Fetched {len(df)} IV records for {site_id}")

        return results

    def _order_sites_by_flow(self, sites: List[SiteTimeSeries]) -> List[SiteTimeSeries]:
        """Order sites from upstream to downstream based on direction and distance."""
        # Separate by direction
        upstream = [s for s in sites if s.direction == 'upstream']
        origin = [s for s in sites if s.direction == 'origin']
        downstream = [s for s in sites if s.direction == 'downstream']

        # Sort upstream by distance (furthest first)
        upstream.sort(key=lambda s: s.distance_km or 0, reverse=True)

        # Sort downstream by distance (closest first)
        downstream.sort(key=lambda s: s.distance_km or 0)

        return upstream + origin + downstream

    def _generate_frame_timestamps(
        self,
        start: datetime,
        end: datetime,
        interval_minutes: int
    ) -> List[datetime]:
        """Generate list of timestamps for animation frames."""
        timestamps = []
        current = start
        delta = timedelta(minutes=interval_minutes)

        while current <= end:
            timestamps.append(current)
            current += delta

        return timestamps


def calculate_event_statistics(animation: FloodEventAnimation) -> Dict[str, Any]:
    """
    Calculate summary statistics for a flood event animation.

    Args:
        animation: FloodEventAnimation object

    Returns:
        Dict with event statistics including:
        - total_sites: Number of sites with data
        - peak_discharge: Maximum discharge across all sites
        - propagation_time_hours: Time from first to last peak
        - sites_with_data: List of site IDs with valid data
    """
    stats = {
        'total_sites': len(animation.sites),
        'sites_with_data': 0,
        'peak_discharge': None,
        'peak_site': None,
        'propagation_time_hours': None,
        'first_peak_site': None,
        'last_peak_site': None,
    }

    # Collect peak times and values
    peak_times = []
    peak_values = []

    for site in animation.sites:
        if site.data is not None and not site.data.empty:
            stats['sites_with_data'] += 1
            if site.peak_value:
                peak_values.append((site.site_id, site.peak_value))
            if site.peak_time:
                peak_times.append((site.site_id, site.peak_time))

    # Max discharge
    if peak_values:
        max_site, max_val = max(peak_values, key=lambda x: x[1])
        stats['peak_discharge'] = max_val
        stats['peak_site'] = max_site

    # Propagation time
    if len(peak_times) >= 2:
        sorted_times = sorted(peak_times, key=lambda x: x[1])
        first_site, first_time = sorted_times[0]
        last_site, last_time = sorted_times[-1]
        stats['propagation_time_hours'] = (last_time - first_time).total_seconds() / 3600
        stats['first_peak_site'] = first_site
        stats['last_peak_site'] = last_site

    return stats
