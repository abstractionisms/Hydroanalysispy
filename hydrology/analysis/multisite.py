"""
Multi-site correlation and upstream/downstream analysis.

Provides tools for analyzing relationships between multiple USGS sites,
including lag correlation, flow routing, and watershed-level analysis.

Features:
- Cross-correlation analysis between sites
- Lag time estimation for upstream/downstream pairs
- Flow accumulation analysis
- Synchronized multi-site visualization

Example:
    >>> from hydrology.analysis.multisite import MultiSiteAnalyzer
    >>> analyzer = MultiSiteAnalyzer()
    >>> analyzer.add_site('12422500', name='Spokane River at Spokane')
    >>> analyzer.add_site('12419000', name='Spokane River near Post Falls')
    >>> results = analyzer.analyze_correlation()
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats, signal

from ..core.logging_setup import get_logger
from ..core.parameters import DEFAULT_DISCHARGE_CODE
from ..data.usgs import fetch_daily_values

logger = get_logger(__name__)


@dataclass
class SiteInfo:
    """Information about a monitoring site for multi-site analysis."""
    site_id: str
    name: str = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    drainage_area_sqmi: Optional[float] = None
    upstream_sites: List[str] = field(default_factory=list)
    downstream_sites: List[str] = field(default_factory=list)


@dataclass
class CorrelationResult:
    """Results from cross-correlation analysis between two sites."""
    site_a: str
    site_b: str
    correlation: float
    p_value: float
    lag_days: int
    lag_correlation: float
    relationship: str  # 'upstream', 'downstream', 'parallel', 'unknown'
    n_observations: int


@dataclass
class LagAnalysisResult:
    """Results from lag time analysis."""
    upstream_site: str
    downstream_site: str
    optimal_lag_days: int
    lag_correlation: float
    travel_time_hours: float
    lag_correlations: List[Tuple[int, float]]  # (lag, correlation) pairs


class MultiSiteAnalyzer:
    """
    Analyzer for multi-site hydrological relationships.

    Supports:
    - Cross-correlation between sites
    - Lag time analysis for upstream/downstream pairs
    - Synchronized data retrieval
    - Relationship inference from correlation patterns

    Example:
        >>> analyzer = MultiSiteAnalyzer()
        >>> analyzer.add_site('12422500')
        >>> analyzer.add_site('12419000')
        >>> analyzer.fetch_data('2020-01-01', '2023-12-31')
        >>> results = analyzer.analyze_all_pairs()
    """

    def __init__(self):
        """Initialize the multi-site analyzer."""
        self.sites: Dict[str, SiteInfo] = {}
        self.data: Dict[str, pd.DataFrame] = {}
        self.correlation_results: List[CorrelationResult] = []

    def add_site(
        self,
        site_id: str,
        name: str = "",
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        drainage_area: Optional[float] = None,
        upstream_of: Optional[List[str]] = None,
        downstream_of: Optional[List[str]] = None
    ):
        """
        Add a site to the analyzer.

        Args:
            site_id: USGS site ID
            name: Human-readable site name
            latitude: Site latitude
            longitude: Site longitude
            drainage_area: Drainage area in square miles
            upstream_of: List of site IDs this site is upstream of
            downstream_of: List of site IDs this site is downstream of
        """
        site = SiteInfo(
            site_id=site_id,
            name=name or site_id,
            latitude=latitude,
            longitude=longitude,
            drainage_area_sqmi=drainage_area,
            upstream_sites=upstream_of or [],
            downstream_sites=downstream_of or []
        )
        self.sites[site_id] = site
        logger.info(f"Added site: {site_id} ({name})")

    def remove_site(self, site_id: str):
        """Remove a site from the analyzer."""
        if site_id in self.sites:
            del self.sites[site_id]
            if site_id in self.data:
                del self.data[site_id]
            logger.info(f"Removed site: {site_id}")

    def fetch_data(
        self,
        start_date: str,
        end_date: str,
        param_code: str = DEFAULT_DISCHARGE_CODE
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for all sites.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            param_code: USGS parameter code

        Returns:
            Dict mapping site_id to DataFrame
        """
        for site_id in self.sites:
            try:
                df = fetch_daily_values(site_id, param_code, start_date, end_date)
                if df is not None and not df.empty:
                    self.data[site_id] = df
                    logger.info(f"Fetched {len(df)} records for site {site_id}")
                else:
                    logger.warning(f"No data available for site {site_id}")
            except Exception as e:
                logger.error(f"Error fetching data for site {site_id}: {e}")

        return self.data

    def get_synchronized_data(self) -> pd.DataFrame:
        """
        Get synchronized data for all sites with common dates.

        Returns:
            DataFrame with columns for each site's discharge
        """
        if not self.data:
            logger.warning("No data loaded. Call fetch_data() first.")
            return pd.DataFrame()

        # Create a combined dataframe
        dfs = []
        for site_id, df in self.data.items():
            site_df = df[['value']].copy()
            site_df.columns = [site_id]
            dfs.append(site_df)

        if not dfs:
            return pd.DataFrame()

        # Merge on index (datetime)
        combined = dfs[0]
        for df in dfs[1:]:
            combined = pd.merge(
                combined, df,
                left_index=True, right_index=True,
                how='inner'
            )

        logger.info(f"Synchronized data: {len(combined)} common observations")
        return combined

    def calculate_correlation(
        self,
        site_a: str,
        site_b: str,
        method: str = 'pearson'
    ) -> Optional[CorrelationResult]:
        """
        Calculate correlation between two sites.

        Args:
            site_a: First site ID
            site_b: Second site ID
            method: Correlation method ('pearson', 'spearman', 'kendall')

        Returns:
            CorrelationResult or None if insufficient data
        """
        if site_a not in self.data or site_b not in self.data:
            logger.warning(f"Missing data for one or both sites: {site_a}, {site_b}")
            return None

        # Get synchronized data
        df_a = self.data[site_a]
        df_b = self.data[site_b]

        # Merge on common dates
        merged = pd.merge(
            df_a[['value']], df_b[['value']],
            left_index=True, right_index=True,
            suffixes=('_a', '_b'),
            how='inner'
        ).dropna()

        if len(merged) < 30:
            logger.warning(f"Insufficient overlap ({len(merged)} points) for {site_a} vs {site_b}")
            return None

        # Calculate correlation
        if method == 'pearson':
            corr, p_val = stats.pearsonr(merged['value_a'], merged['value_b'])
        elif method == 'spearman':
            corr, p_val = stats.spearmanr(merged['value_a'], merged['value_b'])
        elif method == 'kendall':
            corr, p_val = stats.kendalltau(merged['value_a'], merged['value_b'])
        else:
            raise ValueError(f"Unknown correlation method: {method}")

        # Calculate lag correlation to determine relationship
        lag_result = self.calculate_lag_correlation(site_a, site_b, max_lag=7)
        lag_days = lag_result.optimal_lag_days if lag_result else 0
        lag_corr = lag_result.lag_correlation if lag_result else corr

        # Infer relationship
        if lag_result and lag_result.optimal_lag_days > 0:
            relationship = 'upstream'  # site_a is upstream of site_b
        elif lag_result and lag_result.optimal_lag_days < 0:
            relationship = 'downstream'  # site_a is downstream of site_b
        elif corr > 0.8:
            relationship = 'parallel'  # Similar timing, possibly parallel tributaries
        else:
            relationship = 'unknown'

        result = CorrelationResult(
            site_a=site_a,
            site_b=site_b,
            correlation=corr,
            p_value=p_val,
            lag_days=lag_days,
            lag_correlation=lag_corr,
            relationship=relationship,
            n_observations=len(merged)
        )

        self.correlation_results.append(result)
        return result

    def calculate_lag_correlation(
        self,
        site_upstream: str,
        site_downstream: str,
        max_lag: int = 10
    ) -> Optional[LagAnalysisResult]:
        """
        Calculate optimal lag time between upstream and downstream sites.

        Uses cross-correlation to find the lag that maximizes correlation,
        which indicates travel time between sites.

        Args:
            site_upstream: Upstream site ID
            site_downstream: Downstream site ID
            max_lag: Maximum lag in days to consider

        Returns:
            LagAnalysisResult or None if insufficient data
        """
        if site_upstream not in self.data or site_downstream not in self.data:
            return None

        # Get synchronized data
        df_up = self.data[site_upstream]
        df_down = self.data[site_downstream]

        merged = pd.merge(
            df_up[['value']], df_down[['value']],
            left_index=True, right_index=True,
            suffixes=('_up', '_down'),
            how='inner'
        ).dropna()

        if len(merged) < 30:
            return None

        # Calculate cross-correlation for different lags
        lag_correlations = []
        best_lag = 0
        best_corr = -1

        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                corr = merged['value_up'].corr(merged['value_down'])
            elif lag > 0:
                # Positive lag: upstream leads (expected for upstream site)
                corr = merged['value_up'].iloc[:-lag].corr(
                    merged['value_down'].iloc[lag:].reset_index(drop=True)
                )
            else:
                # Negative lag: downstream leads
                corr = merged['value_up'].iloc[-lag:].reset_index(drop=True).corr(
                    merged['value_down'].iloc[:lag]
                )

            if not np.isnan(corr):
                lag_correlations.append((lag, corr))
                if corr > best_corr:
                    best_corr = corr
                    best_lag = lag

        return LagAnalysisResult(
            upstream_site=site_upstream,
            downstream_site=site_downstream,
            optimal_lag_days=best_lag,
            lag_correlation=best_corr,
            travel_time_hours=best_lag * 24,
            lag_correlations=lag_correlations
        )

    def analyze_all_pairs(self, method: str = 'pearson') -> List[CorrelationResult]:
        """
        Analyze correlations between all pairs of sites.

        Args:
            method: Correlation method

        Returns:
            List of CorrelationResult for all pairs
        """
        results = []
        site_ids = list(self.sites.keys())

        for i, site_a in enumerate(site_ids):
            for site_b in site_ids[i+1:]:
                result = self.calculate_correlation(site_a, site_b, method)
                if result:
                    results.append(result)

        return results

    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        Get correlation matrix for all sites.

        Returns:
            DataFrame with correlation coefficients
        """
        synced = self.get_synchronized_data()
        if synced.empty:
            return pd.DataFrame()

        return synced.corr()

    def identify_upstream_downstream(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Automatically identify upstream/downstream relationships.

        Uses lag correlation analysis to determine which sites are
        upstream or downstream of others.

        Returns:
            Dict with 'upstream' and 'downstream' lists for each site
        """
        relationships = {site_id: {'upstream': [], 'downstream': []}
                        for site_id in self.sites}

        # Analyze all pairs
        results = self.analyze_all_pairs()

        for result in results:
            if result.relationship == 'upstream':
                # site_a is upstream of site_b
                relationships[result.site_a]['downstream'].append(result.site_b)
                relationships[result.site_b]['upstream'].append(result.site_a)
            elif result.relationship == 'downstream':
                # site_a is downstream of site_b
                relationships[result.site_a]['upstream'].append(result.site_b)
                relationships[result.site_b]['downstream'].append(result.site_a)

        return relationships

    def calculate_flow_contribution(
        self,
        downstream_site: str,
        upstream_sites: List[str]
    ) -> Dict[str, float]:
        """
        Estimate flow contribution from upstream sites to downstream.

        Uses regression analysis to estimate what fraction of downstream
        flow comes from each upstream site.

        Args:
            downstream_site: Site ID of downstream location
            upstream_sites: List of upstream site IDs

        Returns:
            Dict mapping upstream site ID to contribution fraction
        """
        if downstream_site not in self.data:
            logger.warning(f"No data for downstream site: {downstream_site}")
            return {}

        # Get synchronized data for all relevant sites
        all_sites = [downstream_site] + upstream_sites
        synced = self.get_synchronized_data()

        if synced.empty or not all(s in synced.columns for s in all_sites):
            logger.warning("Insufficient synchronized data for contribution analysis")
            return {}

        # Simple approach: normalize correlations as contribution estimates
        contributions = {}
        total_corr = 0

        for upstream in upstream_sites:
            corr = synced[downstream_site].corr(synced[upstream])
            if corr > 0:
                contributions[upstream] = corr
                total_corr += corr

        # Normalize to sum to 1
        if total_corr > 0:
            contributions = {k: v / total_corr for k, v in contributions.items()}

        return contributions

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of multi-site analysis.

        Returns:
            Dict with analysis summary
        """
        synced = self.get_synchronized_data()

        return {
            'n_sites': len(self.sites),
            'sites': list(self.sites.keys()),
            'n_common_observations': len(synced) if not synced.empty else 0,
            'date_range': {
                'start': synced.index.min().strftime('%Y-%m-%d') if not synced.empty else None,
                'end': synced.index.max().strftime('%Y-%m-%d') if not synced.empty else None
            },
            'correlations_analyzed': len(self.correlation_results),
            'mean_correlation': np.mean([r.correlation for r in self.correlation_results])
                if self.correlation_results else None
        }


def quick_correlation_check(
    site_ids: List[str],
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Quick correlation analysis between multiple sites.

    Convenience function for rapid multi-site correlation check.

    Args:
        site_ids: List of USGS site IDs
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        Correlation matrix DataFrame

    Example:
        >>> corr = quick_correlation_check(
        ...     ['12422500', '12419000', '12424000'],
        ...     '2020-01-01', '2023-12-31'
        ... )
        >>> print(corr)
    """
    analyzer = MultiSiteAnalyzer()

    for site_id in site_ids:
        analyzer.add_site(site_id)

    analyzer.fetch_data(start_date, end_date)

    return analyzer.get_correlation_matrix()
