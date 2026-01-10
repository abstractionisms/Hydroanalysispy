"""
Unified site analysis script with modular plot selection.

THIS IS YOUR NEW MAIN SCRIPT! Configure everything in a YAML/JSON file:
- Which sites to analyze
- Which plots to generate
- What layout to use

Run:
    python -m hydrology.scripts.analyze_sites --config my_analysis.yaml

Or from Python:
    from hydrology.scripts.analyze_sites import run_analysis
    run_analysis('my_analysis.yaml')
"""

import argparse
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd

from hydrology.core.logging_setup import setup_logging
from hydrology.core.config import load_config
from hydrology.data.usgs import fetch_waterml_data
from hydrology.data.climate import fetch_climate_data
from hydrology.visualization import create_multi_plot, PlotLayout
from hydrology.analysis.trends import calculate_annual_means
from hydrology.core.timezone import ensure_utc

logger = setup_logging(__name__, 'analyze_sites.log')


def parse_waterml_for_discharge(waterml_content, site_id):
    """Parse WaterML and filter for positive discharge values."""
    from hydrology.data.usgs import parse_waterml

    if not waterml_content:
        logger.warning(f"No content to parse: {site_id}")
        return None

    df = parse_waterml(waterml_content)
    if df is None or df.empty:
        return None

    df = df.rename(columns={'value': 'Discharge_cfs'})
    df = df[df['Discharge_cfs'] > 0]  # For log scale

    if df.empty:
        logger.warning(f"No positive discharge values: {site_id}")
        return None

    logger.info(f"Discharge data: {site_id} ({len(df)} values)")
    return df


def analyze_correlation(df_merged):
    """Calculate correlation matrix and p-values."""
    from scipy import stats
    from itertools import combinations
    import numpy as np

    logger.info("Performing correlation analysis...")
    results = {'corr_matrix': None, 'p_values': {}, 'lagged_precip_corr': None, 'lagged_precip_p': None}

    if df_merged is None or df_merged.empty:
        return results

    cols = ['Discharge_cfs', 'Temp_C', 'Precip_mm']
    cols = [c for c in cols if c in df_merged.columns]

    if len(cols) < 2:
        return results

    df_analysis = df_merged[cols].dropna()
    if len(df_analysis) < 3:
        return results

    # Correlation matrix
    try:
        results['corr_matrix'] = df_analysis.corr()
    except Exception as e:
        logger.error(f"Error calculating correlation: {e}")

    # P-values
    for col1, col2 in combinations(cols, 2):
        try:
            if df_analysis[col1].nunique() > 1 and df_analysis[col2].nunique() > 1:
                _, p_val = stats.pearsonr(df_analysis[col1], df_analysis[col2])
                results['p_values'][tuple(sorted((col1, col2)))] = p_val
        except (ValueError, TypeError) as e:
            logger.debug(f"Statistical calculation error: {e}")

    # Lagged precip correlation
    if 'Precip_mm' in cols and 'Discharge_cfs' in cols:
        try:
            df_lag = df_analysis.copy()
            df_lag['Precip_mm_lag1'] = df_lag['Precip_mm'].shift(1)
            df_lag = df_lag.dropna()

            if len(df_lag) >= 3:
                corr, p_val = stats.pearsonr(df_lag['Discharge_cfs'], df_lag['Precip_mm_lag1'])
                results['lagged_precip_corr'] = corr
                results['lagged_precip_p'] = p_val
        except (ValueError, TypeError) as e:
            logger.debug(f"Statistical calculation error: {e}")

    return results


def process_site(site_config: Dict[str, Any], analysis_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single site: fetch data, analyze, create plots.

    Args:
        site_config: Site configuration dict
        analysis_config: Global analysis configuration

    Returns:
        Dict with results and data
    """
    site_id = site_config.get('site_id')
    description = site_config.get('description', f'Site {site_id}')

    if not site_config.get('enabled', True):
        logger.info(f"Skipping disabled site: {site_id}")
        return None

    logger.info(f"=== Processing Site: {site_id} ({description}) ===")

    # Get parameters
    param_cd = site_config.get('param_cd', analysis_config.get('param_cd', '00060'))
    start_date = site_config.get('start_date', analysis_config.get('start_date', '2000-01-01'))
    end_date = site_config.get('end_date', analysis_config.get('end_date', 'today'))
    latitude = site_config.get('latitude')
    longitude = site_config.get('longitude')

    if latitude is None or longitude is None:
        logger.error(f"Missing lat/lon for {site_id}")
        return None

    # Fetch discharge data
    logger.info(f"Fetching discharge data: {site_id}")
    discharge_wml = fetch_waterml_data(site_id, param_cd, start_date, end_date)
    df_q = parse_waterml_for_discharge(discharge_wml, site_id) if discharge_wml else None

    # Fetch climate data
    logger.info(f"Fetching climate data: {site_id}")
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.now() if end_date.lower() == 'today' else datetime.strptime(end_date, '%Y-%m-%d')
    df_climate = fetch_climate_data(latitude, longitude, start_dt, end_dt,
                                   include_temp=True, include_precip=True)

    # Merge data
    df_merged = None
    analysis_results = None

    if df_q is not None and not df_q.empty and df_climate is not None and not df_climate.empty:
        logger.info(f"Merging discharge and climate data: {site_id}")

        # Normalize timezones to UTC
        df_q = ensure_utc(df_q)
        df_climate = ensure_utc(df_climate)

        df_merged = pd.merge(df_q, df_climate, left_index=True, right_index=True, how='inner')

        if not df_merged.empty:
            logger.info(f"Merged data: {len(df_merged)} rows")
            analysis_results = analyze_correlation(df_merged)
        else:
            logger.warning(f"No overlapping data: {site_id}")

    # Prepare data for plots
    plot_data = {
        'df_q': df_q,
        'df_merged': df_merged,
        'analysis_results': analysis_results
    }

    # Get plot configuration
    plots_to_create = site_config.get('plots') or analysis_config.get('plots', ['anomaly', 'hexbin_temp', 'lagged_precip', 'timeseries'])
    layout = site_config.get('layout') or analysis_config.get('layout', 'vertical')

    # Convert layout string to enum
    layout_map = {
        'vertical': PlotLayout.VERTICAL,
        'quad': PlotLayout.QUAD,
        'horizontal': PlotLayout.HORIZONTAL,
        'grid_2x3': PlotLayout.GRID_2x3,
        'grid_3x2': PlotLayout.GRID_3x2,
        'grid_2x5': PlotLayout.GRID_2x5,
        'grid_5x2': PlotLayout.GRID_5x2,
        'auto': PlotLayout.AUTO
    }
    layout_enum = layout_map.get(layout.lower(), PlotLayout.VERTICAL)

    # Create plots
    logger.info(f"Creating plots: {plots_to_create}")
    try:
        fig = create_multi_plot(
            plots=plots_to_create,
            layout=layout_enum,
            data=plot_data,
            site_id=site_id,
            title=description,
            dpi=analysis_config.get('dpi', 150)
        )

        if fig:
            logger.info(f"Successfully created multi-plot for {site_id}")
        else:
            logger.warning(f"Failed to create multi-plot for {site_id}")

    except Exception as e:
        logger.error(f"Error creating plots for {site_id}: {e}")

    logger.info(f"=== Finished Site: {site_id} ===\n")

    return {
        'site_id': site_id,
        'status': 'completed',
        'data': plot_data
    }


def run_analysis(config_path: str = None):
    """
    Run analysis based on configuration file.

    Args:
        config_path: Path to configuration file (YAML or JSON)
    """
    logger.info("==================================================")
    logger.info("  UNIFIED SITE ANALYSIS - MODULAR PLOTTING       ")
    logger.info("==================================================\n")

    # Load configuration
    if config_path is None:
        config_path = 'analysis_config.yaml'

    logger.info(f"Loading configuration: {config_path}")
    config = load_config(config_path, required=False)

    if not config:
        logger.error("Failed to load configuration or empty config")
        return

    # Get analysis parameters and sites
    analysis_params = config.get('analysis_parameters', {})
    sites = config.get('sites_to_process', config.get('sites', []))

    if not sites:
        logger.warning("No sites found in configuration")
        return

    logger.info(f"Found {len(sites)} sites to process\n")

    # Process each site
    results = []
    for site_config in sites:
        result = process_site(site_config, analysis_params)
        if result:
            results.append(result)

    logger.info("\n==================================================")
    logger.info(f"  ANALYSIS COMPLETE: {len(results)} sites processed")
    logger.info("==================================================")

    return results


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description='Unified hydrological site analysis with modular plotting'
    )
    parser.add_argument(
        '--config',
        default='analysis_config.yaml',
        help='Path to configuration file (YAML or JSON)'
    )
    args = parser.parse_args()

    run_analysis(args.config)


if __name__ == '__main__':
    main()
