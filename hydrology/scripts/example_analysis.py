"""
Example analysis script using the hydrology package.

This demonstrates how to use the new shared utilities to write
cleaner, more maintainable analysis scripts.

Run from command line:
    python -m hydrology.scripts.example_analysis

Or import and use:
    from hydrology.scripts.example_analysis import analyze_site
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import from our new package - no more duplicate code!
from hydrology.core.logging_setup import setup_logging
from hydrology.core.config import load_config, get_site_config
from hydrology.core.paths import get_site_plot_dir, CONFIG_DIR
from hydrology.data.usgs import fetch_discharge_data
from hydrology.data.climate import fetch_climate_data, merge_discharge_climate
from hydrology.analysis.trends import (
    calculate_annual_means,
    analyze_trend,
    calculate_correlation
)

# Set up logging
logger = setup_logging(__name__, 'example_analysis.log')


def analyze_site(site_id: str, latitude: float, longitude: float,
                 start_date: str = '2000-01-01', end_date: str = 'today'):
    """
    Example function showing the new workflow.

    Args:
        site_id: USGS site identifier (e.g., '12422500')
        latitude: Site latitude for climate data
        longitude: Site longitude for climate data
        start_date: Analysis start date
        end_date: Analysis end date
    """
    logger.info(f"Analyzing site {site_id}")

    # 1. Fetch discharge data using shared utility
    logger.info("Fetching discharge data...")
    discharge = fetch_discharge_data(site_id, start_date, end_date)

    if discharge is None:
        logger.error("Failed to fetch discharge data")
        return

    logger.info(f"Discharge data: {len(discharge)} days")

    # 2. Fetch climate data using shared utility
    logger.info("Fetching climate data...")
    climate = fetch_climate_data(
        latitude, longitude,
        pd.Timestamp(start_date), pd.Timestamp(end_date),
        include_temp=True, include_precip=True
    )

    if climate is None:
        logger.error("Failed to fetch climate data")
        return

    logger.info(f"Climate data: {len(climate)} days")

    # 3. Merge datasets
    logger.info("Merging discharge and climate data...")
    merged = merge_discharge_climate(discharge, climate)

    if merged.empty:
        logger.error("Merged data is empty")
        return

    logger.info(f"Merged data: {len(merged)} days")

    # 4. Calculate annual means
    logger.info("Calculating annual means...")
    annual_discharge = calculate_annual_means(merged, 'Discharge_cfs')
    annual_temp = calculate_annual_means(merged, 'Temp_C')

    # 5. Trend analysis
    if annual_discharge is not None:
        logger.info("Analyzing discharge trends...")
        discharge_trends = analyze_trend(annual_discharge, 'Annual Discharge')

        if discharge_trends['linear_regression']:
            lr = discharge_trends['linear_regression']
            logger.info(f"  Linear trend: {lr['slope']:.2f} cfs/year "
                       f"(p={lr['p_value']:.4f})")

        if discharge_trends['mann_kendall']:
            mk = discharge_trends['mann_kendall']
            logger.info(f"  Mann-Kendall: {mk['trend']} (p={mk['p_value']:.4f})")

    # 6. Correlation analysis
    logger.info("Analyzing correlations...")
    discharge_temp_corr = calculate_correlation(
        merged, 'Discharge_cfs', 'Temp_C'
    )
    discharge_precip_corr = calculate_correlation(
        merged, 'Discharge_cfs', 'Precip_mm'
    )

    if discharge_temp_corr:
        logger.info(f"  Discharge vs Temperature: r={discharge_temp_corr[0]:.3f}, "
                   f"p={discharge_temp_corr[1]:.4f}")

    if discharge_precip_corr:
        logger.info(f"  Discharge vs Precipitation: r={discharge_precip_corr[0]:.3f}, "
                   f"p={discharge_precip_corr[1]:.4f}")

    # 7. Create simple visualization
    logger.info("Creating plots...")
    plot_dir = get_site_plot_dir(site_id, 'example')

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Discharge time series
    ax = axes[0]
    merged['Discharge_cfs'].plot(ax=ax, alpha=0.7, label='Daily Discharge')
    if annual_discharge is not None:
        annual_discharge.plot(ax=ax, marker='o', linewidth=2,
                             label='Annual Mean', color='red')
    ax.set_ylabel('Discharge (cfs)')
    ax.set_title(f'Discharge Time Series - Site {site_id}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Discharge vs Temperature
    ax = axes[1]
    ax.scatter(merged['Temp_C'], merged['Discharge_cfs'],
              alpha=0.3, s=10)
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Discharge (cfs)')
    ax.set_title('Discharge vs Temperature')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save using portable path
    plot_path = plot_dir / f'site_{site_id}_analysis.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"Plot saved to: {plot_path}")

    plt.close()

    logger.info("Analysis complete!")

    return {
        'discharge': discharge,
        'climate': climate,
        'merged': merged,
        'annual_discharge': annual_discharge,
        'trends': discharge_trends if annual_discharge is not None else None,
        'correlations': {
            'discharge_temp': discharge_temp_corr,
            'discharge_precip': discharge_precip_corr,
        }
    }


def main():
    """
    Example main function.

    You can customize this to load sites from a config file.
    """
    # Example: Spokane River at Spokane, WA
    results = analyze_site(
        site_id='12422500',
        latitude=47.6593,
        longitude=-117.4491,
        start_date='2000-01-01',
        end_date='2023-12-31'
    )

    print("\nAnalysis complete! Check outputs/logs/ for log file and outputs/plots/ for figures.")


if __name__ == '__main__':
    main()
