"""
Test script - generates ONE PNG with ALL 10 available plot types.

Just hit run - no editing needed! This tests that all plots work.
"""

from hydrology.scripts.analyze_sites import run_analysis
from hydrology.data.inventory import get_site_info
from hydrology.core.logging_setup import setup_logging
import matplotlib.pyplot as plt

# Setup logging
logger = setup_logging(__name__, 'test_all_plots.log')

# =============================================================================
# TEST CONFIGURATION
# =============================================================================

TEST_SITE_ID = '12422500'  # Spokane River - good data availability

# ALL 10 available plots
ALL_PLOTS = [
    # Climate-dependent (will show N/A if Meteostat unavailable):
    'anomaly',              # 1. Recent vs historical
    'hexbin_temp',          # 2. Q vs Temperature
    'lagged_precip',        # 3. Q vs lagged precip
    'correlation_matrix',   # 4. Correlation heatmap
    'precip_discharge',     # 5. Precip overlay with cumulative

    # Discharge-only (always work):
    'timeseries',           # 6. Recent discharge timeseries
    'flow_duration',        # 7. Flow duration curve
    'monthly_boxplot',      # 8. Monthly boxplots
    'discharge_heatmap',    # 9. Discharge density heatmap
    'temporal_heatmap',     # 10. 4-panel temporal heatmap
]

# =============================================================================
# RUN TEST
# =============================================================================

if __name__ == '__main__':
    logger.info("=" * 80)
    logger.info("TESTING ALL 10 PLOT TYPES")
    logger.info("=" * 80)
    logger.info(f"Test site: {TEST_SITE_ID}")
    logger.info(f"Testing {len(ALL_PLOTS)} plot types")
    logger.info(f"Layout: AUTO (will choose best arrangement)")
    logger.info("")

    for i, plot in enumerate(ALL_PLOTS, 1):
        logger.info(f"  {i:2d}. {plot}")
    logger.info("")

    # Load site info from inventory
    logger.info("Loading site information from inventory...")
    site = get_site_info(TEST_SITE_ID)

    if not site:
        logger.error(f"Could not find test site {TEST_SITE_ID} in inventory!")
        logger.error("Check nwis_inventory_filtered_data_only.txt")
        exit(1)

    logger.info(f"Site loaded: {site['description']}")
    logger.info(f"Coordinates: {site['latitude']}, {site['longitude']}")
    logger.info("")

    # Build config
    config = {
        'analysis_parameters': {
            'param_cd': '00060',
            'start_date': '2010-01-01',  # 15 years of data
            'end_date': 'today',
            'plots': ALL_PLOTS,
            'layout': 'auto',  # Auto-choose best layout for 10 plots
            'dpi': 150,
        },
        'sites_to_process': [{
            'site_id': site['site_id'],
            'description': site['description'],
            'latitude': site['latitude'],
            'longitude': site['longitude'],
            'enabled': True
        }]
    }

    # Save config temporarily
    import yaml
    from hydrology.core.paths import PROJECT_ROOT
    temp_config = PROJECT_ROOT / 'temp_test_all_plots_config.yaml'

    logger.info("Creating temporary config...")
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)

    # Run analysis
    logger.info("Running analysis with ALL 10 plots...")
    logger.info("")

    try:
        results = run_analysis(str(temp_config))

        logger.info("")
        logger.info("=" * 80)

        if results:
            logger.info("✓ TEST COMPLETED SUCCESSFULLY!")
            logger.info("")
            logger.info(f"All {len(ALL_PLOTS)} plot types tested:")
            for i, plot in enumerate(ALL_PLOTS, 1):
                logger.info(f"  {i:2d}. {plot}")
            logger.info("")
            logger.info("Output saved to:")
            logger.info(f"  outputs/plots/{TEST_SITE_ID}/multi_plot/")
            logger.info("")
            logger.info("✓ Check the PNG file to verify all plots rendered correctly.")
            logger.info("✓ Climate-dependent plots (1-5) may show 'N/A' if Meteostat unavailable.")
            logger.info("✓ Discharge-only plots (6-10) should all render successfully.")
        else:
            logger.warning("✗ TEST FAILED - No results returned")
            logger.warning("Check logs above for errors")

        logger.info("=" * 80)

        # Show plot
        plt.show()

    except Exception as e:
        logger.error("")
        logger.error("=" * 80)
        logger.error("✗ TEST FAILED WITH ERROR:")
        logger.error(f"  {e}")
        logger.error("=" * 80)
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Clean up
        if temp_config.exists():
            temp_config.unlink()
            logger.info("Cleaned up temporary config")
