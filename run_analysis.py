"""
Quick analysis script - just hit play/run in VSCode!

Edit the SITE_IDS, PLOTS, and LAYOUT variables below, then run this script.
Site coordinates are automatically loaded from your inventory file.
"""

from hydrology.scripts.analyze_sites import run_analysis
from hydrology.data.inventory import get_multiple_sites
from hydrology.core.logging_setup import setup_logging
import matplotlib.pyplot as plt

# Setup logging
logger = setup_logging(__name__, 'run_analysis.log')

# =============================================================================
# EDIT THESE SETTINGS
# =============================================================================

# Which site(s) to analyze? (Just list the site IDs)
# Coordinates will be loaded automatically from nwis_inventory_filtered_data_only.txt
SITE_IDS = [
    '12422500',  # Spokane River at Spokane, WA
    # Add more site IDs here:
    # '12424000',  # Hangman Creek at Spokane, WA
    # '12431000',  # Little Spokane River at Dartford, WA
]

# Which plots do you want?
# Options: 'anomaly', 'hexbin_temp', 'lagged_precip', 'correlation_matrix',
#          'timeseries', 'flow_duration', 'monthly_boxplot', 'discharge_heatmap', 'temporal_heatmap'
PLOTS = [
    'timeseries',           # Recent discharge time series
    'flow_duration',        # Flow duration curve
    'monthly_boxplot',      # Monthly discharge distribution boxplots
    'discharge_heatmap',    # Discharge density vs day of year
    # 'temporal_heatmap',   # 4-panel temporal heatmap (5/10/20yr + total) - comment out if too big

    # Climate-dependent plots (uncomment if Meteostat is working):
    # 'anomaly',            # Recent vs historical comparison
    # 'hexbin_temp',        # Discharge vs Temperature
    # 'lagged_precip',      # Discharge vs lagged precipitation
    # 'correlation_matrix', # Correlation heatmap
]

# How to arrange them?
# Options: 'vertical', 'quad', 'horizontal', 'grid_2x3', 'grid_3x2', 'grid_2x5', 'grid_5x2', 'auto'
LAYOUT = 'vertical'  # Change to 'grid_2x5' for 2 columns x 5 rows (great for 10 plots)

# Date range
START_DATE = '2010-01-01'  # Change this to your desired start date
END_DATE = 'today'         # Or specify a date like '2023-12-31'

# =============================================================================
# RUN ANALYSIS (don't edit below here unless you know what you're doing)
# =============================================================================

if __name__ == '__main__':
    logger.info("=" * 80)
    logger.info("QUICK ANALYSIS SCRIPT")
    logger.info("=" * 80)

    # Load site info from inventory
    logger.info(f"Loading site information for: {SITE_IDS}")
    sites_from_inventory = get_multiple_sites(SITE_IDS)

    if not sites_from_inventory:
        logger.error("Could not find any of the requested sites in inventory file!")
        logger.error(f"Requested: {SITE_IDS}")
        logger.error("Check that site IDs exist in: nwis_inventory_filtered_data_only.txt")
        exit(1)

    # Build sites list for config
    sites = []
    for site in sites_from_inventory:
        sites.append({
            'site_id': site['site_id'],
            'description': site['description'],
            'latitude': site['latitude'],
            'longitude': site['longitude'],
            'enabled': True
        })

    logger.info(f"Sites loaded: {len(sites)}")
    for site in sites:
        logger.info(f"  - {site['site_id']}: {site['description']}")

    logger.info(f"Plots: {PLOTS}")
    logger.info(f"Layout: {LAYOUT}")
    logger.info("")

    # Build config
    config = {
        'analysis_parameters': {
            'param_cd': '00060',
            'start_date': START_DATE,
            'end_date': END_DATE,
            'plots': PLOTS,
            'layout': LAYOUT,
            'dpi': 150,
        },
        'sites_to_process': sites
    }

    # Save config temporarily
    import yaml
    from hydrology.core.paths import PROJECT_ROOT
    temp_config = PROJECT_ROOT / 'temp_run_config.yaml'
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)

    logger.info(f"Config saved to: {temp_config}")

    # Run analysis
    try:
        results = run_analysis(str(temp_config))

        if results:
            logger.info("")
            logger.info("=" * 80)
            logger.info("✓ SUCCESS! Analysis complete.")
            logger.info(f"  Processed {len(results)} site(s)")
            logger.info("")
            logger.info("Plots saved to:")
            for result in results:
                site_id = result['site_id']
                logger.info(f"  - outputs/plots/{site_id}/multi_plot/")
            logger.info("=" * 80)

            # Show plots
            plt.show()
        else:
            logger.warning("No results returned. Check logs for errors.")

    except Exception as e:
        logger.error(f"Error running analysis: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Clean up temp config
        if temp_config.exists():
            temp_config.unlink()
            logger.info("Cleaned up temporary config file")
