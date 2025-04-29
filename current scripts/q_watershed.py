# --- Imports ---
# Core Scientific Libraries
import pandas as pd
import numpy as np

# Geospatial Libraries
import geopandas as gpd
import rasterio
import xarray as xr # Often used with rasterio or other spatial data
import contextily as ctx # For adding basemaps

# USGS Data Retrieval & Hydro-specific Libraries
import dataretrieval.nwis as nwis # For USGS water data
try:
    from pynhd import NLDI # For NLDI/NHDPlus web services
except ImportError:
    print("Error: The 'pynhd' library is not installed (needed for NLDI). Install with: pip install pynhd")
    NLDI = None # Allows script to run partially even if NLDI fails

# Analysis & Statistics Libraries
import pymannkendall as mk # For trend analysis
# from scipy import stats # Potentially for other stats, currently not used in core trend analysis

# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns

# System & Utility Libraries
import json # For reading config file
import os # For path handling
import datetime # For handling dates
import logging # Added logging module

# --- Configuration File Path ---
# !!! IMPORTANT: Set this path to where your config.json (or config2.json) is located.
# Use an absolute path (e.g., 'C:/Users/YourUser/.../config2.json') or a relative path.
CONFIG_FILE_PATH = 'c:/Users/Cam/source/repos/Hydrology/Hydrology/config2.json'

# --- NLCD Data Folder Path ---
# !!! IMPORTANT: Set this to the folder on your computer where you download NLCD .tif files !!!
# Obtain NLCD data for relevant years/areas (e.g., from USGS EarthExplorer).
# Example: NLCD_DATA_FOLDER = 'C:/Users/YourUser/Data/NLCD/'
NLCD_DATA_FOLDER = './NLCD_Data/' # Default relative path - Adjust if needed

# --- Output Paths ---
LOG_FILE = 'logs/q_watershed.log' # Log file path
PLOT_BASE_DIR = 'plots/q_watershed' # Base directory for plots

# --- Logging Setup ---
def setup_logging(log_file=LOG_FILE):
    """Configures basic logging to file and console."""
    root_logger = logging.getLogger()
    # Prevent adding duplicate handlers if script is run multiple times
    if root_logger.hasHandlers():
         for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()

    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    root_logger.setLevel(logging.DEBUG) # Set minimum level

    # File Handler
    try:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        print(f"Error setting up file logger for {log_file}: {e}")

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)


# --- Helper Functions ---

def load_config(file_path):
    """Loads configuration from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            config = json.load(f)
        logging.info(f"Config loaded from {file_path}") # Changed print to logging.info
        return config
    except FileNotFoundError:
        logging.error(f"Error: Config file not found at {file_path}") # Changed print to logging.error
        return None
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from {file_path}") # Changed print to logging.error
        return None

def get_watershed_boundary(site_id):
    """
    Gets the upstream watershed boundary for a USGS site ID using pynhd (NLDI).
    Returns a GeoDataFrame or None.
    """
    if NLDI is None:
        logging.warning("Skipping watershed delineation: pynhd library not available.") # Changed print to logging.warning
        return None
    if not site_id:
        logging.error("Error: No site_id provided for watershed delineation.") # Changed print to logging.error
        return None

    logging.info(f"\nQuerying NLDI for watershed boundary for site ID: {site_id}...") # Changed print to logging.info
    try:
        nldi_client = NLDI()
        watershed_gdf = nldi_client.get_basins(site_id)

        if watershed_gdf is None or watershed_gdf.empty:
            logging.warning(f"No watershed found via NLDI for site ID {site_id}.") # Changed print to logging.warning
            return None

        logging.info(f"Watershed boundary found for site ID {site_id}.") # Changed print to logging.info
        # print("Watershed GeoDataFrame head:\n", watershed_gdf.head()) # Optional debug print
        return watershed_gdf

    except Exception as e:
        logging.error(f"Error during watershed delineation for site {site_id}: {e}") # Changed print to logging.error
        return None

def analyze_land_use_in_watershed(watershed_gdf, nlcd_folder_path):
    """
    Analyzes NLCD land use percentages within the watershed for available years.
    Assumes NLCD rasters (.tif) are in nlcd_folder_path.
    Returns a pandas DataFrame of results per year.
    """
    if watershed_gdf is None or watershed_gdf.empty:
        logging.warning("Skipping land use analysis: No watershed boundary.") # Changed print to logging.warning
        return None
    if not os.path.exists(nlcd_folder_path):
        logging.error(f"Error: NLCD data folder not found at {nlcd_folder_path}. Update NLCD_DATA_FOLDER path.") # Changed print to logging.error
        return None

    logging.info(f"Analyzing land use within watershed using data from {nlcd_folder_path}...") # Changed print to logging.info
    land_use_results = {}
    watershed_geom = watershed_gdf.geometry.iloc[0]

    # Simplified NLCD classes - map codes to names. Add others as needed.
    # Refer to NLCD legend for years you use.
    nlcd_class_map = {
        11: 'Open Water', 21: 'Developed, Open Space', 22: 'Developed, Low Intensity',
        23: 'Developed, Medium Intensity', 24: 'Developed, High Intensity',
        41: 'Deciduous Forest', 42: 'Evergreen Forest', 43: 'Mixed Forest',
        81: 'Pasture/Hay', 82: 'Cultivated Crops', 90: 'Woody Wetlands', 95: 'Emergent Wetlands',
        # ... include other relevant codes ...
    }
    developed_classes = [21, 22, 23, 24] # Grouped developed types

    # Common NLCD years - keep only years for which you have files
    nlcd_years_to_check = [2001, 2004, 2006, 2008, 2011, 2013, 2016, 2019, 2021]

    for year in nlcd_years_to_check:
        file_pattern_start = f"nlcd_{year}"
        nlcd_files = [f for f in os.listdir(nlcd_folder_path) if f.startswith(file_pattern_start) and f.endswith(".tif")]

        if not nlcd_files: continue # Skip year if no file found

        nlcd_file_path = os.path.join(nlcd_folder_path, nlcd_files[0])
        # print(f"Processing {nlcd_file_path}...") # Optional debug print

        try:
            with rasterio.open(nlcd_file_path) as src:
                # Reproject watershed to raster CRS for clipping
                watershed_reprojected = watershed_gdf.to_crs(src.crs)
                from rasterio.mask import mask
                out_image, _ = mask(src, [watershed_reprojected.geometry.iloc[0]], crop=True, filled=False)
                land_cover_data = out_image[0] # Assuming single band

                # Mask out nodata values
                nodata = src.nodata
                if nodata is not None:
                    valid_pixels = land_cover_data[land_cover_data != nodata]
                else:
                     valid_pixels = land_cover_data

                unique_classes, counts = np.unique(valid_pixels, return_counts=True)
                total_valid_pixels = counts.sum()
                if total_valid_pixels == 0:
                     logging.warning(f"Warning: No valid NLCD pixels in watershed for {year}.") # Changed print to logging.warning
                     continue

                year_results = {}
                developed_area_pixels = 0
                for class_code, count in zip(unique_classes, counts):
                    class_code_int = int(class_code)
                    class_name = nlcd_class_map.get(class_code_int, f'Class {class_code_int} (Unknown)')
                    percentage = (count / total_valid_pixels) * 100.0
                    year_results[class_name] = percentage

                    if class_code_int in developed_classes:
                        developed_area_pixels += count

                if total_valid_pixels > 0:
                    year_results['% Developed'] = (developed_area_pixels / total_valid_pixels) * 100.0

                land_use_results[year] = year_results

        except rasterio.errors.RasterioIOError as e:
             logging.error(f"Error reading NLCD file {nlcd_file_path}: {e}") # Changed print to logging.error
             continue
        except Exception as e:
            logging.error(f"Error processing NLCD for year {year}: {e}") # Changed print to logging.error
            continue # Continue to next year

    if land_use_results:
        land_use_df = pd.DataFrame.from_dict(land_use_results, orient='index')
        land_use_df.index.name = 'Year'
        land_use_df = land_use_df.sort_index()
        logging.info("\nLand Use Analysis Results (Percentages):") # Changed print to logging.info
        logging.info(land_use_df.to_string()) # Use to_string() for logging DataFrames
        return land_use_df
    else:
        logging.warning("No land use data processed for available years.") # Changed print to logging.warning
        return None

def fetch_streamflow_data(site_id, param_cd, start_date_str, end_date_str):
    """
    Fetches daily streamflow data for a site using dataretrieval.
    Handles 'today' in date strings. Returns pandas DataFrame or None.
    """
    if not nwis:
         logging.warning("Skipping streamflow data fetch: dataretrieval not available.") # Changed print to logging.warning
         return None

    logging.info(f"\nFetching streamflow data for site {site_id} ({param_cd})...") # Changed print to logging.info
    try:
        # Convert "today" string to actual date in YYYY-MM-DD format
        current_start_date = start_date_str
        current_end_date = end_date_str

        if isinstance(current_start_date, str) and current_start_date.lower() == 'today':
            current_start_date = datetime.date.today().strftime('%Y-%m-%d')
            logging.info(f"Converted start_date 'today' to {current_start_date}") # Changed print to logging.info
        if isinstance(current_end_date, str) and current_end_date.lower() == 'today':
            current_end_date = datetime.date.today().strftime('%Y-%m-%d')
            logging.info(f"Converted end_date 'today' to {current_end_date}") # Changed print to logging.info

        # Use nwis.get_dv for daily values
        df, metadata = nwis.get_dv(
            sites=site_id,
            parameterCd=param_cd,
            start=current_start_date,
            end=current_end_date
        )

        if df.empty:
            logging.warning(f"No data found for site {site_id} in the specified range.") # Changed print to logging.warning
            return None

        # Rename discharge column (e.g., '00060_00003' to 'discharge_cfs')
        discharge_col_pattern = f'{param_cd}_'
        q_col = None
        for col in df.columns:
            if col.startswith(discharge_col_pattern) and ('_cd' not in col): # Avoid code columns
                q_col = col
                break

        if q_col:
             df = df.rename(columns={q_col: 'discharge_cfs'})
             # Optional: Drop associated code column if it exists
             code_col = f'{q_col}_cd'
             if code_col in df.columns:
                  df = df.drop(columns=[code_col])
        else:
             logging.warning(f"Warning: Could not identify discharge column for param {param_cd} in site {site_id}. Columns: {df.columns.tolist()}") # Changed print to logging.warning
             # Fallback heuristic for common discharge param if standard pattern fails
             if param_cd == '00060' and '00060' in df.columns:
                  df = df.rename(columns={'00060': 'discharge_cfs'})
                  logging.info(f"Assuming column '00060' is discharge for site {site_id}.") # Changed print to logging.info
             else:
                  logging.warning(f"Could not find discharge column for site {site_id}. Skipping.") # Changed print to logging.warning
                  return None


        # Ensure index is datetime for time series analysis
        if not isinstance(df.index, pd.DatetimeIndex):
             try:
                  df.index = pd.to_datetime(df.index)
             except Exception as e:
                  logging.error(f"Failed to convert index to datetime for {site_id}: {e}") # Changed print to logging.error
                  return None

        logging.info(f"Successfully fetched {len(df)} records for site {site_id}.") # Changed print to logging.info
        return df

    except Exception as e:
        logging.error(f"Error fetching data for site {site_id}: {e}") # Changed print to logging.error
        # print(f"Attempted URL parameters: sites={site_id}, parameterCd={param_cd}, start={current_start_date}, end={current_end_date}") # Optional debug
        return None


def analyze_streamflow_trends(df, site_id,param_cd):
    """
    Calculates trends in annual streamflow metrics (Mean, Q10, Q90) using Mann-Kendall.
    Returns trend summary dict and annual metrics DataFrame.
    """
    if df is None or df.empty or 'discharge_cfs' not in df.columns:
        logging.warning(f"No valid streamflow data for trend analysis for site {site_id}.") # Changed print to logging.warning
        return None, None # Return None for both outputs

    logging.info(f"\nAnalyzing streamflow trends for site {site_id}...") # Changed print to logging.info

    # Calculate annual metrics
    annual_metrics_df = pd.DataFrame(index=pd.to_datetime(df.index).year.unique())
    annual_metrics_df.index.name = 'Year'

    # Ensure index is datetime (should be from fetch_streamflow_data)
    if not isinstance(df.index, pd.DatetimeIndex):
         logging.error("Trend analysis requires DatetimeIndex.") # Changed print to logging.error # Should not happen if fetch worked


    # --- Calculate Metrics and Trends ---
    metrics_to_analyze = {
        'Annual_Mean_Flow': lambda x: x.resample('AS').mean(),
        'Annual_Q90_Flow': lambda x: x.resample('AS').apply(lambda y: y.quantile(0.9)),
        'Annual_Q10_Flow': lambda x: x.resample('AS').apply(lambda y: y.quantile(0.1)),
        # Add other metrics like annual max, min, etc. here
    }

    trend_summary = {}
    for metric_name, resample_func in metrics_to_analyze.items():
        try:
            annual_series = resample_func(df['discharge_cfs'])
            annual_metrics_df[metric_name] = annual_series.values # Store even if NaNs
            annual_series_for_trend = annual_series.dropna() # Drop NaNs for trend test

            if len(annual_series_for_trend) >= 4: # Mann-Kendall requires at least 4 data points
                 logging.info(f" Calculating trend for {metric_name} ({len(annual_series_for_trend)} data points)...") # Changed print to logging.info
                 trend_res = mk.original_mk(annual_series_for_trend)
                 trend_summary[metric_name] = {
                      'trend': trend_res.trend,       # e.g., 'increase', 'decrease', 'no trend'
                      'p_value': trend_res.p,         # P-value of the test
                      'significant': trend_res.h,     # True if trend is significant (default alpha=0.05)
                      'slope': trend_res.slope        # Sen's slope (rate of change)
                 }
                 logging.info(f"  {metric_name} Trend: {trend_res.trend}, p={trend_res.p:.3f}, Slope={trend_res.slope:.2f}") # Changed print to logging.info
            else:
                 logging.warning(f" Not enough data points ({len(annual_series_for_trend)}) for {metric_name} trend analysis (min 4 needed).") # Changed print to logging.warning

        except Exception as e:
            logging.error(f"Error analyzing trend for {metric_name}: {e}") # Changed print to logging.error
            # Continue with other metrics

    if trend_summary:
        logging.info("\nStreamflow Trend Summary:") # Changed print to logging.info
        # Print as DataFrame for better readability
        logging.info(pd.DataFrame(trend_summary).T.to_string()) # Changed print to logging.info and used to_string()
    else:
        logging.warning("No streamflow trends could be calculated.") # Changed print to logging.warning


    return trend_summary, annual_metrics_df # Return trend results and calculated annual metrics


# --- Plotting Functions ---

def plot_watershed_boundary(watershed_gdf, site_coords, site_id, description, plot_base_dir):
     """Plots the delineated watershed boundary with a basemap."""
     if watershed_gdf is None or watershed_gdf.empty or not ctx:
          logging.warning("Skipping watershed plotting: No watershed data or contextily not available.") # Changed print to logging.warning
          return

     logging.info(f"\nPlotting watershed for {description}...") # Changed print to logging.info
     try:
         # Reproject to a suitable projected CRS for plotting (e.g., Web Mercator for contextily)
         plot_crs = "EPSG:3857" # Common CRS for web maps/basemaps
         watershed_plot = watershed_gdf.to_crs(plot_crs)

         fig, ax = plt.subplots(1, 1, figsize=(10, 10))

         # Plot watershed polygon
         watershed_plot.plot(ax=ax, color='blue', edgecolor='black', alpha=0.5, label='Watershed')

         # Plot the original site point
         point_gdf = gpd.GeoDataFrame(
             [{'geometry': gpd.points_from_xy([site_coords['longitude']], [site_coords['latitude']])[0]}],
             crs="EPSG:4326" # CRS of the original lat/lon
         )
         point_plot = point_gdf.to_crs(plot_crs)
         point_plot.plot(ax=ax, color='red', marker='o', markersize=50, label='Gauge Site')

         # Add a basemap
         ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik) # Or try other providers

         ax.set_title(f"Watershed Delineated for {description}\nUSGS Site {site_id}")
         # Turn off axis labels as they are less meaningful for Web Mercator basemaps
         ax.set_xticks([])
         ax.set_yticks([])
         # ax.legend() # Add legend if desired, but can clutter map

         plt.tight_layout()
         # Create site-specific plot directory and save
         site_plot_dir = os.path.join(plot_base_dir, site_id)
         os.makedirs(site_plot_dir, exist_ok=True)
         plot_filename = f"{site_id}_watershed_boundary.png"
         plot_path = os.path.join(site_plot_dir, plot_filename)
         plt.savefig(plot_path, dpi=300)
         logging.info(f"Watershed plot saved as {plot_path}") # Changed print to logging.info
         # plt.show() # Show plot interactively if desired (e.g., in Jupyter/Colab)
         plt.close(fig) # Close figure to free memory

     except Exception as e:
          logging.error(f"Error plotting watershed for {site_id}: {e}") # Changed print to logging.error
          logging.error("Ensure contextily and its dependencies are installed.") # Changed print to logging.error


def plot_annual_streamflow_metrics(annual_metrics_df, streamflow_trend_results, site_id, description, plot_base_dir):
    """Plots key annual streamflow metrics and their trends."""
    if annual_metrics_df is None or annual_metrics_df.empty:
        logging.warning(f"Skipping annual streamflow plotting: No annual data for site {site_id}.")
        return

    logging.info(f"Plotting annual streamflow metrics for {description}...")
    try:
        metrics_to_plot = {
            'Annual_Mean_Flow': {'ylabel': 'Mean Daily Discharge (CFS)', 'color': 'blue'},
            'Annual_Q90_Flow': {'ylabel': 'Discharge (CFS)', 'color': 'orange'},
            'Annual_Q10_Flow': {'ylabel': 'Discharge (CFS)', 'color': 'green'},
        }

        # Create a figure with subplots for each metric
        fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(10, 4 * len(metrics_to_plot)), sharex=True)
        fig.suptitle(f"Annual Streamflow Metrics for {description} ({site_id})", y=1.02)

        # Corrected loop iteration to use metrics_to_plot
        for i, (metric_name, plot_params) in enumerate(metrics_to_plot.items()):
            if metric_name in annual_metrics_df.columns:
                # Handle case of single subplot
                ax = axes[i] if len(metrics_to_plot) > 1 else axes

                # Plot data points
                annual_series = annual_metrics_df[metric_name].dropna()  # Plot only available points
                if not annual_series.empty:
                    ax.plot(annual_series.index, annual_series.values, marker='o', linestyle='-', color=plot_params['color'],
                            label=metric_name.replace('_', ' '))

                    # Plot trend line if calculated and significant
                    # Trend results are stored with the metric name directly
                    trend_key = metric_name
                    if streamflow_trend_results and trend_key in streamflow_trend_results:
                        trend_info = streamflow_trend_results[trend_key]
                        # Check if 'significant' key exists and is True
                        if trend_info.get('significant', False):
                            # Calculate points for the trend line using Sen's slope
                            # Trend line starts at the first non-NaN value's year
                            first_year = annual_series.index[0]
                            last_year = annual_series.index[-1]  # Use index of last data point
                            start_value = annual_series.iloc[0]

                            # Create datetime objects for the trend line years
                            trend_line_years_dt = pd.to_datetime([f'{first_year.year}-01-01', f'{last_year.year}-01-01'])

                            # Slope is change per year, relative to the first year
                            # Need to calculate the number of full years for slope application
                            num_years = (last_year.year - first_year.year)
                            end_value = start_value + trend_info['slope'] * num_years

                            ax.plot(trend_line_years_dt, [start_value, end_value], linestyle='--', color='red',
                                    label=f"MK Trend (Slope={trend_info['slope']:.2f})")

                ax.set_ylabel(plot_params['ylabel'])
                ax.set_title(metric_name.replace('_', ' '))
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.legend()  # Add legend to each subplot

        # Set xlabel only on the bottom subplot
        # Need to handle the case where axes is a single Axes object if only 1 metric
        if isinstance(axes, np.ndarray):
            axes[-1].set_xlabel('Year')
        else: # Case with only one subplot
             axes.set_xlabel('Year')


        plt.tight_layout()
        # Create site-specific plot directory and save
        site_plot_dir = os.path.join(plot_base_dir, site_id)
        os.makedirs(site_plot_dir, exist_ok=True)
        plot_filename = f"{site_id}_annual_metrics_plots.png"
        plot_path = os.path.join(site_plot_dir, plot_filename)
        plt.savefig(plot_path, dpi=300)
        logging.info(f"Annual metrics plot saved as {plot_path}")
        # plt.show() # Show plot interactively if desired
        plt.close(fig)

    except Exception as e:
        logging.error(f"Error plotting annual streamflow metrics for {site_id}: {e}")

def plot_land_use_change_trend(land_use_analysis_df, site_id, description, plot_base_dir, metric='% Developed'):
    """Plots the trend of a specific land use metric over the years."""
    if land_use_analysis_df is None or land_use_analysis_df.empty or metric not in land_use_analysis_df.columns:
        logging.warning(f"Skipping land use trend plotting: No data for metric '{metric}' for site {site_id}.") # Changed print to logging.warning
        return

    logging.info(f"Plotting land use trend for '{metric}' for {description}...") # Changed print to logging.info
    try:
        plt.figure(figsize=(10, 5))
        land_use_analysis_df[metric].plot(marker='o', linestyle='-')
        plt.title(f"'{metric}' Land Use in Watershed for {description}\nOver Time (NLCD Years)")
        plt.xlabel('Year')
        plt.ylabel(f'{metric} (%)')
        plt.grid(True)
        plt.xticks(land_use_analysis_df.index) # Ensure ticks are at NLCD years

        # Create site-specific plot directory and save
        site_plot_dir = os.path.join(plot_base_dir, site_id)
        os.makedirs(site_plot_dir, exist_ok=True)
        plot_filename = f"{site_id}_landuse_{metric.replace('% ', '').lower()}_trend.png"
        plot_path = os.path.join(site_plot_dir, plot_filename)

        plt.savefig(plot_path, dpi=300)
        logging.info(f"Land use trend plot saved as {plot_path}") # Changed print to logging.info
        # plt.show() # Show plot interactively
        plt.close()

    except Exception as e:
        logging.error(f"Error plotting land use trend for {site_id}: {e}") # Changed print to logging.error


def plot_cross_site_correlation(cross_site_df, x_metric, y_metric, x_label, y_label, title, plot_base_dir, filename_suffix):
     """Generates a scatter plot comparing metrics across multiple sites."""
     # Pass plot_base_dir and filename_suffix to construct the output path
     if cross_site_df is None or cross_site_df.empty or x_metric not in cross_site_df.columns or y_metric not in cross_site_df.columns:
          logging.warning(f"Skipping cross-site plot: Missing data ({x_metric} or {y_metric}) or not enough sites.") # Changed print to logging.warning
          return

     logging.info(f"\nGenerating cross-site plot: '{x_metric}' vs '{y_metric}'...") # Changed print to logging.info
     try:
          plt.figure(figsize=(10, 7))
          ax = sns.scatterplot(data=cross_site_df, x=x_metric, y=y_metric, hue='site_id', s=100) # s=size of points

          # Add site labels next to points
          texts = [] # Use a list for adjustText if available
          for i in range(cross_site_df.shape[0]):
               texts.append(ax.text(x=cross_site_df[x_metric].iloc[i],
                       y=cross_site_df[y_metric].iloc[i],
                       s=cross_site_df['site_id'].iloc[i],
                       fontdict=dict(color='black',size=9),
                       bbox=dict(alpha=0.5, fc='white'))) # Add background to text

          # Use adjustText if available to prevent overlapping labels
          try:
              from adjustText import adjust_text
              adjust_text(texts, force_points=(0.2, 0.2), force_text=(0.2, 0.2),
                          expand_points=(1.2, 1.2), expand_text=(1.2, 1.2),
                          arrowprops=dict(arrowstyle='-', color='lightgrey', lw=0.5))
          except ImportError:
              logging.warning("adjustText not installed. Site labels may overlap. Run 'pip install adjustText'.")
              # If adjustText is not installed, the default text placement is used.


          # Add zero lines for reference
          plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
          plt.axvline(0, color='grey', linestyle='--', linewidth=0.8)

          plt.title(title)
          plt.xlabel(x_label)
          plt.ylabel(y_label)
          plt.grid(True, linestyle=':', alpha=0.6)
          ax.get_legend().remove() # Remove the hue legend if using adjustText for labels
          plt.tight_layout() # Adjust layout for plot elements

          # Save the cross-site plot
          # Cross-site plots are not site-specific, save directly in the script's plot directory
          os.makedirs(plot_base_dir, exist_ok=True)
          plot_filename = f"cross_site_{filename_suffix}.png"
          plot_path = os.path.join(plot_base_dir, plot_filename)

          plt.savefig(plot_path, dpi=300)
          logging.info(f"Cross-site plot saved as {plot_path}") # Changed print to logging.info
          # plt.show() # Show plot interactively
          plt.close()

     except Exception as e:
          logging.error(f"Error generating cross-site plot: {e}") # Changed print to logging.error


# --- Main Execution ---

if __name__ == "__main__":
    setup_logging() # Setup logging at the start

    config = load_config(CONFIG_FILE_PATH)

    if not config or "sites_to_process" not in config:
        logging.critical("Failed to load configuration or 'sites_to_process' not found. Exiting.") # Changed print to logging.critical
    else:
        # Filter for enabled sites with essential info
        sites_to_process = [
            site for site in config["sites_to_process"]
            if site.get("enabled", True) and
               all(key in site for key in ["site_id", "latitude", "longitude", "start_date", "end_date"])
        ]

        if not sites_to_process:
            logging.warning("No enabled sites with complete information found in the configuration file.") # Changed print to logging.warning
        else:
            all_site_summary_list = [] # List to build cross-site summary DataFrame

            for site_info in sites_to_process:
                site_id = site_info["site_id"]
                param_cd = site_info.get("param_cd", "00060")
                latitude = site_info["latitude"]
                longitude = site_info["longitude"]
                start_date_str = site_info["start_date"]
                end_date_str = site_info["end_date"]
                description = site_info.get("description", f"Site {site_id}")

                logging.info(f"\n--- Processing Site: {description} ({site_id}) ---") # Changed print to logging.info

                # 1. Get Watershed Boundary
                watershed_boundary_gdf = get_watershed_boundary(site_id)

                # Plot watershed boundary
                if watershed_boundary_gdf is not None:
                     plot_watershed_boundary(watershed_boundary_gdf, {'latitude': latitude, 'longitude': longitude}, site_id, description, PLOT_BASE_DIR)


                # 2. Analyze Land Use within Watershed
                land_use_analysis_df = None # Initialize
                land_use_change_summary = None # Initialize

                # Only attempt land use analysis if watershed boundary was obtained
                if watershed_boundary_gdf is not None:
                    land_use_analysis_df = analyze_land_use_in_watershed(watershed_boundary_gdf, nlcd_folder_path=NLCD_DATA_FOLDER)

                    # Calculate and summarize land use change (e.g., % Developed)
                    if land_use_analysis_df is not None and len(land_use_analysis_df) >= 2 and '% Developed' in land_use_analysis_df.columns:
                         earliest_year_data = land_use_analysis_df.iloc[0]
                         latest_year_data = land_use_analysis_df.iloc[-1]
                         change_developed = latest_year_data.get('% Developed', np.nan) - earliest_year_data.get('% Developed', np.nan) # Handle potential missing columns

                         land_use_change_summary = {
                              'earliest_NLCD_year': earliest_year_data.name,
                              'latest_NLCD_year': latest_year_data.name,
                              'change_%_developed': change_developed
                         }
                         logging.info("\nLand Use Change Summary (Earliest to Latest NLCD):") # Changed print to logging.info
                         logging.info(pd.Series(land_use_change_summary).to_string()) # Changed print to logging.info and used to_string()


                    # Plot land use trend (e.g., % Developed over time)
                    if land_use_analysis_df is not None:
                         plot_land_use_change_trend(land_use_analysis_df, site_id, description, PLOT_BASE_DIR, metric='% Developed')


                # 3. Fetch Streamflow Data
                streamflow_df = fetch_streamflow_data(site_id, param_cd, start_date_str, end_date_str)


                # 4. Analyze Streamflow Trends
                streamflow_trend_results, annual_streamflow_metrics = analyze_streamflow_trends(streamflow_df, site_id, param_cd)


                # 5. Plot Annual Streamflow Metrics
                if annual_streamflow_metrics is not None:
                     plot_annual_streamflow_metrics(annual_streamflow_metrics, streamflow_trend_results, site_id, description, PLOT_BASE_DIR)


                # 6. Store Site Summary for Cross-Site Analysis
                site_summary = {
                     'site_id': site_id,
                     'description': description,
                     'latitude': latitude,
                     'longitude': longitude,
                     'config_start_date': start_date_str, # Store config dates for context
                     'config_end_date': end_date_str
                }
                # Add land use change details
                if land_use_change_summary:
                     site_summary.update(land_use_change_summary)
                # Add streamflow trend details
                if streamflow_trend_results:
                     for metric, trend_info in streamflow_trend_results.items():
                          # Flatten trend results for DataFrame columns
                          site_summary[f"{metric}_slope"] = trend_info.get('slope', np.nan)
                          site_summary[f"{metric}_pvalue"] = trend_info.get('p_value', np.nan)
                          site_summary[f"{metric}_significant"] = trend_info.get('significant', False)
                          site_summary[f"{metric}_trend"] = trend_info.get('trend', 'N/A')


                all_site_summary_list.append(site_summary)


            # --- Cross-Site Analysis and Plotting ---
            # Create DataFrame from all site summaries
            cross_site_df = pd.DataFrame(all_site_summary_list)

            if len(cross_site_df) > 1:
                 logging.info("\n--- Generating Cross-Site Summary and Plots ---") # Changed print to logging.info
                 logging.info("Cross-Site Summary Data:") # Changed print to logging.info
                 logging.info(cross_site_df[['site_id', 'description', 'change_%_developed', 'Annual_Mean_Flow_slope', 'Annual_Q10_Flow_slope', 'Annual_Q90_Flow_slope']].to_string()) # Changed print to logging.info and corrected metric names

                 # Plot Cross-Site Correlation Example: % Developed Change vs. Annual Mean Trend Slope
                 plot_cross_site_correlation(
                     cross_site_df,
                     x_metric='change_%_developed',
                     y_metric='Annual_Mean_Flow_slope', # Corrected metric name
                     x_label=f'% Change in Developed Land Use ({cross_site_df["earliest_NLCD_year"].min()} to {cross_site_df["latest_NLCD_year"].max()} NLCD)' if 'earliest_NLCD_year' in cross_site_df.columns else '% Change in Developed Land Use',
                     y_label='Annual Mean Flow Trend Slope (CFS/Year)',
                     title='Streamflow Trend vs. Developed Land Use Change (Across Sites)',
                     plot_base_dir=PLOT_BASE_DIR, # Pass plot_base_dir
                     filename_suffix='landuse_meanflow_correlation' # Pass suffix
                 )

                 # You can add more cross-site plots here for other metrics

            elif len(cross_site_df) == 1:
                logging.info("\nProcessed one site. Skipping cross-site analysis.") # Changed print to logging.info
            else:
                 logging.warning("\nNo sites were successfully processed for cross-site analysis.") # Changed print to logging.warning