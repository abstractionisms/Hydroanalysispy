# --- Imports ---
# Core Scientific Libraries
import pandas as pd
import numpy as np

# Geospatial Libraries
import geopandas as gpd
import rasterio
from rasterio.mask import mask # Explicit import for mask
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
# from scipy import stats # Potentially for other stats

# Visualization Libraries
import matplotlib.pyplot as plt
import matplotlib.colors # For colormap generation
import matplotlib.cm # For colormap generation
import seaborn as sns # Used for plotting styles

# System & Utility Libraries
import json # For reading config file
import os # For path handling
import datetime 
import logging 
import warnings # To suppress specific warnings if needed
import traceback
import matplotlib.dates as mdates # For date formatting in plots

# --- Configuration File Path ---
# !!! EDIT THIS PATH to point to your config.json or config2.json !!!
CONFIG_FILE_PATH = 'config4.json' 

# --- NLCD Data Folder Path ---
# !!! EDIT THIS PATH if your NLCD data is elsewhere !!!
NLCD_DATA_FOLDER = './NLCD_Data/' # Default relative path

# --- Output Paths ---
LOG_FILE = 'logs/q_watershed.log' # Log file path
PLOT_BASE_DIR = 'plots/q_watershed_output' # Base directory for plots

# --- Constants ---
WEB_MERCATOR_CRS = "EPSG:3857" # CRS for contextily basemaps
WGS84_CRS = "EPSG:4326" # Standard Lat/Lon CRS

# --- Logging Setup ---
def setup_logging(log_file=LOG_FILE):
    """Configures basic logging to file and console."""
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # Use basicConfig for simpler setup, force=True allows re-running in notebooks
    logging.basicConfig(
        level=logging.INFO, # Set minimum level (e.g., INFO, DEBUG)
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler() # Output to console
        ],
        force=True 
    )
    # Suppress noisy logs from specific libraries if needed
    logging.getLogger('dataretrieval').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('fiona').setLevel(logging.WARNING) # Geopandas dependency


# --- Helper Functions ---

def load_config(file_path):
    """Loads configuration from a JSON file."""
    if not os.path.exists(file_path):
        logging.error(f"Config file not found at specified path: {file_path}")
        return None
    try:
        with open(file_path, 'r') as f:
            config = json.load(f)
        logging.info(f"Config loaded successfully from {file_path}")
        return config
    except FileNotFoundError:
        logging.error(f"Error: Config file not found at {file_path}")
        return None
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from {file_path}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error loading config {file_path}: {e}")
        return None

def get_watershed_boundary(site_id):
    """Gets the upstream watershed boundary for a USGS site ID using pynhd (NLDI)."""
    if NLDI is None:
        logging.warning("Cannot get watershed: pynhd library not available.")
        return None
    if not site_id or not isinstance(site_id, str):
        logging.error(f"Invalid site_id provided for watershed delineation: {site_id}")
        return None
    # Ensure site ID has the correct prefix for NLDI if needed
    # NLDI usually expects just the number part for USGS sites
    nldi_site_id = site_id.split(':')[-1] if ':' in site_id else site_id

    logging.info(f"Querying NLDI for watershed boundary for site ID: {nldi_site_id}...")
    try:
        nldi_client = NLDI()
        # Use try-except for the specific NLDI call
        try:
            watershed_gdf = nldi_client.get_basins(f"USGS-{nldi_site_id}") # NLDI often needs USGS prefix here
        except Exception as nldi_e:
             logging.warning(f"NLDI query failed for USGS-{nldi_site_id}: {nldi_e}. Trying without prefix...")
             try: # Retry without prefix just in case
                 watershed_gdf = nldi_client.get_basins(nldi_site_id)
             except Exception as nldi_e2:
                  logging.error(f"NLDI query failed again for {nldi_site_id}: {nldi_e2}")
                  return None

        if watershed_gdf is None or watershed_gdf.empty:
            logging.warning(f"No watershed found via NLDI for site ID {site_id}.")
            return None

        logging.info(f"Watershed boundary found for site ID {site_id}.")
        # Ensure the result is a GeoDataFrame and has geometry
        if not isinstance(watershed_gdf, gpd.GeoDataFrame) or 'geometry' not in watershed_gdf.columns:
             logging.error(f"NLDI result for {site_id} is not a valid GeoDataFrame.")
             return None
        # Ensure CRS is set, default from NLDI is usually WGS84 (EPSG:4326)
        if watershed_gdf.crs is None:
             logging.warning(f"Watershed for {site_id} missing CRS, assuming {WGS84_CRS}.")
             watershed_gdf.crs = WGS84_CRS # Assume WGS84 if missing

        return watershed_gdf

    except Exception as e:
        logging.error(f"Unexpected error during watershed delineation for site {site_id}: {e}")
        return None

def analyze_land_use_in_watershed(watershed_gdf, nlcd_folder_path):
    """Analyzes NLCD land use percentages within the watershed for available years."""
    # --- This function remains largely the same as your original ---
    # --- Adding more logging and error handling ---
    if watershed_gdf is None or watershed_gdf.empty:
        logging.warning("Skipping land use analysis: No watershed boundary.")
        return None, None # Return None for both df and summary
    if not os.path.exists(nlcd_folder_path) or not os.path.isdir(nlcd_folder_path):
        logging.error(f"NLCD data folder not found or invalid: {nlcd_folder_path}. Update NLCD_DATA_FOLDER path.")
        return None, None

    logging.info(f"Analyzing land use within watershed using data from {nlcd_folder_path}...")
    land_use_results = {}
    
    # Ensure watershed has geometry
    if 'geometry' not in watershed_gdf.columns or watershed_gdf.geometry.iloc[0] is None:
         logging.error("Watershed GeoDataFrame is missing valid geometry.")
         return None, None
         
    watershed_geom = watershed_gdf.geometry.iloc[0] # Use first geometry if multiple

    # NLCD class mapping (adjust as needed for your specific NLCD versions)
    nlcd_class_map = {11: 'Open Water', 21: 'Dev-Open', 22: 'Dev-Low', 23: 'Dev-Med', 24: 'Dev-High', 31: 'Barren', 41: 'Deciduous Forest', 42: 'Evergreen Forest', 43: 'Mixed Forest', 51: 'Dwarf Scrub', 52: 'Shrub/Scrub', 71: 'Grassland', 72: 'Sedge', 73: 'Lichens', 74: 'Moss', 81: 'Pasture/Hay', 82: 'Cultivated Crops', 90: 'Woody Wetlands', 95: 'Emergent Wetlands'}
    developed_classes = [21, 22, 23, 24] 
    forest_classes = [41, 42, 43]
    ag_classes = [81, 82]
    wetland_classes = [90, 95]

    nlcd_years_to_check = [2001, 2004, 2006, 2008, 2011, 2013, 2016, 2019, 2021] # Common years

    processed_years = 0
    for year in nlcd_years_to_check:
        # Find NLCD file for the year (handle potential variations in naming)
        nlcd_file_path = None
        for potential_suffix in [".tif", ".img", ".TIF", ".IMG"]: # Check common extensions
             potential_filename = f"nlcd_{year}_land_cover_l48_20210604{potential_suffix}" # Example NLCD filename structure - ADJUST IF YOURS IS DIFFERENT
             test_path = os.path.join(nlcd_folder_path, potential_filename)
             if os.path.exists(test_path):
                  nlcd_file_path = test_path
                  break # Found the file for this year
        
        if not nlcd_file_path:
             # logging.debug(f"No NLCD file found for year {year} in {nlcd_folder_path}") # Optional debug log
             continue # Skip year if no file found

        logging.info(f" Processing NLCD {year}...")
        try:
            with rasterio.open(nlcd_file_path) as src:
                # Reproject watershed to raster CRS for clipping
                try:
                    watershed_reprojected = watershed_gdf.to_crs(src.crs)
                except Exception as crs_e:
                     logging.error(f"Failed to reproject watershed for NLCD {year}: {crs_e}")
                     continue

                # Clip raster to watershed boundary
                try:
                    out_image, _ = mask(src, watershed_reprojected.geometry, crop=True, filled=False) # Pass list of geometries
                    land_cover_data = out_image[0] # Assuming single band
                except Exception as mask_e:
                     logging.error(f"Failed to mask NLCD {year} raster: {mask_e}")
                     continue

                # Calculate statistics
                nodata = src.nodata
                valid_pixels_mask = (land_cover_data != nodata) if nodata is not None else np.ones(land_cover_data.shape, dtype=bool)
                valid_pixels = land_cover_data[valid_pixels_mask]

                unique_classes, counts = np.unique(valid_pixels, return_counts=True)
                total_valid_pixels = counts.sum()
                if total_valid_pixels == 0:
                     logging.warning(f"No valid NLCD pixels found within watershed boundary for {year}.")
                     continue

                year_results = {'Total Pixels': total_valid_pixels}
                developed_pixels = 0; forest_pixels = 0; ag_pixels = 0; wetland_pixels = 0
                for class_code, count in zip(unique_classes, counts):
                    class_code_int = int(class_code)
                    if class_code_int in developed_classes: developed_pixels += count
                    if class_code_int in forest_classes: forest_pixels += count
                    if class_code_int in ag_classes: ag_pixels += count
                    if class_code_int in wetland_classes: wetland_pixels += count
                    # Store individual class percentages if needed (can make DataFrame large)
                    # class_name = nlcd_class_map.get(class_code_int, f'Class_{class_code_int}')
                    # year_results[class_name] = (count / total_valid_pixels) * 100.0

                # Store summary percentages
                year_results['% Developed'] = (developed_pixels / total_valid_pixels) * 100.0
                year_results['% Forest'] = (forest_pixels / total_valid_pixels) * 100.0
                year_results['% Agriculture'] = (ag_pixels / total_valid_pixels) * 100.0
                year_results['% Wetland'] = (wetland_pixels / total_valid_pixels) * 100.0
                
                land_use_results[year] = year_results
                processed_years += 1

        except rasterio.errors.RasterioIOError as e: logging.error(f"Rasterio I/O error for NLCD {year}: {e}")
        except Exception as e: logging.error(f"General error processing NLCD for year {year}: {e}"); traceback.print_exc()

    if land_use_results:
        land_use_df = pd.DataFrame.from_dict(land_use_results, orient='index')
        land_use_df.index.name = 'Year'; land_use_df = land_use_df.sort_index()
        logging.info("\nLand Use Analysis Summary (Percentages):")
        logging.info(land_use_df[['% Developed', '% Forest', '% Agriculture', '% Wetland']].to_string())
        
        # Calculate change summary
        change_summary = None
        if len(land_use_df) >= 2:
             first_year_data = land_use_df.iloc[0]
             last_year_data = land_use_df.iloc[-1]
             change_summary = {
                  'earliest_NLCD_year': first_year_data.name,
                  'latest_NLCD_year': last_year_data.name,
                  'change_%_developed': last_year_data.get('% Developed', np.nan) - first_year_data.get('% Developed', np.nan),
                  'change_%_forest': last_year_data.get('% Forest', np.nan) - first_year_data.get('% Forest', np.nan),
                  'change_%_agriculture': last_year_data.get('% Agriculture', np.nan) - first_year_data.get('% Agriculture', np.nan),
                  'change_%_wetland': last_year_data.get('% Wetland', np.nan) - first_year_data.get('% Wetland', np.nan),
             }
             logging.info("\nLand Use Change Summary (Earliest to Latest NLCD):")
             logging.info(pd.Series(change_summary).to_string())
             
        return land_use_df, change_summary
    else:
        logging.warning("No land use data successfully processed.")
        return None, None

def fetch_streamflow_data(site_id, param_cd, start_date_str, end_date_str):
    """Fetches daily streamflow data for a site using dataretrieval."""
    # --- This function remains largely the same as your original ---
    # --- Adding more logging and refined column finding ---
    if not nwis: logging.warning("Skipping streamflow: dataretrieval not available."); return None
    logging.info(f"Fetching streamflow data for site {site_id} ({param_cd})...")
    try:
        current_start_date = start_date_str
        current_end_date = end_date_str
        if isinstance(current_start_date, str) and current_start_date.lower() == 'today': current_start_date = datetime.date.today().strftime('%Y-%m-%d')
        if isinstance(current_end_date, str) and current_end_date.lower() == 'today': current_end_date = datetime.date.today().strftime('%Y-%m-%d')

        df, metadata = nwis.get_dv(sites=site_id, parameterCd=param_cd, start=current_start_date, end=current_end_date)

        if df.empty: logging.warning(f"No data found for site {site_id} in range."); return None

        # Find and rename discharge column
        discharge_col_pattern = f'{param_cd}_' # e.g., '00060_'
        q_col = None
        # Prioritize column ending with '_Mean' or '_00003' (mean stat code)
        mean_suffixes = ['_Mean', '_00003']
        for suffix in mean_suffixes:
             potential_col = f"{param_cd}{suffix}"
             if potential_col in df.columns:
                  q_col = potential_col
                  break
        # Fallback: just the parameter code
        if not q_col and param_cd in df.columns:
             q_col = param_cd
        # Fallback: parameter code + any other suffix (avoiding _cd)
        if not q_col:
             for col in df.columns:
                  if col.startswith(discharge_col_pattern) and not col.endswith('_cd'):
                       q_col = col
                       break

        if q_col:
             df = df.rename(columns={q_col: 'discharge_cfs'})
             # Drop associated code column if it exists
             code_col = f'{q_col}_cd'
             if code_col in df.columns: df = df.drop(columns=[code_col])
             # Drop other potential code columns like 'agency_cd', 'site_no' if they exist
             df = df.drop(columns=[col for col in ['agency_cd', 'site_no'] if col in df.columns], errors='ignore')
        else:
             logging.error(f"CRITICAL: Could not identify discharge column for param {param_cd} in site {site_id}. Columns: {df.columns.tolist()}")
             return None

        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
             try: df.index = pd.to_datetime(df.index)
             except Exception as e: logging.error(f"Failed to convert index to datetime for {site_id}: {e}"); return None
        df.index.name = 'datetime' # Standardize index name

        # Convert discharge to numeric, coerce errors, drop NaNs
        df['discharge_cfs'] = pd.to_numeric(df['discharge_cfs'], errors='coerce')
        df.dropna(subset=['discharge_cfs'], inplace=True)

        # Final checks
        if df.empty: logging.warning(f"No valid numeric discharge data remaining for {site_id}."); return None
        if 'discharge_cfs' not in df.columns: logging.error(f"Discharge column 'discharge_cfs' missing after processing for {site_id}."); return None

        logging.info(f"Successfully fetched and processed {len(df)} streamflow records for site {site_id}.")
        return df

    except Exception as e:
        logging.error(f"Error fetching/processing streamflow for site {site_id}: {e}")
        return None


def analyze_streamflow_trends(df, site_id, param_cd):
    """Calculates trends in annual streamflow metrics using Mann-Kendall."""
    # --- This function remains largely the same as your original ---
    # --- Adding more logging ---
    if df is None or df.empty or 'discharge_cfs' not in df.columns:
        logging.warning(f"No valid streamflow data for trend analysis for site {site_id}.")
        return None, None 
    logging.info(f"Analyzing streamflow trends for site {site_id}...")

    # Calculate annual metrics
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        logging.error("Trend analysis requires DatetimeIndex."); return None, None
        
    # Use water year for hydrological analysis (starts Oct 1st previous year)
    # df_wy = df.copy()
    # df_wy['water_year'] = df_wy.index.year.where(df_wy.index.month < 10, df_wy.index.year + 1)
    # annual_groups = df_wy.groupby('water_year')['discharge_cfs']
    
    # Using Calendar Year for simplicity here, switch to water year if preferred
    annual_groups = df['discharge_cfs'].groupby(df.index.year)

    annual_metrics_list = []
    for year, group in annual_groups:
        if not group.empty:
             metrics = {
                 'Year': pd.to_datetime(f'{year}-01-01'), # Use start of year for index
                 'Annual_Mean_Flow': group.mean(),
                 'Annual_Q90_Flow': group.quantile(0.90),
                 'Annual_Q10_Flow': group.quantile(0.10),
                 'Annual_Max_Flow': group.max(),
                 'Annual_Min_Flow': group.min()
                 # Add more metrics if needed (e.g., days above threshold)
             }
             annual_metrics_list.append(metrics)

    if not annual_metrics_list:
         logging.warning(f"No annual metrics could be calculated for {site_id}.")
         return None, None
         
    annual_metrics_df = pd.DataFrame(annual_metrics_list).set_index('Year')
    annual_metrics_df = annual_metrics_df.sort_index()


    trend_summary = {}
    # Analyze trends for the calculated metrics
    for metric_name in annual_metrics_df.columns:
        try:
            annual_series_for_trend = annual_metrics_df[metric_name].dropna() 
            if len(annual_series_for_trend) >= 10: # Mann-Kendall recommended minimum
                 logging.info(f" Calculating trend for {metric_name} ({len(annual_series_for_trend)} years)...")
                 # Use original_mk for standard test, or other variants if needed
                 trend_res = mk.original_mk(annual_series_for_trend)
                 trend_summary[metric_name] = {
                      'trend': trend_res.trend, 'p_value': trend_res.p, 
                      'significant': trend_res.h, 'slope': trend_res.slope
                 }
                 logging.info(f"  {metric_name} Trend: {trend_res.trend}, p={trend_res.p:.3f}, Slope={trend_res.slope:.2f}")
            else:
                 logging.warning(f" Insufficient data ({len(annual_series_for_trend)} years) for {metric_name} trend (min 10 recommended).")
        except Exception as e: logging.error(f"Error analyzing trend for {metric_name}: {e}")

    if trend_summary: logging.info("\nStreamflow Trend Summary:"); logging.info(pd.DataFrame(trend_summary).T.to_string())
    else: logging.warning("No streamflow trends calculated.")
    return trend_summary, annual_metrics_df


# --- Plotting Functions ---

def plot_watershed_boundary(watershed_gdf, site_coords, site_id, description, plot_base_dir):
     """Plots the delineated watershed boundary with appropriate zoom."""
     if watershed_gdf is None or watershed_gdf.empty:
          logging.warning(f"Skipping watershed plot for {site_id}: No watershed data.")
          return
     if not ctx:
          logging.warning("Skipping watershed plot: contextily not available.")
          return

     logging.info(f"Plotting watershed for {description}...")
     try:
         # Ensure watershed CRS is set, assume WGS84 if not
         if watershed_gdf.crs is None: watershed_gdf.crs = WGS84_CRS
         
         # Reproject watershed and site point to Web Mercator for plotting with basemap
         watershed_plot = watershed_gdf.to_crs(WEB_MERCATOR_CRS)
         point_gdf = gpd.GeoDataFrame([{'geometry': gpd.points_from_xy([site_coords['longitude']], [site_coords['latitude']])[0]}], crs=WGS84_CRS)
         point_plot = point_gdf.to_crs(WEB_MERCATOR_CRS)

         fig, ax = plt.subplots(1, 1, figsize=(10, 10))

         # Plot watershed polygon
         watershed_plot.plot(ax=ax, color='blue', edgecolor='black', alpha=0.4, label='Watershed') # Slightly more transparent

         # Plot the site point on top
         point_plot.plot(ax=ax, color='red', marker='o', markersize=60, label='Gauge Site', zorder=10) # Ensure point is visible

         # --- IMPROVED ZOOM ---
         # Calculate bounds of the watershed in the plotting CRS
         minx, miny, maxx, maxy = watershed_plot.total_bounds
         # Add a buffer (e.g., 10% of the larger dimension)
         buffer_x = (maxx - minx) * 0.10
         buffer_y = (maxy - miny) * 0.10
         # Set axis limits BEFORE adding basemap
         ax.set_xlim(minx - buffer_x, maxx + buffer_x)
         ax.set_ylim(miny - buffer_y, maxy + buffer_y)
         # --- END IMPROVED ZOOM ---

         # Add basemap using the calculated extent
         try:
             ctx.add_basemap(ax, crs=watershed_plot.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik) # Or try Stamen.Terrain
         except Exception as base_e:
              logging.warning(f"Failed to add basemap for {site_id}: {base_e}. Plotting without basemap.")

         ax.set_title(f"Watershed Delineated for {description}\nUSGS Site {site_id.split(':')[-1]}", fontsize=12)
         ax.set_xticks([]) # Hide axes ticks/labels for map clarity
         ax.set_yticks([])
         # ax.legend() # Optional legend

         plt.tight_layout()
         site_plot_dir = os.path.join(plot_base_dir, site_id.replace(":", "_")) # Sanitize ID for dir name
         os.makedirs(site_plot_dir, exist_ok=True)
         plot_filename = f"{site_id.replace(':', '_')}_watershed_boundary.png"
         plot_path = os.path.join(site_plot_dir, plot_filename)
         plt.savefig(plot_path, dpi=200) # Slightly lower DPI if needed
         logging.info(f"Watershed plot saved: {plot_path}")
         plt.close(fig)

     except Exception as e:
          logging.error(f"Error plotting watershed for {site_id}: {e}")
          traceback.print_exc()

def plot_annual_streamflow_metrics(annual_metrics_df, streamflow_trend_results, site_id, description, plot_base_dir):
    """Plots key annual streamflow metrics and their trends."""
    # --- This function remains the same - it generates the annual metrics plots ---
    if annual_metrics_df is None or annual_metrics_df.empty:
        logging.warning(f"Skipping annual streamflow plotting: No annual data for site {site_id}.")
        return

    logging.info(f"Plotting annual streamflow metrics for {description}...")
    try:
        metrics_to_plot = {
            'Annual_Mean_Flow': {'ylabel': 'Mean Daily Discharge (CFS)', 'color': 'blue'},
            'Annual_Max_Flow': {'ylabel': 'Max Daily Discharge (CFS)', 'color': 'purple'}, # Added Max
            'Annual_Min_Flow': {'ylabel': 'Min Daily Discharge (CFS)', 'color': 'brown'},  # Added Min
            'Annual_Q90_Flow': {'ylabel': 'Q90 Discharge (CFS)', 'color': 'orange'},
            'Annual_Q10_Flow': {'ylabel': 'Q10 Discharge (CFS)', 'color': 'green'},
        }
        
        num_metrics = len([m for m in metrics_to_plot if m in annual_metrics_df.columns])
        if num_metrics == 0: logging.warning(f"No metrics to plot for {site_id}."); return

        fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 3 * num_metrics), sharex=True, squeeze=False) # Ensure axes is always 2D
        fig.suptitle(f"Annual Streamflow Metrics & Trends: {description} ({site_id.split(':')[-1]})", y=1.01, fontsize=14)
        
        plot_row = 0
        for metric_name, plot_params in metrics_to_plot.items():
            if metric_name in annual_metrics_df.columns:
                ax = axes[plot_row, 0] # Access subplot correctly
                annual_series = annual_metrics_df[metric_name].dropna()
                if not annual_series.empty:
                    ax.plot(annual_series.index, annual_series.values, marker='o', markersize=4, linestyle='-', color=plot_params['color'], label=metric_name.replace('_', ' '))
                    trend_key = metric_name
                    if streamflow_trend_results and trend_key in streamflow_trend_results:
                        trend_info = streamflow_trend_results[trend_key]
                        if trend_info.get('significant', False):
                            first_valid_index = annual_series.first_valid_index()
                            last_valid_index = annual_series.last_valid_index()
                            if first_valid_index is not None and last_valid_index is not None:
                                 first_year = first_valid_index; last_year = last_valid_index
                                 start_value = annual_series.loc[first_year]
                                 num_years = (last_year.year - first_year.year) # Calculate based on actual years
                                 end_value = start_value + trend_info['slope'] * num_years
                                 ax.plot([first_year, last_year], [start_value, end_value], linestyle='--', color='red', label=f"MK Trend (p={trend_info['p_value']:.2f})")
                
                ax.set_ylabel(plot_params['ylabel'], fontsize=9)
                # ax.set_title(metric_name.replace('_', ' '), fontsize=10) # Title per subplot can be noisy
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.legend(fontsize=8, loc='best')
                ax.tick_params(axis='y', labelsize=8)
                plot_row += 1 # Increment row index

        # Set common xlabel
        axes[-1, 0].set_xlabel('Year', fontsize=10)
        axes[-1, 0].tick_params(axis='x', labelsize=8) # Ensure bottom x-labels are visible
        # Improve date ticks on bottom axis
        axes[-1, 0].xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=8)) 
        axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout
        site_plot_dir = os.path.join(plot_base_dir, site_id.replace(":", "_")) 
        os.makedirs(site_plot_dir, exist_ok=True)
        plot_filename = f"{site_id.replace(':', '_')}_annual_metrics_plots.png"
        plot_path = os.path.join(site_plot_dir, plot_filename)
        plt.savefig(plot_path, dpi=150)
        logging.info(f"Annual metrics plot saved: {plot_path}")
        plt.close(fig)

    except Exception as e:
        logging.error(f"Error plotting annual streamflow metrics for {site_id}: {e}")
        traceback.print_exc()


def plot_land_use_change_trend(land_use_analysis_df, site_id, description, plot_base_dir, metric='% Developed'):
    """Plots the trend of a specific land use metric over the years."""
    # --- This function remains largely the same as your original ---
    if land_use_analysis_df is None or land_use_analysis_df.empty or metric not in land_use_analysis_df.columns:
        logging.warning(f"Skipping land use trend plot: No data for '{metric}' for site {site_id}.")
        return
    logging.info(f"Plotting land use trend for '{metric}' for {description}...")
    try:
        plt.figure(figsize=(8, 4)) # Smaller figure
        metric_series = land_use_analysis_df[metric].dropna()
        if metric_series.empty: logging.warning(f"No valid data points for metric '{metric}' to plot."); plt.close(); return
        
        plt.plot(metric_series.index, metric_series.values, marker='o', linestyle='-') # Use actual years as x-axis
        plt.title(f"Change in '{metric}' Land Use: {description}\n(NLCD Years Available)", fontsize=11)
        plt.xlabel('NLCD Year', fontsize=9); plt.ylabel(f'{metric} (%)', fontsize=9)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.xticks(metric_series.index, rotation=45, ha="right", fontsize=8) # Show NLCD years clearly
        plt.yticks(fontsize=8)
        plt.tight_layout()
        
        site_plot_dir = os.path.join(plot_base_dir, site_id.replace(":", "_"))
        os.makedirs(site_plot_dir, exist_ok=True)
        safe_metric_name = metric.replace('%', 'percent').replace(' ', '_').lower()
        plot_filename = f"{site_id.replace(':', '_')}_landuse_{safe_metric_name}_trend.png"
        plot_path = os.path.join(site_plot_dir, plot_filename)
        plt.savefig(plot_path, dpi=150); logging.info(f"Land use trend plot saved: {plot_path}"); plt.close()
    except Exception as e: logging.error(f"Error plotting land use trend for {site_id}: {e}")


def plot_cross_site_correlation(cross_site_df, x_metric, y_metric, x_label, y_label, title, plot_base_dir, filename_suffix):
     """Generates a scatter plot comparing metrics across multiple sites."""
     # --- This function remains largely the same as your original ---
     # --- Added check for sufficient data points ---
     if cross_site_df is None or cross_site_df.empty or x_metric not in cross_site_df.columns or y_metric not in cross_site_df.columns:
          logging.warning(f"Skipping cross-site plot: Missing data columns ({x_metric} or {y_metric}).")
          return
     
     # Drop rows where either metric is NaN for correlation calculation/plotting
     plot_df = cross_site_df[[x_metric, y_metric, 'site_id']].dropna()
     if len(plot_df) < 2:
          logging.warning(f"Skipping cross-site plot: Not enough sites ({len(plot_df)}) with valid data for both '{x_metric}' and '{y_metric}'.")
          return

     logging.info(f"Generating cross-site plot: '{x_metric}' vs '{y_metric}'...")
     try:
          plt.figure(figsize=(8, 6)) # Slightly smaller
          ax = sns.scatterplot(data=plot_df, x=x_metric, y=y_metric, hue='site_id', s=80, legend=False) # legend=False as we add text

          # Add site labels (using only site number for brevity)
          texts = [] 
          for i in range(plot_df.shape[0]):
               site_num = plot_df['site_id'].iloc[i].split(':')[-1] # Extract number part
               texts.append(ax.text(x=plot_df[x_metric].iloc[i], y=plot_df[y_metric].iloc[i], s=site_num, fontdict=dict(color='black',size=8)))

          try:
              from adjustText import adjust_text
              adjust_text(texts, arrowprops=dict(arrowstyle='-', color='lightgrey', lw=0.5))
          except ImportError: logging.warning("adjustText not installed. Site labels may overlap.")

          plt.axhline(0, color='grey', linestyle='--', linewidth=0.8); plt.axvline(0, color='grey', linestyle='--', linewidth=0.8)
          plt.title(title, fontsize=12); plt.xlabel(x_label, fontsize=10); plt.ylabel(y_label, fontsize=10)
          plt.grid(True, linestyle=':', alpha=0.6); plt.tick_params(axis='both', which='major', labelsize=9); plt.tight_layout() 

          os.makedirs(plot_base_dir, exist_ok=True)
          plot_filename = f"cross_site_{filename_suffix}.png"
          plot_path = os.path.join(plot_base_dir, plot_filename)
          plt.savefig(plot_path, dpi=150); logging.info(f"Cross-site plot saved: {plot_path}"); plt.close()
     except Exception as e: logging.error(f"Error generating cross-site plot: {e}")

# --- NEW FUNCTION: Plot Combined Watersheds ---
def plot_combined_watersheds(watershed_list, site_info_list, plot_base_dir, config_filename):
    """Plots all watershed boundaries on a single map with different colors."""
    if not watershed_list:
        logging.warning("No watershed boundaries provided for combined plot.")
        return
    if not ctx:
         logging.warning("Skipping combined watershed plot: contextily not available.")
         return

    logging.info(f"\nGenerating combined watershed map for {len(watershed_list)} sites...")

    try:
        # Ensure all GeoDataFrames have the same CRS (WGS84 is standard) before combining
        processed_gdfs = []
        site_ids_for_plot = []
        for i, gdf in enumerate(watershed_list):
            if gdf is None or gdf.empty or 'geometry' not in gdf.columns: continue
            
            # Get site ID from corresponding site_info_list entry
            site_id = site_info_list[i].get('site_id', f'Unknown_{i}')
            site_id_num = site_id.split(':')[-1] # Use just the number for labeling

            # Reproject to WGS84 if necessary
            if gdf.crs is None: gdf.crs = WGS84_CRS # Assume WGS84 if missing
            if gdf.crs != WGS84_CRS:
                try: gdf = gdf.to_crs(WGS84_CRS)
                except Exception as crs_e: logging.error(f"Failed to reproject GDF for {site_id} to {WGS84_CRS}: {crs_e}"); continue
            
            gdf['site_label'] = site_id_num # Add site label column
            processed_gdfs.append(gdf)
            site_ids_for_plot.append(site_id_num) # Keep track of IDs included

        if not processed_gdfs:
             logging.warning("No valid watershed GeoDataFrames to combine.")
             return

        # Combine into a single GeoDataFrame
        combined_gdf = pd.concat(processed_gdfs, ignore_index=True)
        combined_gdf = gpd.GeoDataFrame(combined_gdf, geometry='geometry', crs=WGS84_CRS) # Ensure it's a GeoDataFrame with correct CRS

        # Reproject combined GDF to Web Mercator for plotting
        combined_plot_gdf = combined_gdf.to_crs(WEB_MERCATOR_CRS)

        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        # Determine appropriate colormap based on number of sites
        num_colors = len(site_ids_for_plot)
        cmap = plt.get_cmap('tab10' if num_colors <= 10 else 'tab20' if num_colors <= 20 else 'viridis')
        
        # Plot combined watersheds with colors based on site_label
        combined_plot_gdf.plot(
            column='site_label',
            categorical=True, # Treat site labels as categories for color mapping
            cmap=cmap,
            edgecolor='black',
            linewidth=0.5,
            alpha=0.6, # Transparency
            legend=True,
            legend_kwds={'title': "Site ID", 'loc': 'upper left', 'bbox_to_anchor': (1.02, 1), 'fontsize': 8, 'title_fontsize': 10},
            ax=ax
        )

        # Calculate bounds and add buffer for basemap
        minx, miny, maxx, maxy = combined_plot_gdf.total_bounds
        buffer_x = (maxx - minx) * 0.05 # Smaller buffer for combined map
        buffer_y = (maxy - miny) * 0.05
        ax.set_xlim(minx - buffer_x, maxx + buffer_x)
        ax.set_ylim(miny - buffer_y, maxy + buffer_y)

        # Add basemap
        try:
            ctx.add_basemap(ax, crs=combined_plot_gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron) # Use a lighter basemap
        except Exception as base_e:
             logging.warning(f"Failed to add basemap for combined plot: {base_e}.")

        ax.set_title(f"Combined Watershed Boundaries\nConfig: {config_filename}", fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Adjust layout to fit legend
        plt.subplots_adjust(right=0.8) # Make space on the right for legend

        # Save the combined map
        plot_filename = f"{os.path.splitext(config_filename)[0]}_combined_watersheds.png"
        plot_path = os.path.join(plot_base_dir, plot_filename)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        logging.info(f"Combined watershed plot saved: {plot_path}")
        plt.close(fig)

    except Exception as e:
        logging.error(f"Error generating combined watershed plot: {e}")
        traceback.print_exc()


# --- Main Execution ---
if __name__ == "__main__":
    setup_logging() # Setup logging at the start

    # --- Configuration ---
    # Use the hardcoded path from the top of the script
    config_file_abs_path = os.path.abspath(CONFIG_FILE_PATH)
    config = load_config(config_file_abs_path)
    
    # Use a default output directory if PLOT_BASE_DIR is not set or doesn't exist
    output_dir = PLOT_BASE_DIR if PLOT_BASE_DIR else "plots/q_watershed_output"
    config_name_part = os.path.splitext(os.path.basename(config_file_abs_path))[0]
    # Create a subdirectory based on the config file name
    output_dir_config_specific = os.path.join(output_dir, config_name_part)
    os.makedirs(output_dir_config_specific, exist_ok=True)
    logging.info(f"Output plots will be saved in: {output_dir_config_specific}")

    if not config or "sites_to_process" not in config:
        logging.critical("Failed to load config or 'sites_to_process' missing. Exiting.")
    else:
        sites_to_process = [
            site for site in config["sites_to_process"]
            if site.get("enabled", True) and all(k in site for k in ["site_id", "latitude", "longitude", "start_date", "end_date"])
        ]

        if not sites_to_process:
            logging.warning("No enabled sites with complete info found in config.")
        else:
            all_site_summary_list = [] 
            watershed_gdfs_to_combine = [] # List to store individual GDFs for combined plot
            site_info_for_combined_plot = [] # Store corresponding site info

            for site_info in sites_to_process:
                site_id_raw = site_info["site_id"]
                # Ensure USGS prefix for dataretrieval/consistency
                site_id = f"USGS:{site_id_raw}" if not str(site_id_raw).startswith("USGS:") else str(site_id_raw)
                
                param_cd = site_info.get("param_cd", "00060")
                latitude = site_info["latitude"]
                longitude = site_info["longitude"]
                start_date_str = site_info["start_date"]
                end_date_str = site_info["end_date"]
                description = site_info.get("description", f"Site {site_id}")

                logging.info(f"\n--- Processing Site: {description} ({site_id}) ---")

                # 1. Get Watershed Boundary
                watershed_boundary_gdf = get_watershed_boundary(site_id)
                # Store for combined plot if successful
                if watershed_boundary_gdf is not None and not watershed_boundary_gdf.empty:
                     watershed_gdfs_to_combine.append(watershed_boundary_gdf)
                     site_info_for_combined_plot.append(site_info) # Keep site info aligned

                # Plot INDIVIDUAL watershed boundary (with improved zoom)
                if watershed_boundary_gdf is not None:
                     plot_watershed_boundary(watershed_boundary_gdf, {'latitude': latitude, 'longitude': longitude}, site_id, description, output_dir_config_specific)

                # 2. Analyze Land Use (Optional - depends on NLCD data availability)
                land_use_analysis_df, land_use_change_summary = None, None
                if watershed_boundary_gdf is not None:
                    land_use_analysis_df, land_use_change_summary = analyze_land_use_in_watershed(watershed_boundary_gdf, nlcd_folder_path=NLCD_DATA_FOLDER)
                    if land_use_analysis_df is not None:
                         plot_land_use_change_trend(land_use_analysis_df, site_id, description, output_dir_config_specific, metric='% Developed')
                         # plot_land_use_change_trend(land_use_analysis_df, site_id, description, output_dir_config_specific, metric='% Forest') # Example for another metric

                # 3. Fetch Streamflow Data
                streamflow_df = fetch_streamflow_data(site_id, param_cd, start_date_str, end_date_str)

                # 4. Analyze Streamflow Trends
                streamflow_trend_results, annual_streamflow_metrics = analyze_streamflow_trends(streamflow_df, site_id, param_cd)

                # 5. Plot Annual Streamflow Metrics (KEEP THIS)
                if annual_streamflow_metrics is not None:
                     plot_annual_streamflow_metrics(annual_streamflow_metrics, streamflow_trend_results, site_id, description, output_dir_config_specific)

                # 6. Store Site Summary
                site_summary = {'site_id': site_id, 'description': description, 'latitude': latitude, 'longitude': longitude, 'config_start_date': start_date_str, 'config_end_date': end_date_str}
                if land_use_change_summary: site_summary.update(land_use_change_summary)
                if streamflow_trend_results:
                     for metric, trend_info in streamflow_trend_results.items():
                          site_summary[f"{metric}_slope"] = trend_info.get('slope', np.nan); site_summary[f"{metric}_pvalue"] = trend_info.get('p_value', np.nan); site_summary[f"{metric}_significant"] = trend_info.get('significant', False); site_summary[f"{metric}_trend"] = trend_info.get('trend', 'N/A')
                all_site_summary_list.append(site_summary)

            # --- Generate Combined Watershed Plot (AFTER loop) ---
            if len(watershed_gdfs_to_combine) > 1:
                 plot_combined_watersheds(watershed_gdfs_to_combine, site_info_for_combined_plot, output_dir_config_specific, os.path.basename(config_file_abs_path))
            elif len(watershed_gdfs_to_combine) == 1:
                 logging.info("Only one watershed processed, skipping combined map.")
            else:
                 logging.info("No watershed boundaries successfully processed, skipping combined map.")

            # --- Generate Cross-Site Analysis Plots (AFTER loop) ---
            cross_site_df = pd.DataFrame(all_site_summary_list)
            if len(cross_site_df) > 1:
                 logging.info("\n--- Generating Cross-Site Summary and Plots ---")
                 logging.info("Cross-Site Summary Data (Selected Columns):")
                 cols_to_show = ['site_id', 'description', 'change_%_developed', 'Annual_Mean_Flow_slope', 'Annual_Q10_Flow_slope', 'Annual_Q90_Flow_slope']
                 # Only show columns that actually exist in the dataframe
                 cols_to_show_existing = [col for col in cols_to_show if col in cross_site_df.columns]
                 logging.info(cross_site_df[cols_to_show_existing].to_string())

                 # Plot Cross-Site Correlation Example: % Developed Change vs. Annual Mean Trend Slope
                 if 'change_%_developed' in cross_site_df.columns and 'Annual_Mean_Flow_slope' in cross_site_df.columns:
                      x_label_detail = f"({cross_site_df['earliest_NLCD_year'].min()}-{cross_site_df['latest_NLCD_year'].max()})" if 'earliest_NLCD_year' in cross_site_df.columns else ""
                      plot_cross_site_correlation(
                          cross_site_df, x_metric='change_%_developed', y_metric='Annual_Mean_Flow_slope', 
                          x_label=f'% Change in Developed Land Use {x_label_detail}', y_label='Annual Mean Flow Trend Slope (CFS/Year)', 
                          title='Streamflow Trend vs. Developed Land Use Change',
                          plot_base_dir=output_dir_config_specific, filename_suffix='landuse_meanflow_correlation'
                      )
                 else:
                      logging.warning("Skipping Land Use vs Mean Flow plot due to missing data columns.")

            elif len(cross_site_df) == 1: logging.info("\nProcessed one site. Skipping cross-site analysis.")
            else: logging.warning("\nNo sites successfully processed for cross-site analysis.")