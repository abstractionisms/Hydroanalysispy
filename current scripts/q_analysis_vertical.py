import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import xml.etree.ElementTree as ET # Keep standard XML import
from io import StringIO
import json # For config file
import logging
import os
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.ticker as mticker # Keep for potential formatting
import calendar
import textwrap
import pymannkendall as mk # For temp trend analysis
from itertools import combinations # For p-value pairs
import matplotlib.dates as mdates # For date formatting

# Import Meteostat library
from meteostat import Point, Daily, Stations

# --- Constants ---
DEFAULT_CONFIG_PATH = 'config2.json' # Single config file
# Defaults used if config file is missing values OR for constants
DEFAULT_PARAM_CD = "00060"
DEFAULT_START_DATE = "2000-01-01"
DEFAULT_END_DATE = "today"
DEFAULT_LOG_FILE = "logs/q_analysis_vertical.log" # New log file name with logs/
DEFAULT_INVENTORY_FILE = "nwis_inventory_with_latlon_full_data.txt" # Default if not in config
DISCHARGE_COL = 'Discharge_cfs'
TEMP_COL = 'Temp_C'
PRECIP_COL = 'Precip_mm'
P_SIGNIFICANCE_LEVEL = 0.05 # Alpha level for significance stars
MK_ALPHA = 0.05 # Alpha for Mann-Kendall
PLOT_YEARS = 3 # Number of years for daily time series plot
N_SCATTER_ANNOTATIONS = 10 # Number of points to annotate on monthly scatter
HEXBIN_GRIDSIZE = 30 # Controls the number of hexagons in the hexbin plot
RECENT_DECADE_YEARS = 10 # Number of years for recent decade average

# --- Output Paths ---
PLOT_BASE_DIR = 'plots/q_analysis_vertical' # Base directory for plots

# --- Descriptive Labels for Plots ---
PLOT_LABELS = {
    DISCHARGE_COL: 'Discharge (cfs)',
    TEMP_COL: 'Avg Temp (°C)',
    PRECIP_COL: 'Precipitation (mm)',
    f'{PRECIP_COL}_lag1': 'Previous Day Precip (mm)' # Label for lagged precip
}


# --- Logging Setup ---
def setup_logging(log_file=DEFAULT_LOG_FILE):
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
        print(f"Error setting up file logger for {log_file}: {e}") # Keep print here as logging might not be setup yet

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

# --- Configuration Loading ---
def load_config(config_path=DEFAULT_CONFIG_PATH):
    """Loads configuration from a JSON file."""
    logging.info(f"Attempting to load configuration from: {config_path}") # Changed print to logging.info
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        logging.info("Configuration loaded successfully.") # Changed print to logging.info
        return config_data
    except FileNotFoundError:
        logging.error(f"ERROR: Config file not found at '{config_path}'") # Changed print to logging.error
        return None
    except json.JSONDecodeError as e:
        logging.error(f"ERROR: Invalid JSON in config file '{config_path}': {e}") # Changed print to logging.error
        return None
    except Exception as e:
        logging.error(f"ERROR loading config '{config_path}': {e}") # Changed print to logging.error
        return None

# --- Inventory Loading (Not used by main in this version) ---
_inventory_cache = None
def load_inventory(inventory_path):
    """Loads the master inventory file into a pandas DataFrame, using a simple cache."""
    global _inventory_cache
    if _inventory_cache is not None and isinstance(_inventory_cache, pd.DataFrame):
        logging.info("Using cached inventory data.")
        return _inventory_cache
    if not inventory_path or not os.path.exists(inventory_path):
        logging.error(f"Inventory file not found: '{inventory_path}'")
        return None
    logging.info(f"Loading inventory from: {inventory_path}")
    try:
        # Find header row index and column names, skipping comments
        header_row_index = -1
        column_names = []
        comment_lines_count = 0
        with open(inventory_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line.strip().startswith('#'):
                    comment_lines_count += 1
                    continue
                if header_row_index == -1:
                    header_row_index = i
                    column_names = [col.strip() for col in line.strip().split('\t')]
                    break
        if header_row_index == -1 or 'site_no' not in column_names:
            raise ValueError("Could not detect header row with 'site_no'.")

        # Read the CSV data
        df_inventory = pd.read_csv(
            inventory_path, sep='\t', comment='#',
            header=header_row_index - comment_lines_count,
            names=column_names, low_memory=False, dtype={'site_no': str}
        )
        logging.info(f"Inventory loaded: {len(df_inventory)} sites.")

        # Validate required columns
        required_inventory_cols = ['site_no','station_nm', 'dec_lat_va', 'dec_long_va']
        if not all(col in df_inventory.columns for col in required_inventory_cols):
            missing = [col for col in required_inventory_cols if col not in df_inventory.columns]
            raise ValueError(f"Inventory missing required columns: {missing}")

        # Convert columns to appropriate types
        df_inventory['dec_lat_va'] = pd.to_numeric(df_inventory['dec_lat_va'], errors='coerce')
        df_inventory['dec_long_va'] = pd.to_numeric(df_inventory['dec_long_va'], errors='coerce')
        if 'drain_area_va' in df_inventory.columns:
            df_inventory['drain_area_va'] = pd.to_numeric(df_inventory['drain_area_va'], errors='coerce')

        df_inventory = df_inventory.set_index('site_no', drop=False)
        _inventory_cache = df_inventory
        return df_inventory
    except Exception as e:
        logging.error(f"Error reading inventory file '{inventory_path}': {e}")
        return None

# --- Data Fetching & Parsing ---
def fetch_waterml_data(site_id, param_cd, start_date, end_date):
    """Fetches Discharge WaterML data using requests."""
    logging.info(f"Fetching Discharge: {site_id} ({start_date} to {end_date})...")
    url_wml = "https://waterservices.usgs.gov/nwis/dv"
    params_wml = {
        'format': 'waterml,1.1', 'sites': site_id, 'parameterCd': param_cd,
        'startDT': start_date, 'endDT': end_date
    }
    try:
        response_wml = requests.get(url_wml, params=params_wml, timeout=90)
        response_wml.raise_for_status() # Checks for HTTP errors
        logging.info(f"Discharge fetch OK: {site_id}.")
        return response_wml.text
    except requests.exceptions.Timeout:
        logging.error(f"Discharge fetch timed out: {site_id}.")
    except requests.exceptions.RequestException as e:
        logging.error(f"Discharge fetch failed (RequestException): {site_id}: {e}")
    except Exception as e:
        logging.error(f"Unexpected discharge fetch error: {site_id}: {e}")
    return None

def parse_waterml(waterml_content, site_id):
    """
    Parses Daily Value WaterML string into a pandas DataFrame, keeping provisional data.
    """
    if not waterml_content:
        logging.warning(f"No discharge content to parse: {site_id}.")
        return None
    logging.info(f"Parsing Daily Value discharge (WaterML, keeping provisional): {site_id}...")
    try:
        namespaces = {'ns1': 'http://www.cuahsi.org/waterML/1.1/'}
        xml_io = StringIO(waterml_content)
        tree = ET.parse(xml_io)
        root = tree.getroot()
        value_elements = root.findall('.//ns1:value', namespaces)

        if not value_elements:
            logging.warning(f"No <value> tags in discharge XML: {site_id}.")
            return None

        data = []
        provisional_count = 0
        skipped_invalid_count = 0

        raw_data = [
            (elem.get('dateTime'), elem.text, elem.get('qualifiers', ''))
            for elem in value_elements
        ]

        for ts_str, val_str, qual in raw_data:
            if val_str is not None:
                try:
                    fval = float(val_str)
                    # Ensure value is positive for log scale
                    if fval > 0:
                        timestamp = pd.to_datetime(ts_str, utc=True)
                        data.append({'Timestamp': timestamp, DISCHARGE_COL: fval})
                        if 'P' in qual:
                            provisional_count += 1
                    else:
                         skipped_invalid_count += 1 # Skip non-positive for log scale
                except (ValueError, TypeError):
                    skipped_invalid_count += 1
            else:
                skipped_invalid_count += 1

        if not data:
            logging.warning(f"No valid positive discharge points extracted (DV): {site_id}.")
            return None

        df_q = pd.DataFrame(data).set_index('Timestamp').sort_index()
        logging.info(f"Daily Value Discharge DF created: {site_id} ({len(df_q)} rows).")
        if provisional_count > 0:
            logging.info(f"Included {provisional_count} provisional discharge points: {site_id}.")
        if skipped_invalid_count > 0:
            logging.info(f"Skipped {skipped_invalid_count} points (invalid/non-positive value): {site_id}.")

        return df_q

    except ET.ParseError as e:
        logging.error(f"Discharge XML parse error (DV): {site_id}: {e}")
    except Exception as e:
        logging.error(f"Unexpected discharge parse error (DV): {site_id}: {e}")
    return None


def fetch_climate_data(latitude, longitude, start_datetime, end_datetime):
    """Fetches daily climate data using Meteostat."""
    logging.info(f"Fetching climate near Lat={latitude:.4f}, Lon={longitude:.4f} ({start_datetime.date()} to {end_datetime.date()})...")
    try:
        location = Point(latitude, longitude)
        data = Daily(location, start_datetime, end_datetime)
        data = data.fetch()

        if data.empty:
            logging.warning("Meteostat returned no data for this location/period.")
            return None

        # Select, rename, and ensure UTC index
        climate_df = data[['tavg', 'prcp']].copy()
        climate_df.columns = [TEMP_COL, PRECIP_COL]
        if climate_df.index.tz is None:
             climate_df.index = climate_df.index.tz_localize('UTC')
        else:
             climate_df.index = climate_df.index.tz_convert('UTC')

        # Fill missing values
        climate_df[PRECIP_COL] = climate_df[PRECIP_COL].fillna(0)
        climate_df[TEMP_COL] = climate_df[TEMP_COL].ffill().bfill()

        logging.info(f"Climate data fetched successfully ({len(climate_df)} rows).")
        return climate_df
    except Exception as e:
        logging.error(f"Error fetching/processing climate data: {e}")
        return None

# --- Analysis Functions ---
def analyze_correlation(df_merged):
    """Calculates correlation matrix, p-values, and lagged correlation."""
    logging.info("Performing correlation analysis...")
    results = {'corr_matrix': None, 'p_values': {}, 'lagged_precip_corr': None, 'lagged_precip_p': None}
    if df_merged is None or df_merged.empty:
        logging.warning("No merged data to analyze.")
        return results

    cols_to_analyze = [DISCHARGE_COL, TEMP_COL, PRECIP_COL]
    cols_present = [col for col in cols_to_analyze if col in df_merged.columns]

    if len(cols_present) < 2:
        logging.warning(f"Need >= 2 columns for correlation, found: {cols_present}")
        return results

    df_analysis = df_merged[cols_present].dropna()
    if len(df_analysis) < 3:
        logging.warning(f"Less than 3 valid points for correlation after dropna (found {len(df_analysis)}).")
        return results

    # 1. Correlation Matrix (calculated but not plotted)
    try:
        results['corr_matrix'] = df_analysis.corr()
        logging.info("Correlation Matrix calculated.")
    except Exception as e:
        logging.error(f"Error calculating correlation matrix: {e}")
        # Don't return early, p-values might still be calculable

    # 2. P-values for correlations
    logging.info("Calculating p-values...")
    for col1, col2 in combinations(cols_present, 2):
        pair_key = tuple(sorted((col1, col2)))
        try:
            if df_analysis[col1].nunique() > 1 and df_analysis[col2].nunique() > 1:
                corr_test = stats.pearsonr(df_analysis[col1], df_analysis[col2])
                results['p_values'][pair_key] = corr_test.pvalue
            else:
                logging.warning(f"Skipping p-value calculation for {col1} vs {col2}: insufficient variance.")
                results['p_values'][pair_key] = np.nan
        except Exception as e:
            logging.warning(f"Error calculating p-value for {col1} vs {col2}: {e}")
            results['p_values'][pair_key] = np.nan

    logging.info("P-values calculated.")

    # 3. Lagged Precipitation Correlation
    if PRECIP_COL in cols_present and DISCHARGE_COL in cols_present:
        logging.info("Calculating lagged precipitation correlation...")
        try:
            df_analysis_lag = df_analysis.copy()
            lagged_precip_col = f'{PRECIP_COL}_lag1'
            df_analysis_lag[lagged_precip_col] = df_analysis_lag[PRECIP_COL].shift(1)
            df_analysis_lagged = df_analysis_lag.dropna()

            if len(df_analysis_lagged) >= 3:
                if df_analysis_lagged[DISCHARGE_COL].nunique() > 1 and df_analysis_lagged[lagged_precip_col].nunique() > 1:
                    lag_corr_test = stats.pearsonr(df_analysis_lagged[DISCHARGE_COL], df_analysis_lagged[lagged_precip_col])
                    results['lagged_precip_corr'] = lag_corr_test.statistic
                    results['lagged_precip_p'] = lag_corr_test.pvalue
                    logging.info(f"Lag-1 precip corr calculated (R={results['lagged_precip_corr']:.3f}, p={results['lagged_precip_p']:.3f}).")
                else:
                    logging.warning("Insufficient variance for lagged correlation after dropping NaNs.")
            else:
                logging.warning(f"Insufficient data points ({len(df_analysis_lagged)}) for lagged correlation after shift.")
        except Exception as e:
            logging.error(f"Error calculating lagged correlation: {e}")

    return results


def calculate_annual_means(df, col_name):
    """Calculates annual means for a given column."""
    logging.info(f"Calculating annual means for {col_name}...")
    if df is None or col_name not in df.columns or df.empty:
        logging.warning(f"Invalid DataFrame for annual mean calculation of {col_name}.")
        return None
    try:
        df.index = pd.to_datetime(df.index)
        annual_means = df[col_name].resample('AS').mean().dropna()
        logging.info(f"Calculated annual means for {col_name} ({len(annual_means)} years).")
        return annual_means
    except Exception as e:
        logging.error(f"Error calculating annual means for {col_name}: {e}")
        return None

def perform_trend_analysis(series, series_name="series"):
    """Performs LinReg and MK trend analysis on an annual series."""
    logging.info(f"Performing trend analysis for {series_name}...")
    results = {'linear_regression': None, 'mann_kendall': None}
    if series is None or series.empty or len(series) < 3:
        logging.warning(f"Insufficient data for {series_name} trend ({len(series) if series is not None else 0} pts).")
        return results

    # Linear Regression
    try:
        years = series.index.year.astype(float)
        vals = series.values
        mask = ~np.isnan(years) & ~np.isnan(vals)
        if np.sum(mask) >= 2:
            slope, intercept, r, p, se = stats.linregress(years[mask], vals[mask])
            results['linear_regression'] = {
                'slope': slope, 'intercept': intercept, 'p_value': p,
                'r_squared': r**2, 'std_err': se, 'years': years[mask]
            }
            logging.info(f"  {series_name} LinReg: p={p:.4f}, slope={slope:.2f}")
        else:
             logging.warning(f"Less than 2 valid non-NaN points for {series_name} LinReg.")
    except Exception as e:
        logging.error(f"Error during {series_name} LinReg: {e}")

    # Mann-Kendall Test
    try:
        mk_vals = series.dropna().sort_index().values
        if len(mk_vals) >= 3:
             mk_res = mk.original_test(mk_vals, alpha=MK_ALPHA)
             results['mann_kendall'] = mk_res._asdict()
             logging.info(f"  {series_name} MK: Trend={mk_res.trend}, p={mk_res.p:.4f}, SenSlope={mk_res.slope:.2f}")
        else:
             logging.warning(f"Less than 3 valid non-NaN points for {series_name} MK test.")
    except Exception as e:
        logging.error(f"Error during {series_name} MK test: {e}")

    return results


# --- Plotting Helper Functions ---
def _plot_placeholder(ax, message="Plot N/A"):
    """Helper function to display a placeholder message on an axes object."""
    ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=10,
            transform=ax.transAxes, wrap=True,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.5))
    ax.set_xticks([])
    ax.set_yticks([])

def _get_significance_stars(pval):
    """Helper function to get significance stars based on p-value."""
    if pval is None or pd.isna(pval): return ""
    if pval < 0.001: return "***"
    if pval < 0.01: return "**"
    if pval < P_SIGNIFICANCE_LEVEL: return "*"
    return ""

# --- REMOVED Heatmap Plotting Function ---

# --- Helper for Anomaly Plot ---
def _plot_anomaly(ax, df_merged):
    """
    Plots recent monthly averages against long-term and recent-decade monthly averages.
    Includes Discharge, Temperature, and Precipitation.
    Removes value annotations from lines. Adds Precip lines/shading to ax2.
    """
    plot_title = "Recent Decade vs Historical Monthly Averages" # Base title
    required_cols = [DISCHARGE_COL, TEMP_COL, PRECIP_COL]
    if df_merged is None or df_merged.empty or not all(col in df_merged.columns for col in required_cols):
        _plot_placeholder(ax, f"{plot_title}\nData N/A")
        return

    try:
        df_plot = df_merged[required_cols].copy()
        df_plot.index = pd.to_datetime(df_plot.index)
        df_plot['Month'] = df_plot.index.month
        df_plot['Year'] = df_plot.index.year

        # Get overall historical period
        hist_start_year = df_plot['Year'].min()
        hist_end_year = df_plot['Year'].max()
        plot_title += f" (Hist: {hist_start_year}-{hist_end_year})"

        # Calculate long-term monthly stats (mean and std)
        monthly_stats_overall = df_plot.groupby('Month').agg(
            q_mean_overall=(DISCHARGE_COL, 'mean'),
            q_std_overall=(DISCHARGE_COL, 'std'),
            t_mean_overall=(TEMP_COL, 'mean'),
            t_std_overall=(TEMP_COL, 'std'),
            p_mean_overall=(PRECIP_COL, 'mean'), # Add Precip
            p_std_overall=(PRECIP_COL, 'std')    # Add Precip
        ).reset_index()

        # Calculate recent decade monthly stats (mean only)
        decade_start_year = hist_end_year - RECENT_DECADE_YEARS + 1
        df_decade = df_plot[df_plot['Year'] >= decade_start_year]
        if df_decade.empty:
             logging.warning("No data found for the recent decade calculation.")
             monthly_stats_decade = pd.DataFrame(columns=['Month', 'q_mean_decade', 't_mean_decade', 'p_mean_decade'])
        else:
             monthly_stats_decade = df_decade.groupby('Month').agg(
                 q_mean_decade=(DISCHARGE_COL, 'mean'),
                 t_mean_decade=(TEMP_COL, 'mean'),
                 p_mean_decade=(PRECIP_COL, 'mean') # Add Precip
             ).reset_index()
        decade_label = f"{decade_start_year}-{hist_end_year} Avg"

        # Ensure we have stats for all 12 months for both periods
        all_months = pd.DataFrame({'Month': range(1, 13)})
        monthly_stats_overall = pd.merge(all_months, monthly_stats_overall, on='Month', how='left')
        monthly_stats_decade = pd.merge(all_months, monthly_stats_decade, on='Month', how='left')

        # Combine stats for easier plotting
        monthly_stats = pd.merge(monthly_stats_overall, monthly_stats_decade, on='Month', how='left')
        monthly_stats = monthly_stats.set_index('Month')

        if monthly_stats.isnull().values.any():
             logging.warning(f"Anomaly plot may be incomplete due to insufficient data for stats.")
             # Continue plotting with available data

        # Plotting
        months_num = range(1, 13)
        month_labels = [calendar.month_abbr[i] for i in months_num]

        # Discharge Axis (ax)
        color_q = 'tab:blue'
        ax.set_xlabel("Month")
        ax.set_ylabel(f"Avg {PLOT_LABELS[DISCHARGE_COL]}", color=color_q)
        ln1 = ax.plot(months_num, monthly_stats['q_mean_overall'], color=color_q, linestyle='--', label=f'Overall Avg Q')
        # Ensure std dev doesn't go below zero for fill
        q_std_lower = (monthly_stats['q_mean_overall'] - monthly_stats['q_std_overall']).clip(lower=0)
        q_std_upper = monthly_stats['q_mean_overall'] + monthly_stats['q_std_overall']
        fill1 = ax.fill_between(months_num, q_std_lower, q_std_upper,
                         color=color_q, alpha=0.15, label='_nolegend_')
        ln2 = ax.plot(months_num, monthly_stats['q_mean_decade'], color=color_q, linestyle=':', marker='.', markersize=4, label=f'{decade_label} Q')
        ax.tick_params(axis='y', labelcolor=color_q)
        ax.set_xticks(months_num)
        ax.set_xticklabels(month_labels)
        ax.grid(True, axis='y', linestyle=':', alpha=0.5)
        ax.set_ylim(bottom=0)

        # Temperature & Precipitation Axis (ax2)
        ax2 = ax.twinx()
        color_t = 'tab:red'
        color_p_line = 'black' # Precipitation line color
        color_p_fill = 'tab:green' # Precipitation fill color

        # Plot Temp
        ax2.set_ylabel(f"Avg {PLOT_LABELS[TEMP_COL]} (red) / {PLOT_LABELS[PRECIP_COL]} (black/green)", color='black') # Updated label
        ln4 = ax2.plot(months_num, monthly_stats['t_mean_overall'], color=color_t, linestyle='--', label=f'Overall Avg T')
        # Std dev band for temp can go below zero
        fill2 = ax2.fill_between(months_num,
                          monthly_stats['t_mean_overall'] - monthly_stats['t_std_overall'],
                          monthly_stats['t_mean_overall'] + monthly_stats['t_std_overall'],
                          color=color_t, alpha=0.15, label='_nolegend_')
        ln5 = ax2.plot(months_num, monthly_stats['t_mean_decade'], color=color_t, linestyle=':', marker='.', markersize=4, label=f'{decade_label} T')

        # Plot Precip
        ln7 = ax2.plot(months_num, monthly_stats['p_mean_overall'], color=color_p_line, linestyle='--', label=f'Overall Avg P')
        # Ensure std dev doesn't go below zero for precip fill
        p_std_lower = (monthly_stats['p_mean_overall'] - monthly_stats['p_std_overall']).clip(lower=0)
        p_std_upper = monthly_stats['p_mean_overall'] + monthly_stats['p_std_overall']
        fill3 = ax2.fill_between(months_num, p_std_lower, p_std_upper,
                          color=color_p_fill, alpha=0.15, label='_nolegend_') # Green fill
        ln8 = ax2.plot(months_num, monthly_stats['p_mean_decade'], color=color_p_line, linestyle=':', marker='x', markersize=4, label=f'{decade_label} P') # Different marker

        ax2.tick_params(axis='y', labelcolor='black')
        # Adjust y-limits for ax2 to encompass both temp and precip ranges
        min_limit_t = (monthly_stats['t_mean_overall'] - monthly_stats['t_std_overall']).min()
        max_limit_t = (monthly_stats['t_mean_overall'] + monthly_stats['t_std_overall']).max()
        min_limit_p = 0 # Precip starts at 0
        max_limit_p = (monthly_stats['p_mean_overall'] + monthly_stats['p_std_overall']).max()
        # Set limits based on the wider range, ensuring negative temps are visible
        ax2.set_ylim(bottom=min(0, min_limit_t if pd.notna(min_limit_t) else 0),
                     top=max(max_limit_t if pd.notna(max_limit_t) else 0, max_limit_p if pd.notna(max_limit_p) else 0) * 1.1) # Add some padding


        ax.set_title(plot_title)

        # --- REMOVED value annotations ---

        # Combine handles and labels from both axes for the legend
        handles = ln1 + ln2 + ln4 + ln5 + ln7 + ln8
        labels = [h.get_label() for h in handles]
        # Add manual legend entries for shaded areas
        handles.append(plt.Rectangle((0, 0), 1, 1, fc=color_q, alpha=0.15))
        labels.append('Overall Q ± 1σ')
        handles.append(plt.Rectangle((0, 0), 1, 1, fc=color_t, alpha=0.15))
        labels.append('Overall T ± 1σ')
        handles.append(plt.Rectangle((0, 0), 1, 1, fc=color_p_fill, alpha=0.15)) # Green fill legend entry
        labels.append('Overall P ± 1σ')

        # Place legend inside the plot area
        ax.legend(handles, labels, loc='best', fontsize='xx-small', ncol=2)

    except Exception as e:
        logging.error(f"Anomaly plot error: {e}")
        _plot_placeholder(ax, f"Error plotting\n{plot_title}")


# --- Helper for Hexbin Plots (Optional Log Scale) ---
# Modified to accept plot_base_dir for saving if called standalone (not in this script's flow)
def _plot_hexbin(ax, df_merged, x_col, y_col, analysis_results, use_log_scale=False, add_counts=False):
    """
    Plots a hexbin plot for two columns (using daily data) with correlation info.
    Allows specifying whether to use a logarithmic color scale.
    Optionally adds count text to hexagons.
    """
    y_label = PLOT_LABELS.get(y_col, y_col)
    x_label = PLOT_LABELS.get(x_col, x_col)
    scale_label = "(Log Scale)" if use_log_scale else "(Linear Scale)"
    plot_title = f"Daily {y_label} vs. {x_label} (Density {scale_label})"

    if df_merged is None or df_merged.empty or x_col not in df_merged.columns or y_col not in df_merged.columns:
        _plot_placeholder(ax, f"{plot_title}\nData N/A")
        return

    try:
        df_plot = df_merged[[x_col, y_col]].dropna()

        if not df_plot.empty:
            # Create the hexbin plot
            hexbin_args = {
                'gridsize': HEXBIN_GRIDSIZE,
                'cmap': 'viridis',
                'mincnt': 1, # Ensure hexagons with 1 count are plotted
                'alpha': 0.9 # Slightly less transparent
            }
            if use_log_scale:
                hexbin_args['bins'] = 'log' # Use log scale for color intensity

            hb = ax.hexbin(df_plot[x_col], df_plot[y_col], **hexbin_args)

            ax.set_title(plot_title)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.grid(True, linestyle='--', alpha=0.4) # Lighter grid

            # Add a colorbar to show the density scale
            cb = plt.colorbar(hb, ax=ax)
            cb_label = 'Log(Count in Bin)' if use_log_scale else 'Count in Bin'
            cb.set_label(cb_label)

            # Add count text inside hexagons
            if add_counts:
                counts = hb.get_array()
                centers = hb.get_offsets()
                for count, center in zip(counts, centers):
                    if count > 0: # Only label hexagons with counts
                        # Use white text for better contrast on viridis
                        ax.text(center[0], center[1], f'{int(count)}',
                                ha='center', va='center', color='white', fontsize=5) # Adjusted font size
            # End Add count text

            # Add correlation info text (calculated on DAILY data)
            if analysis_results and analysis_results.get('corr_matrix') is not None:
                corr_matrix = analysis_results['corr_matrix']
                p_values = analysis_results.get('p_values', {})
                if x_col in corr_matrix.columns and y_col in corr_matrix.index:
                    r_val = corr_matrix.loc[y_col, x_col]
                    pair_key = tuple(sorted((x_col, y_col)))
                    p_val = p_values.get(pair_key, np.nan)
                    stars = _get_significance_stars(p_val)
                    corr_info = f"Daily R = {r_val:.2f}{stars}"
                    ax.text(0.02, 0.98, corr_info, transform=ax.transAxes, fontsize=9,
                            ha='left', va='top', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
        else:
            _plot_placeholder(ax, f"{plot_title}\nNo Overlapping Data")

    except Exception as e:
        logging.error(f"Hexbin plot error ({y_col} vs {x_col}): {e}")
        _plot_placeholder(ax, f"Error plotting\n{plot_title}")


def _plot_monthly_lagged_scatter(ax, df_merged, analysis_results):
    """
    Helper function to plot MONTHLY AVERAGE lagged precipitation vs. discharge.
    Includes annotations for the top N highest discharge months.
    """
    lagged_precip_col = f'{PRECIP_COL}_lag1'
    y_label = PLOT_LABELS.get(DISCHARGE_COL, DISCHARGE_COL)
    x_label = PLOT_LABELS.get(lagged_precip_col, lagged_precip_col)
    plot_title = f"Monthly Avg {y_label} vs. Monthly Avg {x_label}"

    if df_merged is None or df_merged.empty:
         _plot_placeholder(ax, f"{plot_title}\nData N/A")
         return

    try:
        df_plot_scatter = df_merged.copy()
        if not all(c in df_plot_scatter.columns for c in [DISCHARGE_COL, PRECIP_COL]):
             _plot_placeholder(ax, f"{plot_title}\nRequired columns missing")
             return

        df_plot_scatter[lagged_precip_col] = df_plot_scatter[PRECIP_COL].shift(1)
        df_plot_scatter = df_plot_scatter.dropna(subset=[DISCHARGE_COL, lagged_precip_col])

        if not df_plot_scatter.empty:
            df_plot_scatter.index = pd.to_datetime(df_plot_scatter.index)
            monthly_avg = df_plot_scatter[[DISCHARGE_COL, lagged_precip_col]].resample('ME').mean()
            monthly_avg = monthly_avg.dropna()
            monthly_avg['Month'] = monthly_avg.index.month

            if not monthly_avg.empty:
                scatter_plot = sns.scatterplot(data=monthly_avg, x=lagged_precip_col, y=DISCHARGE_COL,
                                hue='Month', palette='viridis', s=50, ax=ax, legend='full')

                ax.set_title(plot_title)
                ax.set_xlabel(f"Avg {x_label}")
                ax.set_ylabel(f"Avg {y_label}")
                ax.grid(True, linestyle='--', alpha=0.6)

                handles, labels = ax.get_legend_handles_labels()
                month_names = [calendar.month_abbr[int(label)] for label in labels[1:]]
                ax.legend(handles=handles[1:], labels=month_names, title='Month', fontsize='small', ncol=2)

                # Annotate Top N Points
                top_n_months = monthly_avg.nlargest(N_SCATTER_ANNOTATIONS, DISCHARGE_COL)
                for idx, row in top_n_months.iterrows():
                    ax.annotate(f'{idx.strftime("%Y-%m")}',
                                xy=(row[lagged_precip_col], row[DISCHARGE_COL]), xycoords='data',
                                xytext=(5, -5), textcoords='offset points',
                                ha='left', va='top', fontsize=7, color='black',
                                bbox=dict(boxstyle='round,pad=0.1', fc='yellow', alpha=0.5, ec='none'))

                # Add overall daily lagged correlation info text
                lag_corr = analysis_results.get('lagged_precip_corr')
                lag_p = analysis_results.get('lagged_precip_p')
                if lag_corr is not None and lag_p is not None:
                    lag_sig_str = f"(p={lag_p:.3f})" if lag_p >= 0.001 else "(p<0.001)"
                    lag_info = f"Daily Lag-1 Corr: R={lag_corr:.2f} {lag_sig_str}"
                    ax.text(0.02, 0.98, lag_info, transform=ax.transAxes, fontsize=9,
                            ha='left', va='top', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
            else:
                 _plot_placeholder(ax, f"{plot_title}\nNo Monthly Data")

        else:
            _plot_placeholder(ax, f"{plot_title}\nNo Lagged Data")
    except Exception as e:
        logging.error(f"Monthly lagged scatter plot error: {e}")
        _plot_placeholder(ax, f"Error plotting\n{plot_title}")


def _plot_timeseries(ax, df_q):
    """
    Helper function to plot the recent discharge time series with a LOG Y-AXIS.
    Annotates Max/Min for EACH YEAR shown, plus last point.
    Removed precipitation plotting. Ensures min annotation is offset.
    """
    if df_q is None or df_q.empty:
         _plot_placeholder(ax, "Time Series Plot N/A (Discharge Missing)")
         return

    try:
        end_plot_date = df_q.index.max()
        start_plot_date = end_plot_date - pd.DateOffset(years=PLOT_YEARS)
        # Filter for the plot period AND ensure discharge > 0 for log scale
        df_plot_q = df_q.loc[start_plot_date:end_plot_date].copy()
        df_plot_q = df_plot_q[df_plot_q[DISCHARGE_COL] > 0] # Keep only positive values

        if not df_plot_q.empty:
            color1 = 'tab:blue'
            ax.set_xlabel('Date')
            ax.set_ylabel(f"{PLOT_LABELS.get(DISCHARGE_COL, DISCHARGE_COL)} (Log Scale)", color=color1) # Update label
            line_q, = ax.plot(df_plot_q.index, df_plot_q[DISCHARGE_COL], color=color1, linewidth=1.5, label='Discharge')
            ax.tick_params(axis='y', labelcolor=color1)
            ax.grid(True, linestyle='--', alpha=0.6)
            # ax.set_ylim(bottom=0) # Commented out: Log scale cannot start at 0
            ax.set_yscale('log') # <<< Set y-axis to logarithmic scale >>>

            plot_title = f"Recent Discharge - {PLOT_YEARS} Years (Log Scale)" # Update title
            ax.set_title(plot_title)

            # Add Yearly Min/Max Annotations
            q_series = df_plot_q[DISCHARGE_COL]
            years_in_plot = q_series.index.year.unique()

            for year in years_in_plot:
                q_year = q_series[q_series.index.year == year]
                if not q_year.empty:
                    try:
                        idx_max = q_year.idxmax()
                        val_max = q_year.max()
                        idx_min = q_year.idxmin()
                        val_min = q_year.min()

                        # Annotate Max for the year
                        ax.annotate(f'{year} Max: {val_max:.0f}',
                                    xy=(idx_max, val_max), xycoords='data',
                                    xytext=(0, 10), textcoords='offset points', # Keep original offsets for now
                                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
                                    ha='center', va='bottom', fontsize=7, color=color1,
                                    bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.6, ec='none'))

                        # Annotate Min for the year - Adjust vertical offset
                        ax.annotate(f'{year} Min: {val_min:.0f}',
                                    xy=(idx_min, val_min), xycoords='data',
                                    xytext=(0, 15), textcoords='offset points', # Keep original offsets for now
                                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-.2"),
                                    ha='center', va='bottom', fontsize=7, color=color1,
                                    bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.6, ec='none'))
                    except Exception as ann_err:
                        logging.warning(f"Could not annotate min/max for year {year}: {ann_err}")

            # Add Annotation for the last point overall
            if not q_series.empty:
                idx_last = q_series.index[-1]
                val_last = q_series.iloc[-1]
                ax.annotate(f'Last: {val_last:.0f}\n{idx_last.strftime("%Y-%m-%d")}',
                            xy=(idx_last, val_last), xycoords='data',
                            xytext=(-10, 20), textcoords='offset points', # Keep original offsets for now
                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
                            ha='right', va='bottom', fontsize=8, color=color1,
                            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6, ec='none'))

            # Removed Precip plotting logic entirely

            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

        else:
            _plot_placeholder(ax, "Time Series Plot N/A (No Positive Recent Discharge for Log Scale)")
    except Exception as e:
        logging.error(f"Time series plot error: {e}")
        _plot_placeholder(ax, "Error plotting Log Time Series")


# --- Main Plotting Function (Refactored) ---
# Modified to accept plot_base_dir
def plot_correlation_results(df_q, df_merged, analysis_results, temp_trend_results,
                             site_id, description, start_date_str, end_date_str, plot_base_dir):
    """
    Generates plots visualizing data and correlations.
    MODIFIED: Creates 4 plots: Anomaly Q/T/P, Hexbin Q vs T (Log Scale + Counts),
              Monthly Avg Q vs Lag P, Daily Q Timeseries (LOG SCALE).
    Saves plots to plot_base_dir/site_id/.
    """
    logging.info(f"Generating revised plots for site {site_id}...") # Version bump

    has_discharge = df_q is not None and not df_q.empty
    has_merged = df_merged is not None and not df_merged.empty

    # Ensure there is at least discharge data to attempt plotting something
    if not has_discharge and not has_merged:
        logging.warning(f"No discharge or merged data available for plot: {site_id}.")
        return

    try:
        # Create 4 subplots
        fig, axes = plt.subplots(4, 1, figsize=(10, 26), constrained_layout=True)

        # Determine plot period string
        if has_merged:
            actual_start_m = df_merged.index.min().strftime('%Y-%m-%d')
            actual_end_m = df_merged.index.max().strftime('%Y-%m-%d')
            plot_period_str = f"Overlap: {actual_start_m} to {actual_end_m}"
        elif has_discharge: # Use full discharge period if no overlap but discharge exists
            actual_start_q = df_q.index.min().strftime('%Y-%m-%d')
            actual_end_q = df_q.index.max().strftime('%Y-%m-%d')
            plot_period_str = f"Discharge: {actual_start_q} to {actual_end_q}"
        else:
            plot_period_str = "No Data Available"


        wrapped_title = "\n".join(textwrap.wrap(f"Site {site_id}: {description}", width=60))
        fig.suptitle(f"Discharge & Climate Analysis ({plot_period_str})\n{wrapped_title}", fontsize=14)

        # Call new/revised plotting helpers
        # Plot 1: Monthly Anomaly Plot (Comparing Recent Decade, Overall, with Precip)
        _plot_anomaly(axes[0], df_merged if has_merged else None)
        # Plot 2: Hexbin Q vs T (Use LOG scale for color, add counts)
        _plot_hexbin(axes[1], df_merged if has_merged else None, TEMP_COL, DISCHARGE_COL, analysis_results if has_merged else None, use_log_scale=True, add_counts=True)
        # Plot 3: Monthly Avg Q vs Monthly Avg Lag P (with top N annotations)
        _plot_monthly_lagged_scatter(axes[2], df_merged if has_merged else None, analysis_results if has_merged else None)
        # Plot 4: Daily Q Timeseries (LOG SCALE, with yearly min/max annotations)
        _plot_timeseries(axes[3], df_q if has_discharge else None) # <-- Pass df_q if available


        # Save plot
        site_plot_dir = os.path.join(plot_base_dir, site_id) # Construct path using plot_base_dir
        os.makedirs(site_plot_dir, exist_ok=True)
        plot_filename = f"USGS_{site_id}_climate_analysis.png" # Increment version and name
        plot_path = os.path.join(site_plot_dir, plot_filename) # Save to site_plot_dir
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Revised analysis plot saved: {plot_path}")

    except Exception as e:
        logging.error(f"Error generating revised plot figure: {site_id}: {e}")
        plt.close()

# --- REMOVED Summary Text Generation Function ---


# --- Main Orchestration ---
# Modified to accept plot_base_dir
def process_site(site_config, analysis_params, plot_base_dir):
    """Processes a single site based on its configuration."""
    site_id = site_config.get("site_id")
    param_cd = site_config.get("param_cd", analysis_params.get("param_cd", DEFAULT_PARAM_CD))
    start_date_str = site_config.get("start_date", analysis_params.get("start_date", DEFAULT_START_DATE))
    end_date_str = site_config.get("end_date", analysis_params.get("end_date", DEFAULT_END_DATE))
    description = site_config.get("description", f"Site {site_id}")

    if not site_config.get("enabled", False):
        logging.info(f"Skipping disabled site: {site_id or 'Missing ID'}")
        return None

    # Get and validate Lat/Lon early
    latitude_raw = site_config.get("latitude")
    longitude_raw = site_config.get("longitude")

    # Ensure latitude and longitude are present and valid before proceeding
    if latitude_raw is None or longitude_raw is None:
        logging.warning(f"Skipping {site_id or 'N/A'}: Missing latitude or longitude in config.")
        return {site_id: {'status': 'Error (Config Missing Coords)'}}

    try:
        latitude = float(latitude_raw)
        longitude = float(longitude_raw)
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise ValueError("Lat/Lon out of valid range")
    except (ValueError, TypeError):
        logging.warning(f"Skipping {site_id}: Invalid latitude ('{latitude_raw}') or longitude ('{longitude_raw}') in config.")
        return {site_id: {'status': 'Error (Invalid Coords)'}}

    # Validate other required fields after successful coordinate validation
    if not all([site_id, param_cd, start_date_str, end_date_str]):
         logging.warning(f"Skipping {site_id or 'N/A'}: Missing site_id, param_cd, or dates in config.")
         return {site_id: {'status': 'Error (Config Missing Other Fields)'}}


    try:
        end_date = datetime.now() if end_date_str.lower() == 'today' else datetime.strptime(end_date_str, '%Y-%m-%d')
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        if start_date >= end_date:
            logging.warning(f"Skipping {site_id}: Start date ({start_date_str}) not before end date ({end_date_str}).")
            return {site_id: {'status': 'Error (Invalid Dates)'}}
        start_date_nwis = start_date.strftime('%Y-%m-%d')
        end_date_nwis = end_date.strftime('%Y-%m-%d')
    except ValueError as e:
        logging.error(f"Skipping {site_id}: Invalid date format - {e}.")
        return {site_id: {'status': 'Error (Date Format)'}}

    logging.info(f"--- Processing Site: {site_id} ({description}) ---")

    # Removed creation of site_output_dir here as it's done in plot_correlation_results
    # site_output_dir = os.path.join(output_base_dir, site_id)
    # try:
    #     os.makedirs(site_output_dir, exist_ok=True)
    # except Exception as e:
    #     logging.error(f"Could not create output dir for {site_id}: {e}.")


    site_results = {'status': 'Started'}
    df_q = None
    df_climate = None
    df_merged = None
    analysis_results = None
    temp_trend_results = None

    # Fetch discharge data
    discharge_wml = fetch_waterml_data(site_id, param_cd, start_date_nwis, end_date_nwis)
    if discharge_wml:
        # parse_waterml now filters non-positives
        df_q = parse_waterml(discharge_wml, site_id)

    # Fetch climate (precipitation) data - latitude and longitude are now guaranteed to be defined if we reached this point
    df_climate = fetch_climate_data(latitude, longitude, start_date, end_date)

    # Attempt to merge data if both are available and df_q is not empty after parsing
    if df_q is not None and not df_q.empty and df_climate is not None and not df_climate.empty:
         logging.info(f"Merging discharge and climate data for site {site_id}...")
         df_merged = pd.merge(df_q, df_climate, left_index=True, right_index=True, how='inner')
         logging.info(f"Merged DataFrame for analysis shape: {df_merged.shape}")

         if not df_merged.empty:
             analysis_results = analyze_correlation(df_merged)
             site_results['correlation_results'] = analysis_results

             logging.info(f"Analyzing temperature trend: {site_id}...")
             temp_annual_means = calculate_annual_means(df_merged, TEMP_COL)
             if temp_annual_means is not None:
                 temp_trend_results = perform_trend_analysis(temp_annual_means, "Annual Temp")
                 site_results['temp_trend_results'] = temp_trend_results
             else:
                 logging.warning(f"Could not calculate annual temp means: {site_id}")

             site_results['status'] = 'Processed (Full)'
         else:
             logging.warning(f"Merged DataFrame empty (no overlapping data or non-positive discharge): {site_id}. Analysis skipped.")
             site_results['status'] = 'Processed (No Overlap/Non-positive Q)'
             # df_merged remains None/empty, plots will handle this

    elif df_q is None or df_q.empty:
         logging.error(f"Discharge data missing or empty (or no positive values for log scale) for {site_id}. Analysis skipped.")
         site_results['status'] = 'Error (Discharge Missing/Non-positive)'
         # df_merged remains None

    elif df_climate is None or df_climate.empty:
        logging.warning(f"Climate data fetch failed or empty for {site_id}. Correlation analysis skipped.")
        site_results['status'] = 'Processed (Climate Missing)'
        # df_merged remains None


    # Plotting - Pass all results needed for the 4 plots and plot_base_dir
    # plot_correlation_results handles cases where df_q or df_merged is None/empty
    plot_correlation_results(df_q, df_merged, analysis_results, temp_trend_results,
                             site_id, description, start_date_nwis, end_date_nwis, plot_base_dir)


    logging.info(f"--- Finished Processing Site: {site_id} ---")
    return {site_id: site_results}


def main():
    """Main function: Loads config, sets up logging, processes sites."""
    config = load_config()
    if not config:
        # setup_logging failed, print error
        print("CRITICAL: Failed to load configuration. Exiting.")
        return

    # Use the defined LOG_FILE constant directly for setup
    setup_logging(log_file=DEFAULT_LOG_FILE)
    logging.info("--- Starting Climate Correlation Script (Unified Config Mode) ---")

    # Use the defined PLOT_BASE_DIR constant
    # output_base_dir = config.get('project2_settings', {}).get('output_directory', DEFAULT_OUTPUT_DIR) # Removed
    plot_base_dir = PLOT_BASE_DIR # Use the constant

    try:
        # Create the base plot directory if it doesn't exist
        os.makedirs(plot_base_dir, exist_ok=True)
        logging.info(f"Base plot directory: {plot_base_dir}")
    except Exception as e:
        logging.critical(f"Failed create base plot dir '{plot_base_dir}': {e}. Exiting.")
        return


    analysis_params = config.get("analysis_parameters", {})
    sites_to_process = config.get("sites_to_process", [])

    if not sites_to_process:
        logging.warning("No sites found in 'sites_to_process' list in config file.")
        logging.info("--- Climate Correlation Script Finished (No Sites) ---")
        return

    all_site_results = {}
    for site_config in sites_to_process:
        # Pass plot_base_dir to process_site
        result = process_site(site_config, analysis_params, plot_base_dir)
        if result:
             all_site_results.update(result)

    logging.info("--- Climate Correlation Script Finished ---")
    # Optional: Add code here to save all_site_results to a summary file if needed


# --- Script Entry Point ---
if __name__ == "__main__":
    main()