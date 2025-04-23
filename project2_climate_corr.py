import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import xml.etree.ElementTree as ET
from io import StringIO
import json # For config file
import logging
import os
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.ticker as mticker
import calendar
import textwrap
import pymannkendall as mk # For temp trend analysis

# Import Meteostat library
from meteostat import Point, Daily, Stations

# --- Constants ---
DEFAULT_CONFIG_PATH = 'config.json' # Single config file
# Defaults used if config file is missing values OR for constants
DEFAULT_PARAM_CD = "00060"
DEFAULT_START_DATE = "2000-01-01"
DEFAULT_END_DATE = "today"
DEFAULT_OUTPUT_DIR = "climate_corr_results"
DEFAULT_LOG_FILE = "climate_corr.log"
DEFAULT_INVENTORY_FILE = "nwis_inventory_with_latlon.txt" # Default if not in config
DISCHARGE_COL = 'Discharge_cfs'
TEMP_COL = 'Temp_C'
PRECIP_COL = 'Precip_mm'
P_SIGNIFICANCE_LEVEL = 0.05 # Alpha level for significance stars
MK_ALPHA = 0.05 # Alpha for Mann-Kendall

# --- Logging Setup ---
def setup_logging(log_file=DEFAULT_LOG_FILE): # Use constant for default
    """Configures basic logging to file and console."""
    root_logger = logging.getLogger()
    # Prevent adding duplicate handlers if script is run multiple times in same session
    if root_logger.hasHandlers() and len(root_logger.handlers) >= 2:
         # If handlers exist, assume logging is already configured by previous run/import
         # Or optionally, remove existing handlers:
         for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close() # Close handler before removing
         # return # Keep commented out to ensure fresh setup if needed

    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    # --- CHANGE HERE: Set level to DEBUG ---
    root_logger.setLevel(logging.DEBUG) # Set minimum level to DEBUG
    # --------------------------------------

    # File Handler
    try:
        # Ensure directory for log file exists if it includes a path
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        file_handler = logging.FileHandler(log_file, mode='w') # Overwrite log each time
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        print(f"Error setting up file logger for {log_file}: {e}") # Print error if logger fails early

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    # Optional: Set console handler level separately if you only want INFO on console
    # console_handler.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)



# --- Configuration Loading ---
def load_config(config_path=DEFAULT_CONFIG_PATH):
    """Loads configuration from a JSON file."""
    print(f"Attempting to load configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f: config_data = json.load(f); print("Configuration loaded successfully."); return config_data
    except Exception as e: print(f"ERROR loading config '{config_path}': {e}"); return None

# --- Inventory Loading (Needed if config structure changes back) ---
# This function is NOT used by main() in this version, but kept for reference
_inventory_cache = None
def load_inventory(inventory_path):
    """Loads the master inventory file into a pandas DataFrame, using a simple cache."""
    global _inventory_cache;
    if _inventory_cache is not None and isinstance(_inventory_cache, pd.DataFrame): logging.info("Using cached inventory data."); return _inventory_cache
    if not inventory_path or not os.path.exists(inventory_path): logging.error(f"Inventory file not found: '{inventory_path}'"); return None;
    logging.info(f"Loading inventory from: {inventory_path}");
    try: # Start of try block for inventory loading
        header_row_index = -1; column_names = []; comment_lines_count = 0
        with open(inventory_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line.strip().startswith('#'): comment_lines_count += 1; continue
                if header_row_index == -1: header_row_index = i; column_names = [col.strip() for col in line.strip().split('\t')]; break
        if header_row_index == -1 or 'site_no' not in column_names: raise ValueError("Could not detect header row.")
        df_inventory = pd.read_csv(inventory_path, sep='\t', comment='#', header=header_row_index - comment_lines_count, names=column_names, low_memory=False, dtype={'site_no': str})

        # --- Lines 90-92 (Approximate based on file content) ---
        # These lines are correctly placed *inside* the main try block for this function.
        # They perform necessary checks after the initial CSV read attempt.
        logging.info(f"Inventory loaded: {len(df_inventory)} sites."); # Line ~90
        required_inventory_cols = ['site_no','station_nm', 'dec_lat_va', 'dec_long_va'] # Line ~91
        if not all(col in df_inventory.columns for col in required_inventory_cols): raise ValueError(f"Inventory missing columns: {required_inventory_cols}") # Line ~92 (Raises error if check fails, caught by the except block below)
        # --- End Section 90-92 ---

        df_inventory['dec_lat_va'] = pd.to_numeric(df_inventory['dec_lat_va'], errors='coerce')
        df_inventory['dec_long_va'] = pd.to_numeric(df_inventory['dec_long_va'], errors='coerce')
        if 'drain_area_va' in df_inventory.columns: df_inventory['drain_area_va'] = pd.to_numeric(df_inventory['drain_area_va'], errors='coerce')
        df_inventory = df_inventory.set_index('site_no', drop=False); _inventory_cache = df_inventory; return df_inventory
    except Exception as e: # This except block correctly handles errors from the try block above (including lines 90-92)
        logging.error(f"Error reading inventory file '{inventory_path}': {e}");
        return None

# --- Data Fetching & Parsing ---
# CORRECTED fetch_waterml_data with proper indentation
def fetch_waterml_data(site_id, param_cd, start_date, end_date):
    """Fetches Discharge WaterML data using requests."""
    logging.info(f"Fetching Discharge: {site_id} ({start_date} to {end_date})...")
    url_wml = "https://waterservices.usgs.gov/nwis/dv"
    params_wml = {
        'format': 'waterml,1.1',
        'sites': site_id,
        'parameterCd': param_cd,
        'startDT': start_date,
        'endDT': end_date
    }
    try:
        response_wml = requests.get(url_wml, params=params_wml, timeout=90)
        response_wml.raise_for_status() # Checks for HTTP errors (4xx or 5xx)
        logging.info(f"Discharge fetch OK: {site_id}.")
        return response_wml.text
    except requests.exceptions.Timeout:
        logging.error(f"Discharge fetch timed out: {site_id}.")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Discharge fetch failed (RequestException): {site_id}: {e}")
        return None
    except Exception as e: # Catch any other unexpected errors during fetch
        logging.error(f"Unexpected discharge fetch error: {site_id}: {e}")
        return None

# MODIFIED parse_waterml to handle DAILY VALUE WaterML (KEEPING provisional data)
def parse_waterml(waterml_content, site_id):
    """
    Parses Daily Value WaterML string into a pandas DataFrame,
    KEEPING provisional data.
    """
    if not waterml_content: logging.warning(f"No discharge content to parse: {site_id}."); return None
    logging.info(f"Parsing Daily Value discharge (WaterML, keeping provisional): {site_id}...");
    try:
        namespaces = {'ns1': 'http://www.cuahsi.org/waterML/1.1/'}
        xml_io = StringIO(waterml_content)
        tree = ET.parse(xml_io); root = tree.getroot(); value_elements = root.findall('.//ns1:value', namespaces)
        if not value_elements: logging.warning(f"No <value> tags in discharge XML: {site_id}."); return None
        data = []; skipped_invalid_count = 0
        provisional_count = 0
        for elem in value_elements:
            ts = elem.get('dateTime'); val = elem.text; qual = elem.get('qualifiers', '')
            # *** MODIFIED: Keep data even if 'P' is in qualifiers ***
            if val is not None: # Only skip if value itself is missing
                try:
                    fval = float(val)
                    timestamp = pd.to_datetime(ts, utc=True)
                    data.append({'Timestamp': timestamp, DISCHARGE_COL: fval})
                    if 'P' in qual:
                        provisional_count += 1
                except (ValueError, TypeError):
                    skipped_invalid_count += 1 # Count points skipped due to non-numeric value
            else:
                skipped_invalid_count += 1 # Count points skipped due to missing value

        if not data: logging.warning(f"No valid discharge points extracted (DV): {site_id}."); return None

        df_q = pd.DataFrame(data).set_index('Timestamp').sort_index()
        logging.info(f"Daily Value Discharge DF created: {site_id} ({len(df_q)} rows).")
        if provisional_count > 0: logging.info(f"Included {provisional_count} provisional discharge points: {site_id}.")
        if skipped_invalid_count > 0: logging.info(f"Skipped {skipped_invalid_count} points (invalid value): {site_id}.")

        # No resampling needed for DV data
        return df_q

    except ET.ParseError as e: logging.error(f"Discharge XML parse error (DV): {site_id}: {e}"); return None # Line ~139
    except Exception as e: logging.error(f"Unexpected discharge parse error (DV): {site_id}: {e}"); return None # Line ~140


# CORRECTED fetch_climate_data with proper indentation
# --- Lines 144-149 (Approximate based on file content) ---
# This section defines the function and starts the main try block for fetching climate data.
# The try block begins correctly on line ~147.
def fetch_climate_data(latitude, longitude, start_datetime, end_datetime): # Line ~144
    """Fetches daily climate data using Meteostat.""" # Line ~145
    logging.info(f"Fetching climate near Lat={latitude:.4f}, Lon={longitude:.4f} ({start_datetime.date()} to {end_datetime.date()})...") # Line ~146
    try: # Line ~147 (Start of try block)
        location = Point(latitude, longitude) # Line ~148
        data = Daily(location, start_datetime, end_datetime) # Line ~149
        # --- End Section 144-149 ---
        data = data.fetch() # Attempt to fetch data
        if data.empty:
            logging.warning("Meteostat returned no data for this location/period.")
            return None # Return None if fetch is empty
        # Proceed if data is not empty
        climate_df = data[['tavg', 'prcp']].copy()
        climate_df.columns = [TEMP_COL, PRECIP_COL]
        climate_df.index = climate_df.index.tz_localize('UTC') # Ensure UTC index
        climate_df[PRECIP_COL] = climate_df[PRECIP_COL].fillna(0)
        climate_df[TEMP_COL] = climate_df[TEMP_COL].ffill().bfill()
        logging.info(f"Climate data fetched successfully ({len(climate_df)} rows).")
        return climate_df
    except Exception as e: # This except block correctly handles errors from the try block above (lines ~147-160)
        logging.error(f"Error fetching/processing climate data: {e}")
        return None

# --- Analysis Functions ---
# CORRECTED analyze_correlation with proper indentation for ALL try/except blocks
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
        logging.warning(f"Need >= 2 columns for corr, found: {cols_present}")
        return results

    df_analysis = df_merged[cols_present].dropna()
    if len(df_analysis) < 3:
        logging.warning("Less than 3 valid points for correlation after dropna.")
        return results

    # 1. Correlation Matrix
    try:
        results['corr_matrix'] = df_analysis.corr()
        logging.info("Correlation Matrix calculated.")
    except Exception as e:
        logging.error(f"Error calculating correlation matrix: {e}")
        return results # Return early if matrix calculation fails

    # 2. P-values for correlations
    logging.info("Calculating p-values...")
    for col1 in cols_present:
        for col2 in cols_present:
            if col1 == col2: continue
            pair_key = tuple(sorted((col1, col2)))
            if pair_key in results['p_values']: continue
            try: # Start try block for p-value calculation
                if df_analysis[col1].nunique() > 1 and df_analysis[col2].nunique() > 1:
                    corr_test = stats.pearsonr(df_analysis[col1], df_analysis[col2])
                    results['p_values'][pair_key] = corr_test.pvalue
                else:
                    logging.warning(f"Skipping p-value {col1} vs {col2}: no variance.")
                    results['p_values'][pair_key] = np.nan
            except Exception as e: # Correctly indented except block
                logging.warning(f"P-value error {col1} vs {col2}: {e}")
                results['p_values'][pair_key] = np.nan

    logging.info("P-values calculated.")

    # 3. Lagged Precipitation Correlation
    if PRECIP_COL in cols_present and DISCHARGE_COL in cols_present:
        logging.info("Calculating lagged precipitation correlation...")
        # --- CORRECTED try/except block ---
        try: # Start try block for lagged correlation
            df_analysis_lag = df_analysis.copy()
            df_analysis_lag['Precip_lag1'] = df_analysis_lag[PRECIP_COL].shift(1)
            df_analysis_lagged = df_analysis_lag.dropna()
            # Check length AFTER dropping NaNs from shift
            if len(df_analysis_lagged) >= 3:
                # Check variance AFTER potentially dropping rows
                if df_analysis_lagged[DISCHARGE_COL].nunique() > 1 and df_analysis_lagged['Precip_lag1'].nunique() > 1:
                    lag_corr_test = stats.pearsonr(df_analysis_lagged[DISCHARGE_COL], df_analysis_lagged['Precip_lag1'])
                    results['lagged_precip_corr'] = lag_corr_test.statistic
                    results['lagged_precip_p'] = lag_corr_test.pvalue
                    logging.info(f"Lag-1 precip corr calculated (R={results['lagged_precip_corr']:.3f}, p={results['lagged_precip_p']:.3f}).")
                else:
                    logging.warning("Not enough variance for lagged correlation.")
            else:
                logging.warning("Not enough data points for lagged correlation (after shift).")
        # **** THIS except block MUST be aligned with the try ****
        except Exception as e:
            logging.error(f"Error calculating lagged correlation: {e}")
        # --- End corrected block ---

    return results


# --- Temp Trend Analysis Functions (Copied/Adapted from Project 1) ---
# CORRECTED calculate_annual_means with proper indentation
def calculate_annual_means(df, col_name):
    """Calculates annual means for a given column."""
    logging.info(f"Calculating annual means for {col_name}...");
    if df is None or col_name not in df.columns or df.empty:
        logging.warning(f"Invalid DataFrame for annual mean calculation of {col_name}.")
        return None
    try:
        df.index = pd.to_datetime(df.index) # Ensure index is datetime
        annual_means = df[col_name].resample('AS').mean().dropna()
        logging.info(f"Calculated annual means for {col_name} ({len(annual_means)} years).")
        return annual_means
    except Exception as e:
        logging.error(f"Error calculating annual means for {col_name}: {e}")
        return None

# CORRECTED perform_trend_analysis with proper indentation
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
            results['linear_regression'] = {'slope': slope, 'intercept': intercept, 'p_value': p, 'r_squared': r**2, 'std_err': se, 'years': years[mask]}
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


# --- Plotting Function (Corrected Signature and Logic) ---
# Accepts df_q separately, handles None for merged/analysis/temp_trend
def plot_correlation_results(df_q, df_merged, analysis_results, temp_trend_results,
                             site_id, description, start_date, end_date, output_dir):
    """Generates plots visualizing available correlation results."""
    logging.info(f"Generating correlation plots for site {site_id}...")

    has_discharge = df_q is not None and not df_q.empty
    has_merged = df_merged is not None and not df_merged.empty
    has_corr_results = has_merged and analysis_results is not None and analysis_results.get('corr_matrix') is not None
    has_temp_trend = has_merged and temp_trend_results is not None

    if not has_discharge: logging.warning(f"No discharge data available for plot: {site_id}."); return

    try: # Main try block for the entire plotting function
        fig, axes = plt.subplots(3, 1, figsize=(10, 19))
        actual_start = df_q.index.min().strftime('%Y-%m-%d'); actual_end = df_q.index.max().strftime('%Y-%m-%d')
        plot_period_str = f"Discharge: {actual_start} to {actual_end}"
        if has_merged: actual_start_m = df_merged.index.min().strftime('%Y-%m-%d'); actual_end_m = df_merged.index.max().strftime('%Y-%m-%d'); plot_period_str = f"Overlap: {actual_start_m} to {actual_end_m}"
        wrapped_title = "\n".join(textwrap.wrap(f"Site {site_id}: {description}", width=60)); fig.suptitle(f"Discharge & Climate Correlation ({plot_period_str})\n{wrapped_title}", fontsize=14, y=0.99)

        # Plot 1: Heatmap
        ax = axes[0]
        if has_corr_results:
            corr_matrix = analysis_results['corr_matrix']; p_values = analysis_results.get('p_values', {})
            annot_fmt = {};
            for col1 in corr_matrix.columns:
                for col2 in corr_matrix.index: r = corr_matrix.loc[col1, col2]; pval = p_values.get(tuple(sorted((col1,col2))), 1.0); stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < P_SIGNIFICANCE_LEVEL else ""; annot_fmt[(col1, col2)] = f"{r:.2f}{stars}"
            annot_labels = corr_matrix.copy().astype(str);
            for r_idx, row_label in enumerate(corr_matrix.index):
                 for c_idx, col_label in enumerate(corr_matrix.columns): annot_labels.iloc[r_idx, c_idx] = annot_fmt.get((row_label, col_label), f"{corr_matrix.iloc[r_idx, c_idx]:.2f}")
            sns.heatmap(corr_matrix, annot=annot_labels, fmt='', cmap='coolwarm', linewidths=.5, cbar=True, ax=ax, vmin=-1, vmax=1); ax.set_title(f"Correlation Matrix ('*' p<{P_SIGNIFICANCE_LEVEL})"); ax.tick_params(axis='x', rotation=45); ax.tick_params(axis='y', rotation=0)
        else: _plot_placeholder(ax, "Correlation Matrix N/A\n(Climate Data Missing or No Overlap)")

        # Plot 2: Scatter
        ax = axes[1]
        if has_merged:
            try: # Specific try block for scatter plot generation
                df_plot_scatter = df_merged.copy(); df_plot_scatter['Precip_lag1'] = df_plot_scatter[PRECIP_COL].shift(1)
                df_plot = df_plot_scatter.dropna(subset=[DISCHARGE_COL, 'Precip_lag1'])
                if not df_plot.empty:
                     def get_season(month): return 'Winter' if month in [12, 1, 2] else 'Spring' if month in [3, 4, 5] else 'Summer' if month in [6, 7, 8] else 'Fall'
                     df_plot = df_plot.copy(); df_plot['Season'] = df_plot.index.month.map(get_season)
                     sns.scatterplot(data=df_plot, x='Precip_lag1', y=DISCHARGE_COL, hue='Season', alpha=0.3, s=15, ax=ax, legend='brief'); ax.set_title(f"Discharge vs. Previous Day's Precip"); ax.set_xlabel(f"Previous Day Precip ({PRECIP_COL})"); ax.set_ylabel(f"Discharge ({DISCHARGE_COL})"); ax.grid(True, linestyle='--', alpha=0.6); ax.legend(title='Season', fontsize='small')
                     lag_corr = analysis_results.get('lagged_precip_corr'); lag_p = analysis_results.get('lagged_precip_p');
                     if lag_corr is not None: lag_sig_str = f"(p={lag_p:.3f})" if lag_p >= 0.001 else "(p<0.001)"; lag_info = f"Lag-1 Corr: R={lag_corr:.2f} {lag_sig_str}"; ax.text(0.98, 0.02, lag_info, transform=ax.transAxes, fontsize=9, ha='right', va='bottom', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
                else: _plot_placeholder(ax, "Scatter Plot N/A (No Lagged Data)") # Line ~336
            except Exception as e: logging.error(f"Scatter plot error: {e}"); _plot_placeholder(ax, "Error plotting Scatter") # Line ~337 (Handles errors from the try block just above)
        # --- Lines 338-340 (Approximate based on file content) ---
        # Line 338 is the 'else' corresponding to 'if has_merged:' (line ~321).
        # It correctly handles the case where merged data wasn't available for the scatter plot.
        # It is NOT part of a try/except structure itself.
        else: _plot_placeholder(ax, "Scatter Plot N/A\n(Climate Data Missing or No Overlap)") # Line ~338

        # Plot 3: Time Series # Line ~340 (Start of next plot section)
        # --- End Section 338-340 ---
        ax = axes[2]
        try: # Specific try block for time series plot generation
            years_to_plot = 3; end_plot_date = df_q.index.max(); start_plot_date = end_plot_date - pd.DateOffset(years=years_to_plot)
            df_plot_q = df_q.loc[start_plot_date:end_plot_date]
            if not df_plot_q.empty:
                color1 = 'tab:blue'; ax.set_xlabel('Date'); ax.set_ylabel(DISCHARGE_COL, color=color1); ax.plot(df_plot_q.index, df_plot_q[DISCHARGE_COL], color=color1, linewidth=1.5, label='Discharge'); ax.tick_params(axis='y', labelcolor=color1); ax.grid(True, linestyle='--', alpha=0.6)
                plot_title = f"Recent Discharge ({years_to_plot} Years)"; legend_handles = ax.get_legend_handles_labels()[0]; legend_labels = ax.get_legend_handles_labels()[1]
                if has_merged and PRECIP_COL in df_merged.columns:
                    df_plot_clim = df_merged.loc[df_plot_q.index.min():df_plot_q.index.max(), PRECIP_COL]
                    if not df_plot_clim.empty:
                         ax2 = ax.twinx(); color2 = 'tab:green'; ax2.set_ylabel(f"{PRECIP_COL} (Inverted)", color=color2); ax2.plot(df_plot_clim.index, df_plot_clim, color=color2, linewidth=1.0, linestyle='-', alpha=0.7, label='Precipitation'); ax2.tick_params(axis='y', labelcolor=color2); ax2.invert_yaxis(); ax2.set_ylim(bottom=0)
                         plot_title = f"Recent Discharge and Precipitation ({years_to_plot} Years)"; lines2, labels2 = ax2.get_legend_handles_labels(); legend_handles.extend(lines2); legend_labels.extend(labels2)
                    else: logging.warning(f"Climate data empty for recent plot period: {site_id}"); ax.text(0.5, 0.5, "Recent Climate Data Unavailable", ha='center', va='center', fontsize=12, color='orange', alpha=0.6, transform=ax.transAxes)
                else: ax.text(0.5, 0.5, "Climate Data Unavailable", ha='center', va='center', fontsize=12, color='red', alpha=0.5, transform=ax.transAxes)
                ax.set_title(plot_title); ax.legend(legend_handles, legend_labels, loc='upper right', fontsize='small')
            else: _plot_placeholder(ax, "Time Series Plot N/A")
        except Exception as e: logging.error(f"Time series plot error: {e}"); _plot_placeholder(ax, "Error plotting Time Series")

        # Summary Text
        summary_text = _generate_corr_summary_text(analysis_results, temp_trend_results)
        fig.text(0.5, 0.01, summary_text, ha='center', va='bottom', fontsize=9, wrap=True, bbox=dict(boxstyle='round,pad=0.5', fc='lightyellow', alpha=0.8))

        # Layout and Save
        plt.subplots_adjust(bottom=0.18, hspace=0.45);
        # --- PNG Saving Logic ---
        # This section correctly ensures the output directory exists before saving.
        os.makedirs(output_dir, exist_ok=True) # Ensures the specific site directory exists
        plot_filename = f"USGS_{site_id}_climate_correlation_summary.png";
        plot_path = os.path.join(output_dir, plot_filename);
        plt.savefig(plot_path, dpi=150); # Saves the PNG file
        plt.close(fig);
        logging.info(f"Correlation summary plot saved: {plot_path}")
        # --- End PNG Saving Logic ---

    except Exception as e: # Catches errors from the main plotting try block (e.g., subplot creation)
        logging.error(f"Error generating correlation plot: {site_id}: {e}");
        plt.close() # Ensure figure is closed even if saving fails

# CORRECTED _generate_corr_summary_text with proper indentation and handling None
def _generate_corr_summary_text(analysis_results, temp_trend_results):
    """Generates summary text including correlations and temp trend."""
    summary_lines = ["Correlation Summary:"]

    def get_stars(pval): # Nested helper function
        """Returns significance stars based on p-value."""
        if pval is None or pd.isna(pval): return ""
        if pval < 0.001: return "***"
        if pval < 0.01: return "**"
        if pval < P_SIGNIFICANCE_LEVEL: return "*"
        return ""

    if analysis_results: # Check if analysis was performed
        corr_matrix = analysis_results.get('corr_matrix')
        p_values = analysis_results.get('p_values', {})
        lag_corr = analysis_results.get('lagged_precip_corr')
        lag_p = analysis_results.get('lagged_precip_p')

        if corr_matrix is not None:
            try: # This try block handles potential errors when accessing dict/Series keys
                # Use .get() on Series for safety if columns might be missing
                r_qp = corr_matrix.get(DISCHARGE_COL, {}).get(PRECIP_COL, np.nan)
                # Use sorted tuple key for p-values dict
                p_qp = p_values.get(tuple(sorted((DISCHARGE_COL, PRECIP_COL))), np.nan)
                summary_lines.append(f"- Q vs Precip: R={r_qp:.2f}{get_stars(p_qp)}" if not pd.isna(r_qp) else "- Q vs Precip: N/A")

                r_qt = corr_matrix.get(DISCHARGE_COL, {}).get(TEMP_COL, np.nan)
                p_qt = p_values.get(tuple(sorted((DISCHARGE_COL, TEMP_COL))), np.nan)
                summary_lines.append(f"- Q vs Temp:   R={r_qt:.2f}{get_stars(p_qt)}" if not pd.isna(r_qt) else "- Q vs Temp: N/A")

                if lag_corr is not None:
                     summary_lines.append(f"- Q vs Lag-1 Precip: R={lag_corr:.2f}{get_stars(lag_p)}")
            except Exception as e: # This except block correctly catches errors from the try above
                logging.error(f"Err formatting corr summary: {e}")
                summary_lines.append("- (Error extracting correlation values)")
        else:
             summary_lines.append("- Correlation Matrix N/A")
    else: # Handle case where analysis_results itself is None
        summary_lines.append("- Correlation Analysis N/A (Missing Data or Merge Failed)")

    # Temp Trend Summary
    summary_lines.append("\nTemperature Trend Summary:")
    if temp_trend_results and temp_trend_results.get('mann_kendall'):
        mk_temp = temp_trend_results['mann_kendall']
        trend = mk_temp.get('trend','N/A'); sig = "Sig." if mk_temp.get('h') else "Not Sig."
        slope = mk_temp.get('slope', np.nan); p = mk_temp.get('p', np.nan)
        summary_lines.append(f"- Annual Temp MK: {trend} ({sig}, p={p:.3f}, Slope={slope:.3f} C/yr)")
    else:
        summary_lines.append("- Annual Temp MK Trend: N/A")

    summary_lines.append(f"\n(Signif: * p<{P_SIGNIFICANCE_LEVEL}, ** p<0.01, *** p<0.001)")
    return "\n".join(summary_lines)

def _plot_placeholder(ax, message="Plot N/A"): ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=10, transform=ax.transAxes, wrap=True, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.5)); ax.set_xticks([]); ax.set_yticks([])

# --- Main Orchestration (Corrected for Config and Partial Plotting) ---
def main():
    """Main function runs climate correlation for sites in config, reading detailed site info."""
    config = load_config();
    if not config:
        # Use print because logging might not be set up
        print("CRITICAL: Failed to load configuration. Exiting.")
        return

    proj2_settings = config.get('project2_settings', {})
    # Use constants for defaults if keys missing
    log_filename = proj2_settings.get('log_file', DEFAULT_LOG_FILE)
    output_base_dir = proj2_settings.get('output_directory', DEFAULT_OUTPUT_DIR)
    setup_logging(log_file=log_filename)
    logging.info("--- Starting Climate Correlation Script (Unified Config Mode) ---")

    # Inventory lookup removed - using detailed config list now

    try:
        # --- Base Output Directory Creation ---
        # This ensures the main results directory exists.
        os.makedirs(output_base_dir, exist_ok=True)
        logging.info(f"Base output directory: {output_base_dir}")
    except Exception as e:
        logging.critical(f"Failed create output dir: {e}. Exiting.")
        return

    analysis_params = config.get("analysis_parameters", {})
    default_param_cd = analysis_params.get("param_cd", DEFAULT_PARAM_CD)
    default_start_date_str = analysis_params.get("start_date", DEFAULT_START_DATE)
    default_end_date_str = analysis_params.get("end_date", DEFAULT_END_DATE)

    sites_to_process = config.get("sites_to_process", []) # Get list of site dicts
    if not sites_to_process:
        logging.warning("No sites found in 'sites_to_process' list.")
        return # Exit if no sites

    all_site_results = {}

    for site_config in sites_to_process: # Loop through dicts in config list
        # --- Get Site Details from Config Dictionary ---
        if not site_config.get("enabled", False):
            logging.info(f"Skipping disabled: {site_config.get('site_id', 'Missing ID')}")
            continue

        site_id = site_config.get("site_id")
        param_cd = site_config.get("param_cd")
        start_date_str = site_config.get("start_date")
        end_date_str = site_config.get("end_date", "today")
        description = site_config.get("description", f"Site {site_id}")
        latitude_raw = site_config.get("latitude") # Get raw value first
        longitude_raw = site_config.get("longitude")

        # Validate essential fields exist before trying conversion
        if not all([site_id, param_cd, start_date_str, latitude_raw is not None, longitude_raw is not None]):
            logging.warning(f"Skipping {site_id or 'N/A'}: Missing required config data.")
            continue

        # --- CORRECTED try/except block for Lat/Lon validation ---
        try:
            latitude = float(latitude_raw)
            longitude = float(longitude_raw)
            # Check range *after* successful conversion
            if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
                # Raise error *inside* the try block if range is invalid
                raise ValueError("Lat/Lon out of valid range")
        # Ensure except block is correctly aligned with try
        except (ValueError, TypeError):
            logging.warning(f"Skipping {site_id}: Invalid latitude ('{latitude_raw}') or longitude ('{longitude_raw}') in config.")
            continue # Skip to next site if coordinates are invalid
        # --- End corrected block ---

        logging.info(f"--- Processing Site: {site_id} ({description}) ---")
        try: # Date handling
            if end_date_str.lower() == 'today':
                end_date = datetime.now()
            else:
                end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            if start_date >= end_date:
                logging.warning(f"Start date not before end date: {site_id}. Skipping.")
                continue
            start_date_nwis = start_date.strftime('%Y-%m-%d')
            end_date_nwis = end_date.strftime('%Y-%m-%d')
        except ValueError as e:
            logging.error(f"Invalid date format: {site_id}: {e}. Skipping.")
            continue

        # --- Create Specific Site Output Dir ---
        # This is where the PNG for the specific site will be saved.
        site_output_dir = os.path.join(output_base_dir, site_id)
        try:
            # This call ensures the site-specific folder exists before plotting.
            os.makedirs(site_output_dir, exist_ok=True)
        except Exception as e:
            logging.error(f"Could not create output dir {site_id}: {e}. Skipping.")
            continue

        # --- Workflow ---
        site_results = {'status': 'Started'}; df_q = None; df_climate = None; df_merged = None; analysis_results = None; temp_trend_results = None

        discharge_wml = fetch_waterml_data(site_id, param_cd, start_date_nwis, end_date_nwis)
        if discharge_wml:
            df_q = parse_waterml(discharge_wml, site_id)

        # Fetch climate using validated latitude and longitude
        df_climate = fetch_climate_data(latitude, longitude, start_date, end_date)

        # --- Process based on available data ---
        # --- ADDED DEBUG LOGGING ---
        if df_q is not None and not df_q.empty:
            # Log the last 20 available discharge values
            logging.debug(f"Most recent 20 discharge values for {site_id}:\n{df_q[DISCHARGE_COL].tail(20).to_string()}")
        elif df_q is not None: # Parsed but empty
             logging.debug(f"Discharge parsing resulted in an empty DataFrame for {site_id}.")
        # --- END DEBUG LOGGING ---

        # Check if discharge failed or is empty
        if df_q is None or df_q.empty:
             logging.error(f"Discharge data missing or empty for {site_id}. Plotting skipped.")
             site_results['status'] = 'Error (Discharge Missing)'
        # Check if climate failed or is empty (only if discharge was successful)
        elif df_climate is None or df_climate.empty:
             logging.warning(f"Climate data fetch failed for {site_id}. Correlation analysis skipped.")
             site_results['status'] = 'Processed (Climate Missing)'
             # Call plotting function with discharge data but None for climate-dependent args
             # The plot function will handle missing climate/merged data internally.
             plot_correlation_results(df_q, None, None, None, site_id, description, start_date_nwis, end_date_nwis, site_output_dir)
        else:
            # Both datasets exist, proceed with merge and analysis
            logging.info(f"Merging data for site {site_id}...")
            # Inner merge keeps only overlapping dates
            df_merged = pd.merge(df_q, df_climate, left_index=True, right_index=True, how='inner')

            if df_merged.empty:
                logging.error(f"Merged DataFrame empty (no overlapping data): {site_id}. Plotting discharge only.")
                site_results['status'] = 'Processed (No Overlap)';
                # Call plotting function with discharge data but None for climate-dependent args
                # The plot function will handle missing merged data internally.
                plot_correlation_results(df_q, None, None, None, site_id, description, start_date_nwis, end_date_nwis, site_output_dir)
            else:
                logging.info(f"Merged DataFrame for analysis shape: {df_merged.shape}")
                analysis_results = analyze_correlation(df_merged)
                site_results['correlation_results'] = analysis_results
                # Analyze Temperature Trend on the MERGED data
                logging.info(f"Analyzing temperature trend: {site_id}...")
                temp_annual_means = calculate_annual_means(df_merged, TEMP_COL)
                if temp_annual_means is not None:
                    temp_trend_results = perform_trend_analysis(temp_annual_means, "Annual Temp")
                    site_results['temp_trend_results'] = temp_trend_results
                else:
                    logging.warning(f"Could not calc annual temp means: {site_id}")
                site_results['status'] = 'Processed (Full)'
                # Plotting - Pass all results
                # The plot function receives the correct site_output_dir where the PNG will be saved.
                plot_correlation_results(df_q, df_merged, analysis_results, temp_trend_results, site_id, description, start_date_nwis, end_date_nwis, site_output_dir)

        all_site_results[site_id] = site_results
        logging.info(f"--- Finished Processing Site: {site_id} ---")

    logging.info("--- Climate Correlation Script Finished ---")
    # Optional: Save all site results to a JSON file (implement if needed)
    # try:
    #     results_filename = os.path.join(output_base_dir, "all_site_analysis_results.json")
    #     # Need to handle non-serializable objects if any (like DataFrames or complex objects)
    #     # A helper function to convert results to basic types might be needed.
    #     # with open(results_filename, 'w') as f:
    #     #     json.dump(all_site_results, f, indent=4)
    #     # logging.info(f"All site results saved to {results_filename}")
    # except Exception as e:
    #     logging.error(f"Failed to save all site results summary: {e}")

# --- Script Entry Point ---
if __name__ == "__main__":
    main()

