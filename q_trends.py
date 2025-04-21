import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import xml.etree.ElementTree as ET
from io import StringIO
import json # For config file
import logging # For logging
import os # For creating directories, paths
from datetime import datetime # For 'today' date handling
import pymannkendall as mk # Import Mann-Kendall library
import matplotlib.ticker as mticker # For formatting log ticks
import calendar # For month names
import textwrap # For wrapping long titles/text

# --- Constants ---
DEFAULT_CONFIG_PATH = 'config.json'
DISCHARGE_COL = 'Discharge_cfs' # Standard column name after parsing
MK_ALPHA = 0.05 # Significance level for Mann-Kendall

# --- Logging Setup ---
# (Setup logging function remains the same)
def setup_logging(log_file='trend_analysis.log'):
    root_logger = logging.getLogger()
    if root_logger.hasHandlers() and len(root_logger.handlers) >= 2: return
    for handler in root_logger.handlers[:]: root_logger.removeHandler(handler); handler.close()
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    root_logger.setLevel(logging.INFO)
    try:
        file_handler = logging.FileHandler(log_file, mode='w'); file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)
    except Exception as e: print(f"Error setting up file logger for {log_file}: {e}")
    console_handler = logging.StreamHandler(); console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

# --- Configuration Loading ---
# (load_config function remains the same)
def load_config(config_path=DEFAULT_CONFIG_PATH):
    print(f"Attempting to load configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f: config_data = json.load(f)
        print("Configuration loaded successfully.")
        return config_data
    except FileNotFoundError: print(f"ERROR: File not found: {config_path}"); return None
    except json.JSONDecodeError as e: print(f"ERROR: Error decoding JSON: {e}"); return None
    except Exception as e: print(f"ERROR: Unexpected error loading config: {e}"); return None

# --- Data Fetching ---
# (fetch_waterml_data function remains the same)
def fetch_waterml_data(site_id, param_cd, start_date, end_date):
    logging.info(f"Attempting to fetch WaterML for site {site_id} ({start_date} to {end_date})...")
    url_wml = "https://waterservices.usgs.gov/nwis/dv"; params_wml = {'format': 'waterml,1.1', 'sites': site_id, 'parameterCd': param_cd, 'startDT': start_date, 'endDT': end_date}
    try:
        response_wml = requests.get(url_wml, params=params_wml, timeout=90); response_wml.raise_for_status()
        logging.info(f"WaterML fetch successful: {site_id}. Length: {len(response_wml.text)}")
        return response_wml.text
    except requests.exceptions.Timeout: logging.error(f"Request timed out: {site_id}."); return None
    except requests.exceptions.RequestException as e: logging.error(f"Requests error: {site_id}: {e}"); return None
    except Exception as e: logging.error(f"Unexpected fetch error: {site_id}: {e}"); return None

# --- Data Parsing ---
# (parse_waterml function remains the same)
def parse_waterml(waterml_content, site_id):
    if not waterml_content: logging.warning(f"No content to parse: {site_id}."); return None
    logging.info(f"Parsing WaterML: {site_id}...")
    try:
        namespaces = {'ns1': 'http://www.cuahsi.org/waterML/1.1/'}; xml_io = StringIO(waterml_content)
        tree = ET.parse(xml_io); root = tree.getroot()
        value_elements = root.findall('./ns1:timeSeries/ns1:values/ns1:value', namespaces)
        if not value_elements: logging.warning(f"No <value> tags found: {site_id}."); return None
        data = []; skipped_count = 0
        for value_elem in value_elements:
            timestamp_str = value_elem.get('dateTime'); value_str = value_elem.text; qualifiers = value_elem.get('qualifiers', '')
            if 'P' not in qualifiers and value_str is not None:
                try: data.append({'Timestamp': pd.to_datetime(timestamp_str), DISCHARGE_COL: float(value_str)})
                except (ValueError, TypeError): skipped_count += 1
            else: skipped_count += 1
        if not data: logging.warning(f"No valid data points extracted: {site_id}."); return None
        df = pd.DataFrame(data).set_index('Timestamp').sort_index()
        logging.info(f"DataFrame created: {site_id} ({len(df)} rows).")
        if skipped_count > 0: logging.info(f"Skipped {skipped_count} points: {site_id}.")
        return df
    except ET.ParseError as e: logging.error(f"XML parse error: {site_id}: {e}"); return None
    except Exception as e: logging.error(f"Unexpected parse error: {site_id}: {e}"); return None

# --- Analysis Functions ---
# (calculate_annual_means function remains the same)
def calculate_annual_means(df, discharge_col=DISCHARGE_COL):
    logging.info("Calculating annual means...");
    if df is None or discharge_col not in df.columns or df.empty: logging.warning("Invalid DF for annual mean."); return None
    try:
        df.index = pd.to_datetime(df.index); annual_mean_flow = df[discharge_col].resample('AS').mean().dropna()
        logging.info(f"Calculated annual means for {len(annual_mean_flow)} years.")
        return annual_mean_flow
    except Exception as e: logging.error(f"Error calculating annual means: {e}"); return None

# (perform_annual_trend_analysis function remains the same)
def perform_annual_trend_analysis(series):
    logging.info("Performing ANNUAL trend analysis..."); results = {'linear_regression': None, 'mann_kendall': None}
    if series is None or series.empty or len(series) < 3: logging.warning(f"Insuff. data for annual trend ({len(series) if series is not None else 0} points)."); return results
    try: # LinReg
        years = series.index.year.astype(float); flow_values = series.values; mask = ~np.isnan(years) & ~np.isnan(flow_values)
        if np.sum(mask) >= 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(years[mask], flow_values[mask])
            results['linear_regression'] = {'slope': slope, 'intercept': intercept, 'p_value': p_value, 'r_squared': r_value**2, 'std_err': std_err, 'years': years[mask]}
            logging.info(f"  Annual LinReg: p={p_value:.4f}, slope={slope:.2f}")
        else: logging.warning(" < 2 valid points for Annual LinReg.")
    except Exception as e: logging.error(f"Error during Annual LinReg: {e}")
    try: # MK Test
        mk_result = mk.original_test(series.sort_index().values, alpha=MK_ALPHA)
        results['mann_kendall'] = {'trend': mk_result.trend, 'h': mk_result.h, 'p': mk_result.p, 'z': mk_result.z, 'Tau': mk_result.Tau, 's': mk_result.s, 'var_s': mk_result.var_s, 'slope': mk_result.slope, 'intercept': mk_result.intercept}
        logging.info(f"  Annual MK: Trend={mk_result.trend}, p={mk_result.p:.4f}, SenSlope={mk_result.slope:.2f}")
    except Exception as e: logging.error(f"Error during Annual MK: {e}")
    return results

# (calculate_monthly_means function remains the same)
def calculate_monthly_means(df, discharge_col=DISCHARGE_COL):
    logging.info("Calculating monthly means...");
    if df is None or discharge_col not in df.columns or df.empty: logging.warning("Invalid DF for monthly mean."); return None
    try:
        df.index = pd.to_datetime(df.index); monthly_mean_flow = df[discharge_col].resample('MS').mean().dropna()
        logging.info(f"Calculated monthly means for {len(monthly_mean_flow)} months.")
        return monthly_mean_flow
    except Exception as e: logging.error(f"Error calculating monthly means: {e}"); return None

# (perform_monthly_trend_analysis function remains the same)
def perform_monthly_trend_analysis(monthly_series):
    logging.info("Performing MONTHLY trend analysis (Mann-Kendall)...")
    if monthly_series is None or monthly_series.empty: logging.warning("No monthly data for trend analysis."); return None
    monthly_results = {};
    try:
        monthly_series.index = pd.to_datetime(monthly_series.index)
        for month in range(1, 13):
            month_data = monthly_series[monthly_series.index.month == month]; month_name = calendar.month_abbr[month]
            results_key = f"{month:02d}_{month_name}"
            if len(month_data) < 3: logging.warning(f"  Insuff. data (< 3 yrs) for {month_name}."); monthly_results[results_key] = None; continue
            try:
                mk_result = mk.original_test(month_data.sort_index().values, alpha=MK_ALPHA)
                monthly_results[results_key] = {'trend': mk_result.trend, 'h': mk_result.h, 'p': mk_result.p, 'slope': mk_result.slope}
                logging.info(f"  Month {month_name}: Trend={mk_result.trend}, p={mk_result.p:.4f}, SenSlope={mk_result.slope:.2f}")
            except Exception as e: logging.error(f"MK error for {month_name}: {e}"); monthly_results[results_key] = None
        return monthly_results
    except Exception as e: logging.error(f"Error processing monthly trends: {e}"); return None

# (calculate_fdc function remains the same)
def calculate_fdc(df_daily, discharge_col=DISCHARGE_COL):
    logging.info("Calculating Flow Duration Curve (FDC)...");
    if df_daily is None or discharge_col not in df_daily.columns or df_daily.empty: return None
    try:
        discharge = df_daily[discharge_col].dropna(); n = len(discharge)
        if n == 0: logging.warning("No valid values for FDC."); return None
        discharge_sorted = discharge.sort_values(ascending=False); rank = np.arange(1, n + 1); exceedance_prob = (rank / (n + 1)) * 100
        fdc_df = pd.DataFrame({'discharge': discharge_sorted.values, 'exceedance_probability': exceedance_prob})
        logging.info(f"FDC calculated with {n} points.")
        return fdc_df
    except Exception as e: logging.error(f"Error calculating FDC: {e}"); return None


# --- Plotting Functions ---
# Modified signatures to accept description, start_date, end_date
def plot_trend(annual_series, trend_results, site_id, description, param_cd, start_date, end_date, output_dir):
    """Generates and saves the individual annual trend plot."""
    logging.info(f"Generating individual trend plot for site {site_id}...")
    if annual_series is None or annual_series.empty: logging.warning(f"No annual data to plot trend: {site_id}"); return
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        # Pass new args to helper
        _plot_trend_on_axes(ax, annual_series, trend_results, site_id, description, param_cd, start_date, end_date)
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plot_filename = f"USGS_{site_id}_{param_cd}_annual_trend.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path); plt.close(fig)
        logging.info(f"Individual annual trend plot saved: {plot_path}")
    except Exception as e: logging.error(f"Error saving individual trend plot: {site_id}: {e}"); plt.close()

def plot_fdc(fdc_df, site_id, description, param_cd, start_date, end_date, output_dir):
    """Generates and saves the individual Flow Duration Curve plot."""
    logging.info(f"Generating individual FDC plot for site {site_id}...")
    if fdc_df is None or fdc_df.empty: logging.warning(f"No FDC data to plot: {site_id}"); return
    try:
        fig, ax = plt.subplots(figsize=(10, 7))
         # Pass new args to helper
        _plot_fdc_on_axes(ax, fdc_df, site_id, description, param_cd, start_date, end_date)
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plot_filename = f"USGS_{site_id}_{param_cd}_fdc.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path); plt.close(fig)
        logging.info(f"Individual FDC plot saved: {plot_path}")
    except Exception as e: logging.error(f"Error saving individual FDC plot: {site_id}: {e}"); plt.close()

# Modified signature to accept description, start_date, end_date
def plot_combined_results(df_daily, annual_series, annual_trend_results, fdc_data, monthly_trend_results,
                          site_id, description, param_cd, start_date, end_date, output_dir):
    """Generates and saves a combined 2x2 plot with site description and summary."""
    logging.info(f"Generating combined plot for site {site_id}...")

    has_annual = annual_series is not None and not annual_series.empty
    has_fdc = fdc_data is not None and not fdc_data.empty
    has_daily = df_daily is not None and not df_daily.empty
    has_monthly_trends = monthly_trend_results is not None

    if not (has_annual or has_fdc or has_daily): logging.warning(f"No data for combined plot: {site_id}"); return

    try:
        fig, axes = plt.subplots(2, 2, figsize=(17, 13)) # Adjusted figsize
        # Wrap long descriptions for title
        wrapped_title = "\n".join(textwrap.wrap(f"Site {site_id}: {description}", width=80))
        fig.suptitle(f"Hydrology Summary ({start_date} to {end_date}) - Param: {param_cd}\n{wrapped_title}", fontsize=14, y=0.99)
        ax_flat = axes.flatten()

        # --- Top-Left: Annual Trend ---
        if has_annual: _plot_trend_on_axes(axes[0, 0], annual_series, annual_trend_results, site_id, description, param_cd, start_date, end_date, is_subplot=True)
        else: _plot_placeholder(axes[0, 0], 'Annual Trend Data N/A')

        # --- Bottom-Left: FDC ---
        if has_fdc: _plot_fdc_on_axes(axes[1, 0], fdc_data, site_id, description, param_cd, start_date, end_date, is_subplot=True)
        else: _plot_placeholder(axes[1, 0], 'FDC Data N/A')

        # --- Top-Right: Monthly Box Plot ---
        if has_daily: _plot_monthly_boxplot_on_axes(axes[0, 1], df_daily, DISCHARGE_COL, is_subplot=True)
        else: _plot_placeholder(axes[0, 1], 'Monthly Boxplot N/A')

        # --- Bottom-Right: Monthly MK Trends ---
        if has_monthly_trends: _plot_monthly_mk_trends_on_axes(axes[1, 1], monthly_trend_results, is_subplot=True)
        else: _plot_placeholder(axes[1, 1], 'Monthly Trend N/A')

        # --- Add Summary Text Box Below Plots ---
        summary_text = _generate_summary_text(annual_trend_results, monthly_trend_results, len(annual_series) if has_annual else 0)
        # Add text outside the tight_layout boundary initially
        fig.text(0.5, 0.01, summary_text, ha='center', va='bottom', fontsize=9, wrap=True,
                 bbox=dict(boxstyle='round,pad=0.5', fc='lightgray', alpha=0.5))

        # Adjust layout - may need fine-tuning depending on text length
        plt.subplots_adjust(bottom=0.15, hspace=0.3, wspace=0.25) # Increase bottom margin, add spacing

        # Save the combined plot
        os.makedirs(output_dir, exist_ok=True)
        plot_filename = f"USGS_{site_id}_{param_cd}_combined_summary.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=150); plt.close(fig)
        logging.info(f"Combined plot saved to: {plot_path}")

    except Exception as e: logging.error(f"Error generating combined plot: {site_id}: {e}"); plt.close()


# --- Helper Plotting Functions ---
# Modified signatures
def _plot_trend_on_axes(ax, annual_series, trend_results, site_id, description, param_cd, start_date, end_date, is_subplot=False):
    """Helper to draw annual trend onto given axes."""
    ax.plot(annual_series.index, annual_series.values, marker='o', linestyle='-', markersize=4, label='Mean Annual Flow', zorder=5)
    title_extra = ""; legend_handles = [ax.get_lines()[0]]
    lr_stats = trend_results.get('linear_regression'); mk_stats = trend_results.get('mann_kendall')
    # Add LinReg line
    if lr_stats:
        years = annual_series.index.year; intercept = lr_stats['intercept']; slope = lr_stats['slope']; p_value_lr = lr_stats['p_value']
        line, = ax.plot(annual_series.index, intercept + slope * years, 'r--', linewidth=1.5, label=f'LinReg (p={p_value_lr:.3f}, slope={slope:.2f})', zorder=10)
        legend_handles.append(line)
    # Prep MK info
    mk_info = "MK Trend: N/A"
    if mk_stats:
        trend_type = mk_stats['trend']; p_value_mk = mk_stats['p']; sens_slope = mk_stats['slope']; significance = "Sig." if mk_stats['h'] else "Not Sig."
        mk_info = f"MK Trend: {trend_type} ({significance} p={p_value_mk:.3f}), SenSlope={sens_slope:.2f} cfs/yr"

    # Set Title
    if is_subplot:
         ax.set_title(f'Annual Trend ({annual_series.index.year.min()}-{annual_series.index.year.max()})')
         # Add MK info as annotation within subplot
         ax.text(0.98, 0.02, mk_info, transform=ax.transAxes, fontsize=8, va='bottom', ha='right', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    else: # Standalone plot
         wrapped_title = "\n".join(textwrap.wrap(f"USGS Site {site_id}: {description}", width=60))
         ax.set_title(f"Mean Annual Discharge Trend ({start_date} to {end_date})\n{wrapped_title}\n{mk_info}")
         line, = ax.plot([], [], linestyle='', label=mk_info); legend_handles.append(line) # Add MK to legend
         ax.legend(handles=legend_handles, fontsize='small')

    ax.set_xlabel("Year"); ax.set_ylabel("Mean Annual Flow (cfs)")
    ax.grid(True)


def _plot_fdc_on_axes(ax, fdc_df, site_id, description, param_cd, start_date, end_date, is_subplot=False):
    """Helper to draw FDC onto given axes."""
    ax.plot(fdc_df['exceedance_probability'], fdc_df['discharge'])
    ax.set_yscale('log'); ax.set_xlabel("Exceedance Probability (%)"); ax.set_ylabel("Daily Q (cfs) - Log Scale")
    if is_subplot: ax.set_title('Flow Duration Curve')
    else: ax.set_title(f"Flow Duration Curve ({start_date} to {end_date})\nUSGS Site {site_id}: {description[:60]}{'...' if len(description)>60 else ''}") # Truncate description
    ax.grid(True, which='both'); ax.yaxis.set_major_formatter(mticker.ScalarFormatter()); ax.yaxis.get_major_formatter().set_scientific(False); ax.yaxis.get_major_formatter().set_useOffset(False)

def _plot_monthly_boxplot_on_axes(ax, df_daily, discharge_col, is_subplot=False):
    """Helper to draw monthly boxplot onto given axes."""
    if df_daily is None or discharge_col not in df_daily.columns or df_daily.empty: _plot_placeholder(ax, 'Monthly Boxplot N/A'); return
    try:
        df_temp = df_daily.copy(); df_temp['Month'] = df_temp.index.month
        df_temp.boxplot(column=discharge_col, by='Month', ax=ax, showfliers=False, patch_artist=True, medianprops=dict(color='red', linewidth=1.5))
        ax.set_yscale('log'); ax.set_xlabel("Month"); ax.set_ylabel("Daily Q (cfs) - Log Scale")
        title = 'Monthly Discharge Distribution'; ax.set_title(title); plt.suptitle(''); ax.figure.suptitle(''); ax.set_title(title) # Clean up pandas title
        ax.grid(True, which='both', axis='y'); ax.set_xticks(range(1,13)); ax.set_xticklabels(calendar.month_abbr[1:]) # Set month labels
    except Exception as e: logging.error(f"Error generating monthly boxplot: {e}"); _plot_placeholder(ax, 'Error plotting Boxplot')

def _plot_monthly_mk_trends_on_axes(ax, monthly_trend_results, is_subplot=False):
    """Helper function to draw monthly MK trend results bar chart onto given axes."""
    if not monthly_trend_results: _plot_placeholder(ax, 'Monthly Trend Results N/A'); return
    try:
        months = range(1, 13); month_names = [calendar.month_abbr[m] for m in months]
        slopes = []; colors = []; significance_markers = []
        for m in months:
            key = f"{m:02d}_{calendar.month_abbr[m]}"
            result = monthly_trend_results.get(key)
            if result:
                slope = result.get('slope', 0); is_significant = result.get('h', False)
                slopes.append(slope)
                if is_significant: colors.append('red' if slope < 0 else 'green'); significance_markers.append('*')
                else: colors.append('grey'); significance_markers.append('')
            else: slopes.append(0); colors.append('lightgrey'); significance_markers.append('N/A')
        bars = ax.bar(month_names, slopes, color=colors)
        ax.set_xlabel("Month"); ax.set_ylabel("Sen's Slope (cfs / year)")
        ax.set_title("Monthly Discharge Trend (Mann-Kendall)"); ax.grid(True, axis='y')
        ax.axhline(0, color='black', linewidth=0.8)
        for bar, sig in zip(bars, significance_markers): # Add significance markers
            if sig == '*': ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), sig, ha='center', va='bottom' if bar.get_height() >= 0 else 'top', fontsize=14, color='black')
        from matplotlib.patches import Patch # Custom legend
        legend_elements = [Patch(facecolor='green', label=f'Incr. (p<{MK_ALPHA})'), Patch(facecolor='red', label=f'Decr. (p<{MK_ALPHA})'), Patch(facecolor='grey', label='Not Sig.')]
        ax.legend(handles=legend_elements, fontsize='small', loc='best')
    except Exception as e: logging.error(f"Error generating monthly trends plot: {e}"); _plot_placeholder(ax, 'Error plotting Monthly Trends')

def _plot_placeholder(ax, message="Plot Data Not Available"):
     """Helper to display a message on an Axes object."""
     ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=10, transform=ax.transAxes, wrap=True, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.5))
     ax.set_xticks([]); ax.set_yticks([])

# --- NEW: Helper to generate summary text ---
def _generate_summary_text(annual_trend_results, monthly_trend_results, num_years):
    """Generates a summary string of key trend results."""
    summary_lines = [f"Analysis Period: {num_years} years"]
    # Annual Trend Summary
    if annual_trend_results and annual_trend_results.get('mann_kendall'):
        mk_annual = annual_trend_results['mann_kendall']
        trend_ann = mk_annual.get('trend','N/A')
        sig_ann = "Significant" if mk_annual.get('h') else "Not Significant"
        slope_ann = mk_annual.get('slope', np.nan)
        p_ann = mk_annual.get('p', np.nan)
        summary_lines.append(f"Annual MK Trend: {trend_ann} ({sig_ann}, p={p_ann:.3f}, Slope={slope_ann:.2f} cfs/yr)")
    else:
        summary_lines.append("Annual MK Trend: N/A or insufficient data")

    # Monthly Trend Summary
    if monthly_trend_results:
        sig_monthly_count = sum(1 for res in monthly_trend_results.values() if res and res.get('h'))
        summary_lines.append(f"Significant Monthly Trends (p<{MK_ALPHA}): {sig_monthly_count} out of 12 months")
        # Add details on first few significant months if needed
        sig_details = []
        limit = 3 # Limit number of details shown
        count = 0
        for month_key, res in monthly_trend_results.items():
            if res and res.get('h'):
                 month_abbr = month_key.split('_')[1]
                 slope_mon = res.get('slope', np.nan)
                 sig_details.append(f"{month_abbr} ({slope_mon:.1f})")
                 count += 1
                 if count >= limit: break
        if sig_details: summary_lines.append(f"  e.g., {', '.join(sig_details)}{'...' if count < sig_monthly_count else ''}")

    else:
        summary_lines.append(f"Monthly Trends: Analysis N/A")

    return "\n".join(summary_lines)

# --- Main Orchestration ---
def main():
    """Main function uses the unified config file with detailed site entries."""
    # --- Setup ---
    config = load_config() # Load the single config file
    if not config:
        # Cannot proceed without config, exit here before logging setup
        # Use print because logging might not be set up
        print("CRITICAL: Failed to load configuration. Exiting.")
        return

    # --- Get settings specific to Project 1 from the config ---
    proj1_settings = config.get('project1_settings', {}) # Get the nested dictionary for P1

    # Use settings from proj1_settings, with defaults if keys are missing
    log_filename = proj1_settings.get('log_file', 'trend_analysis.log')
    setup_logging(log_file=log_filename) # Setup logging using the correct file name
    logging.info("--- Starting Trend Analysis Script (Unified Config Mode) ---")

    # Get base output directory for Project 1 results
    output_base_dir = proj1_settings.get('output_directory', 'discharge_analysis_results')
    try:
        os.makedirs(output_base_dir, exist_ok=True)
        logging.info(f"Base output directory for trends: {output_base_dir}")
    except Exception as e:
        logging.critical(f"Failed create output dir '{output_base_dir}': {e}. Exiting.")
        return

    # --- Process Sites from Config List ---
    sites_to_process = config.get("sites_to_process", []) # Get the list of site dictionaries
    if not sites_to_process:
        logging.warning("No sites found in 'sites_to_process' list in config.")
        return # Exit if no sites to run

    all_site_results = {} # To store results for final summary

    # Loop through the list of site *dictionaries* provided in the config
    for site_config in sites_to_process:
        # --- Get Site Details Directly from site_config dictionary ---
        if not site_config.get("enabled", False): # Check enabled flag first
            logging.info(f"Skipping disabled site config: {site_config.get('site_id', 'Missing ID')}")
            continue

        # Extract required fields directly from the dictionary for this site
        site_id = site_config.get("site_id")
        param_cd = site_config.get("param_cd")
        start_date_str = site_config.get("start_date")
        end_date_str = site_config.get("end_date", "today") # Use default if missing
        description = site_config.get("description", f"Site {site_id}") # Use ID if desc missing
        # Latitude/Longitude are present in the config dict but ignored by this script

        # Validate essential fields needed from the site_config for this script
        if not all([site_id, param_cd, start_date_str]):
            logging.warning(f"Skipping site config due to missing 'site_id', 'param_cd', or 'start_date': {site_config}")
            continue

        logging.info(f"--- Processing Site: {site_id} ({description}) ---")

        # --- Handle Dates ---
        try:
            if end_date_str.lower() == 'today':
                end_date = datetime.now().strftime('%Y-%m-%d')
            else:
                # Validate date format
                datetime.strptime(end_date_str, '%Y-%m-%d')
                end_date = end_date_str
            # Validate start date format too
            datetime.strptime(start_date_str, '%Y-%m-%d')
        except ValueError:
            logging.error(f"Invalid date format found for site {site_id} (start='{start_date_str}', end='{end_date_str}'). Skipping.")
            continue

        # --- Create Output Dir ---
        # Use the output_base_dir obtained from project1_settings
        site_output_dir = os.path.join(output_base_dir, site_id)
        try:
            os.makedirs(site_output_dir, exist_ok=True)
        except Exception as e:
            logging.error(f"Could not create output dir for {site_id}: {e}. Skipping.")
            continue

        # --- Workflow ---
        # (Initialize results for this site)
        # Store description extracted from config for later summary use
        site_results = {'status': 'Started', 'description': description}
        df_daily = None; monthly_means = None; monthly_trend_results = None; annual_means = None; annual_trend_results = None; fdc_data = None; # Removed low/high flow init as they weren't in user's provided main

        waterml_data = fetch_waterml_data(site_id, param_cd, start_date_str, end_date)
        if waterml_data:
            df_daily = parse_waterml(waterml_data, site_id)

        if df_daily is not None and not df_daily.empty:
            site_results['status'] = 'Processed'; site_results['daily_data_rows'] = len(df_daily)

            fdc_data = calculate_fdc(df_daily) # Calc FDC
            if fdc_data is not None:
                # Pass description, dates to individual plot
                plot_fdc(fdc_data, site_id, description, param_cd, start_date_str, end_date, site_output_dir)
                site_results['fdc_calculated'] = True
            else: site_results['fdc_calculated'] = False

            monthly_means = calculate_monthly_means(df_daily) # Calc Monthly Means
            if monthly_means is not None:
                monthly_trend_results = perform_monthly_trend_analysis(monthly_means)
            site_results['monthly_trend_results'] = monthly_trend_results # Store even if None

            annual_means = calculate_annual_means(df_daily) # Calc Annual Means
            if annual_means is not None and not annual_means.empty:
                 site_results['annual_means_calculated'] = True
                 # NOTE: Using the function name from the user's provided script
                 annual_trend_results = perform_annual_trend_analysis(annual_means)
                 site_results['annual_trend_results'] = annual_trend_results
                 # Pass description, dates to individual plot
                 # NOTE: Using the function name from the user's provided script
                 plot_trend(annual_means, annual_trend_results, site_id, description, param_cd, start_date_str, end_date, site_output_dir) # Removed plot_suffix
            else:
                 logging.warning(f"Annual means failed: {site_id}. Annual trend skipped.")
                 site_results['annual_means_calculated'] = False

            # NOTE: Low flow / High flow calculations were not in the user's provided main()
            # Add them back here if needed, otherwise they remain commented out/removed

            # Plot Combined Summary - Pass description, dates
            # NOTE: Using the function name from the user's provided script
            # NOTE: Passing None for low/high flow results as they are not calculated here
            plot_combined_results(
                df_daily, annual_means, site_results.get('annual_trend_results'),
                fdc_data, site_results.get('monthly_trend_results'),
                # None, None, # Pass None for low/high flow results if not calculated
                site_id, description, param_cd, start_date_str, end_date, site_output_dir
            )
        else: # Handle fetch/parse failure
             status = 'Parse Failed' if waterml_data else 'Fetch Failed'
             logging.warning(f"Analysis skipped: {site_id} ({status})."); site_results['status'] = status

        all_site_results[site_id] = site_results # Store results for this site
        logging.info(f"--- Finished Processing Site: {site_id} ---")

    # --- Post-Processing Summary Log ---
    logging.info("--- Summary of Trend Results (Mann-Kendall) ---");
    if not all_site_results: logging.info("No sites processed.")
    else:
        sorted_site_ids = sorted(all_site_results.keys())
        logging.info(f"{'Site ID':<14} | {'Description':<45} | {'MK Trend':<14} | {'MK Sig':<6} | {'MK p-val':<8} | {'Sen Slope':<10}"); logging.info("-" * 105)

        # Get descriptions from the results dictionary we built during the loop
        processed_site_details = {sid: res.get('description', 'N/A') for sid, res in all_site_results.items()}

        for site_id in sorted_site_ids:
            results = all_site_results[site_id]; description = processed_site_details.get(site_id, 'N/A')[:45]; # Truncate description for log
            status = results.get('status')
            if status != 'Processed' and status is not None: logging.info(f"{site_id:<14} | {description:<45} | {status:<55}"); continue

            # Log Annual Trend Summary
            trend_data = results.get('annual_trend_results', {}); mk_data = trend_data.get('mann_kendall')
            if mk_data:
                trend = mk_data.get('trend', 'N/A'); h = mk_data.get('h', None); p = mk_data.get('p', np.nan); slope = mk_data.get('slope', np.nan)
                sig_str = str(h) if h is not None else 'N/A'; logging.info(f"{site_id:<14} | {description:<45} | {'Annual Mean':<14} | {trend:<14} | {sig_str:<6} | {p:<8.4f} | {slope:<10.2f}")
            elif results.get('annual_means_calculated', False): logging.info(f"{site_id:<14} | {description:<45} | {'Annual Mean':<14} | {'Trend N/A':<44}")
            else: logging.info(f"{site_id:<14} | {description:<45} | {'Annual Mean':<14} | {'Calc Failed':<44}")

        logging.info("-" * 105); logging.info("--- Significant Monthly Trends (Mann-Kendall, p<0.05) ---")
        any_monthly_trends = False
        for site_id in sorted_site_ids:
            results = all_site_results[site_id]; description = processed_site_details.get(site_id, 'N/A')[:45]; status = results.get('status')
            if status != 'Processed' and status is not None: continue
            monthly_trends = results.get('monthly_trend_results')
            if monthly_trends:
                sig_trends = [];
                for month_key, trend_info in monthly_trends.items():
                    if trend_info and trend_info.get('h'): sig_trends.append(f"{month_key.split('_')[1]}:{trend_info.get('slope',0):.1f}")
                if sig_trends: logging.info(f"  {site_id} ({description}): {', '.join(sig_trends)}"); any_monthly_trends = True
        if not any_monthly_trends: logging.info("  No significant monthly trends found for any processed site.")
        logging.info("--- End Monthly Summary ---")

    logging.info("--- Trend Analysis Script Finished ---")
    logging.info(f"Log file: {log_filename}")
    logging.info(f"Output directory: {output_base_dir}")

# --- Script Entry Point ---
if __name__ == "__main__":
    main()