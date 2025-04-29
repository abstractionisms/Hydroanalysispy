import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
# Import find_peaks for local extrema detection
from scipy.signal import find_peaks
# Import adjustText for non-overlapping annotations
try:
    from adjustText import adjust_text
    adjustText_installed = True
except ImportError:
    adjustText_installed = False
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
# Removed pymannkendall as climate trend analysis is removed
# Removed itertools combinations as correlation analysis is removed
import matplotlib.dates as mdates # For date formatting
# Add back Meteostat imports for precipitation
from meteostat import Point, Daily, Stations

# --- Constants ---
DEFAULT_CONFIG_PATH = 'config2.json' # Single config file
# Defaults used if config file is missing values OR for constants
DEFAULT_PARAM_CD = "00060"
DEFAULT_START_DATE = "2000-01-01" # Still needed for initial fetch range
DEFAULT_END_DATE = "today" # Still needed for initial fetch range
DEFAULT_LOG_FILE = "logs/q_focus_quad.log" # New log file name with logs/
DEFAULT_INVENTORY_FILE = "nwis_inventory_with_latlon.txt" # Default if not in config
DISCHARGE_COL = 'Discharge_cfs'
PRECIP_COL = 'Precip_mm' # Added back precipitation column
PLOT_YEARS = 3 # Number of years for analysis window
# Peak finding parameters (adjust as needed)
PEAK_PROMINENCE_THRESHOLD = 50 # Minimum prominence for local extrema annotation
MAX_LOCAL_EXTREMA_PER_YEAR = 1 # Max number of *additional* local peaks/troughs to annotate per year
HEATMAP_DISCHARGE_BINS = 50 # Number of bins for discharge axis in heatmap
HEATMAP_DOY_BINS = 50 # Number of bins for day-of-year axis in heatmap

# --- Output Paths ---
PLOT_BASE_DIR = 'plots/q_focus_quad' # Base directory for plots

# --- Descriptive Labels for Plots ---
PLOT_LABELS = {
    DISCHARGE_COL: 'Discharge (cfs)',
    PRECIP_COL: 'Precipitation (mm)', # Added back precip label
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
    Filters out non-positive values as they cannot be plotted on a log scale.
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
        skipped_nonpositive_count = 0 # Track non-positive values

        raw_data = [
            (elem.get('dateTime'), elem.text, elem.get('qualifiers', ''))
            for elem in value_elements
        ]

        for ts_str, val_str, qual in raw_data:
            if val_str is not None:
                try:
                    fval = float(val_str)
                    # Ensure value is positive for log scale plots
                    if fval > 0:
                        timestamp = pd.to_datetime(ts_str, utc=True)
                        data.append({'Timestamp': timestamp, DISCHARGE_COL: fval})
                        if 'P' in qual:
                            provisional_count += 1
                    else:
                         skipped_nonpositive_count += 1 # Skip non-positive
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
            logging.info(f"Skipped {skipped_invalid_count} points (invalid value): {site_id}.")
        if skipped_nonpositive_count > 0:
            logging.info(f"Skipped {skipped_nonpositive_count} points (non-positive value): {site_id}.")

        return df_q

    except ET.ParseError as e:
        logging.error(f"Discharge XML parse error (DV): {site_id}: {e}")
    except Exception as e:
        logging.error(f"Unexpected discharge parse error (DV): {site_id}: {e}")
    return None

# --- Added back climate data fetching (for precipitation) ---
def fetch_climate_data(latitude, longitude, start_datetime, end_datetime):
    """Fetches daily climate data (specifically precipitation) using Meteostat."""
    logging.info(f"Fetching Precipitation near Lat={latitude:.4f}, Lon={longitude:.4f} ({start_datetime.date()} to {end_datetime.date()})...")
    try:
        location = Point(latitude, longitude)
        # Ensure start/end are datetime objects for Meteostat
        start_dt = pd.to_datetime(start_datetime)
        end_dt = pd.to_datetime(end_datetime)
        data = Daily(location, start_dt, end_dt)
        data = data.fetch()

        if data.empty:
            logging.warning("Meteostat returned no climate data for this location/period.")
            return None

        # Select, rename, and ensure UTC index
        # Only keep precipitation ('prcp')
        climate_df = data[['prcp']].copy()
        climate_df.columns = [PRECIP_COL]
        if climate_df.index.tz is None:
             climate_df.index = climate_df.index.tz_localize('UTC')
        else:
             climate_df.index = climate_df.index.tz_convert('UTC')

        # Fill missing precipitation values with 0
        climate_df[PRECIP_COL] = climate_df[PRECIP_COL].fillna(0)

        logging.info(f"Precipitation data fetched successfully ({len(climate_df)} rows).")
        return climate_df
    except Exception as e:
        logging.error(f"Error fetching/processing precipitation data: {e}")
        return None

# --- Plotting Helper Functions ---
def _plot_placeholder(ax, message="Plot N/A"):
    """Helper function to display a placeholder message on an axes object."""
    ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=10,
            transform=ax.transAxes, wrap=True,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.5))
    ax.set_xticks([])
    ax.set_yticks([])

# --- MODIFIED FUNCTION ---
def _plot_discharge_timeseries(ax, df_plot_q, df_plot_precip):
    """
    Plots the discharge time series with a LOG Y-AXIS for the specified period.
    Annotates Yearly Max/Min, Last Point, and Significant Local Extrema using adjustText.
    All annotations show Q value, the date (YYYY-MM-DD), and the SUMMED precipitation
    since the PREVIOUS annotation.
    Assumes df_plot_q contains only positive discharge values for the plot window.
    df_plot_precip should contain precipitation data for the same window.
    """
    global adjustText_installed # Use the global flag

    if df_plot_q is None or df_plot_q.empty:
         _plot_placeholder(ax, f"Time Series Plot N/A\n(No Positive Discharge in last {PLOT_YEARS} years)")
         return

    has_precip_data = df_plot_precip is not None and not df_plot_precip.empty and PRECIP_COL in df_plot_precip.columns
    if not has_precip_data:
        logging.warning("Precipitation data missing or invalid for timeseries plot annotations.")

    try:
        q_series = df_plot_q[DISCHARGE_COL] # Use this for analysis/annotations

        color1 = 'tab:blue'
        local_extrema_color = 'darkorange' # Color for local extrema annotations
        ax.set_xlabel('Date')
        ax.set_ylabel(f"{PLOT_LABELS.get(DISCHARGE_COL, DISCHARGE_COL)} (Log Scale)", color=color1)
        line_q, = ax.plot(df_plot_q.index, q_series, color=color1, linewidth=1.5, label='Discharge')
        ax.tick_params(axis='y', labelcolor=color1)
        ax.grid(True, which='both', linestyle='--', alpha=0.6)
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.get_major_formatter().set_scientific(False)
        ax.yaxis.get_major_formatter().set_useOffset(False)

        plot_title = f"Recent Discharge - {PLOT_YEARS} Years (Log Scale)"
        ax.set_title(plot_title)

        # --- Annotation Point Selection Logic (same as before) ---
        potential_annotations = []
        years_in_plot = q_series.index.year.unique()
        for year in years_in_plot:
            q_year = q_series[q_series.index.year == year]
            if not q_year.empty:
                try:
                    idx_max = q_year.idxmax(); val_max = q_year.max()
                    idx_min = q_year.idxmin(); val_min = q_year.min()
                    potential_annotations.append({'idx': idx_max, 'val': val_max, 'type': 'Yearly Max', 'year': year})
                    potential_annotations.append({'idx': idx_min, 'val': val_min, 'type': 'Yearly Min', 'year': year})
                except Exception: pass # Ignore errors finding yearly extrema
        try:
            peak_indices, peak_props = find_peaks(q_series.values, prominence=PEAK_PROMINENCE_THRESHOLD)
            trough_indices, trough_props = find_peaks(-q_series.values, prominence=PEAK_PROMINENCE_THRESHOLD)
            for j, i in enumerate(peak_indices): potential_annotations.append({'idx': q_series.index[i], 'val': q_series.iloc[i], 'prom': peak_props['prominences'][j], 'type': 'Local Peak'})
            for j, i in enumerate(trough_indices): potential_annotations.append({'idx': q_series.index[i], 'val': q_series.iloc[i], 'prom': trough_props['prominences'][j], 'type': 'Local Trough'})
        except Exception as peak_err: logging.error(f"Error during local peak/trough finding: {peak_err}")
        if not q_series.empty: potential_annotations.append({'idx': q_series.index[-1], 'val': q_series.iloc[-1], 'type': 'Last Point'})

        unique_annotations_dict = {ann['idx']: ann for ann in reversed(potential_annotations)}
        final_annotations_list = []
        local_extrema_annotated_count = {year: {'Local Peak': 0, 'Local Trough': 0} for year in years_in_plot}
        yearly_max_min_indices_set = {ann['idx'] for ann in unique_annotations_dict.values() if ann['type'] in ['Yearly Max', 'Yearly Min']}

        for ann in unique_annotations_dict.values():
            ann_type = ann['type']; idx = ann['idx']; year = idx.year
            if ann_type in ['Yearly Max', 'Yearly Min', 'Last Point']:
                final_annotations_list.append(ann)
            elif ann_type in ['Local Peak', 'Local Trough'] and idx not in yearly_max_min_indices_set:
                 if year in local_extrema_annotated_count and local_extrema_annotated_count[year][ann_type] < MAX_LOCAL_EXTREMA_PER_YEAR:
                     final_annotations_list.append(ann); local_extrema_annotated_count[year][ann_type] += 1
        final_annotations_list.sort(key=lambda x: x['idx'])
        # --- End Annotation Point Selection ---


        # --- Create Annotation Texts and Objects for adjustText ---
        texts = [] # List to hold annotation text objects for adjustText
        last_annotation_idx = None # Track the index of the previous annotation

        for i, ann in enumerate(final_annotations_list):
            idx = ann['idx']
            val = ann['val']
            ann_type = ann['type']
            date_str = idx.strftime('%Y-%m-%d') # Format date string

            # Calculate summed precipitation since the last annotation
            precip_sum_str = "(ΣP N/A)"
            if has_precip_data and last_annotation_idx is not None and last_annotation_idx < idx:
                try:
                    # Use original index of df_plot_q for date slicing consistency
                    start_sum_date = df_plot_q.index[df_plot_q.index.get_loc(last_annotation_idx, method='nearest') + 1]
                    precip_period = df_plot_precip.loc[start_sum_date:idx]
                    precip_sum = precip_period[PRECIP_COL].sum(skipna=True)
                    precip_sum_str = f"{precip_sum:.1f} mm ΣP"
                except Exception as sum_err: logging.warning(f"Could not sum precipitation for annotation at {idx} (after {last_annotation_idx}): {sum_err}")
            elif i == 0 and has_precip_data: # Handle first annotation
                 try:
                     start_plot_idx = df_plot_q.index.min()
                     precip_period = df_plot_precip.loc[start_plot_idx:idx]
                     precip_sum = precip_period[PRECIP_COL].sum(skipna=True)
                     precip_sum_str = f"{precip_sum:.1f} mm ΣP*"
                 except Exception as sum_err: logging.warning(f"Could not sum precipitation for first annotation at {idx}: {sum_err}")


            # Determine annotation text and style
            text_content = ""
            color = color1
            fontsize = 7

            if ann_type == 'Yearly Max':
                text_content = f"{ann['year']} Max: {val:.0f}\n{date_str}\n{precip_sum_str}"
            elif ann_type == 'Yearly Min':
                text_content = f"{ann['year']} Min: {val:.0f}\n{date_str}\n{precip_sum_str}"
            elif ann_type == 'Local Peak':
                text_content = f"Local Pk: {val:.0f}\n{date_str}\n{precip_sum_str}"
                color = local_extrema_color
                fontsize = 6
            elif ann_type == 'Local Trough':
                text_content = f"Local Tr: {val:.0f}\n{date_str}\n{precip_sum_str}"
                color = local_extrema_color
                fontsize = 6
            elif ann_type == 'Last Point':
                text_content = f"Last: {val:.0f}\n{date_str}\n{precip_sum_str}"
                fontsize = 8

            # Create the text object for adjustText
            # Store the text object in the list
            # Use log scale for y-position
            texts.append(ax.text(idx, val, text_content, color=color, fontsize=fontsize,
                                 bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.6, ec='none')))

            # Update the index of the last successful annotation
            last_annotation_idx = idx

        # --- Use adjustText to position annotations ---
        if texts and adjustText_installed:
            try:
                # Define standard arrow properties for adjustText
                arrowprops = dict(arrowstyle="->", color='gray', lw=0.5, connectionstyle="arc3,rad=0.1") # Standard arrow
                # Pass the ax.collections for scatter/line objects to avoid text overlapping them
                adjust_text(texts, x=df_plot_q.index, y=q_series.values, # Pass x and y data explicitly for adjustText
                            ax=ax, arrowprops=arrowprops,
                            # objects=[line_q], # Include plot lines/scatter as objects to avoid
                            # Example tuning parameters (uncomment and adjust as needed):
                            # force_points=(0.1, 0.2), # Increase repulsion from points
                            # force_text=(0.2, 0.4),   # Increase repulsion between texts
                            # expand_points=(1.1, 1.1) # Expand space around points
                            )
            except Exception as adj_err:
                 logging.error(f"Error during adjustText execution: {adj_err}")
                 # Fallback: Keep the manually placed texts if adjustText fails
        elif not adjustText_installed:
             logging.error("The 'adjustText' library is not installed. Annotations may overlap. Run 'pip install adjustText'")
             # Texts are already added to the plot, they just might overlap

        # Format x-axis dates
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    except Exception as e:
        logging.error(f"Time series plot error: {e}")
        _plot_placeholder(ax, "Error plotting Log Time Series")

def _plot_discharge_heatmap(ax, df_plot_q):
    """
    Plots a heatmap showing density of discharge values vs. day of year
    for the specified period (df_plot_q).
    Uses log scale for discharge axis.
    """
    if df_plot_q is None or df_plot_q.empty:
        _plot_placeholder(ax, "Discharge Heatmap N/A\n(No Data)")
        return

    try:
        # Prepare data
        discharge_vals = df_plot_q[DISCHARGE_COL].values
        day_of_year = df_plot_q.index.dayofyear.values

        # Use logarithmic bins for discharge
        min_q_val = discharge_vals.min()
        max_q_val = discharge_vals.max()
        # Handle edge case where min/max are the same or very close
        if min_q_val <= 0 or max_q_val <= min_q_val:
             logging.warning("Cannot create log bins for heatmap due to non-positive or uniform discharge values.")
             _plot_placeholder(ax, "Discharge Heatmap N/A\n(Invalid Data Range for Log Scale)")
             return

        # Ensure min_q is derived from a positive value for log
        min_q = np.floor(np.log10(max(min_q_val, 1e-6))) # Use a small value if min_q_val is zero or negative
        max_q = np.ceil(np.log10(max_q_val))

        # Ensure min_q < max_q for bin creation
        if min_q >= max_q:
             max_q = min_q + 1 # Add arbitrary difference if they are too close

        log_bins_q = np.logspace(min_q, max_q, HEATMAP_DISCHARGE_BINS)

        # Create 2D histogram
        counts, _, _, im = ax.hist2d(
            day_of_year,
            discharge_vals,
            bins=[HEATMAP_DOY_BINS, log_bins_q],
            cmap='viridis',
            cmin=1
        )

        ax.set_yscale('log')
        ax.set_title(f'Discharge Density vs. Day of Year ({PLOT_YEARS} yrs)')
        ax.set_xlabel('Day of Year')
        ax.set_ylabel(f"{PLOT_LABELS[DISCHARGE_COL]} (Log Scale)")
        ax.set_xlim(1, 366)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Number of Days')

        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.get_major_formatter().set_scientific(False)
        ax.yaxis.get_major_formatter().set_useOffset(False)

    except Exception as e:
        logging.error(f"Discharge heatmap plot error: {e}")
        _plot_placeholder(ax, "Error plotting Discharge Heatmap")


def _plot_flow_duration(ax, df_plot_q):
    """
    Plots a flow duration curve for the specified period (df_plot_q).
    Uses log scale for discharge axis and probability scale for y-axis.
    """
    if df_plot_q is None or df_plot_q.empty:
        _plot_placeholder(ax, "Flow Duration Curve N/A\n(No Data)")
        return

    try:
        # Prepare data
        flows = df_plot_q[DISCHARGE_COL].sort_values(ascending=False).values
        n = len(flows)
        exceedance_prob = (np.arange(1.0, n + 1) / (n + 1)) * 100 # Plotting position

        # Plotting
        ax.plot(flows, exceedance_prob, marker='.', linestyle='-', markersize=2, color='tab:green')

        # Formatting
        ax.set_title(f'Flow Duration Curve ({PLOT_YEARS} yrs)')
        ax.set_xlabel(f"{PLOT_LABELS[DISCHARGE_COL]} (Log Scale)")
        ax.set_ylabel('Exceedance Probability (%)')
        ax.set_yscale('log') # Use log scale for probability
        ax.set_xscale('log') # Use log scale for discharge
        ax.grid(True, which='both', linestyle='--', alpha=0.6)

        # Format axes for better readability
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.get_major_formatter().set_scientific(False)
        ax.xaxis.get_major_formatter().set_useOffset(False)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
        # Set explicit ticks for probability y-axis
        yticks = [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
        ax.set_yticks(yticks)


    except Exception as e:
        logging.error(f"Flow duration curve plot error: {e}")
        _plot_placeholder(ax, "Error plotting Flow Duration Curve")

def _plot_monthly_boxplots(ax, df_plot_q):
    """
    Plots monthly boxplots of discharge for the specified period (df_plot_q).
    Uses log scale for discharge axis.
    """
    if df_plot_q is None or df_plot_q.empty:
        _plot_placeholder(ax, "Monthly Boxplots N/A\n(No Data)")
        return

    try:
        # Prepare data
        # Ensure 'Month' column exists, create if not
        # Create a temporary copy for plotting to avoid modifying the original filtered df
        df_plot_q_copy = df_plot_q.copy()
        df_plot_q_copy['Month'] = df_plot_q_copy.index.month


        month_order = range(1, 13)
        month_labels = [calendar.month_abbr[i] for i in month_order]

        # Create boxplot
        sns.boxplot(
            x='Month',
            y=DISCHARGE_COL,
            data=df_plot_q_copy,
            ax=ax,
            order=month_order,
            showfliers=False,
            palette='viridis'
        )

        # Formatting
        ax.set_title(f'Monthly Discharge Distribution ({PLOT_YEARS} yrs)')
        ax.set_xlabel('Month')
        ax.set_ylabel(f"{PLOT_LABELS[DISCHARGE_COL]} (Log Scale)")
        ax.set_xticklabels(month_labels)
        ax.set_yscale('log')
        ax.grid(True, axis='y', which='both', linestyle='--', alpha=0.6)

        # Format y-axis ticks
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.get_major_formatter().set_scientific(False)
        ax.yaxis.get_major_formatter().set_useOffset(False)

    except Exception as e:
        logging.error(f"Monthly boxplot error: {e}")
        _plot_placeholder(ax, "Error plotting Monthly Boxplots")


# --- Main Plotting Function ---
# Modified to accept climate data (for precip) and PLOT_BASE_DIR
def plot_discharge_details(df_q, df_climate, site_id, description, plot_base_dir):
    """
    Generates a multi-panel plot focusing on discharge details for the last PLOT_YEARS.
    Requires df_climate for precipitation data used in timeseries annotations.
    Saves plots to plot_base_dir/site_id/.
    """
    logging.info(f"Generating discharge focus plots for site {site_id}...")

    if df_q is None or df_q.empty:
        logging.error(f"No discharge data available for plotting: {site_id}.")
        return

    # --- Filter data for the plot window ---
    df_plot_q = None
    df_plot_precip = None
    try:
        end_plot_date = df_q.index.max()
        start_plot_date = end_plot_date - pd.DateOffset(years=PLOT_YEARS)
        # Filter discharge for the plot period AND ensure positive values
        df_plot_q_filtered = df_q.loc[start_plot_date:end_plot_date].copy()
        df_plot_q = df_plot_q_filtered[df_plot_q_filtered[DISCHARGE_COL] > 0]

        if df_plot_q.empty:
             logging.warning(f"No positive discharge data in the last {PLOT_YEARS} years for plotting: {site_id}.")
             # Create a figure with just placeholders
             fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
             axes = axes.flatten()
             plot_period_str = f"Attempted: {start_plot_date.strftime('%Y-%m-%d')} to {end_plot_date.strftime('%Y-%m-%d')}"
             wrapped_title = "\n".join(textwrap.wrap(f"Site {site_id}: {description}", width=70))
             fig.suptitle(f"Discharge Analysis ({plot_period_str})\n{wrapped_title}", fontsize=16)
             for i in range(4):
                 _plot_placeholder(axes[i], f"Plot N/A\n(No Positive Discharge in last {PLOT_YEARS} years)")
             # Save placeholder plot
             site_plot_dir = os.path.join(plot_base_dir, site_id) # Construct path using plot_base_dir
             os.makedirs(site_plot_dir, exist_ok=True)
             plot_filename = f"USGS_{site_id}_discharge_focus.png"
             plot_path = os.path.join(site_plot_dir, plot_filename) # Save to site_plot_dir
             plt.savefig(plot_path, dpi=150, bbox_inches='tight')
             plt.close(fig)
             logging.info(f"Placeholder plot saved due to no positive discharge data: {plot_path}")
             return # Exit plotting function

        # Filter precipitation data for the same window if available
        if df_climate is not None and not df_climate.empty and PRECIP_COL in df_climate.columns:
            # Ensure index alignment before filtering
            df_climate_aligned, df_q_aligned = df_climate.align(df_q, join='right', axis=0) # Align climate to discharge dates
            df_plot_precip = df_climate_aligned.loc[start_plot_date:end_plot_date].copy()
            # Ensure PRECIP_COL exists after filtering and alignment
            if PRECIP_COL not in df_plot_precip.columns:
                 df_plot_precip = None # Treat as missing if column disappears
            elif df_plot_precip.empty:
                 df_plot_precip = None # Treat as missing if empty after filtering

    except Exception as filter_err:
        logging.error(f"Error filtering data for plot window: {site_id} - {filter_err}")
        return

    # --- Create Figure and Axes (2x2 grid) ---
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
        axes = axes.flatten() # Flatten the 2x2 array for easy indexing

        # Determine plot period string from actual filtered data
        actual_start_plot = df_plot_q.index.min().strftime('%Y-%m-%d')
        actual_end_plot = df_plot_q.index.max().strftime('%Y-%m-%d')
        plot_period_str = f"{actual_start_plot} to {actual_end_plot}"

        wrapped_title = "\n".join(textwrap.wrap(f"Site {site_id}: {description}", width=70))
        fig.suptitle(f"Discharge Analysis ({plot_period_str})\n{wrapped_title}", fontsize=16)

        # --- Call Plotting Helpers ---
        # Plot 1: Detailed Time Series (Log Scale) - Pass precip data
        _plot_discharge_timeseries(axes[0], df_plot_q, df_plot_precip)

        # Plot 2: Discharge Density Heatmap
        _plot_discharge_heatmap(axes[1], df_plot_q)

        # Plot 3: Flow Duration Curve
        _plot_flow_duration(axes[2], df_plot_q)

        # Plot 4: Monthly Boxplots
        _plot_monthly_boxplots(axes[3], df_plot_q)

        # --- Save Plot ---
        site_plot_dir = os.path.join(plot_base_dir, site_id) # Construct path using plot_base_dir
        os.makedirs(site_plot_dir, exist_ok=True)
        plot_filename = f"USGS_{site_id}_discharge_focus.png" # Incremented version
        plot_path = os.path.join(site_plot_dir, plot_filename) # Save to site_plot_dir
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Discharge focus plot saved: {plot_path}")

    except Exception as e:
        logging.error(f"Error generating discharge focus plot figure: {site_id}: {e}")
        plt.close() # Ensure figure is closed even if error occurs


# --- Main Orchestration ---
# Modified to fetch precip and pass it along and use plot_base_dir
def process_site(site_config, analysis_params, plot_base_dir):
    """Processes a single site based on its configuration."""
    site_id = site_config.get("site_id")
    param_cd = site_config.get("param_cd", analysis_params.get("param_cd", DEFAULT_PARAM_CD))
    start_date_fetch_str = site_config.get("start_date", analysis_params.get("start_date", DEFAULT_START_DATE))
    end_date_fetch_str = site_config.get("end_date", analysis_params.get("end_date", DEFAULT_END_DATE))
    description = site_config.get("description", f"Site {site_id}")
    # Need Lat/Lon for precipitation fetching
    latitude_raw = site_config.get("latitude")
    longitude_raw = site_config.get("longitude")

    if not site_config.get("enabled", False):
        logging.info(f"Skipping disabled site: {site_id or 'Missing ID'}")
        return None

    # Validation requires lat/lon now for precip
    if not all([site_id, param_cd, start_date_fetch_str, end_date_fetch_str, latitude_raw is not None, longitude_raw is not None]):
        logging.warning(f"Skipping {site_id or 'N/A'}: Missing required config data (site_id, param_cd, dates, latitude, longitude).")
        return {site_id: {'status': 'Error (Config Missing)'}}

    # Validate Lat/Lon
    try:
        latitude = float(latitude_raw)
        longitude = float(longitude_raw)
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise ValueError("Lat/Lon out of valid range")
    except (ValueError, TypeError):
        logging.warning(f"Skipping {site_id}: Invalid latitude ('{latitude_raw}') or longitude ('{longitude_raw}') in config.")
        return {site_id: {'status': 'Error (Invalid Coords)'}}


    # Validate dates
    try:
        end_date_fetch = datetime.now() if end_date_fetch_str.lower() == 'today' else datetime.strptime(end_date_fetch_str, '%Y-%m-%d')
        start_date_fetch = datetime.strptime(start_date_fetch_str, '%Y-%m-%d')
        if start_date_fetch >= end_date_fetch:
            logging.warning(f"Skipping {site_id}: Start date ({start_date_fetch_str}) not before end date ({end_date_fetch_str}).")
            return {site_id: {'status': 'Error (Invalid Dates)'}}
        # Ensure fetch period covers the plot window
        min_fetch_start_date = end_date_fetch - pd.DateOffset(years=PLOT_YEARS) - pd.DateOffset(days=1)
        if start_date_fetch > min_fetch_start_date:
            logging.info(f"Adjusting fetch start date for {site_id} to ensure {PLOT_YEARS}-year plot window coverage.")
            start_date_fetch = min_fetch_start_date

        start_date_nwis = start_date_fetch.strftime('%Y-%m-%d')
        end_date_nwis = end_date_fetch.strftime('%Y-%m-%d')

    except ValueError as e:
        logging.error(f"Skipping {site_id}: Invalid date format - {e}.")
        return {site_id: {'status': 'Error (Date Format)'}}

    logging.info(f"--- Processing Site: {site_id} ({description}) ---")

    # Removed creation of site_output_dir here as it's done in plot_discharge_details
    # site_output_dir = os.path.join(output_base_dir, site_id)
    # try:
    #     os.makedirs(site_output_dir, exist_ok=True)
    # except Exception as e:
    #     logging.error(f"Could not create output dir for {site_id}: {e}.")


    site_results = {'status': 'Started'}
    df_q = None
    df_climate = None # For precip

    # Fetch discharge data
    discharge_wml = fetch_waterml_data(site_id, param_cd, start_date_nwis, end_date_nwis)
    if discharge_wml:
        df_q = parse_waterml(discharge_wml, site_id)

    # Fetch climate (precipitation) data
    # Use the original fetch start/end datetimes for consistency
    df_climate = fetch_climate_data(latitude, longitude, start_date_fetch, end_date_fetch)

    # Plotting - Pass both discharge and climate dataframes and plot_base_dir
    if df_q is not None and not df_q.empty:
        # Pass df_climate even if it's None, plotting function handles it
        plot_discharge_details(df_q, df_climate, site_id, description, plot_base_dir) # Pass plot_base_dir
        site_results['status'] = 'Processed'
        if df_climate is None:
             site_results['warnings'] = ['Precipitation data missing, annotations lack summed values.'] # Updated warning
    else:
        logging.error(f"No valid discharge data obtained for {site_id}. Plotting skipped.")
        site_results['status'] = 'Error (Discharge Missing/Invalid)'
        # Even if discharge is missing, attempt to save a placeholder plot if needed
        plot_discharge_details(df_q, df_climate, site_id, description, plot_base_dir)


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
    logging.info("--- Starting Discharge Focus Plotting Script") # Updated log message

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
        logging.info("--- Discharge Focus Script Finished (No Sites) ---")
        return

    all_site_results = {}
    for site_config in sites_to_process:
        # Pass plot_base_dir to process_site
        result = process_site(site_config, analysis_params, plot_base_dir)
        if result:
             all_site_results.update(result)

    logging.info("--- Discharge Focus Script Finished ---")


# --- Script Entry Point ---
if __name__ == "__main__":
    main()