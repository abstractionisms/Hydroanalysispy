# Save as: local_multi_site_trend_comparison.py
import argparse
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import linregress
from datetime import date
import dataretrieval.nwis as nwis
import traceback # For printing detailed errors

# --- Data Fetching Function ---
def fetch_data_for_sites(
    sites_config_list,
    data_cache_dir=None
):
    """Fetches data for multiple sites using dataretrieval."""
    all_sites_data = {}
    if data_cache_dir:
        os.makedirs(data_cache_dir, exist_ok=True)
    
    today_date_str = date.today().strftime('%Y-%m-%d')

    for site_config_entry in sites_config_list:
        if not site_config_entry.get("enabled", True): # Skip disabled sites
            print(f"Skipping disabled site: {site_config_entry.get('site_id', 'Unknown ID')}")
            continue
            
        site_id_from_config = site_config_entry["site_id"]
        # Ensure USGS prefix for dataretrieval
        site_id = f"USGS:{site_id_from_config}" if not str(site_id_from_config).startswith("USGS:") else str(site_id_from_config)
        
        parameter_code = site_config_entry["param_cd"]
        start_date = site_config_entry["start_date"]
        # Handle "today" for end date
        end_date = today_date_str if site_config_entry["end_date"].lower() == "today" else site_config_entry["end_date"]
        
        raw_csv_path = None
        df_site_raw = None # Initialize
        safe_site_id_filename = site_id.replace(':', '_') # Sanitize for filename

        # --- Caching Logic ---
        if data_cache_dir:
            raw_csv_filename = f"{safe_site_id_filename}_{parameter_code}_{start_date}_to_{end_date}_raw.csv"
            raw_csv_path = os.path.join(data_cache_dir, raw_csv_filename)
            if os.path.exists(raw_csv_path):
                print(f"Loading cached raw data for site {site_id} from {raw_csv_path}")
                try:
                    df_site_raw = pd.read_csv(raw_csv_path, index_col=0, parse_dates=True)
                except Exception as e_cache:
                    print(f"Error loading cached data for {site_id}: {e_cache}. Will re-fetch.")
                    df_site_raw = None
        
        # --- Fetching Logic ---
        if df_site_raw is None:
            print(f"Fetching data for site: {site_id}, Param: {parameter_code}, Period: {start_date} to {end_date}")
            try:
                # Use get_dv for daily values. Add error handling.
                df_site_raw_temp, meta_site = nwis.get_dv(
                    sites=site_id,
                    parameterCd=parameter_code,
                    start=start_date,
                    end=end_date,
                )
                df_site_raw = df_site_raw_temp # Assign after successful fetch
                # Save to cache if fetched successfully and caching enabled
                if data_cache_dir and df_site_raw is not None and not df_site_raw.empty:
                    try:
                        df_site_raw.to_csv(raw_csv_path)
                        print(f"Saved raw data for {site_id} to cache: {raw_csv_path}")
                    except Exception as e_save:
                         print(f"Warning: Failed to save cache for {site_id}: {e_save}")

            except Exception as e_fetch:
                print(f"ERROR fetching data for site {site_id}: {e_fetch}")
                # traceback.print_exc() # Uncomment for more detailed fetch errors
                continue # Skip to next site if fetching fails

        # --- Processing Logic ---
        if df_site_raw is not None and not df_site_raw.empty:
            data_col_name = None
            target_col_suffix = "_Mean" # Prioritize Mean for daily values
            # Find the data column (e.g., "00060_Mean")
            for col in df_site_raw.columns:
                if str(parameter_code) in col and target_col_suffix in col:
                    data_col_name = col; break
            if not data_col_name: # Fallback 1: parameter code without suffix
                 for col in df_site_raw.columns:
                    if str(parameter_code) in col:
                        data_col_name = col; break
            if not data_col_name and len(df_site_raw.columns) > 0: # Fallback 2: first numeric non-ID column
                for col_try in df_site_raw.columns:
                     # Check if column is numeric-like and not an ID column
                     if pd.api.types.is_numeric_dtype(df_site_raw[col_try]) and col_try not in ['site_no', 'agency_cd']:
                         data_col_name = col_try
                         print(f"Warning: Using fallback data column '{data_col_name}' for site {site_id}")
                         break
            
            # Process if a data column was identified
            if data_col_name:
                df_site = df_site_raw[[data_col_name]].copy()
                df_site.rename(columns={data_col_name: 'value'}, inplace=True)
                # Ensure index is DatetimeIndex
                if not isinstance(df_site.index, pd.DatetimeIndex):
                    df_site.index = pd.to_datetime(df_site.index)
                df_site.index.name = 'datetime'
                # Convert value column to numeric, coercing errors
                df_site['value'] = pd.to_numeric(df_site['value'], errors='coerce')
                # Drop rows where conversion failed or value is missing
                df_site.dropna(subset=['value'], inplace=True)
                # Remove duplicate index entries if any, keep first
                df_site = df_site[~df_site.index.duplicated(keep='first')]
                # Ensure chronological order
                df_site = df_site.sort_index()

                if not df_site.empty:
                    all_sites_data[site_id] = df_site
                    print(f"Successfully processed data for site {site_id}, {len(df_site)} records.")
                else:
                    print(f"No valid numeric data after processing for site {site_id}")
            else:
                print(f"Could not identify data column for parameter {parameter_code} for site {site_id}. Available columns: {list(df_site_raw.columns)}")
        elif df_site_raw is None:
             print(f"No data fetched for site {site_id} and param {parameter_code}.")
        # else: (df_site_raw is empty) - already handled by check
    return all_sites_data

# --- Trend Calculation Function ---
def get_trend_from_series(series_data_param, time_col='datetime', value_col='value'):
    """Estimates a linear trend from a time series using linear regression."""
    if series_data_param is None or series_data_param.empty or value_col not in series_data_param.columns:
        return None, None, None, None
    
    data_to_use = None
    # Check if time_col exists and set as index if needed
    if time_col in series_data_param.columns:
        # Ensure time_col is datetime before setting index
        series_data_param[time_col] = pd.to_datetime(series_data_param[time_col])
        temp_df = series_data_param.set_index(time_col)
        if value_col in temp_df:
            data_to_use = temp_df[value_col].dropna()
    elif series_data_param.index.name == time_col:
        # Ensure index is datetime
        if not isinstance(series_data_param.index, pd.DatetimeIndex):
             series_data_param.index = pd.to_datetime(series_data_param.index)
        data_to_use = series_data_param[value_col].dropna()
    else: # Fallback assuming index is datetime
         if isinstance(series_data_param.index, pd.DatetimeIndex) and value_col in series_data_param.columns:
            data_to_use = series_data_param[value_col].dropna()

    if data_to_use is None or data_to_use.empty:
        # print(f"Could not prepare data for trend calculation from input. Ensure '{time_col}' and '{value_col}' are correct.")
        return None, None, None, None

    # Need sufficient points for a meaningful trend
    if len(data_to_use) < 20:
        print(f"Not enough data points ({len(data_to_use)}) to calculate reliable trend.")
        return None, None, None, None

    # Create numeric x-axis (days since start)
    x_start_date = data_to_use.index.min()
    x_numeric = (data_to_use.index - x_start_date).days.astype(float)
    y_values = data_to_use.values

    try:
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = linregress(x_numeric, y_values)
        # Return slope (units per day), p_value, intercept, and the start date for context
        return slope, p_value, intercept, x_start_date
    except ValueError as ve:
        print(f"Linear regression failed: {ve}. Check for NaNs or non-numeric data.")
        return None, None, None, None
    except Exception as e:
        print(f"An unexpected error occurred in linear regression: {e}")
        return None, None, None, None

# --- Main Execution Logic ---
def main(config_file_path, output_base_dir_arg="output_data/local_multi_site_trends", cache_raw_data_arg=False):
    print(f"Starting local trend analysis using config: {config_file_path}")
    try:
        with open(config_file_path, 'r') as f:
            config_data_main = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Config file not found at {config_file_path}")
        return
    except json.JSONDecodeError:
        print(f"ERROR: Could not parse JSON in config file: {config_file_path}")
        return

    if "sites_to_process" not in config_data_main or not config_data_main["sites_to_process"]:
        print(f"No 'sites_to_process' list found or it's empty in {config_file_path}.")
        return

    sites_info_list_main = config_data_main["sites_to_process"]
    # Filter for enabled sites BEFORE passing to fetcher
    enabled_sites_info = [s for s in sites_info_list_main if s.get("enabled", True)]
    if not enabled_sites_info:
        print("No enabled sites found in the config file.")
        return

    os.makedirs(output_base_dir_arg, exist_ok=True)
    raw_data_cache_dir = os.path.join(output_base_dir_arg, "raw_data_cache") if cache_raw_data_arg else None

    # Fetch data for all enabled sites
    all_sites_data_val = fetch_data_for_sites(enabled_sites_info, data_cache_dir=raw_data_cache_dir)

    site_trends_summary_dict = {}
    num_sites = len(all_sites_data_val)
    if num_sites == 0:
        print("No site data was successfully fetched or processed. Exiting trend analysis.")
        return

    # --- Plotting Setup ---
    ncols = min(3, num_sites)
    nrows = (num_sites + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 7, nrows * 5), squeeze=False, sharex=False)
    axes_flat = axes.flatten()
    plot_idx = 0

    # --- Trend Calculation and Plotting Loop ---
    for site_id_key, df_site_val in all_sites_data_val.items():
        if df_site_val.empty:
            print(f"Skipping empty dataframe for site: {site_id_key}")
            continue

        print(f"\n--- Processing trends for site: {site_id_key} (using observed data) ---")
        # Pass the DataFrame with datetime index directly
        slope, p_value, intercept, x_start_date_val = get_trend_from_series(df_site_val, time_col='datetime', value_col='value')

        if slope is not None and intercept is not None and x_start_date_val is not None:
            site_trends_summary_dict[site_id_key] = {
                'slope_per_day': slope,
                'p_value': p_value,
                'intercept': intercept,
                'trend_source': "observed_data",
                'trend_start_date': x_start_date_val.strftime('%Y-%m-%d')
            }
            print(f"Site {site_id_key} trend (observed): Slope={slope:.4f} (units/day), P-value={p_value:.4f}")

            # Plotting
            if plot_idx < len(axes_flat):
                ax = axes_flat[plot_idx]
                ax.plot(df_site_val.index, df_site_val['value'], alpha=0.6, label=f"Observed", linewidth=1)

                # Create trend line for plotting
                x_numeric_plot = (df_site_val.index - x_start_date_val).days.astype(float)
                trend_line_plot = intercept + slope * x_numeric_plot
                ax.plot(df_site_val.index, trend_line_plot, color='red', linestyle='--', linewidth=2,
                        label=f"Trend (Slope: {slope:.2e}, p={p_value:.2f})")

                ax.set_title(f"{site_id_key}", fontsize=10)
                ax.legend(fontsize=8)
                ax.tick_params(axis='x', rotation=30, labelsize=8)
                ax.tick_params(axis='y', labelsize=8)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m')) # Format x-axis dates
                ax.xaxis.set_major_locator(plt.MaxNLocator(6)) # Limit number of x-ticks
                if plot_idx % ncols == 0: # Only add Y label to first column
                    ax.set_ylabel("Value (e.g., CFS)", fontsize=9)
                plot_idx += 1
            else:
                 print(f"Warning: Ran out of plot axes for site {site_id_key}")
        else:
            print(f"Could not calculate trend for site {site_id_key}.")

    # --- Finalize Plot and Save ---
    # Hide any unused subplots
    for i in range(plot_idx, len(axes_flat)):
        fig.delaxes(axes_flat[i])

    config_basename = os.path.splitext(os.path.basename(config_file_path))[0]
    fig.suptitle(f"Observed Data and Linear Trends for Config: {config_basename}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for title and labels
    trend_plot_path = os.path.join(output_base_dir_arg, f"{config_basename}_all_sites_trend_plots_observed.png")
    try:
        plt.savefig(trend_plot_path)
        print(f"\nIndividual trend plots (observed) saved to: {trend_plot_path}")
    except Exception as e_plot:
        print(f"ERROR saving trend plot: {e_plot}")
    plt.close(fig)

    # --- Save Summary ---
    if site_trends_summary_dict:
        trend_summary_df = pd.DataFrame.from_dict(site_trends_summary_dict, orient='index')
        print("\nTrend Summary (Slope in units/day based on observed data):")
        print(trend_summary_df.to_string()) # Print full dataframe
        summary_csv_path = os.path.join(output_base_dir_arg, f"{config_basename}_trend_summary_observed.csv")
        try:
            trend_summary_df.to_csv(summary_csv_path)
            print(f"Trend summary CSV (observed) saved to: {summary_csv_path}")
        except Exception as e_csv:
            print(f"ERROR saving trend summary CSV: {e_csv}")
    else:
        print("\nNo trends were successfully calculated for any site.")

# --- Configuration and Main Call ---
if __name__ == "__main__":
    # --- EDIT THIS LINE TO CHOOSE YOUR CONFIG FILE ---
    CONFIG_FILE_TO_USE = "config.json"  # Or change to "config2.json" 
    # --- EDIT THIS LINE TO SET OUTPUT DIRECTORY ---
    OUTPUT_DIRECTORY = "output_data/local_run_results" # Or customize as needed
    # --- SET WHETHER TO CACHE RAW DATA ---
    USE_RAW_DATA_CACHE = True # Set to False if you don't want caching

    # --- NO NEED TO EDIT BELOW THIS LINE ---
    # Construct default output dir based on script name if not customized above
    if OUTPUT_DIRECTORY == "output_data/local_run_results":
         script_name = os.path.splitext(os.path.basename(__file__))[0] # Gets script name without .py
         config_name = os.path.splitext(os.path.basename(CONFIG_FILE_TO_USE))[0] # Gets config name without .json
         OUTPUT_DIRECTORY = os.path.join("output_data", f"{script_name}_{config_name}")

    # Make sure the config file exists relative to the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(script_dir, CONFIG_FILE_TO_USE)

    if not os.path.exists(config_file_path):
        print(f"ERROR: Config file specified does not exist: {config_file_path}")
        print("Please make sure CONFIG_FILE_TO_USE is set correctly in the script.")
    else:
         # Call the main function with the hardcoded values
         main(config_file_path, OUTPUT_DIRECTORY, USE_RAW_DATA_CACHE)