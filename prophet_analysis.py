# Save as: prophet_analysis.py
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date, timedelta
import dataretrieval.nwis as nwis
import traceback
import sys
import seaborn as sns # Make sure seaborn is imported

# --- IMPORTANT: Install Prophet first: pip install prophet ---
# --- Install seaborn if needed: pip install seaborn ---
try:
    from prophet import Prophet
    from prophet.plot import plot_components
except ImportError:
    print("ERROR: Prophet library not found.")
    print("Please install it: pip install prophet")
    sys.exit(1)

# --- Data Fetching Function ---
def fetch_data_for_sites(sites_config_list, data_cache_dir=None):
    """Fetches data for multiple sites using dataretrieval."""
    all_sites_data = {}
    if data_cache_dir: os.makedirs(data_cache_dir, exist_ok=True)
    today_date_str = date.today().strftime('%Y-%m-%d')
    for site_config_entry in sites_config_list:
        if not site_config_entry.get("enabled", True): print(f"Skipping disabled: {site_config_entry.get('site_id')}"); continue
        site_id_from_config = site_config_entry["site_id"]; site_id = f"USGS:{site_id_from_config}" if not str(site_id_from_config).startswith("USGS:") else str(site_id_from_config)
        parameter_code = site_config_entry["param_cd"]; start_date = site_config_entry["start_date"]; end_date = today_date_str if site_config_entry["end_date"].lower() == "today" else site_config_entry["end_date"]
        raw_csv_path = None; df_site_raw = None; safe_site_id_filename = site_id.replace(':', '_')
        if data_cache_dir:
            raw_csv_filename = f"{safe_site_id_filename}_{parameter_code}_{start_date}_to_{end_date}_raw.csv"
            raw_csv_path = os.path.join(data_cache_dir, raw_csv_filename)
            if os.path.exists(raw_csv_path):
                print(f"Loading cache: {site_id} from {raw_csv_path}")
                try: df_site_raw = pd.read_csv(raw_csv_path, index_col='datetime', parse_dates=True)
                except Exception as e: print(f"Cache err: {e}. Re-fetching."); df_site_raw = None
        if df_site_raw is None:
            print(f"Fetching: {site_id} P:{parameter_code} D:{start_date} to {end_date}")
            try: df_site_raw_temp, meta = nwis.get_dv(sites=site_id, parameterCd=parameter_code, start=start_date, end=end_date); df_site_raw = df_site_raw_temp
            except Exception as e: print(f"ERROR fetch {site_id}: {e}"); continue
            if data_cache_dir and df_site_raw is not None and not df_site_raw.empty:
                 try:
                      if df_site_raw.index.name is None: df_site_raw.index.name = 'datetime'
                      df_site_raw.to_csv(raw_csv_path); print(f"Saved cache: {site_id}")
                 except Exception as e: print(f"Warn: Save cache fail {site_id}: {e}")
        if df_site_raw is not None and not df_site_raw.empty:
            data_col_name = None; target_col_suffix = "_Mean";
            for col in df_site_raw.columns:
                if str(parameter_code) in col and target_col_suffix in col: data_col_name = col; break
            if not data_col_name:
                 for col in df_site_raw.columns:
                    if str(parameter_code) in col: data_col_name = col; break
            if not data_col_name and len(df_site_raw.columns) > 0:
                 for col_try in df_site_raw.columns:
                     if pd.api.types.is_numeric_dtype(df_site_raw[col_try]) and col_try not in ['site_no', 'agency_cd']: data_col_name = col_try; print(f"Warn: Fallback col '{data_col_name}' for {site_id}"); break
            if data_col_name:
                df_site = df_site_raw[[data_col_name]].copy(); df_site.rename(columns={data_col_name: 'value'}, inplace=True)
                if not isinstance(df_site.index, pd.DatetimeIndex): df_site.index = pd.to_datetime(df_site.index)
                df_site.index.name = 'datetime'; df_site['value'] = pd.to_numeric(df_site['value'], errors='coerce'); df_site.dropna(subset=['value'], inplace=True)
                df_site = df_site[~df_site.index.duplicated(keep='first')]; df_site = df_site.sort_index()
                if not df_site.empty:
                    value_col_for_prophet = 'value'
                    if (df_site['value'] > 1e-9).all():
                        try: df_site['value_log'] = np.log(df_site['value'] + 1e-9); print(f"Applied log transform for {site_id}"); value_col_for_prophet = 'value_log'
                        except Exception as e: print(f"Warn: Log transform failed for {site_id}: {e}. Using original."); df_site['value_log'] = np.nan
                    else: print(f"Warn: Non-positive values in {site_id}. Using original."); df_site['value_log'] = np.nan
                    all_sites_data[site_id] = {'df': df_site, 'value_col': value_col_for_prophet}; print(f"Processed: {site_id}, {len(df_site)} recs.")
                else: print(f"No valid numeric data for {site_id}")
            else: print(f"Could not ID data col for {parameter_code} for {site_id}. Cols: {list(df_site_raw.columns)}")
        elif df_site_raw is not None and df_site_raw.empty: print(f"Empty dataframe after fetch/cache for {site_id}")
        elif df_site_raw is None: print(f"No data fetched for {site_id}")
    return all_sites_data

# --- Prophet Analysis Function ---
def analyze_with_prophet(df_input, value_col, site_id, output_dir):
    """Performs Prophet analysis, saves plots and anomaly data."""

    # Prepare data for Prophet (requires 'ds' and 'y' columns)
    # Ensure input df index is datetime before resetting
    if not isinstance(df_input.index, pd.DatetimeIndex):
         df_input.index = pd.to_datetime(df_input.index)
    df_prophet = df_input.reset_index().rename(columns={'datetime': 'ds', value_col: 'y'})

    # --- CRITICAL FIX for Timezone Error ---
    # Ensure 'ds' column is datetime type first
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    # Check if 'ds' column has timezone info and remove it
    if df_prophet['ds'].dt.tz is not None:
        print(f"Removing timezone information from 'ds' column for site {site_id}")
        df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
    # --- End Timezone Fix ---

    # Drop rows where target 'y' is NaN (important after potential log transform issues)
    df_prophet.dropna(subset=['y'], inplace=True)

    if len(df_prophet) < 10: # Need minimum data for Prophet
        print(f"Skipping Prophet for {site_id}: Insufficient valid data points ({len(df_prophet)}) after NaN drop.")
        return None, None

    print(f"Running Prophet for {site_id} on column '{value_col}'...")

    # Initialize and Fit Prophet Model
    model = Prophet( yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)

    try:
        model.fit(df_prophet[['ds', 'y']]) # Pass only ds and y columns
    except Exception as e_fit:
        print(f"ERROR fitting Prophet model for {site_id}: {e_fit}")
        traceback.print_exc()
        return None, None

    # Make Predictions (In-sample forecast)
    try:
        # Ensure the dataframe passed to predict only has 'ds' column
        future_df = df_prophet[['ds']]
        forecast = model.predict(future_df)
    except Exception as e_predict:
        print(f"ERROR predicting with Prophet model for {site_id}: {e_predict}")
        traceback.print_exc()
        return None, None

    # Identify Anomalies
    # Merge forecast back with original prepared data (df_prophet)
    results = pd.merge(df_prophet, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend', 'yearly']], on='ds', how='left')
    results['is_anomaly'] = (results['y'] < results['yhat_lower']) | (results['y'] > results['yhat_upper'])

    # Create Plots
    safe_site_id = site_id.replace(":", "_")
    plot_subdir = os.path.join(output_dir, safe_site_id)
    os.makedirs(plot_subdir, exist_ok=True)

    try:
        # Plot 1: Forecast Plot
        fig_forecast = model.plot(forecast) # Returns matplotlib figure
        anomalies_df = results[results['is_anomaly']]
        # Plot anomalies using 'y' values (which might be log-transformed)
        plt.scatter(anomalies_df['ds'].dt.to_pydatetime(), anomalies_df['y'], color='red', s=15, label='Anomaly', zorder=5, alpha=0.7)
        plot_title = f"Prophet Forecast & Anomalies: {site_id}\n(Modeled: {value_col})"
        plt.title(plot_title, fontsize=12)
        plt.xlabel("Date", fontsize=10)
        plt.ylabel(f"Value ({value_col})", fontsize=10) # Label axis with modeled value (original or log)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        handles, labels = plt.gca().get_legend_handles_labels()
        from matplotlib.lines import Line2D
        anomaly_proxy = Line2D([0], [0], marker='o', color='w', label='Anomaly', markerfacecolor='red', markersize=4)
        if not any('Anomaly' in str(l) for l in labels) and not anomalies_df.empty:
            handles.append(anomaly_proxy)
        plt.legend(handles=handles, fontsize=8, loc='best')
        plt.grid(True, alpha=0.5)
        forecast_plot_path = os.path.join(plot_subdir, f"{safe_site_id}_prophet_forecast_anomalies.png")
        fig_forecast.savefig(forecast_plot_path, dpi=150, bbox_inches='tight')
        print(f"  Saved forecast plot: {forecast_plot_path}")
        plt.close(fig_forecast)

        # Plot 2: Components Plot
        fig_components = model.plot_components(forecast)
        plt.suptitle(f"Prophet Components: {site_id} (Modeled: {value_col})", y=1.02, fontsize=12)
        plt.tight_layout() # Let tight_layout use its default adjustments
        components_plot_path = os.path.join(plot_subdir, f"{safe_site_id}_prophet_components.png")
        fig_components.savefig(components_plot_path, dpi=150, bbox_inches='tight') # Add bbox_inches here too for robustness
        print(f"  Saved components plot: {components_plot_path}")
        plt.close(fig_components)

    except Exception as e_plot:
        print(f"ERROR generating Prophet plots for {site_id}: {e_plot}")
        traceback.print_exc()

    results.set_index('ds', inplace=True)
    # Add original value back if log transform was used
    if value_col == 'value_log' and 'value' in df_input.columns:
        try:
             # Ensure df_input index is datetime before joining
             if not isinstance(df_input.index, pd.DatetimeIndex):
                 df_input.index = pd.to_datetime(df_input.index)
             # Join based on index (which should be datetime 'ds' in results and 'datetime' in df_input)
             results = results.join(df_input[['value']], how='left')
        except Exception as e_join:
             print(f"Warning: Could not join original 'value' column back for {site_id}: {e_join}")
             results['value'] = np.nan # Add column even if join fails

    return results, model

# --- Main Execution Logic ---
def main(config_file_path, output_base_dir_arg="output_data/prophet_analysis", cache_raw_data_arg=True):
    print(f"Starting Prophet analysis using config: {config_file_path}")
    try:
        with open(config_file_path, 'r') as f: config_data_main = json.load(f)
    except Exception as e: print(f"ERROR reading config file {config_file_path}: {e}"); return
    if "sites_to_process" not in config_data_main or not config_data_main["sites_to_process"]: print(f"No 'sites_to_process' in {config_file_path}."); return
    sites_info_list_main = config_data_main["sites_to_process"]; enabled_sites_info = [s for s in sites_info_list_main if s.get("enabled", True)];
    if not enabled_sites_info: print("No enabled sites."); return
    os.makedirs(output_base_dir_arg, exist_ok=True); raw_data_cache_dir = os.path.join(output_base_dir_arg, "raw_data_cache") if cache_raw_data_arg else None
    all_sites_data_dict = fetch_data_for_sites(enabled_sites_info, data_cache_dir=raw_data_cache_dir)
    prophet_results = {}; print("\n--- Running Prophet Analysis ---")
    for site_id_key, site_data_info in all_sites_data_dict.items():
        df_site = site_data_info['df']; value_col_to_use = site_data_info['value_col']
        if df_site.empty: continue
        results_df, fitted_model = analyze_with_prophet(df_site, value_col_to_use, site_id_key, output_base_dir_arg)
        if results_df is not None:
             prophet_results[site_id_key] = results_df; safe_site_id = site_id_key.replace(":", "_"); site_output_dir = os.path.join(output_base_dir_arg, safe_site_id)
             results_csv_path = os.path.join(site_output_dir, f"{safe_site_id}_prophet_results.csv")
             try: results_df.to_csv(results_csv_path); print(f"  Saved Prophet results CSV: {results_csv_path}")
             except Exception as e: print(f"ERROR saving Prophet results CSV for {site_id_key}: {e}")
    print("\nProphet analysis complete.")

    # --- Optional: Correlation/Lagged Correlation on Prophet Anomalies ---
    if prophet_results:
        print("\n--- Calculating Correlations based on Prophet Anomalies ---")
        anomaly_timelines = {}; common_date_index = None; sites_with_prophet_anomalies = []
        for site_id_key, results_df in prophet_results.items():
            if results_df is not None and 'is_anomaly' in results_df.columns:
                if not isinstance(results_df.index, pd.DatetimeIndex): results_df.index = pd.to_datetime(results_df.index)
                resampled_anomalies = results_df['is_anomaly'].astype(int).resample('D').max().fillna(0)
                if resampled_anomalies.std() > 0:
                    anomaly_timelines[site_id_key] = resampled_anomalies; sites_with_prophet_anomalies.append(site_id_key)
                    if common_date_index is None: common_date_index = resampled_anomalies.index
                    else: common_date_index = common_date_index.intersection(resampled_anomalies.index)
                else: print(f"Skipping {site_id_key} from correlation (Prophet found no anomalies).")
        if not anomaly_timelines or common_date_index is None or common_date_index.empty: print("\nNo common dates or sites with actual anomalies found by Prophet."); return
        aligned_anomaly_df = pd.DataFrame(index=common_date_index)
        sites_to_include = [site for site in sites_with_prophet_anomalies if site in anomaly_timelines] # Use only sites with anomalies
        for site_id_key in sites_to_include: aligned_anomaly_df[site_id_key] = anomaly_timelines[site_id_key].reindex(common_date_index).fillna(0)
        if aligned_anomaly_df.shape[1] < 2: print(f"\nNeed >= 2 sites with detected anomalies for correlation. Found: {aligned_anomaly_df.shape[1]}"); return
        print(f"\nCalculating Prophet anomaly correlation matrix for sites: {sites_to_include}")
        correlation_matrix = aligned_anomaly_df.corr(); correlation_matrix.dropna(axis=0, how='all', inplace=True); correlation_matrix.dropna(axis=1, how='all', inplace=True)
        if correlation_matrix.shape[0] < 2: print("Not enough sites remaining after handling NaNs in correlation matrix."); return
        config_basename = os.path.splitext(os.path.basename(config_file_path))[0]; run_name_for_outputs = f"{config_basename}_prophet_analysis"
        print("Generating Prophet anomaly clustermap...")
        num_sites_plot = correlation_matrix.shape[0]; fig_size_heat = min(max(8, num_sites_plot * 0.7), 25); annot_kws_size = max(6, 11 - int(num_sites_plot / 5)); show_annotations = num_sites_plot <= 15
        try:
            if not np.all(np.isfinite(correlation_matrix.fillna(0))): raise ValueError("Non-finite values prevent clustering.")
            clustergrid = sns.clustermap(correlation_matrix, annot=show_annotations, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1, figsize=(fig_size_heat, fig_size_heat), linewidths=.5, annot_kws={"size": annot_kws_size}, dendrogram_ratio=(.15, .15))
            plt.suptitle(f"Correlation of Prophet-Detected Anomalies\nConfig: {config_basename} (Clustered)", fontsize=14, y=1.03); plt.setp(clustergrid.ax_heatmap.get_xticklabels(), rotation=60, ha="right", fontsize=9); plt.setp(clustergrid.ax_heatmap.get_yticklabels(), rotation=0, fontsize=9); clustergrid.cax.set_visible(True); clustergrid.cax.set_ylabel('Correlation', rotation=-90, va='bottom'); clustergrid.cax.tick_params(labelsize=8)
            correlation_plot_path = os.path.join(output_base_dir_arg, f"{run_name_for_outputs}_anomaly_correlation_clustermap.png"); clustergrid.savefig(correlation_plot_path, dpi=150, bbox_inches='tight'); print(f"Prophet anomaly correlation clustermap saved to: {correlation_plot_path}"); plt.close(clustergrid.figure)
        except Exception as e_clustermap:
             print(f"INFO: Could not generate clustermap ({e_clustermap}). Generating simple heatmap.")
             try:
                 plt.figure(figsize=(fig_size_heat, fig_size_heat * 0.9)); ax = sns.heatmap(correlation_matrix, annot=show_annotations, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1, annot_kws={"size": annot_kws_size}, linewidths=.5, linecolor='lightgray'); plt.title(f"Correlation of Prophet-Detected Anomalies\nConfig: {config_basename} (Heatmap)", fontsize=12, pad=20); ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9); ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9); plt.tight_layout()
                 correlation_plot_path = os.path.join(output_base_dir_arg, f"{run_name_for_outputs}_anomaly_correlation_heatmap_fallback.png"); plt.savefig(correlation_plot_path, dpi=150); plt.close(); print(f"Saved fallback heatmap: {correlation_plot_path}")
             except Exception as e_heatmap: print(f"ERROR generating fallback heatmap: {e_heatmap}")
        if aligned_anomaly_df.shape[1] >= 2:
            site1_id = aligned_anomaly_df.columns[0]; site2_id = aligned_anomaly_df.columns[1]; s1_anom = aligned_anomaly_df[site1_id]; s2_anom = aligned_anomaly_df[site2_id]
            max_lag = 7; lags = range(-max_lag, max_lag + 1); lagged_corrs = []
            if s1_anom.std() > 0 and s2_anom.std() > 0: lagged_corrs = [s1_anom.corr(s2_anom.shift(lag)) if pd.notna(s1_anom.corr(s2_anom.shift(lag))) else 0 for lag in lags]
            else: lagged_corrs = [np.nan] * len(lags); print(f"\nWarn: Cannot calculate lagged correlation between {site1_id}/{site2_id} (zero variance).")
            peak_corr = np.nan; peak_lag = 0; valid_corrs_indices = [i for i, c in enumerate(lagged_corrs) if pd.notna(c)]
            if valid_corrs_indices: valid_corrs = [lagged_corrs[i] for i in valid_corrs_indices]; abs_corrs = [abs(c) for c in valid_corrs]; max_abs_corr_index_in_valid = np.argmax(abs_corrs); peak_corr = valid_corrs[max_abs_corr_index_in_valid]; peak_lag = lags[valid_corrs_indices[max_abs_corr_index_in_valid]]
            plt.figure(figsize=(8, 5)); plt.plot(lags, lagged_corrs, marker='.', linestyle='-', markersize=8, color='tab:blue', label="Correlation"); peak_text = f"Peak: {peak_corr:.2f} @ Lag={peak_lag}d" if pd.notna(peak_corr) else "Peak: N/A"; plt.figtext(0.5, 0.01, peak_text, ha="center", fontsize=9, bbox={"facecolor":"white", "alpha":0.7, "pad":3})
            if pd.notna(peak_corr): plt.axvline(peak_lag, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'Peak Lag ({peak_lag}d)'); plt.legend(fontsize=9)
            site1_short = site1_id.split(':')[-1]; site2_short = site2_id.split(':')[-1]; title_str = (f"Lagged Correlation of Prophet Anomalies\n{site1_short} vs {site2_short} (Config: {config_basename})"); plt.title(title_str, fontsize=12)
            plt.xlabel(f"Lag (Days {site2_short} is Shifted)", fontsize=10); plt.ylabel("Correlation Coefficient", fontsize=10); plt.axhline(0, color='black', linestyle='-', linewidth=0.7); 
            min_val = np.nanmin(lagged_corrs); max_val = np.nanmax(lagged_corrs); y_min = min(-0.1, min_val*1.1 if pd.notna(min_val) and min_val < -0.01 else -0.1)-0.05; y_max = max(0.1, max_val*1.1 if pd.notna(max_val) and max_val > 0.01 else 0.1)+0.05; plt.ylim(max(-1.05, y_min), min(1.05, y_max)); plt.xticks(lags); plt.grid(True, linestyle=':', alpha=0.6); plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            safe_site1 = site1_id.replace(':','_'); safe_site2 = site2_id.replace(':','_')
            lagged_corr_path = os.path.join(output_base_dir_arg, f"{run_name_for_outputs}_{safe_site1}_vs_{safe_site2}_prophet_lagged_corr.png")
            try: plt.savefig(lagged_corr_path, dpi=150); print(f"Prophet lagged anomaly correlation plot saved to: {lagged_corr_path}")
            except Exception as e: print(f"ERROR saving lagged plot: {e}")
            plt.close()
        elif aligned_anomaly_df.shape[1] >= 2 : print("Could not generate lagged correlation plot for first pair (likely due to lack of variance).")

# --- Configuration and Main Call (No Argparse) ---
if __name__ == "__main__":
    CONFIG_FILE_TO_USE = "config.json" 
    OUTPUT_DIRECTORY = "output_data/prophet_analysis_results" 
    USE_RAW_DATA_CACHE = True 
    script_name = os.path.splitext(os.path.basename(__file__))[0]; config_name = os.path.splitext(os.path.basename(CONFIG_FILE_TO_USE))[0] 
    if OUTPUT_DIRECTORY == "output_data/prophet_analysis_results": OUTPUT_DIRECTORY = os.path.join("output_data", f"{script_name}_{config_name}")
    script_dir = os.path.dirname(os.path.abspath(__file__)); config_file_path = os.path.join(script_dir, CONFIG_FILE_TO_USE)
    if not os.path.exists(config_file_path): print(f"ERROR: Config file not found: {config_file_path}")
    else: main(config_file_path, OUTPUT_DIRECTORY, USE_RAW_DATA_CACHE)