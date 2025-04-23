import pandas as pd
from meteostat import Point, Daily, Stations
from datetime import datetime
import os
import logging
import numpy as np # For checking NaN

# --- Configuration ---
# Make sure this points to your inventory file with lat/lon columns
INVENTORY_FILE = 'nwis_inventory_with_latlon.txt'
# Optional: Define target date range to check against station inventory
TARGET_START_YEAR = 2000
TARGET_END_YEAR = datetime.now().year
OUTPUT_FILE_GOOD_SITES = 'sites_with_climate_potential.txt'
LOG_FILE = 'climate_screening.log'

# --- Logging Setup ---
def setup_logging(log_file=LOG_FILE):
    """Configures basic logging."""
    root_logger = logging.getLogger()
    # Prevent duplicate handlers
    if root_logger.hasHandlers() and len(root_logger.handlers) >= 2: return
    for handler in root_logger.handlers[:]: root_logger.removeHandler(handler); handler.close()
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    root_logger.setLevel(logging.INFO)
    try:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir): os.makedirs(log_dir)
        file_handler = logging.FileHandler(log_file, mode='w'); file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)
    except Exception as e: print(f"Error setting up file logger {log_file}: {e}")
    console_handler = logging.StreamHandler(); console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

# --- Inventory Loading ---
# (Using the robust loading function)
def load_inventory(inventory_path):
    """Loads the master inventory file into a pandas DataFrame."""
    if not inventory_path or not os.path.exists(inventory_path):
        logging.error(f"Inventory file not found: '{inventory_path}'")
        return None
    logging.info(f"Loading inventory: {inventory_path}")
    try:
        header_row_index = -1; column_names = []; comment_lines_count = 0
        with open(inventory_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line.strip().startswith('#'): comment_lines_count += 1; continue
                if header_row_index == -1: header_row_index = i; column_names = [col.strip() for col in line.strip().split('\t')]; break
        if header_row_index == -1 or 'site_no' not in column_names: raise ValueError("Could not detect header row.")

        df_inventory = pd.read_csv(inventory_path, sep='\t', comment='#', 
        header=header_row_index - comment_lines_count, names=column_names, low_memory=False, dtype={'site_no': str})
        logging.info(f"Inventory loaded: {len(df_inventory)} sites.")
        required_cols = ['site_no','station_nm', 'dec_lat_va', 'dec_long_va']
        if not all(col in df_inventory.columns for col in required_cols): raise ValueError(f"Inventory missing columns: {required_cols}")
        df_inventory['dec_lat_va'] = pd.to_numeric(df_inventory['dec_lat_va'], errors='coerce')
        df_inventory['dec_long_va'] = pd.to_numeric(df_inventory['dec_long_va'], errors='coerce')
        # Keep site_no as a column for iteration, don't set as index here
        return df_inventory
    except Exception as e: logging.error(f"Error reading inventory file '{inventory_path}': {e}"); return None

# --- Main Screening Logic ---
def screen_sites_for_climate_data(df_inventory):
    """
    Screens sites in the inventory DataFrame for potential Meteostat data availability.
    Returns a list of site IDs that have a nearby station with potentially overlapping data.
    """
    if df_inventory is None or df_inventory.empty:
        logging.error("Inventory DataFrame is empty or None. Cannot screen.")
        return []

    logging.info("Starting climate availability screening...")
    promising_sites = []
    stations_api = Stations() # Initialize Stations API once

    for index, row in df_inventory.iterrows():
        site_id = row.get('site_no', 'N/A')
        latitude = row.get('dec_lat_va')
        longitude = row.get('dec_long_va')
        station_name = row.get('station_nm', '')

        # Validate coordinates before querying Meteostat
        if pd.isna(latitude) or pd.isna(longitude) or not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
            logging.warning(f"Skipping Site {site_id} ({station_name[:30]}...): Invalid or missing coordinates (Lat={latitude}, Lon={longitude}).")
            continue

        logging.info(f"Checking site {site_id} at Lat={latitude:.4f}, Lon={longitude:.4f}...")

        try:
            # Find nearest station(s)
            nearby_stations_df = stations_api.nearby(latitude, longitude)
            nearest_station_info = nearby_stations_df.fetch(1) # Get the closest one

            if not nearest_station_info.empty:
                station_id_meteo = nearest_station_info.index[0]
                station_name_meteo = nearest_station_info.iloc[0]['name']
                distance = nearest_station_info.iloc[0]['distance'] # Distance in meters

                logging.info(f"  Found nearby Meteostat station: {station_id_meteo} ({station_name_meteo}) at distance {distance:.0f}m.")

                # --- Optional: Check station's data inventory ---
                # This adds more checks but might slow down the process significantly
                # try:
                #     daily_inventory = Daily(station_id_meteo).inventory()
                #     if daily_inventory and 'daily' in daily_inventory:
                #         start_year = daily_inventory['daily'].get('start', pd.Timestamp.max).year
                #         end_year = daily_inventory['daily'].get('end', pd.Timestamp.min).year
                #         logging.info(f"    Station daily data range: {start_year}-{end_year}")
                #         # Check if station range overlaps significantly with target range
                #         if start_year <= TARGET_START_YEAR and end_year >= TARGET_END_YEAR - 1: # Allow end year to be slightly less than current
                #             logging.info(f"    --> Station {station_id_meteo} looks promising for site {site_id}.")
                #             promising_sites.append(site_id)
                #         else:
                #             logging.warning(f"    Station {station_id_meteo} data range ({start_year}-{end_year}) doesn't fully cover target ({TARGET_START_YEAR}-{TARGET_END_YEAR}).")
                #     else:
                #         logging.warning(f"    No daily data inventory found for station {station_id_meteo}.")
                # except Exception as inv_e:
                #     logging.error(f"    Error checking inventory for station {station_id_meteo}: {inv_e}")
                # --- End Optional Inventory Check ---

                # --- Simpler Check: Just finding a nearby station ---
                # If you just want to know if *any* station is nearby, uncomment this:
                logging.info(f"    --> Site {site_id} has a nearby station. Adding to potential list.")
                promising_sites.append(site_id)
                # --- End Simpler Check ---

            else:
                logging.warning(f"  No nearby Meteostat station found for site {site_id}.")

        except Exception as e:
            logging.error(f"  Error checking Meteostat for site {site_id}: {e}")
            # Continue to next site even if one errors out

    logging.info(f"Screening complete. Found {len(promising_sites)} sites with potential climate data.")
    return promising_sites

# --- Main Script Execution ---
if __name__ == "__main__":
    setup_logging()
    logging.info("--- Starting Climate Availability Screening Script ---")

    inventory_df = load_inventory(INVENTORY_FILE)

    if inventory_df is not None:
        potential_sites = screen_sites_for_climate_data(inventory_df)

        if potential_sites:
            # Save the list of promising site IDs to a file
            try:
                output_dir = os.path.dirname(OUTPUT_FILE_GOOD_SITES)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                with open(OUTPUT_FILE_GOOD_SITES, 'w') as f:
                    for site_id in potential_sites:
                        f.write(f"{site_id}\n")
                logging.info(f"List of potentially usable sites saved to: {OUTPUT_FILE_GOOD_SITES}")
                print(f"\nList of potentially usable sites saved to: {OUTPUT_FILE_GOOD_SITES}")
                print(f"You can use this list to update the 'sites_to_run' in your config.json.")
            except Exception as e:
                logging.error(f"Error saving output file {OUTPUT_FILE_GOOD_SITES}: {e}")
        else:
            logging.info("No sites found with potential climate data based on screening criteria.")
            print("\nNo sites found with potential climate data based on screening criteria.")

    logging.info("--- Climate Availability Screening Script Finished ---")

    print("\n--- Climate Availability Screening Script Finished ---")
    print("Check the log file for details: ", LOG_FILE) 