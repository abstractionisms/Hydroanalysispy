import pandas as pd
from meteostat import Point, Daily, Stations
from datetime import datetime, timedelta
import os
import logging
import numpy as np # For checking NaN
import requests # Needed for USGS check
import xml.etree.ElementTree as ET # Needed for parsing USGS check response
from io import StringIO # Needed for parsing USGS check response

# --- Configuration ---
# Make sure this points to your inventory file with lat/lon columns
INVENTORY_FILE = 'nwis_inventory_with_latlon.txt'
# Optional: Define target date range to check against station inventory (for Meteostat)
TARGET_START_YEAR = 2000
TARGET_END_YEAR = datetime.now().year
OUTPUT_FILE_GOOD_SITES = 'sites_with_climate_potential_and_usgs_data.txt' # Updated filename
LOG_FILE = 'climate_screening_usgs.log' # Updated filename

# --- USGS Data Check Configuration ---
CHECK_USGS_DATA = True # Set to False to skip the USGS availability check
USGS_PARAM_CD_CHECK = "00060" # Parameter code to check (00060 = Discharge, cfs)
USGS_CHECK_DAYS = 7 # How many recent days to check for USGS data

# --- Logging Setup ---
def setup_logging(log_file=LOG_FILE):
    """Configures basic logging."""
    root_logger = logging.getLogger()
    # Prevent duplicate handlers if run multiple times in the same environment
    if root_logger.hasHandlers() and len(root_logger.handlers) >= 2:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()

    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    root_logger.setLevel(logging.INFO) # Set root logger level

    # File Handler
    try:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        file_handler = logging.FileHandler(log_file, mode='w') # Overwrite log each run
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        print(f"Error setting up file logger {log_file}: {e}") # Use print as logger might fail

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
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
        header_row_index = -1
        column_names = []
        comment_lines_count = 0
        # First pass to find the header row after comments
        with open(inventory_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line.strip().startswith('#'):
                    comment_lines_count += 1
                    continue
                # Assume the first non-comment line is the header
                if header_row_index == -1:
                    header_row_index = i
                    column_names = [col.strip() for col in line.strip().split('\t')]
                    break # Found header, stop reading lines

        if header_row_index == -1 or 'site_no' not in column_names:
            raise ValueError("Could not detect a header row with 'site_no' after comments.")

        # Read the CSV using the detected header row index relative to the start of data
        df_inventory = pd.read_csv(
            inventory_path,
            sep='\t',
            comment='#',
            header=header_row_index - comment_lines_count, # Adjust header index based on skipped comments
            names=column_names, # Provide column names explicitly
            low_memory=False,
            dtype={'site_no': str} # Ensure site_no is read as string
        )
        logging.info(f"Inventory loaded: {len(df_inventory)} sites.")

        # Validate required columns
        required_cols = ['site_no','station_nm', 'dec_lat_va', 'dec_long_va']
        if not all(col in df_inventory.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df_inventory.columns]
            raise ValueError(f"Inventory missing required columns: {missing_cols}")

        # Convert coordinate columns to numeric, coercing errors
        df_inventory['dec_lat_va'] = pd.to_numeric(df_inventory['dec_lat_va'], errors='coerce')
        df_inventory['dec_long_va'] = pd.to_numeric(df_inventory['dec_long_va'], errors='coerce')

        # Keep site_no as a column for iteration, don't set as index here
        return df_inventory
    except ValueError as ve:
        logging.error(f"Value error reading inventory file '{inventory_path}': {ve}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error reading inventory file '{inventory_path}': {e}")
        return None

# --- USGS Data Availability Check ---
def check_usgs_data_availability(site_id, param_cd, days_to_check):
    """
    Checks if USGS NWIS has *any* recent daily data for a given site and parameter.
    Returns True if data is likely available, False otherwise.
    """
    logging.debug(f"Checking USGS availability: Site {site_id}, Param {param_cd}, Days {days_to_check}")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_to_check)
    url_wml = "https://waterservices.usgs.gov/nwis/dv"
    params_wml = {
        'format': 'waterml,1.1', # WaterML is parsable and indicates data structure
        'sites': site_id,
        'parameterCd': param_cd,
        'startDT': start_date.strftime('%Y-%m-%d'),
        'endDT': end_date.strftime('%Y-%m-%d')
    }
    try:
        response = requests.get(url_wml, params=params_wml, timeout=30) # Shorter timeout for check
        response.raise_for_status() # Check for HTTP errors

        # Check if the response text actually contains data values
        # A simple check is to see if the <value> tag exists.
        # More robustly, parse slightly to confirm.
        content = response.text
        if not content:
            logging.warning(f"  USGS Check {site_id}: Empty response content.")
            return False

        # Check for common "no data" messages if possible (might vary)
        if "No sites found" in content or "No data available" in content:
             logging.info(f"  USGS Check {site_id}: Service reported no data available.")
             return False

        # Attempt a minimal parse to find value tags
        try:
            namespaces = {'ns1': 'http://www.cuahsi.org/waterML/1.1/'}
            xml_io = StringIO(content)
            tree = ET.parse(xml_io)
            root = tree.getroot()
            value_elements = root.findall('.//ns1:value', namespaces)
            if not value_elements:
                logging.info(f"  USGS Check {site_id}: Response received, but no <value> tags found for param {param_cd}.")
                # Could be valid site but no data for this param/period
                return False
            else:
                logging.info(f"  USGS Check {site_id}: Found {len(value_elements)} <value> tags. Data likely available.")
                return True # Found value tags, data exists
        except ET.ParseError:
            logging.warning(f"  USGS Check {site_id}: Received response, but failed to parse XML. Assuming no data.")
            # This might happen if the response is an error page not caught by raise_for_status
            return False
        except Exception as parse_e:
             logging.warning(f"  USGS Check {site_id}: Error during minimal parse: {parse_e}. Assuming no data.")
             return False

    except requests.exceptions.Timeout:
        logging.warning(f"  USGS Check {site_id}: Request timed out.")
        return False # Treat timeout as unavailable for screening
    except requests.exceptions.HTTPError as http_err:
        # Handle specific HTTP errors if needed (e.g., 404 Not Found might mean bad site ID)
        logging.warning(f"  USGS Check {site_id}: HTTP Error {http_err.response.status_code}.")
        return False
    except requests.exceptions.RequestException as req_e:
        logging.error(f"  USGS Check {site_id}: Request Exception {req_e}.")
        return False # Treat request errors as unavailable
    except Exception as e:
        logging.error(f"  USGS Check {site_id}: Unexpected error: {e}")
        return False

# --- Main Screening Logic ---
def screen_sites_for_climate_data(df_inventory):
    """
    Screens sites in the inventory DataFrame for potential Meteostat data availability
    AND checks for recent USGS data availability for a specific parameter code.
    Returns a list of site IDs that pass both checks.
    """
    if df_inventory is None or df_inventory.empty:
        logging.error("Inventory DataFrame is empty or None. Cannot screen.")
        return []

    logging.info("Starting climate & USGS availability screening...")
    promising_sites = []
    stations_api = Stations() # Initialize Stations API once

    # Iterate through DataFrame rows
    for index, row in df_inventory.iterrows():
        site_id = row.get('site_no', 'N/A')
        latitude = row.get('dec_lat_va')
        longitude = row.get('dec_long_va')
        station_name = row.get('station_nm', '') # USGS station name

        logging.info(f"--- Checking Site: {site_id} ({station_name[:40]}...) ---")

        # 1. Validate coordinates
        if pd.isna(latitude) or pd.isna(longitude) or not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
            logging.warning(f"  Skipping: Invalid or missing coordinates (Lat={latitude}, Lon={longitude}).")
            continue # Skip to the next site

        # 2. Check USGS Data Availability (if enabled)
        usgs_data_ok = False
        if CHECK_USGS_DATA:
            usgs_data_ok = check_usgs_data_availability(site_id, USGS_PARAM_CD_CHECK, USGS_CHECK_DAYS)
            if not usgs_data_ok:
                logging.warning(f"  Skipping: Failed USGS data availability check for param {USGS_PARAM_CD_CHECK}.")
                continue # Skip to the next site
        else:
            usgs_data_ok = True # Assume OK if check is disabled
            logging.debug(f"  Skipping USGS data check as per configuration.")

        # 3. Check Meteostat Nearby Station (only if coordinates and USGS check passed)
        meteo_station_found = False
        try:
            logging.info(f"  Checking Meteostat near Lat={latitude:.4f}, Lon={longitude:.4f}...")
            nearby_stations_df = stations_api.nearby(latitude, longitude)
            nearest_station_info = nearby_stations_df.fetch(1) # Get the closest one

            if not nearest_station_info.empty:
                station_id_meteo = nearest_station_info.index[0]
                station_name_meteo = nearest_station_info.iloc[0]['name']
                distance = nearest_station_info.iloc[0]['distance']
                logging.info(f"    Found nearby Meteostat station: {station_id_meteo} ({station_name_meteo}) at distance {distance:.0f}m.")
                meteo_station_found = True

                # Optional: Add back the inventory date range check here if needed
                # (Remember it slows things down significantly)

            else:
                logging.warning(f"    No nearby Meteostat station found.")
                meteo_station_found = False # Explicitly set

        except Exception as e:
            logging.error(f"    Error checking Meteostat: {e}")
            meteo_station_found = False # Treat error as not found

        # 4. Add to list if ALL checks passed
        if usgs_data_ok and meteo_station_found:
            logging.info(f"  --> Site {site_id} PASSED both USGS and Meteostat checks. Adding to list.")
            promising_sites.append(site_id)
        else:
             logging.warning(f"  Skipping: Site {site_id} failed one or more checks (USGS OK: {usgs_data_ok}, Meteo OK: {meteo_station_found}).")


    logging.info(f"--- Screening complete. Found {len(promising_sites)} sites passing all checks. ---")
    return promising_sites

# --- Main Script Execution ---
if __name__ == "__main__":
    setup_logging() # Configure logging first
    logging.info("--- Starting Climate & USGS Availability Screening Script ---")

    # Load the inventory data
    inventory_df = load_inventory(INVENTORY_FILE)

    # Proceed only if inventory loaded
    if inventory_df is not None:
        # Run the screening process
        potential_sites = screen_sites_for_climate_data(inventory_df)

        # Save results if any sites were found
        if potential_sites:
            try:
                output_dir = os.path.dirname(OUTPUT_FILE_GOOD_SITES)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                with open(OUTPUT_FILE_GOOD_SITES, 'w') as f:
                    for site_id in potential_sites:
                        f.write(f"{site_id}\n")

                logging.info(f"List of sites passing checks saved to: {OUTPUT_FILE_GOOD_SITES}")
                print(f"\nList of sites passing checks saved to: {OUTPUT_FILE_GOOD_SITES}")
                print(f"Found {len(potential_sites)} sites.")
                print(f"Use this list for 'sites_to_process' in config.json for the main correlation script.")

            except Exception as e:
                logging.error(f"Error saving output file {OUTPUT_FILE_GOOD_SITES}: {e}")
                print(f"Error saving output file {OUTPUT_FILE_GOOD_SITES}: {e}")
        else:
            logging.info("No sites found passing both USGS and Meteostat checks.")
            print("\nNo sites found passing both USGS and Meteostat checks.")

    else:
        logging.error("Inventory loading failed. Cannot proceed.")
        print("\nInventory loading failed. Cannot proceed. Check log for details.")


    logging.info("--- Climate & USGS Availability Screening Script Finished ---")
    print("\n--- Climate & USGS Availability Screening Script Finished ---")
    print(f"Check the log file for details: {LOG_FILE}")
