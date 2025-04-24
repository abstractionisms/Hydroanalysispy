import os
import logging

# --- Configuration ---
# Input file 1: Your original inventory file (contains all data, comments, header)
INPUT_FULL_INVENTORY = 'nwis_inventory_with_latlon.txt'
# Input file 2: The list of site IDs (one per line) you want to keep
# *** CORRECTED: Make sure this file ONLY contains the list of site IDs ***
INPUT_GOOD_SITE_LIST = 'sites_with_climate_potential_and_usgs_data.txt'
# Output file: Where to save ONLY the data lines matching the good sites
OUTPUT_FILTERED_DATA = 'nwis_inventory_filtered_data_only.txt'
# Log file for this script
LOG_FILE = 'inventory_filtering_simple.log'

# --- Logging Setup ---
def setup_logging(log_file=LOG_FILE):
    """Configures basic logging for the filtering script."""
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__) # Use a specific logger name
    logger.setLevel(logging.INFO)

    # Prevent adding handlers multiple times
    if logger.hasHandlers():
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()

    # File Handler
    try:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Error setting up file logger {log_file}: {e}")

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    return logger

# --- Main Filtering Function ---
def filter_data_lines(full_inventory_path, good_sites_path, output_path, logger):
    """
    Filters the full inventory, writing ONLY data lines where the first column
    matches a site ID from the good sites list. Skips comments and header.
    """
    logger.info("Starting simple inventory filtering process (data lines only).")
    logger.info(f"Full inventory: {full_inventory_path}")
    logger.info(f"Good site list: {good_sites_path}")
    logger.info(f"Output file: {output_path}")

    # Check if input files exist
    if not os.path.exists(full_inventory_path):
        logger.error(f"Full inventory file not found: {full_inventory_path}")
        print(f"Error: Full inventory file not found: {full_inventory_path}")
        return False
    if not os.path.exists(good_sites_path):
        logger.error(f"Good site list file not found: {good_sites_path}")
        print(f"Error: Good site list file not found: {good_sites_path}")
        return False

    try:
        # 1. Read the good site IDs into a set
        logger.info(f"Reading good site IDs from {good_sites_path}...")
        with open(good_sites_path, 'r', encoding='utf-8') as f_sites:
            # Read lines, strip whitespace, and filter out empty lines
            good_sites = {line.strip() for line in f_sites if line.strip()}
        logger.info(f"Loaded {len(good_sites)} unique site IDs to keep.")

        # Check if the set is empty OR contains log-like entries
        if not good_sites:
             logger.error("The good site list file is empty or contains no valid site IDs. Cannot filter.")
             print("Error: The good site list file is empty or contains no valid site IDs.")
             return False
        # Basic check if it looks like a log file was read instead of site IDs
        if any(line.startswith('2025-') for line in good_sites):
             logger.error(f"The file '{good_sites_path}' appears to contain log entries, not site IDs. Please check the filename.")
             print(f"Error: The file '{good_sites_path}' appears to contain log entries, not site IDs.")
             return False


        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            logger.info(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir)

        # 2. Process the full inventory file
        logger.info(f"Processing full inventory file: {full_inventory_path}...")
        lines_read = 0
        lines_written = 0
        lines_skipped = 0

        with open(full_inventory_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:

            for line in infile:
                lines_read += 1
                stripped_line = line.strip()

                # Skip empty lines and comment lines
                if not stripped_line or stripped_line.startswith('#'):
                    lines_skipped += 1
                    continue

                # Split the line by tab to get the first column (potential site ID)
                try:
                    # Split only once to get the first part efficiently
                    # Check if the line actually contains a tab before splitting
                    if '\t' in stripped_line:
                        first_col = stripped_line.split('\t', 1)[0]
                    else:
                        # Handle lines without tabs (like the format specifier line)
                        lines_skipped += 1
                        continue # Skip lines without tabs

                    # Check if the first column is in our set of good sites
                    if first_col in good_sites:
                        outfile.write(line) # Write the original line (with newline)
                        lines_written += 1
                    else:
                        # Skip if first column doesn't match (includes header line)
                        lines_skipped += 1

                except IndexError:
                    # Should not happen with the check above, but as a safeguard
                    logger.warning(f"Line {lines_read}: Skipping line - could not split or find first column: '{stripped_line}'")
                    lines_skipped += 1
                except Exception as e:
                    logger.warning(f"Line {lines_read}: Error processing line ({e}). Skipping line: '{stripped_line}'")
                    lines_skipped += 1


        logger.info("Filtering complete.")
        logger.info(f"Total lines read from inventory: {lines_read}")
        logger.info(f"Data lines written (matching good sites): {lines_written}")
        logger.info(f"Lines skipped (comments, header, format line, non-matching): {lines_skipped}")

        if lines_written > 0:
            print(f"\nFiltering complete. Output saved to: {output_path}")
            print(f"NOTE: This output file contains ONLY the data lines for the {lines_written} specified sites.")
        else:
            print(f"\nFiltering complete, but NO matching data lines were found for the sites listed in '{good_sites_path}'.")
            print(f"Please verify '{good_sites_path}' contains the correct site IDs and that they exist in '{full_inventory_path}'.")

        return True

    except Exception as e:
        logger.error(f"An error occurred during filtering: {e}")
        print(f"An error occurred during filtering: {e}")
        return False

# --- Script Execution ---
if __name__ == "__main__":
    logger = setup_logging()
    success = filter_data_lines(
        INPUT_FULL_INVENTORY,
        INPUT_GOOD_SITE_LIST,
        OUTPUT_FILTERED_DATA,
        logger
    )
    if success:
        logger.info("--- Simple Inventory Filtering Script Finished ---")
        print("--- Simple Inventory Filtering Script Finished ---")
    else:
        logger.error("--- Simple Inventory Filtering Script Finished With Errors ---")
        print("--- Simple Inventory Filtering Script Finished With Errors ---")
    print(f"Check log file ('{LOG_FILE}') for details.")
