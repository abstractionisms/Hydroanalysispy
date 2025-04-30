import pandas as pd
import plotly.express as px
import os

# --- Configuration ---
data_file_name = 'nwis_inventory_filtered_data_only.txt'

# Dot Settings
marker_size = 6 # Slightly smaller default size might be better for more points
# --- End Configuration ---

# Get the directory where the script is located
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

file_path = os.path.join(script_dir, data_file_name)

print(f"Attempting to read data from: {file_path}")

if not os.path.exists(file_path):
    print(f"Error: Data file not found at {file_path}")
    print("Please make sure the data file is in the same directory as the script.")
else:
    try:
        # Read data
        use_cols = [0, 1, 2, 3]
        df = pd.read_csv(file_path, sep='\t', header=None, usecols=use_cols,
                         names=['site_no', 'station_nm', 'dec_lat_va', 'dec_long_va'],
                         low_memory=False, encoding='utf-8', on_bad_lines='skip')

        # Data Cleaning
        df_plot = df[['site_no', 'station_nm', 'dec_lat_va', 'dec_long_va']].copy()
        df_plot['dec_lat_va'] = pd.to_numeric(df_plot['dec_lat_va'], errors='coerce')
        df_plot['dec_long_va'] = pd.to_numeric(df_plot['dec_long_va'], errors='coerce')
        df_plot.dropna(subset=['dec_lat_va', 'dec_long_va'], inplace=True)
        df_plot['site_no'] = df_plot['site_no'].astype(str).str.strip()
        df_plot.rename(columns={
            'site_no': 'Site Number',
            'station_nm': 'Site Name',
            'dec_lat_va': 'Latitude',
            'dec_long_va': 'Longitude'
        }, inplace=True)

        total_sites = len(df_plot)
        print(f"Total sites loaded and ready to plot: {total_sites}")

        # *** REMOVED the filtering step for Washington State ***

        if total_sites == 0:
             print("Warning: No sites found in the file.")
        else:
            # *** Calculate center of all points for initial view ***
            map_center_lat = df_plot['Latitude'].mean()
            map_center_lon = df_plot['Longitude'].mean()
            # *** Adjust initial zoom to show more area ***
            map_zoom_level = 4 # Lower zoom level (try 3, 4, or 5)

            print(f"Setting initial map center to Lat: {map_center_lat:.4f}, Lon: {map_center_lon:.4f}")
            print(f"Setting initial zoom level to: {map_zoom_level}")

            # Create the map figure using ALL cleaned data (df_plot)
            fig = px.scatter_mapbox(df_plot, # Use df_plot (all sites)
                                    lat="Latitude",
                                    lon="Longitude",
                                    hover_name="Site Name",
                                    hover_data=["Site Number", "Latitude", "Longitude"],
                                    # Use calculated center and adjusted zoom
                                    center=dict(lat=map_center_lat, lon=map_center_lon),
                                    zoom=map_zoom_level,
                                    height=700,
                                    # Updated title
                                    title=f"Interactive Map - All Sites ({total_sites} total)")

            fig.update_layout(mapbox_style="open-street-map")
            fig.update_layout(margin={"r":0,"t":50,"l":0,"b":10})
            fig.update_traces(marker=dict(size=marker_size))

            # Show the interactive figure in a browser window
            print("Displaying interactive map in browser...")
            fig.show()
            print("Map display command executed. Check your browser.")


    except Exception as e:
        print(f"An error occurred while processing the file: {e}")