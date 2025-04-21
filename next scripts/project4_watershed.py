import geopandas as gpd
import matplotlib.pyplot as plt
from pygeoapi import nhdplus # Note: Uses pynhd library internally
import contextily as ctx # For adding basemaps

# --- Configuration ---
# Approximate coordinates for USGS 12422500 (Spokane River at Spokane)
# Get these from NWIS site info or a mapping tool
LATITUDE = 47.660 # Approximate Latitude
LONGITUDE = -117.415 # Approximate Longitude
POINT_CRS = "EPSG:4326" # WGS84 Lat/Lon

print(f"Finding watershed for point: Lat={LATITUDE}, Lon={LONGITUDE}")

# --- Use pygeoapi (pynhd) to get watershed ---
try:
    # Define the point as a GeoDataFrame
    point_gdf = gpd.GeoDataFrame(
        [{'geometry': gpd.points_from_xy([LONGITUDE], [LATITUDE])[0]}],
        crs=POINT_CRS
    )

    # Use nhdplus.navigate.get_wb_basin to get the watershed
    # This function finds the nearest NHD flowline and returns its upstream basin
    print("Querying NLDI/NHDPlus web services for watershed...")
    watershed_gdf = nhdplus.navigate.get_wb_basin(point_gdf)

    if watershed_gdf.empty:
        print("No watershed found for the given point.")
    else:
        print("Watershed boundary found.")
        print("Watershed GeoDataFrame head:\n", watershed_gdf.head())
        print("Original CRS:", watershed_gdf.crs)

        # --- Plotting ---
        # Reproject to a suitable projected CRS for plotting (e.g., Web Mercator for contextily)
        plot_crs = "EPSG:3857"
        watershed_plot = watershed_gdf.to_crs(plot_crs)
        point_plot = point_gdf.to_crs(plot_crs)

        print(f"Plotting watershed (reprojected to {plot_crs})...")
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # Plot watershed polygon
        watershed_plot.plot(ax=ax, color='blue', edgecolor='black', alpha=0.5, label='Delineated Watershed')

        # Plot the original point
        point_plot.plot(ax=ax, color='red', marker='o', markersize=50, label='Query Point')

        # Add a basemap using contextily
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik) # Or try other providers

        ax.set_title(f"Watershed Delineated for Point near USGS 12422500")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        # Turn off axis labels which are less meaningful in Web Mercator
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.legend() # Legend might clutter map

        plt.tight_layout()
        plt.savefig("project4_watershed_boundary.png", dpi=300)
        print("\nPlot saved as project4_watershed_boundary.png")
        plt.show()

        # --- Next Steps ---
        print("\n--- Next Steps for Watershed Characterization ---")
        print("1. Obtain DEM (Elevation) data for this watershed boundary (e.g., using py3dep).")
        print("2. Obtain Land Cover data (e.g., NLCD) for the area.")
        print("3. Use geopandas and rasterio to 'clip' the DEM/Land Cover rasters to the watershed polygon.")
        print("4. Calculate statistics from the clipped rasters (e.g., mean slope from DEM, percentage of each land cover type).")

except ImportError:
     print("Error: Could not import 'pygeoapi' or 'contextily'. Make sure they are installed in the environment.")
     print("Try: pip install pygeoapi-client contextily") # Note: pygeoapi depends on pynhd etc.
except Exception as e:
    print(f"An error occurred: {e}")
    print("Check your internet connection and coordinates. NLDI services might be temporarily down.")
