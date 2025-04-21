import pandas as pd
import matplotlib.pyplot as plt
from dataretrieval import nwis
import xarray as xr
import boto3
import s3fs # Makes S3 access easier for xarray

# --- Configuration ---
SITE_NUMBER = "12422500" # Spokane River at Spokane, WA
PARAM_CD_IV = "00060" # Discharge, cfs (Instantaneous Value)
SERVICE_IV = "iv" # Instantaneous Values
# Get recent data (e.g., last 7 days)
END_DATE = pd.Timestamp.now(tz='UTC') # Use UTC for consistency
START_DATE = END_DATE - pd.Timedelta(days=7)

# --- NWM Configuration (NEEDS ADJUSTMENT BY USER) ---
# You need to find the NWM v2.1 Feature ID corresponding to USGS 12422500
# This often requires consulting NWM documentation or lookup tables.
# Example - THIS IS LIKELY INCORRECT, replace with actual ID:
NWM_FEATURE_ID = 7080711 # Example ID - REPLACE THIS
# NWM output is often in m^3/s, USGS is cfs. 1 m^3/s = 35.3147 cfs
CMS_TO_CFS = 35.3147

print(f"Fetching recent USGS instantaneous discharge data for site {SITE_NUMBER}...")

# Fetch recent USGS data
df_usgs, md_usgs = nwis.get_record(
    sites=SITE_NUMBER,
    service=SERVICE_IV,
    start=START_DATE.strftime('%Y-%m-%d'),
    end=END_DATE.strftime('%Y-%m-%d'),
    parameterCd=PARAM_CD_IV,
)

print("USGS data fetched.")

if df_usgs.empty:
    print(f"No recent USGS data found for site {SITE_NUMBER}.")
    df_usgs_processed = pd.DataFrame() # Create empty df to avoid later errors
else:
    # Process USGS data
    discharge_col_iv = next((col for col in df_usgs.columns if PARAM_CD_IV in col), None)
    if discharge_col_iv:
        print(f"Using USGS discharge column: {discharge_col_iv}")
        df_usgs_processed = df_usgs[[discharge_col_iv]].copy()
        df_usgs_processed.columns = ['USGS_Discharge_cfs']
        df_usgs_processed['USGS_Discharge_cfs'] = pd.to_numeric(df_usgs_processed['USGS_Discharge_cfs'], errors='coerce')
        # Ensure index is timezone-aware (UTC)
        df_usgs_processed.index = df_usgs_processed.index.tz_convert('UTC')
        df_usgs_processed = df_usgs_processed.dropna().sort_index()
        print("Processed USGS data head:\n", df_usgs_processed.head())
    else:
        print("Could not find USGS discharge column.")
        df_usgs_processed = pd.DataFrame()

# --- !!! NWM Data Fetching Outline !!! ---
print("\n--- NWM Data Fetching (Outline - Requires AWS Credentials & Correct Feature ID) ---")
df_nwm_processed = pd.DataFrame() # Initialize empty dataframe

try:
    # 1. Determine NWM Forecast Cycle (e.g., latest available)
    # NWM produces forecasts multiple times a day. You need to choose one.
    # Let's target a forecast from ~1 day ago to ensure it exists
    forecast_date = END_DATE - pd.Timedelta(days=1)
    forecast_hour = 12 # Example forecast cycle hour (UTC)
    forecast_cycle_str = forecast_date.strftime('%Y%m%d') + f"t{forecast_hour:02d}z"
    print(f"Targeting NWM forecast cycle: {forecast_cycle_str}")

    # 2. Construct S3 Path for NWM Channel Route file
    # Format varies slightly by version and forecast type (short_range, medium_range...)
    # This is an example path for NWM v2.1 short_range - VERIFY THIS
    s3_bucket = 'noaa-nwm-pds'
    s3_key = f'nwm.{forecast_date.strftime("%Y%m%d")}/short_range/nwm.t{forecast_hour:02d}z.short_range.channel_rt.f018.conus.nc' # f018 = forecast hour 18
    # NOTE: You might need to list files to find the exact one, or loop through forecast hours (f001, f002...)

    s3_uri = f"s3://{s3_bucket}/{s3_key}"
    print(f"Attempting to access NWM file (requires credentials configured): {s3_uri}")

    # 3. Access S3 and Read Data using s3fs and xarray
    # This step requires AWS credentials configured (e.g., via ~/.aws/credentials or environment variables)
    # Use s3fs for easier access
    s3 = s3fs.S3FileSystem(anon=True) # Use anon=True for public bucket access

    # Check if file exists before trying to open
    if s3.exists(s3_uri):
        with s3.open(s3_uri) as f:
            ds_nwm = xr.open_dataset(f, engine='netcdf4')
            print("NWM NetCDF file opened successfully.")

            # 4. Extract data for the specific Feature ID
            # Select the streamflow variable for the specific feature_id
            nwm_flow_cms = ds_nwm['streamflow'].sel(feature_id=NWM_FEATURE_ID)

            # Convert to pandas DataFrame/Series
            df_nwm = nwm_flow_cms.to_dataframe()

            # Convert flow to CFS and rename column
            df_nwm['NWM_Discharge_cfs'] = df_nwm['streamflow'] * CMS_TO_CFS

            # Make index timezone-aware (UTC)
            df_nwm.index = df_nwm.index.tz_localize('UTC')

            df_nwm_processed = df_nwm[['NWM_Discharge_cfs']].copy().sort_index()
            print("Processed NWM data head:\n", df_nwm_processed.head())
            ds_nwm.close()
    else:
        print(f"NWM file not found at: {s3_uri}")
        print("Check NWM Feature ID, forecast cycle time, and S3 path format.")

except Exception as e:
    print(f"Could not fetch or process NWM data. Error: {e}")
    print("Ensure AWS credentials are configured if needed, s3fs is installed,")
    print("and the NWM_FEATURE_ID and S3 path are correct.")
    df_nwm_processed = pd.DataFrame() # Ensure it's empty on error
# --- End NWM Outline ---


# --- Plotting Comparison ---
plt.figure(figsize=(12, 6))

if not df_usgs_processed.empty:
    plt.plot(df_usgs_processed.index, df_usgs_processed['USGS_Discharge_cfs'], label='USGS Observed', marker='.', linestyle='-', color='blue')

if not df_nwm_processed.empty:
    # You might need to adjust the time range if NWM covers a different period
    plt.plot(df_nwm_processed.index, df_nwm_processed['NWM_Discharge_cfs'], label=f'NWM Forecast (Feature {NWM_FEATURE_ID})', marker='x', linestyle='--', color='red')

if not df_usgs_processed.empty or not df_nwm_processed.empty:
    plt.title(f"USGS Observed vs NWM Forecast\nSite {SITE_NUMBER}")
    plt.xlabel("Date / Time (UTC)")
    plt.ylabel("Discharge (cfs)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("project3_nwm_vs_usgs.png")
    print("\nPlot saved as project3_nwm_vs_usgs.png")
    plt.show()
else:
    print("\nNo data available to plot.")
