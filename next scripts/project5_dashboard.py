import streamlit as st
import pandas as pd
import plotly.express as px
from dataretrieval import nwis
from datetime import timedelta

# --- Page Configuration (Basic) ---
st.set_page_config(page_title="Spokane River Dashboard", layout="wide")
st.title("Spokane River Streamflow (USGS 12422500)")

# --- Configuration ---
SITE_NUMBER = "12422500"
PARAM_CD_IV = "00060" # Discharge, cfs
SERVICE_IV = "iv"

# --- Data Fetching Function (Cached) ---
# Use st.cache_data to avoid refetching data on every interaction
@st.cache_data(ttl=timedelta(minutes=15)) # Cache data for 15 minutes
def get_usgs_data(site, service, days_back):
    """Fetches USGS instantaneous data for the last N days."""
    print(f"CACHE MISS: Fetching data for site {site} for last {days_back} days...")
    end_date = pd.Timestamp.now(tz='UTC')
    start_date = end_date - timedelta(days=days_back)
    try:
        df, md = nwis.get_record(
            sites=site,
            service=service,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            parameterCd=PARAM_CD_IV,
        )

        # Process data
        discharge_col = next((col for col in df.columns if PARAM_CD_IV in col), None)
        if discharge_col and not df.empty:
            df_processed = df[[discharge_col]].copy()
            df_processed.columns = ['Discharge_cfs']
            df_processed['Discharge_cfs'] = pd.to_numeric(df_processed['Discharge_cfs'], errors='coerce')
            # Keep timezone information provided by dataretrieval
            df_processed = df_processed.dropna().sort_index()
            # Get site name from metadata
            site_name = md.site_name.iloc[0] if not md.empty else "Unknown Site"
            return df_processed, site_name
        else:
            return pd.DataFrame(columns=['Discharge_cfs']), "No Data Found"
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame(columns=['Discharge_cfs']), "Error"

# --- Fetch and Display Data ---
days_to_fetch = 7
df_flow, site_name_disp = get_usgs_data(SITE_NUMBER, SERVICE_IV, days_to_fetch)

st.header(f"Site: {site_name_disp}")

if not df_flow.empty:
    # Display Current Conditions
    st.subheader("Current Conditions")
    latest_reading = df_flow.iloc[-1]
    latest_time = latest_reading.name # Get the timestamp from the index
    latest_flow = latest_reading['Discharge_cfs']

    # Format time for display (convert to local time - Pacific)
    try:
        local_tz = 'America/Los_Angeles'
        latest_time_local = latest_time.tz_convert(local_tz)
        time_str = latest_time_local.strftime('%Y-%m-%d %I:%M:%S %p %Z')
    except Exception: # Handle cases where timezone conversion might fail
        time_str = latest_time.strftime('%Y-%m-%d %H:%M:%S %Z') + " (Original Timezone)"

    st.metric(label=f"Latest Discharge ({time_str})", value=f"{latest_flow:.2f} cfs")

    # Plot Recent Flow
    st.subheader(f"Recent Flow ({days_to_fetch} Days)")
    # Create an interactive plot with Plotly
    fig = px.line(
        df_flow,
        x=df_flow.index,
        y='Discharge_cfs',
        title=f"Discharge over Last {days_to_fetch} Days",
        labels={'index': 'Date / Time', 'Discharge_cfs': 'Discharge (cfs)'}
        )
    fig.update_layout(xaxis_title="Date / Time", yaxis_title="Discharge (cfs)")
    st.plotly_chart(fig, use_container_width=True)

    # Display Raw Data (Optional)
    if st.checkbox("Show Raw Data Table"):
        st.dataframe(df_flow.sort_index(ascending=False))
else:
    st.warning("Could not retrieve or process recent flow data for this site.")

st.markdown("---")
st.markdown(f"Data sourced from USGS NWIS for site {SITE_NUMBER}.")
