# Hydrology Analysis Dashboard

A Streamlit web application for analyzing USGS water discharge data with climate correlation analysis.

## Features

- **Interactive Site Map** - View 736+ USGS monitoring sites across the Pacific Northwest
- **23 Plot Types** - Time series, flow duration curves, flood frequency, baseflow separation, recession analysis, climate correlations, heatmaps, and more
- **5 Analysis Modes**:
  - **Site Map** - Interactive map of all monitoring sites
  - **Single Analysis** - Deep dive into one site with multiple plots
  - **Compare Time Periods** - Same site across two time periods
  - **Compare Sites** - Multiple sites (2-4) for the same period
  - **2×2 Comparison** - Two sites × two time periods
- **Accurate Data Availability** - Exact per-parameter dates from USGS series catalog (not estimates)
- **Flexible Date Selection** - Year sliders with fine-tune date inputs
- **Dark Theme** - Easy on the eyes

## Plot Types

| Category | Plots |
|----------|-------|
| **Time Series** | Discharge timeseries, Precip-discharge overlay, Anomaly detection |
| **Flow Analysis** | Flow duration curve, Monthly boxplots, Annual/low-flow trends |
| **Frequency Analysis** | Flood frequency (Log-Pearson III), 7Q10 low flow analysis |
| **Hydrograph Analysis** | Baseflow separation, Recession curves, Rating curve |
| **Climate Correlation** | Hexbin temp, Lagged precip, Seasonal scatter, Lag correlation |
| **Heatmaps** | Discharge density, Temporal (5/10/20yr panels) |
| **Advanced** | Double mass curve, Cumulative departure, Spectral analysis, Anomaly (Q/T/P) |

## Quick Start

```bash
# Clone and install
git clone https://github.com/abstractionisms/Hydroanalysispy
cd Hydroanalysispy
pip install -e .

# Run the dashboard
python run_dashboard.py
```

Then open http://localhost:8501 in your browser.

## Data Sources

- **USGS NWIS** - Discharge and gage height via Water Services API
- **Meteostat** - Temperature and precipitation from nearest weather stations

## Requirements

- Python 3.9+
- See `requirements.txt` for dependencies

## License

MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

- USGS for streamflow data via NWIS API
- Meteostat for climate data
- Streamlit for the web framework
