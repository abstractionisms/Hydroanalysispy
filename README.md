# Hydrology Analysis Dashboard

A Streamlit web application for analyzing USGS water discharge data with climate correlation analysis.

## Features

- **Interactive Site Map** - View 736+ USGS monitoring sites across the Pacific Northwest
- **13 Plot Types** - Time series, flow duration curves, climate correlations, heatmaps, and more
- **Comparison Modes** - Compare time periods, sites, or both in a 2x2 grid
- **Data Availability Indicators** - Quick visual check (✅/⚠️/❌) for discharge, gage height, and climate data
- **Dark Theme** - Easy on the eyes

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

- **USGS NWIS** - Discharge and gage height data
- **Meteostat** - Temperature and precipitation from nearest weather stations

## Screenshots

*Coming soon*

## Requirements

- Python 3.9+
- See `requirements.txt` for dependencies

## License

MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

- USGS for streamflow data via NWIS API
- Meteostat for climate data
- Streamlit for the web framework
