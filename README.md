# Hydrology Analysis Package

A professional Python package for hydrological data analysis featuring a Streamlit web dashboard, 30+ visualization types, and integrations with USGS, NOAA, and climate data services.

**Live Demo:** [hydroplot.streamlit.app](https://hydroplot.streamlit.app)

## Features

### Web Dashboard
- **Interactive Site Map** - Explore 736+ USGS monitoring sites across the Pacific Northwest
  - Dark theme Folium map with marker clustering
  - HUC watershed boundary overlays (Regions, Subregions, Basins, Subbasins)
  - Color-coded markers by record age
  - Watershed quick-jump dropdown for WA, OR, and ID regions
  - Click-to-zoom site tables

- **6 Analysis Modes**
  1. **Site Map** - Interactive exploration of all monitoring sites
  2. **Single Analysis** - Deep dive into one site with guided plot presets
  3. **Compare Time Periods** - Same site across two date ranges
  4. **Compare Sites** - 2-4 sites for the same period
  5. **2x2 Comparison** - Two sites x two time periods
  6. **Reach Analysis** - Upstream/downstream gain, loss, and low-flow diagnostics

- **Real-time Data** - Live fetching from USGS NWIS with accurate data availability dates
- **Flexible Date Selection** - Year sliders with fine-tune inputs
- **Metric Cards** - Record length, data points, mean/peak flow statistics
- **Streamlit Cloud Ready** - Pinned Pandas compatibility for Meteostat and current Streamlit width API usage

### 30+ Plot Types

| Category | Plots |
|----------|-------|
| **Time Series** | Discharge timeseries, Precip-discharge overlay, Anomaly detection |
| **Flow Analysis** | Flow duration curve, Monthly boxplots, Annual trend, Low-flow trend |
| **Frequency Analysis** | Flood frequency (Log-Pearson III), 7Q10 low flow analysis |
| **Hydrograph Analysis** | Baseflow separation (Lyne-Hollick), Recession curves, Stage-discharge rating curve |
| **Climate Correlation** | Hexbin temperature, Lagged precipitation, Seasonal scatter, Correlation matrix, Lag analysis (0-30 days) |
| **Heatmaps** | Discharge density by day-of-year, Temporal panels (5/10/20 year) |
| **Advanced** | Double mass curve, Cumulative departure, Spectral analysis (FFT), Climate anomaly (Q/T/P) |
| **Reach Analysis** | Reach comparison, reach index, paired annual lows, low-flow window comparison, threshold exceedance, precipitation response, summer climate context, seasonal gain/loss |

## Installation

```bash
# Clone the repository
git clone https://github.com/abstractionisms/Hydroanalysispy
cd Hydroanalysispy

# Install in development mode
pip install -e .

# Or install with optional dependencies
pip install -e .[dev]        # Development tools (pytest, black, flake8)
pip install -e .[dashboard]  # Streamlit dashboard
```

## Usage

### Web Dashboard

```bash
# Run locally
python run_dashboard.py

# With network access (for remote connections)
python run_dashboard.py --network

# Custom port
python run_dashboard.py --port 8502
```

Then open http://localhost:8501 in your browser.

### Streamlit Cloud

The deployed app entrypoint is:

```text
hydrology/app/streamlit_app.py
```

Streamlit Cloud should install from `requirements.txt`, `packages.txt`, and `runtime.txt`.
The app uses Meteostat as the primary dashboard climate source and normalizes climate columns to `Temp_C` and `Precip_mm`. HyRiver/Daymet remains available as a fallback for workflows that can use it, but the main dashboard plots do not require Daymet credentials.

### Command-Line Analysis

```bash
# Run analysis with config file
python -m hydrology.scripts.analyze_sites --config configs/my_analysis.yaml
```

### Python API

```python
from hydrology.data.usgs import fetch_daily_values
from hydrology.data.climate import fetch_climate_data
from hydrology.analysis.trends import analyze_trend
from hydrology.visualization.plots import plot_flow_duration

# Fetch discharge data
site_id = "12354500"  # Clark Fork at St. Regis, MT
discharge = fetch_daily_values(site_id, "2020-01-01", "2024-01-01")

# Fetch climate data
climate = fetch_climate_data(lat=47.3, lon=-115.1, start="2020-01-01", end="2024-01-01")

# Analyze trends
trend_result = analyze_trend(discharge)
print(f"Mann-Kendall p-value: {trend_result['p_value']}")

# Generate plots
fig = plot_flow_duration(discharge, site_id)
fig.savefig("flow_duration.png")
```

## Package Structure

```
hydrology/
├── core/                    # Configuration and utilities
│   ├── config.py           # JSON/YAML config management
│   ├── parameters.py       # USGS parameter codes (discharge, stage, temp, etc.)
│   ├── paths.py            # Portable path handling
│   ├── logging_setup.py    # Centralized logging
│   ├── timezone.py         # Timezone normalization (UTC)
│   └── huc_regions.py      # HUC watershed region definitions
│
├── data/                    # Data fetching modules
│   ├── usgs.py             # USGS NWIS daily/instantaneous values
│   ├── climate.py          # Meteostat temperature/precipitation
│   ├── inventory.py        # Local site inventory parsing
│   ├── national_inventory.py  # National USGS inventory by HUC-2
│   ├── nwm.py              # NOAA National Water Model forecasts
│   ├── nldi.py             # USGS NLDI river network navigation
│   └── hyriver.py          # Optional HyRiver/Daymet/NHD helpers
│
├── analysis/                # Statistical analysis
│   ├── trends.py           # Mann-Kendall, linear regression trends
│   ├── stage_discharge.py  # Power-law rating curve fitting
│   ├── alerts.py           # Threshold-based alert monitoring
│   ├── multisite.py        # Cross-site correlation analysis
│   └── flood_events.py     # Flood event detection and animation
│
├── visualization/           # Plotting
│   ├── plots.py            # 30+ plot functions
│   └── composer.py         # Multi-panel layout system
│
├── app/                     # Streamlit dashboard
│   ├── streamlit_app.py    # Main application
│   ├── plot_config.py      # Plot categorization
│   ├── styles.py           # Custom CSS and UI components
│   └── page_modules/       # Dashboard page implementations
│
└── scripts/                 # CLI tools
    ├── analyze_sites.py    # Config-driven batch analysis
    └── example_analysis.py # Example workflow
```

## Data Sources

| Source | Data | API |
|--------|------|-----|
| **USGS NWIS** | Discharge, stage, water quality | https://waterservices.usgs.gov/nwis/ |
| **Meteostat** | Primary dashboard temperature and precipitation | Python library |
| **Daymet / HyRiver** | Optional gridded watershed climate fallback | pydaymet / HyRiver |
| **USGS NLDI** | River network topology | https://api.water.usgs.gov/nldi/ |
| **NOAA NWM** | Water model forecasts | https://api.water.noaa.gov/nwps/v1 |
| **USGS WBD** | HUC watershed boundaries | WMS layers |

## Configuration

Analysis can be configured via YAML or JSON files:

```yaml
# configs/example.yaml
sites:
  - site_id: "12354500"
    name: "Clark Fork at St. Regis"
    lat: 47.298
    lon: -115.087

analysis:
  start_date: "2020-01-01"
  end_date: "2024-01-01"
  plots:
    - flow_duration
    - annual_trend
    - flood_frequency

alerts:
  flood_threshold: 15000  # cfs
  low_flow_threshold: 100  # cfs
```

## Requirements

- Python 3.11 on Streamlit Cloud (`runtime.txt`)
- pandas >= 2.0.0, < 3.0.0
- numpy >= 1.24.0
- scipy >= 1.10.0
- matplotlib >= 3.7.0
- streamlit >= 1.28.0
- folium >= 0.14.0
- meteostat == 1.6.8
- scikit-learn >= 1.2.0
- pymannkendall >= 1.4.3

See `requirements.txt` for complete list.

## License

MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

- **USGS** - Streamflow data via NWIS API and HUC watershed boundary WMS layers
- **NOAA** - National Water Model forecasts via Water Prediction Service
- **Meteostat** - Climate data integration
- **Streamlit** - Web application framework
- **Folium** - Interactive mapping
