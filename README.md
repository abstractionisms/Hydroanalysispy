# HydroPlot

Streamlit dashboard for exploring USGS streamflow, climate context, and reach-scale hydrology.

**Use the app:** [hydroplot.streamlit.app](https://hydroplot.streamlit.app)

## Features

- Interactive map of Pacific Northwest USGS monitoring sites
- Single-site discharge, climate, trend, frequency, and drought diagnostics
- Time-period, site-to-site, and 2x2 comparison workflows
- Reach analysis for upstream/downstream gain, loss, low-flow windows, and aquifer contribution
- Live USGS NWIS streamflow and Meteostat climate data
- 30+ static and interactive hydrology plots

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
