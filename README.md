# Hydrology Analysis Project

A Python package for hydrological analysis, focusing on streamflow data and its relationship with climate variables. Features a modular plotting system that lets you select which plots to generate and how to arrange them.

**Please note:** This project is being worked on in my spare time, so development may be intermittent.

---

## Quick Start

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/abstractionisms/Hydroanalysispy
   cd Hydrology
   ```

2. Install in development mode:
   ```bash
   pip install -e .
   ```

   This installs all dependencies and makes the `hydrology` package available.

### Run Your First Analysis

```bash
# Single site with vertical layout
python -m hydrology.scripts.analyze_sites --config configs/analysis/example_single_site.yaml

# Multiple sites with custom layouts
python -m hydrology.scripts.analyze_sites --config configs/analysis/example_multi_site.yaml

# Quad (2x2) dashboard layout
python -m hydrology.scripts.analyze_sites --config configs/analysis/example_quad_layout.yaml
```

Plots are saved to: `outputs/plots/<site_id>/multi_plot/`

---

## What's New: Modular Plotting System

The repository has been refactored to use a **configuration-driven modular plotting system** that eliminates code duplication and makes it easy to customize analyses.

### Before (Old Way)
```bash
# Different scripts for different layouts
python q_analysis_vertical.py   # Hardcoded vertical layout
python q_focus_quad.py           # Hardcoded quad layout

# To change plots: Edit the script code
# To change sites: Edit config path in script
```

### After (New Way)
```bash
# One unified script, configured via YAML
python -m hydrology.scripts.analyze_sites --config my_analysis.yaml

# To change plots: Edit YAML config
# To change sites: Edit YAML config
# To change layout: Edit YAML config
```

### Configuration Example

Create a YAML file (e.g., `my_analysis.yaml`):

```yaml
analysis_parameters:
  param_cd: "00060"
  start_date: "2000-01-01"
  end_date: "today"

  # Select which plots to generate
  plots:
    - anomaly          # Recent vs historical comparison
    - hexbin_temp      # Q vs Temperature
    - lagged_precip    # Q vs lagged precipitation
    - timeseries       # Recent discharge timeseries

  # Choose layout: vertical, quad, horizontal, grid_2x3, grid_3x2, auto
  layout: "vertical"

  dpi: 150

sites_to_process:
  - site_id: "12422500"
    description: "Spokane River at Spokane, WA"
    latitude: 47.6593
    longitude: -117.4491
    enabled: true

    # Optional: Override global settings for this site
    # plots: [anomaly, hexbin_temp, flow_duration, correlation_matrix]
    # layout: "quad"
```

See `configs/analysis/README.md` for complete configuration guide.

---

## Available Plots

**NEW:** See [PLOT_GUIDE.md](PLOT_GUIDE.md) for detailed explanations in plain English!

### Climate-Dependent Plots
(Require temperature and precipitation data from Meteostat)

| Plot Name | Description |
|-----------|-------------|
| `anomaly` | Recent decade vs historical monthly averages (Q, T, P) |
| `hexbin_temp` | Discharge vs Temperature hexbin density plot (log scale) |
| `lagged_precip` | Monthly discharge vs lagged precipitation scatter |
| `correlation_matrix` | Correlation heatmap with statistical significance stars |
| `precip_discharge` | Precipitation overlay with discharge and cumulative precip |

### Discharge-Only Plots
(Work even without climate data)

| Plot Name | Description |
|-----------|-------------|
| `timeseries` | Recent discharge time series (log scale) |
| `flow_duration` | Flow duration curve (exceedance probability) |
| `monthly_boxplot` | Monthly discharge distribution boxplots |
| `discharge_heatmap` | Discharge density vs day of year (2D histogram) |
| `temporal_heatmap` | 4-panel temporal heatmap (5/10/20yr + total record) |

**Total: 10 plot types available**

## Layout Options

| Layout | Description | Best For |
|--------|-------------|----------|
| `vertical` | All plots stacked vertically | Detailed comparison, 3-7 plots |
| `quad` | 2x2 grid (exactly 4 plots) | Dashboard, 4 plots |
| `horizontal` | Side-by-side | Wide displays, 2-3 plots |
| `grid_2x3` | 2 columns, 3 rows | 4-6 plots |
| `grid_3x2` | 3 columns, 2 rows | 4-6 plots |
| `grid_2x5` | 2 columns, 5 rows | 7-10 plots (NEW) |
| `grid_5x2` | 5 columns, 2 rows | 7-10 plots wide layout (NEW) |
| `auto` | Automatic based on plot count | Any number (recommended) |

---

## Project Structure

```
Hydrology/
├── hydrology/                  # Main Python package
│   ├── core/                   # Core utilities
│   │   ├── paths.py           # Portable path management
│   │   ├── logging_setup.py   # Shared logging
│   │   └── config.py          # Configuration loading
│   ├── data/                   # Data fetching & parsing
│   │   ├── usgs.py            # USGS NWIS API
│   │   └── climate.py         # Meteostat climate data
│   ├── analysis/               # Analysis functions
│   │   ├── trends.py          # Mann-Kendall, linear regression
│   │   └── stage_discharge.py # Rating curves, FDC
│   ├── visualization/          # NEW - Modular plotting
│   │   ├── plots.py           # Individual plot functions
│   │   └── composer.py        # Multi-panel layouts
│   └── scripts/                # Executable scripts
│       ├── analyze_sites.py   # NEW - Unified analysis script
│       └── ...
│
├── configs/                    # Configuration files
│   ├── analysis/              # Analysis configurations
│   │   ├── example_single_site.yaml
│   │   ├── example_multi_site.yaml
│   │   └── example_quad_layout.yaml
│   └── sites/                 # Site definitions
│
├── outputs/                    # All outputs (NEW)
│   ├── plots/                 # Generated plots
│   ├── logs/                  # Log files
│   └── cache/                 # Cached data
│
├── current scripts/            # Active scripts being refactored
│   ├── q_analysis_vertical.py # Refactored climate analysis
│   └── ...
│
├── archive/                    # OLD - Preserved for reference
│   ├── deprecated/            # Old scripts with README
│   └── variants/              # Location-specific variants
│
├── setup.py                   # Package installation
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

---

## Features

### Data Sources
- **USGS NWIS**: Streamflow (discharge) and gage height data via WaterML API
- **Meteostat**: Temperature and precipitation data for any location

### Analysis Capabilities
- Correlation analysis (discharge vs temperature, discharge vs precipitation)
- Trend analysis (Mann-Kendall test, linear regression)
- Annual and monthly aggregations
- Lagged precipitation correlation
- Flow duration curves
- Stage-discharge rating curves

### Visualization
- Multi-panel figures with configurable layouts
- Hexbin density plots (log-scale capable)
- Time series plots
- Anomaly comparison plots (recent vs historical)
- Correlation heatmaps with significance testing
- Flow duration curves

---

## Usage Examples

### From Command Line

```bash
# Basic usage
python -m hydrology.scripts.analyze_sites --config my_config.yaml

# Use a pre-made example
python -m hydrology.scripts.analyze_sites --config configs/analysis/example_quad_layout.yaml
```

### From Python

```python
from hydrology.scripts.analyze_sites import run_analysis

# Run analysis
results = run_analysis('configs/analysis/my_config.yaml')
```

### Create Custom Plots

```python
from hydrology.visualization import create_multi_plot, PlotLayout
from hydrology.data.usgs import fetch_discharge_data
from hydrology.data.climate import fetch_climate_data

# Fetch data
df_q = fetch_discharge_data('12422500', '2020-01-01', '2023-12-31')
df_climate = fetch_climate_data(47.6593, -117.4491, ...)

# Create plot
fig = create_multi_plot(
    plots=['anomaly', 'hexbin_temp', 'timeseries'],
    layout=PlotLayout.VERTICAL,
    data={'df_q': df_q, 'df_merged': merged_data},
    site_id='12422500',
    title='My Custom Analysis'
)
```

---

## Dependencies

Core requirements (automatically installed with `pip install -e .`):
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- seaborn >= 0.13.0
- scipy >= 1.10.0
- requests >= 2.31.0
- pymannkendall >= 1.4.3
- meteostat >= 1.6.5
- pyyaml >= 6.0

See `requirements.txt` for complete list.

---

## Development Setup

If you want to develop or modify the package:

1. **Clone and install in editable mode**:
   ```bash
   git clone https://github.com/abstractionisms/Hydroanalysispy
   cd Hydrology
   pip install -e .
   ```

2. **Make changes to the code** - they'll be immediately available

3. **Test your changes**:
   ```bash
   python -m hydrology.scripts.analyze_sites --config configs/analysis/example_single_site.yaml
   ```

---

## Output Locations

All outputs are organized under the `outputs/` directory:

```
outputs/
├── plots/
│   └── <site_id>/
│       └── multi_plot/
│           └── USGS_<site_id>_<layout>_<plots>.png
├── logs/
│   └── analyze_sites.log
└── cache/
    └── (cached data files)
```

Example:
```
outputs/plots/12422500/multi_plot/USGS_12422500_vertical_anomaly_hexbin_temp_lagged_precip.png
```

---

## Migrating from Old Scripts

If you used the old scripts (`q_analysis_vertical.py`, `q_focus_quad.py`, etc.):

1. **Old scripts are preserved** in `archive/deprecated/` with full documentation
2. **Refactored versions** of key scripts are in `current scripts/` and still work
3. **New modular system** provides same functionality with more flexibility

See `archive/deprecated/README.md` for detailed migration guide.

---

## Configuration Guide

### Minimal Configuration

```yaml
analysis_parameters:
  param_cd: "00060"
  start_date: "2000-01-01"
  end_date: "today"

sites_to_process:
  - site_id: "12422500"
    description: "Spokane River at Spokane, WA"
    latitude: 47.6593
    longitude: -117.4491
    enabled: true
```

This uses default plots (`anomaly`, `hexbin_temp`, `lagged_precip`, `timeseries`) and vertical layout.

### Advanced Configuration

```yaml
analysis_parameters:
  param_cd: "00060"
  start_date: "2000-01-01"
  end_date: "today"

  # Global defaults
  plots: [anomaly, timeseries]
  layout: "vertical"
  dpi: 150

sites_to_process:
  # Site 1: Override to use quad layout
  - site_id: "12422500"
    description: "Spokane River at Spokane, WA"
    latitude: 47.6593
    longitude: -117.4491
    enabled: true
    plots: [anomaly, hexbin_temp, lagged_precip, flow_duration]
    layout: "quad"

  # Site 2: Use global defaults
  - site_id: "12424000"
    description: "Hangman Creek at Spokane, WA"
    latitude: 47.6582
    longitude: -117.3960
    enabled: true

  # Site 3: Disabled (won't be processed)
  - site_id: "12419000"
    description: "St. Joe River at Calder, ID"
    latitude: 47.2677
    longitude: -116.1858
    enabled: false
```

See `configs/analysis/README.md` for complete options.

---

## Benefits of New System

### For Users
- **No code editing**: Change sites, plots, layouts via YAML
- **Mix and match**: Any plots in any layout
- **Multi-site**: Process many sites in one run
- **Reproducible**: Config file documents exactly what was done

### For Developers
- **DRY**: Fix bugs once, benefit everywhere
- **Portable**: No hardcoded paths
- **Testable**: Modular functions are easy to test
- **Extensible**: Add new plots without touching layout code

---

## Legacy Scripts

Old scripts are preserved in `archive/deprecated/` with documentation explaining:
- What each script did
- How to replicate with new system
- Where to find example outputs

Key refactored scripts in `current scripts/`:
- `q_analysis_vertical.py` - Climate correlation analysis (refactored from 1021 → 755 lines)
- More scripts being updated incrementally

---

## Contributing

Contributions are welcome! Since this is a spare-time project, responses may be delayed.

To contribute:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with example configs
5. Submit a pull request

---

## License

[Your license here]

---

## Contact

[Your contact info here]

---

## Acknowledgments

- USGS for streamflow data (NWIS API)
- Meteostat for climate data
- Open source Python community for excellent libraries
