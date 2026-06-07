# HydroPlot Architecture Notes

HydroPlot is organized around five user-facing workflows. New features should land in the workflow where the user decision happens, not in a catch-all tools page.

## Workflows

- **Stations** (`overview`): find a stream gage, inspect station map context, live/current condition checks, and record coverage.
- **Site Analysis** (`single-analysis`): one selected gage, hydrographs, duration curves, exports, drought indicators, SPI, and baseflow proxy views.
- **Compare Sites** (`comparisons`): multi-gage overlays, period comparisons, and site relationship analysis.
- **Reach Analysis** (`reach-analysis`): selected upstream/downstream gage pairs, NHD reach map, gain/loss screening, reach plots, baseflow waterfall, and future groundwater observation support.
- **Watershed** (`watershed`): broader basin, HUC, land-cover, dam, flowline, and future well/log context.

## Groundwater Source Rules

Groundwater support should be implemented as public-source tiers with explicit eligibility.

1. **USGS groundwater field measurements**
   - Public monitoring data.
   - Eligible for time-series screening when enough records exist.
   - First implementation target.

2. **Washington Ecology EIM groundwater data**
   - Public Ecology/partner monitoring data when accessed through published search/download/API paths.
   - Eligible only after schema and public access are verified.

3. **Washington Ecology well logs / well reports**
   - Public context data, but locations and private-well details can be sensitive or approximate.
   - Context only unless a record is clearly a monitoring time series with usable public measurements.
   - Do not expose owner, address, parcel, phone/email, or other private fields.

## Groundwater UI Placement

- Primary: **Reach Analysis**
  - Button-driven "Find public groundwater data" for the selected reach.
  - Show monitoring wells separately from context-only well logs.
  - Map markers must identify source and eligibility.
  - Trend/correlation summaries only for eligible monitoring time series.

- Secondary: **Watershed**
  - Basin-scale well/log inventory context.
  - No reach-level interpretation.

- Later/optional: **Site Analysis**
  - Single-gage nearby groundwater context if it supports an actual user workflow.

## Data And Performance Rules

- Do not fetch groundwater data automatically on every Streamlit rerun.
- Cache public-data calls by source, reach bounds, dates, and buffer distance.
- Normalize provider output into a safe schema before UI rendering.
- Drop private fields at the data boundary.
- Label approximate locations and context-only sources in the UI.
- Avoid claiming calibrated groundwater modeling; use "screening", "proxy", and "observation support".

## Verification Pattern

Each new data source needs:

- Unit tests for normalization.
- Eligibility tests that prevent context-only records from being analyzed.
- Fixture/mocked tests for missing values, sparse time series, and private-field removal.
- One validation case in `docs/cases/` once the workflow produces a useful output.

