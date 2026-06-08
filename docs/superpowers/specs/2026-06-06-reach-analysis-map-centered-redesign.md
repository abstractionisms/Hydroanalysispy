# Reach Analysis Map-Centered Redesign

Date: 2026-06-06

## Purpose

Reach Analysis should feel like one gage-centered workspace, not a stack of disconnected expanders. The selected reach, processable candidate gages, NHD river context, and analysis readiness must be visible together so a user can understand what they are evaluating before generating plots.

## Approved Direction

Use the Map-Centered Workspace layout:

- A compact top row selects state, searches for one anchor gage, controls network search radius, toggles tributary discovery, and runs candidate discovery.
- The main pane is always visible and centered on the selected reach.
- Processable candidate reaches are shown next to the map as recommended pairs, not as a large default table.
- Selected reach summary sits beside the map with upstream/downstream IDs, inferred network length, data readiness, and a primary Run Analysis button.
- Raw candidate tables, plot descriptions, manual length override, DPI, and layout controls are collapsed behind advanced/detail disclosures.

## Non-Negotiable UX Requirements

- Use `gage`, not `gage`, in visible text.
- Do not make the user type reach length for normal cases. Infer network length from NLDI navigation distances when available. Manual length stays advanced-only and only applies when inference fails.
- Do not show unprocessable candidate gages in the primary picker. NLDI sites outside the HydroPlot inventory can be counted or exposed in details, but they should not be default choices.
- Do not bury the reach map inside a collapsed expander. The map is the main workspace.
- The reach map must auto-fit the selected upstream/downstream gage coordinates. It should not open zoomed out far enough to show repeated world tiles.
- The selected mainstem reach should be visually dominant. Tributary and nearby NHD context should be present but secondary.
- The top of the page should not present countless choices before the user understands the selected reach.

## Layout

### Top Search Bar

Replace the current separate "Select reach", "Candidate gages", "Selected reach", and "Run analysis" sections with a compact control row:

- State selector.
- Single gage search text input.
- Anchor gage selector filtered by state/search.
- Search radius control with a conservative default.
- Tributaries toggle.
- Find Reaches button.

The search row should be dense and scan-friendly. It should not include plot settings, DPI, manual length, or raw table controls.

### Main Workspace

Render a three-column workspace after an anchor gage is selected:

- Left: candidate reach list.
- Center: NHD reach map.
- Right: selected reach summary and Run Analysis action.

The candidate list should prefer recommended upstream/downstream pair cards sorted by network position and distance. The anchor gage can be one endpoint. Tributary candidates should appear as context or explicit alternative choices, not silently mixed into mainstem defaults.

### Reach Map

The map should be visible by default and should mount outside collapsed Streamlit expanders.

Map behavior:

- Fit bounds to the selected upstream/downstream gage coordinates, with enough padding to see reach context.
- Disable repeated world wrapping where supported by the tile layer or map options.
- Draw selected reach flowlines in a strong highlight.
- Draw tributary/context flowlines in a muted secondary style.
- Mark upstream and downstream gages with distinct colors and labels.
- When flowline geometry is unavailable, still fit to the two gage coordinates and clearly show both markers.

### Advanced Details

Move lower-priority controls into collapsed disclosures below the workspace:

- Full candidate gage table.
- Omitted NLDI sites not present in HydroPlot inventory.
- Manual reach length override.
- Plot selection and plot descriptions.
- Export/DPI/layout controls.

## Data Flow

1. Load HydroPlot inventory.
2. Filter anchor gage options by state and search text.
3. On Find Reaches, call NLDI related-site discovery for both directions with optional tributaries.
4. Filter discovered sites to processable HydroPlot inventory before primary display.
5. Build candidate reach options from anchor plus processable upstream/downstream/tributary sites.
6. Default selected reach should be a valid processable pair, preferring an upstream-to-anchor or anchor-to-downstream mainstem pair when available.
7. Estimate network length from NLDI signed distances.
8. Render map and selected reach summary before analysis generation.
9. On Run Analysis, fetch paired discharge data, compute gain/loss summary, then render plots and baseflow proxy outputs below the workspace.

## Verification

Unit tests should cover helper behavior without adding expensive runtime checks:

- Candidate filtering excludes non-inventory sites from primary options.
- Default reach selection chooses different upstream/downstream gages when possible.
- Network length inference works from signed NLDI distances.
- Map bounds are computed from selected gage coordinates, not full flowline extent.
- Map component key changes when selected pair or bounds changes.

Manual verification should be done in Streamlit:

- Select a Spokane River anchor gage.
- Find reaches with tributaries enabled.
- Confirm only processable primary candidates appear.
- Select different candidate pairs and confirm upstream/downstream do not reset to the anchor.
- Confirm the map remains visible and zooms to the selected reach.
- Confirm selected reach highlight is visually stronger than tributary/context lines.
- Confirm Run Analysis produces the automated summary and plots below the workspace.

## Out Of Scope For This Pass

- Full drag-and-drop candidate assignment.
- Global expansion beyond PNW inventory.
- A physically based groundwater model.
- QUAL2K, TTools canopy modeling, or QAPP-derived regulatory workflows.
- Rebuilding all app navigation outside Reach Analysis.

Those can be separate feature specs after the core Reach Analysis workflow is usable.
