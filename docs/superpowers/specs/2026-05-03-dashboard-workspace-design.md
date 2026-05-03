# Dashboard Workspace Broad Pass Design

Date: 2026-05-03

## Goal

Move the Streamlit hydrology dashboard toward a coherent analysis workspace while making a broad first pass on performance, navigation, and visualization quality. The pass should preserve the current analytical reach of the app, including multi-site workflows and contextual help.

## Direction

Use a combined "analysis workspace plus command bar" direction:

- A persistent selected-site context should carry across overview, single-site analysis, comparisons, monitoring, and export workflows.
- Navigation should be organized around user intent: Explore, Analyze, Compare, Monitor, and Export.
- The app should expose a global station or HUC search concept and a visible saved-view direction, but full saved-view persistence can be deferred until the core workflow is proven.
- Existing pages can remain in place for this pass, but their entry points and shared controls should feel connected.

## Performance Scope

The first pass should target obvious slowdown sources without rewriting the app:

- Avoid heavy imports at app startup when they are only needed by optional pages or features.
- Keep geospatial, map, advanced statistics, and plotting imports local to the actions that use them.
- Reuse cached site data, availability metadata, and processed discharge data where page reruns currently repeat work.
- Avoid fetching data for analysis panels that are collapsed, unchecked, or not yet requested.
- Keep expensive multi-site processing explicit and bounded by user selection.

## Navigation Scope

The app should make it clear where the user is and what they can do next:

- Provide one consistent selected-site state that site pickers and map clicks update.
- When a user selects a site on the map, make the next action clear and direct, such as opening analysis for that site rather than only showing a passive note.
- Group analysis modes by required context:
  - Single-site analyses: one station with discharge data.
  - Stage analyses: one station with stage data.
  - Climate analyses: one station with discharge plus climate data.
  - Multi-site analyses: two to four compatible stations over a shared date range.
  - Reach analyses: upstream and downstream station pair with overlapping records.
- Make unavailable analyses explain themselves in place instead of disappearing without context.

## Analysis Eligibility Requirements

Each analysis option should carry lightweight metadata that the UI can use before running it:

- Minimum number of sites.
- Maximum number of sites when relevant.
- Required data variables, such as discharge, stage, climate, peak flow, or paired reach data.
- Minimum record length or event count when known.
- Whether the analysis supports single-site, multi-site, reach, or comparison mode.

The UI should use this metadata to show enabled, disabled, or warning states with short explanations. Examples:

- Flood frequency requires enough annual peak records for a meaningful fit.
- 7Q10 and low-flow trend analyses require enough years to avoid misleading output.
- Climate correlation requires climate data availability near the station.
- Reach gain/loss analyses require two stations with overlapping discharge records.

Exact thresholds should follow existing analysis functions where available; otherwise, the first pass should encode conservative defaults and keep the explanation visible.

## Visualization Scope

Visualization changes should improve clarity without removing analytical detail:

- Keep interactive Plotly charts as the default for common workflows.
- Preserve hover tooltips, captions, and help popovers. Do not remove explanatory mouse-over behavior.
- Add or maintain short explanations for plot options, eligibility warnings, statistical methods, and unusual visual encodings.
- Improve chart defaults where safe: clearer titles, consistent axis labels, readable legends, and restrained annotations.
- Keep static matplotlib export available for generated plot grids.
- For multi-site charts, make station identity, record overlap, and date range visually clear.

## Accessibility And Help

The dashboard should remain self-explanatory:

- All non-obvious controls should have `help=` text or nearby captions.
- Disabled analyses should explain the missing requirement.
- Hover behavior should supplement visible labels, not replace them.
- Long method explanations should live in expanders or info popovers so the app stays dense but understandable.

## Deferred Work

The first pass should not attempt:

- Full saved-view persistence.
- Drag-and-drop chart layout.
- A complete rewrite of every page into a new shell.
- New analytical methods beyond what is needed to support eligibility and navigation.

## Testing And Verification

Implementation should include focused tests where logic is extracted into testable helpers, especially for analysis eligibility metadata. Verification should include:

- Existing unit tests.
- Import smoke checks for the Streamlit app modules.
- Manual or browser verification that the main navigation still loads.
- Spot checks that explanatory help text and disabled-analysis explanations are still present.
