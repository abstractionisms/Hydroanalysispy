# Dashboard Workspace Broad Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the Streamlit hydrology dashboard across performance, navigation, analysis eligibility, and visualization clarity while preserving multi-site workflows and explanatory hover/help behavior.

**Architecture:** Add small, testable app-layer modules for analysis eligibility and workspace state, then wire them into existing pages. Keep the current Streamlit multipage structure, avoid a full saved-view rewrite, and use local imports for optional heavy features.

**Tech Stack:** Python, Streamlit, pandas, Plotly, matplotlib, pytest.

---

## File Structure

- Create `hydrology/app/eligibility.py`: metadata and pure helper functions for analysis requirements, record-length checks, and user-facing disabled/warning messages.
- Create `hydrology/app/workspace.py`: Streamlit-facing helpers for persistent selected-site state, query param sync, and a compact sidebar workspace panel.
- Create `tests/test_app_eligibility.py`: unit tests for metadata and eligibility decisions.
- Create `tests/test_app_workspace.py`: unit tests for pure workspace option helpers.
- Modify `hydrology/app/shared.py`: delegate site selection state to `workspace.py`, keep existing site picker behavior, preserve help text.
- Modify `hydrology/app/app.py`: reduce startup imports where possible and add workspace-oriented page labels.
- Modify `hydrology/app/plot_config.py`: attach eligibility metadata and help strings to plot selectors.
- Modify `hydrology/app/pages/overview.py`: make map-selected sites drive global selected-site state and show direct next actions.
- Modify `hydrology/app/pages/single_analysis.py`: use eligibility messages for plot controls and keep explanatory hover/caption text.
- Modify `hydrology/app/pages/comparisons.py`: show multi-site eligibility, overlap, and requirement warnings before expensive processing.
- Modify `hydrology/visualization/interactive.py`: improve multi-site hover labels and chart subtitles without removing existing hover details.

## Task 1: Analysis Eligibility Metadata

**Files:**
- Create: `hydrology/app/eligibility.py`
- Test: `tests/test_app_eligibility.py`

- [ ] **Step 1: Write failing tests for eligibility metadata**

Create `tests/test_app_eligibility.py`:

```python
import pandas as pd

from hydrology.app.eligibility import (
    AnalysisContext,
    get_requirement,
    evaluate_analysis,
    summarize_record,
)


def test_flood_frequency_requires_peak_records():
    context = AnalysisContext(site_count=1, peak_count=8, years=40, has_discharge=True)

    result = evaluate_analysis("flood_frequency", context)

    assert result.enabled is False
    assert "10 annual peak records" in result.reason


def test_low_flow_warns_when_record_is_short():
    context = AnalysisContext(site_count=1, years=7, has_discharge=True)

    result = evaluate_analysis("7q10_analysis", context)

    assert result.enabled is True
    assert result.warning is not None
    assert "10 years" in result.warning


def test_climate_plot_requires_climate_data():
    context = AnalysisContext(site_count=1, years=20, has_discharge=True, has_climate=False)

    result = evaluate_analysis("correlation_matrix", context)

    assert result.enabled is False
    assert "climate data" in result.reason


def test_reach_plot_requires_two_sites_and_overlap():
    context = AnalysisContext(site_count=1, years=20, has_discharge=True, overlap_years=12)

    result = evaluate_analysis("reach_comparison", context)

    assert result.enabled is False
    assert "2 sites" in result.reason


def test_multi_site_accepts_two_to_four_sites():
    context = AnalysisContext(site_count=3, years=12, has_discharge=True, overlap_years=12)

    result = evaluate_analysis("multi_site_overlay", context)

    assert result.enabled is True
    assert result.reason is None


def test_summarize_record_uses_datetime_index_year_span():
    df = pd.DataFrame(
        {"Discharge_cfs": [1.0, 2.0, 3.0]},
        index=pd.to_datetime(["2000-01-01", "2005-01-01", "2010-01-01"]),
    )

    summary = summarize_record(df)

    assert summary["years"] == 11
    assert summary["rows"] == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_app_eligibility.py -v
```

Expected: FAIL because `hydrology.app.eligibility` does not exist.

- [ ] **Step 3: Implement eligibility helpers**

Create `hydrology/app/eligibility.py`:

```python
"""Analysis eligibility metadata for the Streamlit dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class AnalysisRequirement:
    key: str
    label: str
    min_sites: int = 1
    max_sites: Optional[int] = 1
    requires_discharge: bool = True
    requires_stage: bool = False
    requires_climate: bool = False
    requires_peak_records: int = 0
    min_years: int = 0
    warning_years: int = 0
    min_overlap_years: int = 0
    explanation: str = ""


@dataclass(frozen=True)
class AnalysisContext:
    site_count: int = 1
    years: int = 0
    overlap_years: int = 0
    peak_count: int = 0
    has_discharge: bool = False
    has_stage: bool = False
    has_climate: bool = False


@dataclass(frozen=True)
class EligibilityResult:
    enabled: bool
    reason: Optional[str] = None
    warning: Optional[str] = None


ANALYSIS_REQUIREMENTS: dict[str, AnalysisRequirement] = {
    "timeseries": AnalysisRequirement(
        key="timeseries",
        label="Recent Time Series",
        explanation="Requires discharge data for the selected station and date range.",
    ),
    "flow_duration": AnalysisRequirement(
        key="flow_duration",
        label="Flow Duration Curve",
        explanation="Requires discharge data; longer records produce more stable exceedance estimates.",
    ),
    "monthly_boxplot": AnalysisRequirement(
        key="monthly_boxplot",
        label="Monthly Distribution",
        warning_years=3,
        explanation="Best with multiple years so each month has repeated observations.",
    ),
    "annual_trend": AnalysisRequirement(
        key="annual_trend",
        label="Annual Mean Trend",
        warning_years=10,
        explanation="Trend interpretation is more reliable with at least 10 years of discharge data.",
    ),
    "low_flow_trend": AnalysisRequirement(
        key="low_flow_trend",
        label="7-Day Low Flow Trend",
        warning_years=10,
        explanation="Low-flow trends need enough years to avoid over-reading short dry or wet periods.",
    ),
    "7q10_analysis": AnalysisRequirement(
        key="7q10_analysis",
        label="7Q10 Low Flow Analysis",
        warning_years=10,
        explanation="7Q10 is more defensible with at least 10 years of daily discharge records.",
    ),
    "flood_frequency": AnalysisRequirement(
        key="flood_frequency",
        label="Flood Frequency",
        requires_peak_records=10,
        explanation="Requires annual peak streamflow records for distribution fitting.",
    ),
    "rating_curve": AnalysisRequirement(
        key="rating_curve",
        label="Stage-Discharge Rating Curve",
        requires_stage=True,
        explanation="Requires gage height and discharge observations for the same site.",
    ),
    "correlation_matrix": AnalysisRequirement(
        key="correlation_matrix",
        label="Correlation Matrix",
        requires_climate=True,
        warning_years=3,
        explanation="Requires merged discharge and nearby climate observations.",
    ),
    "anomaly": AnalysisRequirement(
        key="anomaly",
        label="Monthly Anomaly Analysis",
        requires_climate=True,
        warning_years=3,
        explanation="Requires discharge and climate data to compare departures from normal.",
    ),
    "hexbin_temp": AnalysisRequirement(
        key="hexbin_temp",
        label="Discharge vs Temperature",
        requires_climate=True,
        explanation="Requires merged discharge and temperature records.",
    ),
    "lagged_precip": AnalysisRequirement(
        key="lagged_precip",
        label="Lagged Precipitation Scatter",
        requires_climate=True,
        explanation="Requires precipitation and discharge records over the same date range.",
    ),
    "multi_site_overlay": AnalysisRequirement(
        key="multi_site_overlay",
        label="Multi-Site Overlay",
        min_sites=2,
        max_sites=4,
        min_overlap_years=1,
        explanation="Requires 2 to 4 stations with overlapping discharge records.",
    ),
    "reach_comparison": AnalysisRequirement(
        key="reach_comparison",
        label="Reach Comparison",
        min_sites=2,
        max_sites=2,
        min_overlap_years=1,
        explanation="Requires upstream and downstream stations with overlapping discharge records.",
    ),
}


def get_requirement(key: str) -> AnalysisRequirement:
    return ANALYSIS_REQUIREMENTS.get(
        key,
        AnalysisRequirement(
            key=key,
            label=key.replace("_", " ").title(),
            explanation="Requires discharge data for the selected context.",
        ),
    )


def evaluate_analysis(key: str, context: AnalysisContext) -> EligibilityResult:
    requirement = get_requirement(key)

    if context.site_count < requirement.min_sites:
        return EligibilityResult(False, f"Requires at least {requirement.min_sites} sites.")

    if requirement.max_sites is not None and context.site_count > requirement.max_sites:
        return EligibilityResult(False, f"Supports no more than {requirement.max_sites} sites.")

    if requirement.requires_discharge and not context.has_discharge:
        return EligibilityResult(False, "Requires discharge data for the selected date range.")

    if requirement.requires_stage and not context.has_stage:
        return EligibilityResult(False, "Requires gage height data at this station.")

    if requirement.requires_climate and not context.has_climate:
        return EligibilityResult(False, "Requires climate data near the selected station.")

    if requirement.requires_peak_records and context.peak_count < requirement.requires_peak_records:
        return EligibilityResult(
            False,
            f"Requires at least {requirement.requires_peak_records} annual peak records.",
        )

    if requirement.min_overlap_years and context.overlap_years < requirement.min_overlap_years:
        return EligibilityResult(
            False,
            f"Requires at least {requirement.min_overlap_years} year of overlapping records.",
        )

    if requirement.min_years and context.years < requirement.min_years:
        return EligibilityResult(False, f"Requires at least {requirement.min_years} years of data.")

    if requirement.warning_years and context.years < requirement.warning_years:
        return EligibilityResult(
            True,
            warning=f"Interpret carefully with less than {requirement.warning_years} years of data.",
        )

    return EligibilityResult(True)


def summarize_record(df: pd.DataFrame | None) -> dict[str, int]:
    if df is None or df.empty:
        return {"rows": 0, "years": 0}

    index = pd.to_datetime(df.index)
    start_year = int(index.min().year)
    end_year = int(index.max().year)
    return {"rows": int(len(df)), "years": max(0, end_year - start_year + 1)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/test_app_eligibility.py -v
```

Expected: PASS for all tests in `tests/test_app_eligibility.py`.

- [ ] **Step 5: Commit**

Run:

```bash
git add hydrology/app/eligibility.py tests/test_app_eligibility.py
git commit -m "feat: add dashboard analysis eligibility metadata"
```

## Task 2: Workspace State Helpers

**Files:**
- Create: `hydrology/app/workspace.py`
- Test: `tests/test_app_workspace.py`
- Modify: `hydrology/app/shared.py`

- [ ] **Step 1: Write failing tests for pure workspace helpers**

Create `tests/test_app_workspace.py`:

```python
import pandas as pd

from hydrology.app.workspace import build_site_options, resolve_default_site_index


def test_build_site_options_uses_id_and_description():
    inventory = pd.DataFrame(
        [{"site_id": "12422500", "description": "Spokane River at Spokane, WA"}]
    )

    options = build_site_options(inventory)

    assert options == ["12422500 - Spokane River at Spokane, WA"]


def test_resolve_default_site_index_prefers_query_site():
    options = ["111 - First", "222 - Second", "333 - Third"]

    index = resolve_default_site_index(options, query_site="333", session_site="222")

    assert index == 2


def test_resolve_default_site_index_falls_back_to_session_site():
    options = ["111 - First", "222 - Second", "333 - Third"]

    index = resolve_default_site_index(options, query_site=None, session_site="222")

    assert index == 1


def test_resolve_default_site_index_uses_zero_for_missing_site():
    options = ["111 - First", "222 - Second"]

    index = resolve_default_site_index(options, query_site="999", session_site=None)

    assert index == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_app_workspace.py -v
```

Expected: FAIL because `hydrology.app.workspace` does not exist.

- [ ] **Step 3: Implement workspace helpers**

Create `hydrology/app/workspace.py`:

```python
"""Workspace state helpers for the Streamlit dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st


GLOBAL_SITE_KEY = "global_last_site"
GLOBAL_START_YEAR_KEY = "global_start_year"
GLOBAL_END_YEAR_KEY = "global_end_year"


def build_site_options(inventory_df: pd.DataFrame) -> list[str]:
    options: list[str] = []
    for _, row in inventory_df.iterrows():
        site_id = str(row["site_id"])
        description = str(row.get("description", ""))[:60]
        options.append(f"{site_id} - {description}")
    return options


def resolve_default_site_index(
    site_options: list[str],
    query_site: str | None,
    session_site: str | None,
) -> int:
    target_site = query_site or session_site
    if target_site:
        for index, option in enumerate(site_options):
            if option.startswith(str(target_site)):
                return index
    return 0


def set_selected_site(site_id: str | None) -> None:
    if not site_id:
        return
    st.session_state[GLOBAL_SITE_KEY] = str(site_id)
    st.query_params["site"] = str(site_id)


def get_selected_site() -> str | None:
    query_site = st.query_params.get("site")
    if query_site:
        return str(query_site)
    site_id = st.session_state.get(GLOBAL_SITE_KEY)
    return str(site_id) if site_id else None


def render_workspace_sidebar() -> None:
    st.sidebar.markdown("### Workspace")
    selected_site = get_selected_site()
    if selected_site:
        st.sidebar.caption(f"Selected site: `{selected_site}`")
    else:
        st.sidebar.caption("Select a site to carry it across pages.")

    with st.sidebar.expander("Saved views", expanded=False):
        st.caption("Saved-view persistence is deferred for this pass.")
        st.markdown("- Spokane low-flow watch")
        st.markdown("- Upstream/downstream comparison")
        st.markdown("- Flood frequency review")
```

- [ ] **Step 4: Run workspace helper tests**

Run:

```bash
pytest tests/test_app_workspace.py -v
```

Expected: PASS for all tests in `tests/test_app_workspace.py`.

- [ ] **Step 5: Wire `shared.site_picker` to workspace helpers**

Modify `hydrology/app/shared.py`:

```python
from hydrology.app.workspace import (
    build_site_options,
    resolve_default_site_index,
    set_selected_site,
)
```

Replace the local `site_options` construction and default-index block inside `site_picker` with:

```python
    site_options = build_site_options(filtered)

    if show_search and (search or state_filter != "All States"):
        container.caption(f"{len(site_options)} sites found")

    if multi:
        kwargs = {"max_selections": max_selections} if max_selections else {}
        selected = container.multiselect(label, site_options, key=f"{key}_multi", **kwargs)
        return [extract_site_id(s) for s in selected]
    else:
        query_site = st.query_params.get("site")
        default_index = resolve_default_site_index(
            site_options,
            query_site=query_site,
            session_site=st.session_state.get("global_last_site"),
        )

        selected = container.selectbox(label, site_options, index=default_index,
                                       key=f"{key}_select")
        site_id = extract_site_id(selected)
        set_selected_site(site_id)
        return site_id
```

- [ ] **Step 6: Run focused tests**

Run:

```bash
pytest tests/test_app_workspace.py tests/test_app_eligibility.py -v
```

Expected: PASS for both test files.

- [ ] **Step 7: Commit**

Run:

```bash
git add hydrology/app/workspace.py hydrology/app/shared.py tests/test_app_workspace.py
git commit -m "feat: add shared dashboard workspace state"
```

## Task 3: App Shell And Startup Performance

**Files:**
- Modify: `hydrology/app/app.py`
- Modify: `hydrology/app/shared.py`

- [ ] **Step 1: Add workspace shell import and reduce direct plot import**

Modify `hydrology/app/app.py`:

```python
from hydrology.app.styles import apply_custom_css, render_footer
from hydrology.app.shared import get_inventory
from hydrology.app.workspace import render_workspace_sidebar
```

Remove:

```python
from hydrology.visualization.plots import AVAILABLE_PLOTS
```

- [ ] **Step 2: Update page grouping labels**

Modify the `pages` dictionary in `hydrology/app/app.py` to:

```python
pages = {
    "Explore": [
        st.Page(overview.show, title="Overview", icon="📊", default=True, url_path="overview"),
    ],
    "Analyze": [
        st.Page(single_analysis.show, title="Single Site", icon="📈", url_path="single-analysis"),
        st.Page(advanced.show, title="Advanced", icon="🔬", url_path="advanced"),
        st.Page(indicators.show, title="Indicators", icon="🌡️", url_path="indicators"),
    ],
    "Compare": [
        st.Page(comparisons.show, title="Multi-Site", icon="🔄", url_path="comparisons"),
        st.Page(reach_analysis.show, title="Reach", icon="🌊", url_path="reach-analysis"),
    ],
    "Monitor": [
        st.Page(alerts.show, title="Alerts", icon="🚨", url_path="alerts"),
    ],
}
```

- [ ] **Step 3: Render workspace sidebar and avoid eager plot-count import**

Replace the sidebar footer block in `hydrology/app/app.py` with:

```python
render_workspace_sidebar()

inventory_df = get_inventory()
st.sidebar.markdown("---")
site_count = len(inventory_df) if not inventory_df.empty else 0
st.sidebar.caption(f"Sites: {site_count}")
```

- [ ] **Step 4: Move optional imports out of `shared.py` top level**

In `hydrology/app/shared.py`, remove top-level imports that are not used by always-rendered controls:

```python
from hydrology.data.nwm import NWMClient, compare_nwm_usgs, get_forecast_skill
from hydrology.analysis.alerts import (
    AlertMonitor, AlertThreshold, create_flood_alert, create_low_flow_alert
)
from hydrology.analysis.multisite import MultiSiteAnalyzer, quick_correlation_check
from hydrology.analysis.flood_events import FloodEventAnalyzer, calculate_event_statistics
from hydrology.data.usgs import fetch_peak_streamflow, get_top_flood_events
from hydrology.data.national_inventory import get_national_inventory, get_region_inventory, get_inventory_summary
from hydrology.core.huc_regions import HUC2_REGIONS, get_region_name, get_region_center, US_CENTER
from hydrology.visualization import create_multi_plot, PlotLayout
from hydrology.visualization.plots import AVAILABLE_PLOTS
```

If any removed symbol is needed by a function in `shared.py`, import it inside that function immediately before use.

- [ ] **Step 5: Run import smoke check**

Run:

```bash
python -c "import hydrology.app.app; import hydrology.app.shared; print('app imports ok')"
```

Expected: prints `app imports ok` and exits with code 0.

- [ ] **Step 6: Commit**

Run:

```bash
git add hydrology/app/app.py hydrology/app/shared.py
git commit -m "perf: lighten dashboard startup and add workspace shell"
```

## Task 4: Plot Selector Eligibility And Help

**Files:**
- Modify: `hydrology/app/plot_config.py`
- Modify: `hydrology/app/pages/single_analysis.py`

- [ ] **Step 1: Import eligibility metadata**

Modify `hydrology/app/plot_config.py`:

```python
from hydrology.app.eligibility import AnalysisContext, evaluate_analysis, get_requirement
```

- [ ] **Step 2: Add helper to render requirement captions**

Add to `hydrology/app/plot_config.py`:

```python
def render_requirement_note(plot_key: str, context: AnalysisContext) -> None:
    requirement = get_requirement(plot_key)
    result = evaluate_analysis(plot_key, context)
    if result.reason:
        st.caption(f"Unavailable: {result.reason}")
    elif result.warning:
        st.caption(f"Use with care: {result.warning}")
    elif requirement.explanation:
        st.caption(requirement.explanation)
```

- [ ] **Step 3: Add eligibility-aware selected plot filtering**

Add to `hydrology/app/plot_config.py`:

```python
def filter_enabled_plots(plot_keys: list[str], context: AnalysisContext) -> list[str]:
    return [key for key in plot_keys if evaluate_analysis(key, context).enabled]
```

- [ ] **Step 4: Use context in `single_analysis.py` static plot selector**

In `hydrology/app/pages/single_analysis.py`, add imports:

```python
from hydrology.app.eligibility import AnalysisContext, summarize_record
from hydrology.app.plot_config import filter_enabled_plots, render_requirement_note
```

Before the static plot grid selector, create:

```python
record_summary = summarize_record(data["df_q"])
analysis_context = AnalysisContext(
    site_count=1,
    years=record_summary["years"],
    has_discharge=data["df_q"] is not None and not data["df_q"].empty,
    has_stage=has_stage,
    has_climate=data["df_merged"] is not None and not data["df_merged"].empty,
)
```

After `selected_plots = plot_selector(...)`, add:

```python
        for plot_key in selected_plots:
            render_requirement_note(plot_key, analysis_context)

        selected_plots = filter_enabled_plots(selected_plots, analysis_context)
```

- [ ] **Step 5: Preserve help text on interactive controls**

Ensure these controls in `single_analysis.py` retain or add `help=`:

```python
agg = st.radio(
    "Aggregation",
    ["daily", "weekly", "monthly"],
    horizontal=True,
    key="hydrograph_agg",
    help="Controls how daily discharge is summarized in the interactive hydrograph.",
)
show_nwm = st.checkbox(
    "NWM forecast overlay",
    value=False,
    key="nwm_overlay",
    help="Adds recent National Water Model streamflow where available.",
)
show_stage = st.checkbox(
    "Show stage (dual axis)",
    value=False,
    disabled=not has_stage,
    key="stage_overlay",
    help="Overlays gage height on a second axis when stage data are available.",
)
koehler = st.checkbox(
    "Koehler (2025) dQ/dt coloring",
    value=True,
    key="fdc_koehler",
    help="Colors flow-duration points by whether discharge was rising, falling, or stable.",
)
```

- [ ] **Step 6: Run tests and import smoke check**

Run:

```bash
pytest tests/test_app_eligibility.py -v
python -c "import hydrology.app.pages.single_analysis; print('single analysis imports ok')"
```

Expected: eligibility tests pass and import smoke check prints `single analysis imports ok`.

- [ ] **Step 7: Commit**

Run:

```bash
git add hydrology/app/plot_config.py hydrology/app/pages/single_analysis.py
git commit -m "feat: show analysis eligibility in plot selection"
```

## Task 5: Overview Map Navigation

**Files:**
- Modify: `hydrology/app/pages/overview.py`
- Modify: `hydrology/app/workspace.py`

- [ ] **Step 1: Add page link helper**

Add to `hydrology/app/workspace.py`:

```python
def render_open_analysis_action(site_id: str) -> None:
    st.link_button(
        "Open analysis for selected site",
        f"/single-analysis?site={site_id}",
        help="Keeps this station selected and opens the single-site analysis workspace.",
    )
```

- [ ] **Step 2: Use workspace state when map markers are clicked**

In `hydrology/app/pages/overview.py`, import:

```python
from hydrology.app.workspace import set_selected_site, render_open_analysis_action
```

Replace the map click success block with:

```python
                set_selected_site(clicked_id)
                st.success(f"Selected: **{clicked_id}** - {site['description']}")
                render_open_analysis_action(clicked_id)
```

- [ ] **Step 3: Add help text to geospatial toggles**

Verify these existing checkbox help strings remain present:

```python
help="Show contributing area polygon (requires HyRiver)"
help="Show upstream flowline traces"
help="Show dams from National Inventory"
```

- [ ] **Step 4: Run import smoke check**

Run:

```bash
python -c "import hydrology.app.pages.overview; print('overview imports ok')"
```

Expected: prints `overview imports ok` and exits with code 0.

- [ ] **Step 5: Commit**

Run:

```bash
git add hydrology/app/pages/overview.py hydrology/app/workspace.py
git commit -m "feat: connect map selection to analysis workspace"
```

## Task 6: Multi-Site Eligibility And Overlap Messaging

**Files:**
- Modify: `hydrology/app/pages/comparisons.py`
- Modify: `hydrology/app/eligibility.py`

- [ ] **Step 1: Add overlap helper**

Add to `hydrology/app/eligibility.py`:

```python
def estimate_overlap_years(frames: list[pd.DataFrame]) -> int:
    valid_frames = [df for df in frames if df is not None and not df.empty]
    if len(valid_frames) < 2:
        return 0

    starts = [pd.to_datetime(df.index).min() for df in valid_frames]
    ends = [pd.to_datetime(df.index).max() for df in valid_frames]
    overlap_start = max(starts)
    overlap_end = min(ends)
    if overlap_end < overlap_start:
        return 0
    return int(overlap_end.year - overlap_start.year + 1)
```

- [ ] **Step 2: Add test for overlap helper**

Append to `tests/test_app_eligibility.py`:

```python
def test_estimate_overlap_years_returns_shared_span():
    from hydrology.app.eligibility import estimate_overlap_years

    a = pd.DataFrame({"Discharge_cfs": [1, 2]}, index=pd.to_datetime(["2000-01-01", "2010-01-01"]))
    b = pd.DataFrame({"Discharge_cfs": [1, 2]}, index=pd.to_datetime(["2005-01-01", "2015-01-01"]))

    assert estimate_overlap_years([a, b]) == 6
```

- [ ] **Step 3: Run test to verify overlap helper passes**

Run:

```bash
pytest tests/test_app_eligibility.py::test_estimate_overlap_years_returns_shared_span -v
```

Expected: PASS.

- [ ] **Step 4: Show multi-site requirement before processing**

In `hydrology/app/pages/comparisons.py`, import:

```python
from hydrology.app.eligibility import AnalysisContext, evaluate_analysis, estimate_overlap_years
```

After `selected_sites` is set in `_compare_sites`, add:

```python
    preflight = evaluate_analysis(
        "multi_site_overlay",
        AnalysisContext(site_count=len(selected_sites), has_discharge=True, overlap_years=1),
    )
    if not preflight.enabled:
        st.info(preflight.reason)
        return
```

- [ ] **Step 5: Show overlap after data load**

After `all_site_data` is built in `_compare_sites`, add:

```python
    overlap_years = estimate_overlap_years([
        data["df_q"] for data in all_site_data.values() if data.get("df_q") is not None
    ])
    overlap_result = evaluate_analysis(
        "multi_site_overlay",
        AnalysisContext(
            site_count=len(all_site_data),
            has_discharge=bool(all_site_data),
            overlap_years=overlap_years,
        ),
    )
    if not overlap_result.enabled:
        st.warning(overlap_result.reason)
        return
    st.caption(f"Shared record overlap: about {overlap_years} year(s)")
```

- [ ] **Step 6: Run tests and import smoke check**

Run:

```bash
pytest tests/test_app_eligibility.py -v
python -c "import hydrology.app.pages.comparisons; print('comparisons imports ok')"
```

Expected: eligibility tests pass and import smoke check prints `comparisons imports ok`.

- [ ] **Step 7: Commit**

Run:

```bash
git add hydrology/app/eligibility.py hydrology/app/pages/comparisons.py tests/test_app_eligibility.py
git commit -m "feat: add multi-site eligibility messaging"
```

## Task 7: Visualization Polish For Multi-Site Charts

**Files:**
- Modify: `hydrology/visualization/interactive.py`

- [ ] **Step 1: Update multi-site hover template**

In `interactive_comparison`, replace the trace `hovertemplate` with:

```python
            hovertemplate=(
                f"<b>{label}</b><br>"
                "%{x|%Y-%m-%d}<br>"
                "Discharge: %{y:,.0f} cfs"
                "<extra></extra>"
            )
```

- [ ] **Step 2: Add range slider and clearer legend placement**

In `interactive_comparison`, update `fig.update_layout(...)` to include:

```python
        xaxis=dict(rangeslider=dict(visible=True), type="date"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
```

Keep the existing `hovermode='x unified'`, log y-axis, and margin settings.

- [ ] **Step 3: Run visualization import smoke check**

Run:

```bash
python -c "from hydrology.visualization.interactive import interactive_comparison; print('interactive imports ok')"
```

Expected: prints `interactive imports ok` and exits with code 0.

- [ ] **Step 4: Commit**

Run:

```bash
git add hydrology/visualization/interactive.py
git commit -m "style: clarify multi-site interactive chart hovers"
```

## Task 8: Full Verification

**Files:**
- Verify only.

- [ ] **Step 1: Run unit tests**

Run:

```bash
pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Run app import smoke checks**

Run:

```bash
python -c "import hydrology.app.app; import hydrology.app.pages.overview; import hydrology.app.pages.single_analysis; import hydrology.app.pages.comparisons; print('dashboard imports ok')"
```

Expected: prints `dashboard imports ok` and exits with code 0.

- [ ] **Step 3: Start dashboard manually for visual check**

Run:

```bash
streamlit run hydrology/app/app.py
```

Expected: Streamlit starts and prints a local URL. Check that the sidebar shows Workspace, pages are grouped as Explore, Analyze, Compare, and Monitor, hover/help text still appears on analysis controls, and multi-site comparison shows requirement/overlap messaging.

- [ ] **Step 4: Capture final status**

Run:

```bash
git status --short
```

Expected: only intentional uncommitted changes remain. If pre-existing user changes are still present, list them separately and do not revert them.
