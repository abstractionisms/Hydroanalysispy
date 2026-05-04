# Dashboard Visual UX Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make HydroPlot feel like a polished hydrology workspace: map-first, visually rich, clear in navigation, and still fully configurable for advanced plot generation.

**Architecture:** Keep Streamlit as the app shell, but move page composition into reusable workspace components. Centralize visual primitives in `hydrology/app/styles.py` and small helper renderers so Overview, Single Analysis, Current Check, and Climate Indicators share the same visual language without duplicating HTML.

**Tech Stack:** Streamlit 1.37, Folium/streamlit-folium, Plotly, pandas, existing `hydrology.app` helpers, pytest.

---

## File Structure

- Modify `hydrology/app/styles.py`: visual system tokens, hero/workspace components, navigation cards, status cards, selected-site panel, layer state chips.
- Modify `hydrology/app/streamlit_app.py`: app-level shell copy and placement of global navigation/workspace modules.
- Modify `hydrology/app/page_modules/overview.py`: map-first station workspace, selected-site panel, map layer control/status model, quick actions.
- Modify `hydrology/app/page_modules/single_analysis.py`: polished analysis header and plot-builder presentation.
- Modify `hydrology/app/plot_config.py`: expose plot preset metadata for card rendering.
- Modify `hydrology/app/shared.py`: selected-site summary helper and optional current-condition detail helper if the Overview page needs a single stable data contract.
- Create `tests/test_app_visual_shell.py`: protects navigation labels, shell components, and no-sidebar workflow from regression.
- Create `tests/test_app_overview_workspace.py`: pure tests for selected-site panel data shaping and layer state labels.
- Extend `tests/test_app_plot_config.py`: verifies preset metadata supports card-based UI without losing full catalog access.

---

## Phase 1: Visual System Foundation

### Task 1: Add Reusable Workspace UI Components

**Files:**
- Modify: `hydrology/app/styles.py`
- Test: `tests/test_app_visual_shell.py`

- [ ] **Step 1: Write failing tests for reusable visual components**

Create `tests/test_app_visual_shell.py` with:

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_styles_define_workspace_components():
    text = (ROOT / "hydrology/app/styles.py").read_text(encoding="utf-8")

    assert "def render_workspace_panel" in text
    assert "def render_status_chips" in text
    assert "def render_action_cards" in text
    assert ".workspace-panel" in text
    assert ".status-chip" in text
    assert ".action-card" in text


def test_visual_system_avoids_sidebar_first_language():
    text = (ROOT / "hydrology/app/styles.py").read_text(encoding="utf-8")

    assert "Search in the sidebar" not in text
    assert "sidebar is not the primary" in text
```

- [ ] **Step 2: Run tests to verify RED**

Run: `pytest tests/test_app_visual_shell.py -q`

Expected: FAIL because `render_workspace_panel`, `render_status_chips`, and `render_action_cards` do not exist yet.

- [ ] **Step 3: Add component CSS and render helpers**

In `hydrology/app/styles.py`, add CSS inside `apply_custom_css()`:

```css
.workspace-panel {
    border: 1px solid var(--hydro-border);
    border-radius: 8px;
    background: linear-gradient(180deg, rgba(10, 21, 36, 0.92), rgba(7, 17, 29, 0.88));
    padding: 0.95rem;
}

.workspace-panel h3 {
    margin: 0 0 0.25rem 0;
    font-size: 1rem;
    color: var(--hydro-text);
}

.workspace-panel p {
    margin: 0;
    color: var(--hydro-muted);
    font-size: 0.82rem;
}

.status-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    margin-top: 0.65rem;
}

.status-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    border-radius: 999px;
    padding: 0.22rem 0.55rem;
    font-size: 0.72rem;
    font-weight: 700;
    border: 1px solid rgba(118, 169, 192, 0.22);
    color: var(--hydro-text);
    background: rgba(255, 255, 255, 0.055);
}

.status-chip.ready { border-color: rgba(78, 205, 196, 0.48); }
.status-chip.limited { border-color: rgba(255, 193, 7, 0.55); }
.status-chip.blocked { border-color: rgba(255, 107, 107, 0.55); }

.action-card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 0.55rem;
    margin: 0.75rem 0;
}

.action-card {
    display: block;
    border: 1px solid rgba(118, 169, 192, 0.20);
    border-radius: 8px;
    padding: 0.75rem;
    background: rgba(12, 21, 34, 0.72);
    color: var(--hydro-text);
    text-decoration: none;
}

.action-card:hover {
    border-color: rgba(78, 205, 196, 0.62);
    background: rgba(78, 205, 196, 0.11);
    text-decoration: none;
}

.action-card strong {
    display: block;
    font-size: 0.9rem;
}

.action-card span {
    display: block;
    margin-top: 0.25rem;
    color: var(--hydro-muted);
    font-size: 0.76rem;
    line-height: 1.25;
}
```

Add helper functions near the existing render helpers:

```python
def render_workspace_panel(title: str, body: str, chips: list[dict] | None = None):
    """Render a reusable workspace panel."""
    chip_html = ""
    if chips:
        chip_html = '<div class="status-chip-row">' + "".join(
            f'<span class="status-chip {chip.get("state", "ready")}">{chip["label"]}</span>'
            for chip in chips
        ) + "</div>"
    st.markdown(
        f'<section class="workspace-panel"><h3>{title}</h3><p>{body}</p>{chip_html}</section>',
        unsafe_allow_html=True,
    )


def render_status_chips(chips: list[dict]):
    """Render standalone status chips."""
    if not chips:
        return
    st.markdown(
        '<div class="status-chip-row">' + "".join(
            f'<span class="status-chip {chip.get("state", "ready")}">{chip["label"]}</span>'
            for chip in chips
        ) + "</div>",
        unsafe_allow_html=True,
    )


def render_action_cards(cards: list[dict]):
    """Render action cards that link to app pages."""
    html = []
    for card in cards:
        html.append(
            f'<a class="action-card" href="{card["href"]}" target="_self">'
            f'<strong>{card["title"]}</strong><span>{card["body"]}</span></a>'
        )
    st.markdown('<div class="action-card-grid">' + "".join(html) + "</div>", unsafe_allow_html=True)
```

- [ ] **Step 4: Run component tests**

Run: `pytest tests/test_app_visual_shell.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hydrology/app/styles.py tests/test_app_visual_shell.py
git commit -m "style: add reusable workspace visual components"
```

---

## Phase 2: Replace Four Tiles With Full Workspace Action Strip

### Task 2: Upgrade Workflow Strip to Seven Role-Specific Cards

**Files:**
- Modify: `hydrology/app/styles.py`
- Modify: `tests/test_app_navigation_labels.py`

- [ ] **Step 1: Extend navigation-label tests**

Update `tests/test_app_navigation_labels.py`:

```python
def test_workflow_tiles_cover_all_navigation_roles():
    text = (ROOT / "hydrology/app/styles.py").read_text(encoding="utf-8")

    for label in [
        "<strong>Stations</strong>",
        "<strong>Site Analysis</strong>",
        "<strong>Compare Sites</strong>",
        "<strong>Reach Tools</strong>",
        "<strong>Current Check</strong>",
        "<strong>Climate Indicators</strong>",
        "<strong>More Tools</strong>",
    ]:
        assert label in text
```

- [ ] **Step 2: Run test to verify RED**

Run: `pytest tests/test_app_navigation_labels.py -q`

Expected: FAIL because the workflow strip currently has only four cards.

- [ ] **Step 3: Modify workflow strip**

Replace `render_workflow_strip()` in `hydrology/app/styles.py` with seven cards:

```python
def render_workflow_strip():
    """Render visible navigation intent without replacing Streamlit navigation."""
    render_action_cards([
        {"title": "Stations", "href": "overview", "body": "Find gages, inspect map layers, live flow context, and record coverage."},
        {"title": "Site Analysis", "href": "single-analysis", "body": "Run guided or fully custom plot sets for one selected gage."},
        {"title": "Compare Sites", "href": "comparisons", "body": "Check overlap and contrast records before heavier multi-site work."},
        {"title": "Reach Tools", "href": "reach-analysis", "body": "Evaluate paired gages, gain/loss patterns, and reach behavior."},
        {"title": "Current Check", "href": "alerts", "body": "Run manual threshold checks against latest available readings."},
        {"title": "Climate Indicators", "href": "indicators", "body": "Review drought, SPI, precipitation, and climate-linked signals."},
        {"title": "More Tools", "href": "advanced", "body": "Open specialized analysis utilities without cluttering core workflows."},
    ])
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_app_navigation_labels.py tests/test_app_visual_shell.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hydrology/app/styles.py tests/test_app_navigation_labels.py
git commit -m "style: expand workflow cards to full workspace navigation"
```

---

## Phase 3: Map-First Overview Workspace

### Task 3: Build Selected-Site Summary Model

**Files:**
- Modify: `hydrology/app/shared.py`
- Test: `tests/test_app_overview_workspace.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_app_overview_workspace.py`:

```python
from hydrology.app.shared import build_site_summary


def test_build_site_summary_formats_core_fields():
    site_info = {
        "site_id": "12422500",
        "description": "Spokane River at Spokane, WA",
        "latitude": 47.6593,
        "longitude": -117.4491,
        "begin_date": "1891-01-01",
    }
    condition = {
        "flow_cfs": 5775.4,
        "percentile": 82.5,
        "source": "USGS seasonal percentile",
    }

    summary = build_site_summary("12422500", site_info, condition)

    assert summary["title"] == "Spokane River at Spokane, WA"
    assert summary["subtitle"] == "USGS 12422500 | 47.6593, -117.4491"
    assert summary["chips"] == [
        {"label": "Flow 5,775 cfs", "state": "ready"},
        {"label": "Above Normal", "state": "ready"},
        {"label": "Record since 1891", "state": "ready"},
    ]
```

- [ ] **Step 2: Run test to verify RED**

Run: `pytest tests/test_app_overview_workspace.py -q`

Expected: FAIL because `build_site_summary` does not exist.

- [ ] **Step 3: Implement summary helper**

Add to `hydrology/app/shared.py` near cached data helpers:

```python
def build_site_summary(site_id: str, site_info: dict, condition: dict | None = None) -> dict:
    """Build display-ready selected-site summary data."""
    from hydrology.visualization.map_utils import get_condition_label

    condition = condition or {}
    desc = site_info.get("description") or site_id
    lat = site_info.get("latitude")
    lon = site_info.get("longitude")
    subtitle = f"USGS {site_id}"
    if lat is not None and lon is not None:
        subtitle = f"{subtitle} | {float(lat):.4f}, {float(lon):.4f}"

    chips = []
    flow = condition.get("flow_cfs")
    if flow is not None:
        chips.append({"label": f"Flow {flow:,.0f} cfs", "state": "ready"})

    pctile = condition.get("percentile")
    if pctile is not None:
        chips.append({"label": get_condition_label(pctile), "state": "ready"})

    begin_date = str(site_info.get("begin_date") or "")
    if len(begin_date) >= 4 and begin_date[:4].isdigit():
        chips.append({"label": f"Record since {begin_date[:4]}", "state": "ready"})

    return {
        "title": desc,
        "subtitle": subtitle,
        "chips": chips,
    }
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_app_overview_workspace.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hydrology/app/shared.py tests/test_app_overview_workspace.py
git commit -m "feat: add selected site summary model"
```

### Task 4: Recompose Overview as Map-First Workspace

**Files:**
- Modify: `hydrology/app/page_modules/overview.py`
- Modify: `hydrology/app/styles.py`
- Test: `tests/test_app_sidebar_layout.py`

- [ ] **Step 1: Add regression expectations**

Extend `tests/test_app_sidebar_layout.py`:

```python
def test_overview_uses_map_first_workspace_copy():
    text = (ROOT / "hydrology/app/page_modules/overview.py").read_text(encoding="utf-8")

    assert "Station Workspace" in text
    assert "Selected Site" in text
    assert "Map Layers" in text
```

- [ ] **Step 2: Run test to verify RED**

Run: `pytest tests/test_app_sidebar_layout.py -q`

Expected: FAIL until Overview copy and section labels are updated.

- [ ] **Step 3: Update Overview layout**

In `hydrology/app/page_modules/overview.py`:

- Replace the initial `Site Workspace` heading with `Station Workspace`.
- Use two columns before map rendering:
  - left: site search and selected-site panel
  - right: quick actions
- Move `_render_station_map(inventory_df, site_id)` directly below that workspace, before quick stats and deeper text blocks.
- Use `build_site_summary()` and `render_workspace_panel()`:

```python
from hydrology.app.shared import build_site_summary, get_site_condition_details
from hydrology.app.styles import render_workspace_panel, render_action_cards
```

Render:

```python
condition = get_site_condition_details([site_id]).get(site_id, {})
summary = build_site_summary(site_id, site_info, condition)
render_workspace_panel("Selected Site", summary["subtitle"], summary["chips"])
render_action_cards([
    {"title": "Open Site Analysis", "href": f"single-analysis?site={site_id}", "body": "Run guided plots and static export grids."},
    {"title": "Compare Sites", "href": f"comparisons?site={site_id}", "body": "Check overlap against nearby or selected gages."},
    {"title": "Current Check", "href": f"alerts?site={site_id}", "body": "Run manual threshold checks for this gage."},
])
```

- [ ] **Step 4: Rename map layer expander**

In `_render_station_map()`, change:

```python
with st.expander("Optional slow map layers", expanded=False):
```

to:

```python
with st.expander("Map Layers", expanded=False):
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_app_sidebar_layout.py tests/test_app_overview_workspace.py -q`

Expected: PASS.

- [ ] **Step 6: Run app smoke test**

Run: `python -m py_compile hydrology/app/page_modules/overview.py hydrology/app/shared.py hydrology/app/styles.py`

Expected: no output.

- [ ] **Step 7: Commit**

```bash
git add hydrology/app/page_modules/overview.py hydrology/app/shared.py hydrology/app/styles.py tests/test_app_sidebar_layout.py tests/test_app_overview_workspace.py
git commit -m "feat: make overview a map-first station workspace"
```

---

## Phase 4: Layer Status and Slow-Service Feedback

### Task 5: Add Map Layer State Labels

**Files:**
- Modify: `hydrology/app/page_modules/overview.py`
- Test: `tests/test_app_overview_workspace.py`

- [ ] **Step 1: Write failing test**

Add:

```python
from hydrology.app.page_modules.overview import build_layer_status


def test_build_layer_status_labels_requested_layers():
    status = build_layer_status(
        show_boundary=True,
        show_flowlines=True,
        show_dams=False,
        has_pynhd=True,
        has_pygeohydro=True,
    )

    assert status == [
        {"label": "Boundary requested", "state": "limited"},
        {"label": "Flowlines requested", "state": "limited"},
        {"label": "Dams off", "state": "blocked"},
    ]
```

- [ ] **Step 2: Run test to verify RED**

Run: `pytest tests/test_app_overview_workspace.py -q`

Expected: FAIL because `build_layer_status` does not exist.

- [ ] **Step 3: Implement pure helper**

Add to `hydrology/app/page_modules/overview.py`:

```python
def build_layer_status(show_boundary: bool, show_flowlines: bool, show_dams: bool, has_pynhd: bool, has_pygeohydro: bool) -> list[dict]:
    """Build display-ready map layer status chips."""
    return [
        {"label": "Boundary requested" if show_boundary else ("Boundary unavailable" if not has_pynhd else "Boundary off"), "state": "limited" if show_boundary else "blocked"},
        {"label": "Flowlines requested" if show_flowlines else ("Flowlines unavailable" if not has_pynhd else "Flowlines off"), "state": "limited" if show_flowlines else "blocked"},
        {"label": "Dams requested" if show_dams else ("Dams unavailable" if not has_pygeohydro else "Dams off"), "state": "limited" if show_dams else "blocked"},
    ]
```

- [ ] **Step 4: Render status chips near map controls**

Import `render_status_chips` from `hydrology.app.styles` and render:

```python
render_status_chips(build_layer_status(show_boundary, show_flowlines, show_dams, has_pynhd, has_pygeohydro))
```

immediately after the `Map Layers` expander.

- [ ] **Step 5: Run tests and compile**

Run:

```bash
pytest tests/test_app_overview_workspace.py -q
python -m py_compile hydrology/app/page_modules/overview.py
```

Expected: PASS and no compile output.

- [ ] **Step 6: Commit**

```bash
git add hydrology/app/page_modules/overview.py tests/test_app_overview_workspace.py
git commit -m "feat: show map layer status chips"
```

---

## Phase 5: Make Plot Builder Visually Premium Without Losing Optionality

### Task 6: Convert Plot Presets to Card Metadata

**Files:**
- Modify: `hydrology/app/plot_config.py`
- Modify: `tests/test_app_plot_config.py`

- [ ] **Step 1: Extend plot config tests**

Add:

```python
def test_plot_presets_have_card_metadata():
    for preset_name, preset in plot_config.PLOT_PRESETS.items():
        assert preset_name
        assert preset["description"]
        assert "intent" in preset
        assert "plots" in preset
```

- [ ] **Step 2: Run test to verify RED**

Run: `pytest tests/test_app_plot_config.py -q`

Expected: FAIL because presets do not have `intent`.

- [ ] **Step 3: Add intent metadata**

Update each preset in `hydrology/app/plot_config.py`:

```python
"Quick site summary": {
    "intent": "First-pass read",
    "description": "A compact read on recent behavior, distribution, seasonality, and trend.",
    "plots": ["timeseries", "flow_duration", "monthly_boxplot", "annual_trend"],
},
```

Use:
- `Manual selection`: `intent: "Power user"`
- `Flood frequency`: `intent: "High-flow risk"`
- `Drought / low-flow`: `intent: "Low-flow risk"`
- `Climate relationship`: `intent: "Weather response"`
- `Compare sites`: `intent: "Cross-gage context"`

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_app_plot_config.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hydrology/app/plot_config.py tests/test_app_plot_config.py
git commit -m "feat: add plot preset card metadata"
```

### Task 7: Render Plot Builder With Preset Cards and Selected Summary

**Files:**
- Modify: `hydrology/app/plot_config.py`
- Modify: `hydrology/app/styles.py`
- Test: `tests/test_app_plot_config.py`

- [ ] **Step 1: Add test for selected summary helper**

Add:

```python
def test_describe_selected_plots_groups_counts():
    summary = plot_config.describe_selected_plots(["timeseries", "flood_frequency", "lag_correlation"])

    assert summary == "3 selected: Flow behavior, Extremes / frequency, Climate linkage"
```

- [ ] **Step 2: Run test to verify RED**

Run: `pytest tests/test_app_plot_config.py -q`

Expected: FAIL because `describe_selected_plots` does not exist.

- [ ] **Step 3: Implement `describe_selected_plots`**

Add to `hydrology/app/plot_config.py`:

```python
def describe_selected_plots(selected_plots: List[str]) -> str:
    """Return a compact selected plot summary."""
    if not selected_plots:
        return "No plots selected"

    labels = []
    for group in PURPOSE_GROUPS:
        if any(plot in selected_plots for plot in group["plots"]):
            labels.append(group["label"])

    return f"{len(selected_plots)} selected: " + ", ".join(labels)
```

- [ ] **Step 4: Render summary in `multi_plot_selector`**

After `selected` is finalized, add:

```python
st.caption(describe_selected_plots(selected))
```

- [ ] **Step 5: Add CSS for preset cards only if using HTML cards**

If the radio remains, do not add unused CSS. If replacing radio with cards is not practical in Streamlit without custom JS, keep radio and make visual improvement through copy, grouping, and selected summary.

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_app_plot_config.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add hydrology/app/plot_config.py tests/test_app_plot_config.py
git commit -m "feat: summarize selected plot builder output"
```

---

## Phase 6: Visual QA and Preview

### Task 8: Run Verification and Relaunch Preview

**Files:**
- No code files unless defects are found.

- [ ] **Step 1: Run full tests**

Run: `pytest -q`

Expected: all tests pass.

- [ ] **Step 2: Compile touched app modules**

Run:

```bash
python -m py_compile hydrology/app/styles.py hydrology/app/streamlit_app.py hydrology/app/page_modules/overview.py hydrology/app/page_modules/single_analysis.py hydrology/app/plot_config.py hydrology/app/shared.py
```

Expected: no output.

- [ ] **Step 3: Restart preview**

Run:

```powershell
Stop-Process -Id <current_streamlit_pid> -Force
Start-Process -FilePath "C:\Users\Cam\anaconda3\python.exe" -ArgumentList @("-m","streamlit","run","hydrology\app\streamlit_app.py","--server.port","55645","--server.address","0.0.0.0","--server.headless","true") -WorkingDirectory "C:\Users\Cam\source\repos\Hydrology\Hydrology\.worktrees\current-cloud" -WindowStyle Hidden -PassThru
```

Expected: a new Python process is returned.

- [ ] **Step 4: Check HTTP**

Run:

```powershell
Invoke-WebRequest -Uri http://127.0.0.1:55645 -UseBasicParsing -TimeoutSec 20
```

Expected: `StatusCode` is `200`.

- [ ] **Step 5: Commit any verification fixes**

If fixes were needed:

```bash
git add <changed-files>
git commit -m "fix: polish dashboard visual ux upgrade"
```

---

## Self-Review

- Spec coverage: Covers visual system, seven-card workflow strip, map-first Overview, selected-site panel, layer status feedback, and plot-builder polish while preserving full plot optionality.
- Completeness scan: No unresolved markers or unspecified implementation steps remain. Each code task includes concrete tests, implementation snippets, commands, and expected results.
- Type consistency: Helper names are consistent across tasks: `render_workspace_panel`, `render_status_chips`, `render_action_cards`, `build_site_summary`, `build_layer_status`, and `describe_selected_plots`.

---

## Execution Options

Plan complete and saved to `docs/superpowers/plans/2026-05-04-dashboard-visual-ux-upgrade.md`.

**1. Subagent-Driven (recommended)** - dispatch a focused worker per phase, review between phases, fastest path with checkpoints.

**2. Inline Execution** - execute the phases in this session with explicit checkpoints after each task.
