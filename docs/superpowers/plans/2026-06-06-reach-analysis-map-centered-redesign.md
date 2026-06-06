# Reach Analysis Map-Centered Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild Reach Analysis as a map-centered gage workflow where candidate reach selection, NHD context, inferred reach length, and analysis readiness are visible together.

**Architecture:** Keep the first pass scoped to `hydrology/app/page_modules/reach_analysis.py` plus focused helper tests. Do not introduce a new frontend framework or long-running runtime checks. Extract small pure helpers only where they make Streamlit layout and selection state easier to test.

**Tech Stack:** Streamlit, streamlit-folium, Folium/Leaflet, pandas, pytest, existing HydroPlot inventory/NLDI helpers.

---

## File Structure

- Modify: `hydrology/app/page_modules/reach_analysis.py`
  - Add pure helpers for recommended reach pairs, selected-pair state, candidate labels, map creation, and compact summary rows.
  - Restructure `show()` so search controls, candidate list, map, and selected reach summary render as one workspace.
  - Move plot settings and manual length override into collapsed advanced sections.
- Modify: `tests/test_app_reach_analysis.py`
  - Add fast unit tests for recommended pair selection, non-resetting selected pair state, map bounds/key behavior, and user-facing labels.
- No new production dependency.
- No full app navigation rewrite in this pass.

---

### Task 1: Add Reach Candidate Pair Helpers

**Files:**
- Modify: `hydrology/app/page_modules/reach_analysis.py`
- Test: `tests/test_app_reach_analysis.py`

- [ ] **Step 1: Write failing tests for pair recommendations**

Add these imports to `tests/test_app_reach_analysis.py`:

```python
from hydrology.app.page_modules.reach_analysis import (
    _build_recommended_reach_pairs,
    _pair_key,
)
```

Add tests:

```python
def test_build_recommended_reach_pairs_prefers_mainstem_pairs():
    candidates = [
        {"site_id": "anchor", "position": "Anchor", "distance_km": 0.0, "label": "Anchor | anchor | 0.0 km | Anchor"},
        {"site_id": "up", "position": "Upstream", "distance_km": 5.0, "label": "Upstream | up | 5.0 km | Upstream"},
        {"site_id": "trib", "position": "Tributary", "distance_km": 3.0, "label": "Tributary | trib | 3.0 km | Tributary"},
        {"site_id": "down", "position": "Downstream", "distance_km": 8.0, "label": "Downstream | down | 8.0 km | Downstream"},
    ]

    pairs = _build_recommended_reach_pairs("anchor", candidates, max_pairs=5)

    assert pairs[0]["upstream_id"] == "up"
    assert pairs[0]["downstream_id"] == "anchor"
    assert pairs[1]["upstream_id"] == "anchor"
    assert pairs[1]["downstream_id"] == "down"
    assert all(pair["upstream_id"] != pair["downstream_id"] for pair in pairs)
    assert any(pair["kind"] == "tributary context" for pair in pairs)


def test_pair_key_is_stable_and_readable():
    assert _pair_key("12419000", "12422000") == "12419000__12422000"
```

- [ ] **Step 2: Run failing tests**

Run:

```bash
python -m pytest tests/test_app_reach_analysis.py::test_build_recommended_reach_pairs_prefers_mainstem_pairs tests/test_app_reach_analysis.py::test_pair_key_is_stable_and_readable -q
```

Expected: fails because `_build_recommended_reach_pairs` and `_pair_key` are not defined.

- [ ] **Step 3: Implement minimal helpers**

Add to `hydrology/app/page_modules/reach_analysis.py` near existing candidate helpers:

```python
def _pair_key(upstream_id, downstream_id):
    """Return a stable selected-reach key."""
    return f"{upstream_id}__{downstream_id}"


def _build_recommended_reach_pairs(origin_site_id, candidates, max_pairs=8):
    """Build processable upstream/downstream reach pairs for the workspace."""
    origin_site_id = str(origin_site_id)
    by_position = {"Upstream": [], "Downstream": [], "Tributary": []}
    for candidate in candidates:
        site_id = str(candidate.get("site_id", ""))
        if not site_id or site_id == origin_site_id:
            continue
        position = candidate.get("position")
        if position in by_position:
            by_position[position].append(candidate)

    def distance_value(candidate):
        distance = candidate.get("distance_km")
        return float(distance) if distance is not None else 9999.0

    for values in by_position.values():
        values.sort(key=distance_value)

    pairs = []
    for upstream in by_position["Upstream"]:
        pairs.append({
            "key": _pair_key(upstream["site_id"], origin_site_id),
            "upstream_id": str(upstream["site_id"]),
            "downstream_id": origin_site_id,
            "label": f'{upstream["site_id"]} -> {origin_site_id}',
            "kind": "mainstem upstream",
            "distance_km": upstream.get("distance_km"),
        })
    for downstream in by_position["Downstream"]:
        pairs.append({
            "key": _pair_key(origin_site_id, downstream["site_id"]),
            "upstream_id": origin_site_id,
            "downstream_id": str(downstream["site_id"]),
            "label": f'{origin_site_id} -> {downstream["site_id"]}',
            "kind": "mainstem downstream",
            "distance_km": downstream.get("distance_km"),
        })
    for tributary in by_position["Tributary"]:
        pairs.append({
            "key": _pair_key(tributary["site_id"], origin_site_id),
            "upstream_id": str(tributary["site_id"]),
            "downstream_id": origin_site_id,
            "label": f'{tributary["site_id"]} -> {origin_site_id}',
            "kind": "tributary context",
            "distance_km": tributary.get("distance_km"),
        })

    seen = set()
    unique_pairs = []
    for pair in pairs:
        if pair["key"] in seen:
            continue
        seen.add(pair["key"])
        unique_pairs.append(pair)
    return unique_pairs[:max_pairs]
```

- [ ] **Step 4: Verify tests pass**

Run:

```bash
python -m pytest tests/test_app_reach_analysis.py::test_build_recommended_reach_pairs_prefers_mainstem_pairs tests/test_app_reach_analysis.py::test_pair_key_is_stable_and_readable -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add hydrology/app/page_modules/reach_analysis.py tests/test_app_reach_analysis.py
git commit -m "feat: recommend processable reach pairs"
```

---

### Task 2: Make Reach Selection State Stable

**Files:**
- Modify: `hydrology/app/page_modules/reach_analysis.py`
- Test: `tests/test_app_reach_analysis.py`

- [ ] **Step 1: Write failing tests for selected-pair state**

Add import:

```python
from hydrology.app.page_modules.reach_analysis import _resolve_selected_pair_key
```

Add tests:

```python
def test_resolve_selected_pair_key_keeps_valid_existing_selection():
    pairs = [
        {"key": "up__anchor", "upstream_id": "up", "downstream_id": "anchor"},
        {"key": "anchor__down", "upstream_id": "anchor", "downstream_id": "down"},
    ]
    session_state = {"reach_selected_pair_key": "anchor__down"}

    assert _resolve_selected_pair_key(pairs, session_state) == "anchor__down"


def test_resolve_selected_pair_key_falls_back_to_first_pair():
    pairs = [{"key": "up__anchor", "upstream_id": "up", "downstream_id": "anchor"}]
    session_state = {"reach_selected_pair_key": "missing__pair"}

    assert _resolve_selected_pair_key(pairs, session_state) == "up__anchor"


def test_resolve_selected_pair_key_handles_no_pairs():
    assert _resolve_selected_pair_key([], {}) is None
```

- [ ] **Step 2: Run failing tests**

Run:

```bash
python -m pytest tests/test_app_reach_analysis.py::test_resolve_selected_pair_key_keeps_valid_existing_selection tests/test_app_reach_analysis.py::test_resolve_selected_pair_key_falls_back_to_first_pair tests/test_app_reach_analysis.py::test_resolve_selected_pair_key_handles_no_pairs -q
```

Expected: fails because `_resolve_selected_pair_key` is not defined.

- [ ] **Step 3: Implement helper**

Add near `_pair_key`:

```python
def _resolve_selected_pair_key(reach_pairs, session_state):
    """Keep a selected reach pair if it remains valid; otherwise choose the first available pair."""
    if not reach_pairs:
        return None
    valid_keys = {pair["key"] for pair in reach_pairs}
    current = session_state.get("reach_selected_pair_key")
    if current in valid_keys:
        return current
    return reach_pairs[0]["key"]
```

- [ ] **Step 4: Verify tests pass**

Run:

```bash
python -m pytest tests/test_app_reach_analysis.py::test_resolve_selected_pair_key_keeps_valid_existing_selection tests/test_app_reach_analysis.py::test_resolve_selected_pair_key_falls_back_to_first_pair tests/test_app_reach_analysis.py::test_resolve_selected_pair_key_handles_no_pairs -q
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add hydrology/app/page_modules/reach_analysis.py tests/test_app_reach_analysis.py
git commit -m "fix: keep selected reach pair stable"
```

---

### Task 3: Extract Map Builder And Fit Bounds Explicitly

**Files:**
- Modify: `hydrology/app/page_modules/reach_analysis.py`
- Test: `tests/test_app_reach_analysis.py`

- [ ] **Step 1: Write failing tests for map options and bounds**

Add import:

```python
from hydrology.app.page_modules.reach_analysis import _leaflet_fit_bounds_script
```

Add test:

```python
def test_leaflet_fit_bounds_script_targets_selected_bounds():
    script = _leaflet_fit_bounds_script([[47.0, -118.0], [48.0, -117.0]])

    assert "fitBounds" in script
    assert "[[47.0, -118.0], [48.0, -117.0]]" in script
    assert "paddingTopLeft" in script
    assert "paddingBottomRight" in script
```

- [ ] **Step 2: Run failing test**

Run:

```bash
python -m pytest tests/test_app_reach_analysis.py::test_leaflet_fit_bounds_script_targets_selected_bounds -q
```

Expected: fails because `_leaflet_fit_bounds_script` is not defined.

- [ ] **Step 3: Implement explicit Leaflet fit helper**

Add near `_reach_map_component_key`:

```python
def _leaflet_fit_bounds_script(bounds):
    """Return a Folium-compatible script that forces Leaflet to fit selected reach bounds."""
    return (
        "<script>"
        "setTimeout(function(){"
        "for (const key in window) {"
        "const value = window[key];"
        "if (value && value.fitBounds && value.eachLayer) {"
        f"value.fitBounds({bounds}, {{paddingTopLeft:[24,24], paddingBottomRight:[24,24], maxZoom:13}});"
        "}"
        "}"
        "}, 250);"
        "</script>"
    )
```

- [ ] **Step 4: Update map rendering code**

In `show()`, replace the nested `with st.expander("Reach Map", expanded=False):` block with a `render_reach_map` helper call in the main workspace task. For this task only, keep existing code path but add:

```python
from folium import Element
...
m.fit_bounds(reach_bounds, padding=(24, 24), max_zoom=13)
m.get_root().html.add_child(Element(_leaflet_fit_bounds_script(reach_bounds)))
st_folium(
    m,
    width=None,
    height=460,
    returned_objects=[],
    key=_reach_map_component_key(upstream_id, downstream_id, reach_bounds),
)
```

Also create the map with:

```python
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=11,
    tiles=None,
    max_bounds=True,
    control_scale=True,
)
```

And set the tile layer with:

```python
folium.TileLayer(
    "CartoDB dark_matter",
    name="Base map",
    no_wrap=True,
    detect_retina=True,
).add_to(m)
```

- [ ] **Step 5: Verify tests pass**

Run:

```bash
python -m pytest tests/test_app_reach_analysis.py::test_leaflet_fit_bounds_script_targets_selected_bounds tests/test_app_reach_analysis.py::test_map_bounds_focuses_on_selected_gage_markers_not_flowline_extent tests/test_app_reach_analysis.py::test_reach_map_component_key_changes_with_selected_pair_and_bounds -q
```

Expected: `3 passed`.

- [ ] **Step 6: Commit**

```bash
git add hydrology/app/page_modules/reach_analysis.py tests/test_app_reach_analysis.py
git commit -m "fix: force reach map to selected gage bounds"
```

---

### Task 4: Restructure Reach Analysis Into Map-Centered Workspace

**Files:**
- Modify: `hydrology/app/page_modules/reach_analysis.py`
- Test: `tests/test_app_reach_analysis.py`

- [ ] **Step 1: Write failing tests for visible labels**

Add tests:

```python
def test_reach_page_source_uses_gage_not_gauge():
    source = __import__("inspect").getsource(__import__("hydrology.app.page_modules.reach_analysis", fromlist=["show"]))

    assert "gauge" not in source.lower()
    assert "gage" in source.lower()


def test_reach_page_source_does_not_bury_map_in_expander():
    source = __import__("inspect").getsource(__import__("hydrology.app.page_modules.reach_analysis", fromlist=["show"]))

    assert 'st.expander("Reach Map"' not in source
```

- [ ] **Step 2: Run failing tests**

Run:

```bash
python -m pytest tests/test_app_reach_analysis.py::test_reach_page_source_uses_gage_not_gauge tests/test_app_reach_analysis.py::test_reach_page_source_does_not_bury_map_in_expander -q
```

Expected: second test fails until the map is moved out of the expander.

- [ ] **Step 3: Replace top expanders with compact search row**

In `show()`, replace:

```python
with st.expander("Select reach", expanded=True):
    ...
```

with direct columns:

```python
search_col1, search_col2, search_col3, search_col4, search_col5 = st.columns([1.1, 2.0, 3.0, 0.9, 1.2])
with search_col1:
    anchor_state = st.selectbox("State", states, key="reach_anchor_state")
with search_col2:
    anchor_search = st.text_input(
        "Find gage",
        placeholder="River, station, or USGS ID...",
        key="reach_anchor_search",
    )
...
with search_col5:
    include_tributaries = st.toggle("Tributaries", value=True, key="reach_include_tributaries")
    find_gages = st.button("Find Reaches", width="stretch", key="reach_find_related")
```

Keep the existing `anchor_filtered`, `anchor_options`, `anchor_id`, `anchor_info`, and NLDI discovery logic.

- [ ] **Step 4: Replace candidate table default with pair selector**

After `candidate_records` is built, add:

```python
reach_pairs = _build_recommended_reach_pairs(anchor_id, candidate_records)
selected_pair_key = _resolve_selected_pair_key(reach_pairs, st.session_state)
if selected_pair_key:
    st.session_state["reach_selected_pair_key"] = selected_pair_key
selected_pair = next((pair for pair in reach_pairs if pair["key"] == selected_pair_key), None)
```

Use a `st.radio` or compact `st.selectbox` in the left workspace column:

```python
pair_labels = {pair["label"]: pair["key"] for pair in reach_pairs}
selected_pair_label = next(label for label, key in pair_labels.items() if key == selected_pair_key)
chosen_label = st.radio(
    "Candidate reaches",
    list(pair_labels.keys()),
    index=list(pair_labels.keys()).index(selected_pair_label),
    key="reach_pair_radio",
)
st.session_state["reach_selected_pair_key"] = pair_labels[chosen_label]
```

Then set:

```python
upstream_id = selected_pair["upstream_id"]
downstream_id = selected_pair["downstream_id"]
```

- [ ] **Step 5: Build three-column workspace**

Replace the separate `Candidate gages`, `Selected reach`, and nested `Reach Map` sections with:

```python
candidate_col, map_col, summary_col = st.columns([1.05, 2.15, 1.0])
with candidate_col:
    st.subheader("Candidate Reaches")
    ...
with map_col:
    st.subheader("Reach Map")
    ...
with summary_col:
    st.subheader("Selected Reach")
    ...
```

Render network length metric and readiness in the summary column. Keep `manual_reach_km = 0.0` unless advanced override changes it below.

- [ ] **Step 6: Move lower-priority controls below workspace**

Create collapsed sections below the workspace:

```python
with st.expander("Candidate gage details", expanded=False):
    ...
with st.expander("Analysis settings", expanded=False):
    ...
with st.expander("Advanced reach length override", expanded=False):
    ...
```

Keep plot defaults as current: `reach_comparison`, `reach_index`, `seasonal_gain_loss`.

- [ ] **Step 7: Verify tests pass**

Run:

```bash
python -m pytest tests/test_app_reach_analysis.py -q
```

Expected: all reach app tests pass.

- [ ] **Step 8: Commit**

```bash
git add hydrology/app/page_modules/reach_analysis.py tests/test_app_reach_analysis.py
git commit -m "feat: center reach analysis on map workflow"
```

---

### Task 5: Verify App Behavior And Deploy

**Files:**
- Modify only if verification finds a bug.

- [ ] **Step 1: Run focused tests**

Run:

```bash
python -m pytest tests/test_app_reach_analysis.py tests/test_reach_gain_loss.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run related dashboard tests**

Run:

```bash
python -m pytest tests/test_app_indicators.py tests/test_frequency.py tests/test_nwm.py -q
```

Expected: all tests pass, confirming recent dashboard fixes were not broken.

- [ ] **Step 3: Run full suite if focused tests pass**

Run:

```bash
python -m pytest -q
```

Expected: full test suite passes.

- [ ] **Step 4: Push branch**

Run:

```bash
git push
```

Expected: branch pushes to origin.

- [ ] **Step 5: Deploy to Streamlit host**

Run:

```bash
ssh cam@192.168.1.126 'cd /home/cam/source/repos/Hydrology && git pull --ff-only'
ssh cam@192.168.1.126 'pkill -f "streamlit run hydrology/app/app.py" || true'
ssh cam@192.168.1.126 'cd /home/cam/source/repos/Hydrology && source venv/bin/activate && setsid streamlit run hydrology/app/app.py --server.headless true --server.address 0.0.0.0 --server.port 8501 > /tmp/hydroplot-streamlit.log 2>&1 < /dev/null &'
ssh cam@192.168.1.126 'curl -s -o /tmp/hydroplot_home.html -w "%{http_code}\n" http://127.0.0.1:8501/'
```

Expected: HTTP `200`.

- [ ] **Step 6: Manual Streamlit verification**

Open `http://192.168.1.126:8501/` and verify:

- Search for a Spokane River gage.
- Click Find Reaches with tributaries enabled.
- Candidate Reaches shows processable pairs, not a giant unfiltered NLDI table.
- Selecting a different pair changes only the selected pair; it does not reset both endpoints to the anchor gage.
- Reach Map is visible without opening an expander.
- Reach Map is zoomed into the selected pair, with no repeated-world view.
- Selected reach line is visually stronger than tributary/context flowlines.
- Run Analysis produces the automated reach summary and plots below the workspace.

- [ ] **Step 7: Commit fixes only if verification finds issues**

Use targeted commits such as:

```bash
git add hydrology/app/page_modules/reach_analysis.py tests/test_app_reach_analysis.py
git commit -m "fix: refine reach map workspace behavior"
```
