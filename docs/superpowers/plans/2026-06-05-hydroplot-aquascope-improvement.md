# HydroPlot AquaScope-Informed Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve HydroPlot by adopting the strongest reproducibility, validation, baseflow, groundwater, reach, and trend ideas from AquaScope-style demos while avoiding notebook-only, duplicated, or non-PNW implementation patterns.

**Architecture:** Keep HydroPlot as the source of truth. Implement reusable, tested analysis modules first, add PNW validation cases second, then wire proven outputs into the existing Streamlit dashboard. Do not add AquaScope as a runtime dashboard dependency; port useful algorithms and case-study discipline into HydroPlot-native modules.

**Tech Stack:** Python 3.11, pandas, numpy, scipy, pymannkendall, pytest, Streamlit, Plotly/Matplotlib, existing USGS/NLDI/HyRiver helpers, GitHub feature branch + pull request workflow.

---

## Operating Model

Work on a feature branch:

```powershell
git status --short
git switch -c feature/hydroplot-groundwater-reach-validation
```

Use this loop for every task:

```powershell
python -m pytest tests/<focused_test_file>.py -q
git diff
git add <changed files>
git commit -m "<type>: <specific change>"
```

Push and open a draft PR after the first two passing task commits:

```powershell
git push -u origin feature/hydroplot-groundwater-reach-validation
```

Best way to see changes as we go:

- **Code review:** `git diff` after every task, with one focused commit per behavior.
- **Verification:** focused pytest command per task, then broader regression suite after dashboard wiring.
- **Scientific outputs:** generated CSV/PNG/Markdown under `docs/cases/<case>/outputs/`.
- **Dashboard:** local Streamlit run after UI tasks, with screenshots or browser checks for changed pages.
- **GitHub:** draft PR containing task checklist, verification log, and remaining risk notes.

## Commit Hygiene Rules

Every commit should look like a deliberate human-authored change:

- One coherent behavior per commit. Do not mix analysis logic, dashboard UI, docs, and unrelated cleanup in the same commit.
- Tests travel with the behavior they verify. A new algorithm commit includes its focused tests.
- Avoid formatting churn. Do not reformat untouched files or reorder imports globally.
- Avoid broad refactors unless the task explicitly requires moving code to preserve clarity.
- Commit messages should explain the user-visible or developer-visible capability, not the mechanical edit.
- Generated case outputs get their own commit only when they are intentional review artifacts.
- If a task touches more than four files, pause and check whether it should split into two commits.
- Before committing, inspect `git diff --stat` and `git diff` for accidental edits.
- PR description should group commits by capability: baseflow, signatures, changepoints, reach groundwater, validation cases, dashboard wiring.

Good commit examples:

```text
feat: add reusable baseflow method comparison
feat: add reach groundwater gain loss summaries
feat: show baseflow method comparison in dashboard
docs: add HydroPlot validation case template
```

Bad commit examples:

```text
update stuff
ai changes
fix tests and dashboard and docs
misc improvements
```

## Lean Verification Rules

Do not add defensive checks everywhere. The code should stay readable and fast.

- Keep runtime validation at module boundaries: public analysis functions, case runner inputs, and user-triggered dashboard actions.
- Prefer tests over production checks when a condition is a developer contract rather than a likely user/data failure.
- Avoid repeated `isinstance`, schema, and range checks inside loops or vectorized calculations.
- Do not add broad `try/except Exception` around analysis math. Let programming errors fail loudly.
- Use explicit, narrow handling only for expected external failures: network fetch failure, missing site data, empty paired reach data, or insufficient record length.
- Expensive verification belongs in pytest, case-study runners, or on-demand dashboard buttons, not page-load paths.
- Cache or reuse dashboard results through existing Streamlit patterns; do not recompute baseflow, signatures, or changepoints multiple times per rerun.
- Keep PASS/FLAG validation out of core algorithm internals. Algorithms return results; validators inspect results in tests/cases/UI summaries.

---

## Current Assessment

### AquaScope Strengths To Adopt

- Reproducible case-study structure with scenario, method, data source, outputs, and validation.
- Flood-frequency validation against published reference values.
- Dual-method baseflow comparison with Lyne-Hollick and Eckhardt filters.
- Hydrologic signatures for quick basin behavior summaries.
- Mann-Kendall trend plus Sen's slope and Pettitt changepoint.
- Output artifacts that make results inspectable outside the app.

### AquaScope Weaknesses To Improve On

- Notebook-level fetch/cache code is duplicated across cases.
- Some `dataretrieval` calls shown in demos emit deprecation warnings.
- Broad `except Exception` fallbacks hide failure modes.
- Validation constants live in notebooks instead of reusable configs/tests.
- National examples are useful demos but not tailored to HydroPlot's PNW reach and groundwater purpose.
- Not a dashboard-first implementation.

### HydroPlot Strengths To Preserve

- Existing Streamlit dashboard and page module organization.
- Existing `hydrology.analysis.frequency`, `hydrology.analysis.trends`, and `hydrology.analysis.indicators`.
- Existing USGS retry/chunking fetchers.
- Existing PNW/site inventory orientation.
- Existing reach-analysis and NLDI scaffolding.
- Existing pytest suite.

---

## File Structure

Create:

- `hydrology/analysis/baseflow.py`  
  Reusable baseflow filters, result dataclasses, BFI comparison, and method quality flags.

- `hydrology/analysis/signatures.py`  
  Hydrologic signatures: flow duration quantiles, flashiness, high/low-flow frequency, seasonality, BFI summary, recession summary hooks.

- `hydrology/analysis/changepoints.py`  
  Pettitt changepoint and optional PELT-compatible interface without making heavy dependencies mandatory.

- `hydrology/analysis/reach_groundwater.py`  
  Reach-scale gain/loss, normalized contribution, low-flow contribution, classification, and confidence flags.

- `hydrology/analysis/reach_topology.py`  
  Small pure helpers for validating station-pair direction and pairing metadata from NLDI/navigation results. This keeps topology checks separate from gain/loss math.

- `hydrology/analysis/temperature_context.py`  
  Lightweight riparian/thermal sensitivity context. This is not QUAL2K; it prepares defensible reach descriptors inspired by TTools/Shade/QUAL2K workflows. Keep this as a simple screening helper, not a large rules engine.

- `hydrology/analysis/validation.py`  
  Shared validation result objects and PASS/FLAG helpers for case studies.

- `hydrology/app/page_modules/groundwater.py`  
  Dashboard panel for baseflow comparison and reach groundwater summaries, only after analysis modules are verified.

- `docs/cases/_template/README.md`  
  HydroPlot-native version of the AquaScope case template.

- `docs/cases/spokane_groundwater_reach/README.md`  
  PNW validation case for groundwater/reach behavior.

- `docs/cases/pnw_baseflow_signatures/README.md`  
  PNW validation case for baseflow and signatures.

- `docs/cases/red_river_trend_benchmark/README.md`  
  Non-PNW benchmark case used only to verify changepoint/trend behavior against a well-documented published example.

- `scripts/run_case_study.py`  
  CLI runner for case configs that produces repeatable output artifacts.

Modify:

- `hydrology/analysis/__init__.py`  
  Export new public analysis functions.

- `hydrology/analysis/indicators.py`  
  Deprecate or delegate current BFI implementation to `baseflow.py` while keeping backward compatibility.

- `hydrology/analysis/trends.py`  
  Add Sen's slope fields if missing and delegate changepoints to `changepoints.py`.

- `hydrology/analysis/frequency.py`  
  Add Q-Q/P-P diagnostic table helpers and deterministic bootstrap seed support.

- `hydrology/app/page_modules/indicators.py`  
  Replace single-method BFI display with method comparison where available.

- `hydrology/app/page_modules/reach_analysis.py`  
  Add reach gain/loss and groundwater contribution summaries.

- `hydrology/app/page_modules/single_analysis.py`  
  Add flood-frequency diagnostics and changepoint output in focused sections.

- `requirements.txt`  
  Add only lightweight dependencies if needed. Avoid AquaScope as a dependency.

Test:

- `tests/test_baseflow.py`
- `tests/test_signatures.py`
- `tests/test_changepoints.py`
- `tests/test_reach_topology.py`
- `tests/test_reach_groundwater.py`
- `tests/test_temperature_context.py`
- `tests/test_validation_cases.py`
- Existing app tests that cover page imports and plot config.

---

## Task 1: Branch And Baseline Verification

**Files:**
- No code changes.

- [ ] **Step 1: Check workspace state**

Run:

```powershell
git status --short
```

Expected: review any existing uncommitted files. Do not overwrite unrelated user changes.

- [ ] **Step 2: Create feature branch**

Run:

```powershell
git switch -c feature/hydroplot-groundwater-reach-validation
```

Expected: branch created from current checkout.

- [ ] **Step 3: Run baseline tests**

Run:

```powershell
python -m pytest tests/test_trends.py tests/test_usgs.py tests/test_app_plot_config.py -q
```

Expected: PASS or documented existing failures before feature work starts.

- [ ] **Step 4: Commit plan if not already committed**

Run:

```powershell
git add docs/superpowers/plans/2026-06-05-hydroplot-aquascope-improvement.md
git commit -m "docs: plan HydroPlot groundwater and validation upgrades"
```

Expected: one docs commit.

---

## Task 2: Baseflow Module With Lyne-Hollick And Eckhardt

**Files:**
- Create: `hydrology/analysis/baseflow.py`
- Modify: `hydrology/analysis/__init__.py`
- Modify: `hydrology/analysis/indicators.py`
- Test: `tests/test_baseflow.py`

- [ ] **Step 1: Write failing baseflow tests**

Create `tests/test_baseflow.py`:

```python
import numpy as np
import pandas as pd

from hydrology.analysis.baseflow import (
    BaseflowResult,
    compare_baseflow_methods,
    eckhardt_filter,
    lyne_hollick_filter,
)


def _daily_flow(values):
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=len(values), freq="D"))


def test_lyne_hollick_returns_components_bounded_by_total_flow():
    flow = _daily_flow([100, 120, 180, 140, 110, 90, 95, 105, 130, 115])

    result = lyne_hollick_filter(flow, alpha=0.925, passes=3)

    assert isinstance(result, BaseflowResult)
    assert list(result.components.columns) == ["total_flow", "baseflow", "quickflow"]
    assert (result.components["baseflow"] >= 0).all()
    assert (result.components["baseflow"] <= result.components["total_flow"]).all()
    assert 0 <= result.bfi <= 1
    assert result.method == "lyne_hollick"


def test_eckhardt_filter_responds_to_bfi_max_and_stays_bounded():
    flow = _daily_flow([100, 110, 130, 125, 115, 105, 95, 100, 108, 102])

    conservative = eckhardt_filter(flow, alpha=0.98, bfi_max=0.50)
    permissive = eckhardt_filter(flow, alpha=0.98, bfi_max=0.80)

    assert (conservative.components["baseflow"] <= conservative.components["total_flow"]).all()
    assert (permissive.components["baseflow"] <= permissive.components["total_flow"]).all()
    assert permissive.bfi >= conservative.bfi
    assert conservative.method == "eckhardt"


def test_compare_baseflow_methods_flags_large_disagreement():
    flow = _daily_flow(np.r_[np.repeat(100.0, 20), 500.0, np.repeat(90.0, 20)])

    comparison = compare_baseflow_methods(flow)

    assert {"lyne_hollick_bfi", "eckhardt_bfi", "bfi_difference", "agreement"}.issubset(comparison)
    assert comparison["agreement"] in {"strong", "moderate", "weak"}
    assert comparison["bfi_difference"] >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest tests/test_baseflow.py -q
```

Expected: FAIL because `hydrology.analysis.baseflow` does not exist.

- [ ] **Step 3: Implement `baseflow.py`**

Create `hydrology/analysis/baseflow.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from ..core.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class BaseflowResult:
    method: str
    components: pd.DataFrame
    bfi: float
    parameters: Dict[str, float]


def _clean_daily_flow(daily_q: pd.Series) -> pd.Series:
    series = pd.Series(daily_q).dropna().astype(float)
    series = series[np.isfinite(series)]
    series = series[series >= 0]
    return series.sort_index()


def _result(method: str, total: pd.Series, baseflow: np.ndarray, parameters: Dict[str, float]) -> BaseflowResult:
    base = np.clip(baseflow.astype(float), 0, total.values.astype(float))
    quick = total.values.astype(float) - base
    components = pd.DataFrame(
        {
            "total_flow": total.values.astype(float),
            "baseflow": base,
            "quickflow": quick,
        },
        index=total.index,
    )
    total_sum = float(components["total_flow"].sum())
    bfi = float(components["baseflow"].sum() / total_sum) if total_sum > 0 else float("nan")
    return BaseflowResult(method=method, components=components, bfi=bfi, parameters=parameters)


def lyne_hollick_filter(daily_q: pd.Series, alpha: float = 0.925, passes: int = 3) -> BaseflowResult:
    flow = _clean_daily_flow(daily_q)
    if flow.empty:
        return _result("lyne_hollick", flow, np.array([], dtype=float), {"alpha": alpha, "passes": passes})
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if passes < 1:
        raise ValueError("passes must be >= 1")

    q = flow.values.astype(float)
    quick = q.copy()
    direction = 1
    for _ in range(passes):
        work = quick if direction == 1 else quick[::-1]
        filtered = np.zeros_like(work)
        for i in range(1, len(work)):
            filtered[i] = alpha * filtered[i - 1] + ((1 + alpha) / 2.0) * (work[i] - work[i - 1])
            filtered[i] = min(max(filtered[i], 0.0), work[i])
        quick = filtered if direction == 1 else filtered[::-1]
        direction *= -1

    baseflow = q - quick
    return _result("lyne_hollick", flow, baseflow, {"alpha": alpha, "passes": float(passes)})


def eckhardt_filter(daily_q: pd.Series, alpha: float = 0.98, bfi_max: float = 0.80) -> BaseflowResult:
    flow = _clean_daily_flow(daily_q)
    if flow.empty:
        return _result("eckhardt", flow, np.array([], dtype=float), {"alpha": alpha, "bfi_max": bfi_max})
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if not 0 < bfi_max < 1:
        raise ValueError("bfi_max must be between 0 and 1")

    q = flow.values.astype(float)
    base = np.zeros_like(q)
    base[0] = min(q[0], q[0] * bfi_max)
    denominator = 1 - alpha * bfi_max
    for i in range(1, len(q)):
        numerator = (1 - bfi_max) * alpha * base[i - 1] + (1 - alpha) * bfi_max * q[i]
        base[i] = min(q[i], max(0.0, numerator / denominator))

    return _result("eckhardt", flow, base, {"alpha": alpha, "bfi_max": bfi_max})


def compare_baseflow_methods(daily_q: pd.Series) -> Dict[str, float | str]:
    lh = lyne_hollick_filter(daily_q)
    ek = eckhardt_filter(daily_q)
    diff = abs(lh.bfi - ek.bfi)
    if diff <= 0.05:
        agreement = "strong"
    elif diff <= 0.10:
        agreement = "moderate"
    else:
        agreement = "weak"
    return {
        "lyne_hollick_bfi": lh.bfi,
        "eckhardt_bfi": ek.bfi,
        "bfi_difference": diff,
        "agreement": agreement,
    }
```

- [ ] **Step 4: Export and preserve indicator compatibility**

Modify `hydrology/analysis/__init__.py` to export:

```python
from .baseflow import BaseflowResult, compare_baseflow_methods, eckhardt_filter, lyne_hollick_filter
```

Modify `calculate_baseflow_index_timeseries()` in `hydrology/analysis/indicators.py` to delegate internally:

```python
from .baseflow import lyne_hollick_filter

result = lyne_hollick_filter(daily_q, alpha=alpha, passes=1)
df = result.components.copy()
```

Keep the rolling BFI logic unchanged after `df` is created.

- [ ] **Step 5: Run focused tests**

Run:

```powershell
python -m pytest tests/test_baseflow.py tests/test_app_indicators.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```powershell
git add hydrology/analysis/baseflow.py hydrology/analysis/__init__.py hydrology/analysis/indicators.py tests/test_baseflow.py
git commit -m "feat: add reusable baseflow method comparison"
```

---

## Task 3: Hydrologic Signatures

**Files:**
- Create: `hydrology/analysis/signatures.py`
- Modify: `hydrology/analysis/__init__.py`
- Test: `tests/test_signatures.py`

- [ ] **Step 1: Write failing signature tests**

Create `tests/test_signatures.py`:

```python
import numpy as np
import pandas as pd

from hydrology.analysis.signatures import compute_hydrologic_signatures


def test_compute_hydrologic_signatures_returns_core_metrics():
    index = pd.date_range("2020-01-01", periods=366, freq="D")
    flow = pd.Series(100 + 30 * np.sin(np.linspace(0, 2 * np.pi, 366)), index=index)

    result = compute_hydrologic_signatures(flow)

    assert result["n_days"] == 366
    assert result["mean_flow"] > 0
    assert result["q05"] > result["q50"] > result["q95"]
    assert 0 <= result["baseflow_index_lh"] <= 1
    assert result["richards_baker_flashiness"] >= 0
    assert 1 <= result["peak_month"] <= 12


def test_compute_hydrologic_signatures_handles_empty_series():
    result = compute_hydrologic_signatures(pd.Series(dtype=float))

    assert result == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest tests/test_signatures.py -q
```

Expected: FAIL because module does not exist.

- [ ] **Step 3: Implement signatures**

Create `hydrology/analysis/signatures.py`:

```python
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .baseflow import lyne_hollick_filter


def compute_hydrologic_signatures(daily_q: pd.Series) -> Dict[str, float]:
    flow = pd.Series(daily_q).dropna().astype(float)
    flow = flow[np.isfinite(flow)]
    flow = flow[flow >= 0].sort_index()
    if flow.empty:
        return {}

    diffs = flow.diff().abs().dropna()
    flashiness = float(diffs.sum() / flow.sum()) if flow.sum() > 0 else float("nan")
    monthly = flow.groupby(flow.index.month).mean() if isinstance(flow.index, pd.DatetimeIndex) else pd.Series(dtype=float)
    peak_month = int(monthly.idxmax()) if not monthly.empty else 0
    low_month = int(monthly.idxmin()) if not monthly.empty else 0
    lh = lyne_hollick_filter(flow)

    return {
        "n_days": float(len(flow)),
        "mean_flow": float(flow.mean()),
        "median_flow": float(flow.median()),
        "min_flow": float(flow.min()),
        "max_flow": float(flow.max()),
        "q05": float(flow.quantile(0.95)),
        "q10": float(flow.quantile(0.90)),
        "q50": float(flow.quantile(0.50)),
        "q90": float(flow.quantile(0.10)),
        "q95": float(flow.quantile(0.05)),
        "coefficient_of_variation": float(flow.std(ddof=1) / flow.mean()) if flow.mean() > 0 else float("nan"),
        "richards_baker_flashiness": flashiness,
        "baseflow_index_lh": float(lh.bfi),
        "high_flow_frequency": float((flow > flow.quantile(0.90)).sum() / len(flow)),
        "low_flow_frequency": float((flow < flow.quantile(0.10)).sum() / len(flow)),
        "peak_month": float(peak_month),
        "low_month": float(low_month),
    }
```

- [ ] **Step 4: Export**

Modify `hydrology/analysis/__init__.py`:

```python
from .signatures import compute_hydrologic_signatures
```

- [ ] **Step 5: Run focused tests**

Run:

```powershell
python -m pytest tests/test_signatures.py tests/test_baseflow.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```powershell
git add hydrology/analysis/signatures.py hydrology/analysis/__init__.py tests/test_signatures.py
git commit -m "feat: add hydrologic signature metrics"
```

---

## Task 4: Pettitt Changepoint And Sen Slope

**Files:**
- Create: `hydrology/analysis/changepoints.py`
- Modify: `hydrology/analysis/trends.py`
- Modify: `hydrology/analysis/__init__.py`
- Test: `tests/test_changepoints.py`
- Test: `tests/test_trends.py`

- [ ] **Step 1: Write failing changepoint tests**

Create `tests/test_changepoints.py`:

```python
import pandas as pd

from hydrology.analysis.changepoints import pettitt_test
from hydrology.analysis.trends import mann_kendall_test


def test_pettitt_detects_known_step_change():
    index = pd.Index(range(1900, 1940), name="year")
    values = pd.Series([10] * 20 + [30] * 20, index=index)

    result = pettitt_test(values)

    assert result["change_index"] in {19, 20}
    assert result["change_point"] in {1919, 1920}
    assert result["p_value"] < 0.05
    assert result["mean_before"] < result["mean_after"]


def test_mann_kendall_exposes_sens_slope_when_available():
    values = pd.Series([1, 2, 3, 4, 5, 6])

    result = mann_kendall_test(values)

    if result is not None:
        assert "sens_slope" in result
        assert result["sens_slope"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest tests/test_changepoints.py -q
```

Expected: FAIL because `changepoints.py` does not exist or `sens_slope` is missing.

- [ ] **Step 3: Implement Pettitt**

Create `hydrology/analysis/changepoints.py`:

```python
from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd


def pettitt_test(series: pd.Series) -> Dict[str, Any]:
    clean = pd.Series(series).dropna().sort_index()
    n = len(clean)
    if n < 6:
        return {}

    values = clean.values.astype(float)
    ranks = pd.Series(values).rank().values
    u = np.array([2 * np.sum(ranks[: k + 1]) - (k + 1) * (n + 1) for k in range(n)])
    k = int(np.argmax(np.abs(u)))
    statistic = float(abs(u[k]))
    p_value = float(2 * np.exp((-6 * statistic**2) / (n**3 + n**2)))
    before = values[: k + 1]
    after = values[k + 1 :]

    return {
        "change_index": k,
        "change_point": clean.index[k].year if hasattr(clean.index[k], "year") else clean.index[k],
        "statistic": statistic,
        "p_value": min(max(p_value, 0.0), 1.0),
        "mean_before": float(np.mean(before)) if len(before) else float("nan"),
        "mean_after": float(np.mean(after)) if len(after) else float("nan"),
        "n_points": n,
    }
```

- [ ] **Step 4: Add Sen slope to trend output**

Modify `mann_kendall_test()` in `hydrology/analysis/trends.py` result dict:

```python
"sens_slope": getattr(mk_result, "slope", np.nan),
"sens_intercept": getattr(mk_result, "intercept", np.nan),
```

- [ ] **Step 5: Export**

Modify `hydrology/analysis/__init__.py`:

```python
from .changepoints import pettitt_test
```

- [ ] **Step 6: Run focused tests**

Run:

```powershell
python -m pytest tests/test_changepoints.py tests/test_trends.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```powershell
git add hydrology/analysis/changepoints.py hydrology/analysis/trends.py hydrology/analysis/__init__.py tests/test_changepoints.py tests/test_trends.py
git commit -m "feat: add changepoint and Sen slope trend outputs"
```

---

## Task 5: Reach Topology And Station-Pair Validation

**Files:**
- Create: `hydrology/analysis/reach_topology.py`
- Modify: `hydrology/analysis/__init__.py`
- Test: `tests/test_reach_topology.py`

**Purpose:** prove HydroPlot can reason about upstream/downstream station pairs before applying gain/loss math. This task does not call NLDI directly; it validates metadata returned by existing NLDI helpers and dashboard selections. Network-backed NLDI discovery remains in `hydrology/data/nldi.py`.

**Research basis:** agency reach workflows do not infer gaining/losing conditions from two arbitrary gauges. They require a defensible reach definition, station order, flow-period pairing, and caveats for tributaries/diversions/withdrawals. This task creates the lightweight software boundary for that discipline.

- [ ] **Step 1: Write failing topology tests**

Create `tests/test_reach_topology.py`:

```python
from hydrology.analysis.reach_topology import (
    ReachPair,
    classify_pair_direction,
    validate_reach_pair,
)


def test_classify_pair_direction_accepts_downstream_metadata():
    sites = [
        {"site_id": "up", "direction": "upstream", "distance_km": 8.0},
        {"site_id": "down", "direction": "downstream", "distance_km": 12.0},
    ]

    assert classify_pair_direction("up", "down", sites, origin_site_id="origin") == "ordered"


def test_validate_reach_pair_flags_same_station():
    pair = validate_reach_pair("12422500", "12422500", related_sites=[])

    assert pair.status == "invalid"
    assert "same station" in pair.notes[0]


def test_validate_reach_pair_flags_unverified_direction():
    pair = validate_reach_pair(
        "12422500",
        "12424000",
        related_sites=[{"site_id": "12424000", "direction": "upstream"}],
    )

    assert isinstance(pair, ReachPair)
    assert pair.status == "unverified"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest tests/test_reach_topology.py -q
```

Expected: FAIL because `reach_topology.py` does not exist.

- [ ] **Step 3: Implement topology helpers**

Create `hydrology/analysis/reach_topology.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class ReachPair:
    upstream_site_id: str
    downstream_site_id: str
    status: str
    notes: List[str]


def classify_pair_direction(
    upstream_site_id: str,
    downstream_site_id: str,
    related_sites: Iterable[Dict],
    origin_site_id: str | None = None,
) -> str:
    by_id = {str(site.get("site_id")): site for site in related_sites}
    upstream_meta = by_id.get(str(upstream_site_id))
    downstream_meta = by_id.get(str(downstream_site_id))

    if downstream_meta and downstream_meta.get("direction") == "downstream":
        return "ordered"
    if upstream_meta and upstream_meta.get("direction") == "upstream":
        return "ordered"
    if downstream_meta and downstream_meta.get("direction") == "upstream":
        return "reversed_or_tributary"
    if upstream_meta and upstream_meta.get("direction") == "downstream":
        return "reversed_or_tributary"
    return "unknown"


def validate_reach_pair(
    upstream_site_id: str,
    downstream_site_id: str,
    related_sites: Iterable[Dict],
) -> ReachPair:
    if upstream_site_id == downstream_site_id:
        return ReachPair(upstream_site_id, downstream_site_id, "invalid", ["same station selected twice"])

    direction = classify_pair_direction(upstream_site_id, downstream_site_id, related_sites)
    if direction == "ordered":
        return ReachPair(upstream_site_id, downstream_site_id, "verified", ["NLDI/navigation metadata supports station order"])
    if direction == "reversed_or_tributary":
        return ReachPair(upstream_site_id, downstream_site_id, "unverified", ["metadata suggests reversed order or tributary/diversion relationship"])
    return ReachPair(upstream_site_id, downstream_site_id, "unverified", ["station order not verified by navigation metadata"])
```

- [ ] **Step 4: Export**

Modify `hydrology/analysis/__init__.py`:

```python
from .reach_topology import ReachPair, classify_pair_direction, validate_reach_pair
```

- [ ] **Step 5: Run focused tests**

Run:

```powershell
python -m pytest tests/test_reach_topology.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```powershell
git add hydrology/analysis/reach_topology.py hydrology/analysis/__init__.py tests/test_reach_topology.py
git commit -m "feat: add reach station pair validation"
```

---

## Task 6: Reach Groundwater Gain/Loss Analysis

**Files:**
- Create: `hydrology/analysis/reach_groundwater.py`
- Modify: `hydrology/analysis/__init__.py`
- Test: `tests/test_reach_groundwater.py`

- [ ] **Step 1: Write failing reach tests**

Create `tests/test_reach_groundwater.py`:

```python
import pandas as pd

from hydrology.analysis.reach_groundwater import classify_reach_gain_loss, summarize_reach_gain_loss


def test_summarize_reach_gain_loss_classifies_gaining_reach():
    index = pd.date_range("2021-08-01", periods=10, freq="D")
    upstream = pd.Series([100] * 10, index=index)
    downstream = pd.Series([130] * 10, index=index)

    result = summarize_reach_gain_loss(upstream, downstream, reach_km=10, drainage_area_sqmi=100)

    assert result["median_gain_cfs"] == 30
    assert result["median_gain_cfs_per_km"] == 3
    assert result["classification"] == "gaining"
    assert result["confidence"] == "high"


def test_classify_reach_gain_loss_uses_deadband():
    assert classify_reach_gain_loss(0.2, deadband_cfs=1.0) == "neutral"
    assert classify_reach_gain_loss(5.0, deadband_cfs=1.0) == "gaining"
    assert classify_reach_gain_loss(-5.0, deadband_cfs=1.0) == "losing"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest tests/test_reach_groundwater.py -q
```

Expected: FAIL because module does not exist.

- [ ] **Step 3: Implement reach module**

Create `hydrology/analysis/reach_groundwater.py`:

```python
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def classify_reach_gain_loss(median_gain_cfs: float, deadband_cfs: float = 1.0) -> str:
    if not np.isfinite(median_gain_cfs):
        return "insufficient_data"
    if median_gain_cfs > deadband_cfs:
        return "gaining"
    if median_gain_cfs < -deadband_cfs:
        return "losing"
    return "neutral"


def summarize_reach_gain_loss(
    upstream_q: pd.Series,
    downstream_q: pd.Series,
    reach_km: Optional[float] = None,
    drainage_area_sqmi: Optional[float] = None,
    low_flow_quantile: float = 0.25,
    deadband_cfs: float = 1.0,
) -> Dict[str, float | str]:
    upstream = pd.Series(upstream_q).dropna().astype(float)
    downstream = pd.Series(downstream_q).dropna().astype(float)
    paired = pd.concat({"upstream": upstream, "downstream": downstream}, axis=1).dropna()
    if paired.empty:
        return {"classification": "insufficient_data", "confidence": "none", "n_days": 0.0}

    paired["gain_cfs"] = paired["downstream"] - paired["upstream"]
    low_threshold = paired["upstream"].quantile(low_flow_quantile)
    low_flow = paired[paired["upstream"] <= low_threshold]

    median_gain = float(paired["gain_cfs"].median())
    low_flow_gain = float(low_flow["gain_cfs"].median()) if not low_flow.empty else float("nan")
    classification = classify_reach_gain_loss(median_gain, deadband_cfs=deadband_cfs)
    variability = float(paired["gain_cfs"].std(ddof=1)) if len(paired) > 1 else 0.0
    confidence = "high" if len(paired) >= 7 and variability <= max(abs(median_gain), deadband_cfs) else "moderate"
    if len(paired) < 7:
        confidence = "low"

    result: Dict[str, float | str] = {
        "n_days": float(len(paired)),
        "median_gain_cfs": median_gain,
        "mean_gain_cfs": float(paired["gain_cfs"].mean()),
        "low_flow_median_gain_cfs": low_flow_gain,
        "classification": classification,
        "confidence": confidence,
    }
    if reach_km and reach_km > 0:
        result["median_gain_cfs_per_km"] = median_gain / reach_km
    if drainage_area_sqmi and drainage_area_sqmi > 0:
        result["median_gain_cfs_per_sqmi"] = median_gain / drainage_area_sqmi
    return result
```

- [ ] **Step 4: Export**

Modify `hydrology/analysis/__init__.py`:

```python
from .reach_groundwater import classify_reach_gain_loss, summarize_reach_gain_loss
```

- [ ] **Step 5: Run focused tests**

Run:

```powershell
python -m pytest tests/test_reach_groundwater.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```powershell
git add hydrology/analysis/reach_groundwater.py hydrology/analysis/__init__.py tests/test_reach_groundwater.py
git commit -m "feat: add reach groundwater gain loss summaries"
```

---

## Task 7: Lightweight Riparian And Temperature Context

**Files:**
- Create: `hydrology/analysis/temperature_context.py`
- Modify: `hydrology/analysis/__init__.py`
- Test: `tests/test_temperature_context.py`

**Scope rule:** keep this task small. It should produce one explicit screening helper with a clear disclaimer. Do not add GIS transect extraction, QUAL2K inputs, TTools emulation, or many nested threshold checks in this phase.

- [ ] **Step 1: Write failing context tests**

Create `tests/test_temperature_context.py`:

```python
from hydrology.analysis.temperature_context import classify_thermal_sensitivity


def test_classify_thermal_sensitivity_high_for_wide_unshaded_low_flow_reach():
    result = classify_thermal_sensitivity(
        summer_flow_cfs=20,
        channel_width_m=12,
        canopy_cover_pct=15,
        groundwater_gain_cfs=-3,
    )

    assert result["class"] == "high"
    assert "low canopy cover" in result["drivers"]
    assert "losing reach" in result["drivers"]


def test_classify_thermal_sensitivity_lower_for_shaded_gaining_reach():
    result = classify_thermal_sensitivity(
        summer_flow_cfs=80,
        channel_width_m=4,
        canopy_cover_pct=85,
        groundwater_gain_cfs=10,
    )

    assert result["class"] in {"low", "moderate"}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest tests/test_temperature_context.py -q
```

Expected: FAIL because module does not exist.

- [ ] **Step 3: Implement lightweight classifier**

Create `hydrology/analysis/temperature_context.py`:

```python
from __future__ import annotations

from typing import Dict, List


def classify_thermal_sensitivity(
    summer_flow_cfs: float | None,
    channel_width_m: float | None,
    canopy_cover_pct: float | None,
    groundwater_gain_cfs: float | None,
) -> Dict[str, object]:
    score = 0
    drivers: List[str] = []

    if summer_flow_cfs is not None and summer_flow_cfs < 50:
        score += 1
        drivers.append("low summer flow")
    if channel_width_m is not None and channel_width_m >= 10:
        score += 1
        drivers.append("wide channel")
    if canopy_cover_pct is not None and canopy_cover_pct < 40:
        score += 2
        drivers.append("low canopy cover")
    elif canopy_cover_pct is not None and canopy_cover_pct >= 75:
        score -= 1
        drivers.append("high canopy cover")
    if groundwater_gain_cfs is not None and groundwater_gain_cfs < -1:
        score += 1
        drivers.append("losing reach")
    elif groundwater_gain_cfs is not None and groundwater_gain_cfs > 1:
        score -= 1
        drivers.append("gaining reach")

    if score >= 3:
        label = "high"
    elif score >= 1:
        label = "moderate"
    else:
        label = "low"

    return {
        "class": label,
        "score": score,
        "drivers": drivers,
        "note": "Screening context only; this is not a QUAL2K, Heat Source, Shade, or TTools model.",
    }
```

- [ ] **Step 4: Export**

Modify `hydrology/analysis/__init__.py`:

```python
from .temperature_context import classify_thermal_sensitivity
```

- [ ] **Step 5: Run focused tests**

Run:

```powershell
python -m pytest tests/test_temperature_context.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```powershell
git add hydrology/analysis/temperature_context.py hydrology/analysis/__init__.py tests/test_temperature_context.py
git commit -m "feat: add reach thermal sensitivity context"
```

---

## Task 8: Flood Frequency Diagnostics And Deterministic CIs

**Files:**
- Modify: `hydrology/analysis/frequency.py`
- Test: `tests/test_frequency.py`

- [ ] **Step 1: Write failing frequency tests**

Create `tests/test_frequency.py`:

```python
import numpy as np

from hydrology.analysis.frequency import estimate_return_periods, flood_frequency_diagnostics


def test_estimate_return_periods_is_reproducible_with_seed():
    peaks = np.array([100, 120, 130, 150, 180, 220, 260, 300, 360, 420, 500, 620])

    a = estimate_return_periods(peaks, periods=[10, 50], distribution="lp3", random_seed=42)
    b = estimate_return_periods(peaks, periods=[10, 50], distribution="lp3", random_seed=42)

    assert a[["lower_ci", "upper_ci"]].equals(b[["lower_ci", "upper_ci"]])


def test_flood_frequency_diagnostics_returns_plotting_table():
    peaks = np.array([100, 120, 130, 150, 180, 220, 260, 300, 360, 420, 500, 620])

    diagnostics = flood_frequency_diagnostics(peaks, distribution="lp3")

    assert {"observed_flow_cfs", "fitted_flow_cfs", "exceedance_prob", "return_period"}.issubset(diagnostics.columns)
    assert len(diagnostics) == len(peaks)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest tests/test_frequency.py -q
```

Expected: FAIL because `random_seed` and diagnostics are missing.

- [ ] **Step 3: Add deterministic bootstrap seed**

Modify `estimate_return_periods()` signature:

```python
random_seed: int | None = None,
```

Replace bootstrap sampling:

```python
rng = np.random.default_rng(random_seed)
boot_sample = rng.choice(peaks, size=n, replace=True)
```

- [ ] **Step 4: Add diagnostics helper**

Add to `hydrology/analysis/frequency.py`:

```python
def flood_frequency_diagnostics(peaks: np.ndarray, distribution: str = "lp3") -> pd.DataFrame:
    peaks = np.asarray(peaks, dtype=float)
    peaks = peaks[np.isfinite(peaks) & (peaks > 0)]
    if len(peaks) < 10:
        return pd.DataFrame()

    positions = get_plotting_positions(peaks)
    fits = fit_flood_frequency(peaks, distributions=[distribution], return_periods=positions["return_period"].tolist())
    if distribution not in fits:
        return pd.DataFrame()

    fit = fits[distribution]
    rows = []
    for _, row in positions.iterrows():
        rp = float(row["return_period"])
        rows.append(
            {
                "observed_flow_cfs": float(row["flow_cfs"]),
                "fitted_flow_cfs": float(fit.quantiles.get(rp, np.nan)),
                "exceedance_prob": float(row["exceedance_prob"]),
                "return_period": rp,
                "distribution": fit.display_name,
            }
        )
    return pd.DataFrame(rows)
```

- [ ] **Step 5: Run focused tests**

Run:

```powershell
python -m pytest tests/test_frequency.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```powershell
git add hydrology/analysis/frequency.py tests/test_frequency.py
git commit -m "feat: add flood frequency diagnostics"
```

---

## Task 9: Shared Validation Helpers

**Files:**
- Create: `hydrology/analysis/validation.py`
- Modify: `hydrology/analysis/__init__.py`
- Test: `tests/test_validation_cases.py`

**Scope rule:** validation helpers are for case studies and tests. Do not call them inside core baseflow, signatures, changepoint, frequency, or reach algorithms.

- [ ] **Step 1: Write failing validation tests**

Create `tests/test_validation_cases.py`:

```python
from hydrology.analysis.validation import validate_range, validate_relative_error


def test_validate_relative_error_passes_within_tolerance():
    result = validate_relative_error("100yr flood", observed=443000, expected=475000, tolerance=0.10)

    assert result.status == "PASS"
    assert abs(result.relative_error) < 0.10


def test_validate_range_flags_out_of_range():
    result = validate_range("BFI", value=0.95, lower=0.55, upper=0.85)

    assert result.status == "FLAG"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest tests/test_validation_cases.py -q
```

Expected: FAIL because module does not exist.

- [ ] **Step 3: Implement validation helpers**

Create `hydrology/analysis/validation.py`:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationResult:
    metric: str
    status: str
    value: float
    expected: float | str
    relative_error: float | None = None
    message: str = ""


def validate_relative_error(metric: str, observed: float, expected: float, tolerance: float) -> ValidationResult:
    rel = (observed - expected) / expected if expected else float("inf")
    status = "PASS" if abs(rel) <= tolerance else "FLAG"
    return ValidationResult(
        metric=metric,
        status=status,
        value=observed,
        expected=expected,
        relative_error=rel,
        message=f"{metric}: {observed:g} vs {expected:g}, rel error {rel:.3f}, tolerance {tolerance:.3f}",
    )


def validate_range(metric: str, value: float, lower: float, upper: float) -> ValidationResult:
    status = "PASS" if lower <= value <= upper else "FLAG"
    return ValidationResult(
        metric=metric,
        status=status,
        value=value,
        expected=f"{lower:g} to {upper:g}",
        message=f"{metric}: {value:g}, expected {lower:g} to {upper:g}",
    )
```

- [ ] **Step 4: Export**

Modify `hydrology/analysis/__init__.py`:

```python
from .validation import ValidationResult, validate_range, validate_relative_error
```

- [ ] **Step 5: Run focused tests**

Run:

```powershell
python -m pytest tests/test_validation_cases.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```powershell
git add hydrology/analysis/validation.py hydrology/analysis/__init__.py tests/test_validation_cases.py
git commit -m "feat: add validation helpers for case studies"
```

---

## Task 10: HydroPlot Case Study Template And Runner

**Files:**
- Create: `docs/cases/_template/README.md`
- Create: `docs/cases/_template/case.yml`
- Create: `scripts/run_case_study.py`
- Test: `tests/test_validation_cases.py`

- [ ] **Step 1: Add case template docs**

Create `docs/cases/_template/README.md`:

```markdown
---
case: <slug>
title: "<one-line case title>"
hydroplot_version: "<version or commit>"
showcases:
  - <analysis_feature>
data_source: "<dataset, URL, and access date>"
runtime_minutes: 0
created: YYYY-MM-DD
---

# <Title>

## Scenario

Describe the PNW hydrology question, the reach or gauge, and why it matters.

## What This Proves About HydroPlot

- `<feature>`: describe the reusable HydroPlot analysis behavior being validated.

## How To Run

```powershell
python scripts/run_case_study.py docs/cases/<case>/case.yml
```

## Outputs

- `outputs/<file>.csv`: describe table.
- `outputs/<file>.png`: describe figure.

## Validation

List the published report, agency method, or expected range used for PASS/FLAG checks.
```

- [ ] **Step 2: Add case config template**

Create `docs/cases/_template/case.yml`:

```yaml
case: template
site_ids: []
start_date: "2000-01-01"
end_date: "2025-01-01"
analyses:
  - baseflow
  - signatures
validation: []
```

- [ ] **Step 3: Add minimal runner**

Create `scripts/run_case_study.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

import yaml


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/run_case_study.py docs/cases/<case>/case.yml")
        return 2
    config_path = Path(sys.argv[1])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = config_path.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    summary = output_dir / "run_summary.md"
    summary.write_text(
        f"# {config['case']} run summary\n\nConfigured analyses: {', '.join(config.get('analyses', []))}\n",
        encoding="utf-8",
    )
    print(f"Wrote {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run runner smoke test**

Run:

```powershell
python scripts/run_case_study.py docs/cases/_template/case.yml
```

Expected: `docs/cases/_template/outputs/run_summary.md` created.

- [ ] **Step 5: Commit**

Run:

```powershell
git add docs/cases/_template scripts/run_case_study.py
git commit -m "docs: add HydroPlot validation case template"
```

---

## Task 11: PNW Baseflow And Groundwater Validation Cases

**Files:**
- Create: `docs/cases/pnw_baseflow_signatures/README.md`
- Create: `docs/cases/pnw_baseflow_signatures/case.yml`
- Create: `docs/cases/spokane_groundwater_reach/README.md`
- Create: `docs/cases/spokane_groundwater_reach/case.yml`
- Modify: `scripts/run_case_study.py`
- Test: `tests/test_validation_cases.py`

- [ ] **Step 1: Create baseflow case doc**

Create `docs/cases/pnw_baseflow_signatures/README.md` with:

```markdown
# PNW Baseflow And Hydrologic Signatures

## Scenario

This case validates HydroPlot's baseflow and hydrologic signature implementation on a Pacific Northwest streamflow record. It focuses on method agreement, plausible BFI range, flow-duration behavior, and flashiness.

## What This Proves About HydroPlot

- `lyne_hollick_filter`: bounded multi-pass recursive filter.
- `eckhardt_filter`: independent two-parameter filter.
- `compute_hydrologic_signatures`: compact basin-behavior summary.
- `validate_range`: explicit PASS/FLAG result instead of informal notebook checks.

## Outputs

- `outputs/baseflow_components.csv`
- `outputs/signatures.csv`
- `outputs/validation_summary.csv`

## Validation

BFI and flashiness are screened against method-comparison ranges. These checks are not a substitute for a basin-specific groundwater study.
```

- [ ] **Step 2: Create Spokane reach case doc**

Create `docs/cases/spokane_groundwater_reach/README.md` with:

```markdown
# Spokane Groundwater Reach Screening

## Scenario

This case validates HydroPlot's reach-scale groundwater screening on a PNW reach where upstream/downstream discharge differences are important for dry-season interpretation.

## What This Proves About HydroPlot

- `summarize_reach_gain_loss`: paired upstream/downstream gain-loss calculation.
- Low-flow median gain/loss: groundwater contribution during dry windows.
- `classify_thermal_sensitivity`: screening context for shade, low flow, and losing/gaining reach conditions.

## Outputs

- `outputs/reach_gain_loss.csv`
- `outputs/validation_summary.csv`

## Validation

This is a screening workflow. It does not claim to replace seepage runs, groundwater modeling, TTools, Shade, Heat Source, QUAL2K, or QUAL2Kw.
```

- [ ] **Step 3: Create case configs**

Create `docs/cases/pnw_baseflow_signatures/case.yml`:

```yaml
case: pnw_baseflow_signatures
site_ids:
  - "12422500"
start_date: "2000-01-01"
end_date: "2025-01-01"
analyses:
  - baseflow
  - signatures
validation:
  - metric: baseflow_index_lh
    lower: 0.20
    upper: 0.95
```

Create `docs/cases/spokane_groundwater_reach/case.yml`:

```yaml
case: spokane_groundwater_reach
site_ids:
  upstream: "12422500"
  downstream: "12424000"
start_date: "2015-07-01"
end_date: "2015-09-30"
reach_km: 20
analyses:
  - reach_gain_loss
  - thermal_context
validation: []
```

- [ ] **Step 4: Extend runner to produce real outputs**

Modify `scripts/run_case_study.py` to:

```python
from hydrology.analysis.baseflow import compare_baseflow_methods, lyne_hollick_filter, eckhardt_filter
from hydrology.analysis.signatures import compute_hydrologic_signatures
from hydrology.analysis.reach_groundwater import summarize_reach_gain_loss
from hydrology.data.usgs import fetch_discharge_data
```

For `baseflow`, fetch the first `site_ids` entry, run both filters, and write `baseflow_components.csv`.

For `signatures`, write `signatures.csv`.

For `reach_gain_loss`, fetch upstream/downstream and write `reach_gain_loss.csv`.

Keep network failures explicit: if live fetch fails and no cached input exists, return non-zero with a clear message.

- [ ] **Step 5: Run case smoke tests**

Run:

```powershell
python scripts/run_case_study.py docs/cases/pnw_baseflow_signatures/case.yml
```

Expected: outputs written, or explicit network/API failure documented.

Run:

```powershell
python scripts/run_case_study.py docs/cases/spokane_groundwater_reach/case.yml
```

Expected: outputs written, or explicit network/API failure documented.

- [ ] **Step 6: Commit**

Run:

```powershell
git add docs/cases/pnw_baseflow_signatures docs/cases/spokane_groundwater_reach scripts/run_case_study.py
git commit -m "feat: add PNW validation case studies"
```

---

## Task 12: Dashboard Integration For Indicators

**Files:**
- Modify: `hydrology/app/page_modules/indicators.py`
- Test: `tests/test_app_indicators.py`

**Performance rule:** compute method comparison once per selected site/date range and keep it behind existing Streamlit caching or an explicit user action. Do not recompute both filters for every chart render.

- [ ] **Step 1: Write or update app test**

Add to `tests/test_app_indicators.py`:

```python
def test_indicators_imports_baseflow_comparison():
    from hydrology.analysis.baseflow import compare_baseflow_methods

    assert callable(compare_baseflow_methods)
```

- [ ] **Step 2: Run test**

Run:

```powershell
python -m pytest tests/test_app_indicators.py -q
```

Expected: PASS before UI text changes.

- [ ] **Step 3: Wire method comparison**

In `hydrology/app/page_modules/indicators.py`, where BFI is computed, import:

```python
from hydrology.analysis.baseflow import compare_baseflow_methods
```

Display:

- Lyne-Hollick BFI
- Eckhardt BFI
- difference
- agreement label

Keep existing rolling BFI chart to avoid disrupting the page.

- [ ] **Step 4: Run app-focused tests**

Run:

```powershell
python -m pytest tests/test_app_indicators.py tests/test_app_plot_config.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git add hydrology/app/page_modules/indicators.py tests/test_app_indicators.py
git commit -m "feat: show baseflow method comparison in dashboard"
```

---

## Task 13: Dashboard Integration For Reach Groundwater

**Files:**
- Modify: `hydrology/app/page_modules/reach_analysis.py`
- Test: `tests/test_app_shared_conditions.py` or create `tests/test_app_reach_groundwater.py`

**Performance rule:** run gain/loss summaries on already-fetched paired daily series. Do not trigger extra USGS/NLDI calls solely to populate optional context.

- [ ] **Step 1: Write app import test**

Create `tests/test_app_reach_groundwater.py`:

```python
def test_reach_groundwater_helpers_importable():
    from hydrology.analysis.reach_groundwater import summarize_reach_gain_loss
    from hydrology.analysis.temperature_context import classify_thermal_sensitivity

    assert callable(summarize_reach_gain_loss)
    assert callable(classify_thermal_sensitivity)
```

- [ ] **Step 2: Run test**

Run:

```powershell
python -m pytest tests/test_app_reach_groundwater.py -q
```

Expected: PASS.

- [ ] **Step 3: Wire reach summary**

In `hydrology/app/page_modules/reach_analysis.py`, add a compact summary near the upstream/downstream analysis:

- median gain/loss cfs
- median gain/loss cfs/km when reach length is available
- low-flow median gain/loss
- classification
- confidence
- thermal sensitivity note when canopy/width values are provided or estimated

Use plain labels and do not claim this is a calibrated groundwater model.

- [ ] **Step 4: Run app-focused tests**

Run:

```powershell
python -m pytest tests/test_app_reach_groundwater.py tests/test_app_shared_conditions.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git add hydrology/app/page_modules/reach_analysis.py tests/test_app_reach_groundwater.py
git commit -m "feat: add reach groundwater dashboard summary"
```

---

## Task 14: Dashboard Integration For Frequency And Trend Diagnostics

**Files:**
- Modify: `hydrology/app/page_modules/single_analysis.py`
- Test: `tests/test_app_plot_config.py`
- Test: `tests/test_frequency.py`
- Test: `tests/test_changepoints.py`

**Performance rule:** diagnostics must stay inside the existing on-demand flood/trend workflows. Do not add new page-load computations.

- [ ] **Step 1: Add imports behind existing analysis actions**

In `single_analysis.py`, import:

```python
from hydrology.analysis.changepoints import pettitt_test
from hydrology.analysis.frequency import flood_frequency_diagnostics
```

- [ ] **Step 2: Add flood diagnostic table**

After flood frequency fit succeeds, show a compact diagnostics table with observed flow, fitted flow, exceedance probability, and return period.

- [ ] **Step 3: Add trend changepoint summary**

Where annual trend output is available, call `pettitt_test()` and show:

- change year
- p-value
- mean before
- mean after

Only show when there are enough annual values.

- [ ] **Step 4: Run focused tests**

Run:

```powershell
python -m pytest tests/test_frequency.py tests/test_changepoints.py tests/test_app_plot_config.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git add hydrology/app/page_modules/single_analysis.py
git commit -m "feat: add frequency and changepoint diagnostics to dashboard"
```

---

## Task 15: Full Verification And Local Dashboard Check

**Files:**
- No code changes unless fixing issues.

- [ ] **Step 1: Run focused new tests**

Run:

```powershell
python -m pytest tests/test_baseflow.py tests/test_signatures.py tests/test_changepoints.py tests/test_reach_groundwater.py tests/test_temperature_context.py tests/test_frequency.py tests/test_validation_cases.py -q
```

Expected: PASS.

- [ ] **Step 2: Run relevant app tests**

Run:

```powershell
python -m pytest tests/test_app_indicators.py tests/test_app_plot_config.py tests/test_app_shared_conditions.py tests/test_app_reach_groundwater.py -q
```

Expected: PASS.

- [ ] **Step 3: Run broader regression**

Run:

```powershell
python -m pytest -q
```

Expected: PASS or documented unrelated existing failures.

- [ ] **Step 4: Start dashboard locally**

Run:

```powershell
streamlit run hydrology/app/streamlit_app.py
```

Expected: app starts and reports local URL.

- [ ] **Step 5: Manually inspect changed pages**

Open the local Streamlit URL and inspect:

- Indicators page: BFI method comparison appears and existing rolling BFI still renders.
- Reach page: gain/loss classification appears for paired sites.
- Single analysis page: flood diagnostics and changepoint summaries appear only with sufficient data.

- [ ] **Step 6: Commit any verification fixes**

Run:

```powershell
git status --short
git diff
git add <fixed files>
git commit -m "fix: stabilize groundwater dashboard verification"
```

Expected: only needed if verification found issues.

---

## Task 16: Push And Open GitHub PR

**Files:**
- No code changes.

- [ ] **Step 1: Review final diff**

Run:

```powershell
git diff main...HEAD --stat
git log --oneline main..HEAD
```

Expected: focused commits matching this plan.

- [ ] **Step 2: Push branch**

Run:

```powershell
git push -u origin feature/hydroplot-groundwater-reach-validation
```

Expected: branch pushed.

- [ ] **Step 3: Open draft PR**

Use GitHub UI or `gh pr create --draft` if authenticated:

```powershell
gh pr create --draft --title "Improve HydroPlot groundwater, reach, and validation workflows" --body "Adds tested baseflow comparison, hydrologic signatures, changepoints, reach gain/loss summaries, lightweight thermal context, validation case scaffolding, and dashboard integration."
```

Expected: draft PR opened against `main`.

- [ ] **Step 4: Add PR verification checklist**

PR body should include:

```markdown
## Verification

- [ ] `python -m pytest tests/test_baseflow.py tests/test_signatures.py tests/test_changepoints.py tests/test_reach_groundwater.py tests/test_temperature_context.py tests/test_frequency.py tests/test_validation_cases.py -q`
- [ ] `python -m pytest tests/test_app_indicators.py tests/test_app_plot_config.py tests/test_app_shared_conditions.py tests/test_app_reach_groundwater.py -q`
- [ ] `python -m pytest -q`
- [ ] Streamlit dashboard manually checked for Indicators, Reach Analysis, and Single Analysis pages

## Notes

This does not add AquaScope as a dependency and does not claim to implement QUAL2K, QUAL2Kw, TTools, Shade, or Heat Source.
```

---

## Final Verification Matrix

| Area | Verification |
|---|---|
| Baseflow methods | `tests/test_baseflow.py`, dashboard indicators check |
| Signatures | `tests/test_signatures.py`, PNW case output |
| Changepoints | `tests/test_changepoints.py`, single analysis display |
| Flood diagnostics | `tests/test_frequency.py`, single analysis display |
| Reach groundwater | `tests/test_reach_groundwater.py`, reach page display |
| Thermal context | `tests/test_temperature_context.py`, reach page note |
| Case-study discipline | `tests/test_validation_cases.py`, `docs/cases/*/outputs` |
| Regression | `python -m pytest -q` |
| User-visible app | local Streamlit check |
| GitHub traceability | feature branch, task commits, draft PR checklist |

## Scope Boundaries

This plan intentionally does not:

- Add AquaScope as a dependency.
- Expand HydroPlot to every national gauge.
- Implement full QUAL2K/QUAL2Kw.
- Implement TTools GIS transect extraction.
- Claim calibrated groundwater modeling.
- Replace agency QAPP workflows.

It does create the foundation needed to add those heavier workflows later with clear interfaces and validation.
