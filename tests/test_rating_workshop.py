"""Rating workshop sliders must move the equation and the curve."""
import re

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from hydrology.app.page_modules.single_analysis import _snap_slider


def test_snap_slider_lands_on_step_grid():
    lo, hi, step = 10.123, 18.456, 0.01
    snapped = _snap_slider(12.34567, lo, hi, step)
    n = (snapped - lo) / step
    assert abs(n - round(n)) < 1e-8
    assert lo <= snapped <= hi


SCRIPT = """
import numpy as np
import pandas as pd
from hydrology.app.page_modules.single_analysis import _render_rating_curve_workshop

idx = pd.date_range("2020-01-01", periods=80, freq="D")
h = np.linspace(12.0, 22.0, 80)
q = 40.0 * (h - 8.0) ** 1.7
df = pd.DataFrame({"Gage_Height_ft": h, "Discharge_cfs": q}, index=idx)
_render_rating_curve_workshop(df, site_id="TEST", site_desc="Test gage")
"""


def _walk(node):
    yield node
    children = getattr(node, "children", None)
    if isinstance(children, dict):
        for child in children.values():
            yield from _walk(child)


def _equations(at):
    bodies = []
    for node in _walk(at.main):
        if getattr(node, "type", None) == "markdown" and "Equation:" in node.proto.body:
            bodies.append(node.proto.body)
    return bodies


def _chart_html(at) -> str:
    pages = []
    for node in _walk(at.main):
        if getattr(node, "type", None) != "iframe":
            continue
        srcdoc = getattr(node.proto, "srcdoc", "") or ""
        if "Plotly" in srcdoc or "plotly" in srcdoc:
            pages.append(srcdoc)
    if not pages:
        raise AssertionError("rating chart iframe missing")
    return pages[-1]


def _eq_text(bodies) -> str:
    match = re.search(r"`([^`]+)`", bodies[0])
    assert match, bodies[0]
    return match.group(1)


def _on_grid(slider):
    n = (float(slider.value) - float(slider.min)) / float(slider.step)
    assert abs(n - round(n)) < 1e-6


def test_slider_drag_changes_equation_and_curve():
    at = AppTest.from_string(SCRIPT, default_timeout=60).run()
    assert not at.exception

    for slider in at.slider:
        _on_grid(slider)

    before_eq = _equations(at)
    before_html = _chart_html(at)
    assert len(before_eq) == 1
    assert "Q =" in before_eq[0]
    assert _eq_text(before_eq) in before_html
    assert "Your curve" in before_html
    assert "Auto-fit" in before_html
    assert "Your fit · R²=" not in before_html

    exponent = next(slider for slider in at.slider if slider.label.startswith("B"))
    exponent.set_value(float(exponent.min))
    at.run()
    assert not at.exception

    after_eq = _equations(at)
    after_html = _chart_html(at)
    assert after_eq != before_eq
    assert after_html != before_html
    assert _eq_text(after_eq) in after_html
    assert _eq_text(before_eq) not in after_html

    at.button[0].click().run()
    assert not at.exception
    restored = next(slider for slider in at.slider if slider.label.startswith("B"))
    _on_grid(restored)
    assert abs(float(restored.value) - float(exponent.value)) > 0.5
    assert _equations(at) == before_eq


def test_zero_flow_slider_changes_equation():
    at = AppTest.from_string(SCRIPT, default_timeout=60).run()
    assert not at.exception
    at.toggle[0].set_value(True).run()
    assert not at.exception

    h0 = next(slider for slider in at.slider if "H" in slider.label)
    _on_grid(h0)
    before_eq = _equations(at)
    before_html = _chart_html(at)
    h0.set_value(float(h0.max))
    at.run()
    assert not at.exception

    after_eq = _equations(at)
    after_html = _chart_html(at)
    assert after_eq != before_eq
    assert after_html != before_html
    assert _eq_text(after_eq) in after_html
    assert "−" in _eq_text(after_eq)
