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


def test_styles_include_tile_hover_tooltips():
    text = (ROOT / "hydrology/app/styles.py").read_text(encoding="utf-8")
    assert ".tile-tip" in text
    assert "def _tile_tip_html" in text
    assert "section: str = \"all\"" in text or 'section: str = "all"' in text


def test_styles_include_premium_motion_and_reduced_motion_guard():
    """New fluid/animated premium polish (inspired by modern award-winning web UIs)
    must be present in the CSS system and guarded for accessibility.
    """
    text = (ROOT / "hydrology/app/styles.py").read_text(encoding="utf-8")

    for token in [
        "@keyframes fadeInUp",
        "@keyframes cardPop",
        "@keyframes subtlePulse",
        "animation: fadeInUp",
        "animation: cardPop",
        "prefers-reduced-motion: reduce",
        "translateY(-2px)",
        "will-change: transform",
    ]:
        assert token in text, f"Missing premium motion token: {token}"

    # Inspiration comment / docstring reference present
    assert "x.com/twetsfyp/status/2065283731833651709" in text or "award-winning" in text
