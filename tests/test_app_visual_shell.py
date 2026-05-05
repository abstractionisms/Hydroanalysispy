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
