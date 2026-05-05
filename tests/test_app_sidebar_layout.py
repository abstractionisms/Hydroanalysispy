from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_core_pages_do_not_use_sidebar_controls():
    core_pages = [
        ROOT / "hydrology/app/page_modules/overview.py",
        ROOT / "hydrology/app/page_modules/single_analysis.py",
        ROOT / "hydrology/app/page_modules/alerts.py",
        ROOT / "hydrology/app/page_modules/indicators.py",
    ]

    for path in core_pages:
        text = path.read_text(encoding="utf-8")
        assert "st.sidebar" not in text
        assert 'location="sidebar"' not in text
        assert "location='sidebar'" not in text


def test_app_entrypoint_does_not_write_sidebar_footer():
    text = (ROOT / "hydrology/app/streamlit_app.py").read_text(encoding="utf-8")

    assert "st.sidebar" not in text


def test_overview_uses_map_first_workspace_copy():
    text = (ROOT / "hydrology/app/page_modules/overview.py").read_text(encoding="utf-8")

    assert "Station Workspace" in text
    assert "Selected Site" in text
    assert "Map Layers" in text
