from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_main_navigation_uses_role_specific_labels():
    text = (ROOT / "hydrology/app/styles.py").read_text(encoding="utf-8")

    for label in [
        "Stations",
        "Site Analysis",
        "Compare Sites",
        "Reach Analysis",
        "Watershed",
    ]:
        assert label in text

    for old_label in [
        ">Explore<",
        ">Analyze<",
        ">Compare<",
        ">Monitor<",
        "More Tools",
        "Current Check",
        "Climate Indicators",
    ]:
        assert old_label not in text


def test_app_shell_uses_coalesced_page_set():
    text = (ROOT / "hydrology/app/app.py").read_text(encoding="utf-8")

    for title in [
        'title="Stations"',
        'title="Site Analysis"',
        'title="Compare Sites"',
        'title="Reach Analysis"',
        'title="Watershed"',
    ]:
        assert title in text

    for removed in [
        'title="Alerts"',
        'title="Indicators"',
        'title="Advanced"',
        "render_workflow_strip()",
    ]:
        assert removed not in text


def test_compare_page_owns_multisite_relationships():
    text = (ROOT / "hydrology/app/page_modules/comparisons.py").read_text(encoding="utf-8")

    assert "Site Relationships" in text
    assert "_multisite_analysis" in text
