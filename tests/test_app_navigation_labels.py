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


def test_site_analysis_owns_indicators_without_autoloading():
    text = (ROOT / "hydrology/app/page_modules/single_analysis.py").read_text(encoding="utf-8")

    assert 'view == "Indicators"' in text
    assert "render_site_indicators" in text
    assert "Load Indicators" not in text
    assert "indicator_loaded_key" not in text
    assert "Advanced Visualizations" not in text
    assert "Guided Plot Builder" not in text
    assert "Run Frequency Analysis" not in text


def test_compare_sites_uses_one_chart_mode():
    text = (ROOT / "hydrology/app/page_modules/comparisons.py").read_text(encoding="utf-8")

    assert 'key="compare_sites_chart"' in text
    assert "Export grid" in text
    assert "Static Plot Grid" not in text


def test_stations_does_not_offer_current_conditions_check():
    text = (ROOT / "hydrology/app/page_modules/overview.py").read_text(encoding="utf-8")

    assert "Current Conditions Check" not in text
    assert "render_site_current_check" not in text
    assert "Flood Alerts" not in text
    assert "alerts?site=" not in text
