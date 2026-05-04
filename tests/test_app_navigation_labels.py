from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_main_navigation_uses_role_specific_labels():
    text = (ROOT / "hydrology/app/styles.py").read_text(encoding="utf-8")

    for label in [
        "Stations",
        "Site Analysis",
        "Compare Sites",
        "Reach Tools",
        "Current Check",
        "Climate Indicators",
        "More Tools",
    ]:
        assert label in text

    for old_label in [
        ">Explore<",
        ">Analyze<",
        ">Compare<",
        ">Monitor<",
    ]:
        assert old_label not in text


def test_workflow_tiles_match_primary_navigation_terms():
    text = (ROOT / "hydrology/app/styles.py").read_text(encoding="utf-8")

    assert "<strong>Stations</strong>" in text
    assert "<strong>Site Analysis</strong>" in text
    assert "<strong>Compare Sites</strong>" in text
    assert "<strong>Current Check</strong>" in text
