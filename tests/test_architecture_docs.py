from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_architecture_doc_names_current_workflows_and_groundwater_rules():
    text = (ROOT / "docs/architecture.md").read_text(encoding="utf-8")

    for workflow in ["Stations", "Site Analysis", "Compare Sites", "Reach Analysis", "Watershed"]:
        assert workflow in text

    for rule in [
        "USGS groundwater field measurements",
        "Washington Ecology EIM",
        "Washington Ecology well logs",
        "Drop private fields",
        "Do not fetch groundwater data automatically",
    ]:
        assert rule in text
