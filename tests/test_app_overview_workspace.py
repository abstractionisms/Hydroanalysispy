from hydrology.app.shared import build_site_summary
from hydrology.app.page_modules.overview import build_layer_status


def test_build_site_summary_formats_core_fields():
    site_info = {
        "site_id": "12422500",
        "description": "Spokane River at Spokane, WA",
        "latitude": 47.6593,
        "longitude": -117.4491,
        "begin_date": "1891-01-01",
    }
    condition = {
        "flow_cfs": 5775.4,
        "percentile": 82.5,
        "source": "USGS seasonal percentile",
    }

    summary = build_site_summary("12422500", site_info, condition)

    assert summary["title"] == "Spokane River at Spokane, WA"
    assert summary["subtitle"] == "USGS 12422500 | 47.6593, -117.4491"
    assert summary["chips"] == [
        {"label": "Flow 5,775 cfs", "state": "ready"},
        {"label": "Above Normal", "state": "ready"},
        {"label": "Record since 1891", "state": "ready"},
    ]


def test_build_layer_status_labels_requested_layers():
    status = build_layer_status(
        show_boundary=True,
        show_flowlines=True,
        show_dams=False,
        has_pynhd=True,
        has_pygeohydro=True,
    )

    assert status == [
        {"label": "Boundary requested", "state": "limited"},
        {"label": "Flowlines requested", "state": "limited"},
        {"label": "Dams off", "state": "blocked"},
    ]
