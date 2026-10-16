from pathlib import Path

from hydrology.visualization.map_utils import BASEMAP_ATTR, BASEMAP_LABELS, BASEMAP_TILES


ROOT = Path(__file__).resolve().parents[1]


def test_basemap_tiles_do_not_use_carto():
    assert "carto" not in BASEMAP_TILES.lower()
    assert "carto" not in BASEMAP_LABELS.lower()
    assert "{z}" in BASEMAP_TILES and "{y}" in BASEMAP_TILES and "{x}" in BASEMAP_TILES
    assert "Esri" in BASEMAP_ATTR


def test_basemap_stops_at_esri_native_zoom():
    import folium

    from hydrology.visualization.map_utils import add_basemap

    m = folium.Map(location=[47.6, -117.4], zoom_start=10, tiles=None)
    add_basemap(m)
    html = m.get_root().render()
    assert "cartocdn" not in html.lower()
    assert "arcgisonline.com" in html
    assert '"maxZoom": 16' in html
    assert '"maxNativeZoom": 16' in html


def test_dashboard_maps_use_shared_basemap():
    paths = [
        "hydrology/visualization/map_utils.py",
        "hydrology/app/page_modules/overview.py",
        "hydrology/app/page_modules/reach_analysis.py",
        "hydrology/app/page_modules/advanced.py",
        "spokane_flood_explorer.py",
    ]
    for rel in paths:
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "CartoDB" not in text
        assert "add_basemap" in text
