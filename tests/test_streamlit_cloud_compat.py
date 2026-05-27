from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pandas_dependency_is_pinned_below_v3_for_meteostat():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert "pandas>=2.0.0,<3.0.0" in requirements


def test_streamlit_width_api_uses_supported_width_argument():
    offenders = []
    for path in (ROOT / "hydrology").rglob("*.py"):
        if "use_container_width" in path.read_text(encoding="utf-8"):
            offenders.append(path.relative_to(ROOT).as_posix())

    assert offenders == []
