from pathlib import Path

import yaml

from scripts.run_case_study import run_case_study


def test_run_case_study_writes_summary(tmp_path: Path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    config_path = case_dir / "case.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "case": "example_case",
                "analyses": ["baseflow", "signatures"],
            }
        ),
        encoding="utf-8",
    )

    summary_path = run_case_study(config_path)

    assert summary_path == case_dir / "outputs" / "run_summary.md"
    assert "example_case" in summary_path.read_text(encoding="utf-8")
