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


def test_run_case_study_writes_baseflow_and_signature_outputs(tmp_path: Path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    flow_path = case_dir / "daily_flow.csv"
    flow_path.write_text(
        "date,discharge_cfs\n"
        "2020-01-01,100\n"
        "2020-01-02,120\n"
        "2020-01-03,140\n"
        "2020-01-04,110\n"
        "2020-01-05,90\n"
        "2020-01-06,95\n",
        encoding="utf-8",
    )
    config_path = case_dir / "case.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "case": "flow_case",
                "input_daily_flow_csv": str(flow_path),
                "analyses": ["baseflow", "signatures"],
            }
        ),
        encoding="utf-8",
    )

    run_case_study(config_path)

    assert (case_dir / "outputs" / "baseflow_components.csv").exists()
    assert (case_dir / "outputs" / "signatures.csv").exists()
