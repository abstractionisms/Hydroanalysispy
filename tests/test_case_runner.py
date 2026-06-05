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


def test_run_case_study_writes_validation_summary(tmp_path: Path):
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
                "case": "validated_flow_case",
                "input_daily_flow_csv": str(flow_path),
                "analyses": ["baseflow", "signatures"],
                "validation": [
                    {"metric": "baseflow_index_lh", "lower": 0.0, "upper": 1.0},
                    {"metric": "n_days", "expected": 6, "tolerance": 0.0},
                ],
            }
        ),
        encoding="utf-8",
    )

    run_case_study(config_path)

    summary = case_dir / "outputs" / "validation_summary.csv"
    assert summary.exists()
    text = summary.read_text(encoding="utf-8")
    assert "baseflow_index_lh" in text
    assert "n_days" in text
    assert "PASS" in text


def test_run_case_study_fetches_daily_flow_from_site_config(tmp_path: Path, monkeypatch):
    import pandas as pd
    import scripts.run_case_study as runner

    def fake_fetch(site_id, start_date, end_date):
        assert site_id == "12345678"
        assert start_date == "2020-01-01"
        assert end_date == "2020-01-03"
        return pd.DataFrame(
            {"Discharge_cfs": [100.0, 110.0, 120.0]},
            index=pd.date_range("2020-01-01", periods=3, freq="D"),
        )

    monkeypatch.setattr(runner, "fetch_discharge_data", fake_fetch)
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    config_path = case_dir / "case.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "case": "fetched_flow_case",
                "site_ids": ["12345678"],
                "start_date": "2020-01-01",
                "end_date": "2020-01-03",
                "analyses": ["baseflow"],
                "validation": [{"metric": "n_days", "expected": 3, "tolerance": 0.0}],
            }
        ),
        encoding="utf-8",
    )

    run_case_study(config_path)

    assert (case_dir / "outputs" / "baseflow_components.csv").exists()
    validation = (case_dir / "outputs" / "validation_summary.csv").read_text(encoding="utf-8")
    assert "n_days" in validation
    assert "PASS" in validation


def test_pnw_case_configs_are_parseable():
    cases = [
        Path("docs/cases/pnw_baseflow_signatures/case.yml"),
        Path("docs/cases/spokane_groundwater_reach/case.yml"),
    ]

    for case_path in cases:
        config = yaml.safe_load(case_path.read_text(encoding="utf-8"))
        assert config["case"]
        assert config["analyses"]
