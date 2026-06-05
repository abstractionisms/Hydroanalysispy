from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

from hydrology.analysis.baseflow import eckhardt_filter, lyne_hollick_filter
from hydrology.analysis.signatures import compute_hydrologic_signatures


def _load_daily_flow(config: dict, config_path: Path) -> pd.Series | None:
    flow_csv = config.get("input_daily_flow_csv")
    if not flow_csv:
        return None

    path = Path(flow_csv)
    if not path.is_absolute():
        path = config_path.parent / path
    df = pd.read_csv(path, parse_dates=["date"])
    return pd.Series(df["discharge_cfs"].astype(float).values, index=df["date"], name="discharge_cfs")


def _write_flow_outputs(flow: pd.Series, analyses: list[str], output_dir: Path) -> None:
    if "baseflow" in analyses:
        lh = lyne_hollick_filter(flow)
        eckhardt = eckhardt_filter(flow)
        components = lh.components.rename(columns={"baseflow": "baseflow_lyne_hollick"})
        components["baseflow_eckhardt"] = eckhardt.components["baseflow"]
        components["quickflow_lyne_hollick"] = components["total_flow"] - components["baseflow_lyne_hollick"]
        components.to_csv(output_dir / "baseflow_components.csv", index_label="date")

    if "signatures" in analyses:
        signatures = compute_hydrologic_signatures(flow)
        pd.DataFrame([signatures]).to_csv(output_dir / "signatures.csv", index=False)


def run_case_study(config_path: Path) -> Path:
    """Run a lightweight case config and write a summary artifact."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = config_path.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    summary = output_dir / "run_summary.md"
    analyses = config.get("analyses", [])
    flow = _load_daily_flow(config, config_path)
    if flow is not None:
        _write_flow_outputs(flow, analyses, output_dir)
    summary.write_text(
        f"# {config['case']} run summary\n\nConfigured analyses: {', '.join(analyses)}\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/run_case_study.py docs/cases/<case>/case.yml")
        return 2
    summary = run_case_study(Path(sys.argv[1]))
    print(f"Wrote {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
