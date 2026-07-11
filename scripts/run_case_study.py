from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

from hydrology.analysis.baseflow import eckhardt_filter, lyne_hollick_filter
from hydrology.analysis.signatures import compute_hydrologic_signatures
from hydrology.analysis.validation import validate_range, validate_relative_error
from hydrology.data.usgs import fetch_discharge_data


def _load_daily_flow(config: dict, config_path: Path) -> pd.Series | None:
    flow_csv = config.get("input_daily_flow_csv")
    if not flow_csv:
        site_ids = config.get("site_ids", [])
        if not site_ids:
            return None
        df = fetch_discharge_data(str(site_ids[0]), config.get("start_date"), config.get("end_date"))
        if df is None or df.empty:
            return None
        return pd.Series(df["Discharge_cfs"].astype(float).values, index=df.index, name="discharge_cfs")

    path = Path(flow_csv)
    if not path.is_absolute():
        path = config_path.parent / path
    df = pd.read_csv(path, parse_dates=["date"])
    return pd.Series(df["discharge_cfs"].astype(float).values, index=df["date"], name="discharge_cfs")


def _write_flow_outputs(flow: pd.Series, analyses: list[str], output_dir: Path) -> dict[str, float]:
    metrics: dict[str, float] = {"n_days": float(flow.dropna().shape[0])}

    if "baseflow" in analyses:
        lh = lyne_hollick_filter(flow)
        eckhardt = eckhardt_filter(flow)
        metrics["baseflow_index_lh"] = lh.bfi
        metrics["baseflow_index_eckhardt"] = eckhardt.bfi
        metrics["baseflow_index_difference"] = abs(lh.bfi - eckhardt.bfi)
        components = lh.components.rename(columns={"baseflow": "baseflow_lyne_hollick"})
        components["baseflow_eckhardt"] = eckhardt.components["baseflow"]
        components["quickflow_lyne_hollick"] = components["total_flow"] - components["baseflow_lyne_hollick"]
        components.to_csv(output_dir / "baseflow_components.csv", index_label="date")

    if "signatures" in analyses:
        signatures = compute_hydrologic_signatures(flow)
        metrics.update({key: float(value) for key, value in signatures.items() if pd.notna(value)})
        pd.DataFrame([signatures]).to_csv(output_dir / "signatures.csv", index=False)

    return metrics


def _write_validation_summary(config: dict, metrics: dict[str, float], output_dir: Path) -> None:
    rows = []
    for check in config.get("validation", []):
        metric = check["metric"]
        if metric not in metrics:
            rows.append(
                {
                    "metric": metric,
                    "status": "MISSING",
                    "value": "",
                    "expected": "",
                    "relative_error": "",
                    "message": f"{metric}: metric was not produced by configured analyses",
                }
            )
            continue

        value = float(metrics[metric])
        if "lower" in check and "upper" in check:
            result = validate_range(metric, value, float(check["lower"]), float(check["upper"]))
        else:
            result = validate_relative_error(metric, value, float(check["expected"]), float(check.get("tolerance", 0.0)))
        rows.append(
            {
                "metric": result.metric,
                "status": result.status,
                "value": result.value,
                "expected": result.expected,
                "relative_error": "" if result.relative_error is None else result.relative_error,
                "message": result.message,
            }
        )

    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "validation_summary.csv", index=False)


def run_case_study(config_path: Path) -> Path:
    """Run a lightweight case config and write a summary artifact."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = config_path.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    summary = output_dir / "run_summary.md"
    analyses = config.get("analyses", [])
    flow = _load_daily_flow(config, config_path)
    metrics: dict[str, float] = {}
    if flow is not None:
        metrics.update(_write_flow_outputs(flow, analyses, output_dir))
    _write_validation_summary(config, metrics, output_dir)
    summary.write_text(
        (
            f"# {config['case']} run summary\n\n"
            f"Configured analyses: {', '.join(analyses)}\n\n"
            f"Validation checks: {len(config.get('validation', []))}\n"
        ),
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
