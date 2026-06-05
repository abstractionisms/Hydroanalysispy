from __future__ import annotations

import sys
from pathlib import Path

import yaml


def run_case_study(config_path: Path) -> Path:
    """Run a lightweight case config and write a summary artifact."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = config_path.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    summary = output_dir / "run_summary.md"
    analyses = ", ".join(config.get("analyses", []))
    summary.write_text(
        f"# {config['case']} run summary\n\nConfigured analyses: {analyses}\n",
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
