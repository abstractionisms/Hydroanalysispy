from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_tracked_text_uses_gage_terminology():
    forbidden = "gau" + "ge"
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    text_suffixes = {".py", ".md", ".txt", ".yml", ".yaml"}

    offenders = []
    for relative_path in tracked:
        path = ROOT / relative_path
        if path.suffix.lower() not in text_suffixes:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if forbidden in text.lower():
            offenders.append(relative_path)

    assert offenders == []
