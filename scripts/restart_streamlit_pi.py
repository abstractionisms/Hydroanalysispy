#!/usr/bin/env python3
"""Restart HydroPlot Streamlit on the Pi without shell self-match on pgrep -f."""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = 8501
LOG = Path("/tmp/streamlit-hydro.log")


def _pids_listening_on(port: int) -> list[int]:
    try:
        out = subprocess.check_output(
            ["ss", "-lptn", f"sport = :{port}"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        out = ""
    pids: list[int] = []
    for token in out.replace(",", " ").split():
        if token.startswith("pid="):
            try:
                pids.append(int(token.split("=")[1]))
            except ValueError:
                pass
    return sorted(set(pids))


def _kill(pids: list[int]) -> None:
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(1.5)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def main() -> int:
    old = _pids_listening_on(PORT)
    if old:
        print(f"stopping pids on :{PORT}: {old}")
        _kill(old)
        time.sleep(1)

    venv_streamlit = ROOT / "venv" / "bin" / "streamlit"
    if not venv_streamlit.exists():
        print("streamlit binary missing", file=sys.stderr)
        return 1

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    log_f = open(LOG, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            str(venv_streamlit),
            "run",
            "hydrology/app/app.py",
            "--server.headless",
            "true",
            "--server.address",
            "0.0.0.0",
            "--server.port",
            str(PORT),
        ],
        cwd=str(ROOT),
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    time.sleep(3)
    print(f"started pid={proc.pid}")
    try:
        print(LOG.read_text(encoding="utf-8", errors="replace")[-800:])
    except Exception as exc:
        print(f"(log read failed: {exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
