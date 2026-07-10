#!/usr/bin/env python3
"""Restart HydroPlot Streamlit on the Pi without self-matching pkill."""
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path("/home/cam/source/repos/Hydrology")
LOG = ROOT / "outputs" / "logs" / "streamlit.log"
LOG.parent.mkdir(parents=True, exist_ok=True)

killed = 0
out = subprocess.check_output(["ps", "-eo", "pid,cmd"], text=True)
for line in out.splitlines():
    if "streamlit" not in line or "hydrology" not in line:
        continue
    if "restart_streamlit" in line:
        continue
    try:
        pid = int(line.split(None, 1)[0])
    except ValueError:
        continue
    try:
        os.kill(pid, signal.SIGTERM)
        killed += 1
        print("killed", pid)
    except ProcessLookupError:
        pass
print("killed_count", killed)
time.sleep(2)

cmd = [
    str(ROOT / "venv" / "bin" / "streamlit"),
    "run",
    "hydrology/app/app.py",
    "--server.headless",
    "true",
    "--server.address",
    "0.0.0.0",
    "--server.port",
    "8501",
]
with open(LOG, "a", encoding="utf-8") as fh:
    fh.write("\n--- restart ---\n")
    subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
time.sleep(4)
try:
    with urllib.request.urlopen("http://127.0.0.1:8501/", timeout=6) as r:
        print("http", r.status)
except Exception as e:
    print("http_err", e)
    print(LOG.read_text(encoding="utf-8", errors="replace")[-800:])
