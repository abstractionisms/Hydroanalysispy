#!/usr/bin/env python3
import os
import re
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path("/home/cam/source/repos/Hydrology")
LOG = ROOT / "outputs" / "logs" / "streamlit.log"
PORT = 8501


def kill_port(port: int) -> None:
    try:
        out = subprocess.check_output(
            ["ss", "-lptn", f"sport = :{port}"], text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        out = ""
    for pid in re.findall(r"pid=(\d+)", out):
        try:
            os.kill(int(pid), signal.SIGKILL)
            print("killed port holder", pid)
        except Exception as e:
            print("port kill fail", pid, e)


def kill_streamlit() -> None:
    out = subprocess.check_output(["ps", "-eo", "pid,cmd"], text=True)
    for line in out.splitlines():
        if "streamlit" not in line or "hydrology" not in line:
            continue
        if "force_restart" in line or "restart_streamlit" in line:
            continue
        try:
            pid = int(line.split(None, 1)[0])
            os.kill(pid, signal.SIGKILL)
            print("killed streamlit", pid)
        except Exception as e:
            print("streamlit kill fail", e)


kill_port(PORT)
kill_streamlit()
time.sleep(2)

LOG.parent.mkdir(parents=True, exist_ok=True)
cmd = [
    str(ROOT / "venv" / "bin" / "streamlit"),
    "run",
    "hydrology/app/app.py",
    "--server.headless",
    "true",
    "--server.address",
    "0.0.0.0",
    "--server.port",
    str(PORT),
]
with open(LOG, "a", encoding="utf-8") as fh:
    fh.write("\n--- force restart ---\n")
    subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

time.sleep(5)
try:
    print("http", urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=8).status)
except Exception as e:
    print("http_err", e)
    print(LOG.read_text(encoding="utf-8", errors="replace")[-600:])
