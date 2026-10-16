"""Cloud entrypoint for HydroPlot.

Streamlit re-executes this file on every session, but a normal import runs
only once per process. Later sessions would then render a blank page.
Re-executing app.py keeps every session on the same dashboard as a direct
`streamlit run hydrology/app/app.py`.
"""

import sys
from pathlib import Path
import runpy

# Ensure the hydrology package is importable when launched as a script path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

runpy.run_path(str(Path(__file__).with_name("app.py")), run_name="__main__")
