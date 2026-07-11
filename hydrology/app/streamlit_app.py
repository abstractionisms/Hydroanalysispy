"""
Hydrology Analysis Dashboard entrypoint (compat with existing process lines).

Preferred: streamlit run hydrology/app/app.py
Also works: streamlit run hydrology/app/streamlit_app.py
"""

import sys
from pathlib import Path

# Ensure the hydrology package is importable when launched as a script path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Execute the shared shell (single source of truth in app.py)
import hydrology.app.app  # noqa: F401,E402
