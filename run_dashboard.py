#!/usr/bin/env python
"""
Launch the Hydrology Analysis Dashboard.

Usage:
    python run_dashboard.py           # Local only (localhost:8501)
    python run_dashboard.py --network # Allow network access (0.0.0.0:8501)

Or run streamlit directly:
    streamlit run hydrology/app/streamlit_app.py
"""

import subprocess
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Launch Hydrology Dashboard')
    parser.add_argument('--network', action='store_true',
                       help='Allow network access (not just localhost)')
    parser.add_argument('--port', type=int, default=8501,
                       help='Port to run on (default: 8501)')
    args = parser.parse_args()

    # New multipage app entry point (old: streamlit_app.py still works as legacy)
    app_path = Path(__file__).parent / "hydrology" / "app" / "app.py"

    if not app_path.exists():
        print(f"Error: Could not find {app_path}")
        sys.exit(1)

    print("Starting Hydrology Analysis Dashboard...")
    print(f"App location: {app_path}")

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.headless", "true",
        "--server.port", str(args.port)
    ]

    if args.network:
        cmd.extend(["--server.address", "0.0.0.0"])
        print(f"\nNetwork access enabled!")
        print(f"Local:   http://localhost:{args.port}")
        print(f"Network: http://<your-ip>:{args.port}\n")
    else:
        print(f"\nOpening in browser at http://localhost:{args.port}")
        print("(Use --network flag to allow access from other devices)\n")

    subprocess.run(cmd)


if __name__ == "__main__":
    main()
