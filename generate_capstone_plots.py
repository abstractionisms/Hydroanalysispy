#!/usr/bin/env python3
"""
Generate plots for the Spokane River Dry Reach capstone project.

Gages (both from Oct 2017–present):
  12419000 - Spokane River near Post Falls, ID (upstream, dam-controlled by Avista)
  12422000 - Spokane River at Greene St, Spokane (downstream, aquifer-fed)

The Avista Window: Each year Avista reduces Post Falls to ~500 cfs minimum release
for a period in Aug–Sep. During this window, Greene St's minimum has been declining
year over year — evidence of declining aquifer contribution between the gages.

Usage:
    python generate_capstone_plots.py
    python generate_capstone_plots.py --output-dir ./capstone_plots
"""

import argparse
import sys
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd

from hydrology.data.usgs import fetch_discharge_data
from hydrology.data.climate import fetch_climate_data
from hydrology.visualization.plots import AVAILABLE_PLOTS


# --- Configuration ---

UPSTREAM_GAGE = '12419000'    # Post Falls
DOWNSTREAM_GAGE = '12422000'  # Greene Street

UPSTREAM_NAME = 'Post Falls (12419000)'
DOWNSTREAM_NAME = 'Greene St (12422000)'

# Both gages from Oct 2017 only (common overlap period)
START_DATE = '2017-10-01'

# Spokane area coordinates for Meteostat climate data
CLIMATE_LAT = 47.66
CLIMATE_LON = -117.43

# Plots to generate per individual gage (both gages)
SINGLE_GAGE_PLOTS = [
    'timeseries',
    'flow_duration',
    'baseflow_separation',
    'summer_low_flow_trend',
]

# Two-gage comparison plots
TWO_GAGE_PLOTS = [
    'reach_comparison',
    'paired_annual_lows',
    'avista_window_comparison',
    'seasonal_gain_loss',
    'seasonal_gain_loss_annual',
]

# Single-gage plot (Greene St only)
DOWNSTREAM_ONLY_PLOTS = [
    'threshold_exceedance',
]

# Plots that need climate data
CLIMATE_PLOTS = [
    'summer_climate_context',
]


def fetch_gage_data(site_id, name, start_date, end_date):
    """Fetch discharge data for a gage, keeping zeros for dry-reach analysis."""
    print(f"  Fetching {name}...")
    df = fetch_discharge_data(site_id, start_date=start_date, end_date=end_date,
                              positive_only=False)
    if df is None or df.empty:
        print(f"  WARNING: No data returned for {name}")
        return None
    print(f"  Got {len(df)} days: {df.index.min().date()} to {df.index.max().date()}")
    zero_count = (df['Discharge_cfs'] == 0).sum()
    if zero_count > 0:
        print(f"  Found {zero_count} zero-flow days")
    return df


def generate_single_gage_plots(df_q, gage_name, gage_id, output_dir):
    """Generate all single-gage plots."""
    generated = []
    for plot_key in SINGLE_GAGE_PLOTS:
        if plot_key not in AVAILABLE_PLOTS:
            print(f"  SKIP {plot_key}: not in AVAILABLE_PLOTS")
            continue

        plot_info = AVAILABLE_PLOTS[plot_key]
        func = plot_info['function']
        w, h = plot_info.get('default_size', (10, 6))

        print(f"  Generating {plot_key}...")
        try:
            fig, ax = plt.subplots(figsize=(w, h))
            func(ax, df_q=df_q, config={})
            current_title = ax.get_title()
            if current_title:
                ax.set_title(f"{gage_name}\n{current_title}")

            fig.tight_layout()
            fname = f"{gage_id}_{plot_key}.png"
            fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
            plt.close(fig)
            generated.append(fname)
        except Exception as e:
            print(f"  ERROR on {plot_key}: {e}")
            plt.close('all')

    return generated


def generate_two_gage_plots(df_upstream, df_downstream, output_dir, df_climate=None):
    """Generate two-gage comparison plots and climate-dependent plots."""
    generated = []

    # Standard two-gage plots
    for plot_key in TWO_GAGE_PLOTS:
        if plot_key not in AVAILABLE_PLOTS:
            print(f"  SKIP {plot_key}: not in AVAILABLE_PLOTS")
            continue

        plot_info = AVAILABLE_PLOTS[plot_key]
        func = plot_info['function']
        w, h = plot_info.get('default_size', (10, 6))

        print(f"  Generating {plot_key}...")
        try:
            fig, ax = plt.subplots(figsize=(w, h))
            func(ax, df_upstream=df_upstream, df_downstream=df_downstream, config={})
            current_title = ax.get_title()
            if current_title:
                ax.set_title(f"Spokane River: {UPSTREAM_NAME} → {DOWNSTREAM_NAME}\n{current_title}")

            fig.tight_layout()
            fname = f"comparison_{plot_key}.png"
            fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
            plt.close(fig)
            generated.append(fname)
        except Exception as e:
            print(f"  ERROR on {plot_key}: {e}")
            plt.close('all')

    # Downstream-only plots (Greene St)
    for plot_key in DOWNSTREAM_ONLY_PLOTS:
        if plot_key not in AVAILABLE_PLOTS:
            print(f"  SKIP {plot_key}: not in AVAILABLE_PLOTS")
            continue

        plot_info = AVAILABLE_PLOTS[plot_key]
        func = plot_info['function']
        w, h = plot_info.get('default_size', (10, 6))

        print(f"  Generating {plot_key} (Greene St)...")
        try:
            fig, ax = plt.subplots(figsize=(w, h))
            func(ax, df_q=df_downstream, config={})
            current_title = ax.get_title()
            if current_title:
                ax.set_title(f"{DOWNSTREAM_NAME}\n{current_title}")

            fig.tight_layout()
            fname = f"{DOWNSTREAM_GAGE}_{plot_key}.png"
            fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
            plt.close(fig)
            generated.append(fname)
        except Exception as e:
            print(f"  ERROR on {plot_key}: {e}")
            plt.close('all')

    # Climate-dependent plots
    if df_climate is not None:
        for plot_key in CLIMATE_PLOTS:
            if plot_key not in AVAILABLE_PLOTS:
                print(f"  SKIP {plot_key}: not in AVAILABLE_PLOTS")
                continue

            plot_info = AVAILABLE_PLOTS[plot_key]
            func = plot_info['function']
            w, h = plot_info.get('default_size', (10, 6))

            print(f"  Generating {plot_key} (with climate data)...")
            try:
                fig, ax = plt.subplots(figsize=(w, h))
                func(ax, df_upstream=df_upstream, df_downstream=df_downstream,
                     df_climate=df_climate, config={})
                current_title = ax.get_title()
                if current_title:
                    ax.set_title(f"Spokane River: {UPSTREAM_NAME} → {DOWNSTREAM_NAME}\n{current_title}")

                fig.tight_layout()
                fname = f"comparison_{plot_key}.png"
                fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
                plt.close(fig)
                generated.append(fname)
            except Exception as e:
                print(f"  ERROR on {plot_key}: {e}")
                plt.close('all')
    else:
        print("  SKIP climate plots: no climate data available")

    return generated


def main():
    parser = argparse.ArgumentParser(description='Generate Spokane River dry-reach capstone plots')
    parser.add_argument('--output-dir', type=str,
                        default=r'C:\Users\Cam\OneDrive - Spokane Colleges\Winter 26\NATR221\Capstone\Plots\focus plots',
                        help='Output directory for plots')
    parser.add_argument('--end', type=str, default='today',
                        help='End date (YYYY-MM-DD or "today")')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.resolve()}")

    end_date = args.end

    # Fetch data
    print("\n--- Fetching USGS data ---")
    df_upstream = fetch_gage_data(UPSTREAM_GAGE, UPSTREAM_NAME, START_DATE, end_date)
    df_downstream = fetch_gage_data(DOWNSTREAM_GAGE, DOWNSTREAM_NAME, START_DATE, end_date)

    # Fetch climate data (Meteostat)
    print("\n--- Fetching Meteostat climate data ---")
    try:
        start_ts = pd.Timestamp(START_DATE)
        end_ts = pd.Timestamp('today') if end_date == 'today' else pd.Timestamp(end_date)
        df_climate = fetch_climate_data(CLIMATE_LAT, CLIMATE_LON, start_ts, end_ts,
                                        include_temp=True, include_precip=True)
        if df_climate is not None:
            print(f"  Got {len(df_climate)} days of climate data")
        else:
            print("  WARNING: No climate data returned")
    except Exception as e:
        print(f"  ERROR fetching climate data: {e}")
        df_climate = None

    all_generated = []

    # Generate per-gage plots (both gages)
    if df_upstream is not None:
        print(f"\n--- Upstream gage: {UPSTREAM_NAME} ---")
        files = generate_single_gage_plots(df_upstream, UPSTREAM_NAME, UPSTREAM_GAGE, output_dir)
        all_generated.extend(files)

    if df_downstream is not None:
        print(f"\n--- Downstream gage: {DOWNSTREAM_NAME} ---")
        files = generate_single_gage_plots(df_downstream, DOWNSTREAM_NAME, DOWNSTREAM_GAGE, output_dir)
        all_generated.extend(files)

    # Generate two-gage comparison and capstone plots
    if df_upstream is not None and df_downstream is not None:
        print(f"\n--- Two-gage comparisons & capstone plots ---")
        files = generate_two_gage_plots(df_upstream, df_downstream, output_dir,
                                        df_climate=df_climate)
        all_generated.extend(files)
    else:
        print("\nSKIPPING two-gage comparisons (missing data)")

    # Summary
    print(f"\n{'='*60}")
    print(f"Generated {len(all_generated)} plots in {output_dir.resolve()}")
    for f in sorted(all_generated):
        print(f"  {f}")

    if not all_generated:
        print("WARNING: No plots generated! Check data availability.")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
