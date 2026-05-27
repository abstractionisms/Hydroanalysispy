"""
Individual plot functions for hydrology analysis.

Each plot is self-contained and reusable. Plots take data and an axis,
render to that axis, and return nothing. This makes them composable
into any layout you want.

Available plots (27 total):
- anomaly: Recent decade vs historical monthly averages (Q, T, P)
- hexbin_temp: Discharge vs Temperature hexbin (log scale with counts)
- lagged_precip: Monthly avg discharge vs lagged precipitation
- timeseries: Recent discharge time series (log scale)
- flow_duration: Flow duration curve
- correlation_matrix: Correlation heatmap
- monthly_boxplot: Monthly discharge distribution boxplots
- discharge_heatmap: Discharge density vs day of year
- temporal_heatmap: 4-panel temporal discharge density
- precip_discharge: Discharge with precipitation overlay
- seasonal_scatter: Q vs T colored by season
- low_flow_trend: Annual 7-day low flow trend
- annual_trend: Annual mean discharge trend
- baseflow_separation: Lyne-Hollick baseflow filter
- recession_curves: Recession curve analysis with K estimation
- double_mass_curve: Cumulative P vs Q for consistency check
- rating_curve: Stage-discharge power-law fit
- flood_frequency: Log-Pearson Type III flood frequency
- 7q10_analysis: 7Q10 low flow frequency analysis
- lag_correlation: P-Q lag correlation (0-30 days)
- anomaly_detection: Rolling mean +/- 2 sigma outliers
- cumulative_departure: Cumulative departure from mean
- spectral_analysis: FFT-based periodicity detection
- reach_comparison: Reach comparison (gaining/losing shading + trend)
- summer_low_flow_trend: Summer 7-day minimum flow trend
- reach_index: Aquifer contribution index
- paired_annual_lows: Paired 7-day lows during low-flow windows
- avista_window_comparison: Downstream hydrographs overlaid by year
- threshold_exceedance: Days below critical flow thresholds
- precip_response_comparison: Precipitation pulse propagation ratio

Usage:
    from hydrology.visualization.plots import plot_anomaly

    fig, ax = plt.subplots()
    plot_anomaly(ax, df_merged, config={})
    plt.show()
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import calendar
from scipy import stats
from typing import Dict, Any, Optional

from ..core.logging_setup import get_logger

logger = get_logger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    'discharge_col': 'Discharge_cfs',
    'temp_col': 'Temp_C',
    'precip_col': 'Precip_mm',
    'plot_years': 3,
    'recent_decade_years': 10,
    'hexbin_gridsize': 30,
    'n_scatter_annotations': 10,
    'p_significance_level': 0.05,
    'epsilon': 1e-10,  # Small value for division safety
}

# Plot labels
PLOT_LABELS = {
    'Discharge_cfs': 'Discharge (cfs)',
    'Temp_C': 'Avg Temp (°C)',
    'Precip_mm': 'Precipitation (mm)',
    'Precip_mm_lag1': 'Previous Day Precip (mm)'
}


def _plot_placeholder(ax, message="Plot N/A"):
    """Helper to display placeholder message."""
    ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=10,
            transform=ax.transAxes, wrap=True,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.5))
    ax.set_xticks([])
    ax.set_yticks([])


def _get_significance_stars(pval, levels=[0.001, 0.01, 0.05]):
    """Get significance stars based on p-value."""
    if pval is None or pd.isna(pval): return ""
    if pval < levels[0]: return "***"
    if pval < levels[1]: return "**"
    if pval < levels[2]: return "*"
    return ""


# ============================================================================
# PLOT 1: ANOMALY (Recent Decade vs Historical Monthly Averages)
# ============================================================================

def plot_anomaly(ax, df_merged: pd.DataFrame = None, config: Dict[str, Any] = None, **kwargs):
    """
    Plot recent decade vs historical monthly averages (Q, T, P).

    Args:
        ax: Matplotlib axis to plot on
        df_merged: DataFrame with Discharge_cfs, Temp_C, Precip_mm columns
        config: Optional configuration dict
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    DISCHARGE_COL = cfg['discharge_col']
    TEMP_COL = cfg['temp_col']
    PRECIP_COL = cfg['precip_col']
    RECENT_DECADE_YEARS = cfg['recent_decade_years']

    plot_title = "Recent Decade vs Historical Monthly Averages"
    required_cols = [DISCHARGE_COL, TEMP_COL, PRECIP_COL]

    if df_merged is None or df_merged.empty or not all(col in df_merged.columns for col in required_cols):
        _plot_placeholder(ax, f"{plot_title}\nData N/A")
        return

    try:
        df_plot = df_merged[required_cols].copy()
        df_plot.index = pd.to_datetime(df_plot.index)
        df_plot['Month'] = df_plot.index.month
        df_plot['Year'] = df_plot.index.year

        hist_start_year = df_plot['Year'].min()
        hist_end_year = df_plot['Year'].max()
        plot_title += f" (Hist: {hist_start_year}-{hist_end_year})"

        # Long-term stats
        monthly_stats_overall = df_plot.groupby('Month').agg(
            q_mean_overall=(DISCHARGE_COL, 'mean'),
            q_std_overall=(DISCHARGE_COL, 'std'),
            t_mean_overall=(TEMP_COL, 'mean'),
            t_std_overall=(TEMP_COL, 'std'),
            p_mean_overall=(PRECIP_COL, 'mean'),
            p_std_overall=(PRECIP_COL, 'std')
        ).reset_index()

        # Recent decade stats
        decade_start_year = hist_end_year - RECENT_DECADE_YEARS + 1
        df_decade = df_plot[df_plot['Year'] >= decade_start_year]
        if not df_decade.empty:
             monthly_stats_decade = df_decade.groupby('Month').agg(
                 q_mean_decade=(DISCHARGE_COL, 'mean'),
                 t_mean_decade=(TEMP_COL, 'mean'),
                 p_mean_decade=(PRECIP_COL, 'mean')
             ).reset_index()
        else:
             monthly_stats_decade = pd.DataFrame(columns=['Month', 'q_mean_decade', 't_mean_decade', 'p_mean_decade'])

        decade_label = f"{decade_start_year}-{hist_end_year} Avg"

        # Ensure all 12 months
        all_months = pd.DataFrame({'Month': range(1, 13)})
        monthly_stats_overall = pd.merge(all_months, monthly_stats_overall, on='Month', how='left')
        monthly_stats_decade = pd.merge(all_months, monthly_stats_decade, on='Month', how='left')
        monthly_stats = pd.merge(monthly_stats_overall, monthly_stats_decade, on='Month', how='left')
        monthly_stats = monthly_stats.set_index('Month')

        # Plotting
        months_num = range(1, 13)
        month_labels = [calendar.month_abbr[i] for i in months_num]

        # Discharge axis
        color_q = 'tab:blue'
        ax.set_xlabel("Month")
        ax.set_ylabel(f"Avg {PLOT_LABELS[DISCHARGE_COL]}", color=color_q)
        ln1 = ax.plot(months_num, monthly_stats['q_mean_overall'], color=color_q, linestyle='--', label='Overall Avg Q')
        q_std_lower = (monthly_stats['q_mean_overall'] - monthly_stats['q_std_overall']).clip(lower=0)
        q_std_upper = monthly_stats['q_mean_overall'] + monthly_stats['q_std_overall']
        ax.fill_between(months_num, q_std_lower, q_std_upper, color=color_q, alpha=0.15)
        ln2 = ax.plot(months_num, monthly_stats['q_mean_decade'], color=color_q, linestyle=':', marker='.', markersize=4, label=f'{decade_label} Q')
        ax.tick_params(axis='y', labelcolor=color_q)
        ax.set_xticks(months_num)
        ax.set_xticklabels(month_labels)
        ax.grid(True, axis='y', linestyle=':', alpha=0.5)
        ax.set_ylim(bottom=0)

        # Temp & Precip axis
        ax2 = ax.twinx()
        color_t = 'tab:red'
        color_p = 'tab:green'

        ax2.set_ylabel(f"Avg {PLOT_LABELS[TEMP_COL]} (red) / {PLOT_LABELS[PRECIP_COL]} (green)", color='black')
        ln4 = ax2.plot(months_num, monthly_stats['t_mean_overall'], color=color_t, linestyle='--', label='Overall Avg T')
        ax2.fill_between(months_num,
                          monthly_stats['t_mean_overall'] - monthly_stats['t_std_overall'],
                          monthly_stats['t_mean_overall'] + monthly_stats['t_std_overall'],
                          color=color_t, alpha=0.15)
        ln5 = ax2.plot(months_num, monthly_stats['t_mean_decade'], color=color_t, linestyle=':', marker='.', markersize=4, label=f'{decade_label} T')

        ln7 = ax2.plot(months_num, monthly_stats['p_mean_overall'], color='black', linestyle='--', label='Overall Avg P')
        p_std_lower = (monthly_stats['p_mean_overall'] - monthly_stats['p_std_overall']).clip(lower=0)
        p_std_upper = monthly_stats['p_mean_overall'] + monthly_stats['p_std_overall']
        ax2.fill_between(months_num, p_std_lower, p_std_upper, color=color_p, alpha=0.15)
        ln8 = ax2.plot(months_num, monthly_stats['p_mean_decade'], color='black', linestyle=':', marker='x', markersize=4, label=f'{decade_label} P')

        ax2.tick_params(axis='y', labelcolor='black')

        ax.set_title(plot_title)

        # Legend
        handles = ln1 + ln2 + ln4 + ln5 + ln7 + ln8
        labels = [h.get_label() for h in handles]
        handles.extend([
            plt.Rectangle((0, 0), 1, 1, fc=color_q, alpha=0.15),
            plt.Rectangle((0, 0), 1, 1, fc=color_t, alpha=0.15),
            plt.Rectangle((0, 0), 1, 1, fc=color_p, alpha=0.15)
        ])
        labels.extend(['Overall Q ± 1σ', 'Overall T ± 1σ', 'Overall P ± 1σ'])
        ax.legend(handles, labels, loc='best', fontsize='xx-small', ncol=2)

    except Exception as e:
        logger.error(f"Anomaly plot error: {e}")
        _plot_placeholder(ax, f"Error plotting\n{plot_title}")


# ============================================================================
# PLOT 2: HEXBIN (Discharge vs Temperature)
# ============================================================================

def plot_hexbin(ax, df_merged: pd.DataFrame = None, analysis_results: Dict = None,
                x_col: str = 'Temp_C', y_col: str = 'Discharge_cfs',
                use_log_scale: bool = True, add_counts: bool = True,
                config: Dict[str, Any] = None, **kwargs):
    """
    Plot hexbin (discharge vs temperature by default).

    Args:
        ax: Matplotlib axis
        df_merged: Merged data
        analysis_results: Dict with correlation results
        x_col: X-axis column name
        y_col: Y-axis column name
        use_log_scale: Use log scale for color intensity
        add_counts: Add count text to hexagons
        config: Optional configuration
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    y_label = PLOT_LABELS.get(y_col, y_col)
    x_label = PLOT_LABELS.get(x_col, x_col)
    scale_label = "(Log Scale)" if use_log_scale else "(Linear Scale)"
    plot_title = f"Daily {y_label} vs. {x_label} (Density {scale_label})"

    if df_merged is None or df_merged.empty or x_col not in df_merged.columns or y_col not in df_merged.columns:
        _plot_placeholder(ax, f"{plot_title}\nData N/A")
        return

    try:
        df_plot = df_merged[[x_col, y_col]].dropna()

        if not df_plot.empty:
            hexbin_args = {
                'gridsize': cfg['hexbin_gridsize'],
                'cmap': 'viridis',
                'mincnt': 1,
                'alpha': 0.9
            }
            if use_log_scale:
                hexbin_args['bins'] = 'log'

            hb = ax.hexbin(df_plot[x_col], df_plot[y_col], **hexbin_args)
            ax.set_title(plot_title)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.grid(True, linestyle='--', alpha=0.4)

            cb = plt.colorbar(hb, ax=ax)
            cb.set_label('Log(Count in Bin)' if use_log_scale else 'Count in Bin')

            if add_counts:
                counts = hb.get_array()
                centers = hb.get_offsets()
                for count, center in zip(counts, centers):
                    if count > 0:
                        ax.text(center[0], center[1], f'{int(count)}',
                                ha='center', va='center', color='white', fontsize=5)

            # Add correlation info
            if analysis_results and analysis_results.get('corr_matrix') is not None:
                corr_matrix = analysis_results['corr_matrix']
                p_values = analysis_results.get('p_values', {})
                if x_col in corr_matrix.columns and y_col in corr_matrix.index:
                    r_val = corr_matrix.loc[y_col, x_col]
                    pair_key = tuple(sorted((x_col, y_col)))
                    p_val = p_values.get(pair_key, np.nan)
                    stars = _get_significance_stars(p_val)
                    corr_info = f"Daily R = {r_val:.2f}{stars}"
                    ax.text(0.02, 0.98, corr_info, transform=ax.transAxes, fontsize=9,
                            ha='left', va='top', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
        else:
            _plot_placeholder(ax, f"{plot_title}\nNo Overlapping Data")

    except Exception as e:
        logger.error(f"Hexbin plot error: {e}")
        _plot_placeholder(ax, f"Error plotting\n{plot_title}")


# ============================================================================
# PLOT 3: MONTHLY LAGGED SCATTER (Q vs Lagged Precip)
# ============================================================================

def plot_monthly_lagged_scatter(ax, df_merged: pd.DataFrame = None, analysis_results: Dict = None,
                                config: Dict[str, Any] = None, **kwargs):
    """
    Plot monthly average discharge vs lagged precipitation.

    Args:
        ax: Matplotlib axis
        df_merged: Merged data
        analysis_results: Dict with lagged correlation results
        config: Optional configuration
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    DISCHARGE_COL = cfg['discharge_col']
    PRECIP_COL = cfg['precip_col']
    N_ANNOTATIONS = cfg['n_scatter_annotations']

    lagged_precip_col = f'{PRECIP_COL}_lag1'
    plot_title = f"Monthly Avg {PLOT_LABELS[DISCHARGE_COL]} vs. Monthly Avg Previous Day Precip"

    if df_merged is None or df_merged.empty:
         _plot_placeholder(ax, f"{plot_title}\nData N/A")
         return

    try:
        df_plot = df_merged.copy()
        if not all(c in df_plot.columns for c in [DISCHARGE_COL, PRECIP_COL]):
             _plot_placeholder(ax, f"{plot_title}\nRequired columns missing")
             return

        df_plot[lagged_precip_col] = df_plot[PRECIP_COL].shift(1)
        df_plot = df_plot.dropna(subset=[DISCHARGE_COL, lagged_precip_col])

        if not df_plot.empty:
            df_plot.index = pd.to_datetime(df_plot.index)
            monthly_avg = df_plot[[DISCHARGE_COL, lagged_precip_col]].resample('ME').mean().dropna()
            monthly_avg['Month'] = monthly_avg.index.month

            if not monthly_avg.empty:
                sns.scatterplot(data=monthly_avg, x=lagged_precip_col, y=DISCHARGE_COL,
                                hue='Month', palette='viridis', s=50, ax=ax, legend='full')

                ax.set_title(plot_title)
                ax.set_xlabel(f"Avg Previous Day Precip (mm)")
                ax.set_ylabel(f"Avg {PLOT_LABELS[DISCHARGE_COL]}")
                ax.grid(True, linestyle='--', alpha=0.6)

                handles, labels = ax.get_legend_handles_labels()
                month_names = [calendar.month_abbr[int(label)] for label in labels[1:]]
                ax.legend(handles=handles[1:], labels=month_names, title='Month', fontsize='small', ncol=2)

                # Annotate top N
                top_n = monthly_avg.nlargest(N_ANNOTATIONS, DISCHARGE_COL)
                for idx, row in top_n.iterrows():
                    ax.annotate(f'{idx.strftime("%Y-%m")}',
                                xy=(row[lagged_precip_col], row[DISCHARGE_COL]),
                                xytext=(5, -5), textcoords='offset points',
                                ha='left', va='top', fontsize=7, color='black',
                                bbox=dict(boxstyle='round,pad=0.1', fc='yellow', alpha=0.5, ec='none'))

                # Add lag correlation info
                if analysis_results:
                    lag_corr = analysis_results.get('lagged_precip_corr')
                    lag_p = analysis_results.get('lagged_precip_p')
                    if lag_corr is not None and lag_p is not None:
                        lag_info = f"Daily Lag-1 Corr: R={lag_corr:.2f} (p={lag_p:.3f})"
                        ax.text(0.02, 0.98, lag_info, transform=ax.transAxes, fontsize=9,
                                ha='left', va='top', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
            else:
                 _plot_placeholder(ax, f"{plot_title}\nNo Monthly Data")
        else:
            _plot_placeholder(ax, f"{plot_title}\nNo Lagged Data")

    except Exception as e:
        logger.error(f"Monthly lagged scatter plot error: {e}")
        _plot_placeholder(ax, f"Error plotting\n{plot_title}")


# ============================================================================
# PLOT 4: TIMESERIES (Recent Discharge with Log Scale)
# ============================================================================

def plot_timeseries(ax, df_q: pd.DataFrame = None, config: Dict[str, Any] = None, **kwargs):
    """
    Plot recent discharge time series with log scale.

    Args:
        ax: Matplotlib axis
        df_q: Discharge DataFrame
        config: Optional configuration
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    DISCHARGE_COL = cfg['discharge_col']
    PLOT_YEARS = cfg['plot_years']

    if df_q is None or df_q.empty:
         _plot_placeholder(ax, "Time Series Plot N/A (Discharge Missing)")
         return

    try:
        import matplotlib.dates as mdates

        end_plot_date = df_q.index.max()
        start_plot_date = end_plot_date - pd.DateOffset(years=PLOT_YEARS)
        df_plot = df_q.loc[start_plot_date:end_plot_date].copy()
        df_plot = df_plot[df_plot[DISCHARGE_COL] > 0]

        if not df_plot.empty:
            color1 = 'tab:blue'
            ax.set_xlabel('Date')
            ax.set_ylabel(f"{PLOT_LABELS[DISCHARGE_COL]} (Log Scale)", color=color1)
            ax.plot(df_plot.index, df_plot[DISCHARGE_COL], color=color1, linewidth=1.5, label='Discharge')
            ax.tick_params(axis='y', labelcolor=color1)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_yscale('log')
            ax.set_title(f"Recent Discharge - {PLOT_YEARS} Years (Log Scale)")

            # Yearly min/max annotations
            q_series = df_plot[DISCHARGE_COL]
            for year in q_series.index.year.unique():
                q_year = q_series[q_series.index.year == year]
                if not q_year.empty:
                    try:
                        idx_max, val_max = q_year.idxmax(), q_year.max()
                        idx_min, val_min = q_year.idxmin(), q_year.min()

                        ax.annotate(f'{year} Max: {val_max:.0f}',
                                    xy=(idx_max, val_max), xytext=(0, 10), textcoords='offset points',
                                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
                                    ha='center', va='bottom', fontsize=7, color=color1,
                                    bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.6, ec='none'))

                        ax.annotate(f'{year} Min: {val_min:.0f}',
                                    xy=(idx_min, val_min), xytext=(0, 15), textcoords='offset points',
                                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-.2"),
                                    ha='center', va='bottom', fontsize=7, color=color1,
                                    bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.6, ec='none'))
                    except (ValueError, IndexError, TypeError):
                        pass

            # Last point
            if not q_series.empty:
                idx_last, val_last = q_series.index[-1], q_series.iloc[-1]
                ax.annotate(f'Last: {val_last:.0f}\n{idx_last.strftime("%Y-%m-%d")}',
                            xy=(idx_last, val_last), xytext=(-10, 20), textcoords='offset points',
                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
                            ha='right', va='bottom', fontsize=8, color=color1,
                            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6, ec='none'))

            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        else:
            _plot_placeholder(ax, "No Positive Recent Discharge for Log Scale")

    except Exception as e:
        logger.error(f"Time series plot error: {e}")
        _plot_placeholder(ax, "Error plotting Time Series")


# ============================================================================
# PLOT 5: FLOW DURATION CURVE
# ============================================================================

def plot_flow_duration(ax, df_q: pd.DataFrame = None, config: Dict[str, Any] = None, **kwargs):
    """
    Plot flow duration curve.

    Args:
        ax: Matplotlib axis
        df_q: Discharge DataFrame
        config: Optional configuration
    """
    from ..analysis.stage_discharge import flow_duration_curve

    cfg = {**DEFAULT_CONFIG, **(config or {})}
    DISCHARGE_COL = cfg['discharge_col']

    if df_q is None or df_q.empty or DISCHARGE_COL not in df_q.columns:
        _plot_placeholder(ax, "Flow Duration Curve N/A")
        return

    try:
        fdc = flow_duration_curve(df_q[DISCHARGE_COL])

        if not fdc.empty:
            ax.semilogy(fdc['exceedance_pct'], fdc['discharge'], linewidth=2, color='tab:blue')
            ax.set_xlabel('Exceedance Probability (%)')
            ax.set_ylabel(f'{PLOT_LABELS[DISCHARGE_COL]} (Log Scale)')
            ax.set_title('Flow Duration Curve')
            ax.grid(True, alpha=0.3, which='both')
            ax.set_xlim(0, 100)

            # Add percentile markers
            for pct in [1, 10, 50, 90, 99]:
                q_val = fdc[fdc['exceedance_pct'] == pct]['discharge'].values
                if len(q_val) > 0:
                    ax.axvline(pct, color='gray', linestyle='--', alpha=0.3)
                    ax.text(pct, q_val[0], f'Q{pct}', fontsize=8, ha='center', va='bottom')
        else:
            _plot_placeholder(ax, "No Data for Flow Duration Curve")

    except Exception as e:
        logger.error(f"Flow duration curve error: {e}")
        _plot_placeholder(ax, "Error plotting Flow Duration Curve")


# ============================================================================
# PLOT 6: CORRELATION MATRIX HEATMAP
# ============================================================================

def plot_correlation_matrix(ax, df_merged: pd.DataFrame = None, analysis_results: Dict = None,
                            config: Dict[str, Any] = None, **kwargs):
    """
    Plot correlation matrix heatmap.

    Args:
        ax: Matplotlib axis
        df_merged: Merged data
        analysis_results: Dict with correlation results
        config: Optional configuration
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    if analysis_results is None or analysis_results.get('corr_matrix') is None:
        _plot_placeholder(ax, "Correlation Matrix N/A")
        return

    try:
        corr_matrix = analysis_results['corr_matrix']
        p_values = analysis_results.get('p_values', {})

        # Create annotations with significance stars
        annot = corr_matrix.copy().astype(str)
        for i, row_name in enumerate(corr_matrix.index):
            for j, col_name in enumerate(corr_matrix.columns):
                if i != j:
                    pair_key = tuple(sorted((row_name, col_name)))
                    p_val = p_values.get(pair_key, np.nan)
                    stars = _get_significance_stars(p_val)
                    val = corr_matrix.loc[row_name, col_name]
                    annot.loc[row_name, col_name] = f"{val:.2f}{stars}"
                else:
                    annot.loc[row_name, col_name] = "1.00"

        sns.heatmap(corr_matrix, annot=annot, fmt='', cmap='cividis', center=0,
                    vmin=-1, vmax=1, ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_title('Correlation Matrix (* p<0.05, ** p<0.01, *** p<0.001)')

    except Exception as e:
        logger.error(f"Correlation matrix plot error: {e}")
        _plot_placeholder(ax, "Error plotting Correlation Matrix")


# ============================================================================
# PLOT 7: MONTHLY BOXPLOT
# ============================================================================

def plot_monthly_boxplot(ax, df_q: pd.DataFrame = None, config: Dict[str, Any] = None, **kwargs):
    """
    Plot monthly discharge distribution boxplots (log scale).

    Args:
        ax: Matplotlib axis
        df_q: Discharge DataFrame with datetime index
        config: Optional configuration
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    if df_q is None or df_q.empty:
        _plot_placeholder(ax, "Monthly Boxplots\nN/A - No discharge data")
        return

    try:

        # Prepare data - filter zeros before log scale
        df_copy = df_q[df_q['Discharge_cfs'] > 0].copy()
        if df_copy.empty:
            _plot_placeholder(ax, "Monthly Boxplots\nNo positive discharge values")
            return
        df_copy['Month'] = df_copy.index.month

        # Create boxplot
        month_order = range(1, 13)
        month_labels = [calendar.month_abbr[i] for i in month_order]

        sns.boxplot(
            x='Month',
            y='Discharge_cfs',
            data=df_copy,
            ax=ax,
            order=month_order,
            showfliers=False,
            palette='viridis'
        )

        # Formatting
        data_range = f"{df_q.index.min().year} - {df_q.index.max().year}"
        ax.set_title(f'Monthly Discharge Distribution\nData: {data_range}')
        ax.set_xlabel('Month')
        ax.set_ylabel('Discharge (cfs) [Log Scale]')
        ax.set_xticklabels(month_labels)
        ax.set_yscale('log')
        ax.grid(True, axis='y', which='both', linestyle='--', alpha=0.6)
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.get_major_formatter().set_scientific(False)

    except Exception as e:
        logger.error(f"Monthly boxplot error: {e}")
        _plot_placeholder(ax, "Error plotting Monthly Boxplots")


# ============================================================================
# PLOT 8: DISCHARGE DENSITY HEATMAP (Single Panel)
# ============================================================================

def plot_discharge_heatmap(ax, df_q: pd.DataFrame = None, config: Dict[str, Any] = None, **kwargs):
    """
    Plot discharge density heatmap vs day of year (single panel).

    Args:
        ax: Matplotlib axis
        df_q: Discharge DataFrame
        config: Optional configuration
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    if df_q is None or df_q.empty:
        _plot_placeholder(ax, "Discharge Heatmap\nN/A - No discharge data")
        return

    try:
        # Filter zeros before log scale
        df_pos = df_q[df_q['Discharge_cfs'] > 0]
        if df_pos.empty:
            _plot_placeholder(ax, "Discharge Heatmap\nNo positive discharge values")
            return
        discharge_vals = df_pos['Discharge_cfs'].values
        day_of_year = df_pos.index.dayofyear.values

        # Calculate log bins
        min_q = np.floor(np.log10(discharge_vals.min()))
        max_q = np.ceil(np.log10(discharge_vals.max()))
        log_bins_q = np.logspace(min_q, max_q, 75)

        # Create 2D histogram
        counts, _, _, im = ax.hist2d(
            day_of_year,
            discharge_vals,
            bins=[75, log_bins_q],
            cmap='viridis',
            cmin=1
        )

        # Formatting
        data_range = f"{df_q.index.min().year} - {df_q.index.max().year}"
        ax.set_yscale('log')
        ax.set_title(f'Discharge Density Heatmap\nData: {data_range}')
        ax.set_xlabel('Day of Year')
        ax.set_ylabel('Discharge (cfs) [Log Scale]')
        ax.set_xlim(1, 366)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Number of Days')

        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.get_major_formatter().set_scientific(False)

    except Exception as e:
        logger.error(f"Discharge heatmap error: {e}")
        _plot_placeholder(ax, "Error plotting Discharge Heatmap")


# ============================================================================
# PLOT 9: TEMPORAL DISCHARGE HEATMAP (4-Panel)
# ============================================================================

def plot_temporal_heatmap(ax, df_q: pd.DataFrame = None, config: Dict[str, Any] = None, **kwargs):
    """
    Plot 4-panel temporal discharge density heatmap.

    Shows discharge density for: Last 5 years, Last 10 years, Last 20 years, Total record.
    This function expects ax to be a single axis but creates subplots internally.

    NOTE: This plot works differently - it needs to create its own figure.
    When used in composer, it will appear in one subplot but contain 4 mini-panels.

    Args:
        ax: Matplotlib axis (will be replaced with 2x2 subplots)
        df_q: Discharge DataFrame
        config: Optional configuration
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    if df_q is None or df_q.empty:
        _plot_placeholder(ax, "Temporal Heatmap\nN/A - No discharge data")
        return

    try:
        # Clear the provided axis and use its figure
        fig = ax.get_figure()
        ax.set_visible(False)  # Hide the original axis

        # Get position of original axis
        pos = ax.get_position()

        # Create 2x2 subplots within the original axis space
        # This is a bit hacky but allows it to work within the composer
        gs = fig.add_gridspec(2, 2, left=pos.x0, right=pos.x1,
                             bottom=pos.y0, top=pos.y1,
                             hspace=0.3, wspace=0.3)
        axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

        # Define time windows
        scenarios = [
            ("Last 5 Years", 5),
            ("Last 10 Years", 10),
            ("Last 20 Years", 20),
            ("Total Record", None)
        ]

        last_date = df_q.index.max()

        # Filter zeros before log scale
        df_q = df_q[df_q['Discharge_cfs'] > 0]
        if df_q.empty:
            _plot_placeholder(ax, "Temporal Heatmap\nNo positive discharge values")
            return

        # Global min/max for consistent Y-axis
        global_min_q = df_q['Discharge_cfs'].min()
        global_max_q = df_q['Discharge_cfs'].max()

        # Calculate log bins
        log_min = np.floor(np.log10(global_min_q))
        log_max = np.ceil(np.log10(global_max_q))
        log_bins = np.logspace(log_min, log_max, 60)

        for i, (label, years) in enumerate(scenarios):
            ax_sub = axes[i]

            # Slice data
            if years:
                start_date = last_date - pd.DateOffset(years=years)
                subset = df_q.loc[start_date:last_date]
                date_range_str = f"({start_date.year} - {last_date.year})"
            else:
                subset = df_q
                start_year = df_q.index.min().year
                date_range_str = f"({start_year} - {last_date.year})"

            if subset.empty:
                ax_sub.text(0.5, 0.5, "No Data", ha='center', va='center')
                continue

            # Prepare data
            q_vals = subset['Discharge_cfs'].values
            doy = subset.index.dayofyear.values

            # Plot 2D histogram
            counts, _, _, im = ax_sub.hist2d(
                doy, q_vals,
                bins=[60, log_bins],
                cmap='viridis',
                cmin=1
            )

            # Formatting
            ax_sub.set_title(f"{label} {date_range_str}", fontsize=10, fontweight='bold')
            ax_sub.set_yscale('log')
            ax_sub.set_xlim(1, 366)
            ax_sub.set_ylim(global_min_q, global_max_q * 1.5)

            if i >= 2:  # Bottom plots
                ax_sub.set_xlabel("Day of Year", fontsize=9)
            if i % 2 == 0:  # Left plots
                ax_sub.set_ylabel("Discharge (cfs) [Log]", fontsize=9)

            ax_sub.grid(True, which='both', linestyle='--', alpha=0.3)
            ax_sub.yaxis.set_major_formatter(mticker.ScalarFormatter())
            ax_sub.yaxis.get_major_formatter().set_scientific(False)
            ax_sub.tick_params(labelsize=8)

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax_sub)
            cbar.set_label('Days Count', fontsize=8)
            cbar.ax.tick_params(labelsize=7)

    except Exception as e:
        logger.error(f"Temporal heatmap error: {e}")
        _plot_placeholder(ax, "Error plotting Temporal Heatmap")


# ============================================================================
# PLOT 10: PRECIPITATION-DISCHARGE OVERLAY
# ============================================================================

def plot_precip_discharge_overlay(ax, df_merged: pd.DataFrame = None, df_q: pd.DataFrame = None, config: Dict[str, Any] = None, **kwargs):
    """
    Plot discharge with precipitation overlay (dual y-axis) and cumulative precipitation.

    Shows:
    - Discharge (left y-axis, blue line)
    - Daily precipitation (right y-axis, blue bars) - if available
    - Cumulative precipitation (right y-axis, orange line) - if available

    Falls back to discharge-only plot if precipitation data is unavailable.

    Args:
        ax: Matplotlib axis
        df_merged: Merged DataFrame with Discharge_cfs and Precip_mm columns
        df_q: Discharge-only DataFrame (fallback if df_merged unavailable)
        config: Optional configuration
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    # Determine which data source to use
    has_merged = df_merged is not None and not df_merged.empty
    has_precip = has_merged and 'Precip_mm' in df_merged.columns
    has_discharge_only = df_q is not None and not df_q.empty

    if not has_merged and not has_discharge_only:
        _plot_placeholder(ax, "Precip-Discharge Overlay\nN/A - No data available")
        return

    try:
        # Use merged data if available, otherwise fall back to discharge-only
        if has_merged and 'Discharge_cfs' in df_merged.columns:
            df_source = df_merged
        elif has_discharge_only and 'Discharge_cfs' in df_q.columns:
            df_source = df_q
            has_precip = False  # No precip in discharge-only data
        else:
            _plot_placeholder(ax, "Precip-Discharge Overlay\nN/A - No discharge data")
            return

        # Use full data range (user selected dates, not hardcoded last 2 years)
        df_plot = df_source.copy()

        if df_plot.empty:
            _plot_placeholder(ax, "Precip-Discharge Overlay\nNo data in range")
            return

        # Plot discharge on left axis (log scale)
        line1 = ax.plot(df_plot.index, df_plot['Discharge_cfs'],
                       color='steelblue', linewidth=1.5, label='Discharge', alpha=0.8)
        ax.set_yscale('log')
        ax.set_ylabel('Discharge (cfs) [Log Scale]', color='steelblue', fontweight='bold')
        ax.tick_params(axis='y', labelcolor='steelblue')
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.get_major_formatter().set_scientific(False)

        if has_precip:
            # Calculate cumulative precipitation (reset at start of plot period)
            df_plot['Cumulative_Precip_mm'] = df_plot['Precip_mm'].cumsum()

            # Create twin axis for precipitation
            ax2 = ax.twinx()

            # Plot daily precipitation as bars on right axis
            bar1 = ax2.bar(df_plot.index, df_plot['Precip_mm'],
                          color='lightblue', alpha=0.4, width=1, label='Daily Precip')

            # Plot cumulative precipitation as line on right axis
            line2 = ax2.plot(df_plot.index, df_plot['Cumulative_Precip_mm'],
                            color='darkorange', linewidth=2, label='Cumulative Precip', alpha=0.8)

            ax2.set_ylabel('Precipitation (mm)', color='darkblue', fontweight='bold')
            ax2.tick_params(axis='y', labelcolor='darkblue')
            ax2.set_ylim(bottom=0)

            # Combined legend
            lines = line1 + line2
            bars = [bar1]
            labels = [l.get_label() for l in lines] + ['Daily Precip']
            ax.legend(lines + bars, labels, loc='upper left', framealpha=0.9)

            title_suffix = "Discharge vs Precipitation"
        else:
            # Discharge-only legend and title
            ax.legend(loc='upper left', framealpha=0.9)
            title_suffix = "Discharge Only (No Climate Data)"

        # Formatting - show actual data range
        data_range = f"{df_plot.index.min().strftime('%Y-%m-%d')} to {df_plot.index.max().strftime('%Y-%m-%d')}"
        n_days = len(df_plot)
        ax.set_title(f'{title_suffix}\n{data_range} ({n_days:,} days)', fontweight='bold')
        ax.set_xlabel('Date')
        ax.grid(True, alpha=0.3)

        # Rotate x-axis labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    except Exception as e:
        logger.error(f"Precip-discharge overlay error: {e}")
        _plot_placeholder(ax, "Error plotting Precip-Discharge Overlay")




# ============================================================================
# PLOT 11: SEASONAL SCATTER (Q vs T colored by season)
# ============================================================================

def plot_seasonal_scatter(ax, df_merged: pd.DataFrame = None, config: Dict[str, Any] = None, **kwargs):
    """Scatter plot of Discharge vs Temperature colored by season."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    DISCHARGE_COL = cfg['discharge_col']
    TEMP_COL = cfg['temp_col']

    if df_merged is None or df_merged.empty:
        _plot_placeholder(ax, "Seasonal Scatter\nData N/A")
        return

    required_cols = [DISCHARGE_COL, TEMP_COL]
    if not all(col in df_merged.columns for col in required_cols):
        _plot_placeholder(ax, "Seasonal Scatter\nMissing columns")
        return

    try:
        df = df_merged[[DISCHARGE_COL, TEMP_COL]].dropna().copy()
        # Filter zeros before log scale
        df = df[df[DISCHARGE_COL] > 0]
        if df.empty:
            _plot_placeholder(ax, "Seasonal Scatter\nNo valid positive data")
            return

        df['month'] = df.index.month
        summer_mask = (df['month'] >= 5) & (df['month'] <= 9)
        winter_mask = ~summer_mask

        ax.scatter(df.loc[winter_mask, TEMP_COL], df.loc[winter_mask, DISCHARGE_COL],
                  alpha=0.4, s=15, color='#1f77b4', label='Winter (Oct-Apr)')
        ax.scatter(df.loc[summer_mask, TEMP_COL], df.loc[summer_mask, DISCHARGE_COL],
                  alpha=0.4, s=15, color='#2ca02c', label='Summer (May-Sep)')

        ax.set_yscale('log')
        ax.set_xlabel(PLOT_LABELS.get(TEMP_COL, TEMP_COL))
        ax.set_ylabel(f'{PLOT_LABELS.get(DISCHARGE_COL, DISCHARGE_COL)} [Log]')
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

        date_range = f"{df.index.min().strftime('%Y')} - {df.index.max().strftime('%Y')}"
        ax.set_title(f'Seasonal Discharge vs Temperature\n{date_range}', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    except Exception as e:
        logger.error(f"Seasonal scatter error: {e}")
        _plot_placeholder(ax, "Error plotting Seasonal Scatter")


# ============================================================================
# PLOT 12: 7-DAY LOW FLOW TREND
# ============================================================================

def plot_low_flow_trend(ax, df_q: pd.DataFrame = None, config: Dict[str, Any] = None, **kwargs):
    """Annual 7-day low flow with linear trend line."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    DISCHARGE_COL = cfg['discharge_col']

    if df_q is None or df_q.empty:
        _plot_placeholder(ax, "7-Day Low Flow Trend\nData N/A")
        return

    if DISCHARGE_COL not in df_q.columns:
        _plot_placeholder(ax, "7-Day Low Flow\nMissing column")
        return

    try:
        df = df_q[[DISCHARGE_COL]].dropna().copy()
        if len(df) < 365:
            _plot_placeholder(ax, "7-Day Low Flow\nNeed 1+ year data")
            return

        df['Q_7day'] = df[DISCHARGE_COL].rolling(window=7, min_periods=7).min()
        annual_lows = df['Q_7day'].resample('YE').min().dropna()

        if len(annual_lows) < 3:
            _plot_placeholder(ax, "7-Day Low Flow\nNeed 3+ years")
            return

        x_years = annual_lows.index.year.values
        y_vals = annual_lows.values
        z = np.polyfit(x_years, y_vals, 1)
        trend_x = np.linspace(x_years.min(), x_years.max(), 100)
        trend_y = np.poly1d(z)(trend_x)
        slope = z[0]
        epsilon = cfg.get('epsilon', 1e-10)
        y_mean = y_vals.mean()
        trend_pct = (slope * len(x_years)) / max(abs(y_mean), epsilon) * 100 if y_mean != 0 else 0

        ax.bar(x_years, y_vals, color='#d62728', alpha=0.7, width=0.7, label='Annual 7-Day Low')
        ax.plot(trend_x, trend_y, 'k--', lw=2, label=f'Trend ({trend_pct:+.1f}%)')
        ax.set_xlabel('Year')
        ax.set_ylabel('7-Day Low Flow (cfs)')
        ax.set_title(f'Annual 7-Day Low Flow\n{x_years.min()}-{x_years.max()}', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    except Exception as e:
        logger.error(f"Low flow trend error: {e}")
        _plot_placeholder(ax, "Error plotting Low Flow Trend")


# ============================================================================
# PLOT 13: ANNUAL MEAN TREND
# ============================================================================

def plot_annual_trend(ax, df_q: pd.DataFrame = None, config: Dict[str, Any] = None, **kwargs):
    """Annual mean discharge with linear trend."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    DISCHARGE_COL = cfg['discharge_col']

    if df_q is None or df_q.empty:
        _plot_placeholder(ax, "Annual Trend\nData N/A")
        return

    if DISCHARGE_COL not in df_q.columns:
        _plot_placeholder(ax, "Annual Trend\nMissing column")
        return

    try:
        df = df_q[[DISCHARGE_COL]].dropna().copy()
        annual_means = df[DISCHARGE_COL].resample('YE').mean().dropna()

        if len(annual_means) < 5:
            _plot_placeholder(ax, "Annual Trend\nNeed 5+ years")
            return

        x_years = annual_means.index.year.values
        y_vals = annual_means.values
        z = np.polyfit(x_years, y_vals, 1)
        trend_y = np.poly1d(z)(x_years)
        r = np.corrcoef(x_years, y_vals)[0, 1]
        r_squared = r ** 2
        direction = "increasing" if z[0] > 0 else "decreasing"

        ax.plot(x_years, y_vals, 'o-', color='steelblue', markersize=5, linewidth=1, label='Annual Mean')
        ax.plot(x_years, trend_y, 'r--', lw=2, label=f'Trend (R^2={r_squared:.3f})')
        ax.fill_between(x_years, y_vals, alpha=0.2, color='steelblue')
        ax.set_xlabel('Year')
        ax.set_ylabel('Mean Annual Discharge (cfs)')
        ax.set_title(f'Annual Discharge Trend ({direction})\n{x_years.min()}-{x_years.max()}', fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    except Exception as e:
        logger.error(f"Annual trend error: {e}")
        _plot_placeholder(ax, "Error plotting Annual Trend")

# ============================================================================
# PLOT 14: BASEFLOW SEPARATION (Lyne-Hollick Filter)
# ============================================================================

def plot_baseflow_separation(ax, df_q: pd.DataFrame = None, config: Dict[str, Any] = None, **kwargs):
    """
    Plot baseflow separation using Lyne-Hollick digital filter.

    The Lyne-Hollick filter separates baseflow from total streamflow using
    a recursive digital filter: Q_f(t) = alpha * Q_f(t-1) + (1+alpha)/2 * (Q(t) - Q(t-1))
    where Q_f is quickflow, and baseflow = Q - Q_f.

    Args:
        ax: Matplotlib axis to plot on
        df_q: DataFrame with Discharge_cfs column and datetime index
        config: Optional configuration dict with 'alpha' parameter (default 0.925)
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    DISCHARGE_COL = cfg['discharge_col']
    alpha = cfg.get('baseflow_alpha', 0.925)

    if df_q is None or df_q.empty:
        _plot_placeholder(ax, "Baseflow Separation\nData N/A")
        return

    if DISCHARGE_COL not in df_q.columns:
        _plot_placeholder(ax, "Baseflow Separation\nMissing discharge column")
        return

    try:
        # Use full data range (user selected dates)
        df_plot = df_q.copy()

        Q = df_plot[DISCHARGE_COL].dropna().values
        if len(Q) < 10:
            _plot_placeholder(ax, "Baseflow Separation\nInsufficient data")
            return

        # Lyne-Hollick filter (forward pass)
        Q_f = np.zeros_like(Q)  # quickflow
        for t in range(1, len(Q)):
            Q_f[t] = alpha * Q_f[t-1] + (1 + alpha) / 2 * (Q[t] - Q[t-1])
            Q_f[t] = max(0, Q_f[t])  # quickflow can't be negative

        # Baseflow = total - quickflow, but can't exceed total or be negative
        baseflow = np.clip(Q - Q_f, 0, Q)

        # Calculate baseflow index
        bfi = np.sum(baseflow) / np.sum(Q) if np.sum(Q) > 0 else 0

        # Create date index for plot
        dates = df_plot[DISCHARGE_COL].dropna().index

        # Clip floor for log scale (replace zeros with small positive value)
        Q_plot = np.where(Q > 0, Q, np.nan)
        baseflow_plot = np.where(baseflow > 0, baseflow, np.nan)

        # Plot total flow and baseflow
        ax.fill_between(dates, 0.01, Q_plot, color='steelblue', alpha=0.4, label='Total Flow')
        ax.fill_between(dates, 0.01, baseflow_plot, color='darkorange', alpha=0.7, label='Baseflow')
        ax.plot(dates, Q_plot, color='steelblue', linewidth=0.8, alpha=0.8)

        # Formatting
        ax.set_yscale('log')
        ax.set_xlabel('Date')
        ax.set_ylabel('Discharge (cfs) [Log Scale]')
        ax.set_title(f'Baseflow Separation (Lyne-Hollick, alpha={alpha})\nBaseflow Index = {bfi:.2f}', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.get_major_formatter().set_scientific(False)

        # Rotate x labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    except Exception as e:
        logger.error(f"Baseflow separation error: {e}")
        _plot_placeholder(ax, "Error plotting Baseflow Separation")


# ============================================================================
# PLOT 15: RECESSION CURVES
# ============================================================================

def plot_recession_curves(ax, df_q: pd.DataFrame = None, config: Dict[str, Any] = None, **kwargs):
    """
    Plot recession curve analysis showing Q(t) = Q0 * exp(-t/K).

    Identifies periods of consecutive decline in streamflow and fits
    exponential decay curves to estimate recession constants.

    Args:
        ax: Matplotlib axis to plot on
        df_q: DataFrame with Discharge_cfs column and datetime index
        config: Optional configuration dict with 'min_recession_days' (default 5)
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    DISCHARGE_COL = cfg['discharge_col']
    min_days = cfg.get('min_recession_days', 5)

    if df_q is None or df_q.empty:
        _plot_placeholder(ax, "Recession Curves\nData N/A")
        return

    if DISCHARGE_COL not in df_q.columns:
        _plot_placeholder(ax, "Recession Curves\nMissing discharge column")
        return

    try:
        Q = df_q[DISCHARGE_COL].dropna()
        if len(Q) < 30:
            _plot_placeholder(ax, "Recession Curves\nInsufficient data")
            return

        # Find recession periods (consecutive decline)
        Q_vals = Q.values
        dQ = np.diff(Q_vals)
        declining = dQ < 0

        # Find start and end of recession periods
        recessions = []
        in_recession = False
        start_idx = 0

        for i, dec in enumerate(declining):
            if dec and not in_recession:
                in_recession = True
                start_idx = i
            elif not dec and in_recession:
                in_recession = False
                if i - start_idx >= min_days:
                    recessions.append((start_idx, i))

        # Handle case where recession continues to end
        if in_recession and len(declining) - start_idx >= min_days:
            recessions.append((start_idx, len(declining)))

        if not recessions:
            _plot_placeholder(ax, "Recession Curves\nNo recession periods found")
            return

        # Fit recession curves and plot
        K_values = []
        colors = plt.cm.viridis(np.linspace(0, 1, min(len(recessions), 20)))

        for idx, (start, end) in enumerate(recessions[:20]):  # Limit to 20 curves
            t = np.arange(end - start + 1)
            Q_rec = Q_vals[start:end+1]
            Q0 = Q_rec[0]

            if Q0 <= 0 or np.any(Q_rec <= 0):
                continue

            # Fit exponential: ln(Q/Q0) = -t/K -> linear regression
            log_ratio = np.log(Q_rec / Q0)
            try:
                slope, _, r_value, _, _ = stats.linregress(t, log_ratio)
                if slope < 0:
                    K = -1 / slope
                    K_values.append(K)

                    # Normalize and plot
                    ax.plot(t, Q_rec / Q0, color=colors[idx % len(colors)],
                           alpha=0.5, linewidth=1)
            except (ValueError, RuntimeError):
                continue

        if not K_values:
            _plot_placeholder(ax, "Recession Curves\nCould not fit any curves")
            return

        # Plot mean recession curve
        K_mean = np.mean(K_values)
        K_std = np.std(K_values)
        t_plot = np.linspace(0, 30, 100)
        Q_mean = np.exp(-t_plot / K_mean)
        ax.plot(t_plot, Q_mean, 'r-', linewidth=2.5,
               label=f'Mean K = {K_mean:.1f} days (n={len(K_values)})')

        # Formatting
        ax.set_xlabel('Days since recession start')
        ax.set_ylabel('Q/Q0 (Normalized Discharge)')
        ax.set_title(f'Recession Curve Analysis\nK = {K_mean:.1f} +/- {K_std:.1f} days', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        ax.set_xlim(0, 30)

    except Exception as e:
        logger.error(f"Recession curves error: {e}")
        _plot_placeholder(ax, "Error plotting Recession Curves")


# ============================================================================
# PLOT 16: DOUBLE MASS CURVE
# ============================================================================

def plot_double_mass_curve(ax, df_merged: pd.DataFrame = None, config: Dict[str, Any] = None, **kwargs):
    """
    Plot double mass curve of cumulative precipitation vs cumulative discharge.

    Double mass curves are used to check consistency of hydrometeorological data
    and detect changes in the rainfall-runoff relationship over time.
    Points are colored by time to show temporal evolution.

    Args:
        ax: Matplotlib axis to plot on
        df_merged: DataFrame with Discharge_cfs and Precip_mm columns
        config: Optional configuration dict
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    DISCHARGE_COL = cfg['discharge_col']
    PRECIP_COL = cfg['precip_col']

    if df_merged is None or df_merged.empty:
        _plot_placeholder(ax, "Double Mass Curve\nData N/A")
        return

    required_cols = [DISCHARGE_COL, PRECIP_COL]
    if not all(col in df_merged.columns for col in required_cols):
        _plot_placeholder(ax, "Double Mass Curve\nMissing columns")
        return

    try:
        df = df_merged[[DISCHARGE_COL, PRECIP_COL]].dropna().copy()
        if len(df) < 30:
            _plot_placeholder(ax, "Double Mass Curve\nInsufficient data")
            return

        # Calculate cumulative values
        cum_precip = df[PRECIP_COL].cumsum()
        cum_discharge = df[DISCHARGE_COL].cumsum()

        # Color by time (year)
        years = df.index.year
        year_range = years.max() - years.min()
        if year_range > 0:
            colors = (years - years.min()) / year_range
        else:
            colors = np.zeros(len(years))

        # Create scatter plot colored by time
        scatter = ax.scatter(cum_precip, cum_discharge, c=colors, cmap='viridis',
                            s=3, alpha=0.6)

        # Add colorbar for years
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Year')
        # Set colorbar ticks to actual years
        tick_positions = np.linspace(0, 1, min(5, year_range + 1) if year_range > 0 else 1)
        tick_labels = [str(int(years.min() + t * year_range)) for t in tick_positions]
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels(tick_labels)

        # Fit overall linear trend
        slope, intercept, r_value, _, _ = stats.linregress(cum_precip, cum_discharge)
        x_fit = np.array([cum_precip.min(), cum_precip.max()])
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, 'r--', linewidth=2, label=f'Linear fit (R²={r_value**2:.3f})')

        # Formatting
        ax.set_xlabel('Cumulative Precipitation (mm)')
        ax.set_ylabel('Cumulative Discharge (cfs-days)')
        ax.set_title(f'Double Mass Curve\n{years.min()}-{years.max()}', fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

    except Exception as e:
        logger.error(f"Double mass curve error: {e}")
        _plot_placeholder(ax, "Error plotting Double Mass Curve")


# ============================================================================
# PLOT 17: RATING CURVE (Stage-Discharge)
# ============================================================================

def plot_rating_curve(ax, df_q: pd.DataFrame = None, config: Dict[str, Any] = None, **kwargs):
    """
    Plot stage-discharge rating curve with power-law fit: Q = A * H^B.

    Uses the fit_powerlaw_rating_curve function to fit the relationship
    between stage height and discharge, plotting observed points and fitted curve.

    Args:
        ax: Matplotlib axis to plot on
        df_q: DataFrame with Discharge_cfs and Gage_Height_ft columns
        config: Optional configuration dict
    """
    from ..analysis.stage_discharge import fit_powerlaw_rating_curve

    cfg = {**DEFAULT_CONFIG, **(config or {})}
    DISCHARGE_COL = cfg['discharge_col']
    STAGE_COL = cfg.get('stage_col', 'Gage_Height_ft')

    if df_q is None or df_q.empty:
        _plot_placeholder(ax, "Rating Curve\nData N/A")
        return

    if DISCHARGE_COL not in df_q.columns:
        _plot_placeholder(ax, "Rating Curve\nMissing discharge column")
        return

    if STAGE_COL not in df_q.columns:
        _plot_placeholder(ax, "Rating Curve\nMissing stage column (Gage_Height_ft)")
        return

    try:
        df = df_q[[STAGE_COL, DISCHARGE_COL]].dropna()
        df = df[(df[STAGE_COL] > 0) & (df[DISCHARGE_COL] > 0)]

        if len(df) < 10:
            _plot_placeholder(ax, "Rating Curve\nInsufficient data (need 10+ points)")
            return

        # Fit power-law rating curve
        A, B, R2, Q_pred = fit_powerlaw_rating_curve(
            df[STAGE_COL], df[DISCHARGE_COL], min_points=10
        )

        if np.isnan(A) or np.isnan(B):
            _plot_placeholder(ax, "Rating Curve\nCould not fit curve")
            return

        # Plot observed data
        ax.scatter(df[STAGE_COL], df[DISCHARGE_COL], c='steelblue', s=10,
                  alpha=0.4, label='Observed')

        # Plot fitted curve
        H_range = np.linspace(df[STAGE_COL].min(), df[STAGE_COL].max(), 100)
        Q_fit = A * H_range ** B
        ax.plot(H_range, Q_fit, 'r-', linewidth=2.5,
               label=f'Q = {A:.3f} * H^{B:.2f} (R²={R2:.3f})')

        # Formatting
        ax.set_xlabel('Stage Height (ft)')
        ax.set_ylabel('Discharge (cfs)')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_title('Stage-Discharge Rating Curve', fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.get_major_formatter().set_scientific(False)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.get_major_formatter().set_scientific(False)

    except Exception as e:
        logger.error(f"Rating curve error: {e}")
        _plot_placeholder(ax, "Error plotting Rating Curve")


# ============================================================================
# PLOT 18: FLOOD FREQUENCY ANALYSIS
# ============================================================================

def plot_flood_frequency(ax, df_q: pd.DataFrame = None, config: Dict[str, Any] = None, **kwargs):
    """
    Plot flood frequency analysis using Log-Pearson Type III distribution.

    Fits annual maximum flows to Log-Pearson Type III distribution and
    plots with Weibull plotting positions. Shows return period estimates.

    Args:
        ax: Matplotlib axis to plot on
        df_q: DataFrame with Discharge_cfs column and datetime index
        config: Optional configuration dict
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    DISCHARGE_COL = cfg['discharge_col']

    if df_q is None or df_q.empty:
        _plot_placeholder(ax, "Flood Frequency\nData N/A")
        return

    if DISCHARGE_COL not in df_q.columns:
        _plot_placeholder(ax, "Flood Frequency\nMissing discharge column")
        return

    try:
        # Get annual maximum flows
        annual_max = df_q[DISCHARGE_COL].resample('YE').max().dropna()

        if len(annual_max) < 10:
            _plot_placeholder(ax, "Flood Frequency\nNeed 10+ years of data")
            return

        n = len(annual_max)
        Q_sorted = np.sort(annual_max.values)[::-1]  # Descending order

        # Weibull plotting positions: P = m / (n + 1)
        m = np.arange(1, n + 1)
        prob = m / (n + 1)
        return_period = 1 / prob

        # Log-Pearson Type III fit
        log_Q = np.log10(Q_sorted)
        mean_log = np.mean(log_Q)
        std_log = np.std(log_Q, ddof=1)
        skew = stats.skew(log_Q, bias=False)

        # Generate fitted curve
        prob_fit = np.linspace(0.01, 0.99, 100)
        T_fit = 1 / prob_fit

        # Use Pearson Type III quantile function
        K_fit = stats.pearson3.ppf(1 - prob_fit, skew)
        log_Q_fit = mean_log + K_fit * std_log
        Q_fit = 10 ** log_Q_fit

        # Plot observed data
        ax.scatter(return_period, Q_sorted, c='steelblue', s=50, zorder=5,
                  edgecolors='black', linewidths=0.5, label='Annual Maximum')

        # Plot fitted curve
        ax.plot(T_fit, Q_fit, 'r-', linewidth=2, label='Log-Pearson III fit')

        # Add key return period annotations
        for T in [2, 10, 25, 50, 100]:
            if T <= T_fit.max():
                prob_T = 1 / T
                K_T = stats.pearson3.ppf(1 - prob_T, skew)
                Q_T = 10 ** (mean_log + K_T * std_log)
                ax.axvline(T, color='gray', linestyle=':', alpha=0.5)
                ax.text(T, Q_T * 1.1, f'Q{T}={Q_T:.0f}', fontsize=7, ha='center')

        # Formatting
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Return Period (years)')
        ax.set_ylabel('Discharge (cfs)')
        ax.set_title(f'Flood Frequency Analysis (Log-Pearson Type III)\nn={n} years, skew={skew:.2f}', fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.get_major_formatter().set_scientific(False)

    except Exception as e:
        logger.error(f"Flood frequency error: {e}")
        _plot_placeholder(ax, "Error plotting Flood Frequency")


# ============================================================================
# PLOT 19: 7Q10 LOW FLOW ANALYSIS
# ============================================================================

def plot_7q10_analysis(ax, df_q: pd.DataFrame = None, config: Dict[str, Any] = None, **kwargs):
    """
    Plot 7Q10 low flow frequency analysis using log-normal distribution.

    Calculates annual 7-day minimum flows and fits a log-normal distribution
    to estimate the 7Q10 (7-day, 10-year low flow) and other low flow statistics.

    Args:
        ax: Matplotlib axis to plot on
        df_q: DataFrame with Discharge_cfs column and datetime index
        config: Optional configuration dict
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    DISCHARGE_COL = cfg['discharge_col']

    if df_q is None or df_q.empty:
        _plot_placeholder(ax, "7Q10 Analysis\nData N/A")
        return

    if DISCHARGE_COL not in df_q.columns:
        _plot_placeholder(ax, "7Q10 Analysis\nMissing discharge column")
        return

    try:
        # Calculate 7-day rolling minimum (7Q10 definition: 7-day minimum, not mean)
        df = df_q[[DISCHARGE_COL]].dropna().copy()
        df['Q7'] = df[DISCHARGE_COL].rolling(window=7, min_periods=7).min()

        # Get annual minimum of 7-day minimum
        annual_7day_min = df['Q7'].resample('YE').min().dropna()

        if len(annual_7day_min) < 10:
            _plot_placeholder(ax, "7Q10 Analysis\nNeed 10+ years of data")
            return

        # Filter out zero/negative values — both scatter and fit must use same data
        annual_7day_min = annual_7day_min[annual_7day_min > 0]
        if len(annual_7day_min) < 10:
            _plot_placeholder(ax, "7Q10 Analysis\nNeed 10+ years of positive low-flow data")
            return

        n = len(annual_7day_min)
        Q_sorted = np.sort(annual_7day_min.values)  # Ascending for low flows

        # Weibull plotting positions for low flows: P = m / (n + 1)
        m = np.arange(1, n + 1)
        prob = m / (n + 1)  # Non-exceedance probability
        return_period = 1 / prob  # Return period for low flow

        # Log-normal fit (data already filtered to positive values)
        log_Q = np.log(Q_sorted)
        if len(log_Q) < 5:
            _plot_placeholder(ax, "7Q10 Analysis\nInsufficient data for fit")
            return

        mean_log = np.mean(log_Q)
        std_log = np.std(log_Q, ddof=1)

        # Generate fitted curve
        prob_fit = np.linspace(0.01, 0.5, 100)
        T_fit = 1 / prob_fit
        Q_fit = np.exp(mean_log + stats.norm.ppf(prob_fit) * std_log)

        # Calculate key statistics
        Q7_10 = np.exp(mean_log + stats.norm.ppf(0.1) * std_log)  # 10-year return period
        Q7_2 = np.exp(mean_log + stats.norm.ppf(0.5) * std_log)   # 2-year (median)
        Q7_5 = np.exp(mean_log + stats.norm.ppf(0.2) * std_log)   # 5-year

        # Plot observed data
        ax.scatter(return_period, Q_sorted, c='darkorange', s=50, zorder=5,
                  edgecolors='black', linewidths=0.5, label='Annual 7-day Minimum')

        # Plot fitted curve
        ax.plot(T_fit, Q_fit, 'r-', linewidth=2, label='Log-normal fit')

        # Add 7Q10 line
        ax.axhline(Q7_10, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvline(10, color='gray', linestyle=':', alpha=0.5)

        # Add annotations
        ax.text(0.98, 0.95, f'7Q10 = {Q7_10:.1f} cfs\n7Q5 = {Q7_5:.1f} cfs\n7Q2 = {Q7_2:.1f} cfs',
               transform=ax.transAxes, fontsize=9, ha='right', va='top',
               bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

        # Formatting
        ax.set_xscale('log')
        ax.set_xlabel('Return Period (years)')
        ax.set_ylabel('7-Day Low Flow (cfs)')
        ax.set_title(f'7Q10 Low Flow Analysis (Log-Normal)\nn={n} years', fontweight='bold')
        ax.legend(loc='lower left', fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_ylim(bottom=0)

    except Exception as e:
        logger.error(f"7Q10 analysis error: {e}")
        _plot_placeholder(ax, "Error plotting 7Q10 Analysis")


# ============================================================================
# PLOT 20: LAG CORRELATION
# ============================================================================

def plot_lag_correlation(ax, df_merged: pd.DataFrame = None, config: Dict[str, Any] = None, **kwargs):
    """
    Plot lag correlation between precipitation and discharge.

    Calculates Pearson correlation between precipitation and discharge
    at lags from 0 to 30 days to identify response time of the watershed.

    Args:
        ax: Matplotlib axis to plot on
        df_merged: DataFrame with Discharge_cfs and Precip_mm columns
        config: Optional configuration dict with 'max_lag_days' (default 30)
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    DISCHARGE_COL = cfg['discharge_col']
    PRECIP_COL = cfg['precip_col']
    max_lag = cfg.get('max_lag_days', 30)

    if df_merged is None or df_merged.empty:
        _plot_placeholder(ax, "Lag Correlation\nData N/A")
        return

    required_cols = [DISCHARGE_COL, PRECIP_COL]
    if not all(col in df_merged.columns for col in required_cols):
        _plot_placeholder(ax, "Lag Correlation\nMissing columns")
        return

    try:
        df = df_merged[[DISCHARGE_COL, PRECIP_COL]].dropna()

        if len(df) < max_lag + 10:
            _plot_placeholder(ax, "Lag Correlation\nInsufficient data")
            return

        Q = df[DISCHARGE_COL].values
        P = df[PRECIP_COL].values

        # Calculate correlations at each lag
        lags = np.arange(0, max_lag + 1)
        correlations = []
        p_values = []

        for lag in lags:
            if lag == 0:
                r, p = stats.pearsonr(P, Q)
            else:
                r, p = stats.pearsonr(P[:-lag], Q[lag:])
            correlations.append(r)
            p_values.append(p)

        correlations = np.array(correlations)
        p_values = np.array(p_values)

        # Find peak correlation
        peak_lag = lags[np.argmax(correlations)]
        peak_corr = correlations[np.argmax(correlations)]

        # Plot correlation bars
        colors = ['steelblue' if p < 0.05 else 'lightgray' for p in p_values]
        bars = ax.bar(lags, correlations, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

        # Highlight peak
        ax.bar(peak_lag, peak_corr, color='darkorange', alpha=0.9, edgecolor='black', linewidth=1.5)

        # Add significance threshold lines
        n = len(Q) - max_lag
        sig_threshold = 1.96 / np.sqrt(n)  # Approximate 95% CI
        ax.axhline(sig_threshold, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'95% CI (+/-{sig_threshold:.3f})')
        ax.axhline(-sig_threshold, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)

        # Formatting
        ax.set_xlabel('Lag (days)')
        ax.set_ylabel('Pearson Correlation')
        ax.set_title(f'Precipitation-Discharge Lag Correlation\nPeak: r={peak_corr:.3f} at lag={peak_lag} days', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xlim(-0.5, max_lag + 0.5)

        # Add color legend
        ax.text(0.02, 0.02, 'Blue: p < 0.05\nGray: p >= 0.05\nOrange: Peak',
               transform=ax.transAxes, fontsize=7, va='bottom',
               bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

    except Exception as e:
        logger.error(f"Lag correlation error: {e}")
        _plot_placeholder(ax, "Error plotting Lag Correlation")


# ============================================================================
# PLOT 21: ANOMALY DETECTION (Visual Highlighting)
# ============================================================================

def plot_anomaly_detection(ax, df_q: pd.DataFrame = None, config: Dict[str, Any] = None, **kwargs):
    """
    Plot discharge with anomaly detection using rolling mean +/- 2 standard deviations.

    Identifies and highlights outliers that fall outside the rolling window
    envelope defined by mean +/- 2 sigma.

    Args:
        ax: Matplotlib axis to plot on
        df_q: DataFrame with Discharge_cfs column and datetime index
        config: Optional configuration dict with 'anomaly_window_days' (default 30)
                and 'anomaly_sigma' (default 2)
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    DISCHARGE_COL = cfg['discharge_col']
    window = cfg.get('anomaly_window_days', 30)
    sigma = cfg.get('anomaly_sigma', 2)

    if df_q is None or df_q.empty:
        _plot_placeholder(ax, "Anomaly Detection\nData N/A")
        return

    if DISCHARGE_COL not in df_q.columns:
        _plot_placeholder(ax, "Anomaly Detection\nMissing discharge column")
        return

    try:
        # Get recent 3 years for clearer visualization
        end_date = df_q.index.max()
        start_date = end_date - pd.DateOffset(years=3)
        df = df_q.loc[start_date:end_date][[DISCHARGE_COL]].dropna().copy()

        if len(df) < window + 10:
            _plot_placeholder(ax, "Anomaly Detection\nInsufficient data")
            return

        # Filter zeros before log scale
        df = df[df[DISCHARGE_COL] > 0]
        if len(df) < window + 10:
            _plot_placeholder(ax, "Anomaly Detection\nInsufficient positive data")
            return

        Q = df[DISCHARGE_COL]

        # Calculate rolling statistics
        rolling_mean = Q.rolling(window=window, center=True).mean()
        rolling_std = Q.rolling(window=window, center=True).std()

        upper_bound = rolling_mean + sigma * rolling_std
        lower_bound = rolling_mean - sigma * rolling_std
        lower_bound = lower_bound.clip(lower=0)  # Can't be negative

        # Identify anomalies
        anomalies_high = Q > upper_bound
        anomalies_low = Q < lower_bound
        anomalies = anomalies_high | anomalies_low

        n_anomalies = anomalies.sum()
        pct_anomalies = 100 * n_anomalies / len(Q)

        # Plot
        ax.fill_between(df.index, lower_bound, upper_bound, color='lightblue',
                       alpha=0.4, label=f'Mean +/- {sigma}sigma')
        ax.plot(df.index, Q, color='steelblue', linewidth=0.8, alpha=0.7, label='Discharge')
        ax.plot(df.index, rolling_mean, color='darkblue', linewidth=1.5, label=f'{window}-day Mean')

        # Highlight anomalies
        if n_anomalies > 0:
            ax.scatter(df.index[anomalies_high], Q[anomalies_high], color='red', s=20,
                      zorder=5, label=f'High Anomalies ({anomalies_high.sum()})')
            ax.scatter(df.index[anomalies_low], Q[anomalies_low], color='orange', s=20,
                      zorder=5, label=f'Low Anomalies ({anomalies_low.sum()})')

        # Formatting
        ax.set_yscale('log')
        ax.set_xlabel('Date')
        ax.set_ylabel('Discharge (cfs) [Log Scale]')
        ax.set_title(f'Anomaly Detection ({window}-day window, {sigma}sigma)\n{n_anomalies} anomalies ({pct_anomalies:.1f}%)', fontweight='bold')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3, which='both')
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.get_major_formatter().set_scientific(False)

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    except Exception as e:
        logger.error(f"Anomaly detection error: {e}")
        _plot_placeholder(ax, "Error plotting Anomaly Detection")


# ============================================================================
# PLOT 22: CUMULATIVE DEPARTURE
# ============================================================================

def plot_cumulative_departure(ax, df_q: pd.DataFrame = None, config: Dict[str, Any] = None, **kwargs):
    """
    Plot cumulative departure from mean discharge.

    Calculates the cumulative sum of (Q - mean(Q)) to show periods of
    above-normal and below-normal flow. Positive slopes indicate above-normal
    periods, negative slopes indicate below-normal periods.

    Args:
        ax: Matplotlib axis to plot on
        df_q: DataFrame with Discharge_cfs column and datetime index
        config: Optional configuration dict
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    DISCHARGE_COL = cfg['discharge_col']

    if df_q is None or df_q.empty:
        _plot_placeholder(ax, "Cumulative Departure\nData N/A")
        return

    if DISCHARGE_COL not in df_q.columns:
        _plot_placeholder(ax, "Cumulative Departure\nMissing discharge column")
        return

    try:
        Q = df_q[DISCHARGE_COL].dropna()

        if len(Q) < 365:
            _plot_placeholder(ax, "Cumulative Departure\nNeed 1+ year of data")
            return

        # Calculate departure from mean
        Q_mean = Q.mean()
        departure = Q - Q_mean
        cum_departure = departure.cumsum()

        # Normalize by mean for interpretability (in units of "mean flow days")
        # Use epsilon to avoid division by zero
        epsilon = cfg.get("epsilon", 1e-10)
        cum_departure_norm = cum_departure / max(Q_mean, epsilon)

        # Find periods of positive and negative departure
        positive_mask = cum_departure_norm >= 0

        # Plot
        ax.fill_between(Q.index, 0, cum_departure_norm, where=positive_mask,
                       color='steelblue', alpha=0.5, label='Above normal')
        ax.fill_between(Q.index, 0, cum_departure_norm, where=~positive_mask,
                       color='darkorange', alpha=0.5, label='Below normal')
        ax.plot(Q.index, cum_departure_norm, color='black', linewidth=1)
        ax.axhline(0, color='black', linestyle='-', linewidth=1)

        # Add trend line
        x_numeric = np.arange(len(Q))
        z = np.polyfit(x_numeric, cum_departure_norm.values, 1)
        trend = np.poly1d(z)(x_numeric)
        ax.plot(Q.index, trend, 'r--', linewidth=2,
               label=f'Trend (slope: {z[0]*365:.1f}/year)')

        # Formatting
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Departure (mean-flow-days)')
        date_range = f"{Q.index.min().year}-{Q.index.max().year}"
        ax.set_title(f'Cumulative Departure from Mean Discharge\n{date_range} (Mean = {Q_mean:.0f} cfs)', fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    except Exception as e:
        logger.error(f"Cumulative departure error: {e}")
        _plot_placeholder(ax, "Error plotting Cumulative Departure")


# ============================================================================
# PLOT 23: SPECTRAL ANALYSIS (FFT-based)
# ============================================================================

def plot_spectral_analysis(ax, df_q: pd.DataFrame = None, config: Dict[str, Any] = None, **kwargs):
    """
    Plot FFT-based spectral analysis of discharge time series.

    Performs Fast Fourier Transform on discharge data to identify
    dominant periodicities (seasonal, annual, etc.) with smoothed power spectrum.

    Args:
        ax: Matplotlib axis to plot on
        df_q: DataFrame with Discharge_cfs column and datetime index
        config: Optional configuration dict with 'spectral_smoothing' window (default 5)
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    DISCHARGE_COL = cfg['discharge_col']
    smoothing_window = cfg.get('spectral_smoothing', 5)

    if df_q is None or df_q.empty:
        _plot_placeholder(ax, "Spectral Analysis\nData N/A")
        return

    if DISCHARGE_COL not in df_q.columns:
        _plot_placeholder(ax, "Spectral Analysis\nMissing discharge column")
        return

    try:
        Q = df_q[DISCHARGE_COL].dropna()

        if len(Q) < 365 * 2:
            _plot_placeholder(ax, "Spectral Analysis\nNeed 2+ years of data")
            return

        # Resample to ensure daily data and interpolate gaps
        Q_daily = Q.resample('D').mean()
        Q_daily = Q_daily.interpolate(method='linear', limit=7)
        Q_daily = Q_daily.dropna()

        if len(Q_daily) < 365:
            _plot_placeholder(ax, "Spectral Analysis\nInsufficient continuous data")
            return

        # Detrend and remove mean
        Q_values = Q_daily.values
        Q_detrended = Q_values - np.mean(Q_values)

        # Apply FFT
        n = len(Q_detrended)
        fft_result = np.fft.fft(Q_detrended)
        power = np.abs(fft_result) ** 2

        # Get frequencies (in cycles per day)
        freq = np.fft.fftfreq(n, d=1)  # d=1 day

        # Only positive frequencies
        positive_mask = freq > 0
        freq_pos = freq[positive_mask]
        power_pos = power[positive_mask]

        # Convert to period in days
        period = 1 / freq_pos

        # Smooth power spectrum
        if smoothing_window > 1:
            power_smooth = pd.Series(power_pos).rolling(window=smoothing_window, center=True).mean().values
        else:
            power_smooth = power_pos

        # Only show periods up to 5 years (1825 days) and above 2 days
        mask = (period >= 2) & (period <= 1825)
        period_plot = period[mask]
        power_plot = power_smooth[mask]

        # Find dominant periods
        # Look for peaks in specific ranges
        annual_mask = (period_plot >= 300) & (period_plot <= 400)
        semiannual_mask = (period_plot >= 150) & (period_plot <= 210)

        if np.any(annual_mask):
            annual_power = np.max(power_plot[annual_mask])
            annual_period = period_plot[annual_mask][np.argmax(power_plot[annual_mask])]
        else:
            annual_power, annual_period = None, None

        if np.any(semiannual_mask):
            semiannual_power = np.max(power_plot[semiannual_mask])
            semiannual_period = period_plot[semiannual_mask][np.argmax(power_plot[semiannual_mask])]
        else:
            semiannual_power, semiannual_period = None, None

        # Plot
        ax.loglog(period_plot, power_plot, color='steelblue', linewidth=1, alpha=0.8)

        # Mark key periods
        key_periods = [(365, 'Annual'), (182.5, 'Semi-annual'), (30, 'Monthly'), (7, 'Weekly')]
        for p, label in key_periods:
            ax.axvline(p, color='gray', linestyle='--', alpha=0.5)
            ax.text(p, ax.get_ylim()[1] * 0.5, label, rotation=90, va='center',
                   fontsize=7, color='gray')

        # Highlight detected peaks
        if annual_period is not None:
            ax.scatter([annual_period], [annual_power], color='red', s=100, zorder=5,
                      marker='*', label=f'Annual ({annual_period:.0f} days)')
        if semiannual_period is not None:
            ax.scatter([semiannual_period], [semiannual_power], color='orange', s=80, zorder=5,
                      marker='*', label=f'Semi-annual ({semiannual_period:.0f} days)')

        # Formatting
        ax.set_xlabel('Period (days)')
        ax.set_ylabel('Power Spectral Density')
        ax.set_title(f'Spectral Analysis (FFT)\n{len(Q_daily)} days of data', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlim(2, 1825)

    except Exception as e:
        logger.error(f"Spectral analysis error: {e}")
        _plot_placeholder(ax, "Error plotting Spectral Analysis")


# ============================================================================
# PLOT 24: TRANSMISSION LOSS (Two-gage comparison)
# ============================================================================

def plot_reach_comparison(ax, df_upstream: pd.DataFrame = None, df_downstream: pd.DataFrame = None,
                          config: Dict[str, Any] = None, **kwargs):
    """
    Plot reach comparison between upstream and downstream gages.

    Shows upstream Q, downstream Q, with blue shading when gaining
    and red shading when losing. Includes trend line on the gain.

    Args:
        ax: Matplotlib axis
        df_upstream: Discharge DataFrame for upstream gage (DatetimeIndex, Discharge_cfs)
        df_downstream: Discharge DataFrame for downstream gage (DatetimeIndex, Discharge_cfs)
        config: Optional configuration dict
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    DISCHARGE_COL = cfg['discharge_col']

    if df_upstream is None or df_upstream.empty or df_downstream is None or df_downstream.empty:
        _plot_placeholder(ax, "Reach Comparison\nNeed upstream & downstream data")
        return

    if DISCHARGE_COL not in df_upstream.columns or DISCHARGE_COL not in df_downstream.columns:
        _plot_placeholder(ax, "Reach Comparison\nMissing discharge column")
        return

    try:
        # Align on common dates (daily)
        q_up = df_upstream[DISCHARGE_COL].resample('D').mean().dropna()
        q_dn = df_downstream[DISCHARGE_COL].resample('D').mean().dropna()
        common_idx = q_up.index.intersection(q_dn.index)

        if len(common_idx) < 30:
            _plot_placeholder(ax, "Reach Comparison\nInsufficient overlapping data")
            return

        q_up = q_up.loc[common_idx]
        q_dn = q_dn.loc[common_idx]
        gain = q_dn - q_up  # Positive = gaining reach (downstream > upstream)

        # Plot upstream and downstream
        up_name = cfg.get('upstream_name', 'Upstream')
        dn_name = cfg.get('downstream_name', 'Downstream')

        ax.plot(common_idx, q_up, color='steelblue', linewidth=1, alpha=0.8, label=f'{up_name} (upstream)')
        ax.plot(common_idx, q_dn, color='darkorange', linewidth=1, alpha=0.8, label=f'{dn_name} (downstream)')

        # Shade: blue when gaining, red when losing
        ax.fill_between(common_idx, q_up, q_dn,
                        where=q_dn >= q_up,
                        color='steelblue', alpha=0.15, label='Gaining reach')
        ax.fill_between(common_idx, q_up, q_dn,
                        where=q_dn < q_up,
                        color='#d62728', alpha=0.15, label='Losing reach')

        # Trend line on the gain
        x_numeric = np.arange(len(gain))
        mask = ~np.isnan(gain.values)
        if mask.sum() > 10:
            z = np.polyfit(x_numeric[mask], gain.values[mask], 1)
            trend = np.poly1d(z)(x_numeric)
            cfs_per_year = z[0] * 365.25
            ax.plot(common_idx, trend, 'k--', linewidth=2,
                    label=f'Gain trend ({cfs_per_year:+.1f} cfs/yr)')

        # Formatting
        date_range = f"{common_idx.min().strftime('%Y')}-{common_idx.max().strftime('%Y')}"
        ax.set_xlabel('Date')
        ax.set_ylabel('Discharge (cfs)')
        ax.set_title(f'Reach Comparison: {up_name} \u2192 {dn_name}\n{date_range}', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    except Exception as e:
        logger.error(f"Reach comparison error: {e}")
        _plot_placeholder(ax, "Error plotting Reach Comparison")


# ============================================================================
# PLOT 25: SUMMER LOW FLOW TREND
# ============================================================================

def plot_summer_low_flow_trend(ax, df_q: pd.DataFrame = None, config: Dict[str, Any] = None, **kwargs):
    """
    Annual summer (Jun-Sep) 7-day minimum flow with trend line.

    More relevant for drought/dry-reach analysis than the annual 7-day min,
    which can be influenced by winter low flows.

    Args:
        ax: Matplotlib axis
        df_q: Discharge DataFrame
        config: Optional configuration dict
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    DISCHARGE_COL = cfg['discharge_col']

    if df_q is None or df_q.empty:
        _plot_placeholder(ax, "Summer Low Flow Trend\nData N/A")
        return

    if DISCHARGE_COL not in df_q.columns:
        _plot_placeholder(ax, "Summer Low Flow Trend\nMissing column")
        return

    try:
        df = df_q[[DISCHARGE_COL]].dropna().copy()

        # Filter to summer months (June-September)
        df_summer = df[df.index.month.isin([6, 7, 8, 9])]
        if len(df_summer) < 90:
            _plot_placeholder(ax, "Summer Low Flow\nNeed 1+ summer of data")
            return

        # 7-day rolling minimum
        df_summer = df_summer.copy()
        df_summer['Q_7day'] = df_summer[DISCHARGE_COL].rolling(window=7, min_periods=7).min()

        # Annual summer minimum
        annual_summer_lows = df_summer['Q_7day'].resample('YE').min().dropna()

        if len(annual_summer_lows) < 3:
            _plot_placeholder(ax, "Summer Low Flow\nNeed 3+ years")
            return

        x_years = annual_summer_lows.index.year.values
        y_vals = annual_summer_lows.values

        # Trend
        z = np.polyfit(x_years, y_vals, 1)
        trend_x = np.linspace(x_years.min(), x_years.max(), 100)
        trend_y = np.poly1d(z)(trend_x)
        slope = z[0]
        y_mean = y_vals.mean()
        epsilon = cfg.get('epsilon', 1e-10)
        trend_pct = (slope * len(x_years)) / max(abs(y_mean), epsilon) * 100 if y_mean != 0 else 0

        # Color bars by value (red for very low)
        colors = ['#d62728' if v < y_mean * 0.5 else '#ff7f0e' if v < y_mean else '#2ca02c'
                  for v in y_vals]

        # Dynamic color legend — only show colors that appear
        from matplotlib.patches import Patch
        ax.bar(x_years, y_vals, color=colors, alpha=0.7, width=0.7)
        ax.plot(trend_x, trend_y, 'k--', lw=2)

        legend_handles = []
        if '#2ca02c' in colors:
            legend_handles.append(Patch(facecolor='#2ca02c', alpha=0.7, label='Above avg'))
        if '#ff7f0e' in colors:
            legend_handles.append(Patch(facecolor='#ff7f0e', alpha=0.7, label='Below avg'))
        if '#d62728' in colors:
            legend_handles.append(Patch(facecolor='#d62728', alpha=0.7, label='< 50% of avg'))
        legend_handles.append(plt.Line2D([], [], color='k', linestyle='--', lw=2, label=f'Trend ({trend_pct:+.1f}%)'))

        # Annotate the minimum year
        min_idx = np.argmin(y_vals)
        ax.annotate(f'{y_vals[min_idx]:.0f} cfs',
                    xy=(x_years[min_idx], y_vals[min_idx]),
                    xytext=(0, -15), textcoords='offset points',
                    ha='center', fontsize=8, fontweight='bold', color='#d62728')

        ax.set_xlabel('Year')
        ax.set_ylabel('Summer 7-Day Low Flow (cfs)')
        ax.set_title(f'Summer (Jun-Sep) 7-Day Minimum Flow\n{x_years.min()}-{x_years.max()}',
                     fontweight='bold')
        ax.legend(handles=legend_handles, loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    except Exception as e:
        logger.error(f"Summer low flow trend error: {e}")
        _plot_placeholder(ax, "Error plotting Summer Low Flow Trend")


# ============================================================================
# PLOT 26: AQUIFER CONTRIBUTION INDEX
# ============================================================================

def plot_reach_index(ax, df_upstream: pd.DataFrame = None, df_downstream: pd.DataFrame = None,
                     config: Dict[str, Any] = None, **kwargs):
    """
    Plot gaining/losing reach index over time.

    Index = (Q_downstream - Q_upstream) / Q_upstream * 100
    Negative = losing reach (transmission loss).
    Trend shows if loss is accelerating.

    Args:
        ax: Matplotlib axis
        df_upstream: Discharge DataFrame for upstream gage
        df_downstream: Discharge DataFrame for downstream gage
        config: Optional configuration dict
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    DISCHARGE_COL = cfg['discharge_col']

    if df_upstream is None or df_upstream.empty or df_downstream is None or df_downstream.empty:
        _plot_placeholder(ax, "Reach Index\nNeed upstream & downstream data")
        return

    if DISCHARGE_COL not in df_upstream.columns or DISCHARGE_COL not in df_downstream.columns:
        _plot_placeholder(ax, "Reach Index\nMissing discharge column")
        return

    try:
        # Align on common dates (monthly means for cleaner signal)
        q_up = df_upstream[DISCHARGE_COL].resample('ME').mean().dropna()
        q_dn = df_downstream[DISCHARGE_COL].resample('ME').mean().dropna()
        common_idx = q_up.index.intersection(q_dn.index)

        if len(common_idx) < 12:
            _plot_placeholder(ax, "Reach Index\nNeed 1+ year of overlapping data")
            return

        q_up = q_up.loc[common_idx]
        q_dn = q_dn.loc[common_idx]

        # Reach index: % change from upstream to downstream
        # Filter out months with very low upstream flow (avoid division instability)
        epsilon = cfg.get('epsilon', 1e-10)
        reach_idx = ((q_dn - q_up) / q_up.clip(lower=epsilon)) * 100

        # Color by gaining/losing
        gaining = reach_idx >= 0
        losing = reach_idx < 0

        ax.bar(common_idx[gaining], reach_idx[gaining], width=25,
               color='steelblue', alpha=0.7, label='Gaining reach')
        ax.bar(common_idx[losing], reach_idx[losing], width=25,
               color='#d62728', alpha=0.7, label='Losing reach')

        ax.axhline(0, color='black', linewidth=1)

        # Trend line
        x_numeric = np.arange(len(reach_idx))
        z = np.polyfit(x_numeric, reach_idx.values, 1)
        trend = np.poly1d(z)(x_numeric)
        pct_per_year = z[0] * 12  # slope is per month, convert to per year
        ax.plot(common_idx, trend, 'k--', linewidth=2,
                label=f'Trend ({pct_per_year:+.1f}%/yr)')

        # Summary stats
        n_losing = losing.sum()
        n_total = len(reach_idx)
        median_idx = reach_idx.median()
        ax.text(0.02, 0.95,
                f'Median: {median_idx:+.1f}%\nLosing: {n_losing}/{n_total} months',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

        date_range = f"{common_idx.min().strftime('%Y')}-{common_idx.max().strftime('%Y')}"
        ax.set_xlabel('Date')
        ax.set_ylabel('Reach Index (%)')
        ax.set_title(f'Aquifer Contribution Index\n{date_range}', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    except Exception as e:
        logger.error(f"Reach index error: {e}")
        _plot_placeholder(ax, "Error plotting Reach Index")


# ============================================================================
# PLOT 27: PAIRED ANNUAL LOWS (Avista Window)
# ============================================================================

def _find_avista_windows(q_up, threshold=550):
    """
    Identify Avista low-flow windows: contiguous periods where upstream
    daily discharge stays below threshold (cfs). Returns list of
    (year, start_date, end_date) tuples for windows >= 7 days in summer.
    """
    below = q_up < threshold
    windows = []
    in_window = False
    start = None

    for date, val in below.items():
        if val and date.month in [7, 8, 9, 10]:  # Summer/early fall
            if not in_window:
                in_window = True
                start = date
        else:
            if in_window:
                duration = (date - start).days
                if duration >= 7:
                    windows.append((start.year, start, date - pd.Timedelta(days=1)))
                in_window = False
                start = None

    # Close any trailing window
    if in_window and start is not None:
        last_date = q_up.index[-1]
        duration = (last_date - start).days
        if duration >= 7:
            windows.append((start.year, start, last_date))

    return windows


def plot_paired_annual_lows(ax, df_upstream: pd.DataFrame = None, df_downstream: pd.DataFrame = None,
                            config: Dict[str, Any] = None, **kwargs):
    """
    Paired bar chart of 7-day minimum flow during Avista low-flow windows.

    During each year's Avista window (Post Falls < 550 cfs):
    - Blue bars: Post Falls 7-day min (flat, ~500 cfs)
    - Orange bars: Greene St 7-day min (declining year over year)
    - Trend line on Greene St
    - Annotation: gap = aquifer contribution

    Args:
        ax: Matplotlib axis
        df_upstream: Post Falls discharge (DatetimeIndex, Discharge_cfs)
        df_downstream: Greene St discharge (DatetimeIndex, Discharge_cfs)
        config: Optional configuration dict
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    DISCHARGE_COL = cfg['discharge_col']
    up_name = cfg.get('upstream_name', 'Upstream')
    dn_name = cfg.get('downstream_name', 'Downstream')

    if df_upstream is None or df_upstream.empty or df_downstream is None or df_downstream.empty:
        _plot_placeholder(ax, "Paired Annual Lows\nNeed upstream & downstream data")
        return

    if DISCHARGE_COL not in df_upstream.columns or DISCHARGE_COL not in df_downstream.columns:
        _plot_placeholder(ax, "Paired Annual Lows\nMissing discharge column")
        return

    try:
        q_up = df_upstream[DISCHARGE_COL].resample('D').mean().dropna()
        q_dn = df_downstream[DISCHARGE_COL].resample('D').mean().dropna()

        # Find low-flow windows
        windows = _find_avista_windows(q_up)
        if len(windows) < 2:
            _plot_placeholder(ax, "Paired Annual Lows\nNeed 2+ low-flow windows")
            return

        # Compute 7-day min for each gage during each window
        years = []
        pf_mins = []
        gs_mins = []

        for year, w_start, w_end in windows:
            up_window = q_up.loc[w_start:w_end]
            dn_window = q_dn.loc[w_start:w_end]

            if len(up_window) < 7 or len(dn_window) < 7:
                continue

            pf_7d = up_window.rolling(7, min_periods=7).min().min()
            gs_7d = dn_window.rolling(7, min_periods=7).min().min()

            if pd.notna(pf_7d) and pd.notna(gs_7d):
                years.append(year)
                pf_mins.append(pf_7d)
                gs_mins.append(gs_7d)

        if len(years) < 2:
            _plot_placeholder(ax, "Paired Annual Lows\nInsufficient window data")
            return

        years = np.array(years)
        pf_mins = np.array(pf_mins)
        gs_mins = np.array(gs_mins)

        # Grouped bars
        bar_width = 0.35
        x = np.arange(len(years))

        bars_pf = ax.bar(x - bar_width/2, pf_mins, bar_width, color='steelblue',
                         alpha=0.8, label=f'{up_name} 7d min', edgecolor='white')
        bars_gs = ax.bar(x + bar_width/2, gs_mins, bar_width, color='darkorange',
                         alpha=0.8, label=f'{dn_name} 7d min', edgecolor='white')

        # Trend line on downstream
        z = np.polyfit(x, gs_mins, 1)
        trend_y = np.poly1d(z)(x)
        cfs_per_year = z[0]
        ax.plot(x + bar_width/2, trend_y, 'k--', linewidth=2,
                label=f'GS trend ({cfs_per_year:+.0f} cfs/yr)')

        # Annotate aquifer contribution (gap between bars)
        for i in range(len(years)):
            gap = gs_mins[i] - pf_mins[i]
            mid_y = pf_mins[i] + gap / 2
            ax.annotate(f'+{gap:.0f}', xy=(x[i], mid_y),
                        ha='center', va='center', fontsize=7, color='#555555',
                        fontweight='bold')

        # Value labels on bars
        for i, (pf, gs) in enumerate(zip(pf_mins, gs_mins)):
            ax.text(x[i] - bar_width/2, pf + 10, f'{pf:.0f}', ha='center', va='bottom',
                    fontsize=7, color='steelblue', fontweight='bold')
            ax.text(x[i] + bar_width/2, gs + 10, f'{gs:.0f}', ha='center', va='bottom',
                    fontsize=7, color='darkorange', fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(years)
        ax.set_xlabel('Year')
        ax.set_ylabel('7-Day Minimum Flow (cfs)')
        ax.set_title(f'Annual Low Flows During Low-Flow Window\n{up_name} (flat) vs {dn_name} (declining)',
                     fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(bottom=0)

        # Subtitle annotation
        ax.text(0.02, 0.95, 'Gap = aquifer contribution\n(shrinking over time)',
                transform=ax.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.8))

    except Exception as e:
        logger.error(f"Paired annual lows error: {e}")
        _plot_placeholder(ax, "Error plotting Paired Annual Lows")


# ============================================================================
# PLOT 28: AVISTA WINDOW COMPARISON
# ============================================================================

def plot_avista_window_comparison(ax, df_upstream: pd.DataFrame = None, df_downstream: pd.DataFrame = None,
                                  config: Dict[str, Any] = None, **kwargs):
    """
    Overlaid hydrographs of Greene St during Avista low-flow windows.

    X-axis: days since start of window (normalized).
    One line per year, colored from blue (oldest) to red (newest).
    Shows the annual low getting lower each year.

    Args:
        ax: Matplotlib axis
        df_upstream: Post Falls discharge (for window detection)
        df_downstream: Greene St discharge (plotted)
        config: Optional configuration dict
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    DISCHARGE_COL = cfg['discharge_col']
    up_name = cfg.get('upstream_name', 'Upstream')
    dn_name = cfg.get('downstream_name', 'Downstream')

    if df_upstream is None or df_upstream.empty or df_downstream is None or df_downstream.empty:
        _plot_placeholder(ax, "Low-Flow Window Comparison\nNeed upstream & downstream data")
        return

    if DISCHARGE_COL not in df_upstream.columns or DISCHARGE_COL not in df_downstream.columns:
        _plot_placeholder(ax, "Low-Flow Window Comparison\nMissing discharge column")
        return

    try:
        q_up = df_upstream[DISCHARGE_COL].resample('D').mean().dropna()
        q_dn = df_downstream[DISCHARGE_COL].resample('D').mean().dropna()

        windows = _find_avista_windows(q_up)
        if len(windows) < 2:
            _plot_placeholder(ax, "Low-Flow Window Comparison\nNeed 2+ low-flow windows")
            return

        # Fixed comparison length: first 14 days of each window
        COMPARE_DAYS = 14

        # Color map: blue (oldest) → red (newest)
        n_windows = len(windows)
        cmap = plt.cm.viridis
        colors = [cmap(i / max(n_windows - 1, 1)) for i in range(n_windows)]

        for i, (year, w_start, w_end) in enumerate(windows):
            # Take exactly COMPARE_DAYS from start of window
            window_end = w_start + pd.Timedelta(days=COMPARE_DAYS - 1)
            dn_window = q_dn.loc[w_start:window_end]
            if len(dn_window) < 7:
                continue

            days = (dn_window.index - w_start).days
            min_q = dn_window.min()

            ax.plot(days, dn_window.values, color=colors[i], linewidth=2.2,
                    alpha=0.85, label=f'{year} (min={min_q:.0f})')

            # Mark the minimum point
            min_day = days[dn_window.values.argmin()]
            ax.plot(min_day, min_q, 'o', color=colors[i], markersize=5, zorder=5)

        ax.set_xlabel('Days Since Start of Low-Flow Window')
        ax.set_ylabel(f'{dn_name} Discharge (cfs)')
        ax.set_title(f'{dn_name} During First {COMPARE_DAYS} Days of Low-Flow Window\nEach year compared on equal footing',
                     fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, COMPARE_DAYS - 0.5)

        # Reference line at 1000 cfs
        ax.axhline(1000, color='gray', linestyle=':', linewidth=1, alpha=0.7)
        ax.text(COMPARE_DAYS - 1, 1005, '1,000 cfs', ha='right', fontsize=7, color='gray')

    except Exception as e:
        logger.error(f"Avista window comparison error: {e}")
        _plot_placeholder(ax, "Error plotting Low-Flow Window Comparison")


# ============================================================================
# PLOT 29: THRESHOLD EXCEEDANCE
# ============================================================================

def plot_threshold_exceedance(ax, df_q: pd.DataFrame = None, config: Dict[str, Any] = None, **kwargs):
    """
    Days below critical flow thresholds by year.

    Stacked bars showing summer days below configured thresholds, or
    site-specific low-flow quantiles when thresholds are not configured.
    Demonstrates worsening low-flow conditions over time without assuming
    a Greene St-specific threshold set.

    Args:
        ax: Matplotlib axis
        df_q: Discharge DataFrame (single gage)
        config: Optional configuration dict
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    DISCHARGE_COL = cfg['discharge_col']
    threshold_colors = cfg.get('threshold_colors', ['#ffc107', '#ff9800', '#d62728'])

    if df_q is None or df_q.empty:
        _plot_placeholder(ax, "Threshold Exceedance\nData N/A")
        return

    if DISCHARGE_COL not in df_q.columns:
        _plot_placeholder(ax, "Threshold Exceedance\nMissing discharge column")
        return

    try:
        df = df_q[[DISCHARGE_COL]].dropna().copy()
        df['year'] = df.index.year

        # Only count summer months (Jun-Sep) for relevance
        df_summer = df[df.index.month.isin([6, 7, 8, 9])]
        if df_summer.empty:
            _plot_placeholder(ax, "Threshold Exceedance\nNo summer data")
            return

        years = sorted(df_summer['year'].unique())
        if len(years) < 2:
            _plot_placeholder(ax, "Threshold Exceedance\nNeed 2+ years")
            return

        if cfg.get('flow_thresholds'):
            thresholds = sorted([float(t) for t in cfg['flow_thresholds']], reverse=True)
            threshold_source = "configured"
        else:
            quantiles = df_summer[DISCHARGE_COL].quantile([0.25, 0.10, 0.05]).dropna()
            thresholds = sorted(quantiles.unique(), reverse=True)
            threshold_source = "summer flow quantiles"

        if len(thresholds) < 2:
            _plot_placeholder(ax, "Threshold Exceedance\nNeed variable summer flows")
            return

        if len(threshold_colors) < len(thresholds):
            threshold_colors = list(plt.cm.YlOrRd(np.linspace(0.35, 0.9, len(thresholds))))

        # Count days below each threshold per year
        data = {t: [] for t in thresholds}
        for year in years:
            yr_data = df_summer[df_summer['year'] == year][DISCHARGE_COL]
            for t in thresholds:
                data[t].append((yr_data < t).sum())

        x = np.arange(len(years))
        bar_width = 0.6
        lower_counts = np.zeros(len(years))
        bottom = np.zeros(len(years))

        for idx, threshold in enumerate(reversed(thresholds)):
            counts = np.array(data[threshold])
            band = counts - lower_counts
            color = threshold_colors[len(thresholds) - idx - 1]
            if idx == 0:
                label = f"< {threshold:,.0f} cfs"
            else:
                lower = list(reversed(thresholds))[idx - 1]
                label = f"{lower:,.0f}-{threshold:,.0f} cfs"
            ax.bar(x, band, bar_width, bottom=bottom, color=color, label=label)
            bottom += band
            lower_counts = counts

        # Total label on top
        totals = np.array(data[thresholds[0]])
        for i, total in enumerate(totals):
            if total > 0:
                ax.text(x[i], total + 1, str(total), ha='center', va='bottom',
                        fontsize=7, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(years, rotation=45, ha='right')
        ax.set_xlabel('Year')
        ax.set_ylabel('Summer Days Below Threshold (Jun-Sep)')
        ax.set_title(f'Days Below Low-Flow Thresholds\nSummer months only, {threshold_source}',
                     fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(bottom=0)

    except Exception as e:
        logger.error(f"Threshold exceedance error: {e}")
        _plot_placeholder(ax, "Error plotting Threshold Exceedance")


# ============================================================================
# PLOT 30: PRECIPITATION-DISCHARGE RESPONSE COMPARISON
# ============================================================================

def plot_precip_response_comparison(ax, df_upstream: pd.DataFrame = None,
                                    df_downstream: pd.DataFrame = None,
                                    df_climate: pd.DataFrame = None,
                                    config: Dict[str, Any] = None, **kwargs):
    """
    Summer pulse propagation ratio by year.

    For each summer (Jun-Sep) precip event (>=2mm/day), compute the ratio of
    downstream discharge response to upstream discharge response.
    Declining ratio = aquifer contributing less to baseflow recovery.

    Args:
        ax: Matplotlib axis
        df_upstream: Post Falls discharge
        df_downstream: Greene St discharge
        df_climate: Climate DataFrame with Precip_mm column (from Meteostat)
        config: Optional configuration dict
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    DISCHARGE_COL = cfg['discharge_col']
    PRECIP_COL = cfg.get('precip_col', 'Precip_mm')

    if df_upstream is None or df_downstream is None:
        _plot_placeholder(ax, "Precip Response\nNeed upstream & downstream data")
        return

    if df_climate is None or df_climate.empty or PRECIP_COL not in df_climate.columns:
        _plot_placeholder(ax, "Precip Response\nNeed climate data (Precip_mm)")
        return

    try:
        q_up = df_upstream[DISCHARGE_COL].resample('D').mean().dropna()
        q_dn = df_downstream[DISCHARGE_COL].resample('D').mean().dropna()
        precip = df_climate[PRECIP_COL].resample('D').sum().dropna()

        # Normalize timezones — Meteostat is tz-naive, USGS is tz-aware UTC
        if precip.index.tz is None and q_up.index.tz is not None:
            precip.index = precip.index.tz_localize(q_up.index.tz)
        elif precip.index.tz is not None and q_up.index.tz is None:
            precip.index = precip.index.tz_localize(None)

        # Common index across all three datasets
        common_idx = q_up.index.intersection(q_dn.index).intersection(precip.index)
        q_up = q_up.loc[common_idx]
        q_dn = q_dn.loc[common_idx]
        precip = precip.loc[common_idx]

        # Filter to extended summer (May-Sep) for more precip events
        summer_mask = precip.index.month.isin([5, 6, 7, 8, 9])
        precip_summer = precip[summer_mask]
        q_up_summer = q_up[summer_mask]
        q_dn_summer = q_dn[summer_mask]

        avail_years = sorted(precip_summer.index.year.unique())
        if len(avail_years) < 2:
            _plot_placeholder(ax, "Precip Response\nNeed 2+ summers of data")
            return

        years = []
        ratios = []
        n_events_list = []

        for year in avail_years:
            yr_mask = precip_summer.index.year == year
            p_yr = precip_summer[yr_mask]
            up_yr = q_up_summer[yr_mask]
            dn_yr = q_dn_summer[yr_mask]

            if len(p_yr) < 30:
                continue

            # Find precip events (>= 2mm/day)
            event_days = p_yr[p_yr >= 2.0].index
            if len(event_days) < 2:
                continue

            # For each event, measure 1-3 day discharge response at both gages
            event_ratios = []
            for event_date in event_days:
                response_start = event_date + pd.Timedelta(days=1)
                response_end = event_date + pd.Timedelta(days=3)

                # Get pre-event discharge
                pre_dates = up_yr.loc[:event_date]
                if len(pre_dates) == 0:
                    continue
                up_before = pre_dates.iloc[-1]

                pre_dates_dn = dn_yr.loc[:event_date]
                if len(pre_dates_dn) == 0:
                    continue
                dn_before = pre_dates_dn.iloc[-1]

                # Get post-event peak (look in full dataset, not just summer filter)
                up_after = q_up.loc[response_start:response_end]
                dn_after = q_dn.loc[response_start:response_end]

                if len(up_after) == 0 or len(dn_after) == 0:
                    continue

                up_rise = up_after.max() - up_before
                dn_rise = dn_after.max() - dn_before

                # Only count meaningful upstream pulses (> 20 cfs rise)
                if abs(up_rise) > 20:
                    event_ratios.append(dn_rise / up_rise)

            if len(event_ratios) >= 1:
                years.append(year)
                ratios.append(np.median(event_ratios))
                n_events_list.append(len(event_ratios))

        if len(years) < 2:
            _plot_placeholder(ax, "Precip Response\nInsufficient summer precip events")
            return

        years = np.array(years)
        ratios = np.array(ratios)

        # Bar chart
        colors = ['#2ca02c' if r > 0.7 else '#ff7f0e' if r > 0.5 else '#d62728' for r in ratios]
        ax.bar(years, ratios, color=colors, alpha=0.8, width=0.6, edgecolor='white')

        # Trend line
        x_norm = np.arange(len(years))
        z = np.polyfit(x_norm, ratios, 1)
        trend_y = np.poly1d(z)(x_norm)
        ax.plot(years, trend_y, 'k--', linewidth=2, label='Trend')

        # Value labels
        for yr, r, n in zip(years, ratios, n_events_list):
            ax.text(yr, r + 0.02, f'{r:.2f}\n({n})', ha='center', va='bottom',
                    fontsize=7, fontweight='bold')

        # Reference line at 1.0
        ax.axhline(1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.text(years[0] - 0.3, 1.02, 'Equal response', fontsize=7, color='gray')

        ax.set_xlabel('Year')
        ax.set_ylabel('Pulse Propagation Ratio (GS rise / PF rise)')
        ax.set_title('Summer (May-Sep) Precipitation→Discharge Response\nDeclining ratio = weakening aquifer contribution',
                     fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(bottom=0)

        # Explain (n) values
        ax.text(0.02, 0.05, '(n) = number of qualifying\nprecip events (≥2mm/day)',
                transform=ax.transAxes, fontsize=7, va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

    except Exception as e:
        logger.error(f"Precip response comparison error: {e}")
        _plot_placeholder(ax, "Error plotting Precip Response")


# ============================================================================
# PLOT 31: SUMMER CLIMATE CONTEXT
# ============================================================================

def plot_summer_climate_context(ax, df_climate: pd.DataFrame = None,
                                config: Dict[str, Any] = None, **kwargs):
    """
    Summer precipitation totals and mean temperature by year.

    Dual-axis: bars for precip (mm), line for temp (°C).
    Contextualizes flow declines with climate drivers.

    Args:
        ax: Matplotlib axis
        df_climate: Climate DataFrame with Precip_mm and Temp_C columns
        config: Optional configuration dict
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    PRECIP_COL = cfg.get('precip_col', 'Precip_mm')
    TEMP_COL = cfg.get('temp_col', 'Temp_C')

    if df_climate is None or df_climate.empty:
        _plot_placeholder(ax, "Summer Climate Context\nNeed climate data")
        return

    if PRECIP_COL not in df_climate.columns or TEMP_COL not in df_climate.columns:
        _plot_placeholder(ax, "Summer Climate Context\nNeed Precip_mm and Temp_C")
        return

    try:
        df = df_climate[[PRECIP_COL, TEMP_COL]].copy()
        df.index = pd.to_datetime(df.index)

        # Filter to summer (Jun-Sep)
        df_summer = df[df.index.month.isin([6, 7, 8, 9])]
        if len(df_summer) < 60:
            _plot_placeholder(ax, "Summer Climate\nInsufficient data")
            return

        # Annual summer totals/means
        summer_precip = df_summer[PRECIP_COL].resample('YE').sum()
        summer_temp = df_summer[TEMP_COL].resample('YE').mean()

        # Align
        years = summer_precip.index.year.values
        precip_vals = summer_precip.values
        temp_vals = summer_temp.values

        if len(years) < 2:
            _plot_placeholder(ax, "Summer Climate\nNeed 2+ years")
            return

        # Convert to inches and Fahrenheit
        precip_in = precip_vals / 25.4  # mm → inches
        temp_f = temp_vals * 9 / 5 + 32  # °C → °F

        # Precip bars
        precip_mean_in = precip_in.mean()
        bar_colors = ['#d62728' if p < precip_mean_in * 0.5 else '#ff7f0e' if p < precip_mean_in else '#2196F3'
                      for p in precip_in]
        ax.bar(years, precip_in, color=bar_colors, alpha=0.7, width=0.6)

        # Value labels on bars
        for yr, p in zip(years, precip_in):
            ax.text(yr, p + 0.03, f'{p:.1f}"', ha='center', va='bottom', fontsize=7, fontweight='bold')

        ax.set_xlabel('Year')
        ax.set_ylabel('Summer Precipitation Total (in)', color='#2196F3')
        ax.tick_params(axis='y', labelcolor='#2196F3')
        ax.set_ylim(bottom=0)

        # Precip mean reference
        ax.axhline(precip_mean_in, color='#2196F3', linestyle=':', linewidth=1, alpha=0.5)
        ax.text(years[0] - 0.4, precip_mean_in + 0.03, f'Avg: {precip_mean_in:.1f}"',
                fontsize=7, color='#2196F3')

        # Temp line on secondary axis
        ax2 = ax.twinx()
        ax2.plot(years, temp_f, 'o-', color='#d62728', linewidth=2, markersize=6,
                 label='Mean Summer Temp')

        # Temp value labels
        for yr, t in zip(years, temp_f):
            ax2.text(yr, t + 0.3, f'{t:.0f}°F', ha='center', va='bottom',
                     fontsize=7, color='#d62728')

        ax2.set_ylabel('Mean Summer Temperature (°F)', color='#d62728')
        ax2.tick_params(axis='y', labelcolor='#d62728')

        # Temp mean reference
        temp_mean_f = temp_f.mean()
        ax2.axhline(temp_mean_f, color='#d62728', linestyle=':', linewidth=1, alpha=0.3)

        ax.set_title('Summer Climate Context (Jun-Sep)\nPrecipitation & Temperature by Year',
                     fontweight='bold')
        ax.grid(True, alpha=0.2, axis='y')

        # Dynamic legend for precip colors
        from matplotlib.patches import Patch
        legend_handles = []
        if '#2196F3' in bar_colors:
            legend_handles.append(Patch(facecolor='#2196F3', alpha=0.7, label='Precip ≥ avg'))
        if '#ff7f0e' in bar_colors:
            legend_handles.append(Patch(facecolor='#ff7f0e', alpha=0.7, label='Precip below avg'))
        if '#d62728' in bar_colors:
            legend_handles.append(Patch(facecolor='#d62728', alpha=0.7, label='Precip < 50% avg'))
        legend_handles.append(plt.Line2D([], [], color='#d62728', marker='o', linewidth=2,
                                         markersize=6, label='Mean temp'))
        ax.legend(handles=legend_handles, loc='upper left', fontsize=7)

    except Exception as e:
        logger.error(f"Summer climate context error: {e}")
        _plot_placeholder(ax, "Error plotting Summer Climate Context")


# ============================================================================
# PLOT 32: SEASONAL GAIN/LOSS COMPARISON
# ============================================================================

def plot_seasonal_gain_loss(ax, df_upstream: pd.DataFrame = None,
                            df_downstream: pd.DataFrame = None,
                            config: Dict[str, Any] = None, **kwargs):
    """
    Average reach gain/loss by season, comparing early vs recent years.

    Grouped bars: early period vs recent period for each season.
    Shows summer losses worsening while other seasons stay stable.

    Args:
        ax: Matplotlib axis
        df_upstream: Post Falls discharge
        df_downstream: Greene St discharge
        config: Optional configuration dict
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    DISCHARGE_COL = cfg['discharge_col']

    if df_upstream is None or df_upstream.empty or df_downstream is None or df_downstream.empty:
        _plot_placeholder(ax, "Seasonal Gain/Loss\nNeed upstream & downstream data")
        return

    if DISCHARGE_COL not in df_upstream.columns or DISCHARGE_COL not in df_downstream.columns:
        _plot_placeholder(ax, "Seasonal Gain/Loss\nMissing discharge column")
        return

    try:
        q_up = df_upstream[DISCHARGE_COL].resample('D').mean().dropna()
        q_dn = df_downstream[DISCHARGE_COL].resample('D').mean().dropna()
        common_idx = q_up.index.intersection(q_dn.index)

        if len(common_idx) < 365 * 2:
            _plot_placeholder(ax, "Seasonal Gain/Loss\nNeed 2+ years of overlap")
            return

        q_up = q_up.loc[common_idx]
        q_dn = q_dn.loc[common_idx]
        gain = q_dn - q_up  # Positive = gaining

        # Assign seasons
        month = gain.index.month
        seasons = pd.Series('', index=gain.index)
        seasons[month.isin([12, 1, 2])] = 'Winter'
        seasons[month.isin([3, 4, 5])] = 'Spring'
        seasons[month.isin([6, 7, 8, 9])] = 'Summer'
        seasons[month.isin([10, 11])] = 'Fall'

        # Split into early and recent halves
        all_years = sorted(gain.index.year.unique())
        mid_year = all_years[len(all_years) // 2]
        early_label = f'{all_years[0]}-{mid_year - 1}'
        recent_label = f'{mid_year}-{all_years[-1]}'

        early_mask = gain.index.year < mid_year
        recent_mask = gain.index.year >= mid_year

        season_order = ['Winter', 'Spring', 'Summer', 'Fall']
        early_means = []
        recent_means = []

        for s in season_order:
            s_mask = seasons == s
            early_vals = gain[early_mask & s_mask]
            recent_vals = gain[recent_mask & s_mask]
            early_means.append(early_vals.mean() if len(early_vals) > 0 else 0)
            recent_means.append(recent_vals.mean() if len(recent_vals) > 0 else 0)

        early_means = np.array(early_means)
        recent_means = np.array(recent_means)

        # Grouped bars
        x = np.arange(len(season_order))
        bar_width = 0.35

        bars_early = ax.bar(x - bar_width/2, early_means, bar_width,
                            color='steelblue', alpha=0.8, label=early_label, edgecolor='white')
        bars_recent = ax.bar(x + bar_width/2, recent_means, bar_width,
                             color='darkorange', alpha=0.8, label=recent_label, edgecolor='white')

        # Value labels
        for i in range(len(season_order)):
            ax.text(x[i] - bar_width/2, early_means[i] + (10 if early_means[i] >= 0 else -25),
                    f'{early_means[i]:+.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold',
                    color='steelblue')
            ax.text(x[i] + bar_width/2, recent_means[i] + (10 if recent_means[i] >= 0 else -25),
                    f'{recent_means[i]:+.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold',
                    color='darkorange')

        ax.axhline(0, color='black', linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(season_order, fontsize=10)
        ax.set_xlabel('Season')
        ax.set_ylabel('Mean Reach Gain (cfs)\n(Downstream − Upstream)')
        ax.set_title('Seasonal Reach Gain: Early vs Recent Years\nPositive = gaining, Negative = losing',
                     fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        # Highlight summer change
        summer_change = recent_means[2] - early_means[2]
        if abs(summer_change) > 10:
            ax.annotate(f'Summer: {summer_change:+.0f} cfs',
                        xy=(2, recent_means[2]), xytext=(2.8, recent_means[2] - 100),
                        fontsize=9, fontweight='bold', color='#d62728',
                        arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5))

    except Exception as e:
        logger.error(f"Seasonal gain/loss error: {e}")
        _plot_placeholder(ax, "Error plotting Seasonal Gain/Loss")


# ============================================================================
# PLOT 33: SEASONAL GAIN/LOSS BY WATER-YEAR PERIOD (2-YEAR CHUNKS)
# ============================================================================

def plot_seasonal_gain_loss_annual(ax, df_upstream: pd.DataFrame = None,
                                   df_downstream: pd.DataFrame = None,
                                   config: Dict[str, Any] = None, **kwargs):
    """
    Seasonal reach gain/loss evolution using 2-water-year periods.

    Water years (Oct-Sep) respect the snowpack-driven hydrology.
    Each Avista regulation window falls cleanly within one water year.
    8 years of data → 4 periods of 2 water years each, stacked vertically
    as grouped bars by season, showing the progressive summer decline.

    Args:
        ax: Matplotlib axis
        df_upstream: Post Falls discharge
        df_downstream: Greene St discharge
        config: Optional configuration dict
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    DISCHARGE_COL = cfg['discharge_col']
    up_name = cfg.get('upstream_name', 'Upstream')
    dn_name = cfg.get('downstream_name', 'Downstream')

    if df_upstream is None or df_upstream.empty or df_downstream is None or df_downstream.empty:
        _plot_placeholder(ax, "Seasonal Gain/Loss by Period\nNeed upstream & downstream data")
        return

    if DISCHARGE_COL not in df_upstream.columns or DISCHARGE_COL not in df_downstream.columns:
        _plot_placeholder(ax, "Seasonal Gain/Loss by Period\nMissing discharge column")
        return

    try:
        q_up = df_upstream[DISCHARGE_COL].resample('D').mean().dropna()
        q_dn = df_downstream[DISCHARGE_COL].resample('D').mean().dropna()
        common_idx = q_up.index.intersection(q_dn.index)

        if len(common_idx) < 365 * 2:
            _plot_placeholder(ax, "Seasonal Gain/Loss by Period\nNeed 2+ years of overlap")
            return

        q_up = q_up.loc[common_idx]
        q_dn = q_dn.loc[common_idx]
        gain = q_dn - q_up  # Positive = gaining reach

        # Assign water year (Oct-Sep): Oct 2017 → WY 2018, Sep 2018 → WY 2018
        water_year = gain.index.year.where(gain.index.month >= 10, gain.index.year - 1) + 1
        # So WY = calendar year of the January in that water year

        # Assign seasons within water year
        month = gain.index.month
        seasons = pd.Series('', index=gain.index)
        seasons[month.isin([10, 11, 12])] = 'Fall'       # Oct-Dec (start of WY)
        seasons[month.isin([1, 2, 3])] = 'Winter'         # Jan-Mar
        seasons[month.isin([4, 5])] = 'Spring'            # Apr-May (snowmelt)
        seasons[month.isin([6, 7, 8, 9])] = 'Summer'      # Jun-Sep (low flow + Avista)

        # Get complete water years only (need Oct through Sep)
        wy_counts = water_year.value_counts()
        complete_wys = sorted(wy_counts[wy_counts >= 300].index)

        if len(complete_wys) < 4:
            _plot_placeholder(ax, "Seasonal Gain/Loss by Period\nNeed 4+ complete water years")
            return

        # Build 2-water-year chunks
        n_periods = len(complete_wys) // 2
        if n_periods < 2:
            _plot_placeholder(ax, "Seasonal Gain/Loss by Period\nNeed 4+ water years for periods")
            return

        periods = []
        for i in range(n_periods):
            wy1 = complete_wys[i * 2]
            wy2 = complete_wys[i * 2 + 1]
            label = f'WY {wy1}-{wy2}'
            mask = water_year.isin([wy1, wy2])
            periods.append((label, mask))

        # Season order: follow the water year flow
        season_order = ['Fall', 'Winter', 'Spring', 'Summer']

        # Compute means per period per season
        period_means = []
        for label, mask in periods:
            means = []
            for s in season_order:
                s_mask = seasons == s
                vals = gain[mask & s_mask]
                means.append(vals.mean() if len(vals) > 0 else 0)
            period_means.append(np.array(means))

        # Color palette: blue → orange → red progression (early → recent)
        colors = ['#4a90d9', '#7cb342', '#f9a825', '#e53935']
        if len(periods) > len(colors):
            colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(periods)))

        # Grouped bars
        x = np.arange(len(season_order))
        n_bars = len(periods)
        total_width = 0.75
        bar_width = total_width / n_bars

        for i, ((label, _), means) in enumerate(zip(periods, period_means)):
            offset = (i - (n_bars - 1) / 2) * bar_width
            color = colors[i] if i < len(colors) else colors[-1]
            bars = ax.bar(x + offset, means, bar_width * 0.9,
                          color=color, alpha=0.85, label=label, edgecolor='white', linewidth=0.5)

        # Value labels on summer bars only (the key story)
        summer_idx = season_order.index('Summer')
        for i, means in enumerate(period_means):
            offset = (i - (n_bars - 1) / 2) * bar_width
            val = means[summer_idx]
            color = colors[i] if i < len(colors) else colors[-1]
            ax.text(x[summer_idx] + offset, val - 15 if val < 0 else val + 8,
                    f'{val:+.0f}', ha='center', va='top' if val < 0 else 'bottom',
                    fontsize=7, fontweight='bold', color=color)

        ax.axhline(0, color='black', linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(season_order, fontsize=10)
        ax.set_xlabel('Season (Water Year: Oct → Sep)')
        ax.set_ylabel(f'Mean Reach Gain (cfs)\n({dn_name} \u2212 {up_name})')
        ax.set_title('Seasonal Reach Gain by Water-Year Period\nPositive = gaining, Negative = losing',
                     fontweight='bold')
        ax.legend(loc='upper left', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3, axis='y')

        # Annotate summer trend
        summer_vals = [m[summer_idx] for m in period_means]
        if len(summer_vals) >= 2:
            total_change = summer_vals[-1] - summer_vals[0]
            if abs(total_change) > 10:
                ax.annotate(f'Summer trend: {total_change:+.0f} cfs\n({periods[0][0]} → {periods[-1][0]})',
                            xy=(summer_idx + (n_bars - 1) / 2 * bar_width + 0.1, summer_vals[-1]),
                            xytext=(summer_idx + 0.8, min(summer_vals) - 80),
                            fontsize=8, fontweight='bold', color='#d62728',
                            arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5),
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    except Exception as e:
        logger.error(f"Seasonal gain/loss annual error: {e}")
        _plot_placeholder(ax, "Error plotting Seasonal Gain/Loss by Period")


# ============================================================================
# AVAILABLE PLOTS REGISTRY
# ============================================================================

AVAILABLE_PLOTS = {
    'anomaly': {
        'function': plot_anomaly,
        'description': 'Recent decade vs historical monthly averages (Q, T, P)',
        'requires': ['df_merged'],
        'default_size': (10, 6)
    },
    'hexbin_temp': {
        'function': lambda ax, **kwargs: plot_hexbin(ax, x_col='Temp_C', y_col='Discharge_cfs',
                                                     use_log_scale=True, add_counts=True, **kwargs),
        'description': 'Discharge vs Temperature hexbin (log scale)',
        'requires': ['df_merged', 'analysis_results'],
        'default_size': (8, 6)
    },
    'lagged_precip': {
        'function': plot_monthly_lagged_scatter,
        'description': 'Monthly avg discharge vs lagged precipitation',
        'requires': ['df_merged', 'analysis_results'],
        'default_size': (8, 6)
    },
    'timeseries': {
        'function': plot_timeseries,
        'description': 'Recent discharge time series (log scale)',
        'requires': ['df_q'],
        'default_size': (10, 5)
    },
    'flow_duration': {
        'function': plot_flow_duration,
        'description': 'Flow duration curve',
        'requires': ['df_q'],
        'default_size': (8, 6)
    },
    'correlation_matrix': {
        'function': plot_correlation_matrix,
        'description': 'Correlation heatmap with significance',
        'requires': ['df_merged', 'analysis_results'],
        'default_size': (6, 5)
    },
    'monthly_boxplot': {
        'function': plot_monthly_boxplot,
        'description': 'Monthly discharge distribution boxplots',
        'requires': ['df_q'],
        'default_size': (10, 6)
    },
    'discharge_heatmap': {
        'function': plot_discharge_heatmap,
        'description': 'Discharge density vs day of year heatmap',
        'requires': ['df_q'],
        'default_size': (10, 6)
    },
    'temporal_heatmap': {
        'function': plot_temporal_heatmap,
        'description': '4-panel temporal discharge density (5/10/20yr + total)',
        'requires': ['df_q'],
        'default_size': (16, 12)
    },
    'precip_discharge': {
        'function': plot_precip_discharge_overlay,
        'description': 'Discharge with precipitation overlay (dual y-axis)',
        'requires': ['df_q'],  # Can work with just df_q, enhanced with df_merged
        'default_size': (14, 6)
    },
    'seasonal_scatter': {
        'function': plot_seasonal_scatter,
        'description': 'Discharge vs Temperature colored by season (winter/summer)',
        'requires': ['df_merged'],
        'default_size': (10, 6)
    },
    'low_flow_trend': {
        'function': plot_low_flow_trend,
        'description': 'Annual 7-day low flow with trend line',
        'requires': ['df_q'],
        'default_size': (12, 6)
    },
    'annual_trend': {
        'function': plot_annual_trend,
        'description': 'Annual mean discharge trend analysis',
        'requires': ['df_q'],
        'default_size': (12, 6)
    },
    'baseflow_separation': {
        'function': plot_baseflow_separation,
        'description': 'Baseflow separation using Lyne-Hollick digital filter',
        'requires': ['df_q'],
        'default_size': (14, 6)
    },
    'recession_curves': {
        'function': plot_recession_curves,
        'description': 'Recession curve analysis with exponential decay fit',
        'requires': ['df_q'],
        'default_size': (10, 6)
    },
    'double_mass_curve': {
        'function': plot_double_mass_curve,
        'description': 'Cumulative precipitation vs cumulative discharge',
        'requires': ['df_merged'],
        'default_size': (10, 8)
    },
    'rating_curve': {
        'function': plot_rating_curve,
        'description': 'Stage-discharge rating curve with power-law fit',
        'requires': ['df_q'],
        'default_size': (10, 8)
    },
    'flood_frequency': {
        'function': plot_flood_frequency,
        'description': 'Flood frequency analysis (Log-Pearson Type III)',
        'requires': ['df_q'],
        'default_size': (10, 8)
    },
    '7q10_analysis': {
        'function': plot_7q10_analysis,
        'description': '7Q10 low flow frequency analysis (log-normal)',
        'requires': ['df_q'],
        'default_size': (10, 8)
    },
    'lag_correlation': {
        'function': plot_lag_correlation,
        'description': 'Precipitation-discharge lag correlation (0-30 days)',
        'requires': ['df_merged'],
        'default_size': (12, 6)
    },
    'anomaly_detection': {
        'function': plot_anomaly_detection,
        'description': 'Anomaly detection with rolling mean +/- 2 sigma',
        'requires': ['df_q'],
        'default_size': (14, 6)
    },
    'cumulative_departure': {
        'function': plot_cumulative_departure,
        'description': 'Cumulative departure from mean discharge',
        'requires': ['df_q'],
        'default_size': (14, 6)
    },
    'spectral_analysis': {
        'function': plot_spectral_analysis,
        'description': 'FFT-based spectral analysis of discharge',
        'requires': ['df_q'],
        'default_size': (12, 8)
    },
    'reach_comparison': {
        'function': plot_reach_comparison,
        'description': 'Reach comparison: upstream vs downstream (gaining/losing)',
        'requires': ['df_upstream', 'df_downstream'],
        'default_size': (14, 6)
    },
    'summer_low_flow_trend': {
        'function': plot_summer_low_flow_trend,
        'description': 'Summer (Jun-Sep) 7-day minimum flow trend',
        'requires': ['df_q'],
        'default_size': (12, 6)
    },
    'reach_index': {
        'function': plot_reach_index,
        'description': 'Aquifer contribution index (% change upstream to downstream)',
        'requires': ['df_upstream', 'df_downstream'],
        'default_size': (14, 6)
    },
    'paired_annual_lows': {
        'function': plot_paired_annual_lows,
        'description': 'Paired 7-day lows during low-flow windows (upstream flat, downstream declining)',
        'requires': ['df_upstream', 'df_downstream'],
        'default_size': (12, 7)
    },
    'avista_window_comparison': {
        'function': plot_avista_window_comparison,
        'description': 'Downstream hydrographs overlaid across low-flow windows by year',
        'requires': ['df_upstream', 'df_downstream'],
        'default_size': (12, 7)
    },
    'threshold_exceedance': {
        'function': plot_threshold_exceedance,
        'description': 'Days below critical flow thresholds by year (stacked bars)',
        'requires': ['df_q'],
        'default_size': (12, 6)
    },
    'precip_response_comparison': {
        'function': plot_precip_response_comparison,
        'description': 'Precipitation pulse propagation ratio by year (requires climate data)',
        'requires': ['df_upstream', 'df_downstream', 'df_climate'],
        'default_size': (12, 7)
    },
    'summer_climate_context': {
        'function': plot_summer_climate_context,
        'description': 'Summer precip totals and mean temperature by year',
        'requires': ['df_climate'],
        'default_size': (12, 6)
    },
    'seasonal_gain_loss': {
        'function': plot_seasonal_gain_loss,
        'description': 'Seasonal reach gain/loss: early vs recent years',
        'requires': ['df_upstream', 'df_downstream'],
        'default_size': (10, 7)
    },
    'seasonal_gain_loss_annual': {
        'function': plot_seasonal_gain_loss_annual,
        'description': 'Seasonal reach gain/loss by 2-water-year periods (Oct-Sep)',
        'requires': ['df_upstream', 'df_downstream'],
        'default_size': (12, 7)
    },
}
