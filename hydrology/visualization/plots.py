"""
Individual plot functions for hydrology analysis.

Each plot is self-contained and reusable. Plots take data and an axis,
render to that axis, and return nothing. This makes them composable
into any layout you want.

Available plots:
- anomaly: Recent decade vs historical monthly averages (Q, T, P)
- hexbin_temp: Discharge vs Temperature hexbin (log scale with counts)
- lagged_precip: Monthly avg discharge vs lagged precipitation
- timeseries: Recent discharge time series (log scale)
- flow_duration: Flow duration curve
- correlation_matrix: Correlation heatmap

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
                    except:
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

        sns.heatmap(corr_matrix, annot=annot, fmt='', cmap='coolwarm', center=0,
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
        import calendar
        import seaborn as sns

        # Prepare data
        df_copy = df_q.copy()
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
        discharge_vals = df_q['Discharge_cfs'].values
        day_of_year = df_q.index.dayofyear.values

        # Calculate log bins
        min_q = np.floor(np.log10(max(discharge_vals.min(), 0.01)))
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

        # Global min/max for consistent Y-axis
        global_min_q = df_q['Discharge_cfs'].min()
        global_max_q = df_q['Discharge_cfs'].max()
        if global_min_q <= 0:
            global_min_q = 0.1

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

def plot_precip_discharge_overlay(ax, df_merged: pd.DataFrame = None, config: Dict[str, Any] = None, **kwargs):
    """
    Plot discharge with precipitation overlay (dual y-axis) and cumulative precipitation.

    Shows:
    - Discharge (left y-axis, blue line)
    - Daily precipitation (right y-axis, blue bars)
    - Cumulative precipitation (right y-axis, orange line)

    Args:
        ax: Matplotlib axis
        df_merged: Merged DataFrame with Discharge_cfs and Precip_mm columns
        config: Optional configuration
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    if df_merged is None or df_merged.empty:
        _plot_placeholder(ax, "Precip-Discharge Overlay\nN/A - No merged data")
        return

    if 'Precip_mm' not in df_merged.columns:
        _plot_placeholder(ax, "Precip-Discharge Overlay\nN/A - No precipitation data")
        return

    try:
        # Get recent data (last 2 years by default for readability)
        end_date = df_merged.index.max()
        start_date = end_date - pd.DateOffset(years=2)
        df_plot = df_merged.loc[start_date:end_date].copy()

        if df_plot.empty:
            _plot_placeholder(ax, "Precip-Discharge Overlay\nNo recent data")
            return

        # Calculate cumulative precipitation (reset at start of plot period)
        df_plot['Cumulative_Precip_mm'] = df_plot['Precip_mm'].cumsum()

        # Create twin axis for precipitation
        ax2 = ax.twinx()

        # Plot discharge on left axis (log scale)
        line1 = ax.plot(df_plot.index, df_plot['Discharge_cfs'],
                       color='steelblue', linewidth=1.5, label='Discharge', alpha=0.8)
        ax.set_yscale('log')
        ax.set_ylabel('Discharge (cfs) [Log Scale]', color='steelblue', fontweight='bold')
        ax.tick_params(axis='y', labelcolor='steelblue')
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.get_major_formatter().set_scientific(False)

        # Plot daily precipitation as bars on right axis
        bar1 = ax2.bar(df_plot.index, df_plot['Precip_mm'],
                      color='lightblue', alpha=0.4, width=1, label='Daily Precip')

        # Plot cumulative precipitation as line on right axis
        line2 = ax2.plot(df_plot.index, df_plot['Cumulative_Precip_mm'],
                        color='darkorange', linewidth=2, label='Cumulative Precip', alpha=0.8)

        ax2.set_ylabel('Precipitation (mm)', color='darkblue', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='darkblue')
        ax2.set_ylim(bottom=0)

        # Formatting
        data_range = f"{df_plot.index.min().strftime('%Y-%m-%d')} to {df_plot.index.max().strftime('%Y-%m-%d')}"
        ax.set_title(f'Discharge vs Precipitation (Last 2 Years)\nData: {data_range}', fontweight='bold')
        ax.set_xlabel('Date')
        ax.grid(True, alpha=0.3)

        # Combined legend
        lines = line1 + line2
        bars = [bar1]
        labels = [l.get_label() for l in lines] + ['Daily Precip']
        ax.legend(lines + bars, labels, loc='upper left', framealpha=0.9)

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
        if df.empty:
            _plot_placeholder(ax, "Seasonal Scatter\nNo valid data")
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

        df['Q_7day'] = df[DISCHARGE_COL].rolling(window=7, min_periods=7).mean()
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
        trend_pct = (slope * len(x_years)) / y_vals.mean() * 100 if y_vals.mean() != 0 else 0

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
        'requires': ['df_merged'],
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
}
