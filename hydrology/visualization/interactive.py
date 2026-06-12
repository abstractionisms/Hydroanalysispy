"""
Interactive Plotly chart functions for the Hydrology Dashboard.

Provides hover/zoom/pan versions of the most-used plots:
- Hydrograph with percentile bands
- Flow Duration Curve with Koehler (2025) dQ/dt coloring
- Multi-site comparison overlay with unified hover
- Baseflow separation waterfall for reach analysis
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional, List

from ..core.logging_setup import get_logger

logger = get_logger(__name__)


def interactive_hydrograph(
    df_q: pd.DataFrame,
    df_hist: pd.DataFrame = None,
    discharge_col: str = 'Discharge_cfs',
    title: str = "Discharge Hydrograph",
    show_percentile_bands: bool = True,
    aggregation: str = "daily",
) -> go.Figure:
    """
    Interactive Plotly hydrograph with hover, zoom, and optional percentile bands.

    Args:
        df_q: Discharge DataFrame (datetime index, discharge_col column)
        df_hist: Optional historical DataFrame for computing percentile bands
        discharge_col: Column name for discharge values
        title: Plot title
        show_percentile_bands: Whether to show 10/25/50/75/90th percentile envelopes
        aggregation: 'daily', 'weekly', or 'monthly'

    Returns:
        Plotly Figure
    """
    if df_q is None or df_q.empty or discharge_col not in df_q.columns:
        fig = go.Figure()
        fig.add_annotation(text="No discharge data available", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        return fig

    # Resample if requested
    q_series = df_q[discharge_col].dropna()
    if aggregation == "weekly":
        q_series = q_series.resample('W').mean().dropna()
    elif aggregation == "monthly":
        q_series = q_series.resample('ME').mean().dropna()

    fig = go.Figure()

    # Percentile bands from historical data
    hist_source = df_hist if df_hist is not None else df_q
    if show_percentile_bands and hist_source is not None and not hist_source.empty:
        hist_col = discharge_col if discharge_col in hist_source.columns else 'value'
        if hist_col in hist_source.columns:
            hist_vals = hist_source[hist_col].dropna()
            if len(hist_vals) > 365:
                # Compute day-of-year percentiles
                hist_df = pd.DataFrame({'value': hist_vals, 'doy': hist_vals.index.dayofyear})
                percentiles = hist_df.groupby('doy')['value'].quantile([0.10, 0.25, 0.50, 0.75, 0.90]).unstack()
                percentiles.columns = ['p10', 'p25', 'p50', 'p75', 'p90']

                # Map to current year's dates for display
                year = q_series.index.max().year if not q_series.empty else pd.Timestamp.now().year
                band_dates = pd.date_range(f'{year}-01-01', f'{year}-12-31', freq='D')
                band_doy = band_dates.dayofyear

                # Only plot bands for DOYs we have
                valid_doys = percentiles.index.intersection(band_doy)
                if len(valid_doys) > 30:
                    band_x = [band_dates[band_dates.dayofyear == d][0] for d in valid_doys if d in band_dates.dayofyear]
                    p10 = [percentiles.loc[d, 'p10'] for d in valid_doys]
                    p25 = [percentiles.loc[d, 'p25'] for d in valid_doys]
                    p50 = [percentiles.loc[d, 'p50'] for d in valid_doys]
                    p75 = [percentiles.loc[d, 'p75'] for d in valid_doys]
                    p90 = [percentiles.loc[d, 'p90'] for d in valid_doys]

                    # 10-90th band
                    fig.add_trace(go.Scatter(
                        x=band_x + band_x[::-1],
                        y=p90 + p10[::-1],
                        fill='toself', fillcolor='rgba(100, 149, 237, 0.1)',
                        line=dict(width=0), name='10-90th percentile',
                        hoverinfo='skip', showlegend=True
                    ))

                    # 25-75th band
                    fig.add_trace(go.Scatter(
                        x=band_x + band_x[::-1],
                        y=p75 + p25[::-1],
                        fill='toself', fillcolor='rgba(100, 149, 237, 0.2)',
                        line=dict(width=0), name='25-75th percentile',
                        hoverinfo='skip', showlegend=True
                    ))

                    # Median line
                    fig.add_trace(go.Scatter(
                        x=band_x, y=p50,
                        mode='lines', name='Median',
                        line=dict(color='rgba(100, 149, 237, 0.5)', width=1, dash='dash'),
                        hovertemplate='Median: %{y:,.0f} cfs<extra></extra>'
                    ))

    # Main discharge trace
    fig.add_trace(go.Scatter(
        x=q_series.index, y=q_series.values,
        mode='lines', name='Discharge',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='%{x|%Y-%m-%d}<br>%{y:,.0f} cfs<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Discharge (cfs)",
        yaxis_type="log",
        height=450,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=60, b=40),
    )

    return fig


def interactive_fdc(
    df_q: pd.DataFrame,
    discharge_col: str = 'Discharge_cfs',
    title: str = "Flow Duration Curve",
    color_by_dqdt: bool = True,
) -> go.Figure:
    """
    Interactive Flow Duration Curve with optional Koehler (2025) dQ/dt coloring.

    Points colored by rate of change:
    - Blue = rising limb (dQ/dt > 0)
    - Red = falling limb (dQ/dt < 0)
    - Gray = stable (|dQ/dt| < threshold)

    Args:
        df_q: Discharge DataFrame
        discharge_col: Column name
        title: Plot title
        color_by_dqdt: Whether to color points by dQ/dt (Koehler 2025 enhancement)

    Returns:
        Plotly Figure
    """
    from ..analysis.stage_discharge import flow_duration_curve

    if df_q is None or df_q.empty or discharge_col not in df_q.columns:
        fig = go.Figure()
        fig.add_annotation(text="No discharge data available", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        return fig

    q = df_q[discharge_col].dropna().sort_index()

    if color_by_dqdt and len(q) > 2:
        # Calculate dQ/dt (rate of change)
        dq = q.diff()

        # Classify: rising, falling, stable
        threshold = q.std() * 0.01  # 1% of std as stability threshold
        conditions = np.where(dq > threshold, 'rising',
                    np.where(dq < -threshold, 'falling', 'stable'))

        # Sort by discharge (descending) for FDC
        sorted_idx = q.argsort()[::-1]
        q_sorted = q.iloc[sorted_idx].values
        conditions_sorted = conditions[sorted_idx]
        n = len(q_sorted)
        exceedance = np.arange(1, n + 1) / (n + 1) * 100

        fig = go.Figure()

        # Plot each category
        colors = {'rising': '#2196F3', 'falling': '#F44336', 'stable': '#9E9E9E'}
        labels = {'rising': 'Rising (dQ/dt > 0)', 'falling': 'Falling (dQ/dt < 0)', 'stable': 'Stable'}

        for condition, color in colors.items():
            mask = conditions_sorted == condition
            if mask.any():
                fig.add_trace(go.Scattergl(
                    x=exceedance[mask], y=q_sorted[mask],
                    mode='markers', name=labels[condition],
                    marker=dict(color=color, size=3, opacity=0.6),
                    hovertemplate='Exceedance: %{x:.1f}%<br>Flow: %{y:,.0f} cfs<extra></extra>'
                ))

        fig.add_annotation(
            text="Koehler (2025) enhanced FDC: color = dQ/dt",
            xref="paper", yref="paper", x=0.98, y=0.02,
            showarrow=False, font=dict(size=9, color='gray'),
            xanchor='right', yanchor='bottom'
        )

    else:
        # Standard FDC
        fdc = flow_duration_curve(q)
        if fdc.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data for FDC", xref="paper", yref="paper",
                              x=0.5, y=0.5, showarrow=False)
            return fig

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fdc['exceedance_pct'], y=fdc['discharge'],
            mode='lines', name='FDC',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='Exceedance: %{x:.1f}%<br>Flow: %{y:,.0f} cfs<extra></extra>'
        ))

    # Add reference lines for key percentiles
    if not q.empty:
        for pct, label in [(10, 'Q10'), (50, 'Q50'), (90, 'Q90')]:
            q_val = np.percentile(q.values, 100 - pct)
            fig.add_hline(y=q_val, line_dash="dash", line_color="gray",
                         annotation_text=f"{label}: {q_val:,.0f}", annotation_position="right",
                         line_width=1, opacity=0.5)

    fig.update_layout(
        title=title,
        xaxis_title="Exceedance Probability (%)",
        yaxis_title="Discharge (cfs)",
        yaxis_type="log",
        xaxis=dict(range=[0, 100]),
        height=450,
        margin=dict(l=60, r=80, t=60, b=40),
    )

    return fig


def interactive_comparison(
    site_data: Dict[str, pd.DataFrame],
    discharge_col: str = 'Discharge_cfs',
    value_col_fallback: str = 'value',
    title: str = "Multi-Site Comparison",
) -> go.Figure:
    """
    Interactive multi-site comparison overlay with synchronized crosshair.

    Args:
        site_data: Dict mapping site labels to DataFrames
        discharge_col: Column name for discharge
        value_col_fallback: Fallback column name
        title: Plot title

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f']

    for i, (label, df) in enumerate(site_data.items()):
        if df is None or df.empty:
            continue

        col = discharge_col if discharge_col in df.columns else value_col_fallback
        if col not in df.columns:
            continue

        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col],
            mode='lines', name=label,
            line=dict(color=color, width=2),
            hovertemplate=f'{label}<br>%{{x|%Y-%m-%d}}<br>%{{y:,.0f}} cfs<extra></extra>'
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Discharge (cfs)",
        yaxis_type="log",
        height=450,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=60, b=40),
    )

    return fig


def raster_hydrograph(
    df_q: pd.DataFrame,
    discharge_col: str = 'Discharge_cfs',
    title: str = "Raster Hydrograph",
) -> go.Figure:
    """
    Raster hydrograph: x=day-of-year, y=year, color=flow.

    Entire station history at a glance - each pixel represents one day.
    Inspired by USGS hyswap raster hydrograph pattern.

    Args:
        df_q: Discharge DataFrame with datetime index
        discharge_col: Column name for discharge
        title: Plot title

    Returns:
        Plotly Figure
    """
    if df_q is None or df_q.empty or discharge_col not in df_q.columns:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        return fig

    q = df_q[discharge_col].dropna()
    if q.empty:
        fig = go.Figure()
        fig.add_annotation(text="No discharge data", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig

    # Build matrix: rows=years, cols=day-of-year
    df_work = pd.DataFrame({'q': q, 'year': q.index.year, 'doy': q.index.dayofyear})
    years = sorted(df_work['year'].unique())
    n_years = len(years)

    if n_years < 2:
        fig = go.Figure()
        fig.add_annotation(text="Need 2+ years for raster hydrograph",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    # Create matrix (log-scaled for better contrast)
    matrix = np.full((n_years, 366), np.nan)
    for i, year in enumerate(years):
        year_data = df_work[df_work['year'] == year]
        for _, row in year_data.iterrows():
            doy = int(row['doy']) - 1  # 0-indexed
            if 0 <= doy < 366:
                matrix[i, doy] = np.log10(max(row['q'], 0.01))

    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=list(range(1, 367)),
        y=years,
        colorscale='Viridis',
        colorbar=dict(title='log10(cfs)'),
        hovertemplate='Day %{x}<br>Year %{y}<br>Flow: 10^%{z:.1f} cfs<extra></extra>',
        zsmooth='best',
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(title="Day of Year", tickvals=month_starts, ticktext=month_labels),
        yaxis=dict(title="Year", autorange='reversed'),
        height=max(300, n_years * 15 + 100),
        margin=dict(l=60, r=20, t=60, b=40),
    )

    return fig


def percentile_bands_hydrograph(
    df_q: pd.DataFrame,
    discharge_col: str = 'Discharge_cfs',
    title: str = "Percentile Bands Hydrograph",
    current_year: int = None,
) -> go.Figure:
    """
    Percentile rainbow bands showing where current year sits relative to
    the historical record.

    Bands: 10th, 25th, 50th, 75th, 90th percentiles by day-of-year.
    Current year's data overlaid as a bold line.
    """
    if df_q is None or df_q.empty or discharge_col not in df_q.columns:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        return fig

    q = df_q[discharge_col].dropna()
    if current_year is None:
        current_year = q.index.max().year

    df_work = pd.DataFrame({'q': q, 'doy': q.index.dayofyear, 'year': q.index.year})

    hist = df_work[df_work['year'] != current_year]
    if len(hist) < 365:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient historical data for percentile bands",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    pcts = hist.groupby('doy')['q'].quantile([0.10, 0.25, 0.50, 0.75, 0.90]).unstack()
    pcts.columns = ['p10', 'p25', 'p50', 'p75', 'p90']

    doys = pcts.index.tolist()

    fig = go.Figure()

    # 10-90th band
    fig.add_trace(go.Scatter(
        x=doys + doys[::-1],
        y=pcts['p90'].tolist() + pcts['p10'].tolist()[::-1],
        fill='toself', fillcolor='rgba(100, 149, 237, 0.1)',
        line=dict(width=0), name='10-90th', hoverinfo='skip'
    ))

    # 25-75th band
    fig.add_trace(go.Scatter(
        x=doys + doys[::-1],
        y=pcts['p75'].tolist() + pcts['p25'].tolist()[::-1],
        fill='toself', fillcolor='rgba(100, 149, 237, 0.25)',
        line=dict(width=0), name='25-75th', hoverinfo='skip'
    ))

    # Median
    fig.add_trace(go.Scatter(
        x=doys, y=pcts['p50'].tolist(),
        mode='lines', name='Median',
        line=dict(color='rgba(100, 149, 237, 0.6)', width=1.5, dash='dash'),
        hovertemplate='Day %{x}<br>Median: %{y:,.0f} cfs<extra></extra>'
    ))

    # Current year
    current = df_work[df_work['year'] == current_year].sort_values('doy')
    if not current.empty:
        fig.add_trace(go.Scatter(
            x=current['doy'].tolist(), y=current['q'].tolist(),
            mode='lines', name=f'{current_year}',
            line=dict(color='#d62728', width=2.5),
            hovertemplate=f'{current_year} Day %{{x}}<br>%{{y:,.0f}} cfs<extra></extra>'
        ))

    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    fig.update_layout(
        title=title,
        xaxis=dict(title="Day of Year", tickvals=month_starts, ticktext=month_labels),
        yaxis=dict(title="Discharge (cfs)", type="log"),
        height=450,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=60, b=40),
    )

    return fig


def interactive_return_period(
    observed: pd.DataFrame,
    fitted: dict,
    return_period_table: pd.DataFrame = None,
    title: str = "Flood Frequency Analysis",
) -> go.Figure:
    """
    Interactive return period plot showing observed data and fitted distributions.

    Args:
        observed: DataFrame from get_plotting_positions() with return_period and flow_cfs
        fitted: Dict from fit_flood_frequency() mapping dist name to DistributionFit
        return_period_table: Optional DataFrame from estimate_return_periods() with CIs
        title: Plot title

    Returns:
        Plotly Figure with readable return period axis
    """
    fig = go.Figure()

    # Plot observed data (plotting positions)
    if observed is not None and not observed.empty:
        fig.add_trace(go.Scatter(
            x=observed['return_period'], y=observed['flow_cfs'],
            mode='markers', name='Observed',
            marker=dict(color='black', size=6, symbol='circle'),
            hovertemplate='T=%{x:.1f} yr<br>Q=%{y:,.0f} cfs<extra>Observed</extra>',
        ))

    # Plot fitted distribution curves
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    rp_range = np.linspace(1.01, 500, 300)

    for i, (name, fit) in enumerate(fitted.items()):
        quantiles = []
        for rp in rp_range:
            p = 1 / rp
            if name == 'lp3':
                from hydrology.analysis.frequency import _lp3_quantile
                q = _lp3_quantile(fit.params, p)
            elif name == 'gev':
                from scipy import stats as sp_stats
                q = sp_stats.genextreme.ppf(1 - p, *fit.params)
            elif name == 'gumbel':
                from scipy import stats as sp_stats
                q = sp_stats.gumbel_r.ppf(1 - p, *fit.params)
            elif name == 'lognormal':
                from scipy import stats as sp_stats
                q = sp_stats.lognorm.ppf(1 - p, *fit.params)
            elif name == 'pearson3':
                from scipy import stats as sp_stats
                q = sp_stats.pearson3.ppf(1 - p, *fit.params)
            else:
                continue
            quantiles.append(q)

        color = colors[i % len(colors)]
        aic_label = f" (AIC={fit.aic:.0f})" if np.isfinite(fit.aic) else ""
        fig.add_trace(go.Scatter(
            x=rp_range, y=quantiles,
            mode='lines', name=f'{fit.display_name}{aic_label}',
            line=dict(color=color, width=2),
            hovertemplate=f'{fit.display_name}<br>T=%{{x:.1f}} yr<br>Q=%{{y:,.0f}} cfs<extra></extra>',
        ))

    # Confidence interval band for best distribution
    if return_period_table is not None and not return_period_table.empty:
        rp_ci = return_period_table.dropna(subset=['lower_ci', 'upper_ci'])
        if not rp_ci.empty:
            fig.add_trace(go.Scatter(
                x=list(rp_ci['return_period']) + list(rp_ci['return_period'][::-1]),
                y=list(rp_ci['upper_ci']) + list(rp_ci['lower_ci'][::-1]),
                fill='toself', fillcolor='rgba(31, 119, 180, 0.12)',
                line=dict(width=0), name='95% CI',
                hoverinfo='skip',
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Return Period (years)",
        yaxis_title="Peak Discharge (cfs)",
        yaxis_type="log",
        xaxis=dict(
            type="linear",
            range=[0, 105],
            tickmode="array",
            tickvals=[2, 5, 10, 25, 50, 100],
            ticktext=["2", "5", "10", "25", "50", "100"],
        ),
        height=500,
        hovermode='closest',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=60, b=40),
    )

    # Add reference lines for common return periods
    for rp in [10, 50, 100]:
        fig.add_vline(x=rp, line_dash="dot", line_color="gray",
                     line_width=0.5, opacity=0.5,
                     annotation_text=f"{rp}-yr", annotation_position="top")

    return fig


def _lyne_hollick(Q: np.ndarray, alpha: float = 0.925) -> np.ndarray:
    """Apply Lyne-Hollick recursive digital filter to separate baseflow."""
    Q_f = np.zeros_like(Q, dtype=float)
    for t in range(1, len(Q)):
        Q_f[t] = alpha * Q_f[t - 1] + (1 + alpha) / 2 * (Q[t] - Q[t - 1])
        Q_f[t] = max(0, Q_f[t])
    return np.clip(Q - Q_f, 0, Q)


def baseflow_waterfall(
    df_upstream: pd.DataFrame,
    df_downstream: pd.DataFrame,
    discharge_col: str = 'Discharge_cfs',
    upstream_name: str = 'Upstream',
    downstream_name: str = 'Downstream',
    alpha: float = 0.925,
    title: str = "Baseflow Separation Waterfall",
) -> go.Figure:
    """
    Waterfall chart decomposing reach flow into components.

    Shows how upstream flow transforms into downstream flow through:
    - Upstream baseflow and quickflow
    - Reach gain or loss (the difference)
    - Downstream total

    The reach gain/loss quantifies net aquifer contribution in the reach.
    """
    empty = go.Figure()
    empty.add_annotation(text="Insufficient data for baseflow waterfall",
                         xref="paper", yref="paper", x=0.5, y=0.5,
                         showarrow=False, font=dict(size=14))

    if df_upstream is None or df_downstream is None:
        return empty
    if discharge_col not in df_upstream.columns or discharge_col not in df_downstream.columns:
        return empty

    # Align on common dates
    common_idx = df_upstream.index.intersection(df_downstream.index)
    if len(common_idx) < 30:
        return empty

    Q_up = df_upstream.loc[common_idx, discharge_col].dropna().values
    Q_dn = df_downstream.loc[common_idx, discharge_col].dropna().values
    min_len = min(len(Q_up), len(Q_dn))
    if min_len < 30:
        return empty
    Q_up = Q_up[:min_len]
    Q_dn = Q_dn[:min_len]
    dates = common_idx[:min_len]

    # Separate baseflow
    bf_up = _lyne_hollick(Q_up, alpha)
    qf_up = Q_up - bf_up
    bf_dn = _lyne_hollick(Q_dn, alpha)
    qf_dn = Q_dn - bf_dn

    # Compute means
    mean_up_bf = float(np.mean(bf_up))
    mean_up_qf = float(np.mean(qf_up))
    mean_up_total = mean_up_bf + mean_up_qf
    mean_dn_total = float(np.mean(bf_dn)) + float(np.mean(qf_dn))
    reach_delta = mean_dn_total - mean_up_total

    bfi_up = mean_up_bf / mean_up_total if mean_up_total > 0 else 0
    bfi_dn = float(np.mean(bf_dn)) / mean_dn_total if mean_dn_total > 0 else 0

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.45, 0.55],
        vertical_spacing=0.12,
        subplot_titles=["Flow Component Waterfall (Period Means)", "Monthly Baseflow Index"],
    )

    # --- Row 1: Waterfall ---
    labels = [
        f"{upstream_name[:25]}<br>Baseflow",
        f"{upstream_name[:25]}<br>Quickflow",
        "Reach<br>Gain/Loss",
        f"{downstream_name[:25]}<br>Total",
    ]
    values = [mean_up_bf, mean_up_qf, reach_delta, mean_dn_total]
    measures = ["absolute", "relative", "relative", "total"]

    fig.add_trace(
        go.Waterfall(
            x=labels, y=values, measure=measures,
            connector=dict(line=dict(color="#888", width=1)),
            increasing=dict(marker=dict(color="#2ca02c")),
            decreasing=dict(marker=dict(color="#d62728")),
            totals=dict(marker=dict(color="#ff7f0e")),
            texttemplate="%{y:,.0f}",
            textposition="outside",
            hovertemplate="%{x}<br>%{y:,.0f} cfs<extra></extra>",
        ),
        row=1, col=1,
    )

    # --- Row 2: Monthly BFI time series ---
    df_combined = pd.DataFrame({
        'bf_up': bf_up, 'Q_up': Q_up,
        'bf_dn': bf_dn, 'Q_dn': Q_dn,
    }, index=dates)

    monthly = df_combined.resample('ME').sum()
    monthly = monthly[monthly['Q_up'] > 0]
    monthly['bfi_up'] = (monthly['bf_up'] / monthly['Q_up']).clip(0, 1)
    monthly['bfi_dn'] = (monthly['bf_dn'] / monthly['Q_dn']).clip(0, 1)

    fig.add_trace(
        go.Scatter(
            x=monthly.index, y=monthly['bfi_up'],
            mode='lines', name=f'{upstream_name[:25]} BFI',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='%{x|%b %Y}<br>BFI: %{y:.2f}<extra></extra>',
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=monthly.index, y=monthly['bfi_dn'],
            mode='lines', name=f'{downstream_name[:25]} BFI',
            line=dict(color='#ff7f0e', width=2),
            hovertemplate='%{x|%b %Y}<br>BFI: %{y:.2f}<extra></extra>',
        ),
        row=2, col=1,
    )

    # Shade the gap between BFI curves
    fig.add_trace(
        go.Scatter(
            x=list(monthly.index) + list(monthly.index[::-1]),
            y=list(monthly['bfi_dn']) + list(monthly['bfi_up'][::-1]),
            fill='toself', fillcolor='rgba(44, 160, 44, 0.15)',
            line=dict(width=0), showlegend=False,
            hoverinfo='skip',
        ),
        row=2, col=1,
    )

    gain_loss_label = "gaining" if reach_delta >= 0 else "losing"
    annotation_text = (
        f"Reach is <b>{gain_loss_label}</b> {abs(reach_delta):,.0f} cfs on average | "
        f"BFI: {upstream_name[:20]} = {bfi_up:.2f}, {downstream_name[:20]} = {bfi_dn:.2f}"
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        height=650,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=80, b=40),
        annotations=[
            dict(
                text=annotation_text,
                xref="paper", yref="paper",
                x=0.5, y=-0.06, showarrow=False,
                font=dict(size=11, color="#666"),
            )
        ] + list(fig.layout.annotations),
    )

    fig.update_yaxes(title_text="Discharge (cfs)", row=1, col=1)
    fig.update_yaxes(title_text="Baseflow Index", range=[0, 1], row=2, col=1)

    return apply_hydro_theme(fig)


def apply_hydro_theme(fig: "go.Figure") -> "go.Figure":
    """Lightweight consistent theming for Plotly charts to match the dashboard premium visual language.

    Safe to call on any interactive fig returned from this module.
    Uses the same teal accent and dark surfaces as the Streamlit custom CSS.
    (Optional polish; callers can opt-in without behavior change.)
    """
    try:
        fig.update_layout(
            font=dict(family="Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif", size=12, color="#e7eef7"),
            paper_bgcolor="rgba(16, 28, 46, 0.85)",
            plot_bgcolor="rgba(10, 21, 36, 0.6)",
            margin=dict(l=50, r=20, t=60, b=40),
            hoverlabel=dict(bgcolor="#101c2e", bordercolor="#4ecdc4", font=dict(color="#e7eef7")),
        )
        # Subtle grid using the muted border tone
        fig.update_xaxes(gridcolor="rgba(118, 169, 192, 0.12)", zerolinecolor="rgba(118, 169, 192, 0.2)")
        fig.update_yaxes(gridcolor="rgba(118, 169, 192, 0.12)", zerolinecolor="rgba(118, 169, 192, 0.2)")
    except Exception:
        # Never break a chart for theme
        pass
    return fig
