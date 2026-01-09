"""
Plot composer for creating multi-panel figures.

Arranges individual plots into various layouts (vertical, quad, grid, etc.)
and saves them as combined figures.

Usage:
    from hydrology.visualization import create_multi_plot, PlotLayout

    # Vertical layout (all plots stacked)
    fig = create_multi_plot(
        plots=['anomaly', 'hexbin_temp', 'lagged_precip', 'timeseries'],
        layout=PlotLayout.VERTICAL,
        data={'df_merged': df, 'df_q': df_q, 'analysis_results': results},
        site_id='12422500',
        title='Spokane River Analysis'
    )

    # Quad layout (2x2 grid)
    fig = create_multi_plot(
        plots=['anomaly', 'hexbin_temp', 'lagged_precip', 'flow_duration'],
        layout=PlotLayout.QUAD,
        ...
    )
"""

from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import textwrap

from .plots import AVAILABLE_PLOTS
from ..core.logging_setup import get_logger
from ..core.paths import get_site_plot_dir

logger = get_logger(__name__)


class PlotLayout(Enum):
    """Available plot layouts."""
    VERTICAL = "vertical"      # All plots stacked vertically (1 column, N rows)
    QUAD = "quad"              # 2x2 grid (exactly 4 plots)
    GRID_2x3 = "grid_2x3"      # 2 columns, 3 rows (up to 6 plots)
    GRID_3x2 = "grid_3x2"      # 3 columns, 2 rows (up to 6 plots)
    GRID_2x5 = "grid_2x5"      # 2 columns, 5 rows (up to 10 plots)
    GRID_5x2 = "grid_5x2"      # 5 columns, 2 rows (up to 10 plots)
    HORIZONTAL = "horizontal"   # All plots side-by-side (N columns, 1 row)
    AUTO = "auto"              # Automatically choose best layout


def _determine_auto_layout(n_plots: int) -> Tuple[int, int]:
    """
    Determine optimal grid layout for N plots.

    Args:
        n_plots: Number of plots

    Returns:
        Tuple of (n_rows, n_cols)
    """
    if n_plots == 1:
        return (1, 1)
    elif n_plots == 2:
        return (2, 1)
    elif n_plots == 3:
        return (3, 1)
    elif n_plots == 4:
        return (2, 2)
    elif n_plots <= 6:
        return (3, 2)
    elif n_plots <= 9:
        return (3, 3)
    elif n_plots == 10:
        # 5 rows x 2 columns for 10 plots
        return (5, 2)
    else:
        # For more than 10 plots, use a wider grid
        import math
        n_cols = min(3, n_plots)  # Max 3 columns
        n_rows = math.ceil(n_plots / n_cols)
        return (n_rows, n_cols)


def create_multi_plot(
    plots: List[str],
    layout: PlotLayout = PlotLayout.VERTICAL,
    data: Dict[str, Any] = None,
    site_id: Optional[str] = None,
    title: Optional[str] = None,
    config: Dict[str, Any] = None,
    save_path: Optional[Path] = None,
    dpi: int = 150,
    figsize: Optional[Tuple[float, float]] = None
) -> plt.Figure:
    """
    Create a multi-panel plot figure.

    Args:
        plots: List of plot names (from AVAILABLE_PLOTS keys)
        layout: Layout style (VERTICAL, QUAD, etc.)
        data: Dict containing data for plots (df_q, df_merged, analysis_results, etc.)
        site_id: Optional site ID for title and save path
        title: Optional custom title (will be auto-generated if None)
        config: Optional configuration dict passed to plot functions
        save_path: Optional custom save path (auto-generated if None)
        dpi: DPI for saved figure
        figsize: Optional figure size (width, height) in inches

    Returns:
        Matplotlib figure object

    Example:
        >>> fig = create_multi_plot(
        ...     plots=['anomaly', 'hexbin_temp', 'timeseries'],
        ...     layout=PlotLayout.VERTICAL,
        ...     data={'df_merged': df, 'df_q': df_q, 'analysis_results': results},
        ...     site_id='12422500'
        ... )
    """
    if not plots:
        logger.error("No plots specified")
        return None

    if data is None:
        data = {}

    # Validate plot names
    invalid_plots = [p for p in plots if p not in AVAILABLE_PLOTS]
    if invalid_plots:
        logger.error(f"Invalid plot names: {invalid_plots}. Available: {list(AVAILABLE_PLOTS.keys())}")
        return None

    # Determine grid layout
    if layout == PlotLayout.VERTICAL:
        n_rows, n_cols = len(plots), 1
    elif layout == PlotLayout.QUAD:
        if len(plots) != 4:
            logger.warning(f"QUAD layout expects 4 plots, got {len(plots)}. Using AUTO layout instead.")
            n_rows, n_cols = _determine_auto_layout(len(plots))
        else:
            n_rows, n_cols = 2, 2
    elif layout == PlotLayout.GRID_2x3:
        n_rows, n_cols = 3, 2
    elif layout == PlotLayout.GRID_3x2:
        n_rows, n_cols = 2, 3
    elif layout == PlotLayout.GRID_2x5:
        n_rows, n_cols = 5, 2
    elif layout == PlotLayout.GRID_5x2:
        n_rows, n_cols = 2, 5
    elif layout == PlotLayout.HORIZONTAL:
        n_rows, n_cols = 1, len(plots)
    elif layout == PlotLayout.AUTO:
        n_rows, n_cols = _determine_auto_layout(len(plots))
    else:
        logger.error(f"Unknown layout: {layout}")
        return None

    # Determine figure size
    if figsize is None:
        if layout == PlotLayout.VERTICAL:
            # Limit height per plot for vertical stacking
            height_per_plot = min(6, 30 / len(plots))  # Max 30 inches total height
            figsize = (12, height_per_plot * len(plots))
        elif layout == PlotLayout.HORIZONTAL:
            figsize = (8 * len(plots), 6)
        elif layout == PlotLayout.QUAD:
            figsize = (16, 14)
        elif layout == PlotLayout.GRID_2x5 or (n_rows == 5 and n_cols == 2):
            figsize = (16, 20)  # 2 columns x 5 rows
        elif layout == PlotLayout.GRID_5x2 or (n_rows == 2 and n_cols == 5):
            figsize = (40, 10)  # 5 columns x 2 rows
        else:
            # General grid: 8 inches per column, 5 inches per row
            figsize = (8 * n_cols, 5 * n_rows)

    # Create figure and axes with better spacing
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # Flatten axes array for easy indexing
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    # Generate title
    if title is None:
        plot_names = ', '.join([AVAILABLE_PLOTS[p]['description'] for p in plots])
        if site_id:
            title = f"Site {site_id} Analysis"
        else:
            title = "Hydrological Analysis"

        # Add data period if available
        df_merged = data.get('df_merged')
        df_q = data.get('df_q')
        if df_merged is not None and not df_merged.empty:
            start = df_merged.index.min().strftime('%Y-%m-%d')
            end = df_merged.index.max().strftime('%Y-%m-%d')
            title += f"\nData: {start} to {end}"
        elif df_q is not None and not df_q.empty:
            start = df_q.index.min().strftime('%Y-%m-%d')
            end = df_q.index.max().strftime('%Y-%m-%d')
            title += f"\nData: {start} to {end}"

    wrapped_title = "\n".join(textwrap.wrap(title, width=80))
    fig.suptitle(wrapped_title, fontsize=14, fontweight='bold')

    # Create each plot
    for i, plot_name in enumerate(plots):
        if i >= len(axes):
            logger.warning(f"More plots than axes. Skipping plot: {plot_name}")
            break

        ax = axes[i]
        plot_info = AVAILABLE_PLOTS[plot_name]
        plot_func = plot_info['function']

        logger.info(f"Creating plot {i+1}/{len(plots)}: {plot_name}")

        try:
            # Call plot function with appropriate data
            plot_func(ax, **data, config=config)
        except Exception as e:
            logger.error(f"Error creating plot '{plot_name}': {e}")
            ax.text(0.5, 0.5, f"Error creating plot:\n{plot_name}\n{str(e)}",
                   ha='center', va='center', transform=ax.transAxes,
                   bbox=dict(boxstyle='round,pad=0.5', fc='red', alpha=0.3))

    # Hide unused axes
    for i in range(len(plots), len(axes)):
        axes[i].set_visible(False)

    # Save figure if save_path provided or site_id given
    if save_path is None and site_id:
        # Auto-generate save path
        plot_dir = get_site_plot_dir(site_id, 'multi_plot')
        layout_name = layout.value if isinstance(layout, PlotLayout) else str(layout)
        plot_filename = f"USGS_{site_id}_{layout_name}_{'_'.join(plots[:3])}.png"
        save_path = plot_dir / plot_filename

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Multi-plot saved: {save_path}")

    return fig


def create_comparison_plot(
    plot_name: str,
    site_data_list: List[Dict[str, Any]],
    layout: PlotLayout = PlotLayout.GRID_2x3,
    config: Dict[str, Any] = None,
    save_path: Optional[Path] = None,
    dpi: int = 150
) -> plt.Figure:
    """
    Create a comparison figure showing the same plot for multiple sites.

    Args:
        plot_name: Name of plot to create (from AVAILABLE_PLOTS)
        site_data_list: List of dicts, each containing:
                       {'site_id': '12422500', 'data': {...}, 'title': 'Site Name'}
        layout: Layout for arranging site comparisons
        config: Optional configuration
        save_path: Optional save path
        dpi: DPI for saved figure

    Returns:
        Matplotlib figure

    Example:
        >>> sites = [
        ...     {'site_id': '12422500', 'data': {...}, 'title': 'Spokane River'},
        ...     {'site_id': '12424000', 'data': {...}, 'title': 'Hangman Creek'},
        ... ]
        >>> fig = create_comparison_plot('timeseries', sites, layout=PlotLayout.GRID_2x3)
    """
    if plot_name not in AVAILABLE_PLOTS:
        logger.error(f"Invalid plot name: {plot_name}")
        return None

    n_sites = len(site_data_list)
    plot_info = AVAILABLE_PLOTS[plot_name]
    plot_func = plot_info['function']

    # Determine layout
    if layout == PlotLayout.AUTO:
        n_rows, n_cols = _determine_auto_layout(n_sites)
    elif layout == PlotLayout.VERTICAL:
        n_rows, n_cols = n_sites, 1
    elif layout == PlotLayout.HORIZONTAL:
        n_rows, n_cols = 1, n_sites
    elif layout == PlotLayout.GRID_2x3:
        n_rows, n_cols = 3, 2
    elif layout == PlotLayout.GRID_3x2:
        n_rows, n_cols = 2, 3
    else:
        n_rows, n_cols = _determine_auto_layout(n_sites)

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows), constrained_layout=True)

    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    fig.suptitle(f"Multi-Site Comparison: {plot_info['description']}",
                fontsize=14, fontweight='bold')

    # Create plot for each site
    for i, site_info in enumerate(site_data_list):
        if i >= len(axes):
            break

        ax = axes[i]
        site_id = site_info.get('site_id', f'Site {i+1}')
        data = site_info.get('data', {})
        site_title = site_info.get('title', site_id)

        logger.info(f"Creating comparison plot {i+1}/{n_sites}: {site_id}")

        try:
            plot_func(ax, **data, config=config)
            # Override title with site-specific title
            ax.set_title(site_title)
        except Exception as e:
            logger.error(f"Error creating plot for site {site_id}: {e}")
            ax.text(0.5, 0.5, f"Error:\n{str(e)}", ha='center', va='center',
                   transform=ax.transAxes, bbox=dict(boxstyle='round', fc='red', alpha=0.3))

    # Hide unused axes
    for i in range(n_sites, len(axes)):
        axes[i].set_visible(False)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Comparison plot saved: {save_path}")

    return fig
