import numpy as np

from hydrology.analysis.frequency import estimate_return_periods, flood_frequency_diagnostics


def test_estimate_return_periods_is_reproducible_with_seed():
    peaks = np.array([100, 120, 130, 150, 180, 220, 260, 300, 360, 420, 500, 620])

    first = estimate_return_periods(peaks, periods=[10, 50], distribution="lp3", random_seed=42)
    second = estimate_return_periods(peaks, periods=[10, 50], distribution="lp3", random_seed=42)

    assert first[["lower_ci", "upper_ci"]].equals(second[["lower_ci", "upper_ci"]])


def test_flood_frequency_diagnostics_returns_plotting_table():
    peaks = np.array([100, 120, 130, 150, 180, 220, 260, 300, 360, 420, 500, 620])

    diagnostics = flood_frequency_diagnostics(peaks, distribution="lp3")

    expected = {"observed_flow_cfs", "fitted_flow_cfs", "exceedance_prob", "return_period"}
    assert expected.issubset(diagnostics.columns)
    assert len(diagnostics) == len(peaks)


def test_single_analysis_imports_frequency_diagnostics():
    from hydrology.app.page_modules.single_analysis import _format_frequency_diagnostics

    diagnostics = flood_frequency_diagnostics(
        np.array([100, 120, 130, 150, 180, 220, 260, 300, 360, 420, 500, 620]),
        distribution="lp3",
    )

    formatted = _format_frequency_diagnostics(diagnostics)

    assert {"Observed flow", "Fitted flow", "Return period"}.issubset(formatted.columns)


def test_interactive_return_period_uses_readable_linear_x_axis():
    from hydrology.analysis.frequency import fit_flood_frequency, get_plotting_positions
    from hydrology.visualization.interactive import interactive_return_period

    peaks = np.array([100, 120, 130, 150, 180, 220, 260, 300, 360, 420, 500, 620])
    observed = get_plotting_positions(peaks)
    fits = fit_flood_frequency(peaks)
    rp_table = estimate_return_periods(peaks, periods=[2, 5, 10, 25, 50, 100], random_seed=42)

    fig = interactive_return_period(observed, fits, rp_table)

    assert fig.layout.xaxis.type in (None, "linear")
    assert fig.layout.yaxis.type == "log"
    assert list(fig.layout.xaxis.tickvals) == [2, 5, 10, 25, 50, 100]
