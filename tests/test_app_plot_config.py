from pathlib import Path

from hydrology.app import plot_config


def test_plot_preset_resolves_only_available_plots():
    available = {
        "timeseries": {},
        "flow_duration": {},
        "flood_frequency": {},
    }

    plots = plot_config.resolve_plot_preset("Flood frequency", available)

    assert plots == ["timeseries", "flow_duration", "flood_frequency"]


def test_plot_preset_keeps_manual_mode_empty():
    assert plot_config.resolve_plot_preset("Manual selection", {"timeseries": {}}) == []


def test_plot_presets_have_card_metadata():
    for preset_name, preset in plot_config.PLOT_PRESETS.items():
        assert preset_name
        assert preset["description"]
        assert "intent" in preset
        assert "plots" in preset


def test_grouped_plot_options_cover_all_known_plots_once():
    grouped = plot_config.get_grouped_plot_options({plot: {} for plot in plot_config.ALL_PLOTS})
    grouped_plots = [
        plot
        for group in grouped
        for plot in group["plots"]
    ]

    assert sorted(grouped_plots) == sorted(plot_config.ALL_PLOTS)
    assert len(grouped_plots) == len(set(grouped_plots))


def test_single_site_plots_exclude_reach_only_plots():
    assert not set(plot_config.REACH_PLOTS) & set(plot_config.SINGLE_SITE_PLOTS)
    assert "timeseries" in plot_config.SINGLE_SITE_PLOTS
    assert "lag_correlation" in plot_config.SINGLE_SITE_PLOTS


def test_grouped_plot_options_include_purpose_copy():
    grouped = plot_config.get_grouped_plot_options({"flow_duration": {}, "lag_correlation": {}})

    labels = {group["label"]: group for group in grouped}

    assert "Flow behavior" in labels
    assert "Climate linkage" in labels
    assert labels["Flow behavior"]["plots"] == ["flow_duration"]
    assert labels["Climate linkage"]["plots"] == ["lag_correlation"]


def test_describe_selected_plots_groups_counts():
    summary = plot_config.describe_selected_plots([
        "timeseries",
        "flood_frequency",
        "lag_correlation",
    ])

    assert summary == "3 selected: Flow behavior, Extremes / frequency, Climate linkage"


def test_catalog_plot_options_cover_single_site_plots_once():
    options = plot_config.catalog_plot_options({plot: {} for plot in plot_config.SINGLE_SITE_PLOTS})
    keys = [key for _, key in options]

    assert len(keys) == len(set(keys))
    assert set(keys) == set(plot_config.SINGLE_SITE_PLOTS)
    assert all(" · " in label for label, _ in options)


def test_plot_selector_is_one_catalog():
    text = Path(__file__).resolve().parents[1].joinpath("hydrology/app/plot_config.py").read_text(encoding="utf-8")

    assert "catalog_plot_options" in text
    assert "Add from full plot catalog" not in text
    assert "plot_preset_choice" in text


def test_resolve_generated_plots_keeps_interactive_plot_choices():
    selected = ["timeseries", "flow_duration", "monthly_boxplot", "annual_trend"]

    assert plot_config.resolve_generated_plots(selected) == selected
