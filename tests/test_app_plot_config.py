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


def test_grouped_plot_options_include_purpose_copy():
    grouped = plot_config.get_grouped_plot_options({"flow_duration": {}, "lag_correlation": {}})

    labels = {group["label"]: group for group in grouped}

    assert "Flow behavior" in labels
    assert "Climate linkage" in labels
    assert labels["Flow behavior"]["plots"] == ["flow_duration"]
    assert labels["Climate linkage"]["plots"] == ["lag_correlation"]
