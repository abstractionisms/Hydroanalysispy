import pandas as pd

from hydrology.app.interpretation import (
    describe_standardized_index,
    summarize_flow_context,
    summarize_recommendations,
)


def test_summarize_flow_context_includes_current_and_record_depth():
    index = pd.date_range("2010-01-01", periods=365 * 12, freq="D")
    df = pd.DataFrame({"Discharge_cfs": range(1, len(index) + 1)}, index=index)

    cards = summarize_flow_context(df)

    assert cards[0].title == "Current Flow"
    assert cards[0].value.endswith("cfs")
    assert any(card.title == "Record Depth" and "12.0" in card.value for card in cards)
    assert all(getattr(card, "help", "") for card in cards)


def test_summarize_flow_context_handles_empty_data():
    cards = summarize_flow_context(pd.DataFrame())

    assert cards[0].state == "blocked"
    assert cards[0].value == "No data"


def test_summarize_recommendations_reflects_available_inputs():
    cards = summarize_recommendations(has_stage=True, has_climate=True, record_years=11)

    values = {card.value for card in cards}
    assert "Use SPI" in values
    assert "Available" in values
    assert "On demand" in values


def test_describe_standardized_index_explains_dry_and_normal_values():
    dry_label, dry_body = describe_standardized_index(-1.5)
    normal_label, normal_body = describe_standardized_index(0.2)

    assert "dry" in dry_label.lower()
    assert "drier" in dry_body
    assert normal_label == "Near normal"
    assert "historical middle range" in normal_body


def test_build_plot_analysis_report_covers_core_sections():
    from hydrology.app.interpretation import build_plot_analysis_report

    index = pd.date_range("2015-01-01", periods=365 * 5, freq="D")
    # Seasonal signal: higher in spring
    q = []
    for ts in index:
        base = 50 + 200 * (1 if ts.month in (3, 4, 5) else 0.2)
        q.append(base)
    df = pd.DataFrame({"Discharge_cfs": q}, index=index)
    df["Gage_Height_ft"] = 2 + df["Discharge_cfs"] ** 0.3 / 10
    merged = df.copy()
    merged["Precip_mm"] = 1.0
    merged["Temp_C"] = 10.0

    report = build_plot_analysis_report(
        site_id="14018500",
        site_desc="WALLA WALLA RIVER NEAR TOUCHET, WA",
        plot_keys=["timeseries", "flow_duration", "rating_curve"],
        df_q=df,
        df_merged=merged,
    )

    assert "14018500" in report
    assert "Flow duration" in report
    assert "Seasonal pattern" in report
    assert "Climate" in report
    assert "Plots generated" in report
    assert "Metric relevance" in report
    assert "Q50" in report or "Median" in report


def test_metric_relevance_table_has_core_metrics():
    from hydrology.app.interpretation import metric_relevance_table, metric_keys_for_plots

    rows = metric_relevance_table(["q10", "q90", "mean_flow"])
    assert len(rows) == 3
    assert {r["Metric"] for r in rows} == {
        "Q10 (high flow)",
        "Q90 (low flow)",
        "Mean flow",
    }
    keys = metric_keys_for_plots(["flow_duration", "rating_curve"])
    assert "rating_r2" in keys
    assert "q90" in keys


def test_dynamic_metric_relevance_uses_site_hydrology():
    from hydrology.app.interpretation import (
        compute_hydrologic_profile,
        dynamic_metric_relevance,
        format_metric_relevance_markdown,
        metric_help_text,
    )

    index = pd.date_range("2010-01-01", periods=365 * 8, freq="D")
    q = []
    for ts in index:
        # Flashy spring peaks vs low summer baseflow
        if ts.month in (3, 4, 5):
            q.append(800 + (ts.day % 20) * 40)
        elif ts.month in (7, 8):
            q.append(25 + (ts.day % 5))
        else:
            q.append(120 + (ts.day % 10) * 5)
    df = pd.DataFrame({"Discharge_cfs": q}, index=index)
    df["Gage_Height_ft"] = 1.5 + (df["Discharge_cfs"] / 50) ** 0.4
    merged = df.copy()
    merged["Precip_mm"] = 0.8
    merged["Temp_C"] = 9.0

    profile = compute_hydrologic_profile(df, merged)
    assert profile["ok"]
    assert profile["q10"] > profile["q50"] > profile["q90"]
    assert profile["regime"] in {"flashy", "seasonal", "steady"}

    rows = dynamic_metric_relevance(
        df, merged, metric_keys=["mean_flow", "q10", "q50", "q90", "spi_sri", "rating_r2"]
    )
    assert len(rows) == 6
    why = " ".join(r["Why it matters here"] for r in rows)
    assert f"{profile['q50']:,.0f}" in why or "Median" in why or "cfs" in why
    assert "This site/period" in rows[0]
    assert rows[0]["This site/period"] != "—"
    assert any(r["Regime"] == profile["regime_plain"] for r in rows)

    md = format_metric_relevance_markdown(
        ["q50", "q90"], df_q=df, df_merged=merged
    )
    assert "regime" in md.lower()
    assert str(int(profile["q50"])) in md.replace(",", "") or f"{profile['q50']:,.0f}" in md

    tip = metric_help_text("q10", df, merged)
    assert "cfs" in tip.lower() or "Q10" in tip
    assert tip != ""
