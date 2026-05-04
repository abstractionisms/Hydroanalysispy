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
