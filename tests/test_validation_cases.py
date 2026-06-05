from hydrology.analysis.validation import validate_range, validate_relative_error


def test_validate_relative_error_passes_within_tolerance():
    result = validate_relative_error("100yr flood", observed=443000, expected=475000, tolerance=0.10)

    assert result.status == "PASS"
    assert abs(result.relative_error) < 0.10


def test_validate_range_flags_out_of_range():
    result = validate_range("BFI", value=0.95, lower=0.55, upper=0.85)

    assert result.status == "FLAG"
