"""Validation helpers for case studies and tests."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationResult:
    """Result from a simple PASS/FLAG case-study validation check."""

    metric: str
    status: str
    value: float
    expected: float | str
    relative_error: float | None = None
    message: str = ""


def validate_relative_error(
    metric: str,
    observed: float,
    expected: float,
    tolerance: float,
) -> ValidationResult:
    """Validate a value against a reference using relative error."""
    relative_error = (observed - expected) / expected if expected else float("inf")
    status = "PASS" if abs(relative_error) <= tolerance else "FLAG"
    return ValidationResult(
        metric=metric,
        status=status,
        value=observed,
        expected=expected,
        relative_error=relative_error,
        message=(
            f"{metric}: {observed:g} vs {expected:g}, "
            f"relative error {relative_error:.3f}, tolerance {tolerance:.3f}"
        ),
    )


def validate_range(metric: str, value: float, lower: float, upper: float) -> ValidationResult:
    """Validate a value against an inclusive expected range."""
    status = "PASS" if lower <= value <= upper else "FLAG"
    return ValidationResult(
        metric=metric,
        status=status,
        value=value,
        expected=f"{lower:g} to {upper:g}",
        message=f"{metric}: {value:g}, expected {lower:g} to {upper:g}",
    )
