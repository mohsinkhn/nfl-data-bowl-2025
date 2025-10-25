"""Parsing helpers for NFL tracking data."""

from __future__ import annotations

from typing import Optional, Union


def parse_height_to_inches(height_value: Optional[Union[str, int, float]]) -> float:
    """Convert height representations to inches.

    Accepts strings in ``6-2`` format, plain numeric strings, or numeric inputs.
    Returns 0.0 when parsing fails.
    """
    if height_value is None:
        return 0.0
    if isinstance(height_value, (int, float)):
        return float(height_value)
    if isinstance(height_value, str):
        if "-" in height_value:
            try:
                feet, inches = height_value.split("-")
                return float(feet) * 12.0 + float(inches)
            except ValueError:
                return 0.0
        try:
            return float(height_value)
        except ValueError:
            return 0.0
    return 0.0
