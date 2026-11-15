"""Division and rounding helpers that never raise or emit NaN."""
from __future__ import annotations

import math


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide, returning ``default`` for a zero or non-finite denominator/result."""
    if denominator == 0 or not math.isfinite(denominator):
        return default
    result = numerator / denominator
    return result if math.isfinite(result) else default


def round_or_none(value: float | None, digits: int = 6) -> float | None:
    """Round a float, mapping NaN/inf/None to ``None``."""
    if value is None:
        return None
    number = float(value)
    return round(number, digits) if math.isfinite(number) else None
