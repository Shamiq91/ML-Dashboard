"""Evenly spaced downsampling that always keeps the first and last point."""
from __future__ import annotations

import numpy as np


def select_indices(length: int, max_points: int = 40) -> list[int]:
    """Return at most ``max_points`` evenly spaced indices including 0 and ``length - 1``."""
    if length <= 0:
        return []
    if length <= max_points:
        return list(range(length))
    if max_points < 2:
        return [0]
    raw = np.linspace(0, length - 1, max_points)
    return sorted({int(round(x)) for x in raw})

