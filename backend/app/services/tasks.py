"""Task kinds shared across the pipeline.

``classification`` predicts a category; ``regression`` predicts a number.
"""
from __future__ import annotations

REGRESSION_TASKS = frozenset({"regression"})


def is_regression(task: str) -> bool:
    """True for the numeric-target task."""
    return task in REGRESSION_TASKS
