"""Schemas for EDA summary and correlation."""
from __future__ import annotations

from pydantic import ConfigDict

from app.schemas.common import ApiModel


class FeatureSummary(ApiModel):
    """Descriptive statistics for one numeric feature."""

    count: int
    null_count: int
    mean: float | None
    std: float | None
    min: float | None
    median: float | None
    max: float | None
    constant: bool


class EdaSummaryResponse(ApiModel):
    """Per-feature statistics (target excluded)."""

    features: dict[str, FeatureSummary]
    n_features: int
    notes: list[str]

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "features": {
                    "tenure": {
                        "count": 7043, "null_count": 0, "mean": 32.37, "std": 24.56,
                        "min": 0.0, "median": 29.0, "max": 72.0, "constant": False,
                    }
                },
                "n_features": 1,
                "notes": [],
            }
        },
    )


class MulticollinearityFlag(ApiModel):
    """A feature pair whose |r| exceeds the threshold."""

    feature_a: str
    feature_b: str
    r: float


class CorrelationResponse(ApiModel):
    """Pearson correlation matrix across numeric features."""

    columns: list[str]
    matrix: list[list[float | None]]
    multicollinearity_flags: list[MulticollinearityFlag]
    threshold: float
    notes: list[str]

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "columns": ["tenure", "total_charges"],
                "matrix": [[1.0, 0.88], [0.88, 1.0]],
                "multicollinearity_flags": [
                    {"feature_a": "tenure", "feature_b": "total_charges", "r": 0.88}
                ],
                "threshold": 0.85,
                "notes": [],
            }
        },
    )
