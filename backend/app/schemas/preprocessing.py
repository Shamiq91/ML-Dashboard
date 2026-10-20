"""Schemas for missing-value handling and class balance."""
from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from app.schemas.common import ApiModel


class MissingColumn(ApiModel):
    """Missing-value stats for one column."""

    count: int
    percent: float


class MissingResponse(ApiModel):
    """Missing-value scan across the working DataFrame."""

    columns: dict[str, MissingColumn]
    rows: int
    total_missing: int
    total_cells: int
    total_percent: float
    has_missing: bool

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "columns": {"total_charges": {"count": 11, "percent": 0.1562}},
                "rows": 7043,
                "total_missing": 11,
                "total_cells": 147903,
                "total_percent": 0.0074,
                "has_missing": True,
            }
        },
    )


class CleanRequest(ApiModel):
    """Interactive cleaning request."""

    strategy: Literal["drop", "impute"]
    method: Literal["mean", "median", "mode", "knn", "interpolate"] | None = None
    columns: list[str] | None = None
    knn_neighbors: int = Field(default=5, ge=1, le=100)

    @model_validator(mode="after")
    def _method_required_for_impute(self) -> "CleanRequest":
        if self.strategy == "impute" and self.method is None:
            raise ValueError("'method' is required when strategy is 'impute'")
        return self

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {"strategy": "impute", "method": "median", "columns": None, "knn_neighbors": 5}
        },
    )


class CleanResponse(ApiModel):
    """Before/after report of a cleaning operation."""

    strategy: str
    method: str | None
    rows_before: int
    rows_after: int
    missing_before: int
    missing_after: int
    missing_before_by_column: dict[str, int]
    missing_after_by_column: dict[str, int]
    methods_applied: dict[str, str]
    fallbacks: list[str]
    warnings: list[str]
    stage: str

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "strategy": "impute",
                "method": "median",
                "rows_before": 7043,
                "rows_after": 7043,
                "missing_before": 11,
                "missing_after": 0,
                "missing_before_by_column": {"total_charges": 11},
                "missing_after_by_column": {"total_charges": 0},
                "methods_applied": {"total_charges": "median"},
                "fallbacks": [],
                "warnings": [],
                "stage": "CLEANED",
            }
        },
    )


class ResetResponse(ApiModel):
    """Result of undoing cleaning."""

    rows: int
    stage: str


class Histogram(ApiModel):
    """Equal-width histogram: ``len(edges) == len(counts) + 1``."""

    edges: list[float]
    counts: list[int]


class TargetSummaryResponse(ApiModel):
    """Distribution of a continuous (regression) target."""

    target: str
    count: int
    n_unique: int
    mean: float
    std: float
    min: float
    q1: float
    median: float
    q3: float
    max: float
    skew: float
    histogram: Histogram


class ClassShare(ApiModel):
    """One class in the target distribution."""

    label: str
    count: int
    ratio: float


class ClassBalanceResponse(ApiModel):
    """Target class distribution and imbalance verdict."""

    target: str
    total: int
    classes: list[ClassShare]
    minority_class: str | None
    minority_ratio: float
    imbalance_ratio: float
    imbalance_warning: bool
    severity: Literal["STRICT", "NONE"]
    single_class: bool
    message: str

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "target": "churn",
                "total": 7043,
                "classes": [
                    {"label": "No", "count": 5174, "ratio": 0.7347},
                    {"label": "Yes", "count": 1869, "ratio": 0.2653},
                ],
                "minority_class": "Yes",
                "minority_ratio": 0.2653,
                "imbalance_ratio": 2.7683,
                "imbalance_warning": True,
                "severity": "STRICT",
                "single_class": False,
                "message": "Minority class 'Yes' is 26.53% of rows (< 30%).",
            }
        },
    )
