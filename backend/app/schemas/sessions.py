"""Schemas for ingestion and session state."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import ConfigDict, Field

from app.schemas.common import ApiModel

ColumnKind = Literal["numeric", "categorical", "datetime", "boolean"]
StageName = Literal["UPLOADED", "TARGET_SET", "CLEANED", "TRAINED"]


class UploadResponse(ApiModel):
    """Result of a successful file upload."""

    session_id: str
    filename: str
    total_rows: int
    total_columns: int
    memory_usage_mb: float
    dtypes: dict[str, str]
    column_kinds: dict[str, ColumnKind]
    preview: list[dict[str, Any]]

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "session_id": "3f0c1c9e-5d0a-4b50-9d53-6a2f0f6c1d11",
                "filename": "churn.csv",
                "total_rows": 7043,
                "total_columns": 21,
                "memory_usage_mb": 1.134,
                "dtypes": {"tenure": "int64", "churn": "str"},
                "column_kinds": {"tenure": "numeric", "churn": "categorical"},
                "preview": [{"tenure": 1, "churn": "No"}],
            }
        },
    )


class TargetRequest(ApiModel):
    """Choose the prediction target column."""

    target_column: str = Field(min_length=1)
    task: Literal["auto", "classification", "regression"] = Field(
        default="auto",
        description="'auto' picks regression for a numeric target with many distinct values, else classification.",
    )

    model_config = ConfigDict(
        protected_namespaces=(), json_schema_extra={"example": {"target_column": "churn", "task": "auto"}}
    )


class TargetResponse(ApiModel):
    """Description of the selected target."""

    session_id: str
    target_column: str
    dtype: str
    task: Literal["classification", "regression"] = "classification"
    kind: Literal["binary", "multiclass", "continuous", "single_class"]
    n_classes: int
    warnings: list[str]
    stage: StageName

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "session_id": "3f0c1c9e",
                "target_column": "churn",
                "dtype": "str",
                "kind": "binary",
                "n_classes": 2,
                "warnings": [],
                "stage": "TARGET_SET",
            }
        },
    )


class SessionInfo(ApiModel):
    """Current pipeline stage and metadata."""

    session_id: str
    filename: str
    stage: StageName
    target_column: str | None
    task: Literal["classification", "regression"] = "classification"
    rows: int
    columns: int
    raw_rows: int
    has_missing: bool
    trained_models: list[str]
    created_at: str
    expires_in_seconds: int

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "session_id": "3f0c1c9e",
                "filename": "churn.csv",
                "stage": "CLEANED",
                "target_column": "churn",
                "rows": 7043,
                "columns": 21,
                "raw_rows": 7043,
                "has_missing": False,
                "trained_models": ["xgboost"],
                "created_at": "2026-01-01T12:00:00Z",
                "expires_in_seconds": 7100,
            }
        },
    )


class DeleteResponse(ApiModel):
    """Acknowledgement of a deleted session."""

    session_id: str
    deleted: bool = True
