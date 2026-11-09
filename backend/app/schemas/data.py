"""Schemas for the backend-only transform operations."""
from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field

from app.schemas.common import ApiModel


class DropColumns(ApiModel):
    """Remove columns."""

    op: Literal["drop_columns"]
    columns: list[str] = Field(min_length=1)


class DropMissingColumns(ApiModel):
    """Remove columns whose share of missing values exceeds ``threshold``."""

    op: Literal["drop_missing_columns"]
    threshold: float = Field(default=0.5, ge=0, le=1)


class DropConstantColumns(ApiModel):
    """Remove columns with a single value."""

    op: Literal["drop_constant_columns"]


class DropDuplicates(ApiModel):
    """Remove duplicate rows (optionally judged on a subset of columns)."""

    op: Literal["drop_duplicates"]
    subset: list[str] | None = None


class StripWhitespace(ApiModel):
    """Trim leading/trailing spaces in text columns."""

    op: Literal["strip_whitespace"]
    columns: list[str] | None = None


class FillConstant(ApiModel):
    """Fill missing values with a constant."""

    op: Literal["fill_constant"]
    columns: list[str] = Field(min_length=1)
    value: str | float | int


class Cast(ApiModel):
    """Convert a column's type."""

    op: Literal["cast"]
    column: str
    dtype: Literal["numeric", "category", "datetime", "string"]


class LogTransform(ApiModel):
    """``log1p`` of non-negative numeric columns."""

    op: Literal["log_transform"]
    columns: list[str] = Field(min_length=1)


class Clip(ApiModel):
    """Clip a numeric column to fixed bounds."""

    op: Literal["clip"]
    column: str
    lower: float | None = None
    upper: float | None = None


class Outliers(ApiModel):
    """Clip or drop outliers (IQR fences or z-score)."""

    op: Literal["outliers"]
    columns: list[str] | None = None
    method: Literal["iqr", "zscore"] = "iqr"
    factor: float = Field(default=1.5, gt=0, le=10)
    action: Literal["clip", "drop"] = "clip"


class FilterRows(ApiModel):
    """Keep rows matching a condition."""

    op: Literal["filter_rows"]
    column: str
    comparison: Literal["eq", "ne", "gt", "ge", "lt", "le", "in", "notin"]
    value: Any


class Rename(ApiModel):
    """Rename columns (old -> new)."""

    op: Literal["rename"]
    mapping: dict[str, str] = Field(min_length=1)


Operation = Annotated[
    DropColumns | DropMissingColumns | DropConstantColumns | DropDuplicates | StripWhitespace | FillConstant | Cast
    | LogTransform | Clip | Outliers | FilterRows | Rename,
    Field(discriminator="op"),
]


class TransformRequest(ApiModel):
    """Ordered list of operations, applied atomically."""

    operations: list[Operation] = Field(min_length=1, max_length=30)
