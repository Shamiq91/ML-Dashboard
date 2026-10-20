"""File parsing, column typing and target validation."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api import types as ptypes

from app.config import Settings
from app.errors import BadRequestError
from app.services.tasks import is_regression
from app.utils.sanitize import sanitize

ALLOWED_EXTENSIONS = {".csv", ".parquet"}


def parse_upload(filename: str, content: bytes) -> pd.DataFrame:
    """Parse a CSV/Parquet payload into a non-empty DataFrame or raise 400."""
    extension = Path(filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise BadRequestError(
            f"Unsupported file type '{extension or 'none'}'. Upload a .csv or .parquet file.",
            error_code="UNSUPPORTED_FILE_TYPE",
        )
    if not content or not content.strip():
        raise BadRequestError("The uploaded file is empty.", error_code="EMPTY_FILE")
    try:
        df = _read_csv(content) if extension == ".csv" else pd.read_parquet(io.BytesIO(content))
    except BadRequestError:
        raise
    except pd.errors.EmptyDataError as exc:
        raise BadRequestError("The uploaded file is empty.", error_code="EMPTY_FILE") from exc
    except Exception as exc:  # noqa: BLE001 - any parser failure means a corrupt file
        raise BadRequestError(
            f"Could not read the {extension} file: it appears to be corrupt.",
            error_code="CORRUPT_FILE",
            details=str(exc)[:200],
        ) from exc
    if df.shape[1] == 0 or len(df) == 0:
        raise BadRequestError(
            "The file has no data rows (header-only or empty).", error_code="EMPTY_DATASET"
        )
    df.columns = [str(column).strip() for column in df.columns]
    if df.columns.duplicated().any():
        raise BadRequestError("Duplicate column names are not supported.", error_code="DUPLICATE_COLUMNS")
    return df


def _read_csv(content: bytes) -> pd.DataFrame:
    if b"\x00" in content[:4096]:
        raise BadRequestError("The CSV contains binary data and cannot be parsed.", error_code="CORRUPT_FILE")
    try:
        return pd.read_csv(io.BytesIO(content), encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(content), encoding="latin-1")


def column_kind(series: pd.Series) -> str:
    """Classify a column as numeric / categorical / datetime / boolean."""
    if ptypes.is_bool_dtype(series):
        return "boolean"
    if ptypes.is_numeric_dtype(series):
        return "numeric"
    if ptypes.is_datetime64_any_dtype(series):
        return "datetime"
    return "categorical"


def memory_mb(df: pd.DataFrame) -> float:
    """Deep memory usage in MB rounded to 3 decimals."""
    return round(float(df.memory_usage(deep=True).sum()) / (1024**2), 3)


def describe_upload(df: pd.DataFrame) -> dict[str, Any]:
    """Build the dtype/kind/preview part of the upload response."""
    return {
        "total_rows": int(len(df)),
        "total_columns": int(df.shape[1]),
        "memory_usage_mb": memory_mb(df),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "column_kinds": {column: column_kind(df[column]) for column in df.columns},
        "preview": sanitize(df.head(5).to_dict(orient="records")),
    }


def is_numeric_series(series: pd.Series) -> bool:
    """Numeric but not boolean."""
    return bool(ptypes.is_numeric_dtype(series) and not ptypes.is_bool_dtype(series))


def infer_task(series: pd.Series, settings: Settings) -> str:
    """``regression`` for a numeric target with many distinct values, otherwise ``classification``."""
    clean = series.dropna()
    if is_numeric_series(clean) and int(clean.nunique()) > settings.continuous_target_unique:
        return "regression"
    return "classification"


def resolve_task(series: pd.Series, requested: str, settings: Settings, target: str) -> str:
    """Apply an explicit ``classification``/``regression`` choice, or infer it for ``auto``."""
    task = infer_task(series, settings) if requested == "auto" else requested
    if is_regression(task) and not is_numeric_series(series.dropna()):
        raise BadRequestError(
            f"Target '{target}' is not numeric, so it cannot be used for regression.",
            error_code="TARGET_NOT_NUMERIC",
        )
    return task


def describe_target(df: pd.DataFrame, target: str, settings: Settings, task: str | None = None) -> dict[str, Any]:
    """Validate ``target`` and describe its type, class count and warnings."""
    if target not in df.columns:
        raise BadRequestError(
            f"Target column '{target}' does not exist.",
            error_code="TARGET_NOT_FOUND",
            details={"available_columns": list(df.columns)},
        )
    series = df[target].dropna()
    if series.empty:
        raise BadRequestError(f"Target column '{target}' contains only missing values.", error_code="TARGET_ALL_NAN")
    n_classes = int(series.nunique())
    warnings: list[str] = []
    numeric = is_numeric_series(series)
    task = task or infer_task(series, settings)
    if is_regression(task):
        kind = "continuous"
        if n_classes < 2:
            warnings.append("Target has a single value; at least 2 distinct values are needed to train.")
    elif numeric and n_classes > settings.continuous_target_unique:
        kind = "continuous"
        warnings.append(
            f"Target has {n_classes} unique numeric values but task is classification; "
            "every value would be its own class. Switch the task to regression."
        )
    elif n_classes == 1:
        kind = "single_class"
        warnings.append("Target has a single class; at least 2 classes are needed to train.")
    elif n_classes == 2:
        kind = "binary"
    else:
        kind = "multiclass"
    return {
        "dtype": str(df[target].dtype),
        "task": task,
        "kind": kind,
        "n_classes": n_classes,
        "warnings": warnings,
    }
