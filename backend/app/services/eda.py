"""Descriptive statistics and correlation analysis."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes


def numeric_feature_columns(df: pd.DataFrame, target: str | None) -> list[str]:
    """Numeric (non-boolean) columns excluding the target."""
    return [
        c
        for c in df.columns
        if c != target and ptypes.is_numeric_dtype(df[c]) and not ptypes.is_bool_dtype(df[c])
    ]


def _num(value: Any) -> float | None:
    number = float(value)
    return round(number, 6) if math.isfinite(number) else None


def eda_summary(df: pd.DataFrame, target: str | None) -> dict[str, Any]:
    """Count/mean/std/min/median/max for every numeric feature."""
    features: dict[str, Any] = {}
    notes: list[str] = []
    for column in numeric_feature_columns(df, target):
        series = df[column].astype("float64")
        valid = series.dropna()
        constant = bool(valid.nunique() <= 1 and len(valid) > 0)
        if valid.empty:
            notes.append(f"'{column}' is entirely missing.")
        elif constant:
            notes.append(f"'{column}' is constant.")
        features[column] = {
            "count": int(valid.size),
            "null_count": int(series.isna().sum()),
            "mean": _num(valid.mean()) if valid.size else None,
            "std": _num(valid.std(ddof=1)) if valid.size > 1 else (0.0 if valid.size == 1 else None),
            "min": _num(valid.min()) if valid.size else None,
            "median": _num(valid.median()) if valid.size else None,
            "max": _num(valid.max()) if valid.size else None,
            "constant": constant,
        }
    if not features:
        notes.append("No numeric features available.")
    return {"features": features, "n_features": len(features), "notes": notes}


def find_multicollinear_pairs(
    columns: list[str], matrix: list[list[float | None]], threshold: float
) -> list[dict[str, Any]]:
    """Unique upper-triangle pairs with |r| > threshold, sorted by |r| descending."""
    flags: list[dict[str, Any]] = []
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            r = matrix[i][j]
            if r is not None and math.isfinite(r) and abs(r) > threshold:
                flags.append({"feature_a": columns[i], "feature_b": columns[j], "r": r})
    return sorted(flags, key=lambda flag: abs(flag["r"]), reverse=True)


def correlation(df: pd.DataFrame, target: str | None, threshold: float) -> dict[str, Any]:
    """Pearson correlation matrix over numeric features plus multicollinearity flags."""
    columns = numeric_feature_columns(df, target)
    notes: list[str] = []
    if len(columns) < 2:
        notes.append("At least 2 numeric features are needed for a correlation matrix.")
        return {
            "columns": columns,
            "matrix": [[1.0]] if len(columns) == 1 else [],
            "multicollinearity_flags": [],
            "threshold": threshold,
            "notes": notes,
        }
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = df[columns].astype("float64").corr(method="pearson")
    matrix: list[list[float | None]] = [
        [_num(corr.iloc[i, j]) for j in range(len(columns))] for i in range(len(columns))
    ]
    constant = [c for i, c in enumerate(columns) if all(matrix[i][j] is None for j in range(len(columns)) if j != i)]
    if constant:
        notes.append(f"Constant column(s) have undefined correlation: {', '.join(constant)}.")
    return {
        "columns": columns,
        "matrix": matrix,
        "multicollinearity_flags": find_multicollinear_pairs(columns, matrix, threshold),
        "threshold": threshold,
        "notes": notes,
    }
