"""Missing-value scan, interactive cleaning and class-balance analysis."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from app.errors import BadRequestError, ConflictError
from app.utils.safe_math import safe_div


def missing_report(df: pd.DataFrame) -> dict[str, Any]:
    """Per-column and total missing counts/percentages."""
    rows = int(len(df))
    columns = {}
    total = 0
    for column in df.columns:
        count = int(df[column].isna().sum())
        total += count
        columns[column] = {"count": count, "percent": round(safe_div(count, rows) * 100, 4)}
    cells = rows * int(df.shape[1])
    return {
        "columns": columns,
        "rows": rows,
        "total_missing": total,
        "total_cells": cells,
        "total_percent": round(safe_div(total, cells) * 100, 4),
        "has_missing": total > 0,
    }


def _is_numeric(series: pd.Series) -> bool:
    return bool(ptypes.is_numeric_dtype(series) and not ptypes.is_bool_dtype(series))


def _as_nan_object(series: pd.Series) -> pd.Series:
    """Object copy with real ``np.nan`` for missing and strings elsewhere (sklearn-safe)."""
    return series.astype(object).map(lambda v: np.nan if pd.isna(v) else str(v))


def _impute_numeric(out: pd.DataFrame, columns: list[str], method: str) -> None:
    strategy = {"mean": "mean", "median": "median", "mode": "most_frequent"}[method]
    values = SimpleImputer(strategy=strategy).fit_transform(out[columns].astype("float64"))
    out[columns] = values


def _impute_mode(out: pd.DataFrame, column: str) -> None:
    filled = SimpleImputer(strategy="most_frequent", missing_values=np.nan).fit_transform(
        _as_nan_object(out[column]).to_frame()
    )
    out[column] = filled.ravel()


def _impute_knn(
    out: pd.DataFrame,
    targets: list[str],
    exclude: str | None,
    requested_k: int,
    warnings: list[str],
) -> list[str]:
    """KNN-impute numeric ``targets``; returns the columns actually imputed."""
    features = [c for c in out.columns if c != exclude and not out[c].isna().all()]
    numeric = [c for c in features if _is_numeric(out[c])]
    categorical = [c for c in features if c not in numeric]
    matrix_parts: list[np.ndarray] = []
    order: list[str] = []
    if numeric:
        matrix_parts.append(out[numeric].astype("float64").to_numpy())
        order += numeric
    if categorical:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=np.nan, encoded_missing_value=np.nan
        )
        encoded = encoder.fit_transform(pd.DataFrame({c: _as_nan_object(out[c]) for c in categorical}))
        matrix_parts.append(np.asarray(encoded, dtype="float64"))
        order += categorical
    matrix = np.hstack(matrix_parts)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)

    donors = min(int(out[c].notna().sum()) for c in targets)
    k = max(1, min(requested_k, len(out) - 1, donors))
    if k != requested_k:
        warnings.append(f"KNN neighbors reduced from {requested_k} to {k} (small dataset).")
    imputed = scaler.inverse_transform(KNNImputer(n_neighbors=k).fit_transform(scaled))
    done: list[str] = []
    for column in targets:
        position = order.index(column)
        mask = out[column].isna().to_numpy()
        values = out[column].astype("float64").to_numpy().copy()
        values[mask] = imputed[mask, position]
        out[column] = values
        done.append(column)
    return done


def _interpolate_columns(
    out: pd.DataFrame, selected: list[str], sort_by: str | None, methods: dict[str, str], warnings: list[str]
) -> pd.DataFrame:
    """Time-ordered fill: linear interpolation for numbers (the target too), forward fill for the rest.

    Only gaps *inside* a series are filled. Rows before the first / after the last real value of a column are removed
    instead, because back-filling would invent a constant history that never existed.
    """
    if sort_by and sort_by in out.columns:
        order = pd.to_datetime(out[sort_by], errors="coerce")
        out = out.assign(__t=order).sort_values("__t", kind="stable").drop(columns="__t").reset_index(drop=True)
    else:
        warnings.append("No time column set; interpolating in file order.")
    for column in selected:
        if not out[column].isna().any():
            continue
        if out[column].isna().all():
            methods[column] = "skipped (all values missing)"
            warnings.append(f"Column '{column}' is entirely missing and cannot be filled.")
        elif _is_numeric(out[column]):
            out[column] = out[column].astype("float64").interpolate(method="linear", limit_area="inside")
            methods[column] = "interpolate"
        else:
            out[column] = out[column].ffill()
            methods[column] = "forward fill"
    active = [c for c in selected if not out[c].isna().all()]
    complete = out[active].notna().all(axis=1) if active else pd.Series(True, index=out.index)
    if complete.any() and not complete.all():
        first, last = int(complete.idxmax()), int(complete[::-1].idxmax())
        lead, tail = first, len(out) - 1 - last
        if lead or tail:
            when = f" (first complete row: {out.loc[first, sort_by]})" if sort_by and sort_by in out.columns else ""
            parts = [f"{n} {side} row(s)" for n, side in ((lead, "leading"), (tail, "trailing")) if n]
            warnings.append(
                f"Removed {' and '.join(parts)} where a column has no data before/after its first/last real value"
                f"{when}; interpolation cannot invent a history that never existed."
            )
            out = out.iloc[first : last + 1].reset_index(drop=True)
    return out


def clean_dataframe(
    df: pd.DataFrame,
    target: str | None,
    strategy: str,
    method: str | None,
    columns: list[str] | None,
    knn_neighbors: int = 5,
    sort_by: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop or impute missing values and return ``(clean_df, report)``."""
    unknown = [c for c in (columns or []) if c not in df.columns]
    if unknown:
        raise BadRequestError(
            f"Unknown column(s): {', '.join(unknown)}", error_code="UNKNOWN_COLUMNS", details=unknown
        )
    selected = list(columns) if columns else list(df.columns)
    before = missing_report(df)
    out = df.copy()
    methods: dict[str, str] = {}
    fallbacks: list[str] = []
    warnings: list[str] = []

    if strategy == "impute" and method == "interpolate":
        out = _interpolate_columns(out, selected, sort_by, methods, warnings)
    elif strategy == "drop":
        out = out.dropna(subset=selected)
        methods = {c: "drop rows" for c in selected if before["columns"][c]["count"] > 0}
    else:
        assert method is not None
        if target and target in out.columns and out[target].isna().any():
            dropped = int(out[target].isna().sum())
            out = out[out[target].notna()].copy()
            warnings.append(f"Dropped {dropped} row(s) with a missing target (a target cannot be imputed).")
        if out.empty:
            raise ConflictError("Cleaning would leave an empty dataset.", error_code="EMPTY_RESULT")
        todo = [c for c in selected if c != target and out[c].isna().any()]
        skipped = [c for c in todo if out[c].isna().all()]
        for column in skipped:
            methods[column] = "skipped (all values missing)"
            warnings.append(f"Column '{column}' is entirely missing and cannot be imputed.")
        todo = [c for c in todo if c not in skipped]
        numeric = [c for c in todo if _is_numeric(out[c])]
        other = [c for c in todo if c not in numeric]
        if numeric:
            if method == "knn":
                for column in _impute_knn(out, numeric, target, knn_neighbors, warnings):
                    methods[column] = f"knn(k<={knn_neighbors})"
            else:
                _impute_numeric(out, numeric, method)
                methods.update({c: method for c in numeric})
        for column in other:
            _impute_mode(out, column)
            methods[column] = "mode" if method == "mode" else "mode (fallback)"
            if method != "mode":
                fallbacks.append(f"{column}: categorical column, '{method}' not applicable -> mode")

    if out.empty:
        raise ConflictError(
            "Cleaning would leave an empty dataset; choose 'impute' or fewer columns.",
            error_code="EMPTY_RESULT",
        )
    after = missing_report(out)
    report = {
        "strategy": strategy,
        "method": method,
        "rows_before": int(len(df)),
        "rows_after": int(len(out)),
        "missing_before": before["total_missing"],
        "missing_after": after["total_missing"],
        "missing_before_by_column": {c: v["count"] for c, v in before["columns"].items()},
        "missing_after_by_column": {c: v["count"] for c, v in after["columns"].items()},
        "methods_applied": methods,
        "fallbacks": fallbacks,
        "warnings": warnings,
    }
    return out.reset_index(drop=True), report


def target_summary(series: pd.Series, target: str, bins: int = 20) -> dict[str, Any]:
    """Distribution of a continuous target: quantiles, skew and a histogram (regression counterpart of class balance)."""
    clean = pd.to_numeric(series.dropna(), errors="coerce").dropna().astype("float64")
    if clean.empty:
        raise BadRequestError(f"Target '{target}' has no non-missing values.", error_code="TARGET_ALL_NAN")
    q1, median, q3 = (float(v) for v in clean.quantile([0.25, 0.5, 0.75]))
    counts, edges = np.histogram(clean.to_numpy(), bins=min(bins, max(int(clean.nunique()), 1)))
    skew = float(clean.skew()) if len(clean) > 2 else 0.0
    std = float(clean.std()) if len(clean) > 1 else 0.0
    return {
        "target": target,
        "count": int(len(clean)),
        "n_unique": int(clean.nunique()),
        "mean": round(float(clean.mean()), 6),
        "std": round(std if np.isfinite(std) else 0.0, 6),
        "min": round(float(clean.min()), 6),
        "q1": round(q1, 6),
        "median": round(median, 6),
        "q3": round(q3, 6),
        "max": round(float(clean.max()), 6),
        "skew": round(skew, 4) if np.isfinite(skew) else 0.0,
        "histogram": {"edges": [round(float(e), 6) for e in edges], "counts": [int(c) for c in counts]},
    }


def class_balance(series: pd.Series, target: str, threshold: float) -> dict[str, Any]:
    """Class distribution plus imbalance verdict (minority ratio < threshold is STRICT)."""
    clean = series.dropna()
    total = int(len(clean))
    if total == 0:
        raise BadRequestError(f"Target '{target}' has no non-missing values.", error_code="TARGET_ALL_NAN")
    counts = clean.astype(str).value_counts()
    classes = [
        {"label": str(label), "count": int(count), "ratio": round(safe_div(int(count), total), 6)}
        for label, count in counts.items()
    ]
    smallest = min(classes, key=lambda c: c["count"])
    largest = max(classes, key=lambda c: c["count"])
    single = len(classes) == 1
    minority_ratio = round(smallest["ratio"], 10)
    flagged = (not single) and minority_ratio < threshold
    if single:
        message = f"Target '{target}' has a single class; training needs at least 2."
    elif flagged:
        message = (
            f"Minority class '{smallest['label']}' is {minority_ratio * 100:.2f}% of rows "
            f"(below the {threshold * 100:.0f}% threshold) - consider class weights or SMOTE."
        )
    else:
        message = f"Classes are acceptably balanced (minority {minority_ratio * 100:.2f}%)."
    return {
        "target": target,
        "total": total,
        "classes": classes,
        "minority_class": smallest["label"],
        "minority_ratio": minority_ratio,
        "imbalance_ratio": round(safe_div(largest["count"], smallest["count"], default=1.0), 4),
        "imbalance_warning": flagged,
        "severity": "STRICT" if flagged else "NONE",
        "single_class": single,
        "message": message,
    }
