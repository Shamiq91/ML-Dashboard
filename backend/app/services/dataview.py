"""Row preview and per-column profiles for the data-preview panel."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.config import Settings
from app.errors import BadRequestError
from app.services.ingestion import column_kind
from app.utils.sanitize import sanitize


def column_info(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Name, dtype, kind, missing and unique counts per column."""
    return [
        {"name": c, "dtype": str(df[c].dtype), "kind": column_kind(df[c]), "missing": int(df[c].isna().sum()), "unique": int(df[c].nunique(dropna=True))}
        for c in df.columns
    ]


def preview(
    df: pd.DataFrame, offset: int, limit: int, sort: str | None, descending: bool, search: str | None, settings: Settings,
) -> dict[str, Any]:
    """A page of rows with optional sort and case-insensitive substring search over all columns."""
    limit = min(max(limit, 1), settings.preview_max_limit)
    frame = df
    if sort:
        if sort not in df.columns:
            raise BadRequestError(f"Unknown column '{sort}'.", error_code="UNKNOWN_COLUMNS")
        frame = frame.sort_values(sort, ascending=not descending, kind="stable", na_position="last")
    if search:
        needle = search.lower()
        mask = np.zeros(len(frame), dtype=bool)
        for c in frame.columns:
            mask |= frame[c].astype(str).str.lower().str.contains(needle, regex=False, na=False).to_numpy()
        frame = frame[mask]
    page = frame.iloc[offset : offset + limit]
    return {
        "columns": column_info(df), "total": int(len(df)), "filtered": int(len(frame)), "offset": offset, "limit": limit,
        "index": [int(i) for i in page.index],
        "rows": sanitize(page.to_dict(orient="records")),
    }


def profile(df: pd.DataFrame, settings: Settings) -> dict[str, Any]:
    """Distribution summary per column: histogram + quantiles (numeric), top values (categorical), range (dates)."""
    cols: list[dict[str, Any]] = []
    for c in df.columns:
        s = df[c]
        kind = column_kind(s)
        base: dict[str, Any] = {"name": c, "kind": kind, "dtype": str(s.dtype), "missing": int(s.isna().sum()), "unique": int(s.nunique(dropna=True))}
        clean = s.dropna()
        if kind == "numeric" and len(clean):
            v = clean.astype("float64")
            q1, med, q3 = (float(x) for x in v.quantile([0.25, 0.5, 0.75]))
            counts, edges = np.histogram(v.to_numpy(), bins=min(settings.profile_histogram_bins, max(int(v.nunique()), 1)))
            iqr = q3 - q1
            base.update({
                "mean": float(v.mean()), "std": float(v.std()) if len(v) > 1 else 0.0, "min": float(v.min()), "q1": q1, "median": med,
                "q3": q3, "max": float(v.max()), "skew": float(v.skew()) if len(v) > 2 else 0.0,
                "outliers": int(((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)).sum()),
                "histogram": {"edges": [float(e) for e in edges], "counts": [int(x) for x in counts]},
            })
        elif kind in ("categorical", "boolean") and len(clean):
            top = clean.astype(str).value_counts().head(6)
            base["top"] = [{"value": str(k), "count": int(n), "ratio": float(n) / len(clean)} for k, n in top.items()]
        elif kind == "datetime" and len(clean):
            base.update({"min": clean.min().isoformat(), "max": clean.max().isoformat()})
        cols.append(base)
    return {"rows": int(len(df)), "columns": sanitize(cols)}
