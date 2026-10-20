"""Backend-only data transforms (the dashboard exposes imputation only; everything else is reachable via the API)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from app.errors import BadRequestError, ConflictError


def _numeric(series: pd.Series) -> bool:
    return bool(ptypes.is_numeric_dtype(series) and not ptypes.is_bool_dtype(series))


def _need(df: pd.DataFrame, columns: list[str]) -> None:
    unknown = [c for c in columns if c not in df.columns]
    if unknown:
        raise BadRequestError(f"Unknown column(s): {', '.join(unknown)}", error_code="UNKNOWN_COLUMNS", details=unknown)


def _guard(columns: list[str], protected: set[str], action: str) -> None:
    hit = [c for c in columns if c in protected]
    if hit:
        raise BadRequestError(
            f"Cannot {action} the target/time column ({', '.join(hit)}).", error_code="PROTECTED_COLUMN", details=hit
        )


def apply_operation(df: pd.DataFrame, op: dict[str, Any], protected: set[str]) -> tuple[pd.DataFrame, str]:
    """Apply one operation and return ``(new_frame, human summary)``. ``op['op']`` selects the operation."""
    kind = op["op"]
    out = df.copy()
    if kind == "drop_columns":
        cols = op["columns"]
        _need(out, cols)
        _guard(cols, protected, "drop")
        return out.drop(columns=cols), f"dropped {len(cols)} column(s)"
    if kind == "drop_missing_columns":
        share = out.isna().mean()
        cols = [c for c in out.columns if share[c] > op["threshold"] and c not in protected]
        return out.drop(columns=cols), f"dropped {len(cols)} column(s) with > {op['threshold']:.0%} missing"
    if kind == "drop_constant_columns":
        cols = [c for c in out.columns if out[c].nunique(dropna=False) <= 1 and c not in protected]
        return out.drop(columns=cols), f"dropped {len(cols)} constant column(s)"
    if kind == "drop_duplicates":
        subset = op.get("subset") or None
        if subset:
            _need(out, subset)
        before = len(out)
        out = out.drop_duplicates(subset=subset)
        return out, f"removed {before - len(out)} duplicate row(s)"
    if kind == "strip_whitespace":
        cols = op.get("columns") or [c for c in out.columns if not _numeric(out[c]) and not ptypes.is_datetime64_any_dtype(out[c])]
        _need(out, cols)
        for c in cols:
            mask = out[c].notna()
            out.loc[mask, c] = out.loc[mask, c].astype(str).str.strip()
        return out, f"trimmed whitespace in {len(cols)} column(s)"
    if kind == "fill_constant":
        cols = op["columns"]
        _need(out, cols)
        for c in cols:
            value: Any = op["value"]
            if _numeric(out[c]):
                try:
                    value = float(value)
                except (TypeError, ValueError) as exc:
                    raise BadRequestError(f"'{value}' is not numeric but column '{c}' is.", error_code="BAD_VALUE") from exc
            out[c] = out[c].fillna(value)
        return out, f"filled missing values in {len(cols)} column(s) with a constant"
    if kind == "cast":
        c, dtype = op["column"], op["dtype"]
        _need(out, [c])
        try:
            if dtype == "numeric":
                out[c] = pd.to_numeric(out[c], errors="coerce")
            elif dtype == "datetime":
                out[c] = pd.to_datetime(out[c], errors="coerce")
            elif dtype == "category":
                out[c] = out[c].astype("string").where(out[c].notna())
            else:
                out[c] = out[c].astype("string")
        except Exception as exc:  # noqa: BLE001
            raise BadRequestError(f"Cannot convert '{c}' to {dtype}: {exc}", error_code="CAST_FAILED") from exc
        return out, f"converted '{c}' to {dtype} ({int(out[c].isna().sum() - df[c].isna().sum())} new missing)"
    if kind == "log_transform":
        cols = op["columns"]
        _need(out, cols)
        for c in cols:
            if not _numeric(out[c]):
                raise BadRequestError(f"Column '{c}' is not numeric.", error_code="NOT_NUMERIC")
            if (out[c].dropna() < 0).any():
                raise BadRequestError(f"Column '{c}' has negative values; log1p needs values >= 0.", error_code="NEGATIVE_VALUES")
            out[c] = np.log1p(out[c].astype("float64"))
        return out, f"applied log1p to {len(cols)} column(s)"
    if kind == "clip":
        c = op["column"]
        _need(out, [c])
        if not _numeric(out[c]):
            raise BadRequestError(f"Column '{c}' is not numeric.", error_code="NOT_NUMERIC")
        out[c] = out[c].clip(lower=op.get("lower"), upper=op.get("upper"))
        return out, f"clipped '{c}' to [{op.get('lower')}, {op.get('upper')}]"
    if kind == "outliers":
        cols = op.get("columns") or [c for c in out.columns if _numeric(out[c]) and c not in protected]
        _need(out, cols)
        _guard([c for c in cols if op["action"] == "drop"], protected, "drop outlier rows by")
        keep = pd.Series(True, index=out.index)
        found = 0
        for c in cols:
            s = out[c].astype("float64")
            if op["method"] == "iqr":
                q1, q3 = s.quantile([0.25, 0.75])
                lo, hi = q1 - op["factor"] * (q3 - q1), q3 + op["factor"] * (q3 - q1)
            else:
                mu, sd = s.mean(), s.std()
                lo, hi = mu - op["factor"] * sd, mu + op["factor"] * sd
            bad = (s < lo) | (s > hi)
            found += int(bad.sum())
            if op["action"] == "clip":
                out[c] = s.clip(lo, hi)
            else:
                keep &= ~bad
        if op["action"] == "drop":
            out = out[keep]
        return out, f"{'clipped' if op['action'] == 'clip' else 'dropped rows for'} {found} outlier value(s) in {len(cols)} column(s) ({op['method']}, factor {op['factor']})"
    if kind == "filter_rows":
        c, how, value = op["column"], op["comparison"], op["value"]
        _need(out, [c])
        s = out[c]
        if _numeric(s) and how not in ("in", "notin"):
            value = float(value)
        if how == "eq":
            mask = s == value
        elif how == "ne":
            mask = s != value
        elif how == "gt":
            mask = s > value
        elif how == "ge":
            mask = s >= value
        elif how == "lt":
            mask = s < value
        elif how == "le":
            mask = s <= value
        else:
            values = value if isinstance(value, list) else [value]
            mask = s.astype(str).isin([str(v) for v in values])
            mask = mask if how == "in" else ~mask
        before = len(out)
        out = out[mask.fillna(False)]
        return out, f"kept {len(out)} of {before} rows where {c} {how} {value}"
    if kind == "rename":
        mapping = op["mapping"]
        _need(out, list(mapping))
        clash = [n for n in mapping.values() if n in out.columns and n not in mapping]
        if clash:
            raise BadRequestError(f"Column name(s) already exist: {', '.join(clash)}", error_code="DUPLICATE_COLUMNS")
        return out.rename(columns=mapping), f"renamed {len(mapping)} column(s)"
    raise BadRequestError(f"Unknown operation '{kind}'.", error_code="UNKNOWN_OPERATION")


def apply_operations(df: pd.DataFrame, ops: list[dict[str, Any]], protected: set[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run operations in order. Nothing is committed if any of them fails (the caller swaps frames on success)."""
    current = df
    steps: list[dict[str, Any]] = []
    for op in ops:
        before = current.shape
        current, summary = apply_operation(current, op, protected)
        steps.append({"op": op["op"], "summary": summary, "rows": [int(before[0]), int(current.shape[0])], "columns": [int(before[1]), int(current.shape[1])]})
    if current.empty or current.shape[1] == 0:
        raise ConflictError("These operations would leave an empty dataset.", error_code="EMPTY_RESULT")
    return current.reset_index(drop=True), {
        "steps": steps, "rows_before": int(df.shape[0]), "rows_after": int(current.shape[0]),
        "columns_before": int(df.shape[1]), "columns_after": int(current.shape[1]),
    }
