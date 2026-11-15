"""Run an exported dashboard model on new data.

This module has no dependency on the dashboard: it needs only numpy, pandas, joblib and the libraries the model was
built with (scikit-learn, xgboost, ...). The export bundle ships this exact file as ``predict.py``, and the dashboard
trains on frames built by the same ``to_model_frame``, so the two cannot drift apart.

Usage of the exported bundle::

    python predict.py model.joblib new_rows.csv predictions.csv

or from Python::

    import joblib, predict
    bundle = joblib.load("model.joblib")
    out = predict.predict(bundle, df)          # df: rows in the same shape as the original upload

A bundle is a plain dict (``FORMAT``) holding the fitted scikit-learn pipeline (preprocessing + estimator), the
feature columns it expects, the column operations the data went through before training (``recipe``), and the
values used to fill missing cells.
"""
from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

FORMAT = "mldash-model/1"
PREDICTION = "prediction"
PROB_PREFIX = "prob_"


# --- data preparation (must match the dashboard's training path) ---------------------------------------------
def to_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Booleans -> int, numbers -> float, datetimes -> epoch seconds, everything else -> str."""
    out = pd.DataFrame(index=df.index)
    for column in df.columns:
        series = df[column]
        if ptypes.is_bool_dtype(series):
            out[column] = series.astype("int64")
        elif ptypes.is_numeric_dtype(series):
            out[column] = series.astype("float64")
        elif ptypes.is_datetime64_any_dtype(series):
            naive = series.dt.tz_localize(None) if getattr(series.dt, "tz", None) is not None else series
            out[column] = (naive - pd.Timestamp("1970-01-01")).dt.total_seconds()
        else:
            out[column] = series.astype(str)
    return out


def column_kind(series: pd.Series) -> str:
    """``bool`` / ``numeric`` / ``datetime`` / ``text``: the type a feature had when the model was trained."""
    if ptypes.is_bool_dtype(series):
        return "bool"
    if ptypes.is_numeric_dtype(series):
        return "numeric"
    if ptypes.is_datetime64_any_dtype(series):
        return "datetime"
    return "text"


def replay_recipe(df: pd.DataFrame, steps: list[dict[str, Any]], notes: list[str] | None = None) -> pd.DataFrame:
    """Apply the recorded column operations (renames, casts, log1p, clips, trims, constant fills) in order.

    Row filters and column drops are not replayed: every new row gets a prediction and unused columns are ignored.
    A step whose column is absent is skipped (the missing column is reported later if the model needs it).
    """
    out = df.copy()
    notes = notes if notes is not None else []
    for step in steps:
        kind = step["op"]
        if kind == "rename":
            out = out.rename(columns={k: v for k, v in step["mapping"].items() if k in out.columns})
            continue
        cols = [c for c in step.get("columns", [step.get("column")]) if c is not None and c in out.columns]
        for c in cols:
            if kind == "cast":
                dtype = step["dtype"]
                if dtype == "numeric":
                    out[c] = pd.to_numeric(out[c], errors="coerce")
                elif dtype == "datetime":
                    out[c] = pd.to_datetime(out[c], errors="coerce")
                elif dtype == "category":
                    out[c] = out[c].astype("string").where(out[c].notna())
                else:
                    out[c] = out[c].astype("string")
            elif kind == "log_transform":
                values = pd.to_numeric(out[c], errors="coerce").astype("float64")
                if (values.dropna() < 0).any():
                    notes.append(f"'{c}' has negative values; they were set to 0 before log1p (training data had none).")
                    values = values.clip(lower=0)
                out[c] = np.log1p(values)
            elif kind == "clip":
                out[c] = pd.to_numeric(out[c], errors="coerce").clip(lower=step.get("lower"), upper=step.get("upper"))
            elif kind == "strip_whitespace":
                mask = out[c].notna()
                out[c] = out[c].astype(object)
                out.loc[mask, c] = out.loc[mask, c].astype(str).str.strip()
            elif kind == "fill_constant":
                value = step["value"]
                if ptypes.is_numeric_dtype(out[c]):
                    value = float(value)
                out[c] = out[c].fillna(value)
    return out


def _coerce(series: pd.Series, kind: str) -> pd.Series:
    """Give a column the type it had in training (a CSV can infer ``"1", "2"`` as numbers, for example)."""
    if kind == "numeric":
        if ptypes.is_bool_dtype(series):
            return series.astype("float64")
        return pd.to_numeric(series, errors="coerce")
    if kind == "datetime":
        return pd.to_datetime(series, errors="coerce")
    if kind == "bool":
        if ptypes.is_bool_dtype(series):
            return series
        lowered = series.astype(str).str.strip().str.lower()
        mapped = lowered.map({"true": True, "1": True, "1.0": True, "yes": True, "false": False, "0": False, "0.0": False, "no": False})
        return mapped.where(series.notna())
    # text: keep missing as missing, everything else as its string form
    return series.astype(object).where(series.isna(), series.astype(str))


def prepare_input(bundle: dict[str, Any], df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """New rows -> the exact frame the pipeline was trained on, plus a report of what had to be fixed."""
    notes: list[str] = []
    frame = replay_recipe(df, bundle.get("recipe", []), notes)
    features: list[str] = bundle["features"]
    kinds: dict[str, str] = bundle["kinds"]
    fill: dict[str, Any] = bundle["fill"]
    absent = [c for c in features if c not in frame.columns]
    if absent:
        raw = bundle.get("raw_names", {})
        raise ValueError(f"Missing column(s) the model needs: {', '.join(sorted({raw.get(c, c) for c in absent}))}")
    filled: dict[str, int] = {}
    unparsed: dict[str, int] = {}
    cols: dict[str, pd.Series] = {}
    for c in features:
        original = frame[c]
        series = _coerce(original, kinds[c])
        bad = int((series.isna() & original.notna()).sum())
        if bad:
            unparsed[c] = bad
        n_missing = int(series.isna().sum())
        if n_missing:
            value = fill[c]
            if kinds[c] == "numeric":
                series = series.fillna(value)
            elif kinds[c] == "datetime":
                series = series.fillna(pd.Timestamp(value))
            elif kinds[c] == "bool":
                series = series.astype(object).where(series.notna(), bool(value)).astype(bool)
            else:
                series = series.astype(object).where(series.notna(), value)
            filled[c] = n_missing
        cols[c] = series
    model_frame = to_model_frame(pd.DataFrame(cols, index=frame.index)[features])
    unseen: dict[str, int] = {}
    for c, known in (bundle.get("categories") or {}).items():
        if c in model_frame.columns and known is not None:
            n = int((~model_frame[c].isin(set(known))).sum())
            if n:
                unseen[c] = n
    report = {"rows": int(len(model_frame)), "filled": filled, "unparsed": unparsed, "unseen_categories": unseen, "notes": notes}
    return model_frame.reset_index(drop=True), report


# --- prediction -------------------------------------------------------------------------------------------------
def decision_to_scores(d: np.ndarray) -> np.ndarray:
    """Decision-function values -> ``(n, k)`` scores in 0..1.

    Binary: a sigmoid, so a score of 0.5 sits exactly on the model's own decision boundary (decision value 0) and
    "score >= 0.5" flags the same rows as ``predict``. Multiclass: a softmax, whose argmax is the predicted class.
    """
    d = np.asarray(d, dtype="float64")
    if d.ndim == 1:
        p1 = 1.0 / (1.0 + np.exp(-np.clip(d, -500, 500)))
        return np.column_stack([1.0 - p1, p1])
    e = np.exp(d - d.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def class_scores(pipeline: Any, x: pd.DataFrame) -> np.ndarray | None:
    """``(n, k)`` probabilities, or the decision function mapped to 0..1 (``decision_to_scores``), or ``None``."""
    try:
        if hasattr(pipeline, "predict_proba"):
            return np.asarray(pipeline.predict_proba(x), dtype="float64")
    except Exception:  # noqa: BLE001 - e.g. SVC without probability=True
        pass
    try:
        if hasattr(pipeline, "decision_function"):
            return decision_to_scores(pipeline.decision_function(x))
    except Exception:  # noqa: BLE001
        pass
    return None


def predict(bundle: dict[str, Any], df: pd.DataFrame, report: dict[str, Any] | None = None) -> pd.DataFrame:
    """The input rows with a ``prediction`` column (and ``prob_<class>`` columns for classifiers) appended.

    Pass a dict as ``report`` to receive what was fixed on the way (filled cells, unreadable values, unseen categories).
    """
    if bundle.get("format") != FORMAT:
        raise ValueError(f"Not a dashboard model bundle (expected format {FORMAT!r}).")
    x, got = prepare_input(bundle, df)
    if report is not None:
        report.update(got)
    pipeline = bundle["pipeline"]
    raw = np.asarray(pipeline.predict(x))
    out = df.reset_index(drop=True).copy()
    if bundle["task"] == "regression":
        out[PREDICTION] = raw.astype("float64") * float(bundle["y_sd"]) + float(bundle["y_mu"])
        return out
    labels = np.asarray(bundle["class_labels"], dtype=object)
    out[PREDICTION] = labels[raw.astype(int)]
    scores = class_scores(pipeline, x)
    if scores is not None:
        for j, label in enumerate(bundle["class_labels"][: scores.shape[1]]):
            out[f"{PROB_PREFIX}{label}"] = np.round(scores[:, j], 6)
    return out


def _read(path: str) -> pd.DataFrame:
    lower = path.lower()
    if lower.endswith(".parquet"):
        return pd.read_parquet(path)
    if lower.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    if lower.endswith(".json"):
        return pd.read_json(path)
    return pd.read_csv(path)


def main(argv: list[str]) -> int:
    """``python predict.py model.joblib input.csv [output.csv]``."""
    if len(argv) < 3:
        print(__doc__)
        return 2
    import joblib

    bundle = joblib.load(argv[1])
    report: dict[str, Any] = {}
    out = predict(bundle, _read(argv[2]), report)
    target = argv[3] if len(argv) > 3 else "predictions.csv"
    out.to_csv(target, index=False)
    print(f"Wrote {len(out)} predictions to {target}")
    for note in report.get("notes", []):
        print(f"note: {note}")
    for key, label in (("filled", "filled missing values"), ("unparsed", "unreadable values treated as missing"),
                       ("unseen_categories", "values never seen in training")):
        if report.get(key):
            print(f"note: {label}: " + ", ".join(f"{c} ({n})" for c, n in report[key].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
