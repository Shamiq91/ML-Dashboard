"""Export a trained model: a model card and a zip that predicts on new data without the dashboard."""
from __future__ import annotations

import inspect
import io
import json
import zipfile
from datetime import datetime, timezone
from importlib import metadata
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.errors import NotFoundError
from app.services import inference_runtime as rt
from app.services.analysis import FittedModel
from app.services.model_registry import resolve_model

MAX_KNOWN_CATEGORIES = 1000  # categories kept per text column so predict.py can count unseen values

_PIP_NAMES = {
    "sklearn": "scikit-learn", "xgboost": "xgboost", "lightgbm": "lightgbm", "catboost": "catboost",
    "torch": "torch", "pytorch_tabnet": "pytorch-tabnet",
}
_METRICS_CLS = ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "avg_precision")
_METRICS_REG = ("r2", "rmse", "mae")


# --- input schema (built when training starts) -----------------------------------------------------------------
def _raw_names(features: list[str], recipe: list[dict[str, Any]]) -> dict[str, str]:
    """Feature name -> the column name in the original upload (renames undone, last one first)."""
    names = {c: c for c in features}
    for step in reversed(recipe):
        if step["op"] == "rename":
            back = {new: old for old, new in step["mapping"].items()}
            names = {f: back.get(cur, cur) for f, cur in names.items()}
    return names


def _fill_value(series: pd.Series, kind: str) -> Any:
    """Median for numbers and dates, most common value for text and yes/no."""
    clean = series.dropna()
    if clean.empty:
        return {"numeric": 0.0, "datetime": "1970-01-01", "bool": False}.get(kind, "")
    if kind == "numeric":
        return float(clean.astype("float64").median())
    if kind == "datetime":
        return pd.Timestamp(int(clean.astype("int64").median())).isoformat()
    value = clean.mode().iloc[0]
    return bool(value) if kind == "bool" else str(value)


def build_inference_spec(raw_df: pd.DataFrame, working_df: pd.DataFrame, features: list[str],
                         recipe: list[dict[str, Any]], target: str) -> dict[str, Any]:
    """Everything a prediction needs besides the fitted pipeline: feature types, fill values, the replayable recipe."""
    kinds = {c: rt.column_kind(working_df[c]) for c in features}
    raw_names = _raw_names(features, recipe)
    categories: dict[str, list[str] | None] = {}
    for c in features:
        if kinds[c] == "text":
            uniq = working_df[c].dropna().astype(str).unique()
            categories[c] = [str(v) for v in uniq] if len(uniq) <= MAX_KNOWN_CATEGORIES else None
    input_columns = list(dict.fromkeys(raw_names[c] for c in features))
    example_cols = [c for c in input_columns if c in raw_df.columns]
    return {
        "features": list(features), "kinds": kinds, "fill": {c: _fill_value(working_df[c], kinds[c]) for c in features},
        "categories": categories, "recipe": [dict(s) for s in recipe], "raw_names": raw_names,
        "input_columns": input_columns, "target": target, "example_csv": raw_df[example_cols].head(5).to_csv(index=False),
    }


def train_config(body: Any) -> dict[str, Any]:
    """The training request as kept on the session for the model card."""
    return {
        "test_size": body.test_size, "scale_features": body.scale_features, "handle_imbalance": body.handle_imbalance,
        "tune": body.tune, "tune_trials": body.tune_trials if body.tune else 0,
        "cross_validate": body.cross_validate, "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# --- bundle and model card -----------------------------------------------------------------------------------
def fitted_or_404(session: Any, name: str) -> FittedModel:
    """The session's fitted model for ``name`` (key or abbreviation)."""
    spec = resolve_model(name)
    model = session.fitted.get(spec.key)
    if model is None or session.inference is None:
        raise NotFoundError(f"Model '{spec.abbr}' has not been trained in this session.", error_code="MODEL_NOT_TRAINED")
    return model


def make_bundle(session: Any, m: FittedModel) -> dict[str, Any]:
    """The dict saved as ``model.joblib``."""
    spec = session.inference
    return {
        "format": rt.FORMAT, "pipeline": m.pipeline, "task": m.task, "class_labels": list(m.class_labels),
        "y_mu": m.y_mu, "y_sd": m.y_sd, "target": spec["target"], "model": m.key,
        **{k: spec[k] for k in ("features", "kinds", "fill", "categories", "recipe", "raw_names")},
    }


def nested_estimators(est: Any) -> list[Any]:
    """``est`` and every estimator nested in it (stacking, voting, bagging ...)."""
    found = [est]
    params = est.get_params(deep=False) if hasattr(est, "get_params") else {}
    for value in params.values():
        items = value if isinstance(value, (list, tuple)) else [value]
        for item in items:
            inner = item[1] if isinstance(item, tuple) and len(item) == 2 else item
            if hasattr(inner, "fit") and hasattr(inner, "get_params"):
                found.extend(nested_estimators(inner))
    return found


def requirements(m: FittedModel) -> tuple[list[str], bool]:
    """Pinned pip requirements to load the bundle, and whether it loads without the dashboard's own code."""
    packages = {"numpy", "pandas", "joblib", "scikit-learn"}
    portable = True
    for est in nested_estimators(m.pipeline.named_steps["clf"]):
        module = type(est).__module__
        top = module.split(".")[0]
        if top == "app":
            portable = False
            if "torch" in module:
                packages.add("torch")
                if "TabNet" in type(est).__name__:
                    packages.add("pytorch-tabnet")
        elif top in _PIP_NAMES:
            packages.add(_PIP_NAMES[top])
    lines = []
    for name in sorted(packages):
        try:
            lines.append(f"{name}=={metadata.version(name)}")
        except metadata.PackageNotFoundError:
            lines.append(name)
    return lines, portable


def _plain(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if value is None or isinstance(value, (int, str)):
        return value
    return repr(value)[:200]


def model_card(session: Any, m: FittedModel) -> dict[str, Any]:
    """What the model is, how it was trained, how well it did, and what it expects as input."""
    row = session.results.get(m.key, {})
    spec = session.inference
    cfg = session.train_config or {}
    split = session.split or {}
    reqs, portable = requirements(m)
    est = m.pipeline.named_steps["clf"]
    try:
        params = {k: _plain(v) for k, v in sorted(est.get_params(deep=False).items())}
    except Exception:  # noqa: BLE001
        params = {}
    limitations = []
    if not portable:
        limitations.append("This model uses a class from the dashboard backend: loading model.joblib needs the backend folder on PYTHONPATH.")
    if row.get("overfit_warning"):
        limitations.append(f"Flagged as overfit in training (train-test gap {row.get('train_test_gap')}).")
    if row.get("underfit_warning"):
        limitations.append("Flagged as underfit in training: it barely beats chance on its own training rows.")
    # predict.py fills gaps with the median / most common value; only other imputation methods can disagree with it
    methods = sorted({str(v) for c in session.cleaning for v in c.get("methods_applied", {}).values()}
                     - {"median", "mode", "mode (fallback)", "drop rows"})
    if methods:
        limitations.append(
            f"Missing values in the training data were filled with: {', '.join(methods)}. predict.py fills missing cells "
            "with the training median (numbers) or most common value (text), which can differ from that."
        )
    if any(spec["kinds"][c] == "text" and spec["categories"].get(c) is None for c in spec["features"]):
        limitations.append("Some text columns had over 1,000 distinct values; predict.py does not count unseen values for them.")
    if m.key == "svm":
        limitations.extend(w for w in session.warnings or [] if "SVM trained on" in w)
    metric_keys = _METRICS_REG if m.regression else _METRICS_CLS
    return {
        "model": m.key, "abbr": m.abbr, "name": row.get("name") or resolve_model(m.key).name, "family": row.get("family"),
        "task": m.task, "target": spec["target"], "classes": None if m.regression else list(m.class_labels),
        "positive_class": m.class_labels[1] if not m.regression and len(m.class_labels) == 2 else None,
        "dataset": session.filename, "rows_trained": split.get("train_rows"), "rows_tested": split.get("test_rows"),
        "trained_at": cfg.get("trained_at"), "settings": {k: v for k, v in cfg.items() if k != "trained_at"},
        "metrics": {k: row[k] for k in metric_keys if row.get(k) is not None},
        "cv": {"score": row.get("cv_score"), "std": row.get("cv_std"), "metric": (row.get("cv") or {}).get("metric")}
        if row.get("cv_score") is not None else None,
        "train_test_gap": row.get("train_test_gap"),
        "flags": [f for f, on in (("OVERFIT", row.get("overfit_warning")), ("UNDERFIT", row.get("underfit_warning")), ("TUNED", row.get("tuned"))) if on],
        "tuned_params": (row.get("tuning") or {}).get("best_params") if row.get("tuned") else None,
        "params": params,
        "input": {"columns": spec["input_columns"], "recipe": spec["recipe"], "excluded": session.exclusions,
                  "features": [{"name": c, "kind": spec["kinds"][c], "fill": spec["fill"][c]} for c in spec["features"]]},
        "requirements": reqs, "portable": portable, "limitations": limitations, "format": rt.FORMAT,
    }


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return "--" if v is None else str(v)


def describe_step(step: dict[str, Any]) -> str:
    """One recipe step in words."""
    kind = step["op"]
    if kind == "rename":
        return "rename " + ", ".join(f"`{a}` -> `{b}`" for a, b in step["mapping"].items())
    cols = ", ".join(f"`{c}`" for c in step.get("columns", [step.get("column")]) if c)
    return {
        "cast": f"convert {cols} to {step.get('dtype')}",
        "log_transform": f"log1p {cols}",
        "clip": f"clip {cols} to [{_fmt(step.get('lower'))}, {_fmt(step.get('upper'))}]",
        "strip_whitespace": f"trim whitespace in {cols}",
        "fill_constant": f"fill missing {cols} with {step.get('value')!r}",
    }.get(kind, kind)


def model_card_markdown(card: dict[str, Any]) -> str:
    """The card as a readable MODEL_CARD.md."""
    task = "Regression" if card["task"] == "regression" else f"Classification ({', '.join(card['classes'] or [])})"
    lines = [f"# {card['name']} ({card['abbr']})", "",
             f"- **Task:** {task}", f"- **Target:** `{card['target']}`", f"- **Dataset:** {card['dataset']}",
             *([f"- **Positive class:** `{card['positive_class']}` (precision, recall and F1 are about this class)"]
               if card.get("positive_class") else []),
             f"- **Rows:** {_fmt(card['rows_trained'])} trained, {_fmt(card['rows_tested'])} held out for testing",
             f"- **Trained:** {card['trained_at'] or '--'}", "",
             "## Performance (held-out test rows)", "", "| Metric | Value |", "|---|---|"]
    lines += [f"| {k} | {_fmt(v)} |" for k, v in card["metrics"].items()]
    if card["cv"]:
        lines.append(f"| cross-validated {card['cv']['metric'] or 'score'} | {_fmt(card['cv']['score'])} +/- {_fmt(card['cv']['std'])} |")
    lines += [f"| train-test gap | {_fmt(card['train_test_gap'])} |", ""]
    if card["flags"]:
        lines += [f"Flags: {', '.join(card['flags'])}", ""]
    s = card["settings"]
    if s:
        lines += ["## Training settings", "",
                  f"Test split {s.get('test_size')}, scaling {'on' if s.get('scale_features') else 'off'}, imbalance handling "
                  f"`{s.get('handle_imbalance')}`, tuning {'on (' + str(s.get('tune_trials')) + ' trials)' if s.get('tune') else 'off'}.", ""]
    if card["tuned_params"]:
        lines += ["Tuned parameters: " + ", ".join(f"`{k}={v}`" for k, v in card["tuned_params"].items()), ""]
    lines += ["## Input", "", "New rows need these columns, as in the original upload: "
              + ", ".join(f"`{c}`" for c in card["input"]["columns"]) + ". Extra columns are ignored and kept in the output.", ""]
    if card["input"]["recipe"]:
        lines += ["predict.py applies these steps first, as they were applied before training:", ""]
        lines += [f"{i}. {describe_step(st)}" for i, st in enumerate(card["input"]["recipe"], 1)] + [""]
    lines += ["Missing cells are filled with the training median (numbers) or most common value (text, yes/no).", "",
              "## Use it", "", "```bash", "pip install -r requirements.txt",
              "python predict.py model.joblib example_input.csv predictions.csv", "```", "",
              "```python", "import joblib, pandas as pd, predict", "bundle = joblib.load('model.joblib')",
              "out = predict.predict(bundle, pd.read_csv('new_rows.csv'))  # adds 'prediction' (and prob_<class>)", "```", ""]
    if card["limitations"]:
        lines += ["## Limitations", ""] + [f"- {x}" for x in card["limitations"]] + [""]
    return "\n".join(lines)


def export_zip(session: Any, m: FittedModel) -> bytes:
    """``model.joblib`` + ``predict.py`` + card (JSON and Markdown) + ``requirements.txt`` + ``example_input.csv``."""
    card = model_card(session, m)
    model_bytes = io.BytesIO()
    joblib.dump(make_bundle(session, m), model_bytes, compress=3)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("model.joblib", model_bytes.getvalue())
        z.writestr("predict.py", inspect.getsource(rt))
        z.writestr("model_card.json", json.dumps(card, indent=2, default=str))
        z.writestr("MODEL_CARD.md", model_card_markdown(card))
        z.writestr("requirements.txt", "\n".join(card["requirements"]) + "\n")
        z.writestr("example_input.csv", session.inference["example_csv"])
    return buf.getvalue()


def export_name(session: Any, m: FittedModel) -> str:
    """A safe zip file name: ``<dataset>_<target>_<model>.zip``."""
    raw = f"{session.filename.rsplit('.', 1)[0]}_{session.inference['target']}_{m.key}"
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in raw)[:80] + ".zip"
