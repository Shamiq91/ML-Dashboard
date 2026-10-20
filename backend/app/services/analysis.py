"""Model analysis on a fitted pipeline: ROC/PR/threshold curves, residuals, permutation importance and SHAP."""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, r2_score, roc_auc_score, roc_curve,
)

from app.config import Settings
from app.errors import BadRequestError
from app.services.tasks import is_regression
from app.utils.downsample import select_indices

logger = logging.getLogger("mldash.analysis")


@dataclass
class FittedModel:
    """Everything needed to analyse one trained model after the job is over (held in memory per session)."""

    key: str
    abbr: str
    task: str
    pipeline: Any
    x_train: pd.DataFrame  # sample of the rows the model saw (SHAP background)
    x_test: pd.DataFrame
    y_test: np.ndarray  # class codes, or the standardised target for regression
    class_labels: list[str]
    y_mu: float = 0.0
    y_sd: float = 1.0
    cache: dict[str, Any] = field(default_factory=dict)

    @property
    def regression(self) -> bool:
        """True for regression."""
        return is_regression(self.task)


# --- scores ------------------------------------------------------------------------------------------------
def score_matrix(pipeline: Any, x: pd.DataFrame) -> tuple[np.ndarray, str] | None:
    """``(n, k)`` class scores in 0..1 plus their kind, or ``None`` if the model exposes no score."""
    try:
        if hasattr(pipeline, "predict_proba"):
            return np.asarray(pipeline.predict_proba(x), dtype="float64"), "probability"
    except Exception:  # noqa: BLE001 - fall through to the decision function
        logger.debug("predict_proba failed", exc_info=True)
    try:
        if hasattr(pipeline, "decision_function"):
            d = np.asarray(pipeline.decision_function(x), dtype="float64")
            if d.ndim == 1:
                span = float(d.max() - d.min()) or 1.0
                p1 = (d - d.min()) / span
                return np.column_stack([1.0 - p1, p1]), "decision"
            e = np.exp(d - d.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True), "decision"
    except Exception:  # noqa: BLE001
        logger.debug("decision_function failed", exc_info=True)
    return None


def headline_scores(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    """ROC AUC and average precision for the leaderboard (binary, or one-vs-rest macro for >2 classes)."""
    out: dict[str, float | None] = {"roc_auc": None, "avg_precision": None}
    try:
        if scores.shape[1] == 2:
            out["roc_auc"] = float(roc_auc_score(y_true, scores[:, 1]))
            out["avg_precision"] = float(average_precision_score(y_true, scores[:, 1]))
        else:
            out["roc_auc"] = float(roc_auc_score(y_true, scores, multi_class="ovr", average="macro", labels=np.arange(scores.shape[1])))
            onehot = np.eye(scores.shape[1])[y_true]
            out["avg_precision"] = float(average_precision_score(onehot, scores, average="macro"))
    except ValueError:
        pass
    return {k: None if v is None else round(v, 6) for k, v in out.items()}


def _thin(*arrays: np.ndarray, limit: int) -> list[list[float]]:
    idx = select_indices(len(arrays[0]), limit)
    return [[round(float(a[i]), 5) for i in idx] for a in arrays]


def _confusion(y: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    return {
        "tp": int(((pred == 1) & (y == 1)).sum()), "fp": int(((pred == 1) & (y == 0)).sum()),
        "tn": int(((pred == 0) & (y == 0)).sum()), "fn": int(((pred == 0) & (y == 1)).sum()),
    }


def _rates(c: dict[str, int]) -> dict[str, float]:
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    total = tp + fp + tn + fn
    return {
        "precision": precision, "recall": recall, "specificity": specificity,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "accuracy": (tp + tn) / total if total else 0.0,
        "balanced_accuracy": (recall + specificity) / 2,
    }


def curves_payload(m: FittedModel, settings: Settings, threshold: float = 0.5) -> dict[str, Any]:
    """ROC, precision-recall and (binary) threshold analysis on the held-out test rows."""
    if m.regression:
        raise BadRequestError("ROC/PR curves exist only for classification models.", error_code="NOT_CLASSIFICATION")
    got = score_matrix(m.pipeline, m.x_test)
    if got is None:
        raise BadRequestError(f"{m.abbr} exposes no probability or decision score.", error_code="NO_SCORES")
    scores, kind = got
    y = m.y_test
    k = scores.shape[1]
    binary = k == 2
    roc_classes, pr_classes = [], []
    for c in ([1] if binary else range(k)):
        target = (y == c).astype(int)
        if target.min() == target.max():
            continue
        fpr, tpr, _ = roc_curve(target, scores[:, c])
        prec, rec, _ = precision_recall_curve(target, scores[:, c])
        fpr_t, tpr_t = _thin(fpr, tpr, limit=settings.curve_max_points)
        rec_t, prec_t = _thin(rec, prec, limit=settings.curve_max_points)
        label = m.class_labels[c] if c < len(m.class_labels) else str(c)
        roc_classes.append({"label": label, "fpr": fpr_t, "tpr": tpr_t, "auc": round(float(roc_auc_score(target, scores[:, c])), 6)})
        pr_classes.append({
            "label": label, "recall": rec_t, "precision": prec_t,
            "ap": round(float(average_precision_score(target, scores[:, c])), 6), "prevalence": round(float(target.mean()), 6),
        })
    head = headline_scores(y, scores)
    out: dict[str, Any] = {
        "model": m.key, "abbr": m.abbr, "classes": m.class_labels, "binary": binary, "score_kind": kind,
        "positive_class": m.class_labels[1] if binary else None, "n_test": int(len(y)),
        "roc": {"classes": roc_classes, "macro_auc": head["roc_auc"]},
        "pr": {"classes": pr_classes, "macro_ap": head["avg_precision"]},
        "thresholds": None,
    }
    if binary:
        p1 = scores[:, 1]
        grid = np.linspace(0.0, 1.0, settings.threshold_points)
        rows = [_rates(_confusion(y, (p1 >= t).astype(int))) | _confusion(y, (p1 >= t).astype(int)) for t in grid]
        series = {name: [round(float(r[name]), 5) for r in rows] for name in rows[0]}
        best_f1 = int(np.argmax(series["f1"]))
        youden = int(np.argmax([r["recall"] + r["specificity"] - 1 for r in rows]))
        c = _confusion(y, (p1 >= threshold).astype(int))
        out["thresholds"] = {
            "grid": [round(float(t), 5) for t in grid], **series,
            "best_f1": {"threshold": round(float(grid[best_f1]), 5), "f1": series["f1"][best_f1]},
            "youden": {"threshold": round(float(grid[youden]), 5), "balanced_accuracy": series["balanced_accuracy"][youden]},
            "at": {"threshold": round(float(threshold), 5), **c, **{k_: round(float(v), 5) for k_, v in _rates(c).items()}},
            "positive_rate": round(float((p1 >= threshold).mean()), 5),
        }
    return out


def stored_curves(m: FittedModel, settings: Settings) -> dict[str, Any] | None:
    """Small ROC/PR payload saved with a run so runs can be overlaid later without the fitted model."""
    if m.regression:
        return None
    try:
        small = Settings(**{**settings.model_dump(), "curve_max_points": settings.curve_points_stored})
        full = curves_payload(m, small)
    except Exception:  # noqa: BLE001 - curves are optional
        return None
    return {"binary": full["binary"], "roc": full["roc"], "pr": full["pr"]}


def residuals_payload(m: FittedModel, settings: Settings) -> dict[str, Any]:
    """Actual-vs-predicted scatter and residual histogram (original target units)."""
    if not m.regression:
        raise BadRequestError("Residual analysis exists only for regression models.", error_code="NOT_REGRESSION")
    pred = np.asarray(m.pipeline.predict(m.x_test), dtype="float64") * m.y_sd + m.y_mu
    actual = m.y_test * m.y_sd + m.y_mu
    resid = actual - pred
    idx = select_indices(len(actual), 400)
    counts, edges = np.histogram(resid, bins=20)
    return {
        "model": m.key, "abbr": m.abbr, "n_test": int(len(actual)),
        "scatter": {"actual": [round(float(actual[i]), 5) for i in idx], "predicted": [round(float(pred[i]), 5) for i in idx]},
        "histogram": {"edges": [round(float(e), 5) for e in edges], "counts": [int(c) for c in counts]},
        "bias": round(float(resid.mean()), 6), "residual_std": round(float(resid.std()), 6),
        "r2": round(float(r2_score(actual, pred)), 6),
        "worst": sorted(
            [{"index": int(i), "actual": round(float(actual[i]), 4), "predicted": round(float(pred[i]), 4)} for i in np.argsort(-np.abs(resid))[:5]],
            key=lambda r: -abs(r["actual"] - r["predicted"]),
        ),
    }


# --- feature naming -----------------------------------------------------------------------------------------
def transformed_features(prep: Any) -> tuple[list[str], list[str]]:
    """Names of the columns the estimator sees, and the original column each one came from."""
    names: list[str] = []
    owner: list[str] = []
    for name, trans, cols in prep.transformers_:
        if name == "remainder" or not len(cols):
            continue
        if name == "ohe":
            for col, cats in zip(cols, trans.categories_):
                for cat in cats:
                    names.append(f"{col}={cat}")
                    owner.append(col)
        else:
            names.extend(cols)
            owner.extend(cols)
    return names, owner


def _aggregate(values: np.ndarray, owner: list[str]) -> tuple[np.ndarray, list[str]]:
    """Sum columns that share an original feature (one-hot levels collapse into their column)."""
    order = list(dict.fromkeys(owner))
    out = np.zeros(values.shape[:-1] + (len(order),))
    pos = {c: i for i, c in enumerate(order)}
    for j, col in enumerate(owner):
        out[..., pos[col]] += values[..., j]
    return out, order


# --- importance ----------------------------------------------------------------------------------------------
def native_importance(m: FittedModel) -> list[dict[str, Any]] | None:
    """Model-internal importances (tree gain / coefficient size / TabNet masks) per original column."""
    est = m.pipeline.named_steps["clf"]
    est = getattr(est, "model_", est) if hasattr(est, "model_") and hasattr(est.model_, "coef_") else est
    raw = getattr(est, "feature_importances_", None)
    if raw is None and hasattr(est, "coef_"):
        raw = np.abs(np.asarray(est.coef_, dtype="float64"))
        raw = raw.mean(axis=0) if raw.ndim > 1 else raw
    if raw is None:
        return None
    raw = np.asarray(raw, dtype="float64").ravel()
    _, owner = transformed_features(m.pipeline.named_steps["prep"])
    if len(raw) != len(owner):
        return None
    agg, cols = _aggregate(np.abs(raw), owner)
    total = float(agg.sum()) or 1.0
    ranked = sorted(zip(cols, agg / total), key=lambda p: -p[1])
    return [{"feature": c, "importance": round(float(v), 6)} for c, v in ranked]


def importance_payload(m: FittedModel, settings: Settings, repeats: int | None = None) -> dict[str, Any]:
    """Permutation importance on the test rows (drop in score when a column is shuffled) plus native importance."""
    repeats = repeats or settings.perm_repeats
    x, y = m.x_test, m.y_test
    if len(y) > settings.perm_max_rows:
        pick = np.sort(np.random.default_rng(settings.random_seed).choice(len(y), settings.perm_max_rows, replace=False))
        x, y = x.iloc[pick], y[pick]
    scoring = "r2" if m.regression else "balanced_accuracy"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = permutation_importance(
            m.pipeline, x, y, scoring=scoring, n_repeats=repeats, random_state=settings.random_seed, n_jobs=1
        )
    order = np.argsort(-result.importances_mean)
    return {
        "model": m.key, "abbr": m.abbr, "method": "permutation", "metric": scoring, "repeats": repeats,
        "n_rows": int(len(y)),
        "features": [
            {"feature": x.columns[i], "importance": round(float(result.importances_mean[i]), 6),
             "std": round(float(result.importances_std[i]), 6)}
            for i in order
        ],
        "native": native_importance(m),
    }


# --- SHAP ----------------------------------------------------------------------------------------------------
_TREE = {
    "XGBClassifier", "XGBRegressor", "LGBMClassifier", "LGBMRegressor", "CatBoostClassifier", "CatBoostRegressor",
    "RandomForestClassifier", "RandomForestRegressor", "ExtraTreesClassifier", "ExtraTreesRegressor",
    "GradientBoostingClassifier", "GradientBoostingRegressor", "DecisionTreeClassifier", "DecisionTreeRegressor",
}
_MARGIN = {"XGBClassifier", "LGBMClassifier", "CatBoostClassifier", "GradientBoostingClassifier"}
_LINEAR = {
    "LogisticRegression", "LinearRegression", "Ridge", "RidgeClassifier", "SGDClassifier", "SGDRegressor",
    "LinearDiscriminantAnalysis",
}


def _to_cube(values: Any) -> np.ndarray:
    """Normalise shap's many output shapes to ``(n_rows, n_features, n_outputs)``."""
    if isinstance(values, list):
        return np.stack([np.asarray(v) for v in values], axis=-1)
    arr = np.asarray(values)
    return arr[..., None] if arr.ndim == 2 else arr


def _base_vector(expected: Any, n_out: int) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(expected, dtype="float64")).ravel()
    return arr if len(arr) == n_out else np.repeat(arr[:1], n_out)


def _explain(m: FittedModel, xs: np.ndarray, background: np.ndarray, settings: Settings) -> tuple[np.ndarray, np.ndarray, str, str]:
    """``(cube, base, explainer_name, output_space)`` for the transformed sample ``xs``."""
    import shap

    est = m.pipeline.named_steps["clf"]
    name = type(est).__name__
    inner = getattr(est, "model_", None) if name == "ThresholdedLinearRegression" else None
    if name in _TREE:
        try:
            ex = shap.TreeExplainer(est)
            cube = _to_cube(ex.shap_values(xs, check_additivity=False))
            space = "target units" if m.regression else ("log-odds" if name in _MARGIN else "probability")
            return cube, _base_vector(ex.expected_value, cube.shape[-1]), "TreeExplainer", space
        except Exception:  # noqa: BLE001 - e.g. multiclass sklearn GBM: use the model-agnostic path
            logger.info("TreeExplainer failed for %s, using KernelExplainer", name, exc_info=True)
    elif name in _LINEAR or inner is not None:
        try:
            ex = shap.LinearExplainer(inner or est, background)
            cube = _to_cube(ex.shap_values(xs))
            space = "target units" if m.regression else ("log-odds" if name == "LogisticRegression" else "decision score")
            return cube, _base_vector(ex.expected_value, cube.shape[-1]), "LinearExplainer", space
        except Exception:  # noqa: BLE001
            logger.info("LinearExplainer failed for %s, using KernelExplainer", name, exc_info=True)
    if m.regression:
        f = lambda a: np.asarray(est.predict(a), dtype="float64")  # noqa: E731
    elif hasattr(est, "predict_proba"):
        f = est.predict_proba
    else:
        f = lambda a: np.asarray(est.decision_function(a), dtype="float64")  # noqa: E731
    ex = shap.KernelExplainer(f, shap.kmeans(background, min(15, len(background))))
    cube = _to_cube(ex.shap_values(xs[: settings.shap_kernel_rows], nsamples=settings.shap_kernel_nsamples, silent=True))
    space = "target units" if m.regression else ("probability" if hasattr(est, "predict_proba") else "decision score")
    return cube, _base_vector(ex.expected_value, cube.shape[-1]), "KernelExplainer", space


def _shap_cache(m: FittedModel, settings: Settings) -> dict[str, Any]:
    if "shap" in m.cache:
        return m.cache["shap"]
    prep = m.pipeline.named_steps["prep"]
    rng = np.random.default_rng(settings.random_seed)
    n = len(m.x_test)
    take = np.sort(rng.choice(n, min(n, settings.shap_max_samples), replace=False))
    sample = m.x_test.iloc[take]
    bg_n = min(len(m.x_train), settings.shap_background)
    background = np.asarray(prep.transform(m.x_train.iloc[np.sort(rng.choice(len(m.x_train), bg_n, replace=False))]), dtype="float64")
    xs = np.asarray(prep.transform(sample), dtype="float64")
    cube, base, explainer, space = _explain(m, xs, background, settings)
    _, owner = transformed_features(prep)
    agg, cols = _aggregate(np.moveaxis(cube, -1, 0), owner)  # (outputs, rows, features)
    agg = np.moveaxis(agg, 0, -1)
    if m.regression:
        agg, base = agg * m.y_sd, base * m.y_sd + m.y_mu
    rows = sample.iloc[: agg.shape[0]]
    m.cache["shap"] = {
        "values": agg, "base": base, "features": cols, "sample": rows, "positions": take[: agg.shape[0]],
        "explainer": explainer, "space": space,
    }
    return m.cache["shap"]


def shap_payload(m: FittedModel, settings: Settings, class_index: int | None = None, row: int = 0) -> dict[str, Any]:
    """Global (mean |SHAP|, beeswarm points) and local (one row's contributions) explanations."""
    c = _shap_cache(m, settings)
    values: np.ndarray = c["values"]
    n_out = values.shape[-1]
    binary = not m.regression and len(m.class_labels) == 2
    if m.regression:
        k = 0
        v, base = values[..., 0], float(c["base"][0])
    else:
        k = class_index if class_index is not None else (1 if binary else 0)
        k = min(max(k, 0), len(m.class_labels) - 1)
        if n_out == 1 and binary:  # one output = the positive class; the other class is its mirror image
            sign = 1.0 if k == 1 else -1.0
            v, base = sign * values[..., 0], sign * float(c["base"][0])
        else:
            kk = min(k, n_out - 1)
            v, base = values[..., kk], float(c["base"][min(kk, len(c["base"]) - 1)])
    sample: pd.DataFrame = c["sample"]
    feats: list[str] = c["features"]
    mean_abs = np.abs(v).mean(axis=0)
    order = np.argsort(-mean_abs)
    top = order[: settings.shap_top_features]
    swarm = []
    for j in top:
        col = sample[feats[j]]
        norm: list[float | None]
        if ptypes.is_numeric_dtype(col):
            arr = col.to_numpy(dtype="float64")
            lo, hi = np.percentile(arr, [5, 95])
            norm = [None if hi == lo else round(float(np.clip((a - lo) / (hi - lo), 0, 1)), 4) for a in arr]
        else:
            norm = [None] * len(col)
        swarm.append({
            "feature": feats[j],
            "points": [{"shap": round(float(v[i, j]), 5), "norm": norm[i], "value": _scalar(col.iloc[i])} for i in range(len(v))],
        })
    row = min(max(row, 0), len(v) - 1)
    contrib = sorted(
        [{"feature": feats[j], "value": _scalar(sample[feats[j]].iloc[row]), "shap": round(float(v[row, j]), 5)} for j in range(len(feats))],
        key=lambda d: -abs(d["shap"]),
    )
    shown = contrib[: settings.shap_top_features]
    rest = contrib[settings.shap_top_features :]
    pred = m.pipeline.predict(sample.iloc[[row]])[0]
    label = float(pred) * m.y_sd + m.y_mu if m.regression else m.class_labels[int(pred)]
    return {
        "model": m.key, "abbr": m.abbr, "explainer": c["explainer"], "output_space": c["space"],
        "classes": m.class_labels, "class_index": k, "class_label": None if m.regression else m.class_labels[k],
        "n_samples": int(len(v)),
        "global": [{"feature": feats[j], "mean_abs": round(float(mean_abs[j]), 6)} for j in order],
        "swarm": swarm,
        "rows": [
            {"i": i, "actual": _actual(m, int(c["positions"][i])), "predicted": _scalar(
                (float(p) * m.y_sd + m.y_mu) if m.regression else m.class_labels[int(p)])}
            for i, p in enumerate(m.pipeline.predict(sample))
        ][:150],
        "local": {
            "row": row, "base": round(base, 5), "output": round(base + float(v[row].sum()), 5), "prediction": _scalar(label),
            "contributions": shown, "other": {"count": len(rest), "shap": round(float(sum(r["shap"] for r in rest)), 5)},
        },
    }


def _actual(m: FittedModel, pos: int) -> Any:
    y = m.y_test[pos]
    return _scalar(float(y) * m.y_sd + m.y_mu) if m.regression else m.class_labels[int(y)]


def _scalar(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        return round(float(value), 5)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return str(value)
