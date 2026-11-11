"""Model lab: fit one model with hand-set hyper-parameters and return deep diagnostics for it.

The lab never touches the dashboard's trained results. Each request re-reads the working data, splits it the same
way (seeded), fits one model and returns metrics plus charts that are specific to that model: coefficients with
confidence intervals, regularization paths, the fitted tree, boosting loss curves, out-of-bag growth, neighbour
explanations, decision surfaces, cluster profiles. Sweeps (validation curves, 2-D score grids, learning curves) are
a separate request because they refit the model many times; they use cross-validation on training rows only.
"""
from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, calinski_harabasz_score, confusion_matrix,
    davies_bouldin_score, f1_score, log_loss, mean_absolute_error, mean_squared_error, precision_recall_curve,
    precision_score, r2_score, recall_score, roc_auc_score, roc_curve, silhouette_samples, silhouette_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from app.config import Settings
from app.errors import BadRequestError, ConflictError
from app.services import export
from app.services.analysis import FittedModel, _confusion, _rates, score_matrix, transformed_features
from app.services.lab_catalog import (
    CLASSIFICATION, CLUSTERING, REGRESSION, LabModel, Param, build_estimator, get_lab_model, lab_task,
)
from app.services.modeling import _to_model_frame, cap_rows, identify_exclusions, prepare_dataset, split_indices
from app.state.session_store import Session
from app.utils.downsample import select_indices

logger = logging.getLogger("mldash.lab")

LINEAR = ("linear_regression", "logistic_regression")


# --- data ------------------------------------------------------------------------------------------------------
@dataclass
class LabData:
    """The (capped) model frame, target and seeded split the lab works on."""

    task: str
    X: pd.DataFrame
    y: np.ndarray | None  # class codes, original target values (regression) or None (clustering without target)
    class_labels: list[str]
    train: np.ndarray
    test: np.ndarray
    available: list[str]
    used: list[str]
    rows_total: int
    capped: bool
    exclusions: dict[str, list[str]]
    y_mu: float = 0.0
    y_sd: float = 1.0

    @property
    def yfit(self) -> np.ndarray:
        """Target as the estimator sees it (standardised with training statistics for regression)."""
        if self.task == REGRESSION:
            return (self.y - self.y_mu) / self.y_sd
        return self.y


def _check_nans(df: pd.DataFrame, columns: list[str]) -> None:
    nan_counts = {c: int(df[c].isna().sum()) for c in columns if df[c].isna().any()}
    if nan_counts:
        raise ConflictError("Missing values remain; impute them on the dashboard (panel 01) first.",
                            error_code="NAN_REMAINING", details=nan_counts)


def load_data(session: Session, settings: Settings, task: str, features: list[str] | None, test_size: float) -> LabData:
    """Validate the working frame and return the lab's capped data and split."""
    df = session.working_df
    seed = settings.random_seed
    labels: list[str] = []
    if task == CLUSTERING:
        target = session.target if session.target in df.columns else None
        exclusions = identify_exclusions(df, target or "")
        dropped = set(exclusions["id_like"]) | set(exclusions["constant"])
        available = [c for c in df.columns if c != target and c not in dropped]
        _check_nans(df, available)
        X = _to_model_frame(df[available]).reset_index(drop=True)
        y = None
        if target is not None and not df[target].isna().any():
            if session.task == REGRESSION:
                y = df[target].to_numpy(dtype="float64")
            else:
                codes, uniques = pd.factorize(df[target], sort=True)
                y, labels = np.asarray(codes, dtype=int), [str(u) for u in uniques]
    else:
        try:
            prepared = prepare_dataset(df, session.target, settings, task, session.positive_class)
        except ConflictError as exc:
            if exc.error_code != "NAN_REMAINING":
                raise
            raise ConflictError("Missing values remain; impute them on the dashboard (panel 01) first.",
                                error_code="NAN_REMAINING", details=exc.details) from exc
        X, y, labels, exclusions = prepared.X, prepared.y, prepared.class_labels, prepared.exclusions
        available = list(X.columns)
    used = available
    if features:
        wanted = set(features)
        used = [c for c in available if c in wanted]
        if not used:
            raise BadRequestError("None of the chosen features are usable. Pick at least one feature.",
                                  error_code="NO_FEATURES", details={"available": available})
        X = X[used]
    rows_total = len(X)
    idx = np.arange(rows_total)
    capped = False
    if rows_total > settings.lab_max_rows:
        strat = y if (y is not None and task != REGRESSION) else np.zeros(rows_total, dtype=int)
        idx, _, capped = cap_rows(idx, strat, settings.lab_max_rows, seed, task == CLASSIFICATION)
        X = X.iloc[idx].reset_index(drop=True)
        y = y[idx] if y is not None else None
    if task == CLUSTERING:
        all_rows = np.arange(len(X))
        return LabData(task, X, y, labels, all_rows, np.array([], dtype=int), available, used, rows_total, capped, exclusions)
    train, test, _, _ = split_indices(np.arange(len(X)), y, test_size, seed, stratify=task == CLASSIFICATION)
    data = LabData(task, X, y, labels, np.sort(train), np.sort(test), available, used, rows_total, capped, exclusions)
    if task == REGRESSION:
        data.y_mu = float(np.mean(y[data.train]))
        data.y_sd = float(np.std(y[data.train])) or 1.0
    return data


def lab_preprocessor(x: pd.DataFrame, scale: bool, settings: Settings, drop_first: bool = False) -> ColumnTransformer:
    """Scaling / one-hot / ordinal encoding (fit on training rows only). Linear models drop one level per
    one-hot column so the coefficients are identifiable (the dropped level is the baseline)."""
    numeric = [c for c in x.columns if ptypes.is_numeric_dtype(x[c])]
    categorical = [c for c in x.columns if c not in numeric]
    low = [c for c in categorical if x[c].nunique() <= settings.onehot_max_cardinality]
    high = [c for c in categorical if c not in low]
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first" if drop_first else None)
    return ColumnTransformer(
        [
            ("num", StandardScaler() if scale else "passthrough", numeric),
            ("ohe", ohe, low),
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), high),
        ],
        remainder="drop",
    )


def _scale_info(prep: ColumnTransformer) -> list[tuple[float, float] | None]:
    """Per transformed column: (mean, scale) to undo standardisation of a threshold, or None."""
    info: list[tuple[float, float] | None] = []
    for name, trans, cols in prep.transformers_:
        if name == "remainder" or not len(cols):
            continue
        if name == "num":
            if isinstance(trans, StandardScaler):
                info.extend((float(m), float(s)) for m, s in zip(trans.mean_, trans.scale_))
            else:
                info.extend([None] * len(cols))
        elif name == "ohe":
            info.extend([None] * sum(len(c) - (1 if trans.drop_idx_ is not None and trans.drop_idx_[i] is not None else 0)
                                     for i, c in enumerate(trans.categories_)))
        else:
            info.extend([None] * len(cols))
    return info


def _names(prep: ColumnTransformer) -> tuple[list[str], list[str]]:
    """Transformed column names and owners, honouring dropped one-hot levels."""
    names, owner = transformed_features(prep)
    ohe = prep.named_transformers_.get("ohe")
    if ohe is None or getattr(ohe, "drop_idx_", None) is None:
        return names, owner
    dropped = set()
    for col, cats, d in zip(prep.transformers_[1][2], ohe.categories_, ohe.drop_idx_):
        if d is not None:
            dropped.add(f"{col}={cats[d]}")
    keep = [i for i, n in enumerate(names) if n not in dropped]
    return [names[i] for i in keep], [owner[i] for i in keep]


def _aggregate(values: np.ndarray, owner: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for v, o in zip(values, owner):
        out[o] = out.get(o, 0.0) + float(v)
    return out


# --- scoring ---------------------------------------------------------------------------------------------------
def primary_scoring(task: str, n_classes: int) -> tuple[str, str]:
    """sklearn scorer name and its label for the lab's headline metric."""
    if task == REGRESSION:
        return "r2", "R2"
    return ("f1", "F1") if n_classes == 2 else ("f1_weighted", "F1 (weighted)")


def _primary(task: str, n_classes: int, y: np.ndarray, pred: np.ndarray) -> float:
    if task == REGRESSION:
        return float(r2_score(y, pred))
    return float(f1_score(y, pred, average="binary" if n_classes == 2 else "weighted", zero_division=0))


def _clf_metrics(y: np.ndarray, pred: np.ndarray, scores: np.ndarray | None, kind: str | None, k: int) -> dict[str, Any]:
    avg = "binary" if k == 2 else "weighted"
    out: dict[str, Any] = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, average=avg, zero_division=0),
        "recall": recall_score(y, pred, average=avg, zero_division=0),
        "f1": f1_score(y, pred, average=avg, zero_division=0),
        "roc_auc": None, "log_loss": None,
    }
    if scores is not None:
        try:
            if k == 2:
                out["roc_auc"] = roc_auc_score(y, scores[:, 1])
            else:
                out["roc_auc"] = roc_auc_score(y, scores, multi_class="ovr", average="macro", labels=np.arange(k))
        except ValueError:
            pass
        if kind == "probability":
            try:
                p = np.clip(scores, 1e-15, 1)
                out["log_loss"] = log_loss(y, p / p.sum(axis=1, keepdims=True), labels=np.arange(k))
            except ValueError:
                pass
    return {key: (None if v is None else round(float(v), 6)) for key, v in out.items()}


def _reg_metrics(y: np.ndarray, pred: np.ndarray, n_features: int) -> dict[str, Any]:
    r2 = float(r2_score(y, pred))
    n = len(y)
    adj = 1 - (1 - r2) * (n - 1) / (n - n_features - 1) if n - n_features - 1 > 0 else None
    return {
        "r2": round(r2, 6), "adj_r2": None if adj is None else round(float(adj), 6),
        "rmse": round(float(np.sqrt(mean_squared_error(y, pred))), 6), "mae": round(float(mean_absolute_error(y, pred)), 6),
        "target_std": round(float(np.std(y)), 6),
    }


def _cv(model: LabModel, task: str, params: dict[str, Any], data: LabData, scale: bool, settings: Settings) -> dict[str, Any] | None:
    """k-fold cross-validated headline score on (a sample of) the training rows."""
    seed = settings.random_seed
    tr = data.train
    y = data.yfit[tr]
    pick, _, _ = cap_rows(tr, y, settings.lab_cv_max_rows, seed, task == CLASSIFICATION)
    x, y = data.X.iloc[pick], data.yfit[pick]
    folds = settings.lab_cv_folds
    if task == CLASSIFICATION:
        folds = min(folds, int(np.bincount(y).min()))
        if folds < 2:
            return None
        splitter: Any = StratifiedKFold(folds, shuffle=True, random_state=seed)
    else:
        splitter = KFold(folds, shuffle=True, random_state=seed)
    scoring, label = primary_scoring(task, len(data.class_labels))
    pipe = Pipeline([("prep", lab_preprocessor(x, scale, settings, model.key in LINEAR)),
                     ("clf", build_estimator(model, task, params, seed, settings.n_jobs))])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = cross_validate(pipe, x, y, cv=splitter, scoring=scoring, error_score=np.nan)
    scores = np.asarray(res["test_score"], dtype=float)
    if np.all(np.isnan(scores)):
        return None
    return {"metric": label, "folds": [round(float(s), 6) for s in scores], "mean": round(float(np.nanmean(scores)), 6),
            "std": round(float(np.nanstd(scores)), 6), "rows": int(len(y))}


# --- shared charts ---------------------------------------------------------------------------------------------
def _thin(arr: np.ndarray, limit: int) -> list[float]:
    return [round(float(arr[i]), 5) for i in select_indices(len(arr), limit)]


def _hist(values: np.ndarray, bins: int = 20, rng: tuple[float, float] | None = None) -> dict[str, Any]:
    counts, edges = np.histogram(values, bins=bins, range=rng)
    return {"edges": [round(float(e), 5) for e in edges], "counts": [int(c) for c in counts]}


def _clf_performance(y: np.ndarray, pred: np.ndarray, scores: np.ndarray | None, kind: str | None, labels: list[str],
                     settings: Settings) -> dict[str, Any]:
    k = len(labels)
    out: dict[str, Any] = {
        "labels": labels, "binary": k == 2, "score_kind": kind,
        "confusion": confusion_matrix(y, pred, labels=np.arange(k)).tolist(),
        "roc": [], "pr": None, "thresholds": None, "calibration": None, "separation": None,
    }
    if scores is None:
        return out
    for c in ([1] if k == 2 else range(min(k, 8))):
        t = (y == c).astype(int)
        if t.min() == t.max():
            continue
        fpr, tpr, _ = roc_curve(t, scores[:, c])
        idx = select_indices(len(fpr), settings.curve_max_points)
        out["roc"].append({"label": labels[c], "fpr": [round(float(fpr[i]), 5) for i in idx],
                           "tpr": [round(float(tpr[i]), 5) for i in idx], "auc": round(float(roc_auc_score(t, scores[:, c])), 6)})
    if k != 2:
        return out
    p1 = scores[:, 1]
    prec, rec, _ = precision_recall_curve(y, p1)
    idx = select_indices(len(prec), settings.curve_max_points)
    out["pr"] = {"precision": [round(float(prec[i]), 5) for i in idx], "recall": [round(float(rec[i]), 5) for i in idx],
                 "ap": round(float(average_precision_score(y, p1)), 6), "prevalence": round(float(y.mean()), 6)}
    grid = np.linspace(0.0, 1.0, settings.threshold_points)
    rows = []
    for t in grid:
        c = _confusion(y, (p1 >= t).astype(int))
        rows.append({**c, **_rates(c)})
    out["thresholds"] = {"grid": [round(float(t), 4) for t in grid],
                         **{name: [round(float(r[name]), 5) for r in rows] for name in rows[0]}}
    out["separation"] = {"negative": _hist(p1[y == 0], 20, (0.0, 1.0))["counts"],
                         "positive": _hist(p1[y == 1], 20, (0.0, 1.0))["counts"],
                         "edges": [round(float(e), 3) for e in np.linspace(0, 1, 21)]}
    if kind == "probability":
        bins = np.linspace(0, 1, 11)
        which = np.clip(np.digitize(p1, bins) - 1, 0, 9)
        cal = []
        for b in range(10):
            mask = which == b
            if mask.sum():
                cal.append({"bin": b, "predicted": round(float(p1[mask].mean()), 5), "observed": round(float(y[mask].mean()), 5),
                            "count": int(mask.sum())})
        brier = float(np.mean((p1 - y) ** 2))
        out["calibration"] = {"bins": cal, "brier": round(brier, 6)}
    return out


def _reg_performance(actual: np.ndarray, pred: np.ndarray, settings: Settings) -> dict[str, Any]:
    resid = actual - pred
    idx = select_indices(len(actual), settings.lab_plot_points)
    sd = float(resid.std()) or 1.0
    z = np.sort((resid - resid.mean()) / sd)
    q_idx = select_indices(len(z), 120)
    theo = stats.norm.ppf((np.arange(1, len(z) + 1) - 0.5) / len(z))
    worst = np.argsort(-np.abs(resid))[:5]
    return {
        "scatter": [{"actual": round(float(actual[i]), 5), "predicted": round(float(pred[i]), 5),
                     "residual": round(float(resid[i]), 5)} for i in idx],
        "histogram": _hist(resid, 24),
        "qq": [{"theoretical": round(float(theo[i]), 4), "sample": round(float(z[i]), 4)} for i in q_idx],
        "bias": round(float(resid.mean()), 6), "residual_std": round(sd, 6),
        "skew": round(float(stats.skew(resid)), 4) if len(resid) > 2 else None,
        "worst": [{"row": int(i), "actual": round(float(actual[i]), 4), "predicted": round(float(pred[i]), 4)} for i in worst],
    }


def _boundary(model: LabModel, task: str, params: dict[str, Any], Xt: np.ndarray, y: np.ndarray, data: LabData,
              settings: Settings, support: bool = False) -> dict[str, Any] | None:
    """Decision surface of a surrogate model refit on the first two principal components."""
    if Xt.shape[1] < 2:
        return None
    seed = settings.random_seed
    pick, _, _ = cap_rows(np.arange(len(y)), y, 2000, seed, task == CLASSIFICATION)
    xs, ys = Xt[pick], y[pick]
    pca = PCA(2, random_state=seed).fit(xs)
    z = pca.transform(xs)
    sur = build_estimator(model, task, params, seed, settings.n_jobs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sur.fit(z, ys)
    n = settings.lab_boundary_resolution
    lo, hi = z.min(axis=0), z.max(axis=0)
    pad = (hi - lo) * 0.05 + 1e-9
    gx = np.linspace(lo[0] - pad[0], hi[0] + pad[0], n)
    gy = np.linspace(lo[1] - pad[1], hi[1] + pad[1], n)
    mesh = np.array([[a, b] for b in gy for a in gx])
    gp = np.asarray(sur.predict(mesh), dtype=float)
    sur_pred = np.asarray(sur.predict(z), dtype=float)
    if task == REGRESSION:
        gp = gp * data.y_sd + data.y_mu
        fidelity = float(r2_score(ys, sur_pred))
        point_value = ys * data.y_sd + data.y_mu
    else:
        fidelity = float(accuracy_score(ys, sur_pred))
        point_value = ys
    show = select_indices(len(z), settings.lab_plot_points)
    sv = set(int(i) for i in getattr(sur, "support_", [])) if support else set()
    return {
        "x": [round(float(v), 4) for v in gx], "y": [round(float(v), 4) for v in gy],
        "grid": [round(float(v), 4) for v in gp], "n": n,
        "points": [{"x": round(float(z[i, 0]), 4), "y": round(float(z[i, 1]), 4), "v": round(float(point_value[i]), 4),
                    "sv": i in sv} for i in show],
        "explained": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "fidelity": round(fidelity, 4), "fidelity_metric": "R2" if task == REGRESSION else "accuracy",
        "n_support": len(sv) if support else None,
    }


# --- trees -----------------------------------------------------------------------------------------------------
def _test_text(name: str, thr: float, scale: tuple[float, float] | None, left: bool) -> str:
    """Readable branch condition. One-hot columns read 'col = level' / 'col != level'."""
    if "=" in name and abs(thr - 0.5) < 1e-6 and scale is None:
        col, level = name.split("=", 1)
        return f"{col} != {level}" if left else f"{col} = {level}"
    value = thr * scale[1] + scale[0] if scale else thr
    shown = f"{value:.4g}"
    return f"{name} <= {shown}" if left else f"{name} > {shown}"


def _node_counts(tree: Any, Xt: np.ndarray | None, y: np.ndarray | None, k: int) -> np.ndarray | None:
    """Real (unweighted) training rows of each class reaching every node: ``(n_nodes, k)``.

    ``tree_.value`` holds class *weights*, so with class_weight='balanced' (or a bootstrap) its shares are not the
    share of actual rows; routing the training rows through the tree gives the counts a reader expects.
    """
    if Xt is None or y is None or k < 2:
        return None
    try:
        path = tree.decision_path(np.asarray(Xt, dtype="float32"))
        return np.asarray(path.T @ np.eye(k)[np.asarray(y, dtype=int)])
    except Exception:  # noqa: BLE001 - counts are extra information
        logger.debug("node counts unavailable", exc_info=True)
        return None


def _class_info(t: Any, i: int, labels: list[str], counts: np.ndarray | None) -> dict[str, Any]:
    """Prediction (what the model decides, class weights included) plus the real class mix of the rows."""
    v = t.value[i]
    dist = v[0] / (v[0].sum() or 1.0)
    j = int(np.argmax(dist))
    out: dict[str, Any] = {"pred": labels[j] if j < len(labels) else str(j), "dist": [round(float(d), 4) for d in dist],
                           "confidence": round(float(dist[j]), 4)}
    if counts is not None:
        c = counts[i]
        total = float(c.sum()) or 1.0
        out["counts"] = [int(x) for x in c]
        out["mix"] = [round(float(x) / total, 4) for x in c]
    return out


def _sk_tree(tree: Any, names: list[str], scales: list[Any], data: LabData, task: str, depth_limit: int,
             Xt: np.ndarray | None = None, y: np.ndarray | None = None, weighted: bool = False) -> dict[str, Any]:
    """Nested node dict of a fitted sklearn tree (down to ``depth_limit``), thresholds in original units.

    Classification nodes carry ``pred`` (the model's decision), ``counts`` / ``mix`` (real training rows per class)
    and ``dist`` (the class-weighted shares the tree itself uses).
    """
    t = tree.tree_
    labels = data.class_labels
    counts = _node_counts(tree, Xt, y, len(labels)) if task == CLASSIFICATION else None

    def value_of(i: int) -> dict[str, Any]:
        if task == CLASSIFICATION:
            return _class_info(t, i, labels, counts)
        return {"pred": round(float(t.value[i][0][0]) * data.y_sd + data.y_mu, 5)}

    def walk(i: int, depth: int) -> dict[str, Any]:
        rows = int(counts[i].sum()) if counts is not None else int(t.n_node_samples[i])
        node: dict[str, Any] = {"id": int(i), "samples": rows, "impurity": round(float(t.impurity[i]), 5),
                                **value_of(i)}
        left, right = int(t.children_left[i]), int(t.children_right[i])
        if left == -1:
            node["leaf"] = True
            return node
        f, thr = int(t.feature[i]), float(t.threshold[i])
        node.update({"leaf": False, "feature": names[f], "yes": _test_text(names[f], thr, scales[f], True),
                     "no": _test_text(names[f], thr, scales[f], False)})
        if depth >= depth_limit:
            node["hidden"] = int(_count_leaves(t, i))
            return node
        node["children"] = [walk(left, depth + 1), walk(right, depth + 1)]
        return node

    root = walk(0, 0)
    out = {"root": root, "depth": int(tree.get_depth()), "leaves": int(tree.get_n_leaves()), "shown_depth": depth_limit,
           "total": root["samples"], "weighted": bool(weighted and task == CLASSIFICATION)}
    if task != CLASSIFICATION:  # colour scale for predicted values
        preds = [n["pred"] for n in _iter_nodes(root)]
        out["pred_range"] = [min(preds), max(preds)]
    return out


def _iter_nodes(node: dict[str, Any]) -> list[dict[str, Any]]:
    found = [node]
    for child in node.get("children", []):
        found.extend(_iter_nodes(child))
    return found


def _count_leaves(t: Any, i: int) -> int:
    stack, n = [i], 0
    while stack:
        j = stack.pop()
        if t.children_left[j] == -1:
            n += 1
        else:
            stack.extend([t.children_left[j], t.children_right[j]])
    return n


def _rules(tree: Any, names: list[str], scales: list[Any], data: LabData, task: str, limit: int = 12,
           Xt: np.ndarray | None = None, y: np.ndarray | None = None) -> list[dict[str, Any]]:
    """The largest leaves as if-then rules (``mix`` = real share of each class among the training rows there)."""
    t = tree.tree_
    counts = _node_counts(tree, Xt, y, len(data.class_labels)) if task == CLASSIFICATION else None
    out: list[dict[str, Any]] = []
    stack: list[tuple[int, list[str]]] = [(0, [])]
    while stack:
        i, path = stack.pop()
        left, right = t.children_left[i], t.children_right[i]
        if left == -1:
            v = t.value[i]
            extra: dict[str, Any] = {}
            rows = int(counts[i].sum()) if counts is not None else int(t.n_node_samples[i])
            if task == CLASSIFICATION:
                info = _class_info(t, i, data.class_labels, counts)
                pred: Any = info["pred"]
                # confidence = real share of the predicted class among these rows (weighted share if unavailable)
                j = data.class_labels.index(pred) if pred in data.class_labels else 0
                conf: float | None = info["mix"][j] if "mix" in info else info["confidence"]
                extra = {k: info[k] for k in ("counts", "mix") if k in info}
            else:
                pred, conf = round(float(v[0][0]) * data.y_sd + data.y_mu, 5), None
            out.append({"conditions": path, "samples": rows, "pred": pred, "confidence": conf, **extra})
            continue
        f, thr = int(t.feature[i]), float(t.threshold[i])
        stack.append((int(right), [*path, _test_text(names[f], thr, scales[f], False)]))
        stack.append((int(left), [*path, _test_text(names[f], thr, scales[f], True)]))
    out.sort(key=lambda r: -r["samples"])
    return out[:limit]


def _xgb_tree(est: Any, index: int, names: list[str], scales: list[Any], depth_limit: int) -> dict[str, Any]:
    """One boosted tree from XGBoost as the same nested structure (leaf values are raw margins)."""
    df = est.get_booster().trees_to_dataframe()
    n_trees = int(df["Tree"].max()) + 1
    index = max(0, min(index, n_trees - 1))
    sub = df[df["Tree"] == index].set_index("ID")

    def fname(raw: str) -> tuple[int, str]:
        j = int(raw[1:]) if raw.startswith("f") and raw[1:].isdigit() else -1
        return j, (names[j] if 0 <= j < len(names) else raw)

    def walk(node_id: str, depth: int) -> dict[str, Any]:
        r = sub.loc[node_id]
        node: dict[str, Any] = {"id": node_id, "samples": round(float(r["Cover"]), 2)}
        if r["Feature"] == "Leaf":
            node.update({"leaf": True, "pred": round(float(r["Gain"]), 5)})
            return node
        j, name = fname(str(r["Feature"]))
        scale = scales[j] if 0 <= j < len(scales) else None
        thr = float(r["Split"])
        # XGBoost sends x < split to "Yes"; one-hot columns split at 1.0 (0 vs 1)
        if "=" in name and scale is None and abs(thr - 1.0) < 1e-6:
            col, level = name.split("=", 1)
            yes, no = f"{col} != {level}", f"{col} = {level}"
        else:
            value = thr * scale[1] + scale[0] if scale else thr
            yes, no = f"{name} < {value:.4g}", f"{name} >= {value:.4g}"
        node.update({"leaf": False, "feature": name, "yes": yes, "no": no, "gain": round(float(r["Gain"]), 4)})
        if depth >= depth_limit:
            node["hidden"] = True
            return node
        node["children"] = [walk(r["Yes"], depth + 1), walk(r["No"], depth + 1)]
        return node

    return {"root": walk(f"{index}-0", 0), "index": index, "n_trees": n_trees, "shown_depth": depth_limit,
            "depth": None, "leaves": int((sub["Feature"] == "Leaf").sum()), "units": "margin (log-odds / scaled target)"}


# --- model-specific charts ------------------------------------------------------------------------------------
def _vif(Xt: np.ndarray, names: list[str], seed: int) -> dict[str, float]:
    """Variance inflation factor per transformed column (on at most 2,000 rows)."""
    if Xt.shape[1] < 2 or Xt.shape[1] > 60:
        return {}
    rng = np.random.default_rng(seed)
    rows = Xt[rng.choice(len(Xt), min(len(Xt), 2000), replace=False)]
    out: dict[str, float] = {}
    for j, name in enumerate(names):
        others = np.column_stack([np.ones(len(rows)), np.delete(rows, j, axis=1)])
        target = rows[:, j]
        coef, *_ = np.linalg.lstsq(others, target, rcond=None)
        resid = target - others @ coef
        tss = float(((target - target.mean()) ** 2).sum())
        r2 = 1 - float((resid ** 2).sum()) / tss if tss > 0 else 1.0
        out[name] = round(1 / (1 - r2), 3) if r2 < 0.999999 else 1e6
    return out


def _linear_charts(est: Any, params: dict[str, Any], Xt_tr: np.ndarray, y_tr: np.ndarray, Xt_te: np.ndarray,
                   y_te: np.ndarray, names: list[str], data: LabData, settings: Settings) -> dict[str, Any]:
    sd, mu = data.y_sd, data.y_mu
    coef = np.ravel(est.coef_) * sd
    se = lo = hi = pv = None
    if params["regularization"] == "none":
        try:
            import statsmodels.api as sm

            design = sm.add_constant(Xt_tr, has_constant="add") if params["fit_intercept"] else Xt_tr
            res = sm.OLS(y_tr, design).fit()
            off = 1 if params["fit_intercept"] else 0
            ci = np.asarray(res.conf_int())
            se, pv = np.asarray(res.bse)[off:] * sd, np.asarray(res.pvalues)[off:]
            lo, hi = ci[off:, 0] * sd, ci[off:, 1] * sd
        except Exception:  # noqa: BLE001 - inference is optional
            logger.debug("OLS inference failed", exc_info=True)
    vif = _vif(Xt_tr, names, settings.random_seed)
    rows = []
    for j, name in enumerate(names):
        rows.append({
            "feature": name, "coef": round(float(coef[j]), 6),
            "se": None if se is None else round(float(se[j]), 6),
            "lo": None if lo is None else round(float(lo[j]), 6), "hi": None if hi is None else round(float(hi[j]), 6),
            "p": None if pv is None else float(pv[j]), "vif": vif.get(name), "zero": bool(abs(coef[j]) < 1e-10),
        })
    rows.sort(key=lambda r: -abs(r["coef"]))
    # the model predicts the standardised target, so the intercept is shifted back by the training mean
    intercept = float(np.ravel([est.intercept_])[0]) * sd + mu if params["fit_intercept"] else 0.0
    active = [r for r in rows if not r["zero"]]
    eq = f"{intercept:.4g}"
    for r in active[:6]:
        eq += f" {'-' if r['coef'] < 0 else '+'} {abs(r['coef']):.3g}*[{r['feature']}]"
    if len(active) > 6:
        eq += f" + ... ({len(active) - 6} more)"
    # regularization path
    from sklearn.linear_model import ElasticNet, Lasso, Ridge

    kind = params["regularization"] if params["regularization"] != "none" else "ridge"
    top_idx = [names.index(r["feature"]) for r in rows[:8]]
    alphas = np.logspace(-4, 2, 19)
    path = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for a in alphas:
            m = (Ridge(alpha=a) if kind == "ridge" else Lasso(alpha=a, max_iter=5000) if kind == "lasso"
                 else ElasticNet(alpha=a, l1_ratio=params["l1_ratio"], max_iter=5000))
            m.set_params(fit_intercept=params["fit_intercept"])
            m.fit(Xt_tr, y_tr)
            c = np.ravel(m.coef_) * sd
            path.append({"alpha": float(a), "test_r2": round(float(r2_score(y_te, m.predict(Xt_te))), 5),
                         "nonzero": int((np.abs(c) > 1e-10).sum()),
                         **{names[j]: round(float(c[j]), 6) for j in top_idx}})
    return {
        "coefficients": rows[: settings.lab_top_features], "n_coefficients": len(rows),
        "intercept": round(intercept, 6) if params["fit_intercept"] else None, "equation": eq,
        "inference": se is not None,
        "path": {"kind": kind, "features": [names[j] for j in top_idx], "points": path,
                 "current": params["alpha"] if params["regularization"] != "none" else None},
        "n_significant": None if pv is None else int((pv < 0.05).sum()),
    }


def _logistic_charts(est: Any, model: LabModel, params: dict[str, Any], Xt_tr: np.ndarray, y_tr: np.ndarray,
                     Xt_te: np.ndarray, y_te: np.ndarray, names: list[str], data: LabData, settings: Settings) -> dict[str, Any]:
    labels = data.class_labels
    coef = np.atleast_2d(est.coef_)
    binary = coef.shape[0] == 1
    classes = [labels[1]] if binary else labels
    se = pv = lo = hi = None
    if binary and params["penalty"] == "none":
        try:
            import statsmodels.api as sm

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = sm.Logit(y_tr, sm.add_constant(Xt_tr, has_constant="add")).fit(disp=0, maxiter=200)
            ci = np.asarray(res.conf_int())
            se, pv, lo, hi = np.asarray(res.bse)[1:], np.asarray(res.pvalues)[1:], ci[1:, 0], ci[1:, 1]
        except Exception:  # noqa: BLE001 - perfect separation etc.
            logger.debug("Logit inference failed", exc_info=True)
    rows = []
    for j, name in enumerate(names):
        c = coef[:, j]
        rows.append({
            "feature": name, "coef": [round(float(v), 6) for v in c], "odds": [round(float(np.exp(v)), 6) for v in c],
            "se": None if se is None else round(float(se[j]), 6), "p": None if pv is None else float(pv[j]),
            "lo": None if lo is None else round(float(lo[j]), 6), "hi": None if hi is None else round(float(hi[j]), 6),
            "zero": bool(np.all(np.abs(c) < 1e-10)),
        })
    rows.sort(key=lambda r: -max(abs(v) for v in r["coef"]))
    # regularization path over C
    penalty = params["penalty"] if params["penalty"] != "none" else "l2"
    top_idx = [names.index(r["feature"]) for r in rows[:8]]
    pick, _, _ = cap_rows(np.arange(len(y_tr)), y_tr, 3000, settings.random_seed, True)
    path = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for c_val in np.logspace(-3, 3, 13):
            p = {**params, "penalty": penalty, "C": float(c_val), "max_iter": min(params["max_iter"], 1000)}
            m = build_estimator(model, CLASSIFICATION, p, settings.random_seed, settings.n_jobs)
            m.fit(Xt_tr[pick], y_tr[pick])
            cm = np.atleast_2d(m.coef_)
            vals = cm[0] if cm.shape[0] == 1 else np.linalg.norm(cm, axis=0)
            path.append({"C": float(c_val), "test_score": round(_primary(CLASSIFICATION, len(labels), y_te, m.predict(Xt_te)), 5),
                         "nonzero": int((np.abs(cm).max(axis=0) > 1e-10).sum()),
                         **{names[j]: round(float(vals[j]), 6) for j in top_idx}})
    return {
        "coefficients": rows[: settings.lab_top_features], "n_coefficients": len(rows), "classes": classes,
        "intercept": [round(float(v), 6) for v in np.ravel(est.intercept_)], "inference": se is not None,
        "n_zero": int(sum(r["zero"] for r in rows)),
        "path": {"penalty": penalty, "features": [names[j] for j in top_idx], "points": path,
                 "current": params["C"] if params["penalty"] != "none" else None, "multiclass_norm": not binary},
        "n_significant": None if pv is None else int((pv < 0.05).sum()),
    }


def _knn_charts(est: Any, params: dict[str, Any], Xt_tr: np.ndarray, y_tr: np.ndarray, Xt_te: np.ndarray, y_te: np.ndarray,
                data: LabData, task: str, row: int, settings: Settings) -> dict[str, Any]:
    n = min(len(Xt_te), 1500)
    dist, _ = est.kneighbors(Xt_te[:n])
    mean_d = dist.mean(axis=1)
    pred = est.predict(Xt_te[:n])
    # error rate by neighbour-distance quintile
    buckets = []
    edges = np.quantile(mean_d, np.linspace(0, 1, 6))
    for b in range(5):
        mask = (mean_d >= edges[b]) & (mean_d <= edges[b + 1]) if b == 4 else (mean_d >= edges[b]) & (mean_d < edges[b + 1])
        if not mask.any():
            continue
        if task == CLASSIFICATION:
            score = float(accuracy_score(y_te[:n][mask], pred[mask]))
        else:
            score = float(mean_absolute_error(y_te[:n][mask], pred[mask])) * data.y_sd
        buckets.append({"bucket": f"Q{b + 1}", "from": round(float(edges[b]), 4), "to": round(float(edges[b + 1]), 4),
                        "score": round(score, 5), "rows": int(mask.sum())})
    # explain one test row
    row = max(0, min(row, len(Xt_te) - 1))
    d_row, i_row = est.kneighbors(Xt_te[row:row + 1])
    x_tr = data.X.iloc[data.train]
    x_te = data.X.iloc[data.test]
    cols = list(data.X.columns)[:8]

    def show(v: Any) -> Any:
        return round(float(v), 4) if isinstance(v, (int, float, np.integer, np.floating)) else str(v)

    def label(v: Any) -> Any:
        if task == CLASSIFICATION:
            return data.class_labels[int(v)]
        return round(float(v) * data.y_sd + data.y_mu, 4)

    neighbours = [{"distance": round(float(d), 4), "label": label(y_tr[j]),
                   "values": {c: show(x_tr.iloc[j][c]) for c in cols}} for d, j in zip(d_row[0], i_row[0])]
    votes: dict[str, Any] = {}
    if task == CLASSIFICATION:
        for nb in neighbours:
            votes[nb["label"]] = votes.get(nb["label"], 0) + 1
    p_row = est.predict(Xt_te[row:row + 1])[0]
    return {
        "distance_hist": _hist(dist[:, -1], 20), "mean_distance": round(float(mean_d.mean()), 4),
        "by_distance": buckets, "by_distance_metric": "accuracy" if task == CLASSIFICATION else "MAE",
        "explain": {"row": row, "n_rows": int(len(Xt_te)), "columns": cols,
                    "values": {c: show(x_te.iloc[row][c]) for c in cols},
                    "actual": label(y_te[row]), "predicted": label(p_row), "votes": votes, "neighbours": neighbours[:30]},
    }


def _xgb_fit(est: Any, Xt_tr: np.ndarray, y_tr: np.ndarray, task: str, k: int, seed: int) -> dict[str, Any]:
    """Fit on an inner train split, tracking loss on it and on an inner validation split every round."""
    inner, val, _, _ = split_indices(np.arange(len(y_tr)), y_tr, 0.2, seed, stratify=task == CLASSIFICATION)
    metrics = ["mae", "rmse"] if task == REGRESSION else (["error", "logloss"] if k == 2 else ["merror", "mlogloss"])
    est.set_params(eval_metric=metrics)
    est.fit(Xt_tr[inner], y_tr[inner], eval_set=[(Xt_tr[inner], y_tr[inner]), (Xt_tr[val], y_tr[val])], verbose=False)
    ev = est.evals_result()
    rounds = len(ev["validation_0"][metrics[-1]])
    idx = select_indices(rounds, 200)
    curves = {m: {"train": [round(float(ev["validation_0"][m][i]), 6) for i in idx],
                  "val": [round(float(ev["validation_1"][m][i]), 6) for i in idx]} for m in metrics}
    best = getattr(est, "best_iteration", None)
    val_loss = np.asarray(ev["validation_1"][metrics[-1]])
    return {"rounds": [i + 1 for i in idx], "metrics": metrics, "curves": curves, "n_rounds": rounds,
            "best_iteration": int(best) + 1 if best is not None else int(np.argmin(val_loss)) + 1,
            "early_stopped": best is not None and rounds < est.get_params()["n_estimators"],
            "inner_rows": int(len(inner)), "val_rows": int(len(val)), "loss_units": "scaled target" if task == REGRESSION else None}


def _xgb_importance(est: Any, names: list[str], owner: list[str]) -> dict[str, Any]:
    booster = est.get_booster()
    out: dict[str, Any] = {}
    for kind, label in (("total_gain", "gain"), ("weight", "splits"), ("total_cover", "cover")):
        raw = booster.get_score(importance_type=kind)
        vals = np.zeros(len(names))
        for key, v in raw.items():
            j = int(key[1:]) if key.startswith("f") and key[1:].isdigit() else -1
            if 0 <= j < len(names):
                vals[j] = v
        agg = _aggregate(vals, owner)
        total = sum(agg.values()) or 1.0
        out[label] = sorted(({"feature": f, "value": round(v / total, 6)} for f, v in agg.items()), key=lambda r: -r["value"])
    return out


def _forest_growth(est: Any, Xt_tr: np.ndarray, y_tr: np.ndarray, Xt_te: np.ndarray, y_te: np.ndarray, task: str,
                   bootstrap: bool) -> list[dict[str, Any]]:
    """Grow the forest in steps with warm_start, recording out-of-bag and test score (the last step is the model)."""
    total = est.get_params()["n_estimators"]
    steps = sorted({max(1, int(round(total * f))) for f in (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)})
    est.set_params(warm_start=True, oob_score=bootstrap)
    out = []
    te = slice(0, min(len(y_te), 3000))
    for n in steps:
        est.set_params(n_estimators=n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(Xt_tr, y_tr)
        pred = est.predict(Xt_te[te])
        test = accuracy_score(y_te[te], pred) if task == CLASSIFICATION else r2_score(y_te[te], pred)
        oob = getattr(est, "oob_score_", None) if bootstrap else None
        out.append({"trees": n, "test": round(float(test), 5), "oob": None if oob is None or not np.isfinite(oob) else round(float(oob), 5)})
    est.set_params(warm_start=False)
    return out


def _forest_agreement(est: Any, Xt_te: np.ndarray, y_te: np.ndarray, task: str, data: LabData) -> dict[str, Any]:
    n = min(len(Xt_te), 1000)
    xs = Xt_te[:n]
    per_tree = np.array([t.predict(xs) for t in est.estimators_])
    forest = est.predict(xs)
    if task == CLASSIFICATION:
        agree = (per_tree == forest[None, :]).mean(axis=0)
        correct = forest == y_te[:n]
        buckets = []
        for lo_, hi_ in ((0.0, 0.6), (0.6, 0.8), (0.8, 0.95), (0.95, 1.01)):
            mask = (agree >= lo_) & (agree < hi_)
            if mask.any():
                buckets.append({"bucket": f"{int(lo_ * 100)}-{min(100, int(hi_ * 100))}%", "rows": int(mask.sum()),
                                "accuracy": round(float(correct[mask].mean()), 4)})
        return {"kind": "agreement", "hist": _hist(agree, 10, (0.0, 1.0)), "mean": round(float(agree.mean()), 4), "buckets": buckets}
    spread = per_tree.std(axis=0) * data.y_sd
    err = np.abs(forest - y_te[:n]) * data.y_sd
    q = np.quantile(spread, [0, 0.25, 0.5, 0.75, 1.0])
    buckets = []
    for b in range(4):
        mask = (spread >= q[b]) & (spread <= q[b + 1]) if b == 3 else (spread >= q[b]) & (spread < q[b + 1])
        if mask.any():
            buckets.append({"bucket": f"Q{b + 1}", "rows": int(mask.sum()), "mae": round(float(err[mask].mean()), 5)})
    return {"kind": "spread", "hist": _hist(spread, 20), "mean": round(float(spread.mean()), 5), "buckets": buckets}


def _permutation(pipeline: Pipeline, x: pd.DataFrame, y: np.ndarray, scoring: str, seed: int) -> dict[str, dict[str, float]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = permutation_importance(pipeline, x, y, scoring=scoring, n_repeats=3, random_state=seed, n_jobs=1)
    return {c: {"mean": float(m), "std": float(s)} for c, m, s in zip(x.columns, res.importances_mean, res.importances_std)}


def _pruning_path(est: Any, model: LabModel, task: str, params: dict[str, Any], Xt_tr: np.ndarray, y_tr: np.ndarray,
                  Xt_te: np.ndarray, y_te: np.ndarray, data: LabData, settings: Settings) -> dict[str, Any]:
    base = build_estimator(model, task, {**params, "ccp_alpha": 0.0}, settings.random_seed, settings.n_jobs)
    path = base.cost_complexity_pruning_path(Xt_tr, y_tr)
    alphas = np.unique(path.ccp_alphas)
    alphas = alphas[alphas >= 0]
    if len(alphas) > 14:
        alphas = np.unique(np.quantile(alphas, np.linspace(0, 1, 14)))
    k = len(data.class_labels)
    points = []
    for a in alphas:
        m = build_estimator(model, task, {**params, "ccp_alpha": float(a)}, settings.random_seed, settings.n_jobs)
        m.fit(Xt_tr, y_tr)
        points.append({"alpha": float(a), "leaves": int(m.get_n_leaves()),
                       "train": round(_primary(task, k, y_tr, m.predict(Xt_tr)), 5),
                       "test": round(_primary(task, k, y_te, m.predict(Xt_te)), 5)})
    return {"points": points, "current": params["ccp_alpha"]}


def _svm_charts(est: Any, Xt_tr: np.ndarray, Xt_te: np.ndarray, y_te: np.ndarray, task: str, data: LabData,
                params: dict[str, Any]) -> dict[str, Any]:
    n_sv = int(len(est.support_))
    out: dict[str, Any] = {"n_support": n_sv, "support_fraction": round(n_sv / max(1, len(Xt_tr)), 4)}
    if task == CLASSIFICATION:
        out["per_class"] = {data.class_labels[int(c)]: int(n) for c, n in zip(est.classes_, est.n_support_)}
        d = est.decision_function(Xt_te)
        if d.ndim == 1:
            lim = float(np.quantile(np.abs(d), 0.99)) or 1.0
            rng = (-lim, lim)
            out["margins"] = {"kind": "binary", "edges": _hist(d, 30, rng)["edges"],
                              "by_class": {data.class_labels[c]: _hist(d[y_te == c], 30, rng)["counts"] for c in (0, 1)},
                              "inside_margin": round(float((np.abs(d) < 1).mean()), 4)}
        else:
            s = np.sort(d, axis=1)
            gap = s[:, -1] - s[:, -2]
            out["margins"] = {"kind": "multiclass", **_hist(gap, 30), "inside_margin": None}
    else:
        resid = est.predict(Xt_te) - (y_te if y_te is not None else 0)
        eps = params["epsilon"]
        out["margins"] = {"kind": "regression", **_hist(resid, 30), "epsilon": eps,
                          "inside_tube": round(float((np.abs(resid) <= eps).mean()), 4)}
    return out


# --- clustering ------------------------------------------------------------------------------------------------
def _kmeans_payload(model: LabModel, params: dict[str, Any], data: LabData, scale: bool, settings: Settings) -> dict[str, Any]:
    seed = settings.random_seed
    started = time.perf_counter()
    prep = lab_preprocessor(data.X, scale, settings)
    Xt = prep.fit_transform(data.X)
    names, owner = _names(prep)
    km = build_estimator(model, CLUSTERING, params, seed, settings.n_jobs)
    km.fit(Xt)
    fit_seconds = time.perf_counter() - started
    lab = km.labels_
    k = params["n_clusters"]
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(Xt), min(len(Xt), 3000), replace=False)
    sil = float(silhouette_score(Xt[sample], lab[sample])) if len(np.unique(lab[sample])) > 1 else None
    metrics = {
        "silhouette": None if sil is None else round(sil, 6), "inertia": round(float(km.inertia_), 4),
        "davies_bouldin": round(float(davies_bouldin_score(Xt, lab)), 6),
        "calinski_harabasz": round(float(calinski_harabasz_score(Xt, lab)), 4), "n_iter": int(km.n_iter_), "k": k,
    }
    sizes = np.bincount(lab, minlength=k)
    pca = PCA(2, random_state=seed).fit(Xt) if Xt.shape[1] >= 2 else None
    show = rng.choice(len(Xt), min(len(Xt), 800), replace=False)
    if pca is not None:
        z = pca.transform(Xt[show])
        cz = pca.transform(km.cluster_centers_)
    else:
        z = np.column_stack([Xt[show, 0], np.zeros(len(show))])
        cz = np.column_stack([km.cluster_centers_[:, 0], np.zeros(k)])
    target_name = None
    points = []
    for i, j in enumerate(show):
        pt = {"x": round(float(z[i, 0]), 4), "y": round(float(z[i, 1]), 4), "c": int(lab[j])}
        if data.y is not None:
            pt["t"] = data.class_labels[int(data.y[j])] if data.class_labels else round(float(data.y[j]), 4)
        points.append(pt)
    # silhouette plot
    s_pick = rng.choice(len(Xt), min(len(Xt), 2000), replace=False)
    s_vals = silhouette_samples(Xt[s_pick], lab[s_pick]) if len(np.unique(lab[s_pick])) > 1 else np.zeros(len(s_pick))
    s_lab = lab[s_pick]
    silhouette = []
    for c in range(k):
        v = np.sort(s_vals[s_lab == c])[::-1]
        silhouette.append({"cluster": c, "mean": round(float(v.mean()), 4) if len(v) else None,
                           "values": _thin(v, 60) if len(v) else [], "negative": round(float((v < 0).mean()), 4) if len(v) else None})
    # profiles: centroid per transformed feature (z-score for scaled numerics, share for one-hot levels)
    centers = km.cluster_centers_
    spread = centers.std(axis=0)
    order = np.argsort(-spread)[:20]
    units = []
    scales = _scale_info(prep)
    for j in order:
        units.append("share" if "=" in names[j] and scales[j] is None else ("z" if scales[j] else "raw"))
    numeric = [c for c in data.X.columns if ptypes.is_numeric_dtype(data.X[c])][:12]
    raw_means = data.X[numeric].groupby(lab).mean() if numeric else pd.DataFrame()
    profiles = {
        "features": [names[j] for j in order], "units": units,
        "centers": [[round(float(centers[c, j]), 4) for j in order] for c in range(k)],
        "raw_columns": numeric,
        "raw_means": [[round(float(raw_means.loc[c, col]), 4) if c in raw_means.index else None for col in numeric] for c in range(k)],
    }
    target = None
    if data.y is not None:
        target_name = "target"
        if data.class_labels:
            ct = pd.crosstab(lab, data.y).reindex(index=range(k), columns=range(len(data.class_labels)), fill_value=0)
            purity = float(ct.max(axis=1).sum() / max(1, ct.values.sum()))
            target = {"kind": "classes", "labels": data.class_labels, "counts": ct.values.tolist(), "purity": round(purity, 4)}
        else:
            g = pd.Series(data.y).groupby(lab)
            target = {"kind": "numeric", "mean": [round(float(g.mean().get(c, np.nan)), 4) for c in range(k)],
                      "std": [round(float(g.std().get(c, np.nan)), 4) for c in range(k)]}
    notes = []
    if sil is not None:
        verdict = ("strong, well-separated structure" if sil > 0.5 else "reasonable structure" if sil > 0.25
                   else "weak structure: clusters overlap a lot")
        notes.append(f"Silhouette {sil:.2f}: {verdict}. Range -1 to 1; higher means rows sit closer to their own cluster than to the next one.")
    small = int(sizes.min())
    if small < max(5, 0.02 * len(Xt)):
        notes.append(f"The smallest cluster has only {small} rows; k may be too high, or it is catching outliers.")
    if not scale:
        notes.append("Scaling is off, so the column with the largest numbers dominates the distances. Turn scaling on.")
    neg = [s["cluster"] for s in silhouette if s["negative"] and s["negative"] > 0.1]
    if neg:
        notes.append(f"Cluster(s) {', '.join(map(str, neg))} have over 10% of rows with negative silhouette (probably assigned to the wrong cluster).")
    if target and target["kind"] == "classes":
        notes.append(f"Cluster purity against the target is {target['purity']:.0%}: the share of rows that match their cluster's most common class.")
    return {
        "model": model.key, "abbr": model.abbr, "task": CLUSTERING, "params": params, "metrics": metrics,
        "sizes": sizes.tolist(), "fit_seconds": round(fit_seconds, 3),
        "data": _data_info(data, scale),
        "charts": {"scatter": {"points": points, "centroids": [{"x": round(float(a), 4), "y": round(float(b), 4), "c": c}
                                                               for c, (a, b) in enumerate(cz)],
                               "explained": [round(float(v), 4) for v in pca.explained_variance_ratio_] if pca is not None else None,
                               "target_name": target_name},
                   "silhouette": silhouette, "profiles": profiles, "target": target},
        "notes": notes,
    }


def _data_info(data: LabData, scale: bool) -> dict[str, Any]:
    return {"rows_total": data.rows_total, "rows_used": int(len(data.X)), "capped": data.capped,
            "train_rows": int(len(data.train)), "test_rows": int(len(data.test)), "features_available": data.available,
            "features_used": data.used, "scaled": scale, "exclusions": data.exclusions,
            "class_labels": data.class_labels, "y_mean": data.y_mu if data.task == REGRESSION else None,
            "y_std": data.y_sd if data.task == REGRESSION else None}


# --- notes -----------------------------------------------------------------------------------------------------
_OVERFIT_HINTS = {
    "decision_tree": "lower max depth, raise min samples per leaf, or prune with ccp_alpha",
    "random_forest": "limit max depth or raise min samples per leaf",
    "xgboost": "lower max depth or learning rate, raise lambda / min child weight, subsample rows and columns, or turn on early stopping",
    "knn": "raise k",
    "svm": "lower C (and gamma for rbf)",
    "logistic_regression": "lower C (stronger penalty)",
    "linear_regression": "add a ridge / lasso penalty or drop features",
}


def _notes(model: LabModel, task: str, params: dict[str, Any], train: dict[str, Any], test: dict[str, Any],
           cv: dict[str, Any] | None, scale: bool, data: LabData, extra: dict[str, Any], converged: bool) -> list[str]:
    notes: list[str] = []
    key = "r2" if task == REGRESSION else "f1"
    label = "R2" if task == REGRESSION else "F1"
    tr, te = train.get(key), test.get(key)
    if tr is not None and te is not None:
        gap = tr - te
        if gap > 0.1:
            notes.append(f"Overfitting: train {label} {tr:.3f} vs test {te:.3f} (gap {gap:.2f}). To fix it: {_OVERFIT_HINTS.get(model.key, 'simplify the model')}.")
        elif task == CLASSIFICATION and test.get("balanced_accuracy") is not None and train["balanced_accuracy"] < 1 / max(2, len(data.class_labels)) + 0.05:
            notes.append("Underfitting: the model barely beats guessing even on its training rows. Make it more flexible.")
        elif task == REGRESSION and tr < 0.1:
            notes.append(f"Underfitting: train R2 is only {tr:.2f}; the model explains little of the target even on rows it saw.")
        else:
            notes.append(f"Train {label} {tr:.3f} vs test {te:.3f}: the gap of {gap:.2f} is small, so the model generalises well at these settings.")
    if cv and te is not None:
        if abs(cv["mean"] - te) > 2 * max(cv["std"], 0.01):
            notes.append(f"Cross-validated {cv['metric']} {cv['mean']:.3f} differs from the single test split ({te:.3f}); trust the CV number more.")
        if cv["std"] > 0.05:
            notes.append(f"Scores swing a lot between folds (std {cv['std']:.3f}): the model is sensitive to which rows it trains on.")
    if model.scaled and not scale:
        notes.append(f"Scaling is off. {model.name} is sensitive to feature scale, so large-valued columns dominate. Turn scaling on.")
    if not converged:
        notes.append("The solver hit its iteration limit before converging. Raise max iterations or use a stronger penalty.")
    if data.capped:
        notes.append(f"The lab uses a sample of {len(data.X):,} of {data.rows_total:,} rows to stay interactive.")
    if model.key == "linear_regression":
        if extra["coefficients"] and all(c["zero"] for c in extra["coefficients"]):
            notes.append("The penalty is so strong that every coefficient is 0, so the model just predicts the mean. Lower alpha.")
        elif params["regularization"] in ("lasso", "elasticnet"):
            zeros = sum(c["zero"] for c in extra["coefficients"])
            if zeros:
                notes.append(f"The penalty dropped {zeros} feature(s) to exactly 0 (built-in feature selection).")
        n_sig = extra.get("n_significant")
        if n_sig is not None:
            notes.append(f"{n_sig} of {extra['n_coefficients']} coefficients are significant at p < 0.05.")
        high_vif = [c["feature"] for c in extra["coefficients"] if c.get("vif") and c["vif"] > 10]
        if high_vif:
            notes.append(f"High multicollinearity (VIF > 10) on {', '.join(high_vif[:4])}: their coefficients are unstable and hard to interpret.")
    if model.key == "logistic_regression" and extra.get("n_zero"):
        notes.append(f"The penalty set {extra['n_zero']} of {extra['n_coefficients']} coefficients to exactly 0 (dropped features).")
    if model.key == "knn" and params["n_neighbors"] == 1:
        notes.append("k = 1 memorises the training rows: train score is perfect by construction.")
    if model.key == "xgboost":
        b = extra["loss"]
        if b["early_stopped"]:
            notes.append(f"Early stopping kept {b['best_iteration']} of {params['n_estimators']} rounds.")
        elif b["best_iteration"] < 0.6 * b["n_rounds"]:
            notes.append(f"Validation loss bottomed out at round {b['best_iteration']} of {b['n_rounds']}; the later trees mostly add overfitting. Try early stopping or fewer trees.")
    if model.key == "decision_tree":
        notes.append(f"The tree has {extra['tree']['leaves']} leaves and depth {extra['tree']['depth']}.")
    if model.key == "random_forest" and params["bootstrap"]:
        g = extra["growth"][-1]
        if g["oob"] is not None:
            notes.append(f"Out-of-bag {'accuracy' if task == CLASSIFICATION else 'R2'} {g['oob']:.3f} vs test {g['test']:.3f}: OOB is a free validation score from rows each tree never saw.")
    if model.key == "svm":
        notes.append(f"{extra['n_support']:,} support vectors ({extra['support_fraction']:.0%} of training rows) define the boundary. A high share means a soft, uncertain boundary.")
    return notes


# --- the fit itself (shared by the diagnostics and the export) ------------------------------------------------
@dataclass
class LabFit:
    """One fitted lab model and the arrays the diagnostics are drawn from."""

    model: LabModel
    task: str
    params: dict[str, Any]
    data: LabData
    pipeline: Pipeline
    x_tr: pd.DataFrame
    x_te: pd.DataFrame
    y_tr: np.ndarray
    y_te: np.ndarray
    Xt_tr: np.ndarray
    Xt_te: np.ndarray
    extra: dict[str, Any]
    fit_seconds: float
    converged: bool


def _fit(model: LabModel, task: str, params: dict[str, Any], data: LabData, scale: bool, settings: Settings) -> LabFit:
    """Fit ``model`` on the training rows exactly as the lab shows it (XGBoost with its inner validation split,
    random forests grown with warm start)."""
    seed = settings.random_seed
    k = len(data.class_labels)
    x_tr, x_te = data.X.iloc[data.train], data.X.iloc[data.test]
    y_tr, y_te = data.yfit[data.train], data.yfit[data.test]
    if model.key == "svm" and len(y_tr) > settings.lab_svm_max_rows:
        keep, _, _ = cap_rows(np.arange(len(y_tr)), y_tr, settings.lab_svm_max_rows, seed, task == CLASSIFICATION)
        x_tr, y_tr = x_tr.iloc[keep], y_tr[keep]
    prep = lab_preprocessor(x_tr, scale, settings, model.key in LINEAR)
    Xt_tr = np.asarray(prep.fit_transform(x_tr), dtype=float)
    Xt_te = np.asarray(prep.transform(x_te), dtype=float)
    est = build_estimator(model, task, params, seed, settings.n_jobs, fit_time=True)
    extra: dict[str, Any] = {}
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if model.key == "xgboost":
            extra["loss"] = _xgb_fit(est, Xt_tr, y_tr, task, k, seed)
        elif model.key == "random_forest":
            extra["growth"] = _forest_growth(est, Xt_tr, y_tr, Xt_te, y_te, task, params["bootstrap"])
        else:
            est.fit(Xt_tr, y_tr)
    fit_seconds = time.perf_counter() - started
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return LabFit(model, task, params, data, Pipeline([("prep", prep), ("clf", est)]), x_tr, x_te, y_tr, y_te,
                  Xt_tr, Xt_te, extra, fit_seconds, converged)


def _scores(f: LabFit, settings: Settings) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Train (sampled) and test metrics, plus the test-set performance charts."""
    seed = settings.random_seed
    data, task, k = f.data, f.task, len(f.data.class_labels)
    est = f.pipeline.named_steps["clf"]
    Xt_tr, Xt_te, y_tr, y_te = f.Xt_tr, f.Xt_te, f.y_tr, f.y_te
    tr_pick = np.arange(len(y_tr)) if len(y_tr) <= 5000 else np.sort(np.random.default_rng(seed).choice(len(y_tr), 5000, replace=False))
    if task == CLASSIFICATION:
        pred_te = np.asarray(est.predict(Xt_te)).astype(int)
        pred_tr = np.asarray(est.predict(Xt_tr[tr_pick])).astype(int)
        got_te, got_tr = score_matrix(est, Xt_te), score_matrix(est, Xt_tr[tr_pick])
        s_te, kind = got_te if got_te else (None, None)
        s_tr = got_tr[0] if got_tr else None
        test_m = _clf_metrics(y_te, pred_te, s_te, kind, k)
        train_m = _clf_metrics(y_tr[tr_pick], pred_tr, s_tr, kind, k)
        return train_m, test_m, _clf_performance(y_te, pred_te, s_te, kind, data.class_labels, settings)
    pred_te = np.asarray(est.predict(Xt_te), dtype=float) * data.y_sd + data.y_mu
    pred_tr = np.asarray(est.predict(Xt_tr[tr_pick]), dtype=float) * data.y_sd + data.y_mu
    actual_te, actual_tr = data.y[data.test], y_tr[tr_pick] * data.y_sd + data.y_mu
    test_m = _reg_metrics(actual_te, pred_te, Xt_te.shape[1])
    train_m = _reg_metrics(actual_tr, pred_tr, Xt_te.shape[1])
    return train_m, test_m, _reg_performance(actual_te, pred_te, settings)


# --- public entry points ---------------------------------------------------------------------------------------
def fit_payload(session: Session, settings: Settings, key: str, req: Any) -> dict[str, Any]:
    """Fit one lab model with the requested settings and return its metrics and diagnostics."""
    model = get_lab_model(key)
    task = lab_task(model, session.task, session.target is not None)
    params = model.resolve(req.params, task)
    data = load_data(session, settings, task, req.features, req.test_size)
    if task == CLUSTERING:
        return _kmeans_payload(model, params, data, req.scale, settings)
    seed = settings.random_seed
    k = len(data.class_labels)
    f = _fit(model, task, params, data, req.scale, settings)
    pipeline, est, extra = f.pipeline, f.pipeline.named_steps["clf"], f.extra
    x_te, y_tr, y_te, Xt_tr, Xt_te = f.x_te, f.y_tr, f.y_te, f.Xt_tr, f.Xt_te
    prep = pipeline.named_steps["prep"]
    names, owner = _names(prep)
    scales = _scale_info(prep)
    train_m, test_m, perf = _scores(f, settings)
    cv = _cv(model, task, params, data, req.scale, settings) if req.cv else None

    charts: dict[str, Any] = {}
    scoring, score_label = primary_scoring(task, k)
    if model.key == "linear_regression":
        charts["linear"] = extra_lin = _linear_charts(est, params, Xt_tr, y_tr, Xt_te, y_te, names, data, settings)
        extra.update(extra_lin)
    elif model.key == "logistic_regression":
        charts["logistic"] = extra_log = _logistic_charts(est, model, params, Xt_tr, y_tr, Xt_te, y_te, names, data, settings)
        extra.update(extra_log)
        charts["boundary"] = _boundary(model, task, params, Xt_tr, y_tr, data, settings)
    elif model.key == "knn":
        charts["knn"] = _knn_charts(est, params, Xt_tr, y_tr, Xt_te, y_te, data, task, req.row, settings)
        charts["boundary"] = _boundary(model, task, params, Xt_tr, y_tr, data, settings)
    elif model.key == "xgboost":
        charts["loss"] = extra["loss"]
        charts["importance"] = _xgb_importance(est, names, owner)
        charts["tree"] = _xgb_tree(est, req.tree_index, names, scales, req.tree_depth)
    elif model.key == "decision_tree":
        weighted = params.get("class_weight") == "balanced"
        charts["tree"] = extra["tree"] = _sk_tree(est, names, scales, data, task, req.tree_depth, Xt_tr, y_tr, weighted)
        charts["rules"] = _rules(est, names, scales, data, task, Xt=Xt_tr, y=y_tr)
        charts["pruning"] = _pruning_path(est, model, task, params, Xt_tr, y_tr, Xt_te, y_te, data, settings)
        mdi = _aggregate(est.feature_importances_, owner)
        charts["importance"] = sorted(({"feature": f, "value": round(v, 6)} for f, v in mdi.items()), key=lambda r: -r["value"])
    elif model.key == "random_forest":
        charts["growth"] = extra["growth"]
        mdi = _aggregate(est.feature_importances_, owner)
        pick = slice(0, min(len(y_te), settings.lab_perm_rows))
        perm = _permutation(pipeline, x_te.iloc[pick], y_te[pick], scoring, seed)
        charts["importance"] = sorted(
            ({"feature": f, "mdi": round(mdi.get(f, 0.0), 6), "perm": round(perm[f]["mean"], 6), "perm_std": round(perm[f]["std"], 6)}
             for f in data.used), key=lambda r: -r["perm"])
        charts["agreement"] = _forest_agreement(est, Xt_te, y_te, task, data)
        n_trees = len(est.estimators_)
        idx = max(0, min(req.tree_index, n_trees - 1))
        charts["tree"] = {**_sk_tree(est.estimators_[idx], names, scales, data, task, req.tree_depth, Xt_tr, y_tr,
                                     params.get("class_weight") == "balanced"), "index": idx, "n_trees": n_trees}
    elif model.key == "svm":
        charts["svm"] = extra_svm = _svm_charts(est, Xt_tr, Xt_te, y_te, task, data, params)
        extra.update(extra_svm)
        charts["boundary"] = _boundary(model, task, params, Xt_tr, y_tr, data, settings, support=True)

    return {
        "model": model.key, "abbr": model.abbr, "task": task, "params": params,
        "metric": score_label, "train": train_m, "test": test_m, "cv": cv, "fit_seconds": round(f.fit_seconds, 3),
        "data": {**_data_info(data, req.scale), "train_rows": int(len(y_tr)), "n_transformed": len(names)},
        "performance": perf, "charts": charts,
        "notes": _notes(model, task, params, train_m, test_m, cv, req.scale, data, extra, f.converged),
    }


def _enable(model: LabModel, task: str, params: dict[str, Any], name: str, changed: dict[str, Any]) -> None:
    """Make ``name`` take effect: switch the controls it depends on (e.g. gamma needs gamma_mode=custom)."""
    spec = {p.name: p for p in model.params_for(task)}
    param = spec.get(name)
    if param is None:
        return
    for ctrl, allowed in param.when.items():
        if params.get(ctrl) not in allowed:
            params[ctrl] = allowed[0]
            changed[ctrl] = allowed[0]
        _enable(model, task, params, ctrl, changed)


def sweep_payload(session: Session, settings: Settings, key: str, req: Any) -> dict[str, Any]:
    """Validation curve (one parameter), score grid (two) or learning curve, cross-validated on training rows."""
    model = get_lab_model(key)
    task = lab_task(model, session.task, session.target is not None)
    params = model.resolve(req.params, task)
    data = load_data(session, settings, task, req.features, req.test_size)
    seed = settings.random_seed
    started = time.perf_counter()
    k = len(data.class_labels)
    scoring, label = primary_scoring(task, k)

    if req.param == "train_size":
        if task == CLUSTERING:
            raise BadRequestError("Learning curves need a target; K-means is unsupervised.", error_code="NO_LEARNING_CURVE")
        tr = data.train
        pick, _, _ = cap_rows(tr, data.yfit[tr], settings.lab_sweep_max_rows, seed, task == CLASSIFICATION)
        x, y = data.X.iloc[pick], data.yfit[pick]
        pipe = Pipeline([("prep", lab_preprocessor(x, req.scale, settings, model.key in LINEAR)),
                         ("clf", build_estimator(model, task, params, seed, settings.n_jobs))])
        cv = (StratifiedKFold if task == CLASSIFICATION else KFold)(settings.lab_sweep_folds, shuffle=True, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sizes, tr_s, va_s = learning_curve(pipe, x, y, train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0], cv=cv, scoring=scoring,
                                               error_score=np.nan, shuffle=True, random_state=seed)
        return {
            "kind": "learning_curve", "metric": label, "values": [int(s) for s in sizes],
            "train": [round(float(np.nanmean(r)), 5) for r in tr_s], "val": [round(float(np.nanmean(r)), 5) for r in va_s],
            "val_std": [round(float(np.nanstd(r)), 5) for r in va_s], "rows": int(len(y)),
            "folds": settings.lab_sweep_folds, "seconds": round(time.perf_counter() - started, 2), "truncated": False,
        }

    sweepable = model.sweepable(task)
    names = [n for n in (req.param, req.param2) if n]
    for n in names:
        if n not in sweepable:
            raise BadRequestError(f"'{n}' cannot be swept for {model.name}.", error_code="BAD_SWEEP_PARAM",
                                  details={"sweepable": list(sweepable)})
    p1: Param = sweepable[req.param]
    v1 = [p1.coerce(v) for v in (req.values or p1.sweep)]
    p2: Param | None = sweepable[req.param2] if req.param2 else None
    v2 = [p2.coerce(v) for v in (req.values2 or p2.sweep)] if p2 else [None]
    changed: dict[str, Any] = {}
    for n in names:
        _enable(model, task, params, n, changed)

    if task == CLUSTERING:
        if p2:
            raise BadRequestError("K-means sweeps one parameter at a time.", error_code="BAD_SWEEP_PARAM")
        pick = np.random.default_rng(seed).choice(len(data.X), min(len(data.X), settings.lab_sweep_max_rows), replace=False)
        Xt = lab_preprocessor(data.X, req.scale, settings).fit_transform(data.X.iloc[pick])
        inertia, sil, done = [], [], []
        for v in v1:
            if done and time.perf_counter() - started > settings.lab_sweep_budget_sec:
                break
            km = build_estimator(model, CLUSTERING, {**params, p1.name: v}, seed, settings.n_jobs).fit(Xt)
            inertia.append(round(float(km.inertia_), 4))
            sil.append(round(float(silhouette_score(Xt, km.labels_)), 5) if len(np.unique(km.labels_)) > 1 else None)
            done.append(v)
        best = int(np.nanargmax([s if s is not None else np.nan for s in sil])) if any(s is not None for s in sil) else None
        return {"kind": "cluster_sweep", "param": p1.name, "values": done, "inertia": inertia, "silhouette": sil,
                "current": params[p1.name], "best_value": done[best] if best is not None else None,
                "rows": int(len(pick)), "seconds": round(time.perf_counter() - started, 2),
                "truncated": len(done) < len(v1), "changed": changed}

    tr = data.train
    cap = settings.lab_grid_max_rows if p2 else settings.lab_sweep_max_rows
    if model.key == "svm":
        cap = min(cap, 1500)
    pick, _, _ = cap_rows(tr, data.yfit[tr], cap, seed, task == CLASSIFICATION)
    x, y = data.X.iloc[pick], data.yfit[pick]
    folds = settings.lab_sweep_folds
    if task == CLASSIFICATION:
        folds = max(2, min(folds, int(np.bincount(y).min())))
    cv = (StratifiedKFold if task == CLASSIFICATION else KFold)(folds, shuffle=True, random_state=seed)
    prep = lab_preprocessor(x, req.scale, settings, model.key in LINEAR)
    train_grid: list[list[float | None]] = []
    val_grid: list[list[float | None]] = []
    std_grid: list[list[float | None]] = []
    truncated = False
    for a in v1:
        tr_row, va_row, sd_row = [], [], []
        for b in v2:
            if (train_grid or tr_row) and time.perf_counter() - started > settings.lab_sweep_budget_sec:
                truncated = True
                break
            p = {**params, p1.name: a}
            if p2:
                p[p2.name] = b
            pipe = Pipeline([("prep", clone(prep)), ("clf", build_estimator(model, task, p, seed, settings.n_jobs))])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = cross_validate(pipe, x, y, cv=cv, scoring=scoring, return_train_score=True, error_score=np.nan)
            ts, vs = np.asarray(res["train_score"], float), np.asarray(res["test_score"], float)
            tr_row.append(None if np.all(np.isnan(ts)) else round(float(np.nanmean(ts)), 5))
            va_row.append(None if np.all(np.isnan(vs)) else round(float(np.nanmean(vs)), 5))
            sd_row.append(None if np.all(np.isnan(vs)) else round(float(np.nanstd(vs)), 5))
        if tr_row:
            train_grid.append(tr_row)
            val_grid.append(va_row)
            std_grid.append(sd_row)
        if truncated:
            break
    best = None
    for i, row in enumerate(val_grid):
        for j, v in enumerate(row):
            if v is not None and (best is None or v > best[2]):
                best = (i, j, v)
    out: dict[str, Any] = {
        "metric": label, "param": p1.name, "values": v1[: len(val_grid)], "current": params[p1.name],
        "rows": int(len(y)), "folds": folds, "seconds": round(time.perf_counter() - started, 2), "truncated": truncated,
        "changed": changed,
    }
    if p2:
        out.update({"kind": "grid", "param2": p2.name, "values2": v2, "current2": params[p2.name],
                    "val": val_grid, "train": train_grid,
                    "best": None if best is None else {"value": v1[best[0]], "value2": v2[best[1]], "score": best[2]}})
    else:
        out.update({"kind": "curve", "train": [r[0] for r in train_grid], "val": [r[0] for r in val_grid],
                    "val_std": [r[0] for r in std_grid],
                    "best": None if best is None else {"value": v1[best[0]], "score": best[2]}})
    return out


# --- export ----------------------------------------------------------------------------------------------------
def _fit_flags(task: str, train: dict[str, Any], test: dict[str, Any], k: int) -> tuple[float | None, bool, bool]:
    """Train-test gap of the headline score and the over / underfit flags, by the same rules as the lab's notes."""
    key = "r2" if task == REGRESSION else "f1"
    tr, te = train.get(key), test.get(key)
    if tr is None or te is None:
        return None, False, False
    gap = round(tr - te, 6)
    if gap > 0.1:
        return gap, True, False
    if task == CLASSIFICATION:
        under = train.get("balanced_accuracy") is not None and train["balanced_accuracy"] < 1 / max(2, k) + 0.05
    else:
        under = tr < 0.1
    return gap, False, under


def export_model(session: Session, settings: Settings, key: str, req: Any) -> tuple[export.ExportSource, FittedModel]:
    """Fit ``key`` with the lab's settings (the same fit the lab shows) and describe it for the export zip."""
    model = get_lab_model(key)
    task = lab_task(model, session.task, session.target is not None)
    if task == CLUSTERING:
        raise BadRequestError(f"{model.name} groups rows without a target, so there is no prediction model to export.",
                              error_code="NOT_EXPORTABLE")
    params = model.resolve(req.params, task)
    data = load_data(session, settings, task, req.features, req.test_size)
    f = _fit(model, task, params, data, req.scale, settings)
    train_m, test_m, _ = _scores(f, settings)
    cv = _cv(model, task, params, data, req.scale, settings) if req.cv else None
    k = len(data.class_labels)
    gap, over, under = _fit_flags(task, train_m, test_m, k)
    spec = export.build_inference_spec(session.raw_df, session.working_df, list(data.used), session.recipe, session.target)
    row = {
        "name": model.name, "family": "model lab", **test_m, "train_test_gap": gap,
        "overfit_warning": over, "underfit_warning": under,
        **({"cv_score": cv["mean"], "cv_std": cv["std"], "cv": {"metric": cv["metric"]}} if cv else {}),
    }
    limits = []
    if data.capped:
        limits.append(f"The model lab trains on a sample of {len(data.X):,} of {data.rows_total:,} rows.")
    if len(f.y_tr) < len(data.train):
        limits.append(f"SVM trained on {len(f.y_tr):,} of {len(data.train):,} training rows to keep the fit fast.")
    src = export.ExportSource(
        filename=session.filename, spec=spec, row=row, origin="lab", hyperparams=params,
        settings={"test_size": req.test_size, "scale": req.scale, "cv": bool(req.cv), "features": list(data.used),
                  "n_features": len(data.used), "n_features_available": len(data.available)},
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        split={"train_rows": int(len(f.y_tr)), "test_rows": int(len(f.y_te))},
        cleaning=list(session.cleaning), exclusions=data.exclusions, limitations=limits,
    )
    fitted = FittedModel(key=model.key, abbr=model.abbr, task=task, pipeline=f.pipeline, x_train=f.x_tr.head(0),
                         x_test=f.x_te.head(0), y_test=f.y_te[:0], class_labels=list(data.class_labels),
                         y_mu=data.y_mu, y_sd=data.y_sd)
    return src, fitted


def export_zip(session: Session, settings: Settings, key: str, req: Any) -> tuple[bytes, str]:
    """The model zip (as on the dashboard) for the lab's fit of ``key``, and its file name."""
    src, fitted = export_model(session, settings, key, req)
    return export.export_zip(src, fitted), export.export_name(src, fitted)
