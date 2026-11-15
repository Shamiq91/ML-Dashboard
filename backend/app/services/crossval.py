"""k-fold cross-validation of a trained model's configuration, on the training rows only (the test rows are never touched).

The hold-out score from one split can be luck. Cross-validation refits the same pipeline (same hyper-parameters, tuned ones
if tuning was applied) on k different train/validation splits and reports the mean and spread of each metric, so the
dashboard can tell a real difference between two models from noise.
"""
from __future__ import annotations

import logging
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from app.config import Settings
from app.services.model_registry import ModelSpec
from app.services.tasks import is_regression
from app.services.tuning import coerce_params, make_cv

logger = logging.getLogger("mldash.crossval")

_HEAVY = ("tabnet", "torch_mlp")


def _scorers(task: str, average: str | None, n_classes: int) -> dict[str, str]:
    if is_regression(task):
        return {"r2": "r2", "rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"}
    suffix = {"macro": "_macro", "weighted": "_weighted"}.get(average or "weighted", "")
    return {
        "f1": f"f1{suffix}", "accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy",
        "precision": f"precision{suffix}", "recall": f"recall{suffix}",
        "roc_auc": "roc_auc" if n_classes == 2 else "roc_auc_ovr",
    }


def cv_pipeline(
    prep: Any, est: Any, y: np.ndarray, folds: int, *, task: str, n_classes: int, handle_imbalance: str, seed: int,
    apply_imbalance: Any,
) -> tuple[Any, dict[str, Any]]:
    """A pipeline plus fit params that train like the final model does, for use inside CV/tuning folds.

    Class weighting is applied as an estimator setting or as per-row ``sample_weight`` (sliced per fold by sklearn);
    SMOTE runs inside each fold so validation rows are never synthetic.
    """
    fit_params: dict[str, Any] = {}
    extra = apply_imbalance(est, y, n_classes, handle_imbalance)
    if "sample_weight" in extra:
        fit_params["clf__sample_weight"] = np.asarray(extra["sample_weight"])
    if handle_imbalance == "smote" and not is_regression(task):
        smallest = int(np.bincount(np.asarray(y, dtype=int)).min())
        neighbours = min(5, int(smallest * (folds - 1) / folds) - 1)
        if neighbours >= 1:
            try:
                from imblearn.over_sampling import SMOTE
                from imblearn.pipeline import Pipeline as ImbPipeline
            except ImportError:
                logger.warning("SMOTE requested but imbalanced-learn is not installed")
            else:
                return ImbPipeline([("prep", prep), ("smote", SMOTE(random_state=seed, k_neighbors=neighbours)), ("clf", est)]), fit_params
    return Pipeline([("prep", prep), ("clf", est)]), fit_params


def _spread(values: np.ndarray, sign: float = 1.0, scale: float = 1.0) -> dict[str, Any] | None:
    finite = values[np.isfinite(values)] * sign * scale
    if finite.size == 0:
        return None
    return {
        "mean": round(float(finite.mean()), 6),
        "std": round(float(finite.std(ddof=1)), 6) if finite.size > 1 else 0.0,
        "min": round(float(finite.min()), 6),
        "max": round(float(finite.max()), 6),
        "folds": [round(float(v), 6) for v in finite],
    }


def cross_validate_model(
    spec: ModelSpec,
    x: pd.DataFrame,
    y: np.ndarray,
    *,
    task: str,
    n_classes: int,
    average: str | None,
    scale: bool,
    handle_imbalance: str,
    params: dict[str, Any] | None,
    y_scale: float,
    settings: Settings,
    folds: int,
    build_preprocessor: Any,
    apply_imbalance: Any,
) -> dict[str, Any] | None:
    """Cross-validate one model; returns the report stored on its leaderboard row (``None`` if it cannot be scored)."""
    started = time.perf_counter()
    heavy = spec.key in _HEAVY
    k = min(folds, 3) if heavy else folds
    seed = settings.random_seed
    regression = is_regression(task)

    prep = build_preprocessor(x, scale and spec.scaled, settings)
    est = spec.build(seed, settings.n_jobs, task)
    if params:
        est.set_params(**coerce_params(spec.key, params))
    pipeline, fit_params = cv_pipeline(prep, est, y, k, task=task, n_classes=n_classes, handle_imbalance=handle_imbalance,
                                       seed=seed, apply_imbalance=apply_imbalance)
    cv = make_cv(task, y, k, seed)
    scorers = _scorers(task, average, n_classes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = cross_validate(pipeline, x, y, cv=cv, scoring=scorers, error_score=np.nan, n_jobs=1, params=fit_params)
    metrics: dict[str, Any] = {}
    for name in scorers:
        values = np.asarray(raw[f"test_{name}"], dtype="float64")
        if name in ("rmse", "mae"):  # sklearn returns negated errors on the standardised target; undo both
            metrics[name] = _spread(values, sign=-1.0, scale=y_scale)
        else:
            metrics[name] = _spread(values)
    headline = "r2" if regression else "f1"
    if metrics.get(headline) is None:
        logger.warning("cross-validation of %s produced no finite %s scores", spec.key, headline)
        return None
    head = metrics[headline]
    n_ok = len(head["folds"])
    return {
        "folds": int(getattr(cv, "n_splits", k)),
        "folds_ok": n_ok,
        "rows_used": int(len(y)),
        "metric": headline,
        "metrics": {name: m for name, m in metrics.items() if m is not None},
        "score": head["mean"],
        "score_std": head["std"],
        # standard error of the mean: how far the average itself may be from the truth
        "score_sem": round(head["std"] / float(np.sqrt(n_ok)), 6) if n_ok > 1 else None,
        "seconds": round(time.perf_counter() - started, 2),
        "note": (
            f"{spec.abbr} is slow, so it used {k} folds instead of {folds}." if heavy and k != folds else None
        ),
    }
