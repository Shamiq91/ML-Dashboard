"""Hyper-parameter tuning with Optuna (TPE) and cross-validation on training rows only."""
from __future__ import annotations

import logging
import time
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from app.config import Settings
from app.services.model_registry import ModelSpec
from app.services.tasks import is_regression

logger = logging.getLogger("mldash.tuning")

ProgressFn = Callable[[int, int, float | None], None]


def scoring_for(task: str, average: str | None) -> str:
    """sklearn scorer matching the leaderboard's primary score (R^2, or F1 with the chosen average)."""
    if is_regression(task):
        return "r2"
    return {"binary": "f1", "macro": "f1_macro"}.get(average or "weighted", "f1_weighted")


def coerce_params(key: str, params: dict[str, Any]) -> dict[str, Any]:
    """Translate search-space values into constructor values (e.g. ``"64-32"`` -> ``(64, 32)`` for the sklearn MLP)."""
    out = dict(params)
    if key == "mlp" and isinstance(out.get("hidden_layer_sizes"), str):
        out["hidden_layer_sizes"] = tuple(int(p) for p in out["hidden_layer_sizes"].split("-"))
    return out


def _suggest(trial: Any, name: str, rule: tuple[Any, ...]) -> Any:
    kind = rule[0]
    if kind == "int":
        return trial.suggest_int(name, rule[1], rule[2], step=rule[3] if len(rule) > 3 else 1)
    if kind == "float":
        return trial.suggest_float(name, rule[1], rule[2])
    if kind == "log":
        return trial.suggest_float(name, rule[1], rule[2], log=True)
    return trial.suggest_categorical(name, rule[1])


def make_cv(task: str, y: np.ndarray, folds: int, seed: int) -> Any:
    """Stratified CV for classes, plain shuffled KFold for regression."""
    if not is_regression(task):
        smallest = int(np.bincount(np.asarray(y, dtype=int)).min()) if len(y) else 0
        if smallest >= folds:
            return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return KFold(n_splits=folds, shuffle=True, random_state=seed)


def tune_model(
    spec: ModelSpec,
    make_pipeline: Callable[[dict[str, Any]], Pipeline],
    x: pd.DataFrame,
    y: np.ndarray,
    *,
    task: str,
    average: str | None,
    n_trials: int,
    settings: Settings,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Search ``spec``'s space; returns the report (best params are applied by the caller only if ``applied``)."""
    import optuna
    from optuna.exceptions import TrialPruned

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    space = spec.search_space()
    started = time.perf_counter()
    base = {"enabled": True, "metric": scoring_for(task, average), "cv_folds": settings.tune_cv_folds,
            "rows_used": int(len(y)), "requested_trials": n_trials}
    if not space:
        return {**base, "applied": False, "n_trials": 0, "history": [], "best_params": {}, "best_score": None,
                "default_score": None, "improvement": None, "param_importance": {},
                "note": f"{spec.abbr} has no tunable hyper-parameters."}
    cv = make_cv(task, y, settings.tune_cv_folds, settings.random_seed)
    scoring = base["metric"]

    def evaluate(params: dict[str, Any]) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(make_pipeline(coerce_params(spec.key, params)), x, y, cv=cv, scoring=scoring,
                                     error_score=np.nan)
        value = float(np.nanmean(scores)) if np.isfinite(scores).any() else float("nan")
        return value

    default_score = evaluate({})
    history: list[dict[str, Any]] = []
    best_so_far = [default_score if np.isfinite(default_score) else -np.inf]

    def objective(trial: Any) -> float:
        params = {name: _suggest(trial, name, rule) for name, rule in space.items()}
        value = evaluate(params)
        if not np.isfinite(value):
            raise TrialPruned()
        history.append({"trial": len(history) + 1, "score": round(value, 6), "params": params})
        best_so_far[0] = max(best_so_far[0], value)
        if progress:
            progress(len(history), n_trials, float(best_so_far[0]) if np.isfinite(best_so_far[0]) else None)
        return value

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=settings.random_seed))
    study.optimize(objective, n_trials=n_trials, timeout=settings.tune_timeout_sec, catch=(Exception,))
    completed = [t for t in study.trials if t.value is not None]
    if not completed:
        return {**base, "applied": False, "n_trials": 0, "history": [], "best_params": {}, "best_score": None,
                "default_score": _num(default_score), "improvement": None, "param_importance": {},
                "note": "Every trial failed; kept the default hyper-parameters."}
    best = study.best_trial
    applied = bool((not np.isfinite(default_score)) or best.value > default_score + 1e-9)
    importance: dict[str, float] = {}
    if len(completed) >= 8:
        try:
            importance = {k: round(float(v), 4) for k, v in optuna.importance.get_param_importances(study).items()}
        except Exception:  # noqa: BLE001 - importance is a bonus, never a failure
            logger.debug("param importance unavailable", exc_info=True)
    stopped_early = len(completed) < n_trials
    return {
        **base,
        "applied": applied,
        "n_trials": len(completed),
        "best_params": best.params,
        "best_score": round(float(best.value), 6),
        "default_score": _num(default_score),
        "improvement": _num(best.value - default_score) if np.isfinite(default_score) else None,
        "history": history,
        "param_importance": importance,
        "duration_sec": round(time.perf_counter() - started, 2),
        "note": (
            "No trial beat the default hyper-parameters, so the defaults were kept." if not applied
            else (f"Stopped after {len(completed)} of {n_trials} trials (time budget)." if stopped_early else None)
        ),
    }


def _num(value: float) -> float | None:
    return round(float(value), 6) if np.isfinite(value) else None
