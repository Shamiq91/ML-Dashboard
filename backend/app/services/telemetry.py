"""Per-epoch telemetry capture, overfit detection and downsampling."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, r2_score

from app.config import Settings
from app.services.tasks import is_regression
from app.utils.downsample import select_indices

logger = logging.getLogger("mldash.telemetry")

OVERFIT_LABEL = "OVERFIT DETECTED"
UNDERFIT_LABEL = "UNDERFIT DETECTED"


@dataclass(frozen=True)
class OverfitResult:
    """Outcome of the overfit rule."""

    detected: bool
    first_detected_epoch: int | None
    window: int

    @property
    def label(self) -> str | None:
        """Warning text or ``None``."""
        return OVERFIT_LABEL if self.detected else None


def detect_overfit(train_loss: list[float], val_loss: list[float], window: int = 3) -> OverfitResult:
    """Flag overfit when val loss rises ``window`` epochs in a row while train loss falls.

    Pure function over the full-resolution series; series shorter than ``window + 1``
    can never trigger.
    """
    n = min(len(train_loss), len(val_loss))
    if window < 1 or n < window + 1:
        return OverfitResult(False, None, window)
    streak = 0
    for epoch in range(1, n):
        if val_loss[epoch] > val_loss[epoch - 1] and train_loss[epoch] < train_loss[epoch - 1]:
            streak += 1
            if streak >= window:
                return OverfitResult(True, epoch, window)
        else:
            streak = 0
    return OverfitResult(False, None, window)


@dataclass(frozen=True)
class UnderfitResult:
    """Outcome of the underfit rule."""

    detected: bool
    train_balanced_accuracy: float
    skill: float

    @property
    def label(self) -> str | None:
        """Warning text or ``None``."""
        return UNDERFIT_LABEL if self.detected else None


def detect_underfit(train_balanced_accuracy: float, n_classes: int, min_skill: float) -> UnderfitResult:
    """Flag underfit when balanced accuracy on the training rows is barely above chance.

    Chance is ``1/k`` for ``k`` classes, so ``skill`` rescales balanced accuracy to 0 (chance) .. 1 (perfect).
    A model that cannot even fit the data it was trained on has not learned the signal.
    """
    chance = 1.0 / max(n_classes, 2)
    skill = (train_balanced_accuracy - chance) / (1.0 - chance)
    return UnderfitResult(skill < min_skill, round(float(train_balanced_accuracy), 6), round(float(skill), 6))


def detect_underfit_r2(train_r2: float, min_skill: float) -> UnderfitResult:
    """Regression underfit: R^2 on the training rows is barely above 0 (the "always predict the mean" model).

    R^2 is already chance-adjusted, so ``skill`` is the training R^2 itself.
    """
    return UnderfitResult(train_r2 < min_skill, round(float(train_r2), 6), round(float(train_r2), 6))


def detect_gap_overfit(train_balanced_accuracy: float, test_balanced_accuracy: float, max_gap: float) -> bool:
    """Flag overfit when the model does much better on rows it has seen than on held-out rows.

    Works for every model (no learning curve needed), complementing the per-epoch rule above.
    """
    return (train_balanced_accuracy - test_balanced_accuracy) > max_gap


def _safe_log_loss(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> float:
    return float(log_loss(y_true, proba, labels=classes))


def fit_mlp_with_telemetry(
    estimator: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    classes: np.ndarray | None,
    settings: Settings,
    task: str = "classification",
) -> dict[str, Any]:
    """Manual ``partial_fit`` loop recording train/val loss and a val score per epoch.

    Classification: log-loss and accuracy. Regression: mean squared error and R^2.
    """
    regression = is_regression(task)
    train_loss: list[float] = []
    val_loss: list[float] = []
    val_acc: list[float] = []
    stale = 0
    for _ in range(settings.max_epochs):
        if regression:
            estimator.partial_fit(x_train, y_train)
            p_val = estimator.predict(x_val)
            train_loss.append(float(mean_squared_error(y_train, estimator.predict(x_train))))
            val_loss.append(float(mean_squared_error(y_val, p_val)))
            val_acc.append(float(r2_score(y_val, p_val)))
        else:
            estimator.partial_fit(x_train, y_train, classes=classes)
            p_train = estimator.predict_proba(x_train)
            p_val = estimator.predict_proba(x_val)
            train_loss.append(_safe_log_loss(y_train, p_train, classes))
            val_loss.append(_safe_log_loss(y_val, p_val, classes))
            val_acc.append(float(accuracy_score(y_val, classes[p_val.argmax(axis=1)])))
        if len(train_loss) > 1 and train_loss[-2] - train_loss[-1] < settings.convergence_tol:
            stale += 1
        else:
            stale = 0
        if stale >= settings.convergence_patience:
            break
    return {
        "train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_acc,
        "val_metric": "r2" if regression else "accuracy",
    }


def fit_boosting_with_telemetry(
    kind: str,
    estimator: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    fit_kwargs: dict[str, Any],
    task: str = "classification",
) -> dict[str, Any]:
    """Fit XGBoost/LightGBM with an eval_set and read per-round loss plus a val score (accuracy or R^2)."""
    regression = is_regression(task)
    multiclass = n_classes > 2
    if kind == "xgboost":
        estimator.set_params(eval_metric="rmse" if regression else ("mlogloss" if multiclass else "logloss"))
        estimator.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val)], verbose=False, **fit_kwargs)
        history = estimator.evals_result()
        train_loss = [float(v) for v in next(iter(history["validation_0"].values()))]
        val_loss = [float(v) for v in next(iter(history["validation_1"].values()))]

        def predict(rounds: int) -> np.ndarray:
            return estimator.predict(x_val, iteration_range=(0, rounds))
    else:
        metric = "rmse" if regression else ("multi_logloss" if multiclass else "binary_logloss")
        estimator.fit(
            x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val)], eval_metric=metric, **fit_kwargs
        )
        history = estimator.evals_result_
        names = list(history)
        train_loss = [float(v) for v in history[names[0]][metric]]
        val_loss = [float(v) for v in history[names[-1]][metric]]

        def predict(rounds: int) -> np.ndarray:
            return estimator.predict(x_val, num_iteration=rounds)

    score = (lambda p: float(r2_score(y_val, p))) if regression else (lambda p: float(accuracy_score(y_val, p)))  # noqa: E731
    val_acc = [score(predict(i + 1)) for i in range(len(val_loss))]
    return {
        "train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_acc,
        "val_metric": "r2" if regression else "accuracy",
    }


def build_telemetry_payload(curves: dict[str, Any], max_points: int) -> dict[str, Any]:
    """Downsample stored full-resolution curves to at most ``max_points`` samples."""
    total = len(curves["train_loss"])
    indices = select_indices(total, max_points)
    pick = lambda series: [round(float(series[i]), 6) for i in indices]  # noqa: E731
    return {
        "epochs": indices,
        "train_loss": pick(curves["train_loss"]),
        "val_loss": pick(curves["val_loss"]),
        "val_accuracy": pick(curves["val_accuracy"]),
        "val_metric": curves.get("val_metric", "accuracy"),
        "total_epochs": total,
    }
