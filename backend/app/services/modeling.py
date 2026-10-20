"""Training pipeline: dataset preparation, leak-free preprocessing, evaluation and ranking."""
from __future__ import annotations

import inspect
import logging
import time
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from app.config import Settings
from app.errors import BadRequestError, ConflictError
from app.services.analysis import FittedModel, headline_scores, score_matrix
from app.services.crossval import cross_validate_model
from app.services.model_registry import ModelSpec
from app.services.torch_models import fit_torch_with_telemetry
from app.services.tuning import coerce_params, tune_model
from app.services.tasks import is_regression
from app.services.telemetry import (
    detect_gap_overfit,
    detect_overfit,
    detect_underfit,
    detect_underfit_r2,
    fit_boosting_with_telemetry,
    fit_mlp_with_telemetry,
)
from app.state.session_store import JobRecord
from app.utils.safe_math import safe_div

logger = logging.getLogger("mldash.modeling")


@dataclass
class Prepared:
    """A validated, encoded-target dataset ready for splitting."""

    X: pd.DataFrame
    y: np.ndarray
    class_labels: list[str]
    exclusions: dict[str, list[str]]
    warnings: list[str] = field(default_factory=list)
    task: str = "classification"

    @property
    def n_classes(self) -> int:
        """Number of target classes."""
        return len(self.class_labels)


# --- dataset preparation -------------------------------------------------
def identify_exclusions(df: pd.DataFrame, target: str) -> dict[str, list[str]]:
    """Find ID-like (unique per row) and constant columns that should not be features."""
    id_like: list[str] = []
    constant: list[str] = []
    rows = len(df)
    for column in df.columns:
        if column == target:
            continue
        series = df[column]
        unique = int(series.nunique(dropna=False))
        is_floaty = ptypes.is_float_dtype(series)
        if unique <= 1:
            constant.append(column)
        elif rows > 1 and unique == rows and not is_floaty and not ptypes.is_bool_dtype(series):
            id_like.append(column)
    return {"id_like": id_like, "constant": constant}


def _to_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Booleans -> int, datetimes -> epoch seconds, other non-numerics -> str."""
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


def prepare_dataset(
    df: pd.DataFrame, target: str, settings: Settings, task: str = "classification"
) -> Prepared:
    """Validate the working frame and return encoded features/target (409 if NaNs remain)."""
    if target not in df.columns:
        raise BadRequestError(f"Target column '{target}' does not exist.", error_code="TARGET_NOT_FOUND")
    exclusions = identify_exclusions(df, target)
    dropped = set(exclusions["id_like"]) | set(exclusions["constant"])
    features = [c for c in df.columns if c != target and c not in dropped]
    if not features:
        raise BadRequestError("No usable feature columns remain after exclusions.", error_code="NO_FEATURES",
                              details=exclusions)
    nan_counts = {c: int(df[c].isna().sum()) for c in [*features, target] if df[c].isna().any()}
    if nan_counts:
        raise ConflictError(
            "Missing values remain; clean the data first (POST /api/v1/sessions/<id>/clean), then train.",
            error_code="NAN_REMAINING",
            details=nan_counts,
        )
    target_series = df[target]
    numeric_target = ptypes.is_numeric_dtype(target_series) and not ptypes.is_bool_dtype(target_series)
    n_unique = int(target_series.nunique())
    if task == "regression":
        if not numeric_target:
            raise BadRequestError(f"Target '{target}' is not numeric, so it cannot be used for regression.",
                                  error_code="TARGET_NOT_NUMERIC")
        if n_unique < 2:
            raise BadRequestError(f"Target '{target}' has a single value; training needs at least 2.",
                                  error_code="SINGLE_CLASS_TARGET")
    elif numeric_target and n_unique > settings.continuous_target_unique:
        raise BadRequestError(
            f"Target '{target}' has {n_unique} unique numeric values (> {settings.continuous_target_unique}); "
            "switch the task to regression (or bin the target) to train on it.",
            error_code="TARGET_NOT_CLASSIFIABLE",
        )
    elif n_unique < 2:
        raise BadRequestError(f"Target '{target}' has a single class; training needs at least 2.",
                              error_code="SINGLE_CLASS_TARGET")
    if len(df) < settings.min_rows_hard:
        raise BadRequestError(f"Only {len(df)} rows - at least {settings.min_rows_hard} are needed to train.",
                              error_code="TOO_FEW_ROWS")
    warnings: list[str] = []
    if len(df) < settings.min_rows_warning:
        warnings.append(f"Very small dataset ({len(df)} rows); metrics will be unreliable.")
    frame = _to_model_frame(df[features]).reset_index(drop=True)
    if task == "regression":
        return Prepared(
            X=frame, y=target_series.to_numpy(dtype="float64"), class_labels=[], exclusions=exclusions,
            warnings=warnings, task="regression",
        )
    codes, uniques = pd.factorize(target_series, sort=True)
    return Prepared(
        X=frame,
        y=np.asarray(codes, dtype=int),
        class_labels=[str(u) for u in uniques],
        exclusions=exclusions,
        warnings=warnings,
    )


# --- splitting -----------------------------------------------------------
def split_indices(
    indices: np.ndarray, y: np.ndarray, test_size: float, seed: int, stratify: bool = True
) -> tuple[np.ndarray, np.ndarray, bool, str | None]:
    """Stratified split of ``indices``; falls back to non-stratified with a warning (plain random if not stratifying)."""
    if not stratify:
        train, test = train_test_split(indices, test_size=test_size, random_state=seed)
        return train, test, False, None
    try:
        train, test = train_test_split(indices, test_size=test_size, stratify=y[indices], random_state=seed)
        return train, test, True, None
    except ValueError:
        train, test = train_test_split(indices, test_size=test_size, random_state=seed)
        return train, test, False, "Stratified split impossible (a class has too few samples); used a random split."


def resolve_average(n_classes: int, requested: str | None) -> tuple[str, str | None]:
    """Auto-select the F-score average; ``binary`` is invalid for >2 classes."""
    if requested is None:
        return ("binary" if n_classes == 2 else "weighted"), None
    if requested == "binary" and n_classes > 2:
        return "weighted", "average='binary' is invalid for more than 2 classes; using 'weighted'."
    return requested, None


# --- preprocessing & imbalance ------------------------------------------
def build_preprocessor(x_train: pd.DataFrame, scale: bool, settings: Settings) -> ColumnTransformer:
    """One-hot (low cardinality) / ordinal (high cardinality) / optional scaling; fit on train only."""
    numeric = [c for c in x_train.columns if ptypes.is_numeric_dtype(x_train[c])]
    categorical = [c for c in x_train.columns if c not in numeric]
    low = [c for c in categorical if x_train[c].nunique() <= settings.onehot_max_cardinality]
    high = [c for c in categorical if c not in low]
    return ColumnTransformer(
        [
            ("num", StandardScaler() if scale else "passthrough", numeric),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), low),
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), high),
        ],
        remainder="drop",
    )


def apply_imbalance(estimator: Any, y: np.ndarray, n_classes: int, mode: str) -> dict[str, Any]:
    """Configure class weighting on ``estimator``; returns extra ``fit`` kwargs."""
    if mode != "class_weight":
        return {}
    params = estimator.get_params() if hasattr(estimator, "get_params") else {}
    name = type(estimator).__name__
    if "class_weight" in params:
        estimator.set_params(class_weight="balanced")
        return {}
    if name == "XGBClassifier":
        if n_classes == 2:
            estimator.set_params(scale_pos_weight=safe_div(int((y == 0).sum()), int((y == 1).sum()), 1.0))
            return {}
        return {"sample_weight": compute_sample_weight("balanced", y)}
    if name == "CatBoostClassifier":
        estimator.set_params(auto_class_weights="Balanced")
        return {}
    if "sample_weight" in inspect.signature(estimator.fit).parameters:
        return {"sample_weight": compute_sample_weight("balanced", y)}
    return {}


def cap_rows(
    x: np.ndarray, y: np.ndarray, max_rows: int, seed: int, stratify: bool = True
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Stratified random subsample to ``max_rows`` (plain random if a class is too small to stratify)."""
    if len(y) <= max_rows:
        return x, y, False
    idx = np.arange(len(y))
    try:
        if not stratify:
            raise ValueError("regression target: random subsample")
        keep, _ = train_test_split(idx, train_size=max_rows, stratify=y, random_state=seed)
    except ValueError:
        keep, _ = train_test_split(idx, train_size=max_rows, random_state=seed)
    keep = np.sort(keep)
    return x[keep], y[keep], True


def compute_learning_curve(
    spec: ModelSpec,
    prepared: Prepared,
    fit_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    scale: bool,
    handle_imbalance: str,
    settings: Settings,
) -> dict[str, Any] | None:
    """Train/validation score as the training set grows (balanced accuracy, or R^2 for regression).

    Uses only the inner train/validation split, never the test rows. Smallest sizes run first and the loop
    stops early once the time budget is spent, so slow models yield a shorter curve instead of a long wait.
    """
    seed = settings.random_seed
    regression = is_regression(prepared.task)
    stratify = not regression
    n_total = len(fit_idx)
    if n_total < settings.learning_curve_min_rows or len(val_idx) < 2:
        return None
    y_all = prepared.y
    x_val, y_val = prepared.X.iloc[val_idx], y_all[val_idx]
    sizes = sorted({max(int(round(n_total * f)), max(2 * prepared.n_classes, 10)) for f in settings.learning_curve_fractions})
    sizes = [min(n, n_total) for n in sizes]
    started = time.perf_counter()
    points: list[tuple[int, float, float]] = []
    for size in sorted(set(sizes)):
        if len(points) >= 2 and time.perf_counter() - started > settings.learning_curve_time_budget_sec:
            break
        idx, _, _ = cap_rows(fit_idx, y_all[fit_idx], size, seed, stratify)
        y_sub = y_all[idx]
        if len(np.unique(y_sub)) < 2:
            continue
        try:
            x_sub = prepared.X.iloc[idx]
            prep = build_preprocessor(x_sub, scale and spec.scaled, settings)
            prep.fit(x_sub)
            xt = prep.transform(x_sub)
            y_fit = y_sub
            if handle_imbalance == "smote":
                xt, y_fit = _maybe_smote(xt, y_fit, seed, [])
            if spec.key == "svm":
                xt, y_fit, _ = cap_rows(xt, y_fit, settings.svm_max_train_rows, seed, stratify)
            estimator = spec.build(seed, settings.n_jobs, prepared.task)
            fit_kwargs = apply_imbalance(estimator, y_fit, prepared.n_classes, handle_imbalance)
            estimator.fit(xt, y_fit, **fit_kwargs)
            pipeline = Pipeline([("prep", prep), ("clf", estimator)])
            x_eval, y_eval = x_sub, y_sub
            if len(y_eval) > settings.underfit_max_eval_rows:
                pick, _, _ = cap_rows(np.arange(len(y_eval)), y_eval, settings.underfit_max_eval_rows, seed, stratify)
                x_eval, y_eval = x_sub.iloc[pick], y_eval[pick]
            if regression:  # clip so one wild point cannot flatten the chart
                train_score = max(float(r2_score(y_eval, pipeline.predict(x_eval))), -1.0)
                val_score = max(float(r2_score(y_val, pipeline.predict(x_val))), -1.0)
            else:
                train_score = float(balanced_accuracy_score(y_eval, np.asarray(pipeline.predict(x_eval))))
                val_score = float(balanced_accuracy_score(y_val, np.asarray(pipeline.predict(x_val))))
        except Exception:  # noqa: BLE001 - one bad point must not lose the curve (or the model)
            logger.exception("learning-curve point failed for %s at %d rows", spec.key, size)
            continue
        points.append((int(len(idx)), round(train_score, 6), round(val_score, 6)))
    if len(points) < 2:
        return None
    return {
        "train_sizes": [p[0] for p in points],
        "train_score": [p[1] for p in points],
        "val_score": [p[2] for p in points],
        "metric": "r2" if regression else "balanced_accuracy",
    }


def _maybe_smote(x: np.ndarray, y: np.ndarray, seed: int, warnings: list[str]) -> tuple[np.ndarray, np.ndarray]:
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        warnings.append("SMOTE requested but imbalanced-learn is not installed; trained without resampling.")
        return x, y
    smallest = int(np.bincount(y).min())
    if smallest < 2:
        warnings.append("SMOTE skipped: the smallest class has fewer than 2 samples.")
        return x, y
    return SMOTE(random_state=seed, k_neighbors=min(5, smallest - 1)).fit_resample(x, y)


# --- metrics & ranking ---------------------------------------------------
def evaluate(y_true: np.ndarray, y_pred: np.ndarray, average: str, n_classes: int) -> dict[str, Any]:
    """Accuracy, precision, recall, F1 (zero_division=0) and the confusion matrix."""
    kwargs: dict[str, Any] = {"average": average, "zero_division": 0}
    if average == "binary":
        kwargs["pos_label"] = 1
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision": round(float(precision_score(y_true, y_pred, **kwargs)), 6),
        "recall": round(float(recall_score(y_true, y_pred, **kwargs)), 6),
        "f1": round(float(f1_score(y_true, y_pred, **kwargs)), 6),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=np.arange(n_classes)).tolist(),
    }


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray, y_scale: float = 1.0) -> dict[str, Any]:
    """R^2, RMSE and MAE. Inputs are in standardised target units; ``y_scale`` (the training std) converts
    RMSE/MAE back to the original units. R^2 is the same in either unit."""
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 6),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))) * y_scale, 6),
        "mae": round(float(mean_absolute_error(y_true, y_pred)) * y_scale, 6),
    }


def _score(row: dict[str, Any]) -> float:
    """Primary ranking score: R^2 for regression rows, F1 otherwise."""
    return row["score"] if row.get("score") is not None else row["f1"]


def rank_results(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank by primary score desc (F1 or R^2); ties break on accuracy desc / RMSE asc, then train time asc."""

    def key(r: dict[str, Any]) -> tuple[float, ...]:
        if is_regression(r.get("task", "classification")):
            return (-_score(r), r["rmse"], r["train_time_sec"])
        return (-_score(r), -r["accuracy"], r["train_time_sec"])

    ordered = sorted((dict(row) for row in results.values()), key=key)
    top = _score(ordered[0]) if ordered else 0.0
    for position, row in enumerate(ordered, start=1):
        row["rank"] = position
        row["gap_to_top"] = 0.0 if position == 1 else round(max(0.0, top - _score(row)), 6)
    return ordered


def train_signature(test_size: float, scale: bool, imbalance: str, average: str | None) -> tuple[Any, ...]:
    """Settings that must match for results to be comparable on one leaderboard."""
    return (round(test_size, 6), scale, imbalance, average)


def build_train_result(
    results: dict[str, dict[str, Any]],
    failed: dict[str, str],
    split: dict[str, Any] | None,
    warnings: list[str],
    exclusions: dict[str, list[str]],
    average: str | None,
    task: str = "classification",
) -> dict[str, Any]:
    """Assemble the public leaderboard payload."""
    board = rank_results(results)
    top = None
    if board:
        top = {
            "name": board[0]["model"], "abbr": board[0]["abbr"], "f1": board[0].get("f1"),
            "score": _score(board[0]), "score_name": board[0].get("score_name", "F1"),
        }
    return {
        "top_model": top,
        "leaderboard": board,
        "split": split,
        "failed_models": [{"model": k, "error": v} for k, v in failed.items()],
        "warnings": warnings,
        "exclusions": exclusions,
        "average": average,
        "task": task,
    }


# --- tuning / cross-validation helpers ------------------------------------------
def _tune(
    spec: ModelSpec, prepared: Prepared, fit_idx: np.ndarray, *, scale: bool, handle_imbalance: str, avg: str | None,
    n_trials: int, settings: Settings, channel: Any, job: JobRecord,
) -> dict[str, Any]:
    """Optuna search for one model on (a capped sample of) its training rows."""
    seed = settings.random_seed
    regression = is_regression(prepared.task)
    pick, _, capped = cap_rows(np.asarray(fit_idx), prepared.y[fit_idx], settings.tune_max_rows, seed, not regression)
    # cap_rows subsamples rows of its first argument, so passing the indices themselves returns the kept indices
    x_tune, y_tune = prepared.X.iloc[pick], prepared.y[pick]
    heavy = spec.key in ("tabnet", "torch_mlp")
    n_trials = min(n_trials, 8) if heavy else n_trials

    def make_pipeline(params: dict[str, Any]) -> Pipeline:
        prep = build_preprocessor(x_tune, scale and spec.scaled, settings)
        est = spec.build(seed, settings.n_jobs, prepared.task)
        if heavy:
            est.set_params(max_epochs=30)
        if params:
            est.set_params(**params)
        apply_imbalance(est, y_tune, prepared.n_classes, handle_imbalance)
        return Pipeline([("prep", prep), ("clf", est)])

    def progress(done: int, total: int, best: float | None) -> None:
        job.progress[spec.abbr] = {"phase": "tuning", "trial": done, "n_trials": total, "best": best}
        if done == 1 or done == total or done % 5 == 0:
            channel.info("TUNE", f"{spec.abbr} TRIAL {done}/{total} BEST CV={best:.3f}" if best is not None else f"{spec.abbr} TRIAL {done}/{total}")

    channel.info("TUNE", f"{spec.abbr} TUNING STARTED {n_trials} TRIALS ON {len(y_tune)} ROWS")
    job.progress[spec.abbr] = {"phase": "tuning", "trial": 0, "n_trials": n_trials, "best": None}
    report = tune_model(
        spec, make_pipeline, x_tune, y_tune, task=prepared.task, average=avg, n_trials=n_trials, settings=settings,
        progress=progress,
    )
    if capped:
        report["note"] = (report.get("note") or "") + f" Tuned on a sample of {len(y_tune):,} rows."
    job.progress[spec.abbr] = {"phase": "fitting", "trial": report["n_trials"], "n_trials": n_trials, "best": report.get("best_score")}
    if report["applied"]:
        channel.success("TUNE", f"{spec.abbr} TUNED CV {report['default_score']:.3f}->{report['best_score']:.3f}")
    else:
        channel.info("TUNE", f"{spec.abbr} DEFAULTS KEPT NO TRIAL BEAT CV={report.get('default_score') or 0:.3f}")
    return report


def _cross_validate(
    spec: ModelSpec, prepared: Prepared, train_idx: np.ndarray, tuning: dict[str, Any] | None, *, scale: bool,
    handle_imbalance: str, avg: str | None, y_sd: float, folds: int, settings: Settings, channel: Any, job: JobRecord,
) -> dict[str, Any] | None:
    """k-fold CV of one model on (a capped sample of) its training rows. Never fails the model: returns None on error."""
    regression = is_regression(prepared.task)
    try:
        pick, _, capped = cap_rows(np.asarray(train_idx), prepared.y[train_idx], settings.cv_max_rows, settings.random_seed, not regression)
        x_cv, y_cv = prepared.X.iloc[pick], prepared.y[pick]
        params = tuning["best_params"] if tuning and tuning.get("applied") else None
        job.progress[spec.abbr] = {"phase": "cv", "trial": 0, "n_trials": folds, "best": None}
        channel.info("CV", f"{spec.abbr} CROSS-VALIDATION {folds} FOLDS ON {len(y_cv)} ROWS")
        report = cross_validate_model(
            spec, x_cv, y_cv, task=prepared.task, n_classes=prepared.n_classes, average=avg, scale=scale,
            handle_imbalance=handle_imbalance, params=params, y_scale=y_sd, settings=settings, folds=folds,
            build_preprocessor=build_preprocessor, apply_imbalance=apply_imbalance,
        )
        if report is None:
            channel.warn("CV", f"{spec.abbr} CROSS-VALIDATION PRODUCED NO SCORES")
            return None
        if capped:
            report["note"] = ((report.get("note") or "") + f" Used a sample of {len(y_cv):,} of {len(train_idx):,} training rows.").strip()
        channel.info("CV", f"{spec.abbr} CV {report['metric'].upper()}={report['score']:.3f} +/- {report['score_std']:.3f}")
        return report
    except Exception as exc:  # noqa: BLE001 - CV is extra evidence; a failure must not lose the model
        logger.exception("cross-validation of %s failed", spec.key)
        channel.warn("CV", f"{spec.abbr} CROSS-VALIDATION FAILED - {type(exc).__name__}")
        return None


# --- training loop -------------------------------------------------------
def run_training(
    prepared: Prepared,
    specs: list[ModelSpec],
    *,
    test_size: float,
    scale_features: bool,
    handle_imbalance: str,
    average: str | None,
    settings: Settings,
    channel: Any,
    job: JobRecord,
    tune_trials: int = 0,
    cv_folds: int = 0,
) -> dict[str, Any]:
    """Train ``specs`` sequentially; one crash never aborts the rest (runs in a worker thread).

    With ``tune_trials > 0`` every tunable model first gets an Optuna search (cross-validated on its training rows
    only); the best parameters are used for the final fit only if they beat the defaults.
    """
    seed = settings.random_seed
    regression = is_regression(prepared.task)
    stratify = not regression
    warnings = list(prepared.warnings)
    if regression:
        avg, avg_note = None, None
        if handle_imbalance != "none":
            warnings.append("Imbalance handling only applies to classification; ignored for regression.")
            channel.warn("TRAIN", "IMBALANCE HANDLING IGNORED FOR REGRESSION")
            handle_imbalance = "none"
    else:
        avg, avg_note = resolve_average(prepared.n_classes, average)
    if avg_note:
        warnings.append(avg_note)
    all_idx = np.arange(len(prepared.y))
    train_idx, test_idx, stratified, note = split_indices(all_idx, prepared.y, test_size, seed, stratify)
    if note:
        warnings.append(note)
        channel.warn("TRAIN", note)
    inner = None
    if len(train_idx) >= 10:
        fit_idx, val_idx, _, inner_note = split_indices(
            train_idx, prepared.y, settings.val_fraction_of_train, seed, stratify
        )
        inner = (fit_idx, val_idx)
        if inner_note:
            warnings.append(inner_note)
    else:
        warnings.append("Training set too small for a validation split; telemetry disabled.")
    y_sd, y_mu = 1.0, 0.0
    if regression:
        # Train every regressor on a standardised target (statistics from training rows only). R^2 is unchanged by
        # this; RMSE/MAE are scaled back to the original units. SVR/MLP in particular need a unit-scale target.
        y_mu = float(prepared.y[train_idx].mean())
        y_sd = float(prepared.y[train_idx].std()) or 1.0
        prepared = replace(prepared, y=(prepared.y - y_mu) / y_sd)
    x_test, y_test = prepared.X.iloc[test_idx], prepared.y[test_idx]
    split = {
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "stratified": stratified,
        "validation_rows": int(len(inner[1])) if inner else 0,
    }
    if handle_imbalance == "class_weight":
        channel.info("TRAIN", "CLASS WEIGHTING ENABLED")

    rows: dict[str, dict[str, Any]] = {}
    curves: dict[str, dict[str, Any]] = {}
    failed: dict[str, str] = {}
    fitted: dict[str, FittedModel] = {}
    tuning_reports: dict[str, dict[str, Any]] = {}
    classes = np.arange(prepared.n_classes)
    for spec in specs:
        status = ["PENDING", "PENDING", "PENDING"]
        job.model_status[spec.abbr] = status
        job.current = spec.abbr
        channel.info("TRAIN", f"{spec.abbr} TRAINING STARTED")
        stage = 0
        try:
            status[0] = "RUNNING"
            use_telemetry = spec.iterative and inner is not None
            fit_idx = inner[0] if use_telemetry and inner else train_idx
            x_fit, y_fit = prepared.X.iloc[fit_idx], prepared.y[fit_idx]
            prep = build_preprocessor(x_fit, scale_features and spec.scaled, settings)
            prep.fit(x_fit)
            xt_fit = prep.transform(x_fit)
            if handle_imbalance == "smote":
                xt_fit, y_fit = _maybe_smote(xt_fit, y_fit, seed, warnings)
            if spec.key == "svm":
                total = len(y_fit)
                xt_fit, y_fit, capped = cap_rows(xt_fit, y_fit, settings.svm_max_train_rows, seed, stratify)
                if capped:
                    how = "a random sample of" if regression else "a stratified sample of"
                    msg = (f"SVM trained on {how} {len(y_fit):,} of {total:,} training rows "
                           "(kernel SVM is too slow on the full set); it is still scored on the full test set.")
                    warnings.append(msg)
                    channel.warn("TRAIN", f"SVM SAMPLED {len(y_fit)} OF {total} TRAINING ROWS")
            status[0], status[1], stage = "DONE", "RUNNING", 1
            estimator = spec.build(seed, settings.n_jobs, prepared.task)
            report = None
            if tune_trials > 0 and spec.search_space():
                report = _tune(
                    spec, prepared, fit_idx, scale=scale_features, handle_imbalance=handle_imbalance, avg=avg,
                    n_trials=tune_trials, settings=settings, channel=channel, job=job,
                )
                tuning_reports[spec.key] = report
                if report["applied"]:
                    estimator.set_params(**coerce_params(spec.key, report["best_params"]))
            elif tune_trials > 0:
                tuning_reports[spec.key] = {"enabled": True, "applied": False, "n_trials": 0, "history": [],
                                            "best_params": {}, "note": f"{spec.abbr} has no tunable hyper-parameters."}
            fit_kwargs = apply_imbalance(estimator, y_fit, prepared.n_classes, handle_imbalance)

            started = time.perf_counter()
            curve = None
            if use_telemetry and inner:
                xt_val = prep.transform(prepared.X.iloc[inner[1]])
                y_val = prepared.y[inner[1]]
                if spec.key == "mlp":
                    curve = fit_mlp_with_telemetry(
                        estimator, xt_fit, y_fit, xt_val, y_val, None if regression else classes, settings, prepared.task
                    )
                elif spec.key in ("tabnet", "torch_mlp"):
                    curve = fit_torch_with_telemetry(estimator, xt_fit, y_fit, xt_val, y_val, prepared.task)
                else:
                    curve = fit_boosting_with_telemetry(
                        spec.key, estimator, xt_fit, y_fit, xt_val, y_val, prepared.n_classes, fit_kwargs, prepared.task
                    )
            else:
                estimator.fit(xt_fit, y_fit, **fit_kwargs)
            train_time = round(time.perf_counter() - started, 3)
            status[1], status[2], stage = "DONE", "RUNNING", 2

            pipeline = Pipeline([("prep", prep), ("clf", estimator)])
            learning = None
            if settings.learning_curves and inner is not None and not use_telemetry:
                learning = compute_learning_curve(
                    spec, prepared, inner[0], inner[1], scale=scale_features, handle_imbalance=handle_imbalance,
                    settings=settings,
                )
                if learning:
                    curves[spec.key] = learning
                    channel.info("TELEMETRY", f"{spec.abbr} LEARNING CURVE {len(learning['train_sizes'])} POINTS")
            test_pred = np.asarray(pipeline.predict(x_test))
            if regression:
                metrics = evaluate_regression(y_test, test_pred, y_sd)
                test_score = float(r2_score(y_test, test_pred))
            else:
                metrics = evaluate(y_test, test_pred, avg, prepared.n_classes)
                test_score = float(balanced_accuracy_score(y_test, test_pred))
            overfit = detect_overfit(curve["train_loss"], curve["val_loss"], settings.overfit_window) if curve else None
            # Underfit: score the model on (a sample of) the rows it was fit on, before any SMOTE.
            x_seen, y_seen = prepared.X.iloc[fit_idx], prepared.y[fit_idx]
            if len(y_seen) > settings.underfit_max_eval_rows:
                pick, _, _ = cap_rows(
                    np.arange(len(y_seen)), y_seen, settings.underfit_max_eval_rows, seed, stratify
                )
                x_seen, y_seen = x_seen.iloc[pick], y_seen[pick]
            seen_pred = np.asarray(pipeline.predict(x_seen))
            if regression:
                train_score = float(r2_score(y_seen, seen_pred))
                underfit = detect_underfit_r2(train_score, settings.underfit_min_skill)
            else:
                train_score = float(balanced_accuracy_score(y_seen, seen_pred))
                underfit = detect_underfit(train_score, prepared.n_classes, settings.underfit_min_skill)
            # Overfit for every model: train-vs-test gap; iterative models can also trip the per-epoch rule.
            gap = round(train_score - test_score, 6)
            curve_overfit = bool(overfit and overfit.detected)
            gap_check = not spec.memorizes
            gap_overfit = gap_check and detect_gap_overfit(train_score, test_score, settings.overfit_max_gap)
            overfit_label = "OVERFIT DETECTED" if (curve_overfit or gap_overfit) else None
            status[2] = "DONE"
            extra_scores: dict[str, Any] = {}
            if not regression:
                got = score_matrix(pipeline, x_test)
                if got is not None:
                    extra_scores = headline_scores(y_test, got[0])
            cv_report = None
            if cv_folds >= 2:
                cv_report = _cross_validate(
                    spec, prepared, train_idx, report, scale=scale_features, handle_imbalance=handle_imbalance, avg=avg,
                    y_sd=y_sd, folds=cv_folds, settings=settings, channel=channel, job=job,
                )
            summary = None
            if report is not None and report.get("n_trials"):
                summary = {k: report.get(k) for k in ("applied", "n_trials", "default_score", "best_score", "improvement", "metric", "best_params", "note")}
            row = {
                "model": spec.key, "abbr": spec.abbr, "family": spec.family, "hw": spec.hw,
                "name": spec.name.replace("Classifier", "Regressor") if regression else spec.name,
                "task": prepared.task,
                **metrics,
                "score": metrics["r2"] if regression else metrics["f1"],
                "score_name": "R2" if regression else "F1",
                "train_time_sec": train_time,
                "overfit_warning": overfit_label,
                "first_detected_epoch": overfit.first_detected_epoch if curve_overfit else None,
                "underfit_warning": underfit.label,
                "train_score": round(train_score, 6),
                "test_score": round(test_score, 6),
                "score_metric": "r2" if regression else "balanced_accuracy",
                "train_balanced_accuracy": None if regression else underfit.train_balanced_accuracy,
                "test_balanced_accuracy": None if regression else round(test_score, 6),
                "train_test_gap": gap,
                "gap_check": gap_check,
                "curve_overfit": curve_overfit,
                "cte": status,
                "roc_auc": extra_scores.get("roc_auc"),
                "avg_precision": extra_scores.get("avg_precision"),
                "tuned": bool(summary and summary["applied"]),
                "tuning": summary,
                "cv": cv_report,
                "cv_score": cv_report["score"] if cv_report else None,
                "cv_std": cv_report["score_std"] if cv_report else None,
            }
            keep = np.sort(np.random.default_rng(seed).choice(len(fit_idx), min(len(fit_idx), 500), replace=False))
            fitted[spec.key] = FittedModel(
                key=spec.key, abbr=spec.abbr, task=prepared.task, pipeline=pipeline,
                x_train=prepared.X.iloc[np.asarray(fit_idx)[keep]], x_test=x_test, y_test=y_test,
                class_labels=prepared.class_labels, y_mu=y_mu, y_sd=y_sd,
            )
            if curve:
                curves[spec.key] = curve
            if overfit_label:
                status[1] = "WARN"
                why = "VAL LOSS RISING" if curve_overfit else f"TRAIN-TEST GAP={gap:.3f}"
                channel.warn("TELEMETRY", f"OVERFIT DETECTED ON {spec.abbr} {why}")
            if underfit.detected:
                status[1] = "WARN"
                channel.warn(
                    "TELEMETRY",
                    f"UNDERFIT DETECTED ON {spec.abbr} TRAIN {'R2' if regression else 'BAL ACC'}={train_score:.3f}",
                )
            rows[spec.key] = row
            channel.success(
                "TRAIN", f"{spec.abbr} EVALUATION COMPLETE {row['score_name']}={row['score']:.3f}"
            )
        except Exception as exc:  # noqa: BLE001 - isolate the failing model
            logger.exception("model %s failed", spec.key)
            status[stage] = "FAILED"
            failed[spec.key] = f"{type(exc).__name__}: {str(exc)[:200]}"
            channel.error("TRAIN", f"{spec.abbr} FAILED - {type(exc).__name__}")
        finally:
            job.done += 1
    job.current = None
    return {
        "rows": rows, "curves": curves, "failed": failed, "split": split, "warnings": warnings,
        "fitted": fitted, "tuning": tuning_reports,
        "average": avg, "exclusions": prepared.exclusions,
    }
