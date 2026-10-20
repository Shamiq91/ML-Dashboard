"""Schemas for model training, jobs and the leaderboard."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import ConfigDict, Field

from app.schemas.common import ApiModel

CteState = Literal["PENDING", "RUNNING", "DONE", "WARN", "FAILED"]


class TrainRequest(ApiModel):
    """Start a training job."""

    models: list[str] = Field(min_length=1, description="Model keys or abbreviations")
    test_size: float = Field(default=0.2, gt=0.05, lt=0.5)
    scale_features: bool = True
    handle_imbalance: Literal["none", "class_weight", "smote"] = "none"
    average: Literal["binary", "macro", "weighted"] | None = None
    tune: bool = Field(default=False, description="Run an Optuna hyper-parameter search per model before the final fit")
    tune_trials: int = Field(default=15, ge=3, le=60, description="Trials per model when tune is true")
    cross_validate: bool = Field(default=True, description="Also k-fold cross-validate every model on its training rows (adds `cv` to each row)")
    cv_folds: int = Field(default=5, ge=3, le=10, description="Folds when cross_validate is true (slow neural models use at most 3)")
    label: str | None = Field(default=None, max_length=80, description="Name for the saved run")

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "models": ["xgboost", "random_forest", "svm", "mlp", "logistic_regression"],
                "test_size": 0.2,
                "scale_features": True,
                "handle_imbalance": "none",
                "average": None,
            }
        },
    )


class TrainStarted(ApiModel):
    """Immediate acknowledgement of a started job."""

    job_id: str
    status: Literal["STARTED"] = "STARTED"
    models: list[str]

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {"job_id": "9e1b...", "status": "STARTED", "models": ["xgboost", "mlp"]}
        },
    )


class TopModel(ApiModel):
    """Best model by its primary score (F1 for classification, R^2 for regression)."""

    name: str
    abbr: str
    f1: float | None = None
    score: float | None = None
    score_name: str = "F1"


class LeaderboardRow(ApiModel):
    """One ranked model."""

    rank: int
    model: str
    abbr: str
    name: str
    family: str
    hw: str
    task: Literal["classification", "regression"] = "classification"
    # classification metrics (None for regression)
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    # regression metrics (None for classification); RMSE/MAE are in the target's original units
    r2: float | None = None
    rmse: float | None = None
    mae: float | None = None
    # primary ranking score: F1 (classification) or R2 (regression)
    score: float | None = None
    score_name: str = "F1"
    gap_to_top: float
    train_time_sec: float
    overfit_warning: str | None = None
    first_detected_epoch: int | None = None
    underfit_warning: str | None = None
    train_score: float | None = None  # balanced accuracy (classification) or R^2 (regression)
    test_score: float | None = None
    score_metric: str | None = None
    train_balanced_accuracy: float | None = None
    test_balanced_accuracy: float | None = None
    train_test_gap: float | None = None
    gap_check: bool = True
    curve_overfit: bool = False
    roc_auc: float | None = None  # binary AUC, or one-vs-rest macro AUC
    avg_precision: float | None = None
    tuned: bool = False
    tuning: dict[str, Any] | None = None  # summary of the hyper-parameter search (full history: GET /tuning/{model})
    cv: dict[str, Any] | None = None  # cross-validation report: per-metric mean/std/folds (see services/crossval.py)
    cv_score: float | None = None  # mean of the ranking metric (F1 or R2) across folds
    cv_std: float | None = None  # standard deviation across folds
    confusion_matrix: list[list[int]] = []
    cte: list[CteState]


class SplitInfo(ApiModel):
    """Train/test split description."""

    train_rows: int
    test_rows: int
    stratified: bool
    validation_rows: int = 0


class FailedModel(ApiModel):
    """A model that crashed without aborting the job."""

    model: str
    error: str


class TrainResult(ApiModel):
    """Leaderboard payload (job result and ``GET /leaderboard``)."""

    top_model: TopModel | None
    leaderboard: list[LeaderboardRow]
    split: SplitInfo | None
    failed_models: list[FailedModel]
    warnings: list[str] = []
    exclusions: dict[str, list[str]] = {}
    average: str | None = None
    task: Literal["classification", "regression"] = "classification"

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "top_model": {"name": "xgboost", "abbr": "XGB", "f1": 0.874},
                "leaderboard": [
                    {
                        "rank": 1, "model": "xgboost", "abbr": "XGB", "name": "XGBoost",
                        "family": "ENS", "hw": "GPU", "accuracy": 0.88, "precision": 0.87,
                        "recall": 0.88, "f1": 0.874, "gap_to_top": 0.0, "train_time_sec": 1.2,
                        "overfit_warning": None, "first_detected_epoch": None,
                        "underfit_warning": None, "train_balanced_accuracy": 0.93,
                        "test_balanced_accuracy": 0.86, "train_test_gap": 0.07, "gap_check": True,
                        "curve_overfit": False,
                        "confusion_matrix": [[900, 60], [80, 370]], "cte": ["DONE", "DONE", "DONE"],
                    }
                ],
                "split": {"train_rows": 5634, "test_rows": 1409, "stratified": True,
                          "validation_rows": 0},
                "failed_models": [],
                "warnings": [],
                "exclusions": {"id_like": [], "constant": []},
                "average": "binary",
            }
        },
    )


class JobStatus(ApiModel):
    """Polling payload for a training job."""

    job_id: str
    status: Literal["RUNNING", "COMPLETE", "FAILED"]
    models: list[str]
    model_status: dict[str, list[CteState]]
    current: str | None
    done: int
    total: int
    result: TrainResult | None
    error: str | None
    progress: dict[str, dict[str, Any]] = {}
    run_id: str | None = None

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "job_id": "9e1b...",
                "status": "RUNNING",
                "models": ["xgboost", "mlp"],
                "model_status": {"XGB": ["DONE", "DONE", "DONE"], "MLP": ["DONE", "RUNNING", "PENDING"]},
                "current": "MLP",
                "done": 1,
                "total": 2,
                "result": None,
                "error": None,
            }
        },
    )
