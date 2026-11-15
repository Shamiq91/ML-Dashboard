"""Schemas for training telemetry."""
from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict

from app.schemas.common import ApiModel


class TelemetryResponse(ApiModel):
    """Per-epoch curves (``kind="epochs"``) or a learning curve (``kind="learning_curve"``)."""

    model: str
    abbr: str
    kind: Literal["epochs", "learning_curve"] = "epochs"
    # epochs kind (iterative models); empty for learning curves
    epochs: list[int] = []
    train_loss: list[float] = []
    val_loss: list[float] = []
    val_accuracy: list[float] = []  # validation accuracy (classification) or R^2 (regression), see val_metric
    val_metric: Literal["accuracy", "r2"] = "accuracy"
    overfit_warning: str | None = None
    first_detected_epoch: int | None = None
    total_epochs: int = 0
    # learning_curve kind (every other model): scores by number of training rows
    train_sizes: list[int] = []
    train_score: list[float] = []
    val_score: list[float] = []
    metric: str | None = None

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "model": "mlp", "abbr": "MLP", "epochs": [0, 5, 10],
                "train_loss": [0.69, 0.5, 0.4], "val_loss": [0.69, 0.52, 0.45],
                "val_accuracy": [0.58, 0.75, 0.8], "overfit_warning": None,
                "first_detected_epoch": None, "total_epochs": 11,
            }
        },
    )
