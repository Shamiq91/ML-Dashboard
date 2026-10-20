"""Schemas for the model catalog."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import ConfigDict

from app.schemas.common import ApiModel


class RegistryEntry(ApiModel):
    """One model in the catalog."""

    key: str
    abbr: str
    name: str
    family: Literal["LIN", "TREE", "ENS", "SVM", "NN", "NB", "KNN"]
    hw: Literal["CPU", "GPU"]
    hyperparams: dict[str, str | int | float | bool]
    hyperparams_text: str
    est_train_time: str
    available: bool
    reason: str | None = None
    iterative: bool = False
    note: str | None = None
    state: Literal["TRAINED", "AVAIL", "N/A"] = "AVAIL"
    last_f1: float | None = None
    last_score: float | None = None
    regression: bool = True  # False for classification-only models (LR, NB, LDA, QDA)
    tunable: bool = False
    search_space: dict[str, list[Any]] = {}  # tunable parameter -> [kind, *bounds/choices]


class RegistryResponse(ApiModel):
    """Catalog of every model the pipeline can run."""

    models: list[RegistryEntry]
    total: int

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "models": [
                    {
                        "key": "xgboost", "abbr": "XGB", "name": "XGBoost", "family": "ENS",
                        "hw": "GPU", "hyperparams": {"n_estimators": 200},
                        "hyperparams_text": "n_estimators=200", "est_train_time": "~3 s",
                        "available": True, "reason": None, "iterative": True, "note": None,
                        "state": "AVAIL", "last_f1": None,
                    }
                ],
                "total": 22,
            }
        },
    )
