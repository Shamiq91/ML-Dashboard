"""Request bodies for the model lab."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LabFitRequest(BaseModel):
    """Fit one lab model with hand-set hyper-parameters."""

    params: dict[str, Any] = Field(default_factory=dict, description="Hyper-parameters; missing ones use defaults")
    test_size: float = Field(0.2, ge=0.1, le=0.5)
    scale: bool = True
    features: list[str] | None = Field(None, description="Subset of usable columns (default: all)")
    cv: bool = Field(True, description="Also k-fold cross-validate on the training rows")
    row: int = Field(0, ge=0, description="KNN: test row to explain")
    tree_index: int = Field(0, ge=0, description="XGBoost / random forest: which tree to draw")
    tree_depth: int = Field(3, ge=1, le=8, description="Tree drawing depth")


class LabSweepRequest(BaseModel):
    """Validation curve (one parameter), score grid (two) or learning curve (param='train_size')."""

    params: dict[str, Any] = Field(default_factory=dict)
    test_size: float = Field(0.2, ge=0.1, le=0.5)
    scale: bool = True
    features: list[str] | None = None
    param: str = Field(..., min_length=1)
    values: list[Any] | None = Field(None, max_length=20)
    param2: str | None = None
    values2: list[Any] | None = Field(None, max_length=10)
