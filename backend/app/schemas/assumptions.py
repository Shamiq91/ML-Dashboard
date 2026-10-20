"""Schemas for statistical assumption tests."""
from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict

from app.schemas.common import ApiModel

Status = Literal["GREEN", "YELLOW", "RED"]
Result = Literal["PASS", "FAIL", "SKIPPED", "N/A"]


class AssumptionBase(ApiModel):
    """Fields shared by every assumption test result."""

    test: str
    result: Result
    status: Status
    detail: str


class NormalityResult(AssumptionBase):
    """Shapiro-Wilk on residuals (or a single column)."""

    statistic: float | None = None
    p_value: float | None = None


class IndependenceResult(AssumptionBase):
    """Durbin-Watson on residuals."""

    statistic: float | None = None


class HomoscedasticityResult(AssumptionBase):
    """Breusch-Pagan on residuals."""

    lm_statistic: float | None = None
    lm_p_value: float | None = None
    f_statistic: float | None = None
    f_p_value: float | None = None


class AssumptionsResponse(ApiModel):
    """All assumption tests plus the worst-of status."""

    normality: NormalityResult
    independence: IndependenceResult
    homoscedasticity: HomoscedasticityResult
    overall_status: Status
    dropped_columns: dict[str, list[str]]
    n_observations: int

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "normality": {"test": "Shapiro-Wilk", "statistic": 0.97, "p_value": 0.03,
                              "result": "FAIL", "status": "YELLOW", "detail": "..."},
                "independence": {"test": "Durbin-Watson", "statistic": 1.98,
                                 "result": "PASS", "status": "GREEN", "detail": "..."},
                "homoscedasticity": {"test": "Breusch-Pagan", "lm_statistic": 30.1,
                                     "lm_p_value": 0.0001, "f_statistic": 8.2, "f_p_value": 0.0001,
                                     "result": "FAIL", "status": "RED", "detail": "..."},
                "overall_status": "RED",
                "dropped_columns": {"constant": [], "collinear": []},
                "n_observations": 7043,
            }
        },
    )
