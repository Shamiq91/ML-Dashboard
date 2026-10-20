"""Post-training model analysis: ROC/PR/threshold, residuals, importance, SHAP, tuning history."""
from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Query

from app.deps import SessionDep, SettingsDep, require_stage
from app.errors import BadRequestError, NotFoundError
from app.schemas.common import ErrorResponse
from app.services import analysis
from app.services.model_registry import resolve_model
from app.state.session_store import Session, Stage

router = APIRouter(
    prefix="/api/v1/sessions/{session_id}",
    tags=["analysis"],
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)


def _fitted(session: Session, name: str) -> analysis.FittedModel:
    require_stage(session, Stage.TRAINED, "analyse a model")
    spec = resolve_model(name)
    model = session.fitted.get(spec.key)
    if model is None:
        raise NotFoundError(f"Model '{spec.abbr}' has not been trained in this session.", error_code="MODEL_NOT_TRAINED")
    return model


@router.get("/curves/{model_name}", summary="ROC, precision-recall and threshold analysis (classification)", response_model=None)
async def curves(
    model_name: str, session: SessionDep, settings: SettingsDep, threshold: float = Query(0.5, ge=0, le=1)
) -> dict:
    """Test-set ROC and PR curves; for binary targets also a threshold sweep and the confusion matrix at ``threshold``."""
    return await asyncio.to_thread(analysis.curves_payload, _fitted(session, model_name), settings, threshold)


@router.get("/residuals/{model_name}", summary="Actual vs predicted and residuals (regression)", response_model=None)
async def residuals(model_name: str, session: SessionDep, settings: SettingsDep) -> dict:
    """Scatter, residual histogram and bias for a regression model."""
    return await asyncio.to_thread(analysis.residuals_payload, _fitted(session, model_name), settings)


@router.get("/importance/{model_name}", summary="Permutation + native feature importance", response_model=None)
async def importance(
    model_name: str, session: SessionDep, settings: SettingsDep, repeats: int = Query(5, ge=1, le=20)
) -> dict:
    """Drop in score when each original column is shuffled on the test rows (cached per model and repeat count)."""
    model = _fitted(session, model_name)
    key = f"importance:{repeats}"
    if key not in model.cache:
        model.cache[key] = await asyncio.to_thread(analysis.importance_payload, model, settings, repeats)
    return model.cache[key]


@router.get("/shap/{model_name}", summary="SHAP summary and single-row explanation", response_model=None)
async def shap_values(
    model_name: str, session: SessionDep, settings: SettingsDep,
    class_index: int | None = Query(None, ge=0), row: int = Query(0, ge=0),
) -> dict:
    """Mean |SHAP|, beeswarm points and one row's contributions. Computed once per model (a few seconds), then cached."""
    model = _fitted(session, model_name)
    try:
        return await asyncio.to_thread(analysis.shap_payload, model, settings, class_index, row)
    except BadRequestError:
        raise
    except Exception as exc:  # noqa: BLE001 - SHAP can fail on exotic models; report it instead of a 500
        raise BadRequestError(f"SHAP could not explain {model.abbr}: {type(exc).__name__}: {str(exc)[:160]}", error_code="SHAP_FAILED") from exc


@router.get("/tuning/{model_name}", summary="Hyper-parameter search history", response_model=None)
async def tuning(model_name: str, session: SessionDep) -> dict[str, Any]:
    """Every Optuna trial, the default-vs-best CV score and which parameters mattered most."""
    require_stage(session, Stage.TRAINED, "read tuning results")
    spec = resolve_model(model_name)
    report = session.tuning.get(spec.key)
    if report is None:
        raise NotFoundError(f"'{spec.abbr}' was trained without tuning. Re-run with tune=true.", error_code="NOT_TUNED")
    return {"model": spec.key, "abbr": spec.abbr, **report}
