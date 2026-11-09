"""Model lab: one model, hand-set hyper-parameters, deep diagnostics and parameter sweeps."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter

from app.deps import SessionDep, SettingsDep
from app.schemas.common import ErrorResponse
from app.schemas.lab import LabFitRequest, LabSweepRequest
from app.services import lab
from app.services.lab_catalog import catalog_payload

router = APIRouter(tags=["lab"], responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}})


@router.get("/api/v1/lab/models", summary="Model-lab catalog: models, controls and defaults", response_model=None)
async def lab_models() -> dict:
    """Every lab model with its hyper-parameter controls (kind, range, default, help) and default sweep."""
    return catalog_payload()


@router.post("/api/v1/sessions/{session_id}/lab/{model_name}/fit", summary="Fit one model and return its diagnostics",
             response_model=None)
async def lab_fit(model_name: str, body: LabFitRequest, session: SessionDep, settings: SettingsDep) -> dict:
    """Metrics (train / test / cross-validated), performance charts and model-specific diagnostics.

    Uses the session's working data and target; nothing on the dashboard changes.
    """
    return await asyncio.to_thread(lab.fit_payload, session, settings, model_name, body)


@router.post("/api/v1/sessions/{session_id}/lab/{model_name}/sweep", summary="Validation curve, score grid or learning curve",
             response_model=None)
async def lab_sweep(model_name: str, body: LabSweepRequest, session: SessionDep, settings: SettingsDep) -> dict:
    """Refit the model across parameter values (cross-validated on training rows only)."""
    return await asyncio.to_thread(lab.sweep_payload, session, settings, model_name, body)
