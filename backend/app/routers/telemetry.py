"""Telemetry endpoint."""
from __future__ import annotations

from fastapi import APIRouter

from app.deps import SessionDep, SettingsDep, require_stage
from app.errors import NotFoundError
from app.schemas.common import ErrorResponse
from app.schemas.telemetry import TelemetryResponse
from app.services.model_registry import resolve_model
from app.services.telemetry import build_telemetry_payload
from app.state.session_store import Stage

router = APIRouter(
    prefix="/api/v1/sessions/{session_id}",
    tags=["telemetry"],
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)


@router.get(
    "/telemetry/{model_name}",
    response_model=TelemetryResponse,
    summary="Per-epoch curves (XGB/LGB/MLP) or a learning curve (every other model)",
)
async def telemetry(model_name: str, session: SessionDep, settings: SettingsDep) -> TelemetryResponse:
    """Return <=40 evenly spaced points (first and last included) plus the overfit verdict."""
    require_stage(session, Stage.TRAINED, "read telemetry")
    spec = resolve_model(model_name)
    if spec.key not in session.results:
        raise NotFoundError(f"Model '{spec.abbr}' has not been trained in this session.", error_code="MODEL_NOT_TRAINED")
    curves = session.telemetry.get(spec.key)
    if curves is None:
        raise NotFoundError(
            f"No telemetry exists for '{spec.abbr}' (too little data for a validation split, or the curve failed).",
            error_code="NO_TELEMETRY",
        )
    row = session.results[spec.key]
    if "train_sizes" in curves:
        return TelemetryResponse(
            model=spec.key,
            abbr=spec.abbr,
            kind="learning_curve",
            train_sizes=curves["train_sizes"],
            train_score=curves["train_score"],
            val_score=curves["val_score"],
            metric=curves.get("metric"),
        )
    payload = build_telemetry_payload(curves, settings.telemetry_max_points)
    return TelemetryResponse(
        model=spec.key,
        abbr=spec.abbr,
        overfit_warning=row.get("overfit_warning"),
        first_detected_epoch=row.get("first_detected_epoch"),
        **payload,
    )
