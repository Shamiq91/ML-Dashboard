"""Model catalog endpoints."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.deps import SessionDep
from app.schemas.registry import RegistryEntry, RegistryResponse
from app.services.model_registry import CATALOG, ModelSpec
from app.schemas.common import ErrorResponse

router = APIRouter(tags=["models"])


def _entry(spec: ModelSpec, results: dict[str, dict[str, Any]] | None = None) -> RegistryEntry:
    available, reason = spec.availability()
    state = "N/A" if not available else "AVAIL"
    last_f1 = last_score = None
    if results and spec.key in results:
        state, last_f1 = "TRAINED", results[spec.key].get("f1")
        last_score = results[spec.key].get("score", last_f1)
    return RegistryEntry(
        key=spec.key, abbr=spec.abbr, name=spec.name, family=spec.family, hw=spec.hw,
        hyperparams=spec.hyperparams, hyperparams_text=spec.hyperparams_text(),
        est_train_time=spec.est_time, available=available, reason=reason,
        iterative=spec.iterative, note=spec.note, state=state, last_f1=last_f1, last_score=last_score,
        regression=spec.supports("regression"),
        tunable=bool(spec.search_space()),
        search_space={k: list(v) for k, v in spec.search_space().items()},
    )


@router.get("/api/v1/models", response_model=RegistryResponse, summary="Catalog of every available model")
async def list_models() -> RegistryResponse:
    """All 23 models with family, hardware, hyperparameters and availability."""
    models = [_entry(spec) for spec in CATALOG]
    return RegistryResponse(models=models, total=len(models))


@router.get(
    "/api/v1/sessions/{session_id}/models",
    response_model=RegistryResponse,
    summary="Catalog with per-session TRAINED state and last F1",
    responses={404: {"model": ErrorResponse}},
)
async def list_session_models(session: SessionDep) -> RegistryResponse:
    """Catalog annotated with what has been trained in this session."""
    models = [_entry(spec, session.results) for spec in CATALOG]
    return RegistryResponse(models=models, total=len(models))
