"""Export a trained model: model card and a downloadable zip."""
from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter
from fastapi.responses import Response

from app.deps import SessionDep, require_stage
from app.schemas.common import ErrorResponse
from app.services import export
from app.state.session_store import Session, Stage

router = APIRouter(
    prefix="/api/v1/sessions/{session_id}",
    tags=["export"],
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)


def _model(session: Session, name: str) -> export.FittedModel:
    require_stage(session, Stage.TRAINED, "export a model")
    return export.fitted_or_404(session, name)


@router.get("/export/{model_name}/card", summary="Model card: training, scores and the input the model expects", response_model=None)
async def card(model_name: str, session: SessionDep) -> dict[str, Any]:
    """What goes into the export's MODEL_CARD.md, as JSON."""
    return await asyncio.to_thread(export.model_card, session, _model(session, model_name))


@router.get("/export/{model_name}", summary="Download the model as a zip (joblib bundle, predict.py, model card)",
            response_class=Response, responses={200: {"content": {"application/zip": {}}}})
async def download(model_name: str, session: SessionDep) -> Response:
    """``model.joblib``, ``predict.py``, ``MODEL_CARD.md``, ``model_card.json``, ``requirements.txt``, ``example_input.csv``."""
    m = _model(session, model_name)
    data = await asyncio.to_thread(export.export_zip, session, m)
    return Response(data, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{export.export_name(session, m)}"'})
