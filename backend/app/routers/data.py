"""Data preview, column profiles and backend-only transforms."""
from __future__ import annotations

import asyncio
from typing import Literal

from fastapi import APIRouter, Query

from app.deps import RadioDep, SessionDep, SettingsDep
from app.errors import ConflictError
from app.schemas.common import ErrorResponse
from app.schemas.data import TransformRequest
from app.services import dataview, features, transforms
from app.services.ingestion import column_kind
from app.state.session_store import Stage

router = APIRouter(
    prefix="/api/v1/sessions/{session_id}",
    tags=["data"],
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)

Source = Literal["working", "raw"]


@router.get("/preview", summary="Page through the rows (sortable, searchable)", response_model=None)
async def preview(
    session: SessionDep, settings: SettingsDep,
    source: Source = "working", offset: int = Query(0, ge=0), limit: int = Query(25, ge=1),
    sort: str | None = None, descending: bool = False, search: str | None = Query(None, max_length=100),
) -> dict:
    """Rows of the working (cleaned) frame or the raw upload, with per-column dtype/missing/unique counts."""
    df = session.working_df if source == "working" else session.raw_df
    return await asyncio.to_thread(dataview.preview, df, offset, limit, sort, descending, search, settings)


@router.get("/profile", summary="Per-column distribution profile", response_model=None)
async def profile(session: SessionDep, settings: SettingsDep, source: Source = "working") -> dict:
    """Histogram + quantiles for numeric columns, top values for categorical ones."""
    df = session.working_df if source == "working" else session.raw_df
    return await asyncio.to_thread(dataview.profile, df, settings)


@router.post(
    "/transform",
    summary="Apply cleaning/feature operations (backend-only: drop columns, outliers, filters, casts, ...)",
    response_model=None,
    responses={400: {"model": ErrorResponse}},
)
async def transform(body: TransformRequest, session: SessionDep, radio: RadioDep) -> dict:
    """Apply an ordered list of operations atomically; trained results are reset. Undo with ``POST /reset``."""
    protected = {session.target} if session.target else set()
    ops = [op.model_dump() for op in body.operations]
    channel = radio.channel(session)
    async with session.lock:
        cleaned, report = await asyncio.to_thread(transforms.apply_operations, session.working_df, ops, protected)
        session.working_df = cleaned
        session.stage = min(session.stage, Stage.CLEANED)
        session.reset_training()
    for step in report["steps"]:
        channel.info("CLEAN", f"TRANSFORM {step['op'].upper()} {step['summary'].upper()}")
    df = session.working_df
    return {**report, "stage": session.stage.name, "column_kinds": {c: column_kind(df[c]) for c in df.columns},
            "missing_total": int(df.isna().sum().sum())}


@router.get("/features/suggest", summary="Feature relevance and ready-to-apply selection / engineering suggestions", response_model=None)
async def suggest_features(session: SessionDep, settings: SettingsDep) -> dict:
    """Mutual information with the target, redundant pairs, and suggestions that each carry the ``/transform`` operation they would run."""
    if not session.target:
        raise ConflictError("Set a target column first; feature relevance is measured against it.", error_code="STAGE_CONFLICT")
    return await asyncio.to_thread(features.feature_report, session.working_df, session.target, session.task, settings)
