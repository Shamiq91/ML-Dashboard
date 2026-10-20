"""Ingestion and session-state endpoints."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, File, UploadFile

from app.deps import RadioDep, RunsDep, SessionDep, SettingsDep, StoreDep
from app.errors import PayloadTooLargeError
from app.schemas.common import ErrorResponse
from app.schemas.logs import LogsResponse
from app.schemas.sessions import (
    DeleteResponse,
    SessionInfo,
    TargetRequest,
    TargetResponse,
    UploadResponse,
)
from app.services import ingestion
from app.state.session_store import Stage

router = APIRouter(
    prefix="/api/v1/sessions",
    tags=["sessions"],
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)


@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload a CSV/Parquet file and open a session",
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
)
async def upload(
    settings: SettingsDep, store: StoreDep, radio: RadioDep, runs: RunsDep, file: UploadFile = File(...)
) -> UploadResponse:
    """Validate and load a file, create a session and return its profile."""
    content = await file.read(settings.max_upload_bytes + 1)
    if len(content) > settings.max_upload_bytes:
        raise PayloadTooLargeError(f"File exceeds the {settings.max_upload_mb} MB limit.")
    filename = file.filename or ""
    df = await asyncio.to_thread(ingestion.parse_upload, filename, content)
    session = store.create(filename, df)
    session.dataset_id = await asyncio.to_thread(runs.upsert_dataset, df, filename)
    profile = await asyncio.to_thread(ingestion.describe_upload, df)
    radio.channel(None).info("INGEST", f"SESSION {session.session_id[:8]} OPENED")
    radio.channel(session).success(
        "INGEST", f"UPLOAD COMPLETE {filename} {profile['total_rows']} ROWS {profile['total_columns']} COLUMNS"
    )
    return UploadResponse(session_id=session.session_id, filename=filename, **profile)


@router.post(
    "/{session_id}/target",
    response_model=TargetResponse,
    summary="Select the prediction target column",
    responses={400: {"model": ErrorResponse}},
)
async def set_target(body: TargetRequest, session: SessionDep, settings: SettingsDep, radio: RadioDep) -> TargetResponse:
    """Validate the target column, store it and reset any previous training."""
    async with session.lock:
        if body.target_column not in session.working_df.columns:
            ingestion.describe_target(session.working_df, body.target_column, settings)  # raises a 400
        task = ingestion.resolve_task(
            session.working_df[body.target_column], body.task, settings, body.target_column
        )
        info = ingestion.describe_target(session.working_df, body.target_column, settings, task)
        session.target = body.target_column
        session.task = task
        session.stage = Stage.TARGET_SET
        session.reset_training()
    channel = radio.channel(session)
    if info["task"] == "regression":
        channel.info(
            "INGEST",
            f"TARGET SET {body.target_column} ({info['n_classes']} UNIQUE VALUES) - REGRESSION MODE",
        )
    else:
        channel.info("INGEST", f"TARGET SET {body.target_column} ({info['n_classes']} CLASSES)")
    for warning in info["warnings"]:
        channel.warn("INGEST", warning)
    return TargetResponse(
        session_id=session.session_id, target_column=body.target_column, stage=session.stage.name,
        **info,
    )


@router.get("/{session_id}", response_model=SessionInfo, summary="Current pipeline stage and metadata")
async def get_session_info(session: SessionDep, store: StoreDep) -> SessionInfo:
    """Return the stage, shapes and trained models of a session."""
    return SessionInfo(
        session_id=session.session_id,
        filename=session.filename,
        stage=session.stage.name,
        target_column=session.target,
        task=session.task,
        rows=int(len(session.working_df)),
        columns=int(session.working_df.shape[1]),
        raw_rows=int(len(session.raw_df)),
        has_missing=bool(session.working_df.isna().any().any()),
        trained_models=list(session.results),
        created_at=session.created_wall.strftime("%Y-%m-%dT%H:%M:%SZ"),
        expires_in_seconds=store.seconds_left(session),
    )


@router.delete("/{session_id}", response_model=DeleteResponse, summary="Delete a session")
async def delete_session(session: SessionDep, store: StoreDep, radio: RadioDep) -> DeleteResponse:
    """Remove a session and its data."""
    store.delete(session.session_id)
    radio.channel(None).info("INGEST", f"SESSION {session.session_id[:8]} CLOSED")
    return DeleteResponse(session_id=session.session_id)


@router.get("/{session_id}/logs", response_model=LogsResponse, summary="Radio log history")
async def get_logs(session: SessionDep, settings: SettingsDep) -> LogsResponse:
    """Return the session's log lines (same payloads as the WebSocket)."""
    return LogsResponse(session_id=session.session_id, logs=list(session.logs)[-settings.max_log_lines :])
