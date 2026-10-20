"""Interactive preprocessing endpoints."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter

from app.deps import RadioDep, SessionDep, SettingsDep, require_stage
from app.errors import BadRequestError, AppError
from app.schemas.common import ErrorResponse
from app.schemas.preprocessing import (
    CleanRequest,
    CleanResponse,
    ClassBalanceResponse,
    MissingResponse,
    ResetResponse,
    TargetSummaryResponse,
)
from app.services import preprocessing
from app.services.tasks import is_regression
from app.state.session_store import Stage

router = APIRouter(
    prefix="/api/v1/sessions/{session_id}",
    tags=["preprocessing"],
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)


@router.get("/missing", response_model=MissingResponse, summary="Missing-value scan")
async def missing(session: SessionDep, radio: RadioDep) -> MissingResponse:
    """Count and percentage of missing values per column."""
    report = await asyncio.to_thread(preprocessing.missing_report, session.working_df)
    channel = radio.channel(session)
    if report["has_missing"]:
        channel.warn("CLEAN", f"MISSING VALUE SCAN {report['total_missing']} MISSING CELLS FOUND")
    else:
        channel.info("CLEAN", "MISSING VALUE SCAN CLEAN - NO MISSING CELLS")
    return MissingResponse(**report)


@router.post(
    "/clean",
    response_model=CleanResponse,
    summary="Drop or impute missing values",
    responses={400: {"model": ErrorResponse}},
)
async def clean(body: CleanRequest, session: SessionDep, radio: RadioDep) -> CleanResponse:
    """Apply a drop/impute strategy to the working DataFrame (undo with ``/reset``)."""
    require_stage(session, Stage.TARGET_SET, "clean data")
    channel = radio.channel(session)
    async with session.lock:
        channel.info("CLEAN", "CLEAN DATA STARTED")
        try:
            cleaned, report = await asyncio.to_thread(
                preprocessing.clean_dataframe,
                session.working_df,
                session.target,
                body.strategy,
                body.method,
                body.columns,
                body.knn_neighbors,
            )
        except AppError as exc:
            channel.error("CLEAN", f"CLEAN DATA FAILED - {exc.error_code}")
            raise
        session.working_df = cleaned
        session.stage = Stage.CLEANED
        session.reset_training()
    for warning in report["warnings"]:
        channel.warn("CLEAN", warning)
    channel.success(
        "CLEAN",
        f"CLEAN DATA COMPLETE ROWS {report['rows_before']}->{report['rows_after']} "
        f"MISSING {report['missing_before']}->{report['missing_after']}",
    )
    return CleanResponse(**report, stage=session.stage.name)


@router.post("/reset", response_model=ResetResponse, summary="Undo cleaning (restore the raw upload)")
async def reset(session: SessionDep, radio: RadioDep) -> ResetResponse:
    """Restore the working DataFrame from the raw snapshot."""
    async with session.lock:
        session.working_df = session.raw_df.copy()
        session.stage = Stage.TARGET_SET if session.target else Stage.UPLOADED
        session.reset_training()
    radio.channel(session).info("CLEAN", "DATA RESET TO RAW UPLOAD")
    return ResetResponse(rows=int(len(session.working_df)), stage=session.stage.name)


@router.get(
    "/class-balance",
    response_model=ClassBalanceResponse,
    summary="Target class distribution and imbalance verdict",
)
async def class_balance(session: SessionDep, settings: SettingsDep, radio: RadioDep) -> ClassBalanceResponse:
    """Class counts/ratios; minority ratio below 0.30 raises a STRICT warning."""
    require_stage(session, Stage.TARGET_SET, "analyse class balance")
    assert session.target is not None
    if is_regression(session.task):
        raise BadRequestError(
            "The target is continuous (regression mode); use /target-summary instead.", error_code="NOT_CLASSIFICATION"
        )
    report = await asyncio.to_thread(
        preprocessing.class_balance, session.working_df[session.target], session.target, settings.imbalance_threshold
    )
    channel = radio.channel(session)
    if report["imbalance_warning"]:
        channel.warn("CLEAN", f"IMBALANCE WARNING STRICT MINORITY {report['minority_ratio'] * 100:.2f} PERCENT")
    elif report["single_class"]:
        channel.warn("CLEAN", "SINGLE CLASS TARGET")
    else:
        channel.info("CLEAN", "CLASS BALANCE OK")
    return ClassBalanceResponse(**report)


@router.get(
    "/target-summary",
    response_model=TargetSummaryResponse,
    summary="Distribution of a continuous (regression) target",
    responses={400: {"model": ErrorResponse}},
)
async def target_summary(session: SessionDep, radio: RadioDep) -> TargetSummaryResponse:
    """Mean, spread, quantiles, skew and a histogram of the regression target."""
    require_stage(session, Stage.TARGET_SET, "summarise the target")
    assert session.target is not None
    if not is_regression(session.task):
        raise BadRequestError(
            "The target is categorical (classification mode); use /class-balance instead.",
            error_code="NOT_REGRESSION",
        )
    report = await asyncio.to_thread(preprocessing.target_summary, session.working_df[session.target], session.target)
    radio.channel(session).info("CLEAN", f"TARGET SUMMARY MEAN {report['mean']:.4g} STD {report['std']:.4g}")
    return TargetSummaryResponse(**report)
