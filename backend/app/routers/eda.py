"""EDA endpoints."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter

from app.deps import RadioDep, SessionDep, SettingsDep, require_stage
from app.schemas.common import ErrorResponse
from app.schemas.eda import CorrelationResponse, EdaSummaryResponse
from app.services import eda
from app.state.session_store import Stage

router = APIRouter(
    prefix="/api/v1/sessions/{session_id}/eda",
    tags=["eda"],
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)


@router.get("/summary", response_model=EdaSummaryResponse, summary="Mean/Std/Min/Max per numeric feature")
async def summary(session: SessionDep, radio: RadioDep) -> EdaSummaryResponse:
    """Descriptive statistics for every numeric feature (target excluded)."""
    require_stage(session, Stage.TARGET_SET, "compute the EDA summary")
    result = await asyncio.to_thread(eda.eda_summary, session.working_df, session.target)
    radio.channel(session).info("EDA", f"SUMMARY STATISTICS READY {result['n_features']} FEATURES")
    return EdaSummaryResponse(**result)


@router.get("/correlation", response_model=CorrelationResponse, summary="Pearson correlation + multicollinearity flags")
async def correlation(session: SessionDep, settings: SettingsDep, radio: RadioDep) -> CorrelationResponse:
    """Correlation matrix over numeric features; |r| above 0.85 is flagged."""
    require_stage(session, Stage.TARGET_SET, "compute correlations")
    result = await asyncio.to_thread(
        eda.correlation, session.working_df, session.target, settings.multicollinearity_threshold
    )
    channel = radio.channel(session)
    for flag in result["multicollinearity_flags"]:
        channel.warn("EDA", f"MULTICOLLINEARITY {flag['feature_a']} / {flag['feature_b']} R={flag['r']:.2f}")
    if not result["multicollinearity_flags"]:
        channel.info("EDA", "CORRELATION MATRIX READY NO MULTICOLLINEARITY")
    return CorrelationResponse(**result)
