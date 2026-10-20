"""Statistical assumption endpoint."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Query

from app.deps import RadioDep, SessionDep, SettingsDep, require_stage
from app.schemas.assumptions import AssumptionsResponse
from app.schemas.common import ErrorResponse
from app.services import assumptions
from app.state.session_store import Stage

router = APIRouter(
    prefix="/api/v1/sessions/{session_id}",
    tags=["assumptions"],
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)

_LABELS = {
    "normality": "NORMALITY SHAPIRO-WILK",
    "independence": "INDEPENDENCE DURBIN-WATSON",
    "homoscedasticity": "HOMOSCEDASTICITY BREUSCH-PAGAN",
}


@router.get(
    "/assumptions",
    response_model=AssumptionsResponse,
    summary="Normality, independence and homoscedasticity tests",
    responses={400: {"model": ErrorResponse}},
)
async def get_assumptions(
    session: SessionDep,
    settings: SettingsDep,
    radio: RadioDep,
    column: str | None = Query(default=None, description="Run Shapiro-Wilk on this single numeric column"),
) -> AssumptionsResponse:
    """Run the three tests on OLS residuals (optionally Shapiro-Wilk on one column)."""
    require_stage(session, Stage.TARGET_SET, "run assumption tests")
    assert session.target is not None
    result = await asyncio.to_thread(
        assumptions.run_assumptions, session.working_df, session.target, settings, column, session.task
    )
    channel = radio.channel(session)
    for key, label in _LABELS.items():
        item = result[key]
        if item["result"] == "N/A":
            channel.info("ASSUMPTIONS", f"{label} N/A (CATEGORICAL TARGET)")
            continue
        line = f"{label} {item['result']} ({item['status']})"
        (channel.info if item["status"] == "GREEN" else channel.warn)("ASSUMPTIONS", line)
    channel.info("ASSUMPTIONS", f"OVERALL STATUS {result['overall_status']}")
    return AssumptionsResponse(**result)
