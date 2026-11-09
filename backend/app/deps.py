"""FastAPI dependencies shared by routers."""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.config import Settings
from app.errors import StageConflictError
from app.state.session_store import Session, SessionStore, Stage
from app.services.runs import RunStore
from app.ws.radio import Radio


def get_app_settings(request: Request) -> Settings:
    """Return the settings attached to the app."""
    return request.app.state.settings


def get_store(request: Request) -> SessionStore:
    """Return the session store."""
    return request.app.state.store


def get_radio(request: Request) -> Radio:
    """Return the radio broadcaster."""
    return request.app.state.radio


def get_runs(request: Request) -> RunStore:
    """Return the persistent run history."""
    return request.app.state.runs


def get_session(session_id: str, request: Request) -> Session:
    """Resolve ``session_id`` from the path or raise 404."""
    return request.app.state.store.get(session_id)


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
StoreDep = Annotated[SessionStore, Depends(get_store)]
RadioDep = Annotated[Radio, Depends(get_radio)]
RunsDep = Annotated[RunStore, Depends(get_runs)]
SessionDep = Annotated[Session, Depends(get_session)]

_HINTS = {
    Stage.TARGET_SET: "Select a target first with POST /api/v1/sessions/{id}/target.",
    Stage.CLEANED: "Run POST /api/v1/sessions/{id}/clean first.",
    Stage.TRAINED: "Train at least one model first with POST /api/v1/sessions/{id}/train.",
}


def require_stage(session: Session, minimum: Stage, action: str) -> None:
    """Raise 409 when ``session`` has not reached ``minimum`` yet."""
    if session.stage < minimum:
        raise StageConflictError(
            f"Cannot {action}: pipeline is at stage {session.stage.name}. {_HINTS.get(minimum, '')}".strip(),
            details={"current_stage": session.stage.name, "required_stage": minimum.name},
        )
