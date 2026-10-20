"""In-memory session registry with TTL expiry and per-session locks."""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any

import pandas as pd

from app.errors import SessionNotFoundError

logger = logging.getLogger("mldash.sessions")


class Stage(IntEnum):
    """Ordered pipeline stages."""

    UPLOADED = 1
    TARGET_SET = 2
    CLEANED = 3
    TRAINED = 4


@dataclass
class JobRecord:
    """State of one background training job."""

    job_id: str
    models: list[str]
    status: str = "RUNNING"
    model_status: dict[str, list[str]] = field(default_factory=dict)
    current: str | None = None
    done: int = 0
    result: dict[str, Any] | None = None
    error: str | None = None
    progress: dict[str, dict[str, Any]] = field(default_factory=dict)  # per-model tuning progress
    run_id: str | None = None  # id of the run record saved when the job completed


@dataclass
class Session:
    """Everything the pipeline remembers about one upload."""

    session_id: str
    filename: str
    raw_df: pd.DataFrame
    working_df: pd.DataFrame
    created_wall: datetime
    created_clock: float
    last_access: float
    target: str | None = None
    task: str = "classification"
    stage: Stage = Stage.UPLOADED
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    telemetry: dict[str, dict[str, Any]] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)
    jobs: dict[str, JobRecord] = field(default_factory=dict)
    logs: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=1000))
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    train_signature: tuple[Any, ...] | None = None
    split: dict[str, Any] | None = None
    exclusions: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    average: str | None = None
    fitted: dict[str, Any] = field(default_factory=dict)  # key -> analysis.FittedModel
    tuning: dict[str, dict[str, Any]] = field(default_factory=dict)
    dataset_id: str | None = None

    def reset_training(self) -> None:
        """Forget trained models (called whenever the data or target changes)."""
        self.results.clear()
        self.telemetry.clear()
        self.failed.clear()
        self.train_signature = None
        self.split = None
        self.exclusions = {}
        self.warnings = []
        self.average = None
        self.fitted.clear()
        self.tuning.clear()


class SessionStore:
    """Registry of sessions; the only place mutable pipeline state lives."""

    def __init__(
        self,
        ttl_seconds: int,
        max_log_lines: int = 1000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._sessions: dict[str, Session] = {}
        self._ttl = ttl_seconds
        self._max_log_lines = max_log_lines
        self.clock = clock

    def __len__(self) -> int:
        return len(self._sessions)

    def create(self, filename: str, df: pd.DataFrame) -> Session:
        """Register a new session for an uploaded DataFrame."""
        now = self.clock()
        session = Session(
            session_id=str(uuid.uuid4()),
            filename=filename,
            raw_df=df,
            working_df=df.copy(),
            created_wall=datetime.now(timezone.utc),
            created_clock=now,
            last_access=now,
            logs=deque(maxlen=self._max_log_lines),
        )
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Session:
        """Return a live session (refreshing its TTL) or raise 404."""
        session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(f"Session '{session_id}' does not exist or has expired.")
        now = self.clock()
        if now - session.last_access > self._ttl:
            self._sessions.pop(session_id, None)
            raise SessionNotFoundError(f"Session '{session_id}' does not exist or has expired.")
        session.last_access = now
        return session

    def delete(self, session_id: str) -> Session:
        """Remove a session, raising 404 if it is unknown."""
        session = self.get(session_id)
        self._sessions.pop(session_id, None)
        return session

    def seconds_left(self, session: Session) -> int:
        """Seconds until the session expires if untouched."""
        return max(0, int(self._ttl - (self.clock() - session.last_access)))

    def purge_expired(self) -> list[str]:
        """Drop every expired session and return their ids."""
        now = self.clock()
        expired = [sid for sid, s in self._sessions.items() if now - s.last_access > self._ttl]
        for sid in expired:
            self._sessions.pop(sid, None)
        if expired:
            logger.info("purged %d expired session(s)", len(expired))
        return expired

    async def run_cleanup(self, interval_seconds: float) -> None:
        """Background loop that purges expired sessions until cancelled."""
        with contextlib.suppress(asyncio.CancelledError):
            while True:
                await asyncio.sleep(interval_seconds)
                self.purge_expired()
