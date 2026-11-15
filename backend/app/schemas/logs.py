"""Schemas for radio logs."""
from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict

from app.schemas.common import ApiModel


class LogMessage(ApiModel):
    """One radio-style log line (also the WebSocket payload)."""

    timestamp: str
    level: Literal["INFO", "WARN", "ERROR", "SUCCESS"]
    stage: Literal["INGEST", "CLEAN", "EDA", "ASSUMPTIONS", "TRAIN", "TUNE", "CV", "TELEMETRY"]
    message: str
    ts_iso: str


class LogsResponse(ApiModel):
    """Log history for a session."""

    session_id: str
    logs: list[LogMessage]

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "session_id": "3f0c1c9e",
                "logs": [
                    {"timestamp": "T+01:15", "level": "INFO", "stage": "CLEAN",
                     "message": "CLEAN DATA STARTED", "ts_iso": "2026-01-01T12:00:00Z"}
                ],
            }
        },
    )


class HealthResponse(ApiModel):
    """Liveness payload."""

    status: str
    version: str
