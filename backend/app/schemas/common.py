"""Shared schema base classes."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ApiModel(BaseModel):
    """Base for every API schema (allows fields such as ``model_status``)."""

    model_config = ConfigDict(protected_namespaces=())


class ErrorResponse(ApiModel):
    """Consistent error payload returned for every non-2xx response."""

    error_code: str
    message: str
    details: Any = None

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "error_code": "SESSION_NOT_FOUND",
                "message": "Session 'abc' does not exist or has expired.",
                "details": None,
            }
        },
    )
