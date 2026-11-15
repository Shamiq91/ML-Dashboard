"""Domain exceptions mapped onto the consistent ``{error_code, message, details}`` schema."""
from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base class for all expected, user-facing errors."""

    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"

    def __init__(
        self,
        message: str,
        *,
        error_code: str | None = None,
        details: Any = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details
        if error_code is not None:
            self.error_code = error_code
        if status_code is not None:
            self.status_code = status_code


class BadRequestError(AppError):
    """400: the input is syntactically fine but semantically unusable."""

    status_code = 400
    error_code = "BAD_REQUEST"


class SessionNotFoundError(AppError):
    """404: unknown or expired session."""

    status_code = 404
    error_code = "SESSION_NOT_FOUND"


class NotFoundError(AppError):
    """404: a resource inside a valid session does not exist."""

    status_code = 404
    error_code = "NOT_FOUND"


class StageConflictError(AppError):
    """409: the pipeline step was called in the wrong order."""

    status_code = 409
    error_code = "STAGE_CONFLICT"


class ConflictError(AppError):
    """409: the operation conflicts with the current data state."""

    status_code = 409
    error_code = "CONFLICT"


class PayloadTooLargeError(AppError):
    """413: upload exceeds the configured limit."""

    status_code = 413
    error_code = "FILE_TOO_LARGE"
