"""Application factory: CORS, routers, exception handlers, WebSocket and optional static frontend."""
from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import Settings, get_settings
from app.errors import AppError
from app.routers import analysis, assumptions, data, eda, models, preprocessing, runs, sessions, telemetry, training
from app.schemas.logs import HealthResponse
from app.services.runs import RunStore
from app.state.session_store import SessionStore
from app.utils.sanitize import SafeJSONResponse
from app.ws import radio as radio_ws
from app.ws.radio import Radio

logger = logging.getLogger("mldashboard")

TAGS = [
    {"name": "sessions", "description": "Upload, target selection, session state and logs"},
    {"name": "preprocessing", "description": "Missing values, cleaning, class balance"},
    {"name": "eda", "description": "Summary statistics and correlation"},
    {"name": "assumptions", "description": "Normality, independence, homoscedasticity (on OLS residuals)"},
    {"name": "training", "description": "Background training jobs and leaderboard"},
    {"name": "telemetry", "description": "Per-epoch loss curves and overfit detection"},
    {"name": "models", "description": "Model registry"},
    {"name": "data", "description": "Row preview, column profiles and backend-only transforms"},
    {"name": "analysis", "description": "ROC/PR/threshold, residuals, importance, SHAP, tuning history"},
    {"name": "runs", "description": "Saved runs, datasets and comparison"},
    {"name": "system", "description": "Health"},
]


def _error(status: int, code: str, message: str, details: Any = None) -> SafeJSONResponse:
    return SafeJSONResponse(status_code=status, content={"error_code": code, "message": message, "details": details})


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a fully wired FastAPI application."""
    settings = settings or get_settings()
    store = SessionStore(settings.session_ttl_seconds, settings.max_log_lines)
    radio = Radio(store, settings)
    run_store = RunStore(settings.resolved_data_dir() / settings.run_db_name)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await radio.start()
        cleanup = asyncio.create_task(store.run_cleanup(settings.cleanup_interval_seconds))
        radio.publish(None, "INFO", "INGEST", "API ONLINE")
        try:
            yield
        finally:
            cleanup.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await cleanup
            await radio.stop()
            run_store.close()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="REST + WebSocket backend for the interactive ML pipeline dashboard.",
        openapi_tags=TAGS,
        default_response_class=SafeJSONResponse,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.store = store
    app.state.radio = radio
    app.state.runs = run_store
    app.state.tasks = set()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> SafeJSONResponse:
        return _error(exc.status_code, exc.error_code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> SafeJSONResponse:
        details = [
            {"loc": [str(part) for part in err.get("loc", [])], "msg": str(err.get("msg", "")), "type": err.get("type", "")}
            for err in exc.errors()
        ]
        return _error(422, "VALIDATION_ERROR", "Request validation failed.", details)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> SafeJSONResponse:
        code = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}.get(exc.status_code, f"HTTP_{exc.status_code}")
        return _error(exc.status_code, code, str(exc.detail))

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception) -> SafeJSONResponse:
        logger.exception("unhandled error")
        return _error(500, "INTERNAL_ERROR", "An unexpected error occurred.")

    @app.get("/health", response_model=HealthResponse, tags=["system"], summary="Liveness probe")
    async def health() -> HealthResponse:
        """Return service status and version."""
        return HealthResponse(status="ok", version=settings.app_version)

    for module in (sessions, preprocessing, data, eda, assumptions, training, telemetry, analysis, runs, models):
        app.include_router(module.router)
    app.include_router(radio_ws.router)

    dist = settings.resolved_frontend_dist()
    if dist is not None:
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    return app


app = create_app()
