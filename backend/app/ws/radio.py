"""Radio: stamped log lines, REST history and a thread-safe WebSocket broadcaster."""
from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.config import Settings
from app.state.session_store import Session, SessionStore

logger = logging.getLogger("mldash.radio")
router = APIRouter()

_LEVELS = {"INFO": logging.INFO, "WARN": logging.WARNING, "ERROR": logging.ERROR, "SUCCESS": logging.INFO}


def format_offset(seconds: float) -> str:
    """Format an elapsed time as ``T+MM:SS`` (``T+HH:MM:SS`` after one hour)."""
    total = max(0, int(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"T+{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"T+{minutes:02d}:{secs:02d}"


class RadioChannel:
    """A radio bound to one session so services need not pass it around."""

    def __init__(self, radio: "Radio", session: Session | None) -> None:
        self._radio = radio
        self._session = session

    def info(self, stage: str, message: str) -> None:
        """Publish an INFO line."""
        self._radio.publish(self._session, "INFO", stage, message)

    def warn(self, stage: str, message: str) -> None:
        """Publish a WARN line."""
        self._radio.publish(self._session, "WARN", stage, message)

    def error(self, stage: str, message: str) -> None:
        """Publish an ERROR line."""
        self._radio.publish(self._session, "ERROR", stage, message)

    def success(self, stage: str, message: str) -> None:
        """Publish a SUCCESS line."""
        self._radio.publish(self._session, "SUCCESS", stage, message)


class Radio:
    """Connection manager plus a queue-fed broadcaster safe to call from any thread."""

    def __init__(self, store: SessionStore, settings: Settings) -> None:
        self._store = store
        self._settings = settings
        self._by_session: dict[str, set[WebSocket]] = {}
        self._global: set[WebSocket] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[tuple[str | None, dict[str, Any]]] | None = None
        self._task: asyncio.Task[None] | None = None
        self._server_started = store.clock()

    # -- lifecycle ---------------------------------------------------------
    async def start(self) -> None:
        """Bind to the running loop and start the broadcaster task."""
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        self._task = asyncio.create_task(self._pump())

    async def stop(self) -> None:
        """Stop the broadcaster task."""
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None
        self._loop = None

    def channel(self, session: Session | None) -> RadioChannel:
        """Return a channel bound to ``session`` (or the server when ``None``)."""
        return RadioChannel(self, session)

    # -- publishing --------------------------------------------------------
    def publish(self, session: Session | None, level: str, stage: str, message: str) -> dict[str, Any]:
        """Stamp, store, log and enqueue a line. Safe to call from worker threads."""
        origin = session.created_clock if session else self._server_started
        record = {
            "timestamp": format_offset(self._store.clock() - origin),
            "level": level,
            "stage": stage,
            "message": message.upper(),
            "ts_iso": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        if session is not None:
            session.logs.append(record)
        logger.log(_LEVELS.get(level, logging.INFO), "[%s] %s %s", stage, record["timestamp"], record["message"])
        loop, queue = self._loop, self._queue
        if loop is not None and queue is not None:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(queue.put_nowait, (session.session_id if session else None, record))
        return record

    # -- connections -------------------------------------------------------
    async def connect(self, websocket: WebSocket, session: Session | None) -> None:
        """Accept a client, replay recent lines, then register it."""
        await websocket.accept()
        if session is not None:
            for line in list(session.logs)[-self._settings.log_replay :]:
                await websocket.send_json(line)
            self._by_session.setdefault(session.session_id, set()).add(websocket)
        else:
            self._global.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        """Forget a client wherever it is registered."""
        self._global.discard(websocket)
        for sid in list(self._by_session):
            self._by_session[sid].discard(websocket)
            if not self._by_session[sid]:
                del self._by_session[sid]

    async def _send(self, websocket: WebSocket, payload: dict[str, Any]) -> None:
        try:
            await asyncio.wait_for(websocket.send_json(payload), self._settings.ws_send_timeout_seconds)
        except Exception:  # noqa: BLE001 - slow/dead client must never break the others
            self.disconnect(websocket)

    async def _broadcast(self, session_id: str | None, payload: dict[str, Any]) -> None:
        if session_id is None:
            targets = set(self._global)
            for group in self._by_session.values():
                targets |= group
        else:
            targets = set(self._by_session.get(session_id, ()))
        if targets:
            await asyncio.gather(*(self._send(ws, payload) for ws in targets))

    async def _pump(self) -> None:
        assert self._queue is not None
        while True:
            session_id, payload = await self._queue.get()
            await self._broadcast(session_id, payload)

    @property
    def client_count(self) -> int:
        """Number of connected clients."""
        return len(self._global) + sum(len(group) for group in self._by_session.values())


@router.websocket("/ws/radio")
async def radio_socket(websocket: WebSocket, session_id: str | None = None) -> None:
    """Live radio feed. ``session_id`` is optional (server-level events only without it)."""
    radio: Radio = websocket.app.state.radio
    store: SessionStore = websocket.app.state.store
    settings: Settings = websocket.app.state.settings
    session: Session | None = None
    if session_id:
        try:
            session = store.get(session_id)
        except Exception:  # noqa: BLE001
            await websocket.accept()
            await websocket.send_json(
                {"type": "error", "error_code": "SESSION_NOT_FOUND", "message": "UNKNOWN OR EXPIRED SESSION"}
            )
            await websocket.close(code=4404)
            return
    await radio.connect(websocket, session)
    try:
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=settings.ws_heartbeat_seconds)
            except asyncio.TimeoutError:
                await websocket.send_json(
                    {"type": "heartbeat", "ts_iso": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
                )
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        radio.disconnect(websocket)
