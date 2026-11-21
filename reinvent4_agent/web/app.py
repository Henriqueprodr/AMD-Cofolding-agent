# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FastAPI app: static page, WebSocket transcript, structure endpoint.

Follows the in-repo service convention (``services/*/app.py``): module-level
pydantic models, ``/health``, ``HTTPException`` for refusals, blocking work
pushed off the loop.

Security posture, stated plainly: this binds to loopback and has NO
authentication. The control that actually matters for a localhost server is
the ``Origin`` check below -- binding to 127.0.0.1 does not stop a page in the
user's browser from POSTing here, and WebSockets are exempt from CORS.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from reinvent4_agent.web import protocol, structures
from reinvent4_agent.web.hub import SessionHub
from reinvent4_agent.web.session import AgentSession

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

_CSP = (
    "default-src 'self'; "
    "script-src 'self' https://cdnjs.cloudflare.com; "
    "style-src 'self' 'unsafe-inline'; "
    "connect-src 'self'; "
    "img-src 'self' data: blob:; "
    "worker-src 'self' blob:; "
    "frame-ancestors 'none'"
)


class StructureRequest(BaseModel):
    """What the viewer asks for. Exactly one source per request."""

    source: str = Field(description="smiles | shared | pdb_id")
    smiles: str | None = None
    path: str | None = None
    pdb_id: str | None = None
    label: str | None = None


def create_app(args, *, allowed_origins: set[str] | None = None) -> FastAPI:
    """Build the app around one agent session."""
    hub = SessionHub()
    session = AgentSession(args, hub)
    cache_dir = (
        Path(args.log_dir) / "structure_cache" if args.log_dir else None
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await session.start()
        logger.info(
            "agent session %s ready with %d tools",
            session.session_id,
            len(session.tools),
        )
        try:
            yield
        finally:
            await session.close()

    app = FastAPI(title="Drug Discovery Agent", lifespan=lifespan)
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.state.session = session
    app.state.hub = hub

    origins = allowed_origins or set()

    def _check_origin(origin: str | None) -> bool:
        # A same-origin fetch from our own page may omit Origin entirely;
        # a cross-site request never does.
        return origin is None or origin in origins

    @app.middleware("http")
    async def _guard(request: Request, call_next):
        if request.url.path.startswith("/api") and not _check_origin(
            request.headers.get("origin")
        ):
            # Returned, not raised: an HTTPException raised inside
            # BaseHTTPMiddleware escapes FastAPI's handlers and surfaces as
            # a 500, which would misreport a blocked request as a bug.
            return JSONResponse(
                status_code=403, content={"detail": "Bad origin"}
            )
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = _CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    # -- pages ---------------------------------------------------------

    app.mount(
        "/static", StaticFiles(directory=str(STATIC_DIR)), name="static"
    )

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "session_id": session.session_id,
            "tools": len(session.tools),
            "subscribers": hub.subscriber_count,
        }

    @app.get("/api/session")
    async def session_info() -> dict[str, Any]:
        return session.describe()

    @app.get("/api/result/{ref:path}")
    async def full_result(ref: str) -> PlainTextResponse:
        payload = session.get_result(ref)
        if payload is None:
            raise HTTPException(status_code=404, detail="No such result")
        return PlainTextResponse(payload)

    @app.post("/api/structure")
    async def structure(request: StructureRequest) -> dict[str, Any]:
        try:
            return await structures.resolve(
                session, request.model_dump(), cache_dir
            )
        except structures.StructureError as exc:
            raise HTTPException(
                status_code=exc.status, detail=str(exc)
            ) from exc

    # -- transcript ----------------------------------------------------

    @app.websocket("/ws")
    async def websocket(ws: WebSocket) -> None:
        if not _check_origin(ws.headers.get("origin")):
            await ws.close(code=1008)
            return

        await ws.accept()
        queue = hub.subscribe()
        try:
            await ws.send_json(session.describe())
            history = hub.replay()
            if history:
                await ws.send_json(protocol.replay_start(len(history)))
                for message in history:
                    await ws.send_json(message)
                await ws.send_json(protocol.replay_end())

            pump = asyncio.create_task(_pump(ws, queue))
            try:
                while True:
                    message = await ws.receive_json()
                    await _handle_client_message(session, hub, message)
            finally:
                pump.cancel()
        except WebSocketDisconnect:
            logger.debug("client disconnected")
        except Exception:  # noqa: BLE001 - one bad socket must not kill the app
            logger.exception("websocket error")
        finally:
            hub.unsubscribe(queue)

    return app


async def _pump(ws: WebSocket, queue: asyncio.Queue) -> None:
    """Forward published messages to one socket."""
    while True:
        message = await queue.get()
        await ws.send_json(message)


async def _handle_client_message(
    session: AgentSession, hub: SessionHub, message: dict
) -> None:
    kind = message.get("type")
    if kind == protocol.C_PING:
        hub.publish(protocol.pong())
    elif kind == protocol.C_STOP:
        session.request_stop()
        hub.publish(
            protocol.status("Stopping after the current tool call...", "warn")
        )
    elif kind == protocol.C_USER_MESSAGE:
        # Turns are serialised inside the session; run detached so the
        # socket keeps reading (Stop has to get through).
        asyncio.create_task(session.handle_input(message.get("text", "")))
    else:
        logger.debug("ignoring unknown client message %r", kind)
