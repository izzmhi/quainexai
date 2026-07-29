"""WebSocket endpoint and connection registry.

Purpose:
    Provide the persistent, bidirectional channel that later phases need, and
    the connection bookkeeping required to push messages to clients.

Why now, when there is nothing to stream yet:
    Phase 4 (voice) needs to stream partial transcripts and speech as they are
    produced; Phase 6 (phone control) needs to push notifications and live system
    status. Both are push-shaped, and request/response cannot express them.
    Establishing the frame protocol and lifecycle here means those phases add
    message *types*, not transport.

Architecture:
    client --(ws)--> /ws
        -> ConnectionManager.connect()      register socket
        -> receive loop                     parse frame, dispatch by "type"
        -> ConnectionManager.disconnect()   deregister on close or error

    Frames are JSON objects with a ``type`` discriminator:
        in : {"type": "ping"}                  out: {"type": "pong"}
        in : {"type": "echo", "data": "hi"}    out: {"type": "echo", "data": "hi"}
        bad: anything else                     out: {"type": "error", ...}

Security note:
    This endpoint is unauthenticated and is why the default bind address is
    127.0.0.1. It must not be exposed beyond localhost until Phase 6 adds
    authentication.

Dependencies:
    fastapi, starlette, quainex.core.logging

Future improvements:
    * Require a JWT during the handshake (Phase 6).
    * Add per-connection rate limiting and a max-frame-size guard.
    * Replace the ad-hoc frame dict with Pydantic models once the protocol grows.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from quainex.api.middleware import new_correlation_id
from quainex.core.exceptions import QuainexError
from quainex.core.logging import get_logger

router = APIRouter(tags=["realtime"])
_log = get_logger(__name__)

#: Reject frames larger than this to bound memory use per connection.
MAX_FRAME_BYTES = 64 * 1024


class ConnectionManager:
    """Tracks open WebSocket connections and fans messages out to them."""

    def __init__(self) -> None:
        """Initialise an empty registry."""
        self._connections: dict[str, WebSocket] = {}

    @property
    def count(self) -> int:
        """Number of currently open connections."""
        return len(self._connections)

    async def connect(self, websocket: WebSocket) -> str:
        """Accept a connection and register it.

        Args:
            websocket: The socket to accept.

        Returns:
            The identifier assigned to this connection.
        """
        await websocket.accept()
        connection_id = new_correlation_id()
        self._connections[connection_id] = websocket
        _log.info("ws_connected", connection_id=connection_id, active=self.count)
        return connection_id

    def disconnect(self, connection_id: str) -> None:
        """Deregister a connection.

        Safe to call for an unknown ID, so cleanup paths need no guard.

        Args:
            connection_id: The connection to forget.
        """
        if self._connections.pop(connection_id, None) is not None:
            _log.info("ws_disconnected", connection_id=connection_id, active=self.count)

    async def send(self, connection_id: str, payload: dict[str, Any]) -> None:
        """Send one frame to one connection.

        Args:
            connection_id: Target connection.
            payload: JSON-serialisable frame body.
        """
        websocket = self._connections.get(connection_id)
        if websocket is not None:
            await websocket.send_json(payload)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        """Send one frame to every open connection.

        Sockets that fail mid-send are dropped rather than aborting the fan-out,
        so one dead client cannot deny the message to the rest.

        Args:
            payload: JSON-serialisable frame body.
        """
        for connection_id, websocket in list(self._connections.items()):
            try:
                await websocket.send_json(payload)
            except (WebSocketDisconnect, RuntimeError):
                self.disconnect(connection_id)


#: Process-wide registry. Later phases publish to this to push to clients.
manager = ConnectionManager()


def _handle_frame(raw: str) -> dict[str, Any]:
    """Parse an inbound frame and build the reply.

    Kept synchronous and side-effect free so it can be unit tested without a
    live socket.

    Args:
        raw: The raw text frame received from the client.

    Returns:
        The frame to send back.
    """
    if len(raw.encode("utf-8")) > MAX_FRAME_BYTES:
        return {"type": "error", "error": "frame_too_large", "max_bytes": MAX_FRAME_BYTES}

    try:
        frame = json.loads(raw)
    except json.JSONDecodeError:
        return {"type": "error", "error": "invalid_json"}

    if not isinstance(frame, dict):
        return {"type": "error", "error": "frame_must_be_object"}

    match frame.get("type"):
        case "ping":
            return {"type": "pong"}
        case "echo":
            return {"type": "echo", "data": frame.get("data")}
        case unknown:
            return {"type": "error", "error": "unknown_frame_type", "received": unknown}


def _authenticate(websocket: WebSocket) -> bool:
    """Authenticate a socket before accepting it.

    Browsers cannot set an ``Authorization`` header on a WebSocket, so the token
    is taken from the query string, with the header still honoured for native
    clients that can send one. A query-string token does leak into access logs,
    which is why these tokens are short-lived.

    Args:
        websocket: The incoming socket, not yet accepted.

    Returns:
        Whether the caller may connect.
    """
    container = getattr(websocket.app.state, "container", None)
    if container is None or not container.settings.auth_required:
        return True
    if container.tokens is None:
        return False

    header = websocket.headers.get("Authorization", "")
    scheme, _, header_token = header.partition(" ")
    token = header_token.strip() if scheme.lower() == "bearer" else ""
    token = token or websocket.query_params.get("token", "")

    try:
        container.tokens.verify(token)
    except QuainexError:
        return False
    return True


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Serve a WebSocket connection for the lifetime of the client.

    Args:
        websocket: The incoming socket.
    """
    if not _authenticate(websocket):
        # Closed before `accept()`, so an unauthenticated client never reaches
        # the message loop. 1008 is the policy-violation close code.
        _log.warning("ws_rejected", reason="authentication_failed")
        await websocket.close(code=1008, reason="Authentication required")
        return

    connection_id = await manager.connect(websocket)
    try:
        await websocket.send_json({"type": "welcome", "connection_id": connection_id})
        while True:
            raw = await websocket.receive_text()
            await websocket.send_json(_handle_frame(raw))
    except WebSocketDisconnect:
        # Normal client-initiated close; not an error worth a traceback.
        pass
    finally:
        manager.disconnect(connection_id)
