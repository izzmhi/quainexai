"""Tests for the WebSocket endpoint, frame handling and connection registry."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from quainex.api.routes.ws import MAX_FRAME_BYTES, ConnectionManager, _handle_frame

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


# -- frame handling (pure function, no socket needed) ----------------------


def test_ping_returns_pong():
    assert _handle_frame(json.dumps({"type": "ping"})) == {"type": "pong"}


def test_echo_returns_payload():
    assert _handle_frame(json.dumps({"type": "echo", "data": "hello"})) == {
        "type": "echo",
        "data": "hello",
    }


def test_invalid_json_returns_error_frame():
    assert _handle_frame("not json at all") == {"type": "error", "error": "invalid_json"}


def test_non_object_frame_is_rejected():
    assert _handle_frame(json.dumps([1, 2, 3])) == {
        "type": "error",
        "error": "frame_must_be_object",
    }


def test_unknown_frame_type_is_reported():
    result = _handle_frame(json.dumps({"type": "launch_missiles"}))
    assert result["error"] == "unknown_frame_type"
    assert result["received"] == "launch_missiles"


def test_oversized_frame_is_rejected_before_parsing():
    oversized = "x" * (MAX_FRAME_BYTES + 1)
    assert _handle_frame(oversized)["error"] == "frame_too_large"


# -- live socket -----------------------------------------------------------


def test_connection_receives_welcome_then_echoes(client: TestClient):
    with client.websocket_connect("/ws") as websocket:
        welcome = websocket.receive_json()
        assert welcome["type"] == "welcome"
        assert welcome["connection_id"]

        websocket.send_text(json.dumps({"type": "ping"}))
        assert websocket.receive_json() == {"type": "pong"}

        websocket.send_text(json.dumps({"type": "echo", "data": {"n": 1}}))
        assert websocket.receive_json() == {"type": "echo", "data": {"n": 1}}


def test_bad_frame_does_not_close_the_connection(client: TestClient):
    with client.websocket_connect("/ws") as websocket:
        websocket.receive_json()  # welcome

        websocket.send_text("{oops")
        assert websocket.receive_json()["error"] == "invalid_json"

        # The socket must still be usable after a client mistake.
        websocket.send_text(json.dumps({"type": "ping"}))
        assert websocket.receive_json() == {"type": "pong"}


# -- connection registry ---------------------------------------------------


def test_disconnect_is_safe_for_unknown_id():
    manager = ConnectionManager()
    manager.disconnect("never-registered")  # must not raise
    assert manager.count == 0
