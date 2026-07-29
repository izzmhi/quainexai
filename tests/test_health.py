"""Tests for the health endpoint and correlation-ID middleware."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quainex.api.middleware import CORRELATION_ID_HEADER

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


def test_health_returns_ok(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["app"] == "Quainex"
    assert body["environment"] == "dev"
    assert body["uptime_seconds"] >= 0


def test_health_reports_ai_unavailable_without_a_key(client: TestClient):
    # The whole point: no credentials must not make the process look unhealthy.
    body = client.get("/health").json()
    assert body["ai"]["provider"] == "anthropic"
    assert body["ai"]["available"] is False
    assert body["status"] == "ok"


def test_response_carries_a_correlation_id(client: TestClient):
    response = client.get("/health")
    assert response.headers[CORRELATION_ID_HEADER]


def test_inbound_correlation_id_is_honoured(client: TestClient):
    response = client.get("/health", headers={CORRELATION_ID_HEADER: "caller-supplied-id"})
    assert response.headers[CORRELATION_ID_HEADER] == "caller-supplied-id"


def test_correlation_ids_differ_between_requests(client: TestClient):
    first = client.get("/health").headers[CORRELATION_ID_HEADER]
    second = client.get("/health").headers[CORRELATION_ID_HEADER]
    assert first != second


def test_docs_are_served_in_development(client: TestClient):
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200
