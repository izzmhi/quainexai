"""Tests for the error envelope and the internal-detail leak guard."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from quainex.api.app import create_app
from quainex.config.settings import Settings
from quainex.core.exceptions import ProviderNotConfiguredError

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

#: Text that must never reach a client through an error response.
LEAKY_DETAIL = "internal-path-/opt/quainex/secret.key"


def _app_with_failing_routes(settings: Settings) -> TestClient:
    app = create_app(settings)

    @app.get("/_boom")
    async def boom() -> None:
        raise RuntimeError(LEAKY_DETAIL)

    @app.get("/_known_failure")
    async def known_failure() -> None:
        raise ProviderNotConfiguredError("anthropic")

    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def dev_client(settings: Settings) -> Iterator[TestClient]:
    with _app_with_failing_routes(settings) as client:
        yield client


@pytest.fixture
def prod_client(tmp_path: Path) -> Iterator[TestClient]:
    prod_settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        environment="prod",
        log_dir=tmp_path / "logs",
        anthropic_api_key=None,
    )
    with _app_with_failing_routes(prod_settings) as client:
        yield client


def test_known_failure_uses_its_own_code_and_status(dev_client: TestClient):
    response = dev_client.get("/_known_failure")
    assert response.status_code == 503

    error = response.json()["error"]
    assert error["code"] == "provider_not_configured"
    assert "anthropic" in error["message"]
    assert error["correlation_id"]


def test_not_found_uses_the_same_envelope(dev_client: TestClient):
    response = dev_client.get("/does-not-exist")
    assert response.status_code == 404

    error = response.json()["error"]
    assert error["code"] == "not_found"
    assert error["correlation_id"]


def test_unexpected_error_returns_500_with_a_correlation_id(dev_client: TestClient):
    response = dev_client.get("/_boom")
    assert response.status_code == 500
    assert response.json()["error"]["correlation_id"]


def test_production_does_not_leak_internal_detail(prod_client: TestClient):
    response = prod_client.get("/_boom")
    assert response.status_code == 500

    body = response.text
    assert LEAKY_DETAIL not in body, "internal detail must never reach the client"
    assert "RuntimeError" not in body
    assert "Traceback" not in body

    error = response.json()["error"]
    assert error["code"] == "internal_error"
    assert error["correlation_id"], "the caller still gets a handle for support"


def test_development_includes_detail_to_speed_up_debugging(dev_client: TestClient):
    error = dev_client.get("/_boom").json()["error"]
    assert "RuntimeError" in error["message"]


def test_docs_are_disabled_in_production(prod_client: TestClient):
    assert prod_client.get("/docs").status_code == 404
    assert prod_client.get("/openapi.json").status_code == 404
