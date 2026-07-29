"""Shared pytest fixtures.

Every fixture here exists to keep tests isolated from the developer's machine:
no real ``.env`` is read, no real API key is used, and logs are written to a
temporary directory rather than the repo's ``logs/``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from quainex.api.app import create_app
from quainex.config.settings import Settings

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from fastapi import FastAPI


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove QUAINEX_* variables so the developer's shell cannot alter results."""
    import os

    for key in list(os.environ):
        if key.startswith("QUAINEX_"):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Return isolated test settings with no AI credentials.

    ``_env_file=None`` prevents pydantic-settings from reading the real ``.env``.
    """
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        environment="dev",
        log_dir=tmp_path / "logs",
        anthropic_api_key=None,
    )


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    """Return an application built from the isolated test settings."""
    return create_app(settings)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    """Return a test client with the application lifespan running.

    ``raise_server_exceptions=False`` lets tests assert on the 500 response the
    catch-all handler produces, instead of the exception being re-raised.
    """
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
