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
        # Each test gets its own database file, so no test can see another's
        # memories and none of them touch the real store in the repo root.
        database_path=tmp_path / "test.db",
        # Critical isolation, not tidiness: the container overlays credentials
        # from the vault on top of settings. Left at its default this would read
        # the developer's real `~/.quainex/credentials.dat`, so a suite that is
        # supposed to run offline would quietly start making billed API calls —
        # and any test that writes a credential would overwrite a real key.
        credentials_path=tmp_path / "credentials.dat",
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
