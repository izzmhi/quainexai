"""FastAPI application factory.

Purpose:
    Assemble the HTTP/WebSocket application: lifecycle, middleware, error
    handling and routes.

Why a factory (not a module-level ``app``):
    A module-level application is constructed on import, which means importing
    anything from this package configures logging and builds an AI client as a
    side effect. Tests would inherit real configuration, and building a second
    instance with different settings would be impossible. ``create_app()`` makes
    construction explicit and parameterisable.

Architecture:
    create_app(settings)
        |-- lifespan: build Container on startup, close it on shutdown
        |-- middleware: CorrelationIdMiddleware
        |-- error handlers: QuainexError / HTTPException / catch-all
        +-- routers: health (HTTP), ws (WebSocket)

Dependencies:
    fastapi, quainex.api.*, quainex.config.settings, quainex.core.container

Example:
    >>> from quainex.api.app import create_app
    >>> app = create_app()

Future improvements:
    * Mount ``/api/v1`` prefix and version the surface before external clients
      exist (Phase 6).
    * Add CORS with an explicit allow-list when the dashboard lands.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI

from quainex.api.errors import install_error_handlers
from quainex.api.middleware import CorrelationIdMiddleware
from quainex.api.routes import brain, commands, health, ws
from quainex.config.settings import Settings, get_settings
from quainex.core.container import Container

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_DESCRIPTION = """
Quainex — Your Personal AI Operating System.

Phase 1 exposes the foundation only: a health endpoint and a WebSocket channel.
Command execution, voice and memory arrive in later phases.
""".strip()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and configure the Quainex application.

    Args:
        settings: Configuration to run with. Defaults to the cached process
            settings; tests pass an override to avoid touching the real ``.env``.

    Returns:
        A configured FastAPI application.
    """
    resolved = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Build the container on startup and release it on shutdown.

        Args:
            app: The application being started.

        Yields:
            Control back to the server for the lifetime of the application.
        """
        container = Container.create(resolved)
        app.state.container = container
        app.state.started_at = time.monotonic()

        # Named "configured_*" deliberately: a uvicorn CLI invocation can bind a
        # different host/port than settings specify, and an audit log that claims
        # the wrong port is worse than one that admits what it actually knows.
        container.logger.info(
            "application_started",
            configured_host=resolved.host,
            configured_port=resolved.port,
            docs_enabled=not resolved.is_production,
        )
        try:
            yield
        finally:
            await container.aclose()
            app.state.container = None

    app = FastAPI(
        title=resolved.app_name,
        description=_DESCRIPTION,
        version=resolved.version,
        lifespan=lifespan,
        # Interactive docs describe the whole attack surface; keep them out of
        # production builds.
        docs_url=None if resolved.is_production else "/docs",
        redoc_url=None,
        openapi_url=None if resolved.is_production else "/openapi.json",
    )

    app.add_middleware(CorrelationIdMiddleware)
    install_error_handlers(app)

    app.include_router(health.router)
    app.include_router(brain.router)
    app.include_router(commands.router)
    app.include_router(ws.router)

    return app
