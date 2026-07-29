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

from fastapi import Depends, FastAPI

from quainex.api.errors import install_error_handlers
from quainex.api.middleware import CorrelationIdMiddleware
from quainex.api.routes import (
    agent,
    auth,
    brain,
    commands,
    health,
    memory,
    voice,
    ws,
)
from quainex.auth import require_auth
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
        await container.start()
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

    # /health and /auth are reachable without a token — a supervisor must be able
    # to check liveness, and the login route is how you obtain a token in the
    # first place. Everything else is guarded by attaching the dependency to the
    # router, so a route added later inherits the protection rather than relying
    # on someone remembering to list it.
    app.include_router(health.router)
    app.include_router(auth.router)

    protected = Depends(require_auth)
    app.include_router(brain.router, dependencies=[protected])
    app.include_router(commands.router, dependencies=[protected])
    app.include_router(voice.router, dependencies=[protected])
    app.include_router(memory.router, dependencies=[protected])
    app.include_router(agent.agent_router, dependencies=[protected])
    app.include_router(agent.plugin_router, dependencies=[protected])
    app.include_router(agent.telegram_router, dependencies=[protected])

    # The WebSocket authenticates in its own handshake: HTTP dependencies do not
    # apply to a socket upgrade, and browsers cannot set headers on one.
    app.include_router(ws.router)

    return app
