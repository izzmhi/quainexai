"""Quainex entrypoint.

Purpose:
    Start the HTTP/WebSocket server using validated application settings.

Why it is this thin:
    Everything meaningful lives in importable modules. Keeping the entrypoint to
    a launcher means the application can be started by uvicorn directly, by a
    test harness, by a Windows service wrapper, or by a future desktop shell —
    without any of them re-implementing startup.

Usage:
    python main.py

    # or, with auto-reload during development:
    uvicorn quainex.api.app:create_app --factory --reload

Dependencies:
    uvicorn, quainex.config.settings
"""

from __future__ import annotations

import uvicorn

from quainex.config.settings import get_settings


def main() -> None:
    """Start the Quainex server."""
    settings = get_settings()

    # log_config=None hands logging to our own structlog pipeline; uvicorn's
    # default dictConfig would otherwise replace the handlers we installed.
    uvicorn.run(
        "quainex.api.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        reload=not settings.is_production,
        log_config=None,
    )


if __name__ == "__main__":
    main()
