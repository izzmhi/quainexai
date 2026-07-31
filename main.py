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
from quainex.core.logging import get_logger
from quainex.core.single_instance import SingleInstanceLock

_log = get_logger(__name__)


def main() -> None:
    """Start the Quainex server, refusing to start a second one."""
    settings = get_settings()

    # One server per port. If another Quainex already holds the lock, a second one
    # would only 409 its Telegram bridge against the first forever, so it stops here
    # with a clear message rather than starting and misbehaving. The lock is held
    # for the whole run and released when this process exits (or the moment it dies).
    lock = SingleInstanceLock(settings.port)
    if not lock.acquire():
        _log.warning("startup_refused_already_running", port=settings.port)
        print(
            f"Quainex is already running on port {settings.port}. "
            "Not starting a second instance."
        )
        return

    try:
        # log_config=None hands logging to our own structlog pipeline; uvicorn's
        # default dictConfig would otherwise replace the handlers we installed.
        #
        # reload is an explicit opt-in (QUAINEX_RELOAD=true), not tied to the
        # environment. The always-on autostart must be a single process: uvicorn's
        # reloader runs a supervisor *and* a worker, and two of those pairs — one
        # from the autostart, one from a manual launch — is how two Telegram bridges
        # ended up fighting over one bot token. Off by default keeps it single.
        uvicorn.run(
            "quainex.api.app:create_app",
            factory=True,
            host=settings.host,
            port=settings.port,
            reload=settings.reload,
            log_config=None,
        )
    finally:
        lock.release()


if __name__ == "__main__":
    main()
