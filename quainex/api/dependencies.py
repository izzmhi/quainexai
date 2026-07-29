"""FastAPI dependency providers.

Purpose:
    Bridge the DI container onto FastAPI's ``Depends`` system, so route handlers
    declare what they need instead of importing globals.

Why it matters:
    A handler that calls ``get_settings()`` or builds its own API client cannot
    be tested without the real thing. A handler that declares
    ``container: ContainerDep`` can be given a substitute in one line.

Architecture:
    lifespan startup -> app.state.container = Container.create()
    request          -> Depends(get_container) -> reads app.state.container
                     -> handler receives it as a typed argument

Dependencies:
    fastapi, quainex.core.container

Example:
    >>> @router.get("/example")
    ... async def example(container: ContainerDep) -> dict[str, str]:
    ...     return {"provider": container.ai_provider.name}

Future improvements:
    * Add ``CurrentUserDep`` in Phase 6 when JWT authentication lands.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from quainex.core.container import Container
from quainex.core.exceptions import ConfigurationError


def get_container(request: Request) -> Container:
    """Return the application's DI container.

    Args:
        request: The incoming request, carrying a reference to the app.

    Returns:
        The container built during application startup.

    Raises:
        ConfigurationError: The container is missing, meaning the app was
            constructed without running its lifespan.
    """
    container = getattr(request.app.state, "container", None)
    if not isinstance(container, Container):
        raise ConfigurationError(
            "Application container is not initialised; the app lifespan did not run."
        )
    return container


#: Annotated alias so handlers can write `container: ContainerDep`.
ContainerDep = Annotated[Container, Depends(get_container)]
