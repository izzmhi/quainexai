"""Dependency injection composition root.

Purpose:
    Build every long-lived object Quainex needs, in one place, and hand them to
    consumers rather than letting consumers construct their own.

Why hand-rolled (not a DI framework):
    FastAPI already ships a dependency system (``Depends``). Layering a second
    container library on top would add a framework to learn and a lifecycle to
    reconcile, for no capability we lack. ``Container`` is a plain dataclass:
    it holds constructed collaborators, knows how to build them from settings,
    and knows how to shut them down. That is the whole job.

    The value is in the *discipline*, not the machinery — no module reaches for
    a global client or calls ``get_settings()`` deep in business logic. Anything
    a component needs arrives through its constructor or a route dependency, so
    every component can be tested with substitutes.

Architecture:
    startup (``api.app`` lifespan)
        -> ``Container.create(settings)``   builds logger + AI provider
        -> stored on ``app.state.container``
        -> routes receive it via ``Depends(get_container)``
    shutdown
        -> ``await container.aclose()``     releases provider connections

Dependencies:
    quainex.config.settings, quainex.core.logging, quainex.services.ai

Example:
    >>> container = Container.create()
    >>> container.ai_provider.name
    'anthropic'

Future improvements:
    * Add ``memory``, ``command_registry`` and ``scheduler`` fields as Phases 3,
      5 and 10 introduce them — the construction and teardown sites stay here.
    * Swap ``ai_provider`` construction on ``settings.ai_provider`` once a second
      provider exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from quainex.config.settings import AIProviderName, Settings, get_settings
from quainex.core.exceptions import ConfigurationError
from quainex.core.logging import configure_logging, get_logger
from quainex.services.ai.anthropic_provider import AnthropicProvider

if TYPE_CHECKING:
    import structlog

    from quainex.services.ai.provider import AIProvider


@dataclass(slots=True)
class Container:
    """Holds the application's long-lived collaborators.

    Attributes:
        settings: Validated configuration for this process.
        logger: Root application logger, pre-bound with app identity.
        ai_provider: The configured language model backend.
    """

    settings: Settings
    logger: structlog.stdlib.BoundLogger
    ai_provider: AIProvider

    @classmethod
    def create(cls, settings: Settings | None = None) -> Container:
        """Construct the container and everything in it.

        Logging is configured first so that any subsequent construction failure
        is itself recorded through the normal pipeline.

        Args:
            settings: Configuration to build from. Defaults to the cached
                process settings; tests pass an override.

        Returns:
            A fully constructed container.

        Raises:
            ConfigurationError: The configured AI provider has no implementation.
        """
        resolved = settings or get_settings()
        configure_logging(resolved)
        logger = get_logger("quainex", app=resolved.app_name, version=resolved.version)

        ai_provider = cls._build_ai_provider(resolved)

        logger.info(
            "container_initialised",
            environment=resolved.environment.value,
            ai_provider=ai_provider.name,
            ai_available=ai_provider.is_available,
        )
        return cls(settings=resolved, logger=logger, ai_provider=ai_provider)

    @staticmethod
    def _build_ai_provider(settings: Settings) -> AIProvider:
        """Select and construct the AI provider implementation.

        Args:
            settings: Configuration naming the desired provider.

        Returns:
            The constructed provider.

        Raises:
            ConfigurationError: The named provider has no implementation.
        """
        match settings.ai_provider:
            case AIProviderName.ANTHROPIC:
                return AnthropicProvider(settings)
            case _:  # pragma: no cover - unreachable while one provider exists
                raise ConfigurationError(
                    f"No implementation for AI provider '{settings.ai_provider}'"
                )

    async def aclose(self) -> None:
        """Release resources held by contained objects.

        Called from the application's shutdown hook. Safe to call more than once.
        """
        await self.ai_provider.aclose()
        self.logger.info("container_closed")
