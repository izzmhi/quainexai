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
from quainex.core.automation import WindowsDesktopController
from quainex.core.brain import Brain
from quainex.core.commands import CommandExecutor, build_executor
from quainex.core.exceptions import ConfigurationError
from quainex.core.logging import configure_logging, get_logger
from quainex.core.memory import MemoryManager, SqlAlchemyMemoryStore
from quainex.core.speech import WindowsSapiTTS
from quainex.core.voice import FasterWhisperSTT, MicrophoneRecorder, VoiceSession
from quainex.database.engine import Database
from quainex.services.ai.anthropic_provider import AnthropicProvider

if TYPE_CHECKING:
    import structlog

    from quainex.core.automation import DesktopController
    from quainex.services.ai.provider import AIProvider


@dataclass(slots=True)
class Container:
    """Holds the application's long-lived collaborators.

    Attributes:
        settings: Validated configuration for this process.
        logger: Root application logger, pre-bound with app identity.
        ai_provider: The configured language model backend.
        brain: Natural language to structured intent classifier.
        desktop: Platform controller performing OS-level actions.
        commands: Policy-enforcing dispatcher for classified intents.
        voice: Spoken conversation loop.
        database: Connection to the persistent store.
        memory: Short-term conversation context and long-term recall.
    """

    settings: Settings
    logger: structlog.stdlib.BoundLogger
    ai_provider: AIProvider
    brain: Brain
    desktop: DesktopController
    commands: CommandExecutor
    voice: VoiceSession
    database: Database
    memory: MemoryManager

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
        brain = Brain(provider=ai_provider, settings=resolved)
        desktop = WindowsDesktopController(resolved)
        commands = build_executor(desktop=desktop, settings=resolved)

        # The engine is created here but the schema is not: DDL is async, so it
        # happens in `start()` during the application lifespan.
        database = Database.create(resolved)
        memory = MemoryManager(SqlAlchemyMemoryStore(database), resolved)

        # Voice components are constructed unconditionally but load nothing:
        # the Whisper model is fetched on first use, so a missing or undownloaded
        # model surfaces as "unavailable" rather than as a failed startup.
        voice = VoiceSession(
            stt=FasterWhisperSTT(resolved),
            tts=WindowsSapiTTS(resolved),
            recorder=MicrophoneRecorder(resolved),
            brain=brain,
            commands=commands,
            settings=resolved,
            memory=memory,
        )

        logger.info(
            "container_initialised",
            environment=resolved.environment.value,
            ai_provider=ai_provider.name,
            ai_available=ai_provider.is_available,
            commands_registered=len(commands.catalogue),
            destructive_commands_enabled=resolved.allow_destructive_commands,
            voice_available=voice.is_available,
        )
        return cls(
            settings=resolved,
            logger=logger,
            ai_provider=ai_provider,
            brain=brain,
            desktop=desktop,
            commands=commands,
            voice=voice,
            database=database,
            memory=memory,
        )

    async def start(self) -> None:
        """Complete the asynchronous part of startup.

        Separate from ``create()`` because schema creation is async and a
        constructor cannot await. Called from the application lifespan.
        """
        await self.database.create_schema()

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
        await self.database.aclose()
        self.logger.info("container_closed")
