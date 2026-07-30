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

import asyncio
import secrets
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import SecretStr

from quainex.auth import TokenService
from quainex.config.settings import AIProviderName, Settings, get_settings
from quainex.core.agent import AutonomousAgent
from quainex.core.automation import WindowsDesktopController
from quainex.core.brain import Brain
from quainex.core.commands import CommandExecutor, build_executor
from quainex.core.conversation import Conversationalist
from quainex.core.devtools import CodeAssistant, DevRunner
from quainex.core.exceptions import ConfigurationError
from quainex.core.logging import configure_logging, get_logger
from quainex.core.memory import MemoryManager, SqlAlchemyMemoryStore
from quainex.core.speech import WindowsSapiTTS
from quainex.core.voice import FasterWhisperSTT, MicrophoneRecorder, VoiceSession
from quainex.database.engine import Database
from quainex.integrations import TelegramBridge
from quainex.plugins import PluginRegistry
from quainex.security import ConfirmationService, CredentialVault, RateLimiter
from quainex.services.ai.anthropic_provider import AnthropicProvider
from quainex.services.ai.fallback import FallbackProvider
from quainex.services.ai.gemini_provider import GeminiProvider
from quainex.services.ai.openai_compatible import OpenAICompatibleProvider
from quainex.vision import ScreenAnalyst

if TYPE_CHECKING:
    import structlog

    from quainex.core.automation import DesktopController
    from quainex.services.ai.provider import AIProvider


def _secret(value: SecretStr | None) -> str | None:
    """Unwrap an optional secret to plain text.

    Args:
        value: The configured credential, if any.

    Returns:
        The value, or ``None`` when unset or blank. Blank counts as unset: an
        empty ``QUAINEX_GROQ_API_KEY=`` in ``.env`` means "no key", not "a key
        that happens to be the empty string".
    """
    if value is None:
        return None
    return value.get_secret_value().strip() or None


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
        tokens: Issues and verifies access tokens. ``None`` when authentication
            is not configured, which is only permitted on loopback.
        confirmations: Issues and verifies command confirmation tokens.
        rate_limiter: Per-client request throttle.
        dev: Runs allowlisted development operations.
        code: AI-backed code explanation, review and generation.
        conversation: Replies for questions, greetings and unrecognised requests.
        vision: Screen and document understanding.
        plugins: Discovers and dispatches to installed plugins.
        agent: Plans and carries out goals within a budget.
        telegram: Phone bridge. Constructed always, polls only when started.
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
    tokens: TokenService | None
    confirmations: ConfirmationService
    rate_limiter: RateLimiter
    dev: DevRunner
    code: CodeAssistant
    conversation: Conversationalist
    vision: ScreenAnalyst
    plugins: PluginRegistry
    agent: AutonomousAgent
    telegram: TelegramBridge
    #: Settings before vault credentials were overlaid. Kept so a reload can
    #: re-derive from the original source rather than compounding overlays.
    _base_settings: Settings | None = None
    #: Handle on the polling task, so shutdown can cancel it.
    _telegram_task: asyncio.Task[None] | None = None

    @property
    def base_settings(self) -> Settings:
        """Settings as parsed from the environment, before vault credentials."""
        return self._base_settings if self._base_settings is not None else self.settings

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
        base = settings or get_settings()
        configure_logging(base)
        logger = get_logger("quainex", app=base.app_name, version=base.version)

        # Credentials come from two places: the .env file, and the encrypted vault
        # the dashboard writes to. Merging happens here, in the composition root,
        # because "where configuration comes from" is precisely this module's job —
        # no component downstream should know there is more than one source.
        vault = CredentialVault(base.credentials_path)
        resolved = cls._apply_vault(base, vault)

        ai_provider = cls._build_ai_provider(resolved)
        brain = Brain(provider=ai_provider, settings=resolved)
        desktop = WindowsDesktopController(resolved)

        # Confirmation tokens are signed with the auth secret when one is set,
        # and with an ephemeral per-process secret otherwise. The ephemeral case
        # is loopback-only, where the trust boundary is the machine itself; a
        # restart simply invalidates any token in flight, which is harmless given
        # their two-minute lifetime.
        confirmation_secret = (
            resolved.auth_secret.get_secret_value()
            if resolved.auth_secret
            else secrets.token_urlsafe(48)
        )
        confirmations = ConfirmationService(
            confirmation_secret, ttl_seconds=resolved.confirmation_ttl_seconds
        )
        dev = DevRunner(resolved)
        code = CodeAssistant(ai_provider, resolved)
        vision = ScreenAnalyst(ai_provider, desktop, resolved)
        # Memory is built before the executor because the conversational commands
        # need it: a reply with no access to recent turns cannot resolve a
        # follow-up like "and what about tomorrow?".
        #
        # The engine is created here but the schema is not: DDL is async, so it
        # happens in `start()` during the application lifespan.
        database = Database.create(resolved)
        memory = MemoryManager(SqlAlchemyMemoryStore(database), resolved)
        conversation = Conversationalist(ai_provider, resolved, memory)

        commands = build_executor(
            desktop=desktop,
            settings=resolved,
            confirmations=confirmations,
            dev=dev,
            code=code,
            vision=vision,
            conversation=conversation,
        )

        tokens = (
            TokenService(resolved.auth_secret.get_secret_value(), resolved.auth_token_ttl_minutes)
            if resolved.auth_secret
            else None
        )
        rate_limiter = RateLimiter(resolved.rate_limit_per_minute)

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

        plugins = PluginRegistry(
            resolved,
            commands=commands,
            provider=ai_provider,
            memory=memory,
            desktop=desktop,
        )
        agent = AutonomousAgent(
            provider=ai_provider,
            brain=brain,
            commands=commands,
            settings=resolved,
            memory=memory,
        )
        telegram = TelegramBridge(
            resolved, brain=brain, commands=commands, voice=voice, memory=memory
        )

        logger.info(
            "container_initialised",
            environment=resolved.environment.value,
            ai_provider=ai_provider.name,
            ai_available=ai_provider.is_available,
            commands_registered=len(commands.catalogue),
            destructive_commands_enabled=resolved.allow_destructive_commands,
            voice_available=voice.is_available,
            auth_required=resolved.auth_required,
            bound_to_loopback=resolved.is_loopback,
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
            tokens=tokens,
            confirmations=confirmations,
            rate_limiter=rate_limiter,
            dev=dev,
            code=code,
            conversation=conversation,
            vision=vision,
            plugins=plugins,
            agent=agent,
            telegram=telegram,
            _base_settings=base,
        )

    async def start(self) -> None:
        """Complete the asynchronous part of startup.

        Separate from ``create()`` because schema creation is async and a
        constructor cannot await. Called from the application lifespan.
        """
        await self.database.create_schema()

        if self.settings.telegram_autostart and self.telegram.is_configured:
            # Fire-and-forget: the bridge polls forever, so awaiting it here
            # would stop the application from finishing startup. The reference
            # is kept so shutdown can cancel it rather than leaking the task.
            self._telegram_task = asyncio.create_task(self.telegram.run())
            self.logger.info("telegram_autostarted")

    @staticmethod
    def _apply_vault(settings: Settings, vault: CredentialVault) -> Settings:
        """Overlay vault-stored credentials onto the settings.

        The vault wins over ``.env``. That direction is chosen on purpose: if
        ``.env`` took precedence, typing a key into the dashboard would appear to
        succeed and quietly do nothing, which is the worse of the two surprises.
        The settings API reports which source each key came from, so the
        precedence is visible rather than mysterious.

        Args:
            settings: Settings parsed from the environment and ``.env``.
            vault: The credential store to overlay.

        Returns:
            Either the original settings, or a copy carrying the stored secrets.
        """
        stored = vault.load()
        if not stored:
            return settings

        # `model_copy` rather than re-validating: these fields have no validators
        # of their own, and re-running `Settings(...)` here would re-execute the
        # startup invariants — including the one that refuses to boot exposed
        # without credentials — against a half-built object.
        #
        # Values are wrapped in `SecretStr` rather than passed as plain strings.
        # `model_copy` does not coerce, so an unwrapped string would sit in a field
        # typed `SecretStr | None` and break every reader of it, including the
        # `has_ai_credentials` computed field.
        return settings.model_copy(
            update={name: SecretStr(value) for name, value in stored.items()}
        )

    async def reload_ai_providers(self) -> None:
        """Rebuild the AI provider chain from the vault, without a restart.

        Called after the dashboard saves or clears a key. The chain object's
        identity is preserved — see ``FallbackProvider.replace`` — so the Brain,
        agent, plugins and vision analyst pick up the change without being
        reconstructed, and no in-flight request sees a half-swapped container.

        Raises:
            ConfigurationError: A configured provider name has no implementation.
        """
        # Re-derived from the *pre-vault* settings, not from `self.settings`.
        # Overlaying onto already-overlaid settings would make a deleted key
        # immortal: the value would survive in the merged copy with nothing left
        # in the vault to remove it.
        vault = CredentialVault(self.base_settings.credentials_path)
        refreshed = self._apply_vault(self.base_settings, vault)

        chain = self._build_ai_provider(refreshed)
        if isinstance(self.ai_provider, FallbackProvider) and isinstance(chain, FallbackProvider):
            await self.ai_provider.replace(chain.providers)
        else:  # pragma: no cover - both sides are built by _build_ai_provider
            self.ai_provider = chain

        self.logger.info("ai_providers_reloaded", chain=self.ai_provider.name)

    @staticmethod
    def _build_ai_provider(settings: Settings) -> AIProvider:
        """Build the provider chain, in the configured preference order.

        Every provider is constructed whether or not it holds credentials — an
        unconfigured one reports itself unavailable and is skipped at call time,
        so adding a key later starts working without touching this code.

        Args:
            settings: Configuration naming the provider order and credentials.

        Returns:
            A ``FallbackProvider`` over the configured chain.

        Raises:
            ConfigurationError: A provider name has no implementation.
        """
        chain: list[AIProvider] = []

        for choice in settings.ai_providers:
            match choice:
                case AIProviderName.ANTHROPIC:
                    chain.append(AnthropicProvider(settings))
                case AIProviderName.GEMINI:
                    chain.append(GeminiProvider(settings))
                case AIProviderName.GROQ:
                    key = _secret(settings.groq_api_key)
                    chain.append(
                        OpenAICompatibleProvider(
                            name="groq",
                            # Groq is a hosted service, so no key means nowhere to
                            # call. Withholding the URL is what makes the provider
                            # report itself unavailable instead of 401-ing on every
                            # request and dragging the chain down with it.
                            base_url=settings.groq_base_url if key else "",
                            model=settings.groq_model,
                            api_key=key,
                            # Free-tier cap: see `ai_max_tokens_free_tier`. Asking
                            # for the full budget here spends a whole minute's
                            # quota on one call.
                            max_tokens=settings.ai_max_tokens_free_tier,
                        )
                    )
                case AIProviderName.OPENROUTER:
                    key = _secret(settings.openrouter_api_key)
                    chain.append(
                        OpenAICompatibleProvider(
                            name="openrouter",
                            base_url=settings.openrouter_base_url if key else "",
                            model=settings.openrouter_model,
                            api_key=key,
                            max_tokens=settings.ai_max_tokens_free_tier,
                        )
                    )
                case AIProviderName.LOCAL:
                    chain.append(
                        OpenAICompatibleProvider(
                            name="local",
                            # A local server needs a URL and legitimately needs no
                            # key, so availability follows the URL here.
                            base_url=settings.local_base_url,
                            model=settings.local_model,
                            api_key=_secret(settings.local_api_key),
                            # Your own machine has no per-minute quota to fit
                            # inside, so the full budget applies here.
                            max_tokens=settings.ai_max_tokens,
                        )
                    )
                case _:  # pragma: no cover - the enum is exhaustive above
                    raise ConfigurationError(f"No implementation for AI provider '{choice}'")

        return FallbackProvider(chain)

    async def aclose(self) -> None:
        """Release resources held by contained objects.

        Called from the application's shutdown hook. Safe to call more than once.
        """
        self.telegram.stop()
        if self._telegram_task is not None:
            self._telegram_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._telegram_task
            self._telegram_task = None

        await self.ai_provider.aclose()
        await self.database.aclose()
        self.logger.info("container_closed")
