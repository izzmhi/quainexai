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
from quainex.core.browser import BrowserSession
from quainex.core.commands import CommandExecutor, build_executor
from quainex.core.conversation import Conversationalist
from quainex.core.devtools import CodeAssistant, DevRunner
from quainex.core.exceptions import ConfigurationError, FeatureNotConfiguredError
from quainex.core.logging import configure_logging, get_logger
from quainex.core.memory import MemoryManager, SqlAlchemyMemoryStore
from quainex.core.speech import WindowsSapiTTS
from quainex.core.voice import (
    FasterWhisperSTT,
    MicrophoneRecorder,
    VoiceSession,
    WakeWordListener,
)
from quainex.database.engine import Database
from quainex.integrations import TelegramBridge
from quainex.plugins import PluginRegistry
from quainex.security import ConfirmationService, CredentialVault, RateLimiter
from quainex.security.vault import STORABLE_SECRETS
from quainex.services.ai.anthropic_provider import AnthropicProvider
from quainex.services.ai.fallback import FallbackProvider
from quainex.services.ai.gemini_provider import GeminiProvider
from quainex.services.ai.openai_compatible import OpenAICompatibleProvider
from quainex.vision import ScreenAnalyst

if TYPE_CHECKING:
    import structlog

    from quainex.core.automation import DesktopController
    from quainex.services.ai.provider import AIProvider


#: Module logger, for the helpers below that run outside a Container instance.
_log = get_logger(__name__)


def _parse_user_ids(raw: str) -> list[int]:
    """Parse a stored Telegram allowlist into user ids.

    Tolerant of the separators people actually type — commas, spaces, newlines —
    because this is entered by hand in the dashboard. Anything that is not an
    integer is dropped rather than raising: a malformed entry must not stop the
    application from booting, and the dashboard reads the list back so a missing
    id is visible.

    Args:
        raw: The stored value, e.g. ``"123456789, 987654321"``.

    Returns:
        The ids, in order, without duplicates.
    """
    ids: list[int] = []
    for token in raw.replace(",", " ").replace("\n", " ").split():
        try:
            parsed = int(token)
        except ValueError:
            _log.warning("telegram_allowlist_entry_ignored", entry=token[:32])
            continue
        if parsed not in ids:
            ids.append(parsed)
    return ids


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
        browser: A steerable web browser, opened on first use.
        vision: Screen and document understanding.
        plugins: Discovers and dispatches to installed plugins.
        agent: Plans and carries out goals within a budget.
        telegram: Phone bridge. Constructed always, polls only when started.
        listener: Always-on wake-word listener. Constructed always, holds the
            microphone only when started.
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
    browser: BrowserSession
    vision: ScreenAnalyst
    plugins: PluginRegistry
    agent: AutonomousAgent
    telegram: TelegramBridge
    listener: WakeWordListener
    #: Settings before vault credentials were overlaid. Kept so a reload can
    #: re-derive from the original source rather than compounding overlays.
    _base_settings: Settings | None = None
    #: Handle on the polling task, so shutdown can cancel it.
    _telegram_task: asyncio.Task[None] | None = None
    #: Handle on the listening task, so shutdown releases the microphone.
    _listener_task: asyncio.Task[None] | None = None

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
        # Constructed but not launched: the browser opens on the first "browse"
        # command and is torn down on shutdown, so a session that never browses
        # pays nothing.
        browser = BrowserSession(resolved)

        commands = build_executor(
            desktop=desktop,
            settings=resolved,
            confirmations=confirmations,
            dev=dev,
            code=code,
            vision=vision,
            conversation=conversation,
            browser=browser,
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
        listener = WakeWordListener(voice=voice, settings=resolved)
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
            browser=browser,
            vision=vision,
            plugins=plugins,
            agent=agent,
            telegram=telegram,
            listener=listener,
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

        if self.settings.voice_always_listening and self.voice.is_available:
            # Same fire-and-forget shape, and the same reason: the listener runs
            # until stopped. Gated on `is_available` so a machine without the
            # voice extra installed starts normally instead of failing.
            self._listener_task = asyncio.create_task(self.listener.run())
            self.logger.warning(
                "wake_word_listener_autostarted",
                detail="The microphone is open. Set QUAINEX_VOICE_ALWAYS_LISTENING=false to stop.",
            )

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
        # Secrets are wrapped in `SecretStr` rather than passed as plain strings.
        # `model_copy` does not coerce, so an unwrapped string would sit in a field
        # typed `SecretStr | None` and break every reader of it, including the
        # `has_ai_credentials` computed field. The same absence of coercion is why
        # the Telegram allowlist is parsed to `list[int]` here rather than left as
        # the string the vault stores.
        update: dict[str, object] = {
            name: SecretStr(value) for name, value in stored.items() if name in STORABLE_SECRETS
        }
        if raw := stored.get("telegram_allowed_users"):
            update["telegram_allowed_users"] = _parse_user_ids(raw)

        return settings.model_copy(update=update)

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

    async def start_listener(self) -> None:
        """Begin always-on wake-word listening.

        Raises:
            SpeechUnavailableError: There is no microphone or no recogniser.
        """
        if self.listener.is_running:
            return
        # Armed synchronously so that a missing microphone raises *here*, to the
        # caller, rather than inside a background task where nobody sees it — and so
        # the status returned immediately afterwards is already accurate.
        self.listener.arm()
        # Started here rather than in the route for the same reason as Telegram: the
        # container owns the handle, so shutdown can cancel it. A task created in a
        # request handler is a task nobody can stop — and this one holds a
        # microphone.
        self._listener_task = asyncio.create_task(self.listener.run())

    async def stop_listener(self) -> None:
        """Stop listening and release the microphone."""
        self.listener.stop()
        if self._listener_task is not None:
            # Awaited rather than merely cancelled: the current cycle owns an open
            # audio stream, and abandoning the task would leave the device held.
            self._listener_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._listener_task
            self._listener_task = None

    async def reload_telegram(self) -> None:
        """Rebuild the Telegram bridge from the vault, without a restart.

        Called after the dashboard changes the bot token or the allowlist. Unlike
        the provider chain, the bridge cannot be swapped in place: it holds the
        settings it was built with, and it may be sitting in a long-poll. So it is
        stopped, replaced, and restarted only if it was running — which also means
        a user removed from the allowlist stops being obeyed immediately, rather
        than at the next process restart. For an authorisation list, "eventually"
        is not good enough.

        Raises:
            ConfigurationError: A configured provider name has no implementation.
        """
        vault = CredentialVault(self.base_settings.credentials_path)
        refreshed = self._apply_vault(self.base_settings, vault)

        was_running = self.telegram.is_running
        await self.stop_telegram()

        self.settings = refreshed
        self.telegram = TelegramBridge(
            refreshed,
            brain=self.brain,
            commands=self.commands,
            voice=self.voice,
            memory=self.memory,
        )

        if was_running and self.telegram.is_configured:
            self._telegram_task = asyncio.create_task(self.telegram.run())

        self.logger.info(
            "telegram_reloaded",
            configured=self.telegram.is_configured,
            running=self.telegram.is_running,
            allowed_users=len(refreshed.telegram_allowed_users),
        )

    async def start_telegram(self) -> None:
        """Begin polling, replacing any existing polling task.

        Raises:
            FeatureNotConfiguredError: The bridge is not configured.
        """
        if not self.telegram.is_configured:
            raise FeatureNotConfiguredError(
                "Telegram is not configured. It needs both a bot token from "
                "@BotFather and at least one allowed user id from @userinfobot — "
                "a bot with an empty allowlist would take orders from anyone who "
                "found it, so Quainex refuses to poll without one."
            )
        if self.telegram.is_running:
            return

        self._telegram_task = asyncio.create_task(self.telegram.run())

    async def stop_telegram(self) -> None:
        """Stop polling and await the task, so nothing is left half-cancelled."""
        self.telegram.stop()
        if self._telegram_task is not None:
            self._telegram_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._telegram_task
            self._telegram_task = None

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
        await self.stop_telegram()
        await self.stop_listener()
        await self.browser.close()
        await self.ai_provider.aclose()
        await self.database.aclose()
        self.logger.info("container_closed")
