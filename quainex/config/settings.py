"""Application configuration.

Purpose:
    Single, validated source of truth for every tunable value in Quainex.
    Nothing else in the codebase reads ``os.environ`` directly.

Architecture:
    ``.env`` / process environment
        -> ``Settings`` (pydantic-settings: parse + validate + coerce types)
        -> ``get_settings()`` (cached; one instance per process)
        -> injected into consumers via ``quainex.core.container.Container``

Dependencies:
    pydantic, pydantic-settings

Example:
    >>> from quainex.config.settings import get_settings
    >>> settings = get_settings()
    >>> settings.port
    8000

Future improvements:
    * Layered profiles (``.env.dev`` / ``.env.prod``) once deployment targets diverge.
    * Move ``anthropic_api_key`` into an OS keyring (Phase 6) so it never touches disk.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from ipaddress import ip_address
from pathlib import Path
from typing import Any, Self

from pydantic import Field, SecretStr, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root: .../quainex/quainex/config/settings.py -> parents[2] is the repo root.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]


class Environment(StrEnum):
    """Deployment environment the process is running in."""

    DEV = "dev"
    PROD = "prod"


class LogLevel(StrEnum):
    """Minimum severity a log record must have to be emitted."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AIProviderName(StrEnum):
    """Identifier for a backing AI provider implementation."""

    #: Free tier, very fast. Structured output is prompt-guided, not enforced.
    GROQ = "groq"
    #: Free tier, native schema enforcement, and vision.
    GEMINI = "gemini"
    #: Aggregator with free models. One key reaches many vendors.
    OPENROUTER = "openrouter"
    #: Paid, strongest structured output and vision.
    ANTHROPIC = "anthropic"
    #: Any OpenAI-compatible server, including a local Ollama.
    LOCAL = "local"


class AIEffort(StrEnum):
    """How much reasoning effort the model should spend on a request.

    The primary cost and latency lever. Intent routing is a small, well-scoped
    task, so ``MEDIUM`` is the default: it keeps voice commands responsive
    without starving the model on genuinely ambiguous requests.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


class Settings(BaseSettings):
    """Validated application settings, sourced from the environment and ``.env``.

    Every field maps to an environment variable prefixed with ``QUAINEX_``
    (case-insensitive), so ``app_name`` reads from ``QUAINEX_APP_NAME``.
    """

    model_config = SettingsConfigDict(
        env_prefix="QUAINEX_",
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # Reject unknown QUAINEX_* variables: a typo in .env should fail loudly at
        # startup rather than silently fall back to a default.
        extra="forbid",
    )

    # --- Application -----------------------------------------------------
    app_name: str = "Quainex"
    version: str = "0.1.0"
    environment: Environment = Environment.DEV
    debug: bool = True

    # --- HTTP server -----------------------------------------------------
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)

    # --- Logging ---------------------------------------------------------
    log_level: LogLevel = LogLevel.INFO
    log_dir: Path = Path("logs")

    # --- Dashboard -------------------------------------------------------
    # Where the browser interface is served from. Relative paths resolve to the
    # repo root. Set to a non-existent path to run headless: the mount is skipped
    # rather than failing startup, so an API-only deployment needs no code change.
    dashboard_dir: Path = Path("dashboard")
    serve_dashboard: bool = True

    # --- AI provider -----------------------------------------------------
    # Providers are tried in this order, and the first one *configured* wins.
    # Free tiers lead deliberately: Quainex should be fully usable before anyone
    # has paid for anything, with the paid provider as the backstop rather than
    # the entry fee. Reorder freely — an entry with no key is skipped.
    ai_providers: list[AIProviderName] = Field(
        default_factory=lambda: [
            AIProviderName.GROQ,
            AIProviderName.GEMINI,
            AIProviderName.OPENROUTER,
            AIProviderName.ANTHROPIC,
            AIProviderName.LOCAL,
        ]
    )

    # --- Groq (free tier) -------------------------------------------------
    groq_api_key: SecretStr | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # --- Google Gemini (free tier) ---------------------------------------
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-2.0-flash"

    # --- OpenRouter (free tier) ------------------------------------------
    # One key, many vendors. First-class rather than folded into `local` below:
    # its base URL is fixed and known, so requiring the user to also supply a URL
    # would make a pasted key silently do nothing.
    openrouter_api_key: SecretStr | None = None
    # Verified working, including structured output, rather than chosen from a
    # blog post: the previous default (`meta-llama/llama-3.3-70b-instruct:free`)
    # had quietly lost its free tier and returned 404 with "the paid version is
    # available now", which is why this one was checked against the live API
    # before being set.
    #
    # Free slugs come and go. `openrouter/free` auto-routes among whatever is free
    # and so cannot go stale, but it gives no guarantee about which model answers —
    # and the Brain needs dependable JSON. Browse the current list at
    # https://openrouter.ai/models?max_price=0
    openrouter_model: str = "nvidia/nemotron-3-super-120b-a12b:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # --- Local / self-hosted (Ollama, LM Studio) -------------------------
    # Offline mode: no key needed, nothing leaves the machine. Needs a URL — this
    # slot points at a server only you know about, so there is no default.
    local_base_url: str = ""
    local_model: str = "llama3.1"
    local_api_key: SecretStr | None = None

    # Where keys saved from the dashboard are stored, encrypted. Kept under the
    # user's profile rather than the repo so it can never be committed, and so
    # Windows DPAPI's per-user scope lines up with the file's location.
    credentials_path: Path = Field(
        default_factory=lambda: Path.home() / ".quainex" / "credentials.dat"
    )

    # --- Anthropic (paid) -------------------------------------------------
    anthropic_api_key: SecretStr | None = None
    ai_model: str = "claude-opus-5"
    ai_effort: AIEffort = AIEffort.MEDIUM
    # Note: on current Claude models this budget covers reasoning *and* the visible
    # answer, so it is set well above what the answer alone needs. Too low and
    # responses truncate mid-thought.
    ai_max_tokens: int = Field(default=8192, ge=1)

    # Cap used for hosted free tiers instead of the value above.
    #
    # This is not tidiness, it is the difference between working and not. Free
    # tiers meter *requested* tokens against a per-minute budget, and `max_tokens`
    # is a request whether or not the model uses it. Groq's free tier allows 12000
    # tokens per minute, so asking for 8192 on every call means roughly one
    # request per minute before a 429 — which looks like an outage, not a quota.
    #
    # These models also have no hidden reasoning to fund: the budget covers only
    # the visible answer, so a smaller number costs nothing in quality. The reason
    # for the large default above does not apply to them.
    ai_max_tokens_free_tier: int = Field(default=1536, ge=1)

    # Cap for intent classification specifically.
    #
    # The output is a five-field JSON object, so this is what the task actually
    # needs — and because free tiers meter *requested* tokens, the unused
    # remainder of a larger cap is charged against the quota regardless. Sizing
    # this to the job rather than to prose is the cheapest available multiplier on
    # how many commands a day fit inside a free tier.
    #
    # Not smaller: a reasoning model funds its thinking from the same budget, and
    # starving it produces an empty reply rather than a terse one.
    ai_max_tokens_classification: int = Field(default=768, ge=1)

    # --- Brain -----------------------------------------------------------
    # Classifications below this confidence still return their best guess, but
    # are flagged as needing user confirmation before Phase 3 acts on them.
    brain_confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)

    # --- Commands (Phase 3) ----------------------------------------------
    # Master switch for irreversible power actions (shutdown, restart, sleep).
    # Defaults to OFF: an AI operating system that can power off the machine on a
    # misheard word is worse than one that has to be told it may.
    allow_destructive_commands: bool = False

    # Grace period before a scheduled shutdown or restart fires, leaving a window
    # in which `shutdown /a` aborts it.
    shutdown_delay_seconds: int = Field(default=15, ge=0, le=600)

    # Directories file search and folder-opening are confined to. Anything
    # outside these is refused, so a misresolved path cannot reach system files.
    command_search_roots: list[Path] = Field(default_factory=lambda: [Path.home()])
    command_search_max_results: int = Field(default=50, ge=1, le=500)

    # Where screenshots are written.
    screenshot_dir: Path = Field(default_factory=lambda: Path.home() / "Pictures" / "Quainex")

    # --- Remote access and auth (Phase 6) ---------------------------------
    # Signing secret for access tokens. Required whenever authentication is
    # required. Generate with:
    #   python -c "import secrets; print(secrets.token_urlsafe(48))"
    auth_secret: SecretStr | None = None
    # scrypt hash of the remote-access password. Generate with:
    #   python scripts/hash_password.py
    auth_password_hash: str | None = None
    auth_token_ttl_minutes: int = Field(default=60, ge=1, le=10080)

    # Force authentication on even when bound to loopback. Leave unset to let it
    # be derived from the bind address — see `auth_required` below.
    require_auth: bool | None = None

    # Requests per minute per client before throttling. Applies only when
    # authentication is required; localhost is not rate limited.
    rate_limit_per_minute: int = Field(default=120, ge=1, le=10000)

    # How long a confirmation token stays redeemable.
    confirmation_ttl_seconds: int = Field(default=120, ge=10, le=3600)

    # --- Memory (Phase 5) -------------------------------------------------
    # Where the SQLite file lives. Relative paths resolve to the repo root.
    database_path: Path = Path("quainex.db")
    # Override to point at PostgreSQL later, e.g.
    # postgresql+asyncpg://user:pass@host/quainex
    database_url_override: str | None = None

    # How many recent turns the Brain receives as conversation context. Enough to
    # resolve "close it" after "open Spotify"; bounded so token cost stays flat
    # however long a session runs.
    memory_context_turns: int = Field(default=6, ge=0, le=50)

    # --- Telegram bridge --------------------------------------------------
    # Phone control without exposing this machine: the bridge polls outward, so
    # no port is forwarded and no certificate is needed.
    # Get a token from @BotFather; get your user id from @userinfobot.
    telegram_bot_token: SecretStr | None = None
    # Only these Telegram user ids are obeyed. Empty means the bridge stays off:
    # a bot with no allowlist would take orders from anyone who found it.
    telegram_allowed_users: list[int] = Field(default_factory=list)
    # Start polling automatically when the application starts.
    telegram_autostart: bool = False

    # --- Plugins (Phase 9) ------------------------------------------------
    # Where plugins are looked for. Relative paths resolve to the repo root.
    plugin_dir: Path = Path("plugins_installed")
    # Plugins are discovered but never loaded automatically: their code first
    # runs when you enable them, after seeing what permissions they asked for.
    plugin_autoload: list[str] = Field(default_factory=list)

    # --- Autonomous agent (Phase 10) --------------------------------------
    # Ceilings for one unattended run. Deliberately modest: an agent that can
    # run 200 steps unattended is one bug away from doing so.
    agent_max_steps: int = Field(default=12, ge=1, le=100)
    agent_max_seconds: float = Field(default=300.0, gt=0, le=3600)
    agent_max_actions: int = Field(default=20, ge=1, le=200)
    # How often one action may repeat before the run is treated as stuck.
    agent_max_repeats: int = Field(default=3, ge=1, le=20)

    # --- Voice (Phase 4) --------------------------------------------------
    wake_word: str = "quainex"
    # How close a transcribed word must be to the wake word to count. Speech
    # recognition mangles unusual names ("Quinex", "Kwainex", "Quain X"), so
    # exact matching would make the assistant deaf to its own name.
    #
    # Kept at 0.75 deliberately. Lowering it to 0.70 to admit "kwainex" (0.714)
    # also admits "equinox", which scores exactly the same — edit distance cannot
    # tell a real mishearing from an unrelated word. Homophones are handled by
    # phonetic folding in `detect_wake_word` instead, which is targeted where a
    # looser threshold is indiscriminate.
    wake_word_similarity: float = Field(default=0.75, ge=0.0, le=1.0)
    voice_require_wake_word: bool = True

    # Hold the microphone open continuously and act on anything beginning with the
    # wake word.
    #
    # **Off by default, and that is a decision rather than caution.** This opens the
    # microphone indefinitely. Speech is transcribed locally and discarded unless it
    # was addressed to Quainex — nothing is uploaded, nothing is stored — but "the
    # mic is open" is a fact about the room, and other people in it have not agreed
    # to anything. An assistant that started listening because a default said
    # `true` would be a different kind of product.
    #
    # Costs no API tokens while idle: the wake gate returns before the Brain is
    # called, so ambient conversation is heard, discarded, and never leaves the
    # machine.
    voice_always_listening: bool = False

    # Log what was heard but ignored, for diagnosing a wake word that never fires.
    #
    # Off by default and worth being blunt about: turning this on writes overheard
    # speech into the log file. That is the opposite of what the listener normally
    # does with unaddressed audio, which is discard it. It exists because the
    # alternative — "it hears something, ignores it, and will not tell you what" —
    # makes a misheard wake word impossible to diagnose.
    #
    # Turn it on to find out why, then turn it off.
    voice_log_ignored_speech: bool = False

    # Recording bounds. `max_seconds` caps a microphone stuck open; recording
    # normally ends earlier, once speech is followed by `silence_seconds` of quiet.
    voice_max_seconds: float = Field(default=15.0, gt=0)
    voice_silence_seconds: float = Field(default=1.2, gt=0)
    # RMS amplitude (16-bit scale) below which a block counts as silence.
    voice_silence_threshold: float = Field(default=350.0, ge=0)

    # Recordings are deleted after transcription by default: audio of a room is
    # not something to accumulate on disk without being asked.
    keep_recordings: bool = False

    # Whisper model size: tiny | base | small | medium | large-v3. Larger is more
    # accurate, slower, and a bigger first-run download.
    whisper_model: str = "base"
    whisper_compute_type: str = "int8"
    whisper_beam_size: int = Field(default=1, ge=1, le=10)

    # --- Speech output (Phase 4) -----------------------------------------
    tts_enabled: bool = True
    tts_rate: int = Field(default=0, ge=-10, le=10)
    tts_voice: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _explain_renamed_settings(cls, values: Any) -> Any:
        """Turn a stale variable name into an instruction instead of a puzzle.

        ``extra="forbid"`` is what makes a typo in ``.env`` fail loudly rather
        than silently reverting to a default, and that is worth keeping. Its cost
        is that a *renamed* setting produces "Extra inputs are not permitted",
        which tells the reader nothing about what to write instead.

        Handled here rather than by keeping a deprecated alias: an alias means the
        old name goes on working, and two spellings for one concept is how
        configuration files rot.

        Args:
            values: Raw input, before field validation.

        Returns:
            The input unchanged.

        Raises:
            ValueError: A renamed setting is still present.
        """
        if not isinstance(values, dict):  # pragma: no cover - always a mapping here
            return values

        renamed = {
            # Singular -> plural: Quainex tries a chain of providers now rather
            # than exactly one.
            "ai_provider": (
                "QUAINEX_AI_PROVIDERS",
                'a JSON list, e.g. ["groq", "gemini", "anthropic", "local"]',
            ),
        }
        for old, (new, shape) in renamed.items():
            if old in values:
                raise ValueError(
                    f"QUAINEX_{old.upper()} has been renamed to {new}, which takes "
                    f"{shape}. Providers are now tried in order and the first one "
                    f"holding a key answers. Remove the old line from your .env."
                )
        return values

    @model_validator(mode="after")
    def _enforce_production_invariants(self) -> Self:
        """Force debug off in production.

        Debug surfaces internal state (tracebacks, config) that must never reach a
        remote caller. Rather than trusting operators to set both variables
        correctly, production silently wins over an errant ``QUAINEX_DEBUG=true``.
        """
        if self.environment is Environment.PROD and self.debug:
            object.__setattr__(self, "debug", False)
        return self

    @model_validator(mode="after")
    def _resolve_database_path(self) -> Self:
        """Anchor a relative database path to the repo root.

        Same reasoning as the log directory: without this, the store would move
        with the working directory and Quainex would silently start with an empty
        memory depending on where it was launched from.
        """
        if not self.database_path.is_absolute():
            object.__setattr__(self, "database_path", REPO_ROOT / self.database_path)
        if not self.plugin_dir.is_absolute():
            object.__setattr__(self, "plugin_dir", REPO_ROOT / self.plugin_dir)
        if not self.dashboard_dir.is_absolute():
            object.__setattr__(self, "dashboard_dir", REPO_ROOT / self.dashboard_dir)
        return self

    @property
    def database_url(self) -> str:
        """The SQLAlchemy URL for the configured store.

        Returns:
            The override when set, otherwise an async SQLite URL for
            ``database_path``.
        """
        if self.database_url_override:
            return self.database_url_override
        return f"sqlite+aiosqlite:///{self.database_path.as_posix()}"

    @model_validator(mode="after")
    def _resolve_log_dir(self) -> Self:
        """Anchor a relative ``log_dir`` to the repo root.

        Without this, log destinations would move with the working directory,
        scattering audit trails depending on where the process was launched.
        """
        if not self.log_dir.is_absolute():
            object.__setattr__(self, "log_dir", REPO_ROOT / self.log_dir)
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_production(self) -> bool:
        """Whether the process is running in the production environment."""
        return self.environment is Environment.PROD

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_loopback(self) -> bool:
        """Whether the server is bound to this machine only.

        Returns:
            ``True`` when the bind address is a loopback address.
        """
        try:
            return ip_address(self.host).is_loopback
        except ValueError:
            # Not an IP literal — "localhost" is loopback, a hostname is not.
            return self.host.strip().lower() in {"localhost", "localhost.localdomain"}

    @computed_field  # type: ignore[prop-decorator]
    @property
    def auth_required(self) -> bool:
        """Whether callers must authenticate.

        **Derived from the bind address rather than configured**, unless
        explicitly overridden. This is the point: a separate `enable_auth` flag
        makes "exposed to the network with authentication off" a reachable
        configuration, and that configuration is exactly the disaster. Here it is
        unreachable — binding to anything other than loopback turns authentication
        on, and startup refuses to proceed without credentials.

        Returns:
            Whether authentication is enforced.
        """
        if self.require_auth is not None:
            return self.require_auth
        return not self.is_loopback

    @model_validator(mode="after")
    def _require_credentials_when_exposed(self) -> Self:
        """Refuse to start exposed without credentials configured.

        Failing at startup is the whole point. The alternative — booting anyway
        and logging a warning — means the window between "listening on the
        network" and "someone notices the warning" is a window with no front
        door on the machine.

        Raises:
            ValueError: Authentication is required but not configured.
        """
        if not self.auth_required:
            return self

        missing = [
            name
            for name, value in (
                ("QUAINEX_AUTH_SECRET", self.auth_secret),
                ("QUAINEX_AUTH_PASSWORD_HASH", self.auth_password_hash),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                f"Authentication is required (host={self.host!r} is not loopback, or "
                f"QUAINEX_REQUIRE_AUTH is set), but {' and '.join(missing)} "
                f"{'is' if len(missing) == 1 else 'are'} not configured. "
                "Run `python scripts/hash_password.py` to set this up, or bind to "
                "127.0.0.1 to run locally without authentication."
            )
        return self

    @property
    def resolved_search_roots(self) -> tuple[Path, ...]:
        """Return the permitted command roots, canonicalised.

        Resolved here rather than at each use site so that containment checks
        always compare canonical paths — comparing a raw path against a resolved
        root is the classic way a ``..`` traversal slips through.

        Returns:
            Absolute, symlink-free root directories.
        """
        return tuple(root.expanduser().resolve() for root in self.command_search_roots)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_ai_credentials(self) -> bool:
        """Whether any AI provider is configured.

        Quainex boots without one; AI-backed features degrade rather than crash.
        A local endpoint counts, since it needs a URL rather than a key.
        """
        keys = (
            self.groq_api_key,
            self.gemini_api_key,
            self.openrouter_api_key,
            self.anthropic_api_key,
        )
        if any(key is not None and key.get_secret_value().strip() for key in keys):
            return True
        return bool(self.local_base_url.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide ``Settings`` instance.

    Cached so that ``.env`` is read once and every caller observes identical
    configuration. Tests clear the cache via ``get_settings.cache_clear()``.

    Returns:
        The validated settings for this process.
    """
    return Settings()
