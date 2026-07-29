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
from pathlib import Path
from typing import Self

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
    """Identifier for the backing AI provider implementation."""

    ANTHROPIC = "anthropic"


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

    # --- AI provider -----------------------------------------------------
    ai_provider: AIProviderName = AIProviderName.ANTHROPIC
    anthropic_api_key: SecretStr | None = None
    ai_model: str = "claude-opus-5"
    ai_effort: AIEffort = AIEffort.MEDIUM
    # Note: on current models this budget covers reasoning *and* the visible
    # answer, so it is set well above what the answer alone needs. Too low and
    # responses truncate mid-thought.
    ai_max_tokens: int = Field(default=8192, ge=1)

    # --- Brain -----------------------------------------------------------
    # Classifications below this confidence still return their best guess, but
    # are flagged as needing user confirmation before Phase 3 acts on them.
    brain_confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)

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
    def has_ai_credentials(self) -> bool:
        """Whether an API key is configured for the selected AI provider.

        Quainex boots without one; AI-backed features degrade rather than crash.
        """
        return self.anthropic_api_key is not None and bool(
            self.anthropic_api_key.get_secret_value().strip()
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide ``Settings`` instance.

    Cached so that ``.env`` is read once and every caller observes identical
    configuration. Tests clear the cache via ``get_settings.cache_clear()``.

    Returns:
        The validated settings for this process.
    """
    return Settings()
