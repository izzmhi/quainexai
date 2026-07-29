"""Structured logging setup.

Purpose:
    Configure one logging pipeline for the whole process: human-readable output
    on the console for development, and machine-parseable JSON on disk for the
    audit trail the security requirements call for.

Why structured (not plain text):
    Phase 6 exposes Quainex to a phone, and Phase 10 lets it act autonomously.
    Both demand an auditable record of *what ran, on whose behalf, and when*.
    Answering that from ``grep`` over prose logs is guesswork; answering it from
    JSON lines is a query. Retrofitting structure later means rewriting every
    call site, so it is established here in Phase 1.

Architecture:
    application code
        -> ``structlog`` logger (bound key/value context)
        -> shared processor chain (timestamp, level, logger name, redaction)
        -> ``ProcessorFormatter``
             |-- console handler  -> coloured key=value  (dev)
             +-- rotating file    -> one JSON object per line (always)

    Standard-library loggers (uvicorn, anthropic, asyncio) are routed through
    the same chain, so third-party output lands in the audit file too.

Dependencies:
    structlog, quainex.config.settings

Example:
    >>> from quainex.core.logging import configure_logging, get_logger
    >>> configure_logging(settings)
    >>> log = get_logger(__name__)
    >>> log.info("command_executed", command="open_app", target="VS Code")

Future improvements:
    * Ship the JSON stream to a real log sink once Quainex runs headless.
    * Split the security audit trail into its own file with stricter retention.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from structlog.typing import EventDict, Processor, WrappedLogger

    from quainex.config.settings import Settings

# Rotate at 10 MB, keep 5 generations (~50 MB ceiling for the audit trail).
_MAX_LOG_BYTES = 10 * 1024 * 1024
_LOG_BACKUP_COUNT = 5

# Substrings that mark a value as sensitive. Matched case-insensitively against
# the *key*, so `api_key`, `ANTHROPIC_API_KEY` and `authorization` all redact.
_SENSITIVE_KEY_MARKERS = ("key", "token", "secret", "password", "authorization", "credential")

_REDACTED = "***REDACTED***"

# Third-party loggers that are noisy at DEBUG and rarely useful.
_NOISY_LOGGERS = ("httpcore", "httpx", "asyncio", "watchfiles")


def _redact_sensitive(_logger: WrappedLogger, _name: str, event_dict: EventDict) -> EventDict:
    """Mask values whose key suggests they hold a credential.

    A defence in depth measure, not the primary one: secrets are held in
    ``SecretStr`` so they do not stringify by accident. This catches the case
    where a developer logs a raw value by mistake — the log file is the one
    artefact most likely to be copied into a bug report or a chat window.

    Args:
        _logger: The wrapped logger (unused).
        _name: The log method name (unused).
        event_dict: The event's key/value payload.

    Returns:
        The event payload with sensitive values replaced.
    """
    for key in event_dict:
        lowered = key.lower()
        if any(marker in lowered for marker in _SENSITIVE_KEY_MARKERS):
            event_dict[key] = _REDACTED
    return event_dict


def _shared_processors() -> list[Processor]:
    """Build the processor chain applied to every record, whatever its origin.

    Returns:
        Processors that enrich an event before it reaches a renderer.
    """
    return [
        # Context bound via structlog.contextvars (e.g. request correlation IDs)
        # must be merged first so later processors can see and redact it.
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        _redact_sensitive,
    ]


def configure_logging(settings: Settings) -> None:
    """Configure structlog and the standard-library root logger.

    Idempotent: existing root handlers are removed first, so calling this twice
    (as tests do) will not duplicate every log line.

    Args:
        settings: Validated application settings supplying level and log directory.
    """
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    shared = _shared_processors()

    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # JSON to disk — always, in every environment. This is the audit trail.
    file_formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
    )
    file_handler = logging.handlers.RotatingFileHandler(
        filename=settings.log_dir / "quainex.log",
        maxBytes=_MAX_LOG_BYTES,
        backupCount=_LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(file_formatter)

    # Console — readable in dev, JSON in production so container logs stay parseable.
    # ConsoleRenderer formats exceptions itself, so `format_exc_info` is omitted here.
    console_renderer: Processor = (
        structlog.processors.JSONRenderer()
        if settings.is_production
        else structlog.dev.ConsoleRenderer(colors=True)
    )
    console_formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            console_renderer,
        ],
    )
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(console_formatter)

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
        existing.close()
    root.addHandler(file_handler)
    root.addHandler(console_handler)
    root.setLevel(settings.log_level.value)

    for noisy in _NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Uvicorn installs its own handlers; clearing them and enabling propagation
    # routes access/error logs through our chain instead of printing separately.
    for uvicorn_logger in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        target = logging.getLogger(uvicorn_logger)
        target.handlers.clear()
        target.propagate = True


def get_logger(name: str | None = None, **initial_context: Any) -> structlog.stdlib.BoundLogger:
    """Return a logger, optionally pre-bound with context.

    Args:
        name: Logger name; pass ``__name__`` from the calling module.
        **initial_context: Key/value pairs attached to every record from this logger.

    Returns:
        A bound logger ready for structured calls such as
        ``log.info("event_name", key=value)``.
    """
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    if initial_context:
        return logger.bind(**initial_context)
    return logger
