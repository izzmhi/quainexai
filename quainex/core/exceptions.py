"""Quainex exception hierarchy.

Purpose:
    Give every deliberate failure in Quainex a common base class, so the API
    layer can distinguish "something we anticipated" from "an unexpected bug"
    and respond appropriately to each.

Architecture:
    ``QuainexError`` carries the two things an HTTP handler needs — a machine
    readable ``error_code`` and an ``http_status`` — so ``quainex.api.errors``
    can translate any subclass without a mapping table that drifts out of date.

    Exception raised
        -> caught by ``quainex.api.errors.quainex_error_handler``
        -> logged with full context + correlation ID
        -> serialised to a safe JSON envelope

Dependencies:
    Standard library only. This module must stay dependency-free so that any
    package can import it without creating a cycle.

Example:
    >>> raise ProviderNotConfiguredError("anthropic")
    Traceback (most recent call last):
    quainex.core.exceptions.ProviderNotConfiguredError: AI provider 'anthropic' is not configured

Future improvements:
    * Add ``PermissionDeniedError`` when Phase 3 introduces gated commands.
    * Add ``PluginError`` when Phase 9 introduces third-party plugin loading.
"""

from __future__ import annotations

from http import HTTPStatus


class QuainexError(Exception):
    """Base class for every error Quainex raises deliberately.

    Attributes:
        message: Human-readable description, safe to show to a local operator.
        error_code: Stable machine-readable slug for clients to branch on.
        http_status: HTTP status the API layer should return for this error.
    """

    error_code: str = "quainex_error"
    http_status: int = HTTPStatus.INTERNAL_SERVER_ERROR

    def __init__(self, message: str) -> None:
        """Initialise the error.

        Args:
            message: Human-readable description of what went wrong.
        """
        super().__init__(message)
        self.message = message


class ConfigurationError(QuainexError):
    """Configuration is missing, malformed, or internally inconsistent.

    Raised at startup rather than at request time wherever possible — a broken
    configuration should stop the process, not fail one request in a hundred.
    """

    error_code = "configuration_error"
    http_status = HTTPStatus.INTERNAL_SERVER_ERROR


class InvalidUtteranceError(QuainexError):
    """The user's input cannot be interpreted as a request.

    A client-side mistake (empty or oversized input), so it is rejected before a
    request is sent upstream — there is no point paying for a model call to
    classify whitespace.
    """

    error_code = "invalid_utterance"
    http_status = HTTPStatus.BAD_REQUEST


class CommandExecutionError(QuainexError):
    """A command was permitted to run but failed while running.

    Distinct from being refused: the request was legitimate, the action was
    attempted, and the operating system did not cooperate.
    """

    error_code = "command_failed"
    http_status = HTTPStatus.INTERNAL_SERVER_ERROR


class CommandNotAllowedError(QuainexError):
    """A command was refused before any side effect occurred.

    Raised when the target is not allowlisted, escapes its permitted roots, or
    the action is disabled by configuration. The distinction from
    ``CommandExecutionError`` matters for auditing: nothing happened here.
    """

    error_code = "command_not_allowed"
    http_status = HTTPStatus.FORBIDDEN


class SpeechError(QuainexError):
    """Speech recognition or synthesis failed."""

    error_code = "speech_error"
    http_status = HTTPStatus.INTERNAL_SERVER_ERROR


class SpeechUnavailableError(SpeechError):
    """The speech subsystem is not installed or not ready.

    Separate from ``SpeechError`` because the fix is an install or a download,
    not a retry — and because Quainex is designed to run without it. Voice is an
    optional capability layered on a system that works fine by text.
    """

    error_code = "speech_unavailable"
    http_status = HTTPStatus.SERVICE_UNAVAILABLE


class ProviderError(QuainexError):
    """An upstream provider call failed.

    Wraps vendor SDK exceptions so that callers depend on Quainex's error
    surface rather than on Anthropic's (or any future provider's) exception
    types. Swapping providers must not ripple into calling code.
    """

    error_code = "provider_error"
    http_status = HTTPStatus.BAD_GATEWAY


class ProviderNotConfiguredError(ProviderError):
    """A provider was used without the credentials it requires.

    Distinct from ``ProviderError`` because it is an operator problem with a
    clear fix (set the API key), not a transient upstream fault worth retrying.
    """

    error_code = "provider_not_configured"
    http_status = HTTPStatus.SERVICE_UNAVAILABLE

    def __init__(self, provider: str) -> None:
        """Initialise the error.

        Args:
            provider: Name of the provider that lacks credentials.
        """
        super().__init__(f"AI provider '{provider}' is not configured")
        self.provider = provider
