"""API error handling.

Purpose:
    Turn every failure — anticipated or not — into one consistent JSON envelope,
    while making sure internal detail reaches the log and not the caller.

Security rationale:
    A traceback is a map of the system: file paths, package versions, sometimes
    argument values. Phase 6 exposes this API beyond localhost, so the split is
    established now — the *log* gets everything, the *response* gets a stable
    error code, a safe message, and a correlation ID linking the two.

Architecture:
    raised exception
        |-- QuainexError    -> known failure  -> its own error_code + status
        |-- HTTPException   -> framework 4xx  -> normalised into the envelope
        +-- anything else   -> unexpected     -> logged with traceback, 500 out

    All three produce:
        {"error": {"code": ..., "message": ..., "correlation_id": ...}}

Dependencies:
    fastapi, starlette, quainex.core

Future improvements:
    * Add ``Retry-After`` on rate-limited provider errors once Phase 9 adds quotas.
    * Emit a security-audit record for authorisation failures in Phase 6.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from quainex.api.middleware import new_correlation_id
from quainex.core.exceptions import QuainexError
from quainex.core.logging import get_logger

if TYPE_CHECKING:
    from quainex.config.settings import Settings

_log = get_logger(__name__)

_GENERIC_MESSAGE = "An internal error occurred. Check the server logs for details."


def _correlation_id(request: Request) -> str:
    """Return the request's correlation ID, generating one if absent.

    A handler can run before the middleware has attached an ID (for example on a
    malformed request), so this never assumes the attribute exists.

    Args:
        request: The request being handled.

    Returns:
        The correlation ID to report.
    """
    existing = getattr(request.state, "correlation_id", None)
    return existing if isinstance(existing, str) else new_correlation_id()


def _envelope(code: str, message: str, correlation_id: str) -> dict[str, Any]:
    """Build the standard error response body.

    Args:
        code: Stable machine-readable error slug.
        message: Human-readable, caller-safe description.
        correlation_id: ID linking this response to the server logs.

    Returns:
        The response body as a JSON-serialisable dict.
    """
    return {
        "error": {
            "code": code,
            "message": message,
            "correlation_id": correlation_id,
        }
    }


def _settings_of(request: Request) -> Settings | None:
    """Read settings off the app container, if it has been built.

    Args:
        request: The request being handled.

    Returns:
        The active settings, or ``None`` if the container is unavailable.
    """
    container = getattr(request.app.state, "container", None)
    return getattr(container, "settings", None) if container is not None else None


async def quainex_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle deliberate Quainex failures.

    These are anticipated conditions with a known cause, so the message is
    considered safe to return as written.

    Args:
        request: The request being handled.
        exc: The raised ``QuainexError``.

    Returns:
        A JSON response carrying the error envelope.
    """
    error = exc if isinstance(exc, QuainexError) else QuainexError(str(exc))
    correlation_id = _correlation_id(request)

    _log.warning(
        "request_failed",
        error_code=error.error_code,
        error_message=error.message,
        status=error.http_status,
        path=request.url.path,
        method=request.method,
        correlation_id=correlation_id,
    )
    return JSONResponse(
        status_code=error.http_status,
        content=_envelope(error.error_code, error.message, correlation_id),
    )


async def http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Normalise framework ``HTTPException``s into the standard envelope.

    Without this, a 404 would return FastAPI's ``{"detail": ...}`` shape while
    everything else returned ours — clients would need two parsers.

    Args:
        request: The request being handled.
        exc: The raised ``HTTPException``.

    Returns:
        A JSON response carrying the error envelope.
    """
    status = exc.status_code if isinstance(exc, StarletteHTTPException | HTTPException) else 500
    detail = getattr(exc, "detail", None)
    phrase = HTTPStatus(status).phrase
    return JSONResponse(
        status_code=status,
        content=_envelope(
            code=phrase.lower().replace(" ", "_"),
            message=str(detail) if detail else phrase,
            correlation_id=_correlation_id(request),
        ),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle anything not otherwise caught.

    Reaching this handler means a bug. The traceback is logged in full; the
    caller receives only a generic message plus the correlation ID needed to
    find that traceback. In debug mode the exception type and text are included
    to shorten the local edit/test loop.

    Args:
        request: The request being handled.
        exc: The unexpected exception.

    Returns:
        A 500 JSON response carrying the error envelope.
    """
    correlation_id = _correlation_id(request)
    _log.exception(
        "unhandled_exception",
        error_type=type(exc).__name__,
        path=request.url.path,
        method=request.method,
        correlation_id=correlation_id,
    )

    settings = _settings_of(request)
    message = (
        f"{type(exc).__name__}: {exc}"
        if settings is not None and settings.debug
        else _GENERIC_MESSAGE
    )
    return JSONResponse(
        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        content=_envelope("internal_error", message, correlation_id),
    )


def install_error_handlers(app: FastAPI) -> None:
    """Register every exception handler on the application.

    Args:
        app: The FastAPI application to configure.
    """
    app.add_exception_handler(QuainexError, quainex_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
