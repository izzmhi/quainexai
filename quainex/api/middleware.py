"""HTTP middleware.

Purpose:
    Give every request a correlation ID and bind it to the logging context, so
    that all records emitted while handling that request — including ones from
    deep inside the AI provider — can be tied back to it.

Why this exists in Phase 1:
    From Phase 6 onward Quainex accepts commands from a phone, and from Phase 10
    it acts on its own. When something goes wrong the question is always "what
    was the system doing at the time" — answerable only if related log lines
    share an identifier. Adding this later means the earliest, least understood
    incidents are the ones without traces.

Architecture:
    request
        -> CorrelationIdMiddleware  (generate/accept ID, bind to contextvars)
        -> route handler            (logs inherit the ID automatically)
        -> response                 (ID echoed in the X-Correlation-ID header)

Dependencies:
    starlette, structlog, quainex.core.logging

Future improvements:
    * Accept W3C ``traceparent`` so IDs survive across the dashboard/API boundary.
    * Emit a request-completed record with duration once metrics are needed.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import structlog
from starlette.middleware.base import BaseHTTPMiddleware

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from starlette.requests import Request
    from starlette.responses import Response

#: Header used to accept an upstream ID and to echo the one in play.
CORRELATION_ID_HEADER = "X-Correlation-ID"

#: Key under which the ID is bound into the structlog context.
CORRELATION_ID_KEY = "correlation_id"


def new_correlation_id() -> str:
    """Generate a fresh correlation identifier.

    Returns:
        A 32-character hex string.
    """
    return uuid.uuid4().hex


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Attach a correlation ID to every request and its log records."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Bind a correlation ID for the duration of the request.

        An inbound ``X-Correlation-ID`` is honoured so a caller can stitch its
        own trace to ours; otherwise one is generated.

        Args:
            request: The incoming request.
            call_next: The next handler in the middleware chain.

        Returns:
            The downstream response, with the correlation ID header set.
        """
        correlation_id = request.headers.get(CORRELATION_ID_HEADER) or new_correlation_id()

        # clear_contextvars prevents leakage between requests sharing a worker task.
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(**{CORRELATION_ID_KEY: correlation_id})

        # Stored on state as well so exception handlers, which run outside this
        # call stack, can read the same value.
        request.state.correlation_id = correlation_id

        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.clear_contextvars()

        response.headers[CORRELATION_ID_HEADER] = correlation_id
        return response
