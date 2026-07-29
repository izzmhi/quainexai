"""Authentication dependency.

Purpose:
    Guard every route that can see or change anything, in one place.

Why a dependency rather than middleware:
    Middleware would need a path allowlist — a string-matching list that silently
    stops protecting a route the day someone renames it. As a dependency, the
    protection is attached to the router itself: a new route added to a protected
    router is protected by construction, and forgetting to protect a *new* router
    is a visible omission in ``app.py`` rather than an invisible one in a regex.

What is deliberately left open:
    ``/health`` only. A supervisor has to be able to ask whether the process is
    alive without holding a credential, and the response says nothing an attacker
    could not learn by watching the port.

Architecture:
    request -> require_auth
                 |-- auth not required (loopback)  -> allow
                 |-- rate limit exceeded           -> 429
                 |-- no/invalid bearer token       -> 401
                 +-- valid                         -> subject

Dependencies:
    fastapi, quainex.auth.tokens, quainex.security.ratelimit

Future improvements:
    * Per-scope tokens, so a phone client can be granted read-only access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, cast

from fastapi import Depends, Request

from quainex.core.exceptions import AuthenticationError, RateLimitedError

if TYPE_CHECKING:
    # Type-only: importing the container at runtime would be a cycle, since the
    # container constructs the TokenService this module consumes.
    from quainex.core.container import Container

#: Returned as the subject when authentication is not enforced.
LOCAL_SUBJECT = "local"


def _bearer_token(request: Request) -> str:
    """Extract the bearer token from the Authorization header.

    Args:
        request: The incoming request.

    Returns:
        The token, or an empty string when absent or malformed.
    """
    header = request.headers.get("Authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer":
        return ""
    return token.strip()


def require_auth(request: Request) -> str:
    """Authenticate the caller.

    Args:
        request: The incoming request.

    Returns:
        The authenticated subject, or ``LOCAL_SUBJECT`` when auth is not enforced.

    Raises:
        AuthenticationError: The token is missing, expired or invalid.
        RateLimitedError: The caller exceeded the configured request rate.
    """
    container = cast("Container | None", getattr(request.app.state, "container", None))
    if container is None or not container.settings.auth_required:
        return LOCAL_SUBJECT

    client = request.client.host if request.client else "unknown"
    if not container.rate_limiter.check(client):
        raise RateLimitedError("Too many requests. Slow down and try again shortly.")

    if container.tokens is None:
        # Unreachable in practice: settings validation refuses to start when auth
        # is required without a secret. Fails closed rather than assuming so.
        raise AuthenticationError("Authentication is not configured on this instance.")

    return container.tokens.verify(_bearer_token(request))


#: Annotated alias so routes can write `subject: AuthDep`.
AuthDep = Annotated[str, Depends(require_auth)]
