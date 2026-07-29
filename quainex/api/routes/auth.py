"""Authentication endpoints.

Purpose:
    Exchange the remote-access password for a short-lived bearer token.

Why the login route is itself unauthenticated but rate limited:
    It has to be reachable without a token — that is what it is for. What it must
    not be is an unlimited password oracle, so it is throttled through the same
    limiter as everything else, and it reports one indistinguishable failure for
    every rejection.

Architecture:
    POST /auth/token   {password}      -> {access_token, expires_in}
    GET  /auth/session Authorization:  -> who you are, when the token dies

Dependencies:
    fastapi, quainex.auth

Future improvements:
    * Refresh tokens, so a phone client does not re-prompt every hour.
    * Lock out after repeated failures from one address, not just throttle.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, status
from pydantic import BaseModel, Field

from quainex.api.dependencies import ContainerDep
from quainex.auth import AuthDep, verify_password
from quainex.core.exceptions import AuthenticationError, ConfigurationError, RateLimitedError
from quainex.core.logging import get_logger

router = APIRouter(prefix="/auth", tags=["auth"])
_log = get_logger(__name__)


class TokenRequest(BaseModel):
    """Body of a token request.

    Attributes:
        password: The remote-access password.
    """

    password: str = Field(min_length=1)


class TokenResponse(BaseModel):
    """An issued access token.

    Attributes:
        access_token: The bearer token.
        token_type: Always ``bearer``.
        expires_in: Lifetime in seconds.
    """

    access_token: str
    # The OAuth2 token-type literal, not a credential.
    token_type: str = "bearer"  # noqa: S105
    expires_in: int


class SessionResponse(BaseModel):
    """Who the caller is.

    Attributes:
        subject: The authenticated subject.
        auth_required: Whether this instance enforces authentication.
    """

    subject: str
    auth_required: bool


@router.post(
    "/token",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Exchange the password for an access token",
    responses={401: {"description": "The password was not accepted."}},
)
async def issue_token(
    body: TokenRequest, request: Request, container: ContainerDep
) -> TokenResponse:
    """Issue a bearer token in exchange for the remote password.

    Args:
        body: The submitted password.
        request: The incoming request, used for rate limiting.
        container: Injected application container.

    Returns:
        The issued token and its lifetime.

    Raises:
        ConfigurationError: Authentication is not configured on this instance.
        RateLimitedError: Too many attempts from this address.
        AuthenticationError: The password was rejected.
    """
    settings = container.settings
    if not settings.auth_password_hash or container.tokens is None:
        raise ConfigurationError(
            "Authentication is not configured on this instance. "
            "Run `python scripts/hash_password.py` to set a remote password."
        )

    client = request.client.host if request.client else "unknown"
    if not container.rate_limiter.check(client):
        raise RateLimitedError("Too many attempts. Slow down and try again shortly.")

    if not verify_password(body.password, settings.auth_password_hash):
        # Logged with the address but never the attempted password.
        _log.warning("auth_failed", client=client)
        raise AuthenticationError("Incorrect password.")

    token, expires_in = container.tokens.issue()
    _log.info("auth_succeeded", client=client, expires_in=expires_in)
    return TokenResponse(access_token=token, expires_in=expires_in)


@router.get("/session", response_model=SessionResponse, summary="Who am I?")
async def session(subject: AuthDep, container: ContainerDep) -> SessionResponse:
    """Report the authenticated subject.

    Useful to a client checking whether its stored token is still good.

    Args:
        subject: The authenticated subject.
        container: Injected application container.

    Returns:
        The subject and whether this instance enforces authentication.
    """
    return SessionResponse(subject=subject, auth_required=container.settings.auth_required)
