"""Access token issue and verification.

Purpose:
    Prove that a request came from someone who knew the password, without
    sending the password on every request.

Why PyJWT rather than a hand-rolled signed token:
    Quainex already hand-rolls HMAC tokens for command confirmations, and that is
    fine — they are short-lived, single-purpose, and never leave this system.
    Authentication tokens are different: they are the thing an attacker attacks,
    the format a phone client has to interoperate with, and the place where
    subtle mistakes (unverified ``alg``, missing ``exp`` checks, timing leaks)
    have historically been catastrophic. This is the wrong place to be clever.

Architecture:
    POST /auth/token   password -> verify_password -> TokenService.issue()
    every request      Authorization: Bearer <jwt> -> TokenService.verify()

Dependencies:
    pyjwt, quainex.core.exceptions

Future improvements:
    * Refresh tokens, so a phone does not have to re-prompt hourly.
    * A revocation list, so a lost phone can be cut off before its token expires.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import jwt

from quainex.core.exceptions import AuthenticationError

#: HMAC-SHA256. Asymmetric signing buys nothing here: the same process issues and
#: verifies, so there is no third party needing a public key.
_ALGORITHM = "HS256"

#: Identifies tokens minted by this system, so a token from an unrelated service
#: that happens to share a secret cannot be replayed here.
_ISSUER = "quainex"


class TokenService:
    """Issues and verifies bearer tokens."""

    def __init__(self, secret: str, ttl_minutes: int) -> None:
        """Construct the service.

        Args:
            secret: Signing secret.
            ttl_minutes: How long an issued token remains valid.

        Raises:
            ValueError: The secret is empty or implausibly short.
        """
        if len(secret) < 32:
            raise ValueError(
                "The auth secret must be at least 32 characters. "
                'Generate one with: python -c "import secrets; print(secrets.token_urlsafe(48))"'
            )
        self._secret = secret
        self._ttl = timedelta(minutes=ttl_minutes)

    def issue(self, subject: str = "owner") -> tuple[str, int]:
        """Mint a token.

        Args:
            subject: Who the token identifies. Single-user today, but recorded so
                the audit trail does not have to be rewritten when that changes.

        Returns:
            The encoded token and its lifetime in seconds.
        """
        now = datetime.now(UTC)
        expires = now + self._ttl
        payload: dict[str, Any] = {
            "iss": _ISSUER,
            "sub": subject,
            "iat": now,
            "exp": expires,
        }
        token = jwt.encode(payload, self._secret, algorithm=_ALGORITHM)
        return token, int(self._ttl.total_seconds())

    def verify(self, token: str) -> str:
        """Verify a token and return its subject.

        Args:
            token: The encoded token.

        Returns:
            The subject the token identifies.

        Raises:
            AuthenticationError: The token is missing, expired, or invalid.
        """
        if not token:
            raise AuthenticationError("No access token was provided.")

        try:
            payload = jwt.decode(
                token,
                self._secret,
                # Pinned explicitly. Accepting whatever the token's own header
                # claims is the classic JWT vulnerability — including "none".
                algorithms=[_ALGORITHM],
                issuer=_ISSUER,
                options={"require": ["exp", "iss", "sub"]},
            )
        except jwt.ExpiredSignatureError as exc:
            raise AuthenticationError("The access token has expired.") from exc
        except jwt.InvalidTokenError as exc:
            raise AuthenticationError("The access token is not valid.") from exc

        subject = payload.get("sub")
        if not isinstance(subject, str):
            raise AuthenticationError("The access token is malformed.")
        return subject
