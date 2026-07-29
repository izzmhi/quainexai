"""Signed confirmation tokens.

Purpose:
    Make "the user approved this" something a caller must *prove*, not something
    it can simply assert.

The hole this closes:
    Phase 3 shipped ``POST /commands/execute`` with a ``confirmed: true`` flag.
    On localhost that is defensible — the only caller is the user. The moment
    Phase 6 exposes the API to a phone, it stops being defensible: any client
    holding a token could send ``{"intent": {...shutdown...}, "confirmed": true}``
    without ever having shown a human anything. Authentication proves *who* is
    calling; it says nothing about whether a person was actually asked.

How it works:
    When the executor refuses an action pending confirmation, it mints a token
    bound to that exact action and hands it back with the refusal. Executing
    requires presenting that token. Because the token is an HMAC over the intent
    and target, a token issued for "close Spotify" cannot authorise "shut down" —
    and because it carries an expiry and a one-use nonce, it cannot be replayed.

    Binding matters as much as signing. A token that merely said "the user
    confirmed something" would be a skeleton key for whatever the caller asked
    next.

Architecture:
    execute(shutdown)            -> REQUIRES_CONFIRMATION + token
    user sees the prompt, agrees
    execute(shutdown, token)     -> verified, bound, unexpired, unused -> runs

Dependencies:
    Standard library only.

Future improvements:
    * Persist spent nonces, so a restart cannot reopen a replay window.
    * Bind the token to the authenticated subject, so one user's confirmation
      cannot be redeemed by another once Quainex is multi-user.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from typing import TYPE_CHECKING

from quainex.core.logging import get_logger

if TYPE_CHECKING:
    from quainex.core.brain import Intent

_log = get_logger(__name__)

#: Long enough to read a prompt and decide, short enough that a token captured
#: from a log is almost always already dead.
DEFAULT_TTL_SECONDS = 120


class ConfirmationService:
    """Mints and verifies single-use, action-bound confirmation tokens."""

    def __init__(self, secret: str, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        """Construct the service.

        Args:
            secret: HMAC signing secret.
            ttl_seconds: How long a token remains redeemable.
        """
        self._secret = secret.encode("utf-8")
        self._ttl = ttl_seconds
        # Nonces of tokens already redeemed. In-process only: a restart clears
        # it, which is acceptable because every token expires in two minutes
        # anyway, so the widest replay window a restart can open is that.
        self._spent: set[str] = set()

    def issue(self, intent: Intent) -> str:
        """Mint a token authorising one execution of one action.

        Args:
            intent: The action awaiting confirmation.

        Returns:
            An opaque token to hand back with the refusal.
        """
        payload = {
            "intent": intent.intent.value,
            "target": (intent.target or "").strip(),
            "exp": int(time.time()) + self._ttl,
            "nonce": secrets.token_urlsafe(16),
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        signature = hmac.new(self._secret, raw, hashlib.sha256).digest()
        return f"{_b64(raw)}.{_b64(signature)}"

    def verify(self, token: str, intent: Intent) -> bool:
        """Check that a token authorises this exact action, once.

        Args:
            token: The token presented by the caller.
            intent: The action being attempted.

        Returns:
            Whether the token is valid, bound to this action, unexpired and
            unredeemed. Redeems it as a side effect when it is.
        """
        if not token:
            return False

        try:
            raw_b64, signature_b64 = token.split(".", 1)
            raw = _unb64(raw_b64)
            signature = _unb64(signature_b64)
        except (ValueError, TypeError):
            return False

        expected = hmac.new(self._secret, raw, hashlib.sha256).digest()
        if not hmac.compare_digest(signature, expected):
            _log.warning("confirmation_token_bad_signature", intent=intent.intent.value)
            return False

        try:
            payload = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return False

        if not isinstance(payload, dict):
            return False

        if int(payload.get("exp", 0)) < time.time():
            _log.info("confirmation_token_expired", intent=intent.intent.value)
            return False

        # The binding check. Without this, a signed token would authorise
        # anything the caller asked for next.
        if payload.get("intent") != intent.intent.value:
            _log.warning(
                "confirmation_token_wrong_intent",
                issued_for=payload.get("intent"),
                presented_for=intent.intent.value,
            )
            return False
        if payload.get("target", "") != (intent.target or "").strip():
            _log.warning("confirmation_token_wrong_target", intent=intent.intent.value)
            return False

        nonce = payload.get("nonce")
        if not isinstance(nonce, str) or nonce in self._spent:
            _log.warning("confirmation_token_replayed", intent=intent.intent.value)
            return False

        self._spent.add(nonce)
        self._prune()
        return True

    def _prune(self) -> None:
        """Bound the spent-nonce set.

        Every token expires within the TTL, so old nonces stop being useful long
        before this matters; the cap exists so a long-running process cannot grow
        the set without limit.
        """
        if len(self._spent) > 10_000:
            self._spent.clear()


def _b64(data: bytes) -> str:
    """URL-safe base64 without padding.

    Args:
        data: Bytes to encode.

    Returns:
        The encoded string.
    """
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    """Decode URL-safe base64 that may be missing padding.

    Args:
        text: The encoded string.

    Returns:
        The decoded bytes.
    """
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)
