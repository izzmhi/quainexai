"""Request rate limiting.

Purpose:
    Bound how fast one client can drive Quainex.

Why it exists at all:
    Two reasons, and the second is the important one. Rate limiting slows an
    online password-guessing attack against ``/auth/token``. It also caps the
    damage a *legitimate* client can do by looping — Phase 10's autonomous agent
    is a program that issues commands on its own, and a bug in it should hit a
    ceiling rather than execute a thousand actions.

    Applied only when authentication is required. A localhost-only Quainex has
    exactly one caller who can already do anything they want directly; throttling
    them would be friction with no security benefit.

Implementation:
    A fixed-window counter, deliberately. A token bucket or sliding log is more
    precise about burst behaviour, but this runs in one process for one user —
    the precision buys nothing, and the simpler thing has no edge cases to get
    wrong.

Dependencies:
    Standard library only.

Future improvements:
    * Move to a shared store if Quainex ever runs multi-process.
    * A stricter, separate limit on ``/auth/token`` than on ordinary requests.
"""

from __future__ import annotations

import time
from collections import defaultdict

_WINDOW_SECONDS = 60


class RateLimiter:
    """Counts requests per client within a fixed window."""

    def __init__(self, limit_per_minute: int) -> None:
        """Construct the limiter.

        Args:
            limit_per_minute: Requests allowed per client per minute.
        """
        self._limit = limit_per_minute
        self._counts: dict[str, int] = defaultdict(int)
        self._window_started = time.monotonic()

    def check(self, client: str) -> bool:
        """Record a request and report whether it is allowed.

        Args:
            client: Identifier for the caller, typically its address.

        Returns:
            ``True`` if the request is within the limit.
        """
        now = time.monotonic()
        if now - self._window_started >= _WINDOW_SECONDS:
            # New window: clearing the whole map also bounds memory, so a
            # long-running process cannot accumulate an entry per address seen.
            self._counts.clear()
            self._window_started = now

        self._counts[client] += 1
        return self._counts[client] <= self._limit

    def remaining(self, client: str) -> int:
        """Report how many requests the client has left this window.

        Args:
            client: Identifier for the caller.

        Returns:
            Remaining allowance, never negative.
        """
        return max(0, self._limit - self._counts.get(client, 0))
