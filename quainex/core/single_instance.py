"""Single-instance lock — one Quainex server per machine, per port.

Purpose:
    Stop a second server from starting at all, so the "two bridges fighting over
    one bot token" failure cannot happen even when two launches overlap.

Why this exists on top of the reload fix:
    Turning reload off removed *one* source of duplicate processes — uvicorn's
    file-watching supervisor-plus-worker pair. But two *independent* launches (the
    logon autostart and a manual ``python main.py``, say) are still two servers,
    and Telegram lets exactly one of them hold the long poll: the other 409s
    forever, messages arrive erratically, and the bridge still reports itself as
    running. That was hours of debugging once. This makes the second launch a
    clean, immediate refusal instead.

Why an OS advisory lock, not a PID file:
    A PID file left behind by a crash is a stale lock — the next start has to
    guess whether the pid inside is a live Quainex or a reused number, and guessing
    wrong either blocks a healthy start or lets a duplicate through. An advisory
    lock held on an open file handle is released by the kernel the instant the
    holding process dies, crash or not. There is nothing to clean up, and nothing
    to misjudge: either the lock is held right now, or it is not.

Why keyed on the port:
    The thing that must be unique is a server on a given port. Running a second
    instance deliberately on a *different* port (a test, a second profile) is not a
    conflict, and keying the lock on the port allows it while still refusing a true
    duplicate.

Cross-platform:
    ``msvcrt`` on Windows, ``fcntl`` elsewhere — the same non-blocking
    lock-or-fail in both. Quainex targets Windows today, but the lock has no reason
    to be Windows-only, and keeping it portable keeps the test suite honest.
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
from pathlib import Path
from typing import IO

from quainex.core.logging import get_logger

_log = get_logger(__name__)

#: Read into a plain bool so mypy does not narrow ``sys.platform`` to a single
#: platform and then flag the other branch as unreachable — the module genuinely
#: supports both.
_IS_WINDOWS: bool = sys.platform == "win32"


def _try_lock(handle: IO[str]) -> bool:
    """Take a non-blocking exclusive lock on a file handle.

    Args:
        handle: An open file handle.

    Returns:
        Whether the lock was acquired. ``False`` means another process holds it.
    """
    if _IS_WINDOWS:
        import msvcrt

        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    import fcntl

    try:
        # attr-defined: fcntl exists only on POSIX, where this branch runs; mypy
        # checks against Windows typeshed, which does not carry it.
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # type: ignore[attr-defined]
    except OSError:
        return False
    return True


def _unlock(handle: IO[str]) -> None:
    """Release the lock on a file handle.

    Args:
        handle: The locked file handle.
    """
    if _IS_WINDOWS:
        import msvcrt

        with contextlib.suppress(OSError):
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    with contextlib.suppress(OSError):
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]


class SingleInstanceLock:
    """Holds a machine-wide lock for the lifetime of one server process."""

    def __init__(self, port: int) -> None:
        """Prepare a lock for a server on ``port`` (nothing is acquired yet).

        Args:
            port: The port this server will bind, which the lock is keyed on.
        """
        self._path = Path(tempfile.gettempdir()) / f"quainex-{port}.lock"
        self._handle: IO[str] | None = None

    @property
    def path(self) -> Path:
        """The lock file's path."""
        return self._path

    def acquire(self) -> bool:
        """Try to become the single instance.

        Opening the file always succeeds; it is the *lock* on it that is exclusive.
        On success the process id is written for diagnostics — it is informational
        only, never trusted for the decision, which is the lock itself.

        Returns:
            Whether this process is now the single instance.
        """
        handle = open(self._path, "a+", encoding="utf-8")
        if not _try_lock(handle):
            handle.close()
            _log.warning("single_instance_busy", path=str(self._path))
            return False

        with contextlib.suppress(OSError):
            handle.seek(0)
            handle.truncate()
            handle.write(str(os.getpid()))
            handle.flush()
        self._handle = handle
        _log.info("single_instance_acquired", path=str(self._path), pid=os.getpid())
        return True

    def release(self) -> None:
        """Release the lock. Idempotent; safe to call from a ``finally``."""
        if self._handle is None:
            return
        _unlock(self._handle)
        with contextlib.suppress(OSError):
            self._handle.close()
        self._handle = None

    def __enter__(self) -> SingleInstanceLock:
        """Acquire on entry; the caller checks :meth:`acquired`."""
        self.acquire()
        return self

    def __exit__(self, *_exc: object) -> None:
        """Release on exit."""
        self.release()

    @property
    def acquired(self) -> bool:
        """Whether this process currently holds the lock."""
        return self._handle is not None
