"""Tests for the single-instance lock.

The point of the lock is that a second server refuses to start, so the "two
Telegram bridges fighting over one bot token" failure cannot happen. These prove
the mutual exclusion, that the first holder still works, and that releasing frees
it for the next start.
"""

from __future__ import annotations

from quainex.core.single_instance import SingleInstanceLock

# A port unlikely to collide with a real server during the test run.
_PORT = 8123


def test_one_holder_at_a_time():
    first = SingleInstanceLock(_PORT)
    second = SingleInstanceLock(_PORT)
    try:
        assert first.acquire() is True
        # A second instance on the same port cannot acquire while the first holds it.
        assert second.acquire() is False
        assert first.acquired is True
        assert second.acquired is False
    finally:
        first.release()
        second.release()


def test_releasing_frees_it_for_the_next_start():
    first = SingleInstanceLock(_PORT)
    assert first.acquire() is True
    first.release()

    # After the first lets go, a fresh start succeeds — a restart must not be blocked.
    second = SingleInstanceLock(_PORT)
    try:
        assert second.acquire() is True
    finally:
        second.release()


def test_a_different_port_is_not_a_conflict():
    a = SingleInstanceLock(_PORT)
    b = SingleInstanceLock(_PORT + 1)
    try:
        assert a.acquire() is True
        # Deliberately running a second instance on another port is allowed.
        assert b.acquire() is True
    finally:
        a.release()
        b.release()


def test_release_is_idempotent():
    lock = SingleInstanceLock(_PORT)
    assert lock.acquire() is True
    lock.release()
    lock.release()  # must not raise
    assert lock.acquired is False


def test_context_manager_acquires_and_releases():
    with SingleInstanceLock(_PORT) as lock:
        assert lock.acquired is True
    assert lock.acquired is False
