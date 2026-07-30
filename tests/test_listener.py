"""Tests for always-on wake-word listening.

No microphone is opened. The session is a double, so what is verified is the
listener's own contract: that ambient speech costs nothing, that a fault stops the
loop instead of spinning on it, and that the microphone is released.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from quainex.config.settings import Settings
from quainex.core.brain import Intent, IntentType
from quainex.core.commands import CommandStatus
from quainex.core.commands.base import CommandResult
from quainex.core.exceptions import NoSpeechError, ProviderError, SpeechUnavailableError
from quainex.core.voice import Transcript, VoiceTurn, WakeWordListener


def _turn(text: str, *, detected: bool) -> VoiceTurn:
    """Build a voice turn as the session would return one.

    Args:
        text: What was heard.
        detected: Whether the wake word was found.

    Returns:
        The turn.
    """
    intent = (
        Intent(
            intent=IntentType.SCREENSHOT,
            target=None,
            confidence=1.0,
            reasoning="test",
            requires_confirmation=False,
            utterance=text,
        )
        if detected
        else None
    )
    result = (
        CommandResult(
            status=CommandStatus.SUCCEEDED,
            intent="screenshot",
            message="Saved a screenshot.",
            executed=True,
        )
        if detected
        else None
    )
    return VoiceTurn(
        transcript=Transcript(text=text),
        wake_word_detected=detected,
        command_text="take a screenshot" if detected else text,
        intent=intent,
        result=result,
    )


class FakeSession:
    """Yields scripted turns, then stops the listener.

    Attributes:
        cycles: How many times the microphone was opened.
    """

    def __init__(self, turns: list[VoiceTurn | Exception], available: bool = True) -> None:
        self._turns = list(turns)
        self._available = available
        self.cycles = 0
        self.listener: WakeWordListener | None = None

    @property
    def is_available(self) -> bool:
        return self._available

    async def listen_and_respond(self, **_: object) -> VoiceTurn:
        self.cycles += 1
        if not self._turns:
            # Nothing scripted left: end the run rather than looping forever.
            if self.listener is not None:
                self.listener.stop()
            return _turn("", detected=False)
        nxt = self._turns.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def _listener(session: FakeSession, tmp_path: Path) -> WakeWordListener:
    """Build a listener over a fake session.

    Args:
        session: The scripted session.
        tmp_path: Per-test directory.

    Returns:
        The listener, with the session wired to stop it when exhausted.
    """
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        log_dir=tmp_path / "logs",
        database_path=tmp_path / "t.db",
    )
    listener = WakeWordListener(voice=session, settings=settings)  # type: ignore[arg-type]
    session.listener = listener
    return listener


# -- the point of the whole thing ------------------------------------------


async def test_ambient_speech_is_heard_and_discarded(tmp_path):
    """The property that makes always-on listening affordable.

    Unaddressed speech is transcribed *locally* and dropped before the Brain is
    called, so a room full of talking costs CPU and zero API tokens.
    """
    session = FakeSession(
        [
            _turn("so anyway I told him it was fine", detected=False),
            _turn("did you see the match last night", detected=False),
        ]
    )
    listener = _listener(session, tmp_path)

    await asyncio.wait_for(listener.run(), timeout=5)

    status = listener.status()
    assert status["utterances_heard"] == 2
    assert status["requests_acted_on"] == 0


async def test_an_addressed_request_is_acted_on(tmp_path):
    session = FakeSession([_turn("Quainex take a screenshot", detected=True)])
    listener = _listener(session, tmp_path)

    await asyncio.wait_for(listener.run(), timeout=5)

    assert listener.status()["requests_acted_on"] == 1


async def test_silence_is_not_counted_as_an_utterance(tmp_path):
    """Most cycles are silence; counting them would make the counters meaningless."""
    session = FakeSession([_turn("   ", detected=False), _turn("", detected=False)])
    listener = _listener(session, tmp_path)

    await asyncio.wait_for(listener.run(), timeout=5)

    assert listener.status()["utterances_heard"] == 0


# -- refusing to run when it cannot ---------------------------------------


async def test_a_missing_microphone_is_refused_up_front(tmp_path):
    """Rather than failing on the first cycle, repeatedly."""
    listener = _listener(FakeSession([], available=False), tmp_path)

    with pytest.raises(SpeechUnavailableError):
        await listener.run()

    assert listener.is_running is False


async def test_starting_twice_is_refused(tmp_path):
    """Two loops would fight over one microphone."""
    listener = _listener(FakeSession([]), tmp_path)
    listener.arm()

    with pytest.raises(RuntimeError, match="already running"):
        listener.arm()


def test_arming_surfaces_a_missing_microphone_to_the_caller(tmp_path):
    """Not to a background task where nobody sees it.

    ``run()`` is driven as a task, so anything it raises lands in the task. Without
    a synchronous precondition check the start endpoint reported success while
    startup had actually failed — and returned ``running: false`` in the same
    breath, because the coroutine had not begun yet.
    """
    listener = _listener(FakeSession([], available=False), tmp_path)

    with pytest.raises(SpeechUnavailableError):
        listener.arm()

    assert listener.is_running is False


def test_arming_makes_the_state_true_before_the_loop_starts(tmp_path):
    """So a status read straight after starting is accurate rather than racy."""
    listener = _listener(FakeSession([]), tmp_path)

    listener.arm()

    assert listener.is_running is True
    assert listener.status()["running"] is True


async def test_speech_becoming_unavailable_stops_the_loop(tmp_path):
    """An unplugged microphone is not a transient fault.

    Retrying it forever would spin, fill the log, and keep reporting the listener
    as running while nothing worked.
    """
    session = FakeSession([SpeechUnavailableError("microphone gone")])
    listener = _listener(session, tmp_path)

    await asyncio.wait_for(listener.run(), timeout=5)

    assert listener.is_running is False
    assert session.cycles == 1


async def test_silence_is_not_a_failure(tmp_path):
    """The bug that made hands-free mode useless.

    The recorder raises when a recording holds no speech, and the listener counted
    that toward its give-up threshold — so a quiet room drove it to shut down after
    about ninety seconds. Broken in exactly the situation it exists for, and only
    visible by leaving it running.

    Far more silent cycles than the threshold here, and it must still be listening
    at the end.
    """
    from quainex.core.voice.listener import _MAX_CONSECUTIVE_FAILURES

    quiet = [NoSpeechError("No speech was detected.")] * (_MAX_CONSECUTIVE_FAILURES * 3)
    session = FakeSession([*quiet, _turn("Quainex take a screenshot", detected=True)])
    listener = _listener(session, tmp_path)

    await asyncio.wait_for(listener.run(), timeout=10)

    # It survived the silence and acted on the request that followed it.
    assert listener.status()["requests_acted_on"] == 1


async def test_silence_still_counts_as_a_completed_cycle(tmp_path):
    """Otherwise a quiet room looks like a stalled loop.

    ``last_cycle_seconds_ago`` is what distinguishes "listening, nothing said" from
    "hung", so silence has to update it.
    """
    session = FakeSession([NoSpeechError("No speech was detected.")])
    listener = _listener(session, tmp_path)

    await asyncio.wait_for(listener.run(), timeout=5)

    assert isinstance(listener.status()["last_cycle_seconds_ago"], float)


async def test_repeated_failures_give_up_rather_than_spin(tmp_path):
    """Bounded, because the failures that matter here do not resolve themselves."""
    from quainex.core.voice.listener import _MAX_CONSECUTIVE_FAILURES

    session = FakeSession([ProviderError("nothing works")] * (_MAX_CONSECUTIVE_FAILURES + 5))
    listener = _listener(session, tmp_path)

    # No sleeping in the test: the backoff is patched out so this stays fast.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("quainex.core.voice.listener._ERROR_BACKOFF_SECONDS", 0)
        await asyncio.wait_for(listener.run(), timeout=10)

    assert listener.is_running is False
    assert session.cycles == _MAX_CONSECUTIVE_FAILURES


async def test_a_transient_failure_does_not_end_the_loop(tmp_path):
    """One bad cycle must not cost hands-free mode for the rest of the session."""
    session = FakeSession(
        [
            ProviderError("one blip"),
            _turn("Quainex take a screenshot", detected=True),
        ]
    )
    listener = _listener(session, tmp_path)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("quainex.core.voice.listener._ERROR_BACKOFF_SECONDS", 0)
        await asyncio.wait_for(listener.run(), timeout=10)

    assert listener.status()["requests_acted_on"] == 1


# -- state reporting -------------------------------------------------------


async def test_status_reports_evidence_that_the_loop_is_alive(tmp_path):
    """A boolean can be stale; a timestamp cannot.

    The same lesson the Telegram bridge taught, applied before it could bite here.
    """
    listener = _listener(FakeSession([_turn("hello", detected=False)]), tmp_path)

    assert listener.status()["last_cycle_seconds_ago"] is None

    await asyncio.wait_for(listener.run(), timeout=5)

    assert isinstance(listener.status()["last_cycle_seconds_ago"], float)


def test_status_names_the_wake_word(tmp_path):
    """So the dashboard can tell the user what to actually say."""
    listener = _listener(FakeSession([]), tmp_path)

    assert listener.status()["wake_word"] == "quainex"


def test_it_is_off_by_default(tmp_path):
    """Opening a microphone must be a deliberate act, not an inherited default."""
    settings = Settings(_env_file=None, log_dir=tmp_path / "logs")  # type: ignore[call-arg]

    assert settings.voice_always_listening is False
