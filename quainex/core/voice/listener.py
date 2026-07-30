"""Always-on wake-word listening.

Purpose:
    Let "Quainex, take a screenshot" work with nothing pressed and nothing open —
    the part that makes this feel like an assistant rather than a web page.

How it stays cheap:
    Each cycle records one utterance, transcribes it **locally** with Whisper, and
    checks for the wake word. ``VoiceSession.handle_transcript`` returns before the
    Brain is called when the wake word is absent, so ambient conversation costs
    **zero API tokens** — it is heard, discarded, and never leaves the machine.

    Only a request actually addressed to Quainex reaches a provider, and the local
    fast path handles most of those without one either. So a room full of talking
    costs CPU and nothing else.

The microphone is open. That is the deal, stated plainly:
    This holds the microphone continuously. Speech is transcribed on this machine
    by a local model and discarded unless it was addressed to Quainex — no audio
    is uploaded and no transcript is stored unless
    ``QUAINEX_KEEP_RECORDINGS`` is set. But "the mic is open" is a fact about the
    room, not a detail, and someone else in it has not agreed to anything.

    So it is **off by default** and has to be turned on deliberately, it says so in
    the log when it starts, and it can be stopped from the dashboard or the API.
    An assistant that quietly started listening because a setting defaulted to
    ``true`` would be a different kind of product.

Why a loop over ``listen_and_respond`` rather than a streaming detector:
    A dedicated wake-word engine (Porcupine, openWakeWord) runs a small model over
    a rolling buffer and is the right answer eventually — lower latency, lower CPU.
    Each is also a dependency, a licence, and a model file.

    This reuses machinery that already exists and is already tested: the recorder
    stops on silence, the transcript path is the same one the API uses, and the
    wake gate is the same code with the same phonetic folding. The cost is latency
    — Quainex hears you only after you stop speaking, so the wake word and the
    request go in one breath: "Quainex, take a screenshot", not "Quainex" … pause …
    "take a screenshot".

Architecture:
    WakeWordListener.run()
        loop:
          VoiceSession.listen_and_respond(require_wake_word=True)
             |-- record until silence          (local)
             |-- Whisper transcribe            (local, no tokens)
             |-- wake word absent -> discard, loop   <-- the common case
             +-- wake word present -> Brain -> execute -> speak

Dependencies:
    quainex.core.voice.session, quainex.config.settings

Example:
    >>> listener = WakeWordListener(voice=session, settings=settings)
    >>> await listener.run()  # doctest: +SKIP

Future improvements:
    * A real streaming wake-word engine, to cut the "after you stop speaking" delay.
    * A short follow-up window, so a second request needs no second wake word.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from quainex.core.exceptions import QuainexError, SpeechUnavailableError
from quainex.core.logging import get_logger

if TYPE_CHECKING:
    from quainex.config.settings import Settings
    from quainex.core.voice.session import VoiceSession

_log = get_logger(__name__)

#: Pause after a failed cycle, so a persistent fault cannot become a busy loop.
_ERROR_BACKOFF_SECONDS = 3.0

#: Consecutive failures before the listener gives up.
#:
#: Bounded because the failure that matters here is not transient: a microphone
#: unplugged, or a Whisper model that will not load. Retrying that forever burns
#: CPU and fills the log while nothing works, and the status endpoint would keep
#: claiming the listener is running.
_MAX_CONSECUTIVE_FAILURES = 5


class WakeWordListener:
    """Listens continuously and acts only when addressed."""

    def __init__(self, *, voice: VoiceSession, settings: Settings) -> None:
        """Construct the listener.

        Args:
            voice: The session that records, transcribes, understands and speaks.
            settings: Application configuration.
        """
        self._voice = voice
        self._settings = settings
        self._running = False
        #: Observed, not declared — the same lesson the Telegram bridge taught.
        #: A loop can stall while a boolean still says it is running.
        self._last_cycle_at: float | None = None
        self._heard = 0
        self._acted = 0

    @property
    def is_running(self) -> bool:
        """Whether the listening loop is active."""
        return self._running

    def status(self) -> dict[str, object]:
        """Report listener state.

        Returns:
            Whether it is available and running, how long since the last cycle
            completed, and how much it has heard versus acted on. The last pair is
            the honest measure of whether the wake word is working: a large gap
            means it is hearing the room and correctly ignoring it.
        """
        return {
            "available": self._voice.is_available,
            "running": self._running,
            "last_cycle_seconds_ago": (
                None
                if self._last_cycle_at is None
                else round(time.monotonic() - self._last_cycle_at, 1)
            ),
            "utterances_heard": self._heard,
            "requests_acted_on": self._acted,
            "wake_word": self._settings.wake_word,
        }

    def arm(self) -> None:
        """Check the preconditions and mark the listener as running.

        Separate from ``run()`` and synchronous on purpose. ``run()`` is driven as a
        background task, so anything it raises lands in the task rather than in the
        caller — an endpoint would report success while startup had actually failed,
        and the status it returned would say ``running: false`` because the
        coroutine had not begun yet.

        Calling this first means a caller learns about a missing microphone
        immediately, and sees accurate state the moment it returns.

        Raises:
            RuntimeError: The listener is already running.
            SpeechUnavailableError: There is no microphone or no recogniser.
        """
        if self._running:
            raise RuntimeError("The wake-word listener is already running.")
        if not self._voice.is_available:
            raise SpeechUnavailableError(
                'Voice support is not installed. Run: pip install -e ".[voice]"'
            )
        self._running = True

    async def run(self) -> None:
        """Listen until stopped.

        Calls ``arm()`` unless the caller already did, so it works both as a
        directly-awaited coroutine and as a task started after arming.

        Raises:
            RuntimeError: The listener is already running.
            SpeechUnavailableError: There is no microphone or no recogniser.
        """
        if not self._running:
            self.arm()
        # Deliberately prominent. Opening a microphone indefinitely is the kind of
        # thing that should be discoverable afterwards, from the log, by someone who
        # did not turn it on.
        _log.warning(
            "wake_word_listener_started",
            detail=(
                "The microphone is now open continuously. Speech is transcribed "
                "locally and discarded unless it begins with the wake word."
            ),
            wake_word=self._settings.wake_word,
        )

        failures = 0
        try:
            while self._running:
                try:
                    await self._cycle()
                    failures = 0
                except SpeechUnavailableError as exc:
                    # Not transient: the microphone or the model is gone.
                    _log.error("wake_word_listener_unavailable", reason=exc.message)
                    break
                except (QuainexError, OSError) as exc:
                    failures += 1
                    _log.warning(
                        "wake_word_cycle_failed",
                        reason=str(exc),
                        consecutive_failures=failures,
                    )
                    if failures >= _MAX_CONSECUTIVE_FAILURES:
                        _log.error(
                            "wake_word_listener_giving_up",
                            detail=(
                                f"{failures} cycles failed in a row. Stopping rather "
                                f"than looping on a fault that is not going away."
                            ),
                        )
                        break
                    await asyncio.sleep(_ERROR_BACKOFF_SECONDS)
        finally:
            self._running = False
            _log.info(
                "wake_word_listener_stopped",
                utterances_heard=self._heard,
                requests_acted_on=self._acted,
            )

    def stop(self) -> None:
        """Ask the loop to finish after the current cycle.

        The current recording is not interrupted: it is at most a few seconds, and
        cutting the microphone mid-buffer is how a half-written WAV reaches the
        transcriber.
        """
        self._running = False

    # -- internals --------------------------------------------------------

    async def _cycle(self) -> None:
        """Record one utterance and act on it if it was addressed to Quainex."""
        turn = await self._voice.listen_and_respond(require_wake_word=True)
        self._last_cycle_at = time.monotonic()

        if not turn.transcript.text.strip():
            # Silence, which is most of the time. Not logged: a line every few
            # seconds would bury everything that matters.
            return

        self._heard += 1
        if not turn.wake_word_detected:
            # Heard and discarded. Logged at debug with a length rather than the
            # text: recording what was said in the room, when it was not addressed
            # to Quainex, would defeat the point of discarding it.
            _log.debug("wake_word_listener_ignored", characters=len(turn.transcript.text))
            return

        self._acted += 1
        _log.info(
            "wake_word_listener_acted",
            command=turn.command_text[:120],
            intent=None if turn.intent is None else turn.intent.intent.value,
            status=None if turn.result is None else turn.result.status.value,
        )
