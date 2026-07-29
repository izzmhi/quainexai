"""Tests for the voice pipeline: wake-word gating, the turn loop, and endpoints.

No microphone is opened, no model is downloaded, and nothing is spoken aloud.
Every component is behind a Protocol precisely so this suite can run on a machine
with no audio hardware at all.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from quainex.config.settings import Settings
from quainex.core.brain import Brain, IntentClassification, IntentType
from quainex.core.commands import build_executor
from quainex.core.exceptions import SpeechError, SpeechUnavailableError
from quainex.core.voice import Transcript, VoiceSession, detect_wake_word
from tests.test_brain import FakeProvider
from tests.test_commands import FakeDesktopController

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


class FakeSTT:
    """Returns a canned transcript instead of running Whisper."""

    def __init__(self, text: str = "quainex open vs code", available: bool = True) -> None:
        self._text = text
        self._available = available
        self.calls = 0

    @property
    def name(self) -> str:
        return "fake-stt"

    @property
    def is_available(self) -> bool:
        return self._available

    def transcribe(self, audio_path: Path) -> Transcript:
        self.calls += 1
        return Transcript(text=self._text, language="en", duration_seconds=1.0)


class FakeTTS:
    """Records what would have been spoken."""

    def __init__(self, available: bool = True) -> None:
        self.spoken: list[str] = []
        self._available = available

    @property
    def name(self) -> str:
        return "fake-tts"

    @property
    def is_available(self) -> bool:
        return self._available

    def speak(self, text: str) -> None:
        self.spoken.append(text)

    def synthesise(self, text: str, destination: Path) -> Path:
        self.spoken.append(text)
        return destination


class FakeRecorder:
    """Writes a silent WAV instead of opening a microphone."""

    def __init__(self, available: bool = True) -> None:
        self._available = available
        self.calls = 0

    @property
    def is_available(self) -> bool:
        return self._available

    def record(self, destination: Path) -> Path:
        self.calls += 1
        destination.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(destination), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16_000)
            handle.writeframes(b"\x00\x00" * 1600)
        return destination


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    base: dict[str, object] = {
        "_env_file": None,
        "log_dir": tmp_path / "logs",
        "command_search_roots": [tmp_path],
        "screenshot_dir": tmp_path / "shots",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _session(
    tmp_path: Path,
    *,
    stt: FakeSTT | None = None,
    tts: FakeTTS | None = None,
    recorder: FakeRecorder | None = None,
    desktop: FakeDesktopController | None = None,
    intent: IntentType = IntentType.OPEN_APPLICATION,
    **setting_overrides: object,
) -> VoiceSession:
    settings = _settings(tmp_path, **setting_overrides)
    provider = FakeProvider(
        IntentClassification(
            intent=intent,
            target="VS Code",
            confidence=0.97,
            reasoning="test",
        )
    )
    return VoiceSession(
        stt=stt or FakeSTT(),
        tts=tts or FakeTTS(),
        recorder=recorder or FakeRecorder(),
        brain=Brain(provider=provider, settings=settings),
        commands=build_executor(desktop or FakeDesktopController(), settings),
        settings=settings,
    )


# -- wake word -------------------------------------------------------------


@pytest.mark.parametrize(
    ("said", "expected_command"),
    [
        ("quainex open vs code", "open vs code"),
        ("Quainex, open VS Code", "open VS Code"),
        ("Hey Quainex take a screenshot", "take a screenshot"),
        ("QUAINEX lock the screen", "lock the screen"),
    ],
)
def test_wake_word_is_detected_and_stripped(said, expected_command):
    match = detect_wake_word(said, "quainex", 0.75)
    assert match.detected is True
    assert match.command == expected_command


@pytest.mark.parametrize("misheard", ["quinex", "quaynex", "Quain-ex", "kwainex"])
def test_near_misses_still_wake_it(misheard):
    # Recognisers mangle unusual proper nouns; exact matching would leave the
    # assistant deaf to its own name.
    match = detect_wake_word(f"{misheard} open vs code", "quainex", 0.75)
    assert match.detected is True


@pytest.mark.parametrize(
    "said",
    [
        "",
        "open vs code",
        "what time is it",
        "let us talk about the weather instead of anything else quainex",
    ],
)
def test_unaddressed_speech_does_not_wake_it(said):
    # The last case has the wake word too late to be an address.
    assert detect_wake_word(said, "quainex", 0.75).detected is False


@pytest.mark.parametrize(
    "word",
    [
        "connect",
        "context",
        "quiet",
        "queen",
        "index",
        "quick",
        "phoenix",
        "kubernetes",
        "annex",
        "unique",
        "equinox",
    ],
)
def test_similar_sounding_words_do_not_false_trigger(word):
    # The near neighbours most likely to appear in ordinary speech. "equinox"
    # is the load-bearing case: it scores 0.714 against the wake word, exactly
    # the same as the genuine mishearing "kwainex". That tie is why homophones
    # are handled by phonetic folding rather than by lowering this threshold.
    assert detect_wake_word(f"{word} open vs code", "quainex", 0.75).detected is False


# -- the turn loop ---------------------------------------------------------


async def test_addressed_utterance_runs_the_full_pipeline(tmp_path):
    desktop = FakeDesktopController()
    tts = FakeTTS()
    session = _session(tmp_path, tts=tts, desktop=desktop)

    turn = await session.handle_transcript(Transcript(text="quainex open vs code"))

    assert turn.wake_word_detected is True
    assert turn.command_text == "open vs code"
    assert turn.intent is not None
    assert turn.result is not None
    assert turn.result.executed is True
    assert desktop.actions == ["open_application"]
    assert tts.spoken == [turn.result.message]


async def test_unaddressed_speech_stops_before_the_brain(tmp_path):
    desktop = FakeDesktopController()
    tts = FakeTTS()
    session = _session(tmp_path, tts=tts, desktop=desktop)

    turn = await session.handle_transcript(Transcript(text="open vs code"))

    assert turn.wake_word_detected is False
    assert turn.intent is None, "ambient speech must not reach the classifier"
    assert turn.result is None
    assert desktop.calls == [], "ambient speech must not reach the executor"
    assert tts.spoken == []


async def test_wake_word_can_be_waived_for_push_to_talk(tmp_path):
    desktop = FakeDesktopController()
    session = _session(tmp_path, desktop=desktop)

    turn = await session.handle_transcript(Transcript(text="open vs code"), require_wake_word=False)

    assert turn.intent is not None
    assert desktop.actions == ["open_application"]


async def test_name_with_no_request_is_answered_not_classified(tmp_path):
    desktop = FakeDesktopController()
    tts = FakeTTS()
    session = _session(tmp_path, tts=tts, desktop=desktop)

    turn = await session.handle_transcript(Transcript(text="quainex"))

    assert turn.wake_word_detected is True
    assert turn.intent is None
    assert desktop.calls == []
    assert turn.spoken_response is not None
    assert tts.spoken == [turn.spoken_response]


async def test_confirmation_still_gates_voice_commands(tmp_path):
    # Speaking a command must not bypass the Phase 3 safety gates.
    desktop = FakeDesktopController()
    session = _session(
        tmp_path, desktop=desktop, intent=IntentType.SHUTDOWN, allow_destructive_commands=True
    )

    turn = await session.handle_transcript(Transcript(text="quainex shut down the pc"))

    assert turn.result is not None
    assert turn.result.executed is False
    assert turn.result.status.value == "requires_confirmation"
    assert desktop.calls == []


async def test_speech_can_be_suppressed(tmp_path):
    tts = FakeTTS()
    session = _session(tmp_path, tts=tts)

    await session.handle_transcript(Transcript(text="quainex open vs code"), speak=False)

    assert tts.spoken == []


async def test_tts_disabled_setting_is_respected(tmp_path):
    tts = FakeTTS()
    session = _session(tmp_path, tts=tts, tts_enabled=False)

    await session.say("anything at all")

    assert tts.spoken == []


# -- microphone path -------------------------------------------------------


async def test_listen_records_transcribes_and_acts(tmp_path):
    recorder = FakeRecorder()
    stt = FakeSTT("quainex open vs code")
    desktop = FakeDesktopController()
    session = _session(tmp_path, recorder=recorder, stt=stt, desktop=desktop)

    turn = await session.listen_and_respond()

    assert recorder.calls == 1
    assert stt.calls == 1
    assert desktop.actions == ["open_application"]
    assert turn.result is not None


async def test_listen_without_a_microphone_is_an_actionable_error(tmp_path):
    session = _session(tmp_path, recorder=FakeRecorder(available=False))

    with pytest.raises(SpeechUnavailableError, match="voice"):
        await session.listen_and_respond()


async def test_recordings_are_deleted_unless_kept(tmp_path):
    session = _session(tmp_path, keep_recordings=False)
    await session.listen_and_respond()

    kept = list((tmp_path / "logs" / "recordings").glob("*.wav"))
    assert kept == [], "audio of the room must not accumulate unasked"


async def test_recordings_are_kept_when_requested(tmp_path):
    session = _session(tmp_path, keep_recordings=True)
    await session.listen_and_respond()

    kept = list((tmp_path / "logs" / "recordings").glob("*.wav"))
    assert len(kept) == 1


# -- status ----------------------------------------------------------------


def test_status_reports_each_component_separately(tmp_path):
    # Voice degrades in pieces: speech output can work while recognition cannot.
    session = _session(tmp_path, stt=FakeSTT(available=False), tts=FakeTTS(available=True))
    status = session.status()

    assert status["speech_to_text"] is False
    assert status["text_to_speech"] is True
    assert status["fully_available"] is False
    assert status["wake_word"] == "quainex"


# -- transcription errors --------------------------------------------------


async def test_transcribing_a_missing_file_is_an_error(tmp_path):
    from quainex.core.voice.stt import FasterWhisperSTT

    stt = FasterWhisperSTT(_settings(tmp_path))
    with pytest.raises(SpeechError, match="No audio file"):
        stt.transcribe(tmp_path / "nope.wav")


def test_whisper_model_is_not_loaded_at_construction(tmp_path):
    from quainex.core.voice.stt import FasterWhisperSTT

    # Constructing must never trigger a multi-hundred-megabyte download.
    stt = FasterWhisperSTT(_settings(tmp_path))
    assert stt.is_loaded is False


# -- HTTP endpoints --------------------------------------------------------


def _install_fake_voice(client: TestClient, session: VoiceSession) -> None:
    client.app.state.container.voice = session


def test_status_endpoint(client: TestClient, tmp_path):
    _install_fake_voice(client, _session(tmp_path))
    body = client.get("/voice/status").json()
    assert "speech_to_text" in body
    assert body["wake_word"] == "quainex"


def test_turn_endpoint_runs_the_pipeline(client: TestClient, tmp_path):
    desktop = FakeDesktopController()
    _install_fake_voice(client, _session(tmp_path, desktop=desktop))

    response = client.post("/voice/turn", json={"text": "quainex open vs code", "speak": False})
    assert response.status_code == 200

    body = response.json()
    assert body["wake_word_detected"] is True
    assert body["result"]["executed"] is True


def test_turn_endpoint_ignores_unaddressed_speech(client: TestClient, tmp_path):
    desktop = FakeDesktopController()
    _install_fake_voice(client, _session(tmp_path, desktop=desktop))

    body = client.post("/voice/turn", json={"text": "open vs code", "speak": False}).json()

    assert body["wake_word_detected"] is False
    assert body["intent"] is None
    assert desktop.calls == []


def test_empty_upload_is_rejected(client: TestClient, tmp_path):
    _install_fake_voice(client, _session(tmp_path))

    response = client.post("/voice/transcribe", files={"audio": ("empty.wav", b"", "audio/wav")})
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "speech_error"


def test_transcribe_endpoint_returns_a_transcript(client: TestClient, tmp_path):
    _install_fake_voice(client, _session(tmp_path, stt=FakeSTT("hello there")))

    response = client.post(
        "/voice/transcribe",
        files={"audio": ("clip.wav", b"RIFFfake-wav-bytes", "audio/wav")},
    )
    assert response.status_code == 200
    assert response.json()["text"] == "hello there"
