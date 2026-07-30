"""Microphone capture.

Purpose:
    Record a spoken utterance and stop when the speaker stops, without the user
    having to press anything to end the recording.

Why stop on silence rather than a fixed duration:
    A fixed window either truncates "open visual studio code and then check my
    calendar" or makes "yes" take five seconds. Trailing silence is the signal
    that a sentence finished, so recording ends a beat after speech does.

    Detection is a plain RMS energy gate rather than a neural VAD: it needs no
    model, adds no latency, and the failure mode is benign — a noisy room ends
    the recording late, which Whisper then trims anyway.

Architecture:
    MicrophoneRecorder.record()
        -> sounddevice InputStream (16 kHz mono int16 — Whisper's native format)
        -> per-block RMS
             |-- above threshold -> speech; reset the silence timer
             +-- below threshold -> silence; stop once it persists
        -> WAV file on disk -> SpeechToText

Dependencies:
    sounddevice + numpy (optional extra), standard-library ``wave``

Future improvements:
    * Calibrate the threshold from ambient noise at startup instead of a constant.
    * Stream blocks to the transcriber so recognition overlaps with speaking.
"""

from __future__ import annotations

import wave
from typing import TYPE_CHECKING, Protocol

from quainex.core.exceptions import NoSpeechError, SpeechError, SpeechUnavailableError
from quainex.core.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from quainex.config.settings import Settings

_log = get_logger(__name__)

#: Whisper resamples everything to 16 kHz mono; recording there avoids a
#: conversion and keeps files small.
SAMPLE_RATE = 16_000
CHANNELS = 1
_SAMPLE_WIDTH_BYTES = 2  # int16

#: Audio is examined in 100 ms blocks — long enough for a stable energy reading,
#: short enough that end-of-speech is detected promptly.
_BLOCK_MS = 100


class AudioRecorder(Protocol):
    """Captures spoken audio to a file."""

    @property
    def is_available(self) -> bool:
        """Whether a microphone can currently be used."""
        ...

    def record(self, destination: Path) -> Path:
        """Record until the speaker stops, writing a WAV file."""
        ...


class MicrophoneRecorder:
    """Records from the default input device, stopping on trailing silence."""

    def __init__(self, settings: Settings) -> None:
        """Construct the recorder without opening the device.

        Args:
            settings: Configuration supplying duration and silence thresholds.
        """
        self._settings = settings
        # Enumerating audio devices talks to the driver and can take seconds on
        # some machines. Memoised because it is probed on every status check.
        self._available: bool | None = None

    @property
    def is_available(self) -> bool:
        """Whether ``sounddevice`` is importable and an input device exists.

        Memoised: hot-plugging a microphone mid-session will not be noticed until
        restart, which is a fair trade for not stalling every status request.
        """
        if self._available is None:
            self._available = self._probe()
        return self._available

    @staticmethod
    def _probe() -> bool:
        """Test for a usable input device.

        Returns:
            Whether audio capture is possible.
        """
        try:
            import sounddevice
        except (ImportError, OSError):
            # OSError covers a missing PortAudio shared library.
            return False
        try:
            return any(
                device.get("max_input_channels", 0) > 0 for device in sounddevice.query_devices()
            )
        except Exception:
            # Device enumeration is driver-dependent and can fail in many ways;
            # any failure means we cannot promise a microphone.
            return False

    def record(self, destination: Path) -> Path:
        """Record an utterance and write it to ``destination``.

        Recording ends when speech has been followed by
        ``voice_silence_seconds`` of quiet, or when ``voice_max_seconds`` is
        reached — the latter bounds a stuck-open microphone.

        Args:
            destination: Where to write the WAV file.

        Returns:
            The path written.

        Raises:
            SpeechUnavailableError: No microphone support is available.
            SpeechError: Recording failed, or captured nothing.
        """
        try:
            import numpy as np
            import sounddevice as sd
        except (ImportError, OSError) as exc:
            raise SpeechUnavailableError(
                'Microphone capture is not installed. Run: pip install -e ".[voice]"'
            ) from exc

        block_frames = int(SAMPLE_RATE * _BLOCK_MS / 1000)
        max_blocks = int(self._settings.voice_max_seconds * 1000 / _BLOCK_MS)
        silence_blocks_needed = int(self._settings.voice_silence_seconds * 1000 / _BLOCK_MS)

        collected: list[bytes] = []
        silent_run = 0
        heard_speech = False

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                blocksize=block_frames,
            ) as stream:
                _log.info("recording_started", max_seconds=self._settings.voice_max_seconds)
                for _ in range(max_blocks):
                    block, overflowed = stream.read(block_frames)
                    if overflowed:
                        # Dropped samples degrade the transcript but do not
                        # invalidate it; note it and keep going.
                        _log.warning("audio_input_overflow")

                    collected.append(bytes(block))

                    # RMS over the block. float64 avoids int16 overflow when squaring.
                    samples = np.frombuffer(bytes(block), dtype=np.int16).astype(np.float64)
                    rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0

                    if rms >= self._settings.voice_silence_threshold:
                        heard_speech = True
                        silent_run = 0
                    elif heard_speech:
                        silent_run += 1
                        if silent_run >= silence_blocks_needed:
                            break
        except Exception as exc:
            raise SpeechError(f"Could not record from the microphone: {exc}") from exc

        if not heard_speech:
            # A distinct type, not a message: the always-on listener has to tell
            # "the room was quiet" apart from "the microphone broke", and it must
            # not do that by matching prose.
            raise NoSpeechError("No speech was detected.")

        return self._write_wav(destination, b"".join(collected))

    @staticmethod
    def _write_wav(destination: Path, frames: bytes) -> Path:
        """Write raw PCM frames as a WAV file.

        Args:
            destination: Where to write.
            frames: Raw 16-bit mono PCM data.

        Returns:
            The path written.

        Raises:
            SpeechError: The file could not be written.
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with wave.open(str(destination), "wb") as handle:
                handle.setnchannels(CHANNELS)
                handle.setsampwidth(_SAMPLE_WIDTH_BYTES)
                handle.setframerate(SAMPLE_RATE)
                handle.writeframes(frames)
        except (OSError, wave.Error) as exc:
            raise SpeechError(f"Could not save the recording: {exc}") from exc

        seconds = len(frames) / (SAMPLE_RATE * _SAMPLE_WIDTH_BYTES * CHANNELS)
        _log.info("recording_saved", seconds=round(seconds, 2), path=str(destination))
        return destination
