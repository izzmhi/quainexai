"""Speech-to-text.

Purpose:
    Turn recorded audio into text the Brain can classify.

Two decisions that shape this module:

**faster-whisper, not openai-whisper.** Both run the same model; the former uses
CTranslate2 rather than PyTorch, which means a ~40 MB dependency instead of a
~2.5 GB one and roughly 4x faster CPU inference. On a personal machine that has
to stay responsive while the user works, that is the difference between usable
and not.

**The model loads lazily, and its absence is not a startup failure.** Whisper
weights are hundreds of megabytes fetched on first use, and this machine has a
documented history of stalling large CDN transfers. If the import is missing or
the download never completes, ``is_available`` reports false and the rest of
Quainex carries on by text. Voice is a capability layered onto a working system,
not a prerequisite for one.

Architecture:
    audio file
        -> SpeechToText (Protocol)
             |-- FasterWhisperSTT   (implemented; lazy, local, offline once cached)
             +-- CloudSTT           (future: lower latency, higher accuracy, off-machine)
        -> Transcript -> Brain.interpret()

Dependencies:
    faster-whisper (optional extra), quainex.config.settings

Future improvements:
    * Streaming transcription so the Brain can start on a partial utterance.
    * Speaker identification, so Quainex can ignore voices that are not the user.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, Field

from quainex.core.exceptions import SpeechError, SpeechUnavailableError
from quainex.core.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from quainex.config.settings import Settings

_log = get_logger(__name__)


class Transcript(BaseModel):
    """The result of transcribing one recording.

    Attributes:
        text: What was said.
        language: Detected language code, when the engine reports one.
        duration_seconds: Length of the audio processed.
        confidence: Engine confidence 0-1, when reported.
    """

    text: str
    language: str | None = None
    duration_seconds: float = Field(default=0.0, ge=0)
    confidence: float | None = None


class SpeechToText(Protocol):
    """Transcribes recorded audio."""

    @property
    def name(self) -> str:
        """Short identifier for this engine."""
        ...

    @property
    def is_available(self) -> bool:
        """Whether transcription is currently possible."""
        ...

    def transcribe(self, audio_path: Path) -> Transcript:
        """Transcribe an audio file."""
        ...


class FasterWhisperSTT:
    """Local Whisper transcription via CTranslate2."""

    def __init__(self, settings: Settings) -> None:
        """Construct the engine without loading anything.

        Deliberately cheap: constructing this must not download a model or block
        application startup.

        Args:
            settings: Configuration supplying model size and compute type.
        """
        self._settings = settings
        self._model: Any | None = None
        # Model loading is not thread-safe and can take tens of seconds; a lock
        # stops two concurrent requests each triggering their own download.
        self._lock = threading.Lock()
        # Importing faster_whisper costs ~1s (it pulls in CTranslate2 and
        # tokenizers), so the availability probe is memoised. Whether the package
        # is installed cannot change while the process runs.
        self._available: bool | None = None

    @property
    def name(self) -> str:
        """Short identifier for this engine."""
        return f"faster-whisper/{self._settings.whisper_model}"

    @property
    def is_available(self) -> bool:
        """Whether the faster-whisper package is importable.

        Reports on the *dependency*, not on whether weights are cached — the
        first transcription may still need to download them.
        """
        if self._available is None:
            try:
                import faster_whisper  # noqa: F401
            except ImportError:
                self._available = False
            else:
                self._available = True
        return self._available

    @property
    def is_loaded(self) -> bool:
        """Whether the model is resident in memory and ready to transcribe."""
        return self._model is not None

    def transcribe(self, audio_path: Path) -> Transcript:
        """Transcribe an audio file to text.

        Args:
            audio_path: Path to the recording.

        Returns:
            The transcript.

        Raises:
            SpeechUnavailableError: faster-whisper is not installed, or the model
                could not be loaded or downloaded.
            SpeechError: The file is missing or could not be transcribed.
        """
        if not audio_path.is_file():
            raise SpeechError(f"No audio file at '{audio_path}'.")

        model = self._ensure_model()
        try:
            segments, info = model.transcribe(
                str(audio_path),
                beam_size=self._settings.whisper_beam_size,
                vad_filter=True,  # drop silence so a pause is not transcribed as noise
            )
            text = "".join(segment.text for segment in segments).strip()
        except Exception as exc:
            raise SpeechError(f"Could not transcribe the audio: {exc}") from exc

        transcript = Transcript(
            text=text,
            language=getattr(info, "language", None),
            duration_seconds=float(getattr(info, "duration", 0.0) or 0.0),
            confidence=self._language_confidence(info),
        )
        # The transcript text itself is not logged: it is whatever was said in
        # the room, which is not ours to record by default.
        _log.info(
            "audio_transcribed",
            characters=len(transcript.text),
            language=transcript.language,
            duration_seconds=round(transcript.duration_seconds, 2),
        )
        return transcript

    # -- internals --------------------------------------------------------

    def _ensure_model(self) -> Any:
        """Load the Whisper model, downloading weights on first use.

        Returns:
            The loaded model.

        Raises:
            SpeechUnavailableError: The package is missing, or loading failed.
        """
        # Read into a local first: the double-checked pattern below is a
        # concurrency guard, and reading the attribute twice would let a type
        # checker conclude the second check is dead code.
        loaded = self._model
        if loaded is not None:
            return loaded

        with self._lock:
            loaded = self._model
            if loaded is not None:  # another thread won the race
                return loaded

            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise SpeechUnavailableError(
                    'Speech recognition is not installed. Run: pip install -e ".[voice]"'
                ) from exc

            _log.info(
                "whisper_model_loading",
                model=self._settings.whisper_model,
                detail="First load downloads weights and may take several minutes.",
            )
            try:
                loaded = WhisperModel(
                    self._settings.whisper_model,
                    device="cpu",
                    compute_type=self._settings.whisper_compute_type,
                )
            except Exception as exc:
                raise SpeechUnavailableError(
                    f"Could not load the '{self._settings.whisper_model}' model. "
                    f"The weights download may have failed: {exc}"
                ) from exc

            self._model = loaded
            _log.info("whisper_model_loaded", model=self._settings.whisper_model)
            return loaded

    @staticmethod
    def _language_confidence(info: Any) -> float | None:
        """Extract the language-detection probability, if the engine reported one.

        Args:
            info: Engine-specific transcription metadata.

        Returns:
            A probability between 0 and 1, or ``None``.
        """
        probability = getattr(info, "language_probability", None)
        return float(probability) if isinstance(probability, (int, float)) else None
