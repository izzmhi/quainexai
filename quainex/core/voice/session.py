"""Voice conversation orchestrator.

Purpose:
    Join the pieces into the loop the user actually experiences: say something,
    have it understood, have it done, hear the answer.

Architecture:
    microphone -> MicrophoneRecorder.record()
               -> SpeechToText.transcribe()
               -> wake-word gate  ── not addressed to Quainex? stop here
               -> Brain.interpret()          (Phase 2)
               -> CommandExecutor.execute()  (Phase 3)
               -> TextToSpeech.speak()
               -> VoiceTurn (the whole trace, for the caller and the audit log)

Why the wake word is matched fuzzily:
    Speech recognition mangles unusual proper nouns — "Quainex" comes back as
    "Quinex", "Quain X", "Wayne X". Exact matching would leave the assistant deaf
    to its own name. A similarity ratio over the opening words accepts the near
    misses while still ignoring speech that was not addressed to it.

Why the gate is a gate and not a filter:
    When no wake word is present the turn stops *before* the Brain is called. No
    API spend, no classification, and — more importantly — nothing the room
    happens to say can reach the command executor.

Dependencies:
    quainex.core.{brain,commands,speech,voice}, quainex.config.settings

Future improvements:
    * A continuous background listener, so the wake word works without a request.
    * Barge-in: stop speaking when the user starts talking over the response.
"""

from __future__ import annotations

import difflib
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import anyio.to_thread
from pydantic import BaseModel

# Imported at runtime, not under TYPE_CHECKING: `VoiceTurn` is a Pydantic model
# whose fields are typed with these, and Pydantic resolves field annotations
# when the schema is built. Deferring them makes the model "not fully defined"
# and breaks OpenAPI generation at request time rather than at import time.
from quainex.core.brain import Intent
from quainex.core.commands import CommandResult
from quainex.core.exceptions import SpeechUnavailableError
from quainex.core.logging import get_logger
from quainex.core.voice.stt import Transcript

if TYPE_CHECKING:
    from quainex.config.settings import Settings
    from quainex.core.brain import Brain
    from quainex.core.commands import CommandExecutor
    from quainex.core.memory import MemoryManager
    from quainex.core.speech.tts import TextToSpeech
    from quainex.core.voice.audio import AudioRecorder
    from quainex.core.voice.stt import SpeechToText

_log = get_logger(__name__)

#: Spoken when the wake word arrives with no request attached.
#:
#: Deliberately one word. This is the only signal that Quainex is awake and
#: listening, so it has to be instant and unmistakable — a sentence spoken back
#: takes longer than the request it is inviting.
ACKNOWLEDGEMENT = "Yes?"

#: How many leading words may contain the wake word. Addressing something is
#: something people do at the start of a sentence; scanning the whole utterance
#: would let any passing mention trigger it.
_WAKE_SCAN_WORDS = 4

#: Spelling variants that sound identical, folded before comparison.
#:
#: These exist because a similarity threshold alone cannot do this job:
#: "kwainex" (a plausible transcription of "Quainex", since "kw" and "qu" are
#: the same sound) and "equinox" (an ordinary English word) both score 0.714
#: against the wake word. Any threshold that accepts the first accepts the
#: second. Folding the sound makes "kwainex" an exact match while leaving
#: "equinox" exactly where it was — below the bar.
_PHONETIC_FOLDINGS: tuple[tuple[str, str], ...] = (
    ("kw", "qu"),  # kwainex -> quainex
    ("cw", "qu"),  # cwainex -> quainex
    ("ks", "x"),  # quaineks -> quainex
)


#: Longest trailing fragment that may be joined to a candidate wake word.
#:
#: When the recogniser splits the name it leaves a stub — "Quain X", "Quain ex",
#: "Quain nex" — never a whole word. Three characters admits every observed stub
#: and excludes real words, which matters because joining greedily would let
#: "Quain export the file" match as "quainexport" and swallow "export".
_MAX_SPLIT_FRAGMENT = 3


def _fold_phonetics(word: str) -> str:
    """Normalise spellings that sound the same.

    Args:
        word: A lower-cased, alphanumeric-only word.

    Returns:
        The word with homophone digraphs folded to one spelling.
    """
    for variant, canonical in _PHONETIC_FOLDINGS:
        word = word.replace(variant, canonical)
    return word


def _alnum(word: str) -> str:
    """Reduce a token to lower-case letters and digits.

    Args:
        word: A raw token from the transcript.

    Returns:
        The token without punctuation the recogniser added.
    """
    return "".join(character for character in word.lower() if character.isalnum())


def _similarity(candidate: str, folded_target: str) -> float:
    """Score a candidate against the folded wake word.

    Args:
        candidate: An alphanumeric, lower-cased candidate.
        folded_target: The wake word, already phonetically folded.

    Returns:
        A ratio from 0 to 1.
    """
    return difflib.SequenceMatcher(None, _fold_phonetics(candidate), folded_target).ratio()


class WakeWordMatch(BaseModel):
    """Result of testing an utterance for the wake word.

    Attributes:
        detected: Whether Quainex was addressed.
        command: The utterance with the wake word removed.
        matched_word: The word that matched, useful when debugging mishearings.
    """

    detected: bool
    command: str
    matched_word: str | None = None


class VoiceTurn(BaseModel):
    """Everything that happened in one spoken exchange.

    Attributes:
        transcript: What Quainex heard.
        wake_word_detected: Whether it was addressed.
        command_text: The utterance after removing the wake word.
        intent: What it understood, when it got that far.
        result: What it did, when it got that far.
        spoken_response: What it said back, if anything.
    """

    transcript: Transcript
    wake_word_detected: bool
    command_text: str
    intent: Intent | None = None
    result: CommandResult | None = None
    spoken_response: str | None = None


def detect_wake_word(text: str, wake_word: str, similarity: float) -> WakeWordMatch:
    """Test whether an utterance is addressed to Quainex.

    Args:
        text: The transcript.
        wake_word: The configured name to listen for.
        similarity: Minimum ratio (0-1) for a word to count as a match.

    Returns:
        Whether the wake word was found, and the utterance with it removed.
    """
    words = text.split()
    if not words:
        return WakeWordMatch(detected=False, command="")

    target = _fold_phonetics(wake_word.lower().strip())

    for index, word in enumerate(words[:_WAKE_SCAN_WORDS]):
        # Strip punctuation the recogniser added: "Quainex," should still match.
        cleaned = _alnum(word)
        if not cleaned:
            continue

        ratio = _similarity(cleaned, target)
        consumed = 1

        # The recogniser splits an invented proper noun across tokens. Real
        # observed output for "Quainex, take a screenshot" is "Quain X. Take a
        # screenshot." — "Quain" alone already scores 0.83, so the loop matched,
        # stopped, and left "X." at the front of the command. The Brain absorbed it
        # that time; "Quain X. open Notepad" is the case where a stray token
        # becomes part of the extracted target.
        #
        # So the join is tried too, and wins only when it genuinely fits better.
        if index + 1 < len(words) and (tail := _alnum(words[index + 1])):
            # Bounded to a short fragment. Without the limit, "Quain export the
            # file" would join to "quainexport", score 0.78, and eat a real word —
            # a split name leaves a stub behind, not a whole one.
            if len(tail) <= _MAX_SPLIT_FRAGMENT:
                joined = _similarity(cleaned + tail, target)
                if joined > ratio:
                    ratio, consumed = joined, 2

        if ratio >= similarity:
            remainder = " ".join(words[index + consumed :]).strip()
            # Drop a leading comma left behind by "Quainex, open VS Code".
            return WakeWordMatch(
                detected=True,
                command=remainder.lstrip(",.: ").strip(),
                matched_word=" ".join(words[index : index + consumed]),
            )

    return WakeWordMatch(detected=False, command=text.strip())


class VoiceSession:
    """Runs the listen, understand, act, respond loop."""

    def __init__(
        self,
        *,
        stt: SpeechToText,
        tts: TextToSpeech,
        recorder: AudioRecorder,
        brain: Brain,
        commands: CommandExecutor,
        settings: Settings,
        memory: MemoryManager | None = None,
    ) -> None:
        """Construct the session.

        Args:
            stt: Speech recogniser.
            tts: Speech synthesiser.
            recorder: Microphone capture.
            brain: Intent classifier.
            commands: Command executor.
            settings: Application configuration.
            memory: Optional memory. When absent the loop still works, but each
                utterance is understood in isolation — "close it" has no referent.
        """
        self._stt = stt
        self._tts = tts
        self._recorder = recorder
        self._brain = brain
        self._commands = commands
        self._settings = settings
        self._memory = memory

    @property
    def is_available(self) -> bool:
        """Whether a full spoken exchange is currently possible."""
        return self._stt.is_available and self._recorder.is_available

    def status(self) -> dict[str, object]:
        """Report which parts of the voice subsystem are usable.

        Voice degrades in pieces — speech output works with no download, while
        recognition needs a model — so status is reported per component rather
        than as one flag.

        Returns:
            Availability of each component.
        """
        return {
            "microphone": self._recorder.is_available,
            "speech_to_text": self._stt.is_available,
            "text_to_speech": self._tts.is_available,
            "engine": self._stt.name,
            "wake_word": self._settings.wake_word,
            "fully_available": self.is_available,
        }

    async def transcribe(self, audio_path: Path) -> Transcript:
        """Transcribe an audio file off the event loop.

        Args:
            audio_path: Recording to transcribe.

        Returns:
            The transcript.
        """
        return await anyio.to_thread.run_sync(self._stt.transcribe, audio_path)

    async def say(self, text: str) -> None:
        """Speak text aloud, if speech output is enabled.

        Args:
            text: What to say.
        """
        if not self._settings.tts_enabled or not self._tts.is_available:
            return
        await anyio.to_thread.run_sync(self._tts.speak, text)

    async def handle_transcript(
        self,
        transcript: Transcript,
        *,
        require_wake_word: bool | None = None,
        confirmed: bool = False,
        speak: bool = True,
    ) -> VoiceTurn:
        """Take a transcript through the wake gate, the Brain and the executor.

        Args:
            transcript: What was heard.
            require_wake_word: Override the configured wake-word requirement.
            confirmed: Whether the user has pre-approved the resulting action.
            speak: Whether to voice the response.

        Returns:
            The complete trace of the turn.
        """
        gated = (
            require_wake_word
            if require_wake_word is not None
            else (self._settings.voice_require_wake_word)
        )
        match = detect_wake_word(
            transcript.text,
            self._settings.wake_word,
            self._settings.wake_word_similarity,
        )

        if gated and not match.detected:
            # Not addressed to Quainex: stop before the Brain is called, so
            # ambient conversation costs nothing and can reach nothing.
            _log.info("wake_word_absent", characters=len(transcript.text))
            return VoiceTurn(
                transcript=transcript,
                wake_word_detected=False,
                command_text=match.command,
            )

        command_text = match.command if match.detected else transcript.text.strip()
        if not command_text:
            # Short, and an invitation rather than a complaint.
            #
            # This is the only moment Quainex proves it is awake, and the wording
            # carries real weight: "I heard my name, but no request" reports a
            # deficiency, which reads as a rebuke for saying its name. One word that
            # hands the turn back is what an assistant does — and it is also the
            # cue the listener uses to open a follow-up window, so this reply is
            # functional, not decorative.
            response = ACKNOWLEDGEMENT
            if speak:
                await self.say(response)
            return VoiceTurn(
                transcript=transcript,
                wake_word_detected=match.detected,
                command_text="",
                spoken_response=response,
            )

        # Prior turns give pronouns a referent: "close it" only means anything
        # if Quainex remembers what it just opened.
        history = await self._memory.conversation_context() if self._memory else None

        intent = await self._brain.interpret(command_text, history=history)
        result = await self._commands.execute(intent, confirmed=confirmed)

        if self._memory is not None:
            await self._memory.remember_exchange(command_text, intent, result)

        if speak:
            await self.say(result.message)

        _log.info(
            "voice_turn_completed",
            wake_word_detected=match.detected,
            intent=intent.intent.value,
            status=result.status.value,
            executed=result.executed,
        )
        return VoiceTurn(
            transcript=transcript,
            wake_word_detected=match.detected,
            command_text=command_text,
            intent=intent,
            result=result,
            spoken_response=result.message,
        )

    async def listen_and_respond(
        self,
        *,
        require_wake_word: bool | None = None,
        confirmed: bool = False,
    ) -> VoiceTurn:
        """Record from the microphone and run one complete exchange.

        Args:
            require_wake_word: Override the configured wake-word requirement.
            confirmed: Whether the user has pre-approved the resulting action.

        Returns:
            The complete trace of the turn.

        Raises:
            SpeechUnavailableError: Microphone or recognition support is missing.
        """
        if not self._recorder.is_available:
            raise SpeechUnavailableError(
                'No microphone is available. Run: pip install -e ".[voice]"'
            )

        with tempfile.TemporaryDirectory(prefix="quainex-voice-") as tmp:
            recording = Path(tmp) / "utterance.wav"
            await anyio.to_thread.run_sync(self._recorder.record, recording)
            transcript = await self.transcribe(recording)

            if self._settings.keep_recordings:
                kept = self._settings.log_dir / "recordings" / recording.name
                kept.parent.mkdir(parents=True, exist_ok=True)
                kept.write_bytes(recording.read_bytes())
                _log.info("recording_kept", path=str(kept))

        # Outside the temp directory: the audio is gone by here unless the user
        # explicitly asked for it to be kept.
        return await self.handle_transcript(
            transcript,
            require_wake_word=require_wake_word,
            confirmed=confirmed,
        )
