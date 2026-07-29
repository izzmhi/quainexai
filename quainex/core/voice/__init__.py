"""Speech-to-text, microphone capture and the voice conversation loop.

Phase 4. Whisper for transcription, fuzzy wake-word gating, and the orchestrator
that joins recognition to the Brain, the command executor and speech output.
"""

from quainex.core.voice.audio import AudioRecorder, MicrophoneRecorder
from quainex.core.voice.session import (
    VoiceSession,
    VoiceTurn,
    WakeWordMatch,
    detect_wake_word,
)
from quainex.core.voice.stt import FasterWhisperSTT, SpeechToText, Transcript

__all__ = [
    "AudioRecorder",
    "FasterWhisperSTT",
    "MicrophoneRecorder",
    "SpeechToText",
    "Transcript",
    "VoiceSession",
    "VoiceTurn",
    "WakeWordMatch",
    "detect_wake_word",
]
