"""Text-to-speech synthesis.

Phase 4. Windows SAPI by default — offline, no download, no extra dependency —
behind a Protocol that a cloud voice can replace.
"""

from quainex.core.speech.tts import TextToSpeech, WindowsSapiTTS

__all__ = ["TextToSpeech", "WindowsSapiTTS"]
