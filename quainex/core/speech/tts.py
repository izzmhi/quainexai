"""Text-to-speech.

Purpose:
    Give Quainex a voice, without adding a dependency or requiring a download.

Why Windows SAPI over a Python TTS package:
    ``pyttsx3`` and friends wrap the same Windows speech engine that PowerShell
    can reach directly, so the package buys nothing but an extra dependency and
    a COM initialisation quirk in async contexts. SAPI is present on every
    Windows install, works offline, and speaks immediately — no model download,
    which matters on a machine where large transfers are unreliable.

    The Protocol keeps a cloud voice (ElevenLabs, Azure) a drop-in away when
    quality matters more than latency and privacy.

Security note:
    The text spoken is model output, and it is never interpolated into the
    PowerShell script. The script is a fixed file that reads the text from a
    temporary file whose path we control, so no quantity of quotes, backticks or
    ``$(...)`` in a response can escape into the shell.

Architecture:
    VoiceSession -> TextToSpeech (Protocol) -> WindowsSapiTTS
                                                    -> powershell -File speak.ps1
                                                         -> System.Speech.Synthesis

Dependencies:
    Standard library only.

Future improvements:
    * Stream audio to the WebSocket so the phone client can hear responses.
    * Cache synthesised audio for repeated phrases ("Confirmed", "Done").
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from quainex.core.exceptions import SpeechError
from quainex.core.logging import get_logger

if TYPE_CHECKING:
    from quainex.config.settings import Settings

_log = get_logger(__name__)

#: Speaking a long paragraph can legitimately take a while; a minute is a
#: generous ceiling for anything Quainex should ever say in one breath.
_SPEAK_TIMEOUT = 60

#: The synthesis script. Text and paths arrive as arguments, never as inlined
#: script text, so nothing in the spoken content can be interpreted as code.
_SPEAK_SCRIPT = """
param(
    [Parameter(Mandatory=$true)][string]$TextPath,
    [string]$WavPath = '',
    [int]$Rate = 0,
    [string]$VoiceName = ''
)
Add-Type -AssemblyName System.Speech
$text = Get-Content -Raw -LiteralPath $TextPath
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Rate = $Rate
if ($VoiceName -ne '') { try { $synth.SelectVoice($VoiceName) } catch { } }
if ($WavPath -ne '') { $synth.SetOutputToWaveFile($WavPath) }
$synth.Speak($text)
$synth.Dispose()
"""


class TextToSpeech(Protocol):
    """Turns text into speech, either aloud or into a file."""

    @property
    def name(self) -> str:
        """Short identifier for this engine."""
        ...

    @property
    def is_available(self) -> bool:
        """Whether the engine can currently synthesise."""
        ...

    def speak(self, text: str) -> None:
        """Say ``text`` aloud on the local machine, blocking until finished."""
        ...

    def synthesise(self, text: str, destination: Path) -> Path:
        """Render ``text`` to a WAV file and return its path."""
        ...


class WindowsSapiTTS:
    """Speech synthesis through the built-in Windows speech engine."""

    def __init__(self, settings: Settings) -> None:
        """Construct the engine.

        Args:
            settings: Configuration supplying speaking rate and voice name.
        """
        self._settings = settings

    @property
    def name(self) -> str:
        """Short identifier for this engine."""
        return "windows-sapi"

    @property
    def is_available(self) -> bool:
        """Whether synthesis is possible.

        Always true on Windows: SAPI ships with the OS, so there is nothing to
        install and nothing to download.
        """
        return True

    def speak(self, text: str) -> None:
        """Say ``text`` aloud, blocking until speech finishes.

        Args:
            text: What to say.

        Raises:
            SpeechError: The speech engine failed.
        """
        self._run(text, wav_path=None)
        _log.info("speech_spoken", characters=len(text))

    def synthesise(self, text: str, destination: Path) -> Path:
        """Render ``text`` to a WAV file.

        Args:
            text: What to say.
            destination: Where to write the audio.

        Returns:
            The path written.

        Raises:
            SpeechError: The speech engine failed.
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._run(text, wav_path=destination)
        _log.info("speech_synthesised", characters=len(text), path=str(destination))
        return destination

    # -- internals --------------------------------------------------------

    def _run(self, text: str, wav_path: Path | None) -> None:
        """Invoke the synthesis script.

        Args:
            text: Content to speak.
            wav_path: Optional file to render into instead of the speakers.

        Raises:
            SpeechError: The text is empty, or synthesis failed.
        """
        content = text.strip()
        if not content:
            raise SpeechError("There is nothing to say.")

        from quainex.core.automation.windows import _system_executable

        with tempfile.TemporaryDirectory(prefix="quainex-tts-") as tmp:
            workspace = Path(tmp)
            text_file = workspace / "utterance.txt"
            script_file = workspace / "speak.ps1"
            text_file.write_text(content, encoding="utf-8")
            script_file.write_text(_SPEAK_SCRIPT, encoding="utf-8")

            argv = [
                _system_executable("powershell.exe"),
                "-NoProfile",
                "-NonInteractive",
                # Bypass applies to this one invocation of a script we just wrote
                # ourselves; it does not change the machine's policy.
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script_file),
                "-TextPath",
                str(text_file),
                "-Rate",
                str(self._settings.tts_rate),
            ]
            if wav_path is not None:
                argv += ["-WavPath", str(wav_path)]
            if self._settings.tts_voice:
                argv += ["-VoiceName", self._settings.tts_voice]

            try:
                result = subprocess.run(  # noqa: S603 - fixed argv, absolute exe, no shell
                    argv,
                    capture_output=True,
                    text=True,
                    timeout=_SPEAK_TIMEOUT,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise SpeechError(f"Speech synthesis failed: {exc}") from exc

            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "").strip()[:200]
                raise SpeechError(f"Speech synthesis failed. {detail}".strip())
