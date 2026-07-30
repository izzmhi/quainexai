"""Report what the microphone is actually hearing.

Purpose:
    Answer "is it not hearing me, or is the mic not working?" — the first question
    when hands-free mode stays silent, and one the application itself cannot answer
    because a quiet room and a muted microphone look identical to it.

What it prints:
    The input device Quainex will use, then a live RMS level for a few seconds.
    Speak while it runs. If the level stays near zero the microphone is muted,
    disabled, or Windows has selected a different default input — none of which is
    a Quainex problem, and all of which are invisible from inside it.

    The configured silence threshold is printed alongside, because "is there
    signal" and "is there *enough* signal to count as speech" are different
    questions and the second is the one that governs the listener.

Dependencies:
    sounddevice, numpy (both from the voice extra)

Example:
    python scripts/check_microphone.py
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import TYPE_CHECKING

from quainex.config.settings import get_settings

if TYPE_CHECKING:
    from quainex.config.settings import Settings


def _listener_is_holding_the_microphone(settings: Settings) -> bool:
    """Whether always-on listening currently owns the input device.

    Args:
        settings: Configuration naming where the API listens.

    Returns:
        ``True`` when the listener reports itself running. A server that is not
        answering means nothing is holding the microphone, so this returns
        ``False`` rather than refusing to run.
    """
    url = f"http://{settings.host}:{settings.port}/voice/listener"
    try:
        # Loopback only: the URL is built from validated settings, not user input.
        with urllib.request.urlopen(url, timeout=2) as response:
            return bool(json.loads(response.read()).get("running"))
    except (urllib.error.URLError, OSError, ValueError):
        return False


def main() -> int:
    """Print the input device and a live level meter.

    Returns:
        Process exit code: 0 when speech-level audio was seen, 1 otherwise.
    """
    try:
        import numpy
        import sounddevice
    except ImportError:
        print('Voice support is not installed. Run: pip install -e ".[voice]"')
        return 1

    settings = get_settings()
    threshold = settings.voice_silence_threshold

    if _listener_is_holding_the_microphone(settings):
        # Without this check the script lies. Windows gives the second opener a
        # stream that reads silence rather than an error, so a *working* microphone
        # measures as dead purely because hands-free mode already has it. That is
        # exactly the wrong answer to give someone asking why it cannot hear them.
        print("Hands-free listening is running, and it holds the microphone.")
        print("This check would read silence no matter how loudly you spoke.")
        print()
        print("Turn hands-free off first (the button on the Console panel, or:")
        print("  curl -X POST http://127.0.0.1:8000/voice/listener/stop )")
        print("then run this again.")
        return 1

    try:
        device = sounddevice.query_devices(kind="input")
    except Exception as exc:
        # Any driver failure *is* the answer here, so the type does not matter.
        print(f"No input device is available: {exc}")
        return 1

    print(f"Input device      : {device['name']}")
    print(f"Silence threshold : {threshold:.0f} (RMS, 16-bit scale)")
    print(f"Wake word         : {settings.wake_word}")
    print()
    print("Speak now. Six seconds:")

    peak = 0.0
    seconds = 6
    rate = 16000

    with sounddevice.InputStream(samplerate=rate, channels=1, dtype="int16") as stream:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            block, overflowed = stream.read(int(rate * 0.3))
            if overflowed:
                # Worth saying: dropped input is a plausible cause of a missed wake
                # word, and it is not otherwise visible.
                print("  (input overflow - audio was dropped)")
            level = float(numpy.sqrt(numpy.mean(numpy.square(block.astype(numpy.float64)))))
            peak = max(peak, level)
            bars = "#" * min(int(level / 60), 40)
            state = "SPEECH" if level >= threshold else "quiet "
            print(f"  {state} {level:7.0f} {bars}")

    print()
    if peak < 10:
        print("The microphone produced almost no signal.")
        print("Check that it is not muted, and that Windows has the right default")
        print("input device selected: Settings > System > Sound > Input.")
        return 1
    if peak < threshold:
        print(f"Signal reached {peak:.0f}, below the {threshold:.0f} speech threshold.")
        print("The microphone works but is quiet. Either raise its input volume, or")
        print(f"lower QUAINEX_VOICE_SILENCE_THRESHOLD (try {max(peak * 0.6, 50):.0f}).")
        return 1

    print(f"Good: peaked at {peak:.0f}, above the {threshold:.0f} threshold.")
    print(f'Hands-free should hear you. Say: "{settings.wake_word}, take a screenshot"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
