"""Measure this microphone and set the speech threshold to match it.

Purpose:
    End the guessing about why the wake word never fires. The threshold that
    decides "this is speech" ships as one number, and one number cannot suit every
    microphone — a quiet laptop array and a desk condenser differ by more than an
    order of magnitude.

Why interactive:
    Diagnosing this remotely does not work. Measuring "ambient" and "speech" needs
    someone to be quiet and then talk *at the right moments*, and a script is the
    only thing that can ask. It also shows a live meter while you speak, which
    answers the real question — does this microphone react to my voice at all? —
    faster than any amount of reasoning about levels.

What it does:
    1. Measures the room with nobody speaking.
    2. Asks you to talk, and shows the level as you do.
    3. Picks a threshold with headroom above the room and below your voice.
    4. Prints the line to put in ``.env``, and offers to write it.

If speech and silence measure the same, the microphone is not capturing you, and
no threshold will fix that — the script says so rather than choosing a number that
cannot work.

Dependencies:
    sounddevice, numpy (from the voice extra)

Example:
    python scripts/calibrate_microphone.py
"""

from __future__ import annotations

import time

from quainex.config.settings import REPO_ROOT, get_settings

#: Sample rate to measure at. Matches what the recorder uses.
_RATE = 16000

#: Block size, in seconds. Short enough to feel live, long enough for a stable RMS.
_BLOCK = 0.15

#: How far above the room's peak a threshold must sit.
#:
#: The room's *peak* rather than its mean: a threshold below an occasional spike
#: makes the recorder start on a passing noise, capture nothing but that noise, and
#: hand Whisper an utterance it will hallucinate a short phrase from. Which is
#: exactly the symptom that started this.
_AMBIENT_MARGIN = 1.3

#: How far below speech the threshold must sit, so a quiet word still registers.
_SPEECH_MARGIN = 0.55


def _measure(label: str, seconds: float, *, live: bool) -> tuple[float, float]:
    """Record for a while and report peak and mean level.

    Args:
        label: What to print in front of the meter.
        seconds: How long to measure.
        live: Whether to draw a bar per block.

    Returns:
        ``(peak, mean)`` RMS on the 16-bit scale.
    """
    import numpy
    import sounddevice

    levels: list[float] = []
    with sounddevice.InputStream(samplerate=_RATE, channels=1, dtype="int16") as stream:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            block, _ = stream.read(int(_RATE * _BLOCK))
            level = float(numpy.sqrt(numpy.mean(numpy.square(block.astype(numpy.float64)))))
            levels.append(level)
            if live:
                bar = "#" * min(int(level / 25), 50)
                print(f"  {label} {level:7.0f} {bar}")

    return (max(levels), sum(levels) / len(levels)) if levels else (0.0, 0.0)


def _countdown(message: str, seconds: int) -> None:
    """Print a countdown so the user knows when to start.

    Args:
        message: What they should be doing.
        seconds: How long until measurement begins.
    """
    for remaining in range(seconds, 0, -1):
        print(f"  {message} in {remaining}...")
        time.sleep(1)


def main() -> int:
    """Calibrate and report.

    Returns:
        0 when a usable threshold was found, 1 otherwise.
    """
    try:
        import sounddevice
    except ImportError:
        print('Voice support is not installed. Run: pip install -e ".[voice]"')
        return 1

    settings = get_settings()

    try:
        device = sounddevice.query_devices(kind="input")
    except Exception as exc:
        print(f"No input device is available: {exc}")
        return 1

    print(f"Device            : {device['name']}")
    print(f"Current threshold : {settings.voice_silence_threshold:.0f}")
    print()
    print("Step 1 of 2: silence. Please do not speak or type.")
    _countdown("Measuring the room", 3)
    ambient_peak, ambient_mean = _measure("ambient", 4, live=False)
    print(f"  room: peak {ambient_peak:.0f}, average {ambient_mean:.0f}")
    print()

    print("Step 2 of 2: speech.")
    print('Say "Quainex, take a screenshot" over and over, normally, until it stops.')
    print("Watch the bars — if they do not move, the microphone is not hearing you.")
    _countdown("Starting", 3)
    speech_peak, speech_mean = _measure("speech ", 10, live=True)
    print()
    print(f"  speech: peak {speech_peak:.0f}, average {speech_mean:.0f}")
    print()

    # The mean is the honest comparison. A peak can be a chair creaking; a raised
    # *average* over ten seconds is speech and nothing else.
    if speech_mean < max(ambient_mean * 1.5, 2.0):
        print("The microphone is not capturing your voice.")
        print(f"Speaking averaged {speech_mean:.0f} against {ambient_mean:.0f} for an empty room —")
        print("indistinguishable, so there is no threshold that would work.")
        print()
        print("This is a Windows setting, not a Quainex one:")
        print("  Win+R -> mmsys.cpl -> Recording -> double-click your microphone")
        print("    Levels tab   : slider to 100, and Microphone Boost to +20 or +30 dB")
        print("    Advanced tab : untick 'Allow applications to take exclusive control'")
        return 1

    floor = ambient_peak * _AMBIENT_MARGIN
    ceiling = speech_peak * _SPEECH_MARGIN

    if floor >= ceiling:
        # The room is nearly as loud as the voice. Sit between them and say so.
        threshold = (floor + ceiling) / 2
        print(f"Tight margin: room peaks at {ambient_peak:.0f}, speech at {speech_peak:.0f}.")
        print("Raising the microphone level in Windows would widen this.")
    else:
        threshold = (floor + ceiling) / 2

    threshold = max(round(threshold), 5)
    print(f"Recommended threshold: {threshold}")
    print(f"  (room peaks at {ambient_peak:.0f}; your speech peaks at {speech_peak:.0f})")
    print()

    line = f"QUAINEX_VOICE_SILENCE_THRESHOLD={threshold}"
    answer = input(f"Write {line} to .env? [y/N] ").strip().lower()
    if answer != "y":
        print(f"Not written. Add this to .env yourself:\n  {line}")
        return 0

    if not _write_threshold(threshold):
        return 1

    print("Written. Restart Quainex for it to take effect:")
    print('  Stop-ScheduledTask -TaskName "Quainex Server"')
    print('  Start-ScheduledTask -TaskName "Quainex Server"')
    return 0


def _write_threshold(threshold: int) -> bool:
    """Replace or append the threshold in ``.env``.

    Rewrites the existing line rather than appending a second one: two assignments
    of the same variable is a file that behaves differently from how it reads.

    Args:
        threshold: The value to store.

    Returns:
        Whether the file was written.
    """
    path = REPO_ROOT / ".env"
    key = "QUAINEX_VOICE_SILENCE_THRESHOLD"

    if not path.exists():
        print(f"No .env at {path}. Copy .env.example to .env first.")
        return False

    lines = path.read_text(encoding="utf-8").splitlines()
    replaced = False
    for index, line in enumerate(lines):
        if line.strip().startswith(f"{key}="):
            lines[index] = f"{key}={threshold}"
            replaced = True
            break

    if not replaced:
        lines.append(f"{key}={threshold}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


if __name__ == "__main__":
    raise SystemExit(main())
