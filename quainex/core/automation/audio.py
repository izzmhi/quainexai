"""System volume, set to an exact level.

Purpose:
    Make "set volume to 30" actually set it to 30. The media keys can only nudge
    up and down a fixed step at a time, so a percentage was impossible — which is
    the bug this fixes.

Why pycaw:
    Setting an absolute level needs the Core Audio ``IAudioEndpointVolume``
    interface, which is COM. pycaw wraps it, and its only dependency —
    ``comtypes`` — is already present for the webcam. Small, no native build.

Graceful absence:
    Without pycaw the module reports unavailable and the controller falls back to
    nudging with the media keys, so up/down and mute still work. Only exact
    percentages need this, and a machine that cannot install it degrades to what
    it had before rather than losing volume control entirely.

Architecture:
    set_level(30)   -> IAudioEndpointVolume.SetMasterVolumeLevelScalar(0.30)
    nudge(+10)      -> get current scalar, add, clamp, set
    set_mute(True)  -> IAudioEndpointVolume.SetMute(1)

Dependencies:
    pycaw, comtypes

Example:
    >>> set_level(30)  # doctest: +SKIP
    30

Future improvements:
    * Per-application volume, which the same interface exposes.
"""

from __future__ import annotations

from typing import Any

from quainex.core.logging import get_logger

_log = get_logger(__name__)


def is_available() -> bool:
    """Whether exact volume control can run on this machine.

    Returns:
        Whether pycaw imports.
    """
    import importlib.util

    return importlib.util.find_spec("pycaw") is not None


def _endpoint() -> Any:
    """Return the default speaker's volume interface.

    Typed ``Any`` because pycaw ships no annotations; the COM interface's methods
    (``SetMasterVolumeLevelScalar`` and friends) are not visible to the checker.

    Returns:
        The ``IAudioEndpointVolume`` for the default output device.
    """
    from pycaw.pycaw import AudioUtilities

    return AudioUtilities.GetSpeakers().EndpointVolume


def set_level(percent: int) -> int:
    """Set the master volume to an absolute percentage.

    Args:
        percent: Target level, clamped to 0-100.

    Returns:
        The level that was set.
    """
    clamped = max(0, min(100, percent))
    endpoint = _endpoint()
    endpoint.SetMasterVolumeLevelScalar(clamped / 100.0, None)
    # Setting a level above zero on a muted device leaves it silent, which reads
    # as "the command did nothing". Unmute whenever a positive level is set.
    if clamped > 0 and endpoint.GetMute():
        endpoint.SetMute(0, None)
    _log.info("volume_set", percent=clamped)
    return clamped


def nudge(delta: int) -> int:
    """Change the volume by a relative amount.

    Args:
        delta: Percentage points to add (negative to lower).

    Returns:
        The resulting level.
    """
    endpoint = _endpoint()
    current = round(endpoint.GetMasterVolumeLevelScalar() * 100)
    return set_level(current + delta)


def set_mute(*, muted: bool) -> bool:
    """Mute or unmute the master output.

    Args:
        muted: ``True`` to mute, ``False`` to unmute.

    Returns:
        The mute state that was set.
    """
    _endpoint().SetMute(1 if muted else 0, None)
    _log.info("volume_mute", muted=muted)
    return muted
