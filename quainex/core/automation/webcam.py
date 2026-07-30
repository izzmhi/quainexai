"""Webcam still capture.

Purpose:
    Take a photo from the built-in camera, so "send me a webcam picture" from a
    phone shows you who is in front of the machine — the anti-theft case, and the
    "is my delivery here yet" case.

Why pygrabber rather than OpenCV:
    OpenCV would work and is what most examples reach for, but it is a ~40 MB wheel
    that pulls a large native runtime, and this machine's downloads are unreliable.
    ``pygrabber`` drives DirectShow directly through ``comtypes`` — a small,
    pure-Python dependency — and Pillow, already present for screenshots, does the
    encoding. Nothing native to build, nothing large to fetch.

The warm-up frames are not optional:
    A camera's first frames after power-on are dark or green while the sensor sets
    its exposure and white balance. Grabbing frame zero produces a black rectangle
    that looks like a bug. So a few frames are pulled and discarded, and a later
    one is kept — the difference between "the camera is broken" and a usable photo.

Availability, not failure:
    A machine may have no camera, or its camera may be in use by a call, or covered
    by a privacy shutter. None of those are errors in Quainex; each is reported as
    what it is. The import itself is optional — without ``pygrabber`` the feature
    reports unavailable and the rest of the system is unaffected, exactly like the
    voice extra.

Architecture:
    capture_webcam(path)
        |-- FilterGraph -> first input device
        |-- warm up: grab and discard a few frames
        |-- keep the next frame (BGR ndarray)
        +-- Pillow: BGR -> RGB -> JPEG

Dependencies:
    pygrabber, comtypes, Pillow, numpy (the "camera" extra)

Example:
    >>> capture_webcam(Path("shot.jpg"))  # doctest: +SKIP
    Path('shot.jpg')

Future improvements:
    * Choose among multiple cameras rather than always the first.
    * A short countdown beep, so a photo of a person is not a surprise.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from quainex.core.exceptions import CommandExecutionError
from quainex.core.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

_log = get_logger(__name__)

#: Frames pulled and thrown away before keeping one, to let exposure settle.
_WARMUP_FRAMES = 4

#: How long to wait for the camera to deliver a frame before giving up. A camera
#: held by another application never delivers, and without a deadline the capture
#: would hang the request forever.
_CAPTURE_TIMEOUT_SECONDS = 8.0

#: JPEG quality. High enough to recognise a face, low enough to stay well under
#: Telegram's photo limit even at full sensor resolution.
_JPEG_QUALITY = 85


def is_available() -> bool:
    """Whether webcam capture can run on this machine.

    Reports the *library* rather than probing the device: enumerating cameras
    powers the hardware on, and a readiness check should not turn on a light. A
    missing device surfaces at capture time, named for what it is.

    Returns:
        Whether the capture dependencies import.
    """
    import importlib.util

    return all(importlib.util.find_spec(name) for name in ("pygrabber", "PIL", "numpy"))


def list_cameras() -> list[str]:
    """Return the names of connected video input devices.

    Returns:
        Device names, first is the default. Empty when there is no camera or the
        dependencies are missing.
    """
    if not is_available():
        return []
    try:
        from pygrabber.dshow_graph import FilterGraph

        return list(FilterGraph().get_input_devices())
    except Exception as exc:
        _log.warning("webcam_enumeration_failed", error=str(exc))
        return []


def capture_webcam(destination: Path, *, device_index: int = 0) -> Path:
    """Capture one still from a webcam and write it as JPEG.

    Args:
        destination: Where to write the image.
        device_index: Which camera, when there is more than one. Defaults to the
            first.

    Returns:
        ``destination``, for chaining.

    Raises:
        CommandExecutionError: No camera, the camera is busy, or capture failed.
    """
    if not is_available():
        raise CommandExecutionError(
            'Webcam support is not installed. Run: pip install -e ".[camera]"'
        )

    cameras = list_cameras()
    if not cameras:
        raise CommandExecutionError(
            "No camera was found. Check that one is connected and not disabled in "
            "Device Manager or behind a privacy shutter."
        )
    if device_index >= len(cameras):
        raise CommandExecutionError(
            f"Camera {device_index} does not exist; this machine has {len(cameras)}."
        )

    frame = _grab_frame(device_index)
    if frame is None:
        raise CommandExecutionError(
            f"The camera '{cameras[device_index]}' did not produce an image within "
            f"{_CAPTURE_TIMEOUT_SECONDS:.0f}s. It may be in use by another application, "
            f"or covered."
        )

    _encode(frame, destination)
    _log.info("webcam_captured", camera=cameras[device_index], path=str(destination))
    return destination


def _grab_frame(device_index: int) -> object | None:
    """Drive DirectShow to capture a single warmed-up frame.

    The sample-grabber callback fires on the graph's own thread, so the frames are
    collected into a list under the graph's lifetime and the newest kept. A
    threading event, not a sleep, ends the wait the moment enough frames arrive.

    Args:
        device_index: Which camera to open.

    Returns:
        The kept frame as a BGR ndarray, or ``None`` on timeout.
    """
    from pygrabber.dshow_graph import FilterGraph

    frames: list[object] = []
    enough = threading.Event()

    def on_frame(frame: object) -> None:
        frames.append(frame)
        if len(frames) > _WARMUP_FRAMES:
            enough.set()

    graph = FilterGraph()
    graph.add_video_input_device(device_index)
    graph.add_sample_grabber(on_frame)
    graph.add_null_render()
    graph.prepare_preview_graph()
    graph.run()
    try:
        # Repeated grabs rather than one: each `grab_frame` requests a single frame,
        # and warm-up needs several. The event releases as soon as enough arrive.
        deadline = threading.Timer(_CAPTURE_TIMEOUT_SECONDS, enough.set)
        deadline.start()
        try:
            while not enough.is_set():
                graph.grab_frame()
                enough.wait(0.3)
        finally:
            deadline.cancel()
    finally:
        graph.stop()

    return frames[-1] if frames else None


def _encode(frame: object, destination: Path) -> None:
    """Convert a captured BGR frame to RGB and write it as JPEG.

    Args:
        frame: The captured frame, BGR as DirectShow delivers it.
        destination: Where to write the JPEG.

    Raises:
        CommandExecutionError: The frame could not be encoded or written.
    """
    import numpy
    from PIL import Image

    try:
        array = numpy.asarray(frame)
        # DirectShow hands back BGR; Pillow expects RGB, so the channels are
        # reversed. Without this every photo comes out with blue and red swapped —
        # subtle enough to miss in code, obvious the moment a person looks at it.
        rgb = array[:, :, ::-1]
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb, mode="RGB").save(destination, format="JPEG", quality=_JPEG_QUALITY)
    except (OSError, ValueError) as exc:
        raise CommandExecutionError(f"Could not save the webcam image: {exc}") from exc
