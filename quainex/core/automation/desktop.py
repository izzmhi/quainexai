"""Desktop control contract and its data types.

Purpose:
    Define every OS-level action Quainex can perform, as an interface that has no
    platform code in it.

Why an interface for something that only runs on Windows today:
    Two reasons, and the second matters more.

    1. The roadmap targets macOS and Linux later. Those become new
       implementations, not edits threaded through the command layer.
    2. **Tests must never actually shut down the machine.** Every command in
       Phase 3 is a real side effect — killing processes, locking the screen,
       powering off. With the controller behind a Protocol, the test suite
       injects a recording fake and asserts on what *would* have happened. There
       is no version of this that is safe to test against the real thing.

Architecture:
    CommandExecutor
        -> DesktopController (this contract)
             |-- WindowsDesktopController  (implemented)
             |-- MacDesktopController      (future)
             +-- FakeDesktopController     (tests)

Dependencies:
    pydantic

Future improvements:
    * Window management (focus, move, tile) once Phase 8 can see the screen.
    * Per-monitor screenshots rather than the full virtual desktop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, Field

#: Direction or absolute level for volume and brightness adjustments.
LevelChange = Literal["up", "down", "mute", "unmute"] | int


class FileHit(BaseModel):
    """One file matched by a search.

    Attributes:
        path: Absolute path to the file.
        size_bytes: File size, or ``None`` if it could not be read.
    """

    path: str
    size_bytes: int | None = None


class SystemSnapshot(BaseModel):
    """A point-in-time view of machine health.

    Attributes:
        cpu_percent: Overall CPU utilisation.
        memory_percent: Used physical memory as a percentage.
        disk_percent: Used space on the system drive as a percentage.
        battery_percent: Battery charge, or ``None`` on a desktop machine.
        uptime_seconds: Seconds since the machine booted.
    """

    cpu_percent: float = Field(ge=0)
    memory_percent: float = Field(ge=0)
    disk_percent: float = Field(ge=0)
    battery_percent: float | None = None
    uptime_seconds: float = Field(ge=0)


class DesktopController(Protocol):
    """Every OS-level action Quainex can take.

    Implementations raise ``quainex.core.exceptions.CommandExecutionError`` when
    an action cannot be completed, and return a short human-readable string
    describing what happened when it can.
    """

    # --- Applications ---
    def open_application(self, name: str) -> str:
        """Launch an application by friendly name."""
        ...

    def close_application(self, name: str) -> str:
        """Terminate a running application by friendly name."""
        ...

    # --- Navigation ---
    def open_url(self, url: str) -> str:
        """Open a URL in the default browser."""
        ...

    def open_folder(self, path: str) -> str:
        """Reveal a directory in the file explorer."""
        ...

    def create_folder(self, name: str) -> str:
        """Create a folder and reveal it."""
        ...

    def search_files(self, query: str, limit: int) -> list[FileHit]:
        """Find files whose name matches ``query`` under the allowed roots."""
        ...

    def resolve_file_for_sending(self, query: str) -> Path:
        """Find a single file, within permitted roots, to send off the machine."""
        ...

    def save_incoming_file(
        self, data: bytes, *, suggested_name: str, location: str | None
    ) -> Path:
        """Save a file received over Telegram into a permitted folder.

        ``location`` is a known-folder word or sub-path ("downloads",
        "documents/reports"), or ``None`` for the default inbox. ``suggested_name``
        is the sender's file name; it is sanitised and can never traverse out of the
        chosen folder. An existing file is never overwritten. Returns the path
        written.
        """
        ...

    # --- Session and power ---
    def lock_screen(self) -> str:
        """Lock the workstation."""
        ...

    def sleep(self) -> str:
        """Put the machine into a low-power state."""
        ...

    def restart(self, delay_seconds: int) -> str:
        """Reboot the machine after a delay."""
        ...

    def shutdown(self, delay_seconds: int) -> str:
        """Power the machine off after a delay."""
        ...

    # --- Media and display ---
    def set_volume(self, change: LevelChange) -> str:
        """Raise, lower, mute, or set the system volume."""
        ...

    def set_brightness(self, change: LevelChange) -> str:
        """Raise, lower, or set display brightness."""
        ...

    def set_keyboard_light(self, *, enabled: bool) -> str:
        """Turn the keyboard backlight on or off, where the firmware allows it."""
        ...

    def media_control(self, action: str) -> str:
        """Send a media transport command (play/pause/next/previous/stop)."""
        ...

    def control_window(self, action: str, name: str | None) -> str:
        """Minimise, maximise or restore a window, or minimise everything."""
        ...

    def list_running_apps(self, limit: int = 15) -> list[str]:
        """List the applications with a visible window."""
        ...

    def kill_process(self, name: str) -> str:
        """Force-close every running process matching ``name``."""
        ...

    # --- Utilities ---
    def screenshot(self, destination: Path) -> str:
        """Capture the screen to ``destination``."""
        ...

    def capture_webcam(self, destination: Path) -> str:
        """Capture a webcam still to ``destination``."""
        ...

    def set_wifi(self, *, enabled: bool) -> str:
        """Connect to or disconnect from Wi-Fi."""
        ...

    def wifi_status(self) -> str:
        """Report the Wi-Fi connection state."""
        ...

    def read_clipboard(self) -> str:
        """Return the current clipboard text."""
        ...

    def write_clipboard(self, text: str) -> str:
        """Replace the clipboard contents."""
        ...

    def type_text(self, text: str) -> str:
        """Type text into the focused window, as if from the keyboard."""
        ...

    def notify(self, message: str, title: str) -> str:
        """Show a desktop notification."""
        ...

    def system_info(self) -> SystemSnapshot:
        """Return a snapshot of machine health."""
        ...
