"""Windows implementation of ``DesktopController``.

Purpose:
    Perform the actual OS-level actions, safely.

Three rules every method here follows:

    1. **No ``shell=True``, ever.** Every subprocess is started from an argument
       list, so a target containing ``&& del /f /s /q`` arrives as one useless
       argument rather than as a second command.
    2. **Model output is never an argument.** Application names resolve through
       the allowlist in ``applications.py``; paths are resolved and checked to be
       inside permitted roots before use.
    3. **Every path is canonicalised before it is checked.** ``resolve()`` first,
       compare second — otherwise ``~/../../Windows/System32`` passes a naive
       prefix test.

Architecture:
    CommandExecutor -> DesktopController (Protocol) -> WindowsDesktopController
                                                            |-- ctypes  (lock, volume keys)
                                                            |-- psutil  (processes, metrics)
                                                            |-- Pillow  (screen capture)
                                                            +-- subprocess (power, brightness)

Dependencies:
    psutil, pillow, pyperclip, quainex.core.automation.applications

Future improvements:
    * Replace the volume key-event approach with the Core Audio API (pycaw) so a
      specific level can be set rather than nudged.
    * Use the Windows Toast API for notifications instead of a balloon tip.
"""

from __future__ import annotations

import ctypes
import os
import platform
import re
import shutil
import subprocess
import time
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import psutil

from quainex.core.automation.applications import (
    known_application_names,
    resolve_application,
)
from quainex.core.automation.desktop import FileHit, LevelChange, SystemSnapshot
from quainex.core.exceptions import CommandExecutionError, CommandNotAllowedError
from quainex.core.logging import get_logger

if TYPE_CHECKING:
    from quainex.config.settings import Settings
    from quainex.core.automation.applications import ApplicationSpec

_log = get_logger(__name__)

# Virtual key codes for the media keys, used with keybd_event.
_VK_VOLUME_MUTE = 0xAD
_VK_VOLUME_DOWN = 0xAE
_VK_VOLUME_UP = 0xAF
_KEYEVENTF_KEYUP = 0x0002

#: How many times a "up"/"down" request nudges the level. Each press is ~2%.
_VOLUME_STEPS = 5

#: URL schemes that may be opened. `file:` is excluded deliberately — it would
#: turn "open a website" into "open anything on disk".
_ALLOWED_URL_SCHEMES = frozenset({"http", "https"})

#: A syntactically valid hostname or IPv4 literal: dot-separated labels of
#: letters, digits and inner hyphens. Used to decide whether a scheme-less string
#: is a bare domain worth upgrading to https, or simply not an address at all.
_HOSTNAME = re.compile(
    r"^(?=.{1,253}$)[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*\.?$"
)

#: Subprocess timeout. Every command here is near-instant; anything that hangs
#: is a fault, not slow work.
_SUBPROCESS_TIMEOUT = 15


def _netsh_field(output: str, label: str) -> str | None:
    """Extract a ``Label : value`` field from netsh output.

    Matches on the label as a prefix before the colon, so it is unaffected by the
    variable whitespace netsh uses to align its columns. Case-insensitive because
    the exact casing varies across Windows versions.

    Args:
        output: The netsh command output.
        label: The field label to find, e.g. ``"SSID"``.

    Returns:
        The field value, or ``None`` when the label is absent.
    """
    wanted = label.strip().lower()
    for line in output.splitlines():
        if ":" not in line:
            continue
        name, _, value = line.partition(":")
        # Exact label match, not a substring: "SSID" must not also match
        # "BSSID", which sits two lines below it in the same output.
        if name.strip().lower() == wanted:
            return value.strip() or None
    return None


#: Windows KNOWNFOLDERID GUIDs for the folders people name by word.
#:
#: Resolved through ``SHGetKnownFolderPath`` rather than assumed to sit at
#: ``~/Downloads`` and so on, because that assumption is wrong on any machine with
#: OneDrive: this one's Desktop is ``~/OneDrive/Desktop`` and ``~/Desktop`` does not
#: exist at all. Guessing the literal path is exactly why "open desktop" opened the
#: home directory instead.
_KNOWN_FOLDER_IDS: dict[str, str] = {
    "desktop": "{B4BFCC3A-DB2C-424C-B029-7FE99A87C641}",
    "downloads": "{374DE290-123F-4565-9164-39C4925E467B}",
    "documents": "{FDD39AD0-238F-46AF-ADB4-6C85480369C7}",
    "pictures": "{33E28130-4E1E-4676-835A-98395C3BC3BB}",
    "music": "{4BD8D571-6D19-48D3-BE97-422220080E43}",
    "videos": "{18989B1D-99B5-455B-841C-AB7C74E4DDFC}",
    "home": "{5E6C858F-0E22-4760-9AFE-EA3317B67173}",
}

#: Words people use for each known folder, folded to the canonical key above.
_FOLDER_ALIASES: dict[str, str] = {
    "desktop": "desktop",
    "downloads": "downloads",
    "download": "downloads",
    "documents": "documents",
    "document": "documents",
    "docs": "documents",
    "pictures": "pictures",
    "picture": "pictures",
    "photos": "pictures",
    "images": "pictures",
    "music": "music",
    "videos": "videos",
    "video": "videos",
    "movies": "videos",
    "home": "home",
    "profile": "home",
}


class _GUID(ctypes.Structure):
    """A Windows ``GUID``, built from its string form."""

    _fields_ = (
        ("Data1", ctypes.c_ulong),
        ("Data2", ctypes.c_ushort),
        ("Data3", ctypes.c_ushort),
        ("Data4", ctypes.c_ubyte * 8),
    )

    def __init__(self, guid: str) -> None:
        """Parse a ``{...}`` GUID string.

        Args:
            guid: The GUID in registry string form.
        """
        super().__init__()
        ctypes.windll.ole32.CLSIDFromString(guid, ctypes.byref(self))


def resolve_known_folder(name: str) -> Path | None:
    """Return the real path of a named Windows folder, or ``None``.

    Handles OneDrive and other redirections, because it asks Windows where the
    folder actually is rather than assuming it lives under the home directory.

    Args:
        name: A folder word such as ``"desktop"`` or ``"downloads"``. Aliases like
            "docs" and "photos" are accepted.

    Returns:
        The resolved path, or ``None`` when the name is not a known folder or the
        lookup fails.
    """
    key = _FOLDER_ALIASES.get(name.strip().lower())
    if key is None:
        return None

    folder_id = _GUID(_KNOWN_FOLDER_IDS[key])
    pointer = ctypes.c_wchar_p()
    result = ctypes.windll.shell32.SHGetKnownFolderPath(
        ctypes.byref(folder_id), 0, None, ctypes.byref(pointer)
    )
    value = pointer.value
    ctypes.windll.ole32.CoTaskMemFree(pointer)
    # S_OK is 0; anything else means the folder is not present on this machine.
    return Path(value) if result == 0 and value else None


def _system_executable(name: str) -> str:
    """Resolve a Windows system executable to an absolute path.

    Invoking ``powershell.exe`` by bare name defers to ``PATH``, which any
    program that can write a directory earlier in ``PATH`` can hijack. Resolving
    to an absolute path — preferring the System32 directory under
    ``%SystemRoot%`` — means Quainex runs the real binary rather than whatever
    shadowed it.

    Args:
        name: Executable file name, e.g. ``"shutdown.exe"``.

    Returns:
        An absolute path when one can be determined, otherwise ``name``.
    """
    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    canonical = Path(system_root) / "System32" / name
    if canonical.is_file():
        return str(canonical)

    resolved = shutil.which(name)
    return resolved if resolved else name


class WindowsDesktopController:
    """Controls a Windows desktop session."""

    def __init__(self, settings: Settings) -> None:
        """Construct the controller.

        Args:
            settings: Configuration supplying search roots and limits.
        """
        self._settings = settings
        if platform.system() != "Windows":  # pragma: no cover - platform guard
            _log.warning(
                "desktop_controller_platform_mismatch",
                detail=f"WindowsDesktopController on {platform.system()}; actions will fail.",
            )

    # --- Applications ----------------------------------------------------

    def open_application(self, name: str) -> str:
        """Launch an allowlisted application.

        Args:
            name: Friendly application name from the utterance.

        Returns:
            A short description of what was launched.

        Raises:
            CommandNotAllowedError: The application is not in the catalogue.
            CommandExecutionError: The application is allowlisted but not installed.
        """
        spec = self._require_application(name)

        for executable in spec.executables:
            resolved = shutil.which(executable)
            if resolved:
                # Argument list, never a shell string, and no model-supplied args.
                subprocess.Popen([resolved])  # noqa: S603
                _log.info("application_opened", application=spec.key, executable=resolved)
                return f"Opened {spec.display}."

        if spec.uri:
            os.startfile(spec.uri)  # noqa: S606 - fixed protocol URI from the catalogue
            _log.info("application_opened", application=spec.key, uri=spec.uri)
            return f"Opened {spec.display}."

        raise CommandExecutionError(
            f"{spec.display} is allowed but could not be found on this machine."
        )

    def close_application(self, name: str) -> str:
        """Terminate an allowlisted application.

        Sends a terminate signal rather than a kill, so the application can save
        state and prompt about unsaved work.

        Args:
            name: Friendly application name from the utterance.

        Returns:
            A short description of what was closed.

        Raises:
            CommandNotAllowedError: The application is not in the catalogue.
            CommandExecutionError: No matching process was running.
        """
        spec = self._require_application(name)
        wanted = {p.lower() for p in spec.process_names}
        closed = 0

        for process in psutil.process_iter(["name"]):
            process_name = (process.info.get("name") or "").lower()
            if process_name in wanted:
                try:
                    process.terminate()
                    closed += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    # Vanished between listing and terminating, or protected.
                    continue

        if closed == 0:
            raise CommandExecutionError(f"{spec.display} does not appear to be running.")

        _log.info("application_closed", application=spec.key, processes=closed)
        return f"Closed {spec.display} ({closed} process{'es' if closed > 1 else ''})."

    # --- Navigation ------------------------------------------------------

    def open_url(self, url: str) -> str:
        """Open an http(s) URL in the default browser.

        The input is parsed *before* any convenience rewriting. An earlier
        version prepended ``https://`` whenever ``://`` was absent, which
        laundered hostile input: ``javascript:alert(1)`` contains no ``://``, so
        it became ``https://javascript:alert(1)`` — a string that passes a naive
        scheme check. Schemes are now validated on the raw input, and only a
        string that is *already* a bare hostname is upgraded.

        Args:
            url: The URL or bare domain to open.

        Returns:
            A short description of what was opened.

        Raises:
            CommandNotAllowedError: The scheme is not http/https, or the input
                is not a valid web address.
        """
        raw = url.strip()
        parsed = urlparse(raw)

        if parsed.scheme:
            # An explicit scheme is honoured only if it is one we allow. This
            # also catches Windows paths, where urlparse reads "C:" as a scheme.
            if parsed.scheme.lower() not in _ALLOWED_URL_SCHEMES:
                raise CommandNotAllowedError(
                    f"Only http and https addresses may be opened; got '{parsed.scheme}'."
                )
            candidate = raw
        else:
            # No scheme at all: upgrade only if what precedes the first slash is
            # a syntactically valid host (optionally with a port).
            host_part = raw.split("/", 1)[0]
            host_only, _, port = host_part.partition(":")
            if not _HOSTNAME.match(host_only) or (port and not port.isdigit()):
                raise CommandNotAllowedError(f"'{url}' is not a valid web address.")
            candidate = f"https://{raw}"
            parsed = urlparse(candidate)

        host = parsed.hostname
        if not host:
            raise CommandNotAllowedError(f"'{url}' is not a valid web address.")

        webbrowser.open(candidate)
        _log.info("url_opened", host=host)
        return f"Opened {host}."

    def open_folder(self, path: str) -> str:
        """Reveal a directory in File Explorer.

        A bare folder word — "desktop", "downloads", "documents" — is resolved
        through the Windows known-folder API, so it finds the real location even
        when OneDrive has redirected it. Anything else is treated as a path,
        absolute or relative to home, and confined to the permitted roots.

        Args:
            path: Folder name or path.

        Returns:
            A short description of what was opened.

        Raises:
            CommandNotAllowedError: The path escapes the permitted roots.
            CommandExecutionError: The path does not exist or is not a directory.
        """
        resolved = self._resolve_folder(path)
        if not resolved.exists():
            raise CommandExecutionError(
                f"'{resolved}' does not exist. Say 'create a folder called …' to make it."
            )
        if not resolved.is_dir():
            raise CommandExecutionError(f"'{resolved}' is not a folder.")

        os.startfile(resolved)  # noqa: S606 - path validated against permitted roots
        _log.info("folder_opened", path=str(resolved))
        return f"Opened {resolved}."

    def create_folder(self, name: str) -> str:
        """Create a folder and reveal it.

        "projects" makes ``Desktop/projects``; "downloads/reports" makes a
        sub-folder under Downloads. A location word at the front is honoured, so
        "documents/tax 2026" lands in the real Documents folder even when it is
        redirected to OneDrive. Everything is confined to the permitted roots.

        Args:
            name: The folder to create, optionally prefixed with a known folder.

        Returns:
            A confirmation naming the created folder.

        Raises:
            CommandNotAllowedError: The target escapes the permitted roots, or the
                name is empty.
            CommandExecutionError: The folder could not be created.
        """
        cleaned = name.strip().strip("/\\").strip()
        if not cleaned:
            raise CommandNotAllowedError("A folder needs a name.")

        target = self._resolve_new_folder(cleaned)
        try:
            existed = target.exists()
            target.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise CommandExecutionError(f"Could not create '{target}': {exc}") from exc

        os.startfile(target)  # noqa: S606 - path validated against permitted roots
        _log.info("folder_created", path=str(target), existed=existed)
        verb = "Opened existing" if existed else "Created"
        return f"{verb} folder {target}."

    def resolve_file_for_sending(self, query: str) -> Path:
        """Find a single file to send off the machine.

        Three shapes, in order:
          * "latest" / "latest download" / "recent" — the newest file in Downloads.
          * a name containing a path separator or extension — resolved as a path.
          * a bare name — the most recently modified file whose name contains it,
            searched across the permitted roots.

        Every result is confined to the permitted roots, so this cannot be turned
        into "send me any file on the disk".

        Args:
            query: What to send.

        Returns:
            The resolved file path.

        Raises:
            CommandNotAllowedError: The query is empty or escapes the roots.
            CommandExecutionError: No matching file was found.
        """
        needle = query.strip()
        if not needle:
            raise CommandNotAllowedError("Name a file to send.")

        if needle.lower() in {
            "latest",
            "recent",
            "latest download",
            "last download",
            "latest file",
        }:
            newest = self._newest_in_known_folder("downloads")
            if newest is None:
                raise CommandExecutionError("Your Downloads folder is empty.")
            return newest

        # A path-ish query (has a separator or a suffix) is resolved directly.
        if "/" in needle or "\\" in needle or Path(needle).suffix:
            candidate = self._resolve_folder(needle)
            if candidate.is_file():
                return candidate
            # Fall through to a name search when the exact path is not a file:
            # "report.pdf" is a name, not a path, on most phrasings.

        match = self._newest_matching_file(needle)
        if match is None:
            raise CommandExecutionError(f"No file matching '{needle}' was found in your folders.")
        return match

    def search_files(self, query: str, limit: int) -> list[FileHit]:
        """Find files whose name contains ``query`` under the permitted roots.

        Args:
            query: Substring to match against file names.
            limit: Maximum number of results to return.

        Returns:
            Matching files, up to ``limit``.

        Raises:
            CommandNotAllowedError: The query is empty.
        """
        needle = query.strip().lower()
        if not needle:
            raise CommandNotAllowedError("A search needs something to search for.")

        hits: list[FileHit] = []
        for root in self._search_roots():
            if len(hits) >= limit:
                break
            for candidate in self._walk(root, limit - len(hits)):
                if needle in candidate.name.lower():
                    try:
                        size = candidate.stat().st_size
                    except OSError:
                        size = None
                    hits.append(FileHit(path=str(candidate), size_bytes=size))
                    if len(hits) >= limit:
                        break

        _log.info("files_searched", results=len(hits), limit=limit)
        return hits

    # --- Session and power ----------------------------------------------

    def lock_screen(self) -> str:
        """Lock the workstation.

        Returns:
            A short confirmation.

        Raises:
            CommandExecutionError: The lock call was rejected by Windows.
        """
        if ctypes.windll.user32.LockWorkStation() == 0:
            raise CommandExecutionError("Windows refused the lock request.")
        _log.info("screen_locked")
        return "Locked the screen."

    def sleep(self) -> str:
        """Put the machine into a low-power state.

        Returns:
            A short confirmation.
        """
        self._run(
            [_system_executable("rundll32.exe"), "powrprof.dll,SetSuspendState", "0,1,0"],
            failure="Could not put the machine to sleep.",
        )
        _log.info("system_sleep_requested")
        return "Going to sleep."

    def restart(self, delay_seconds: int) -> str:
        """Reboot the machine after a delay.

        The delay is deliberate: it leaves a window in which ``shutdown /a``
        aborts the reboot, which is the last line of defence against a
        misclassified command.

        Args:
            delay_seconds: Seconds to wait before restarting.

        Returns:
            A confirmation naming the abort window.
        """
        self._run(
            [_system_executable("shutdown.exe"), "/r", "/t", str(delay_seconds)],
            failure="Could not schedule a restart.",
        )
        _log.warning("system_restart_scheduled", delay_seconds=delay_seconds)
        return (
            f"Restarting in {delay_seconds} seconds. "
            f"Run 'shutdown /a' within that window to cancel."
        )

    def shutdown(self, delay_seconds: int) -> str:
        """Power the machine off after a delay.

        Args:
            delay_seconds: Seconds to wait before powering off.

        Returns:
            A confirmation naming the abort window.
        """
        self._run(
            [_system_executable("shutdown.exe"), "/s", "/t", str(delay_seconds)],
            failure="Could not schedule a shutdown.",
        )
        _log.warning("system_shutdown_scheduled", delay_seconds=delay_seconds)
        return (
            f"Shutting down in {delay_seconds} seconds. "
            f"Run 'shutdown /a' within that window to cancel."
        )

    # --- Media and display ----------------------------------------------

    def set_volume(self, change: LevelChange) -> str:
        """Adjust the system volume.

        Args:
            change: ``"up"``, ``"down"``, ``"mute"``, or a level 0-100.

        Returns:
            A short confirmation.

        Raises:
            CommandNotAllowedError: The requested change is not understood.
        """
        if change == "mute":
            self._tap_key(_VK_VOLUME_MUTE)
            _log.info("volume_changed", change="mute")
            return "Toggled mute."

        if change in {"up", "down"}:
            key = _VK_VOLUME_UP if change == "up" else _VK_VOLUME_DOWN
            for _ in range(_VOLUME_STEPS):
                self._tap_key(key)
            _log.info("volume_changed", change=change)
            return f"Turned the volume {change}."

        if isinstance(change, int) and 0 <= change <= 100:
            # Media keys can only nudge; mute then step up to approximate a level.
            raise CommandNotAllowedError(
                "Setting an exact volume level is not supported yet; use 'up', 'down' or 'mute'."
            )

        raise CommandNotAllowedError(f"Cannot interpret '{change}' as a volume change.")

    def set_brightness(self, change: LevelChange) -> str:
        """Adjust display brightness via WMI.

        Args:
            change: ``"up"``, ``"down"``, or a level 0-100.

        Returns:
            A short confirmation.

        Raises:
            CommandNotAllowedError: The requested change is not understood.
            CommandExecutionError: The display does not support WMI brightness.
        """
        current = self._current_brightness()
        if isinstance(change, int):
            level = change
        elif change == "up":
            level = min(100, current + 20)
        elif change == "down":
            level = max(0, current - 20)
        else:
            raise CommandNotAllowedError(f"Cannot interpret '{change}' as a brightness change.")

        level = max(0, min(100, level))
        self._run(
            [
                _system_executable("powershell.exe"),
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                # Level is an int we computed, never model text.
                f"(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods)"
                f".WmiSetBrightness(1,{level})",
            ],
            failure="This display does not support software brightness control.",
        )
        _log.info("brightness_changed", level=level)
        return f"Set brightness to {level}%."

    # --- Utilities -------------------------------------------------------

    def screenshot(self, destination: Path) -> str:
        """Capture the full desktop to a PNG file.

        Args:
            destination: Where to write the image.

        Returns:
            A confirmation naming the file.

        Raises:
            CommandExecutionError: The screen could not be captured.
        """
        try:
            from PIL import ImageGrab

            destination.parent.mkdir(parents=True, exist_ok=True)
            image = ImageGrab.grab(all_screens=True)
            image.save(destination, format="PNG")
        except OSError as exc:
            raise CommandExecutionError(f"Could not capture the screen: {exc}") from exc

        _log.info("screenshot_captured", path=str(destination))
        return f"Saved a screenshot to {destination}."

    def capture_webcam(self, destination: Path) -> str:
        """Capture a still from the webcam.

        Delegates to ``automation.webcam`` so the DirectShow and image-encoding
        detail stays out of this controller — and so a machine without the camera
        extra reports it cleanly rather than failing to import.

        Args:
            destination: Where to write the JPEG.

        Returns:
            A confirmation naming the file.

        Raises:
            CommandExecutionError: No camera, camera busy, or capture failed.
        """
        from quainex.core.automation.webcam import capture_webcam

        capture_webcam(destination)
        return f"Captured a webcam photo to {destination}."

    def set_wifi(self, *, enabled: bool) -> str:
        """Connect to or disconnect from Wi-Fi.

        Uses ``netsh wlan`` rather than toggling the adapter, deliberately:
        disabling the radio needs administrator rights, and Quainex runs as a
        normal user. Disconnecting does not, and it achieves what "turn off Wi-Fi"
        means to a person — off the network. Reconnecting joins the first saved
        profile in range.

        Args:
            enabled: ``True`` to connect, ``False`` to disconnect.

        Returns:
            What happened.

        Raises:
            CommandExecutionError: The command failed, or no saved network exists
                to reconnect to.
        """
        netsh = _system_executable("netsh.exe")
        if not enabled:
            self._run([netsh, "wlan", "disconnect"], failure="Could not disconnect from Wi-Fi.")
            _log.info("wifi_changed", enabled=False)
            return "Disconnected from Wi-Fi."

        profile = self._first_wifi_profile()
        if profile is None:
            raise CommandExecutionError(
                "No saved Wi-Fi network to reconnect to. Connect once manually, and "
                "Windows will remember it."
            )
        self._run(
            [netsh, "wlan", "connect", f"name={profile}"],
            failure=f"Could not connect to '{profile}'.",
        )
        _log.info("wifi_changed", enabled=True, profile=profile)
        return f"Connecting to Wi-Fi network '{profile}'."

    def wifi_status(self) -> str:
        """Report the Wi-Fi connection state.

        Returns:
            A one-line summary: connected (with the network name) or not.
        """
        output = self._netsh_output(["wlan", "show", "interfaces"])
        state = _netsh_field(output, "State") or "unknown"
        ssid = _netsh_field(output, "SSID")

        if state.lower() == "connected" and ssid:
            return f"Wi-Fi is connected to '{ssid}'."
        return f"Wi-Fi is {state.lower()}."

    def read_clipboard(self) -> str:
        """Return the clipboard contents.

        Returns:
            The clipboard text, or a note that it is empty.

        Raises:
            CommandExecutionError: The clipboard could not be read.
        """
        import pyperclip

        try:
            text = pyperclip.paste()
        except pyperclip.PyperclipException as exc:
            raise CommandExecutionError(f"Could not read the clipboard: {exc}") from exc

        # The value itself is returned to the caller but deliberately not logged:
        # clipboards routinely hold passwords.
        _log.info("clipboard_read", characters=len(text))
        return text or "The clipboard is empty."

    def write_clipboard(self, text: str) -> str:
        """Replace the clipboard contents.

        Args:
            text: The text to copy.

        Returns:
            A short confirmation.

        Raises:
            CommandExecutionError: The clipboard could not be written.
        """
        import pyperclip

        try:
            pyperclip.copy(text)
        except pyperclip.PyperclipException as exc:
            raise CommandExecutionError(f"Could not write to the clipboard: {exc}") from exc

        _log.info("clipboard_written", characters=len(text))
        return "Copied to the clipboard."

    def notify(self, message: str, title: str) -> str:
        """Show a desktop balloon notification.

        Args:
            message: Notification body.
            title: Notification title.

        Returns:
            A short confirmation.
        """
        # Text is passed as a PowerShell single-quoted literal with embedded
        # quotes doubled, so a message containing quotes cannot break out.
        safe_message = message.replace("'", "''")
        safe_title = title.replace("'", "''")
        script = (
            "Add-Type -AssemblyName System.Windows.Forms;"
            "$n = New-Object System.Windows.Forms.NotifyIcon;"
            "$n.Icon = [System.Drawing.SystemIcons]::Information;"
            "$n.Visible = $true;"
            f"$n.ShowBalloonTip(5000, '{safe_title}', '{safe_message}', 'Info');"
            "Start-Sleep -Seconds 6; $n.Dispose()"
        )
        self._run(
            [
                _system_executable("powershell.exe"),
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                script,
            ],
            failure="Could not show the notification.",
        )
        _log.info("notification_shown", title=title)
        return "Notification shown."

    def system_info(self) -> SystemSnapshot:
        """Collect a snapshot of machine health.

        Returns:
            Current CPU, memory, disk, battery and uptime figures.
        """
        battery = psutil.sensors_battery()
        return SystemSnapshot(
            cpu_percent=psutil.cpu_percent(interval=0.1),
            memory_percent=psutil.virtual_memory().percent,
            disk_percent=psutil.disk_usage(str(Path.home().anchor)).percent,
            battery_percent=battery.percent if battery else None,
            uptime_seconds=max(0.0, time.time() - psutil.boot_time()),
        )

    # --- internals -------------------------------------------------------

    @staticmethod
    def _require_application(name: str) -> ApplicationSpec:
        """Resolve a name to an allowlisted application or refuse.

        Args:
            name: Friendly application name.

        Returns:
            The matching ``ApplicationSpec``.

        Raises:
            CommandNotAllowedError: The name is not in the catalogue.
        """
        spec = resolve_application(name)
        if spec is None:
            available = ", ".join(known_application_names())
            raise CommandNotAllowedError(
                f"'{name}' is not an allowed application. Available: {available}."
            )
        return spec

    def _search_roots(self) -> list[Path]:
        """Return the directories file search is permitted to traverse.

        Returns:
            Existing permitted root directories.
        """
        return [root for root in self._settings.resolved_search_roots if root.is_dir()]

    def _resolve_folder(self, path: str) -> Path:
        """Resolve a folder word or path to a real, permitted location.

        A single known-folder word maps through the Windows API; anything with a
        separator has its first segment resolved as a known folder when it is one
        ("downloads/reports"), so redirection is handled at any depth. The result
        is confined to the permitted roots.

        Args:
            path: Folder name or path from the utterance.

        Returns:
            The resolved, contained path.

        Raises:
            CommandNotAllowedError: The path escapes every permitted root.
        """
        cleaned = path.strip().strip("/\\")

        known = resolve_known_folder(cleaned)
        if known is not None:
            return self._contain(known)

        # "downloads/reports/2026": resolve the leading known folder, keep the rest.
        head, sep, tail = cleaned.replace("\\", "/").partition("/")
        base = resolve_known_folder(head)
        if base is not None and sep:
            return self._contain((base / tail).resolve())

        return self._resolve_within_roots(cleaned)

    def _resolve_new_folder(self, name: str) -> Path:
        """Resolve where a to-be-created folder should live.

        A bare name defaults to the Desktop — where "make a folder" most naturally
        means — while a leading known-folder word places it elsewhere. Confined to
        the permitted roots like everything else.

        Args:
            name: The folder name, possibly prefixed with a location.

        Returns:
            The contained target path.

        Raises:
            CommandNotAllowedError: The target escapes the permitted roots.
        """
        head, sep, tail = name.replace("\\", "/").partition("/")
        base = resolve_known_folder(head)
        if base is not None and sep and tail:
            return self._contain((base / tail).resolve())

        desktop = resolve_known_folder("desktop") or Path.home()
        return self._contain((desktop / name).resolve())

    def _newest_in_known_folder(self, folder: str) -> Path | None:
        """Return the most recently modified file directly in a known folder.

        Args:
            folder: A known-folder word, e.g. ``"downloads"``.

        Returns:
            The newest file, or ``None`` when the folder is empty or absent.
        """
        base = resolve_known_folder(folder)
        if base is None or not base.is_dir():
            return None
        files = [p for p in base.iterdir() if p.is_file()]
        return max(files, key=lambda p: p.stat().st_mtime) if files else None

    def _newest_matching_file(self, needle: str) -> Path | None:
        """Return the newest file under the roots whose name contains ``needle``.

        Args:
            needle: Substring to match against file names, case-insensitively.

        Returns:
            The best match, or ``None``.
        """
        lowered = needle.lower()
        best: Path | None = None
        best_mtime = -1.0
        for root in self._search_roots():
            for candidate in self._walk(root, self._settings.command_search_max_results):
                if lowered not in candidate.name.lower():
                    continue
                try:
                    mtime = candidate.stat().st_mtime
                except OSError:
                    continue
                if mtime > best_mtime:
                    best, best_mtime = candidate, mtime
        return best

    def _contain(self, resolved: Path) -> Path:
        """Confirm a resolved path stays inside a permitted root.

        Args:
            resolved: An already-canonicalised path.

        Returns:
            The path, unchanged.

        Raises:
            CommandNotAllowedError: The path escapes every permitted root.
        """
        roots = self._settings.resolved_search_roots
        if not any(resolved.is_relative_to(root) for root in roots):
            allowed = ", ".join(str(r) for r in roots)
            raise CommandNotAllowedError(
                f"'{resolved}' is outside the folders Quainex may touch. Allowed: {allowed}."
            )
        return resolved

    def _resolve_within_roots(self, path: str) -> Path:
        """Resolve a path and confirm it stays inside a permitted root.

        Canonicalises first so that ``..`` segments and symlinks are collapsed
        before the containment check — the check is meaningless otherwise.

        Args:
            path: Raw path from the utterance.

        Returns:
            The resolved path.

        Raises:
            CommandNotAllowedError: The path escapes every permitted root.
        """
        raw = Path(path.strip()).expanduser()
        resolved = (raw if raw.is_absolute() else Path.home() / raw).resolve()
        return self._contain(resolved)

    @staticmethod
    def _walk(root: Path, budget: int) -> list[Path]:
        """List files under ``root``, stopping once ``budget`` files are seen.

        Unreadable directories are skipped rather than aborting the walk, and the
        budget bounds the cost of a broad query on a large disk.

        Args:
            root: Directory to walk.
            budget: Maximum number of files to consider.

        Returns:
            Files found, up to the budget.
        """
        found: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(root, onerror=lambda _: None):
            # Skip hidden and system directories: noise, and often unreadable.
            dirnames[:] = [d for d in dirnames if not d.startswith((".", "$"))]
            for filename in filenames:
                found.append(Path(dirpath) / filename)
                if len(found) >= budget * 20:  # scan headroom for the name filter
                    return found
        return found

    @staticmethod
    def _tap_key(virtual_key: int) -> None:
        """Press and release a virtual key.

        Args:
            virtual_key: Windows virtual key code.
        """
        user32 = ctypes.windll.user32
        user32.keybd_event(virtual_key, 0, 0, 0)
        user32.keybd_event(virtual_key, 0, _KEYEVENTF_KEYUP, 0)

    def _netsh_output(self, args: list[str]) -> str:
        """Run a read-only ``netsh`` query and return its text.

        Args:
            args: Arguments after the executable, e.g. ``["wlan", "show", ...]``.

        Returns:
            Standard output, or an empty string on failure — a status query that
            cannot reach ``netsh`` should report "unknown", not raise.
        """
        try:
            result = subprocess.run(  # noqa: S603 - fixed argv, absolute exe, no shell
                [_system_executable("netsh.exe"), *args],
                capture_output=True,
                text=True,
                timeout=_SUBPROCESS_TIMEOUT,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            _log.warning("netsh_query_failed", args=args, error=str(exc))
            return ""
        return result.stdout

    def _first_wifi_profile(self) -> str | None:
        """Return the name of the first saved Wi-Fi profile.

        Returns:
            A profile name, or ``None`` when none are saved.
        """
        for line in self._netsh_output(["wlan", "show", "profiles"]).splitlines():
            # Lines read "All User Profile     : NetworkName". The colon splits the
            # label from the value regardless of the localised label text.
            if ":" in line and "profile" in line.lower():
                name = line.split(":", 1)[1].strip()
                if name:
                    return name
        return None

    def _current_brightness(self) -> int:
        """Read the current brightness percentage.

        Returns:
            The current level, or 50 when it cannot be determined.
        """
        try:
            result = subprocess.run(  # noqa: S603 - fixed argv, absolute exe, no shell
                [
                    _system_executable("powershell.exe"),
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    "(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightness)"
                    ".CurrentBrightness",
                ],
                capture_output=True,
                text=True,
                timeout=_SUBPROCESS_TIMEOUT,
                check=False,
            )
            return int(result.stdout.strip().splitlines()[0])
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            return 50

    @staticmethod
    def _run(argv: list[str], *, failure: str) -> None:
        """Run a subprocess from a fixed argument list.

        Args:
            argv: Executable and arguments. Never built from model output.
            failure: Message to raise if the process fails.

        Raises:
            CommandExecutionError: The process failed, timed out, or is missing.
        """
        try:
            result = subprocess.run(  # noqa: S603 - argument list, never shell=True
                argv,
                capture_output=True,
                text=True,
                timeout=_SUBPROCESS_TIMEOUT,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise CommandExecutionError(f"{failure} ({exc})") from exc

        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()[:200]
            raise CommandExecutionError(f"{failure} {detail}".strip())
