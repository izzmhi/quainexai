"""System tray presence and a global hotkey.

Purpose:
    Make Quainex reachable from anywhere on the machine, without a terminal window
    and without hunting for a browser tab.

Why this matters more than it looks:
    Everything before this required *going to* Quainex — finding the terminal,
    finding the tab. That is the difference between a project you run and a tool
    you use. A tray icon and one keystroke removes it.

Why ctypes and not pystray/keyboard:
    Both would work and both add dependencies. ``RegisterHotKey`` and
    ``Shell_NotifyIcon`` are two Win32 calls each, already available through
    ``ctypes``, and this file has no imports outside the standard library as a
    result.

    The ``keyboard`` package deserves a specific mention: it works by installing a
    low-level keyboard hook, which requires administrator rights and sees *every*
    keystroke on the machine. For a global hotkey that is a wildly
    disproportionate amount of access. ``RegisterHotKey`` asks the OS to deliver
    one specific combination and nothing else, needs no elevation, and cannot
    observe anything it was not registered for. When the cheap option is also the
    one that sees less, take it.

What it deliberately does not do:
    Start, stop or supervise the server. This is a *launcher*: it opens the
    dashboard and reports whether the process is up. Putting process control here
    would mean two things that can start Quainex and disagree about whether it is
    running. Autostart belongs to Windows (see ``scripts/install_startup.ps1``).

Architecture:
    tray.py (pythonw, no console)
        |-- RegisterHotKey(Ctrl+Alt+Q)  --> open the dashboard
        |-- Shell_NotifyIcon            --> icon + balloon tips
        +-- GetMessage loop
              |-- WM_HOTKEY   -> open dashboard
              +-- WM_TRAYICON -> left click: open, right click: menu

Dependencies:
    ctypes (Windows only), webbrowser, urllib

Example:
    pythonw -m quainex.desktop.tray

Future improvements:
    * Show the last few commands in the menu.
    * Summon the Tauri overlay instead of a browser tab, once that exists.
"""

from __future__ import annotations

import ctypes
import sys
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from ctypes import wintypes

#: Win32 constants, named rather than inlined so the message loop reads.
_WM_DESTROY = 0x0002
_WM_COMMAND = 0x0111
_WM_HOTKEY = 0x0312
_WM_TRAYICON = 0x0400 + 1  # WM_APP + 1: our own private message
_WM_LBUTTONUP = 0x0202
_WM_RBUTTONUP = 0x0205

_MOD_CONTROL = 0x0002
_MOD_ALT = 0x0001
#: MOD_NOREPEAT: holding the combination fires once, not continuously.
_MOD_NOREPEAT = 0x4000
_VK_Q = 0x51

_NIM_ADD = 0x0000
_NIM_MODIFY = 0x0001
_NIM_DELETE = 0x0002
_NIF_MESSAGE = 0x0001
_NIF_ICON = 0x0002
_NIF_TIP = 0x0004
_NIF_INFO = 0x0010

_IDI_APPLICATION = 32512

_TPM_RIGHTBUTTON = 0x0002
_MF_STRING = 0x0000
_MF_SEPARATOR = 0x0800

_ID_OPEN = 1001
_ID_DOCS = 1002
_ID_QUIT = 1003

_HOTKEY_ID = 1

#: How the hotkey is described to the user. Kept next to the registration so the
#: balloon tip cannot claim a different combination from the one that works.
_HOTKEY_LABEL = "Ctrl+Alt+Q"


def _int_resource(value: int) -> wintypes.LPCWSTR:
    """Represent an integer resource id where Win32 expects a string pointer.

    ``LoadIconW`` takes either a name or a numeric id in the same parameter, which
    the Win32 headers express with a macro. Python has no equivalent, so the
    integer is cast to a pointer explicitly.

    Args:
        value: The resource identifier.

    Returns:
        The identifier, typed as a wide-string pointer.
    """
    return ctypes.cast(ctypes.c_void_p(value), wintypes.LPCWSTR)


#: Win32 signature for a window procedure.
_WNDPROC = ctypes.WINFUNCTYPE(
    ctypes.c_ssize_t, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
)


class _WNDCLASS(ctypes.Structure):
    """Win32 ``WNDCLASSW``."""

    _fields_ = (
        ("style", wintypes.UINT),
        ("lpfnWndProc", _WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HICON),
        ("hCursor", wintypes.HANDLE),
        ("hbrBackground", wintypes.HBRUSH),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
    )


class _NOTIFYICONDATA(ctypes.Structure):
    """Win32 ``NOTIFYICONDATAW``, trimmed to the fields used here."""

    _fields_ = (
        ("cbSize", wintypes.DWORD),
        ("hWnd", wintypes.HWND),
        ("uID", wintypes.UINT),
        ("uFlags", wintypes.UINT),
        ("uCallbackMessage", wintypes.UINT),
        ("hIcon", wintypes.HICON),
        ("szTip", wintypes.WCHAR * 128),
        ("dwState", wintypes.DWORD),
        ("dwStateMask", wintypes.DWORD),
        ("szInfo", wintypes.WCHAR * 256),
        ("uVersion", wintypes.UINT),
        ("szInfoTitle", wintypes.WCHAR * 64),
        ("dwInfoFlags", wintypes.DWORD),
    )


class TrayApplication:
    """A tray icon and a global hotkey that open the Quainex dashboard."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000") -> None:
        """Construct the application.

        Args:
            base_url: Where Quainex is listening.

        Raises:
            ValueError: ``base_url`` is not an ``http``/``https`` URL with a host.
        """
        # Validated rather than trusted. This value reaches both `urlopen` and
        # `webbrowser.open`, and neither restricts the scheme: a `file:` URL would
        # read a local file, and the more exotic schemes Windows registers can
        # launch programs. It is a constructor argument, so checking it here is the
        # one place that covers every use below.
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(
                f"base_url must be an http or https URL with a host, got {base_url!r}."
            )
        self._base_url = base_url.rstrip("/")
        self._user32 = ctypes.windll.user32
        self._shell32 = ctypes.windll.shell32
        self._kernel32 = ctypes.windll.kernel32
        self._hwnd: int = 0
        self._icon: _NOTIFYICONDATA | None = None
        # Kept alive explicitly: ctypes callbacks are garbage collected like any
        # other object, and Windows holding the only reference is not a reference.
        # Letting it collect crashes the process from inside the message loop.
        self._proc: object | None = None
        self._declare_signatures()

    def _declare_signatures(self) -> None:
        """Declare argument and return types for every Win32 function used.

        **Not optional on 64-bit Windows.** Without a declared ``restype``, ctypes
        assumes ``int`` — 32 bits — so any function returning a handle silently
        truncates it, and any function *taking* one gets a value it cannot convert.
        The first version of this file omitted them and died with
        ``OverflowError: int too long to convert`` on ``CreateWindowExW``, which
        names neither the parameter nor the cause.

        Declaring them once here also means every call site below reads as a plain
        function call.
        """
        self._user32.RegisterClassW.argtypes = [ctypes.POINTER(_WNDCLASS)]
        self._user32.RegisterClassW.restype = wintypes.WORD

        self._user32.CreateWindowExW.argtypes = [
            wintypes.DWORD,
            wintypes.LPCWSTR,
            wintypes.LPCWSTR,
            wintypes.DWORD,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.HWND,
            wintypes.HMENU,
            wintypes.HINSTANCE,
            wintypes.LPVOID,
        ]
        self._user32.CreateWindowExW.restype = wintypes.HWND

        self._user32.DefWindowProcW.argtypes = [
            wintypes.HWND,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
        ]
        self._user32.DefWindowProcW.restype = ctypes.c_ssize_t

        self._user32.LoadIconW.argtypes = [wintypes.HINSTANCE, wintypes.LPCWSTR]
        self._user32.LoadIconW.restype = wintypes.HICON

        self._user32.CreatePopupMenu.restype = wintypes.HMENU
        self._user32.RegisterHotKey.argtypes = [
            wintypes.HWND,
            ctypes.c_int,
            wintypes.UINT,
            wintypes.UINT,
        ]
        self._user32.RegisterHotKey.restype = wintypes.BOOL

        self._kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
        self._kernel32.GetModuleHandleW.restype = wintypes.HMODULE

        self._shell32.Shell_NotifyIconW.argtypes = [wintypes.DWORD, ctypes.c_void_p]
        self._shell32.Shell_NotifyIconW.restype = wintypes.BOOL

    # -- lifecycle --------------------------------------------------------

    def run(self) -> int:
        """Create the window, register everything, and pump messages.

        Returns:
            Process exit code.
        """
        self._create_window()
        self._add_icon()

        if not self._user32.RegisterHotKey(
            self._hwnd, _HOTKEY_ID, _MOD_CONTROL | _MOD_ALT | _MOD_NOREPEAT, _VK_Q
        ):
            # Not fatal. Another application may already own the combination, and
            # a tray icon with no hotkey is still useful — but say so rather than
            # leaving the user pressing keys that do nothing.
            self._notify(
                "Hotkey unavailable",
                f"{_HOTKEY_LABEL} is already registered by another program. "
                f"The tray icon still works.",
            )
        else:
            self._notify("Quainex is running", f"Press {_HOTKEY_LABEL} to open the console.")

        self._pump()
        return 0

    def _create_window(self) -> None:
        """Create the hidden window that receives hotkey and icon messages.

        A window is required even though nothing is drawn: ``RegisterHotKey`` and
        the tray icon both deliver messages, and messages need a destination.
        """
        self._proc = _WNDPROC(self._on_message)
        instance = self._kernel32.GetModuleHandleW(None)

        wnd_class = _WNDCLASS()
        wnd_class.lpfnWndProc = self._proc
        wnd_class.hInstance = instance
        wnd_class.lpszClassName = "QuainexTray"

        if not self._user32.RegisterClassW(ctypes.byref(wnd_class)):
            raise ctypes.WinError(ctypes.get_last_error())

        self._hwnd = self._user32.CreateWindowExW(
            0, "QuainexTray", "Quainex", 0, 0, 0, 0, 0, None, None, instance, None
        )
        if not self._hwnd:
            raise ctypes.WinError(ctypes.get_last_error())

    def _pump(self) -> None:
        """Run the message loop until the window is destroyed."""
        message = wintypes.MSG()
        while self._user32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
            self._user32.TranslateMessage(ctypes.byref(message))
            self._user32.DispatchMessageW(ctypes.byref(message))

    # -- the icon ---------------------------------------------------------

    def _add_icon(self) -> None:
        """Put the icon in the notification area."""
        data = _NOTIFYICONDATA()
        data.cbSize = ctypes.sizeof(_NOTIFYICONDATA)
        data.hWnd = self._hwnd
        data.uID = 1
        data.uFlags = _NIF_MESSAGE | _NIF_ICON | _NIF_TIP
        data.uCallbackMessage = _WM_TRAYICON
        # A stock icon rather than a bundled .ico: shipping one means a path to
        # resolve at runtime, and a wrong path produces an invisible tray entry. A
        # placeholder that always loads beats a custom one that sometimes does not.
        data.hIcon = self._user32.LoadIconW(None, _int_resource(_IDI_APPLICATION))
        data.szTip = f"Quainex - {_HOTKEY_LABEL} to open"
        self._shell32.Shell_NotifyIconW(_NIM_ADD, ctypes.byref(data))
        self._icon = data

    def _notify(self, title: str, message: str) -> None:
        """Show a balloon tip.

        Args:
            title: Notification title.
            message: Notification body.
        """
        if self._icon is None:
            return
        self._icon.uFlags = _NIF_INFO
        self._icon.szInfoTitle = title[:63]
        self._icon.szInfo = message[:255]
        self._shell32.Shell_NotifyIconW(_NIM_MODIFY, ctypes.byref(self._icon))

    def _remove_icon(self) -> None:
        """Take the icon out of the notification area.

        Without this the icon lingers as a ghost until something hovers over it.
        """
        if self._icon is not None:
            self._shell32.Shell_NotifyIconW(_NIM_DELETE, ctypes.byref(self._icon))
            self._icon = None

    # -- messages ---------------------------------------------------------

    def _on_message(self, hwnd: int, message: int, wparam: int, lparam: int) -> int:
        """Handle one window message.

        Args:
            hwnd: The window handle.
            message: Message identifier.
            wparam: First parameter.
            lparam: Second parameter.

        Returns:
            The message result.
        """
        if message == _WM_HOTKEY and wparam == _HOTKEY_ID:
            self._open_dashboard()
        elif message == _WM_TRAYICON:
            if lparam == _WM_LBUTTONUP:
                self._open_dashboard()
            elif lparam == _WM_RBUTTONUP:
                self._show_menu()
        elif message == _WM_COMMAND:
            self._on_command(wparam & 0xFFFF)
        elif message == _WM_DESTROY:
            self._remove_icon()
            self._user32.UnregisterHotKey(self._hwnd, _HOTKEY_ID)
            self._user32.PostQuitMessage(0)
            return 0

        return int(self._user32.DefWindowProcW(hwnd, message, wparam, lparam))

    def _show_menu(self) -> None:
        """Show the right-click menu at the cursor."""
        menu = self._user32.CreatePopupMenu()
        self._user32.AppendMenuW(menu, _MF_STRING, _ID_OPEN, f"Open console\t{_HOTKEY_LABEL}")
        self._user32.AppendMenuW(menu, _MF_STRING, _ID_DOCS, "API docs")
        self._user32.AppendMenuW(menu, _MF_SEPARATOR, 0, None)
        self._user32.AppendMenuW(menu, _MF_STRING, _ID_QUIT, "Quit tray")

        point = wintypes.POINT()
        self._user32.GetCursorPos(ctypes.byref(point))
        # SetForegroundWindow first: without it the menu does not dismiss when the
        # user clicks elsewhere, which is a documented Win32 quirk of tray menus.
        self._user32.SetForegroundWindow(self._hwnd)
        self._user32.TrackPopupMenu(menu, _TPM_RIGHTBUTTON, point.x, point.y, 0, self._hwnd, None)
        self._user32.DestroyMenu(menu)

    def _on_command(self, command: int) -> None:
        """Act on a menu selection.

        Args:
            command: The selected menu identifier.
        """
        if command == _ID_OPEN:
            self._open_dashboard()
        elif command == _ID_DOCS:
            webbrowser.open(f"{self._base_url}/docs")
        elif command == _ID_QUIT:
            self._user32.DestroyWindow(self._hwnd)

    # -- actions ----------------------------------------------------------

    def _open_dashboard(self) -> None:
        """Open the dashboard, or explain why it cannot be opened."""
        if self._is_server_up():
            webbrowser.open(f"{self._base_url}/ui/")
            return

        # A browser tab showing a connection error teaches the user nothing about
        # what to do, so the tray says it instead.
        self._notify(
            "Quainex is not running",
            "The server is not responding. Start it with `python main.py`, or set "
            "up automatic startup with scripts\\install_startup.ps1.",
        )

    def _is_server_up(self) -> bool:
        """Check whether the API is answering.

        Returns:
            Whether ``/health`` responded.
        """
        try:
            # The suppression below is justified rather than lazy: S310 asks for
            # scheme validation, and `__init__` refuses anything that is not http
            # or https before this can be reached.
            with urllib.request.urlopen(  # noqa: S310
                f"{self._base_url}/health", timeout=2
            ) as response:
                return bool(response.status == 200)
        except (urllib.error.URLError, OSError, ValueError):
            return False


def main() -> int:
    """Entry point.

    Returns:
        Process exit code.
    """
    if sys.platform != "win32":
        print("The tray application is Windows-only.", file=sys.stderr)
        return 1
    return TrayApplication().run()


if __name__ == "__main__":
    raise SystemExit(main())
