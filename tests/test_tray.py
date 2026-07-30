"""Tests for the tray launcher.

No window is created and no Win32 call is made: the message loop needs a real
desktop, and a test that opens one would be untestable in CI and unpleasant
locally. What is tested is the part with a decision in it — URL validation and
the liveness check — plus the fact that importing the module touches nothing.
"""

from __future__ import annotations

import sys

import pytest

from quainex.desktop.tray import TrayApplication, main


@pytest.mark.skipif(sys.platform != "win32", reason="ctypes.windll is Windows-only.")
def test_construction_touches_no_windows_api():
    """Constructing must be inert.

    The Win32 calls live in ``run()`` rather than ``__init__`` so the class can be
    constructed, inspected and tested without a desktop.
    """
    app = TrayApplication()

    assert app._hwnd == 0
    assert app._icon is None


@pytest.mark.skipif(sys.platform != "win32", reason="ctypes.windll is Windows-only.")
@pytest.mark.parametrize(
    "url",
    [
        # `file:` would make the liveness check read a local file.
        "file:///C:/Windows/win.ini",
        # Windows registers schemes that launch programs.
        "ms-settings:privacy",
        "javascript:alert(1)",
        # No host to connect to.
        "http://",
        "",
        "127.0.0.1:8000",
    ],
)
def test_a_dangerous_or_malformed_base_url_is_refused(url: str):
    """The value reaches both ``urlopen`` and ``webbrowser.open``.

    Neither restricts the scheme, so the check has to happen here.
    """
    with pytest.raises(ValueError, match="http or https"):
        TrayApplication(url)


@pytest.mark.skipif(sys.platform != "win32", reason="ctypes.windll is Windows-only.")
@pytest.mark.parametrize(
    "url",
    ["http://127.0.0.1:8000", "http://localhost:8000/", "https://quainex.example.com"],
)
def test_ordinary_urls_are_accepted(url: str):
    assert TrayApplication(url)._base_url == url.rstrip("/")


@pytest.mark.skipif(sys.platform != "win32", reason="ctypes.windll is Windows-only.")
def test_liveness_is_false_when_nothing_is_listening():
    """A closed port must read as down, not raise.

    The tray shows a balloon tip explaining how to start the server, which only
    works if this returns rather than propagating.
    """
    # Port 1 on loopback: reserved, and nothing legitimate listens there.
    assert TrayApplication("http://127.0.0.1:1")._is_server_up() is False


@pytest.mark.skipif(sys.platform == "win32", reason="Tests the non-Windows refusal.")
def test_other_platforms_are_told_plainly():
    """Rather than failing on a missing ``ctypes.windll``."""
    assert main() == 1
