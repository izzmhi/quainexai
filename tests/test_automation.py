"""Tests for application resolution and path containment.

These are the security tests of Phase 3. They cover the two places where model
output becomes an OS action: the name of a program to launch, and a filesystem
path to open. Only the pure, side-effect-free parts of the Windows controller
are exercised here — anything that would actually launch or terminate something
is covered in ``test_commands.py`` against a fake.
"""

from __future__ import annotations

import pytest

from quainex.config.settings import Settings
from quainex.core.automation.applications import (
    known_application_names,
    resolve_application,
)
from quainex.core.automation.windows import (
    WindowsDesktopController,
    _safe_filename,
)
from quainex.core.exceptions import CommandExecutionError, CommandNotAllowedError

# -- application allowlist -------------------------------------------------


@pytest.mark.parametrize(
    ("spoken", "expected_key"),
    [
        ("vscode", "vscode"),
        ("VS Code", "vscode"),
        ("visual studio code", "vscode"),
        ("my editor", "vscode"),
        ("open the notepad app", "notepad"),
        ("Google Chrome", "chrome"),
        ("chrome", "chrome"),
        ("calc", "calculator"),
        ("file explorer", "explorer"),
        ("SPOTIFY", "spotify"),
    ],
)
def test_spoken_names_resolve_to_the_right_application(spoken, expected_key):
    spec = resolve_application(spoken)
    assert spec is not None
    assert spec.key == expected_key


@pytest.mark.parametrize(
    "spoken",
    [
        "",
        "   ",
        "please",
        "some random program nobody has",
        "cmd /c del /f /s /q C:\\",
        "../../windows/system32/cmd.exe",
    ],
)
def test_unknown_or_hostile_names_do_not_resolve(spoken):
    # Anything not in the catalogue is refused. A name that looks like a shell
    # command is just a name that is not in the catalogue.
    assert resolve_application(spoken) is None


def test_catalogue_names_are_listed_for_error_messages():
    names = known_application_names()
    assert "Visual Studio Code" in names
    assert len(names) == len(set(names)), "display names must be unambiguous"


# -- path containment ------------------------------------------------------


@pytest.fixture
def controller(tmp_path):
    """A controller whose only permitted root is an isolated temp directory."""
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        log_dir=tmp_path / "logs",
        command_search_roots=[tmp_path],
    )
    return WindowsDesktopController(settings)


def test_path_inside_the_root_is_accepted(controller, tmp_path):
    (tmp_path / "projects").mkdir()
    resolved = controller._resolve_within_roots(str(tmp_path / "projects"))
    assert resolved == (tmp_path / "projects").resolve()


@pytest.mark.parametrize(
    "escape",
    [
        "..",
        "../..",
        "../../../../Windows/System32",
        "C:\\Windows\\System32",
        "C:/Windows",
    ],
)
def test_paths_escaping_the_root_are_refused(controller, tmp_path, escape):
    # Traversal is collapsed by resolve() *before* the containment check; the
    # check is meaningless in the other order.
    candidate = str(tmp_path / escape) if escape.startswith("..") else escape
    with pytest.raises(CommandNotAllowedError, match="outside the folders"):
        controller._resolve_within_roots(candidate)


def test_refusal_names_the_allowed_roots(controller):
    with pytest.raises(CommandNotAllowedError) as exc_info:
        controller._resolve_within_roots("C:\\Windows")
    assert "Allowed:" in str(exc_info.value)


# -- URL scheme allowlist --------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "file:///C:/Windows/System32/config/SAM",
        "javascript:alert(1)",
        "ftp://example.com",
        "ms-settings:privacy",
    ],
)
def test_non_web_url_schemes_are_refused(controller, url):
    # Refused before the browser is ever invoked, so no side effect occurs.
    with pytest.raises(CommandNotAllowedError):
        controller.open_url(url)


@pytest.mark.parametrize("url", ["not a url", "http://", "https://"])
def test_urls_without_a_host_are_refused(controller, url):
    with pytest.raises(CommandNotAllowedError):
        controller.open_url(url)


def test_bare_domain_is_upgraded_to_https(controller, monkeypatch):
    opened: list[str] = []
    monkeypatch.setattr(
        "quainex.core.automation.windows.webbrowser.open",
        lambda target: opened.append(target),
    )

    message = controller.open_url("example.com")

    assert opened == ["https://example.com"], "bare domains must not default to http"
    assert "example.com" in message


# -- receiving files: containment, no overwrite, sanitised names -----------


def test_a_received_file_is_saved_inside_the_root(controller, tmp_path):
    dest = tmp_path / "incoming"
    saved = controller.save_incoming_file(b"hello", suggested_name="note.txt", location=str(dest))
    assert saved == dest / "note.txt"
    assert saved.read_bytes() == b"hello"


def test_a_received_file_never_overwrites_an_existing_one(controller, tmp_path):
    dest = tmp_path / "incoming"
    first = controller.save_incoming_file(b"one", suggested_name="note.txt", location=str(dest))
    second = controller.save_incoming_file(b"two", suggested_name="note.txt", location=str(dest))
    assert first.name == "note.txt"
    assert second.name == "note (1).txt"
    assert first.read_bytes() == b"one"
    assert second.read_bytes() == b"two"


def test_a_hostile_file_name_cannot_climb_out_of_the_folder(controller, tmp_path):
    dest = tmp_path / "incoming"
    saved = controller.save_incoming_file(
        b"x", suggested_name="..\\..\\evil.exe", location=str(dest)
    )
    # Only the final component survives, so it lands inside the folder, not above it.
    assert saved.parent == dest
    assert saved.name == "evil.exe"


def test_saving_outside_the_permitted_roots_is_refused(controller):
    with pytest.raises(CommandNotAllowedError, match="outside the folders"):
        controller.save_incoming_file(b"x", suggested_name="a.txt", location="C:/Windows")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("report.pdf", "report.pdf"),
        ("../../etc/passwd", "passwd"),
        ("a\\b\\c.doc", "c.doc"),
        ('bad:name?.txt', "bad_name_.txt"),
    ],
)
def test_file_names_are_reduced_to_a_bare_safe_name(raw, expected):
    assert _safe_filename(raw) == expected


def test_a_reserved_device_name_is_defused():
    assert _safe_filename("CON").startswith("_")


def test_an_empty_or_dotted_name_falls_back(controller):
    assert _safe_filename("///").startswith("file-")
    assert _safe_filename("...").startswith("file-")


def test_type_text_sends_two_keystrokes_per_character(controller, monkeypatch):
    """Each character is a key-down and a key-up, sent via one SendInput call."""
    import ctypes

    captured: dict[str, int] = {}

    def fake_send_input(count, _array, _size):
        captured["count"] = count
        return count  # all events accepted

    monkeypatch.setattr(ctypes.windll.user32, "SendInput", fake_send_input)

    message = controller.type_text("Hi")

    assert captured["count"] == 4  # two characters, each a down and an up
    assert "2 character" in message


def test_typing_nothing_is_refused(controller):
    with pytest.raises(CommandNotAllowedError, match="nothing to type"):
        controller.type_text("   ")


def test_an_application_not_installed_anywhere_is_reported_clearly(controller):
    """Open-any-app changed this from a refusal to a not-found.

    The allowlist is no longer a wall: a name not in it is searched for among the
    machine's installed apps (Start menu, App Paths, PATH). So a genuinely absent
    program is "could not find", not "not allowed" — the allowlist only ever
    *added* names, it never restricted them.
    """
    with pytest.raises(CommandExecutionError) as exc_info:
        # A name that matches nothing curated and nothing installed.
        controller.open_application("Nonexistent Program Xyzzy 12345")

    assert "Could not find" in str(exc_info.value)
