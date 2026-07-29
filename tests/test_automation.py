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
from quainex.core.automation.windows import WindowsDesktopController
from quainex.core.exceptions import CommandNotAllowedError

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


def test_unknown_application_refusal_lists_alternatives(controller):
    with pytest.raises(CommandNotAllowedError) as exc_info:
        controller.open_application("Doom Eternal")
    message = str(exc_info.value)
    assert "not an allowed application" in message
    assert "Visual Studio Code" in message, "a refusal should say what is available"
