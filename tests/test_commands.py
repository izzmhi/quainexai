"""Tests for the command layer: policy gates, dispatch, and the endpoints.

Every test runs against ``FakeDesktopController``. Nothing here launches a
process, writes a clipboard, or powers anything off — the point of putting the
controller behind a Protocol was to make this suite safe to run.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from quainex.config.settings import Settings
from quainex.core.automation.desktop import FileHit, LevelChange, SystemSnapshot
from quainex.core.brain import Intent, IntentType
from quainex.core.commands import CommandStatus, build_executor
from quainex.core.commands.base import Command, CommandOutcome
from quainex.core.commands.executor import CommandRegistry
from quainex.core.exceptions import CommandExecutionError, CommandNotAllowedError

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


class FakeDesktopController:
    """Records what it was asked to do instead of doing it."""

    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[tuple[str, object]] = []
        self._error = error

    def _record(self, action: str, payload: object = None) -> str:
        if self._error is not None:
            raise self._error
        self.calls.append((action, payload))
        return f"fake:{action}"

    def open_application(self, name: str) -> str:
        return self._record("open_application", name)

    def close_application(self, name: str) -> str:
        return self._record("close_application", name)

    def open_url(self, url: str) -> str:
        return self._record("open_url", url)

    def open_folder(self, path: str) -> str:
        return self._record("open_folder", path)

    def search_files(self, query: str, limit: int) -> list[FileHit]:
        self._record("search_files", (query, limit))
        return [FileHit(path="C:/fake/report.pdf", size_bytes=10)]

    def lock_screen(self) -> str:
        return self._record("lock_screen")

    def sleep(self) -> str:
        return self._record("sleep")

    def restart(self, delay_seconds: int) -> str:
        return self._record("restart", delay_seconds)

    def shutdown(self, delay_seconds: int) -> str:
        return self._record("shutdown", delay_seconds)

    def set_volume(self, change: LevelChange) -> str:
        return self._record("set_volume", change)

    def set_brightness(self, change: LevelChange) -> str:
        return self._record("set_brightness", change)

    def screenshot(self, destination: Path) -> str:
        # Actually writes a file: the real controller's contract is that the
        # path exists afterwards, and the vision tests check that a captured
        # screenshot is deleted again. A fake that records without writing
        # would make that assertion vacuous.
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"\x89PNG\r\n\x1a\n fake screenshot")
        return self._record("screenshot", destination)

    def read_clipboard(self) -> str:
        self._record("read_clipboard")
        return "clipboard contents"

    def write_clipboard(self, text: str) -> str:
        return self._record("write_clipboard", text)

    def notify(self, message: str, title: str) -> str:
        return self._record("notify", (title, message))

    def system_info(self) -> SystemSnapshot:
        self._record("system_info")
        return SystemSnapshot(
            cpu_percent=12.0,
            memory_percent=44.0,
            disk_percent=61.0,
            battery_percent=88.0,
            uptime_seconds=3600.0,
        )

    @property
    def actions(self) -> list[str]:
        return [action for action, _ in self.calls]


def _settings(tmp_path: Path, *, allow_destructive: bool = False) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        log_dir=tmp_path / "logs",
        allow_destructive_commands=allow_destructive,
        command_search_roots=[tmp_path],
        screenshot_dir=tmp_path / "shots",
    )


def _intent(
    intent: IntentType = IntentType.OPEN_APPLICATION,
    target: str | None = "VS Code",
    *,
    requires_confirmation: bool = False,
) -> Intent:
    return Intent(
        intent=intent,
        target=target,
        confidence=0.95,
        reasoning="test",
        requires_confirmation=requires_confirmation,
    )


# -- registry --------------------------------------------------------------


def test_duplicate_registration_is_rejected():
    def handler(_desktop, _intent):
        return CommandOutcome(message="x")

    command = Command(intent=IntentType.SCREENSHOT, summary="a", handler=handler)
    with pytest.raises(ValueError, match="Duplicate command"):
        CommandRegistry([command, command])


def test_catalogue_lists_every_registered_command(tmp_path):
    executor = build_executor(FakeDesktopController(), _settings(tmp_path))
    catalogue = executor.catalogue

    assert "open_application" in catalogue
    assert "shutdown" in catalogue
    # Conversational intents are not executable.
    assert "small_talk" not in catalogue
    assert "answer_question" not in catalogue


# -- gate 1: unsupported ---------------------------------------------------


async def test_non_executable_intent_is_unsupported(tmp_path):
    desktop = FakeDesktopController()
    result = await build_executor(desktop, _settings(tmp_path)).execute(
        _intent(IntentType.SMALL_TALK, target=None)
    )

    assert result.status is CommandStatus.UNSUPPORTED
    assert result.executed is False
    assert desktop.calls == []


# -- gate 2: confirmation --------------------------------------------------


async def test_unconfirmed_intent_is_not_executed(tmp_path):
    desktop = FakeDesktopController()
    result = await build_executor(desktop, _settings(tmp_path)).execute(
        _intent(IntentType.CLOSE_APPLICATION, "Spotify", requires_confirmation=True)
    )

    assert result.status is CommandStatus.REQUIRES_CONFIRMATION
    assert result.executed is False
    assert "Spotify" in result.message
    assert desktop.calls == [], "nothing may happen before the user says yes"


async def test_confirmation_unlocks_execution(tmp_path):
    desktop = FakeDesktopController()
    result = await build_executor(desktop, _settings(tmp_path)).execute(
        _intent(IntentType.CLOSE_APPLICATION, "Spotify", requires_confirmation=True),
        confirmed=True,
    )

    assert result.status is CommandStatus.SUCCEEDED
    assert result.executed is True
    assert desktop.actions == ["close_application"]


# -- gate 3: destructive switch -------------------------------------------


@pytest.mark.parametrize("intent_type", [IntentType.SHUTDOWN, IntentType.RESTART, IntentType.SLEEP])
async def test_power_actions_are_blocked_by_default(intent_type, tmp_path):
    desktop = FakeDesktopController()
    result = await build_executor(desktop, _settings(tmp_path)).execute(
        _intent(intent_type, target=None), confirmed=True
    )

    assert result.status is CommandStatus.BLOCKED
    assert result.executed is False
    assert "QUAINEX_ALLOW_DESTRUCTIVE_COMMANDS" in result.message
    assert desktop.calls == []


async def test_power_actions_run_when_explicitly_enabled(tmp_path):
    desktop = FakeDesktopController()
    executor = build_executor(desktop, _settings(tmp_path, allow_destructive=True))
    result = await executor.execute(_intent(IntentType.SHUTDOWN, target=None), confirmed=True)

    assert result.status is CommandStatus.SUCCEEDED
    assert desktop.actions == ["shutdown"]


async def test_confirmation_is_checked_before_the_destructive_switch(tmp_path):
    # Both gates would refuse; the user-facing one must win, so the message is
    # the actionable "confirm?" rather than a configuration lecture.
    desktop = FakeDesktopController()
    result = await build_executor(desktop, _settings(tmp_path)).execute(
        _intent(IntentType.SHUTDOWN, target=None, requires_confirmation=True)
    )

    assert result.status is CommandStatus.REQUIRES_CONFIRMATION
    assert desktop.calls == []


# -- gate 4: missing target ------------------------------------------------


@pytest.mark.parametrize("target", [None, "", "   "])
async def test_missing_target_fails_before_touching_the_os(target, tmp_path):
    desktop = FakeDesktopController()
    result = await build_executor(desktop, _settings(tmp_path)).execute(
        _intent(IntentType.OPEN_APPLICATION, target)
    )

    assert result.status is CommandStatus.FAILED
    assert result.executed is False
    assert desktop.calls == []


# -- dispatch --------------------------------------------------------------


async def test_successful_command_reports_execution(tmp_path):
    desktop = FakeDesktopController()
    result = await build_executor(desktop, _settings(tmp_path)).execute(_intent())

    assert result.status is CommandStatus.SUCCEEDED
    assert result.executed is True
    assert result.ok is True
    assert desktop.calls == [("open_application", "VS Code")]


async def test_search_results_are_returned_as_data(tmp_path):
    desktop = FakeDesktopController()
    result = await build_executor(desktop, _settings(tmp_path)).execute(
        _intent(IntentType.SEARCH_FILES, "report")
    )

    assert result.status is CommandStatus.SUCCEEDED
    assert result.data is not None
    assert result.data["results"][0]["path"] == "C:/fake/report.pdf"


async def test_system_info_is_returned_as_data(tmp_path):
    desktop = FakeDesktopController()
    result = await build_executor(desktop, _settings(tmp_path)).execute(
        _intent(IntentType.SYSTEM_INFO, target=None)
    )

    assert result.data is not None
    assert result.data["cpu_percent"] == 12.0
    assert "CPU 12%" in result.message


async def test_clipboard_write_uses_parameters(tmp_path):
    from quainex.core.brain import IntentParameter

    desktop = FakeDesktopController()
    intent = Intent(
        intent=IntentType.CLIPBOARD,
        target=None,
        confidence=0.9,
        reasoning="test",
        requires_confirmation=False,
        parameters=[
            IntentParameter(key="action", value="write"),
            IntentParameter(key="text", value="hello"),
        ],
    )
    result = await build_executor(desktop, _settings(tmp_path)).execute(intent)

    assert result.status is CommandStatus.SUCCEEDED
    assert desktop.calls == [("write_clipboard", "hello")]


async def test_numeric_level_is_coerced(tmp_path):
    desktop = FakeDesktopController()
    await build_executor(desktop, _settings(tmp_path)).execute(
        _intent(IntentType.SET_BRIGHTNESS, "40%")
    )
    assert desktop.calls == [("set_brightness", 40)]


# -- controller failures ---------------------------------------------------


async def test_refusal_from_the_controller_is_blocked_not_failed(tmp_path):
    # The distinction matters for auditing: nothing happened here.
    desktop = FakeDesktopController(error=CommandNotAllowedError("'Doom' is not allowed"))
    result = await build_executor(desktop, _settings(tmp_path)).execute(_intent(target="Doom"))

    assert result.status is CommandStatus.BLOCKED
    assert result.executed is False
    assert "not allowed" in result.message


async def test_execution_failure_is_reported_as_failed(tmp_path):
    desktop = FakeDesktopController(error=CommandExecutionError("not installed"))
    result = await build_executor(desktop, _settings(tmp_path)).execute(_intent())

    assert result.status is CommandStatus.FAILED
    assert result.executed is False
    assert "not installed" in result.message


# -- HTTP endpoints --------------------------------------------------------


def _install_fake_desktop(client: TestClient, desktop: FakeDesktopController) -> None:
    container = client.app.state.container
    container.desktop = desktop
    container.commands = build_executor(desktop, container.settings)


def test_catalogue_endpoint(client: TestClient):
    response = client.get("/commands")
    assert response.status_code == 200
    assert "open_application" in response.json()


def test_execute_endpoint_runs_a_command(client: TestClient):
    desktop = FakeDesktopController()
    _install_fake_desktop(client, desktop)

    response = client.post(
        "/commands/execute",
        json={"intent": _intent().model_dump(mode="json"), "confirmed": False},
    )
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "succeeded"
    assert body["executed"] is True
    assert desktop.actions == ["open_application"]


def test_execute_endpoint_refuses_unconfirmed_with_200(client: TestClient):
    # A refusal is a definite answer to a legitimate question, not an error.
    desktop = FakeDesktopController()
    _install_fake_desktop(client, desktop)

    response = client.post(
        "/commands/execute",
        json={
            "intent": _intent(
                IntentType.CLOSE_APPLICATION, "Spotify", requires_confirmation=True
            ).model_dump(mode="json"),
            "confirmed": False,
        },
    )
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "requires_confirmation"
    assert body["executed"] is False
    assert desktop.calls == []


def test_ask_endpoint_requires_a_configured_brain(client: TestClient):
    # No API key in the test container, so interpretation cannot happen.
    response = client.post("/commands/ask", json={"utterance": "Open VS Code"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "provider_not_configured"
