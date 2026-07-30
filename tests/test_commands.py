"""Tests for the command layer: policy gates, dispatch, and the endpoints.

Every test runs against ``FakeDesktopController``. Nothing here launches a
process, writes a clipboard, or powers anything off — the point of putting the
controller behind a Protocol was to make this suite safe to run.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from quainex.config.settings import Settings
from quainex.core.automation.desktop import FileHit, LevelChange, SystemSnapshot
from quainex.core.brain import Intent, IntentType
from quainex.core.commands import CommandStatus, build_executor
from quainex.core.commands.base import Command, CommandContext, CommandOutcome
from quainex.core.commands.executor import CommandExecutor, CommandRegistry
from quainex.core.exceptions import CommandExecutionError, CommandNotAllowedError

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


class FakeDesktopController:
    """Records what it was asked to do instead of doing it."""

    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[tuple[str, object]] = []
        self._error = error
        # Files the fake "sends" are written here, so the Telegram document-upload
        # path has something real to read.
        self._file_dir = Path(tempfile.mkdtemp(prefix="quainex-fake-files-"))

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

    def create_folder(self, name: str) -> str:
        self._record("create_folder", name)
        return f"Created folder C:/fake/{name}."

    def search_files(self, query: str, limit: int) -> list[FileHit]:
        self._record("search_files", (query, limit))
        return [FileHit(path="C:/fake/report.pdf", size_bytes=10)]

    def resolve_file_for_sending(self, query: str) -> Path:
        self._record("resolve_file_for_sending", query)
        # A real file so the Telegram bridge can read it back, like the fakes for
        # screenshot and webcam.
        target = self._file_dir / f"{query.replace('/', '_')}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fake file contents")
        return target

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

    def capture_webcam(self, destination: Path) -> str:
        # Writes a file, like the real one: the Telegram bridge reads it back to
        # upload, so a fake that only records would make that path untestable.
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"\xff\xd8\xff fake jpeg")
        return self._record("capture_webcam", destination)

    def set_wifi(self, *, enabled: bool) -> str:
        return self._record("set_wifi", enabled)

    def wifi_status(self) -> str:
        self._record("wifi_status")
        return "Wi-Fi is connected to 'TestNet'."

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
    utterance: str = "",
) -> Intent:
    return Intent(
        intent=intent,
        target=target,
        confidence=0.95,
        reasoning="test",
        requires_confirmation=requires_confirmation,
        utterance=utterance,
    )


class FakeConversationalist:
    """Records what it was asked and returns a fixed reply.

    Attributes:
        asked: Every ``(message, kind)`` pair received, so a test can prove the
            handler passed the right text through.
    """

    def __init__(self, reply: str = "I'm running fine.") -> None:
        self.asked: list[tuple[str, IntentType]] = []
        self._reply = reply

    @property
    def is_available(self) -> bool:
        return True

    async def reply(self, *, message: str, kind: IntentType) -> str:
        self.asked.append((message, kind))
        return self._reply


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


def test_every_intent_the_brain_can_produce_has_a_command(tmp_path):
    """No classification may be a dead end.

    This test replaces one that asserted the opposite — that the conversational
    intents were *not* executable. That was the bug, written down as if it were
    the design: the Brain would correctly classify "how are you doing?" as
    ``small_talk`` and the executor would answer "'small_talk' is not something
    Quainex can execute". Accurate, and useless.

    Anything the Brain can return must now be dispatchable, so adding an
    ``IntentType`` without a handler fails here rather than in front of a user.
    """
    catalogue = build_executor(FakeDesktopController(), _settings(tmp_path)).catalogue

    missing = [intent.value for intent in IntentType if intent.value not in catalogue]
    assert missing == []


# -- gate 1: unsupported ---------------------------------------------------


async def test_an_unregistered_intent_is_unsupported(tmp_path):
    """Gate 1 still holds, even though every built-in intent now has a handler.

    Exercised against an empty registry rather than a conversational intent,
    because the gate's job is to refuse a *dispatch with no implementation* — a
    plugin registering an intent it does not implement, or an ``IntentType`` added
    without a command. That the built-in set is now complete does not retire it.
    """
    desktop = FakeDesktopController()
    executor = CommandExecutor(
        registry=CommandRegistry([]),
        context=CommandContext(desktop=desktop),
        settings=_settings(tmp_path),
    )

    result = await executor.execute(_intent(IntentType.OPEN_APPLICATION))

    assert result.status is CommandStatus.UNSUPPORTED
    assert result.executed is False
    assert desktop.calls == []


# -- conversation ----------------------------------------------------------


@pytest.mark.parametrize(
    "kind",
    [IntentType.SMALL_TALK, IntentType.ANSWER_QUESTION, IntentType.UNKNOWN],
)
async def test_conversational_intents_are_answered_not_refused(tmp_path, kind: IntentType):
    """The fix, stated as three assertions.

    Each of these used to come back as ``unsupported``. An AI operating system
    that cannot answer a greeting is broken in the first thirty seconds of use.
    """
    conversation = FakeConversationalist("Running fine.")
    result = await build_executor(
        FakeDesktopController(), _settings(tmp_path), conversation=conversation
    ).execute(_intent(kind, target=None, utterance="how are you doing?"))

    assert result.status is CommandStatus.SUCCEEDED
    assert result.message == "Running fine."
    assert conversation.asked == [("how are you doing?", kind)]
    # `executed` means "a side effect occurred", and a reply has none. Reporting
    # True would put "small_talk executed=True" in the audit trail, which reads as
    # the machine having been changed.
    assert result.executed is False


async def test_a_greeting_with_no_target_still_reaches_the_responder(tmp_path):
    """``small_talk`` has no target by definition.

    So the handler falls back to the utterance. Registering these commands with
    ``requires_target=True`` would have reintroduced the same failure at the
    executor's fourth gate instead of its first — a refusal wearing a new label.
    """
    conversation = FakeConversationalist()
    result = await build_executor(
        FakeDesktopController(), _settings(tmp_path), conversation=conversation
    ).execute(_intent(IntentType.SMALL_TALK, target=None, utterance="morning"))

    assert result.status is CommandStatus.SUCCEEDED
    assert conversation.asked[0][0] == "morning"


async def test_a_question_prefers_the_extracted_target_over_the_raw_utterance(tmp_path):
    """The classifier's ``target`` is the question stripped of framing."""
    conversation = FakeConversationalist()
    await build_executor(
        FakeDesktopController(), _settings(tmp_path), conversation=conversation
    ).execute(
        _intent(
            IntentType.ANSWER_QUESTION,
            target="how tall is Everest",
            utterance="hey quainex, how tall is Everest?",
        )
    )

    assert conversation.asked[0][0] == "how tall is Everest"


async def test_a_conversational_reply_cannot_touch_the_desktop(tmp_path):
    """The reason these handlers reach ``ctx.conversation`` and nothing else.

    A reply claiming "I've opened that for you" when nothing was opened is worse
    than an error, and the only reliable way to prevent it is to make the action
    unreachable from the conversational path.
    """
    desktop = FakeDesktopController()
    await build_executor(
        desktop, _settings(tmp_path), conversation=FakeConversationalist()
    ).execute(_intent(IntentType.SMALL_TALK, target=None, utterance="open notepad for me"))

    assert desktop.calls == []


async def test_conversation_without_a_responder_is_blocked_not_crashed(tmp_path):
    """A desktop-only executor — most tests — must still refuse cleanly."""
    result = await build_executor(FakeDesktopController(), _settings(tmp_path)).execute(
        _intent(IntentType.SMALL_TALK, target=None, utterance="hello")
    )

    assert result.status is CommandStatus.BLOCKED
    assert result.executed is False
    assert "Conversation" in result.message


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


async def test_create_folder_reaches_the_controller(tmp_path):
    desktop = FakeDesktopController()
    result = await build_executor(desktop, _settings(tmp_path)).execute(
        _intent(IntentType.CREATE_FOLDER, "projects")
    )

    assert result.status is CommandStatus.SUCCEEDED
    assert ("create_folder", "projects") in desktop.calls


async def test_send_file_returns_a_readable_path_for_upload(tmp_path):
    """A real file, reported as no side effect.

    The Telegram bridge reads this path back to upload the file, so it must exist;
    and nothing on the machine changed, so executed must be False.
    """
    desktop = FakeDesktopController()
    result = await build_executor(desktop, _settings(tmp_path)).execute(
        _intent(IntentType.SEND_FILE, "report.pdf")
    )

    assert result.status is CommandStatus.SUCCEEDED
    assert result.executed is False
    assert result.data is not None
    assert Path(result.data["path"]).is_file()


async def test_webcam_captures_and_returns_a_path(tmp_path):
    """The path is what the Telegram bridge reads back to upload the photo."""
    desktop = FakeDesktopController()
    result = await build_executor(desktop, _settings(tmp_path)).execute(
        _intent(IntentType.WEBCAM, target=None)
    )

    assert result.status is CommandStatus.SUCCEEDED
    assert result.data is not None
    assert result.data["path"].endswith(".jpg")
    assert Path(result.data["path"]).is_file()


@pytest.mark.parametrize(
    ("target", "call"),
    [("on", ("set_wifi", True)), ("off", ("set_wifi", False)), ("status", ("wifi_status", None))],
)
async def test_wifi_control_reaches_the_controller(tmp_path, target, call):
    desktop = FakeDesktopController()
    result = await build_executor(desktop, _settings(tmp_path)).execute(
        _intent(IntentType.WIFI, target)
    )

    assert result.status is CommandStatus.SUCCEEDED
    assert call in desktop.calls


async def test_wifi_with_a_nonsense_target_is_refused_clearly(tmp_path):
    result = await build_executor(FakeDesktopController(), _settings(tmp_path)).execute(
        _intent(IntentType.WIFI, "sideways")
    )

    assert result.status is CommandStatus.FAILED
    assert "on, off, or status" in result.message


async def test_web_search_opens_the_browser_without_a_model(tmp_path, monkeypatch):
    """The browser open is deterministic and token-free; the summary is a bonus.

    The instant-answer lookup is stubbed so the test makes no network call — what
    matters here is that the browser was opened with the query, which is the part
    the user asked for.
    """
    import quainex.core.commands.builtin as builtin

    async def no_answer(_query: str) -> None:
        return None

    monkeypatch.setattr(builtin, "_instant_answer", no_answer)

    desktop = FakeDesktopController()
    result = await build_executor(desktop, _settings(tmp_path)).execute(
        _intent(IntentType.WEB_SEARCH, "weather in lagos")
    )

    assert result.status is CommandStatus.SUCCEEDED
    opened = next(payload for action, payload in desktop.calls if action == "open_url")
    assert "duckduckgo.com" in opened
    assert "weather" in opened
    assert result.data is not None
    assert result.data["query"] == "weather in lagos"


async def test_web_search_includes_a_summary_when_one_exists(tmp_path, monkeypatch):
    import quainex.core.commands.builtin as builtin

    async def answer(_query: str) -> str:
        return "Lagos is a city in Nigeria."

    monkeypatch.setattr(builtin, "_instant_answer", answer)

    result = await build_executor(FakeDesktopController(), _settings(tmp_path)).execute(
        _intent(IntentType.WEB_SEARCH, "lagos")
    )

    assert "Lagos is a city in Nigeria." in result.message


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
    # No API key in the test container, so interpretation cannot happen — for
    # anything the local fast path does not already handle.
    response = client.post("/commands/ask", json={"utterance": "bring up my editor"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "provider_not_configured"


def test_common_commands_work_with_no_api_key_at_all(client: TestClient):
    """A property that fell out of the fast path, and is worth keeping on purpose.

    The test container has no credentials, and this still succeeds: the classifier
    never runs. So Quainex's most common operations work before you have signed up
    for anything, when every provider is out of quota, and with the network
    unplugged — which is a far better first impression than a 503.
    """
    response = client.post("/commands/ask", json={"utterance": "take a screenshot"})

    assert response.status_code == 200
    body = response.json()
    assert body["intent"]["intent"] == "screenshot"
    assert body["result"]["status"] == "succeeded"
