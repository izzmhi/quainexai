"""Tests for the Telegram phone bridge.

Nothing here talks to Telegram. What is verified is the logic that matters when
a third party sits in the middle of your commands: who is obeyed, what is
refused, and that a confirmation still has to be a real one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from quainex.config.settings import Settings
from quainex.core.brain import Brain, IntentClassification, IntentType
from quainex.core.commands import build_executor
from quainex.core.exceptions import ProviderError
from quainex.integrations.telegram import (
    TELEGRAM_BLOCKED_INTENTS,
    TelegramBridge,
    _parse_update,
    _truncate,
)
from quainex.security import ConfirmationService
from tests.test_brain import FakeProvider
from tests.test_commands import FakeDesktopController

ALLOWED_USER = 12345
STRANGER = 99999


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    base: dict[str, object] = {
        "_env_file": None,
        "log_dir": tmp_path / "logs",
        "database_path": tmp_path / "t.db",
        "command_search_roots": [tmp_path],
        "screenshot_dir": tmp_path / "shots",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _bridge(
    tmp_path: Path,
    intent: IntentType = IntentType.OPEN_APPLICATION,
    desktop: FakeDesktopController | None = None,
    **setting_overrides: object,
) -> TelegramBridge:
    settings = _settings(tmp_path, **setting_overrides)
    provider = FakeProvider(
        IntentClassification(intent=intent, target="VS Code", confidence=0.97, reasoning="test")
    )
    return TelegramBridge(
        settings,
        brain=Brain(provider=provider, settings=settings),
        commands=build_executor(
            desktop or FakeDesktopController(), settings, ConfirmationService("t" * 48)
        ),
    )


# -- configuration ---------------------------------------------------------


def test_bridge_is_off_without_a_token(tmp_path):
    assert _bridge(tmp_path).is_configured is False


def test_bridge_is_off_without_an_allowlist(tmp_path):
    # A bot with no allowlist would take orders from anyone who found it, so an
    # empty list means off rather than "everyone".
    bridge = _bridge(tmp_path, telegram_bot_token="123:abc", telegram_allowed_users=[])
    assert bridge.is_configured is False


# -- setup diagnostics -----------------------------------------------------


async def test_diagnose_says_so_when_there_is_no_token(tmp_path):
    """A diagnostic must not need the network to report the obvious."""
    result = await _bridge(tmp_path).diagnose()

    assert result["ok"] is False
    assert "No bot token" in str(result["error"])
    assert result["candidates"] == []


def test_senders_are_deduplicated_and_named_for_recognition_only():
    """The id authorises; the name only helps a human recognise it.

    A display name is chosen by its owner and can be anything, including a copy
    of someone else's — which is precisely why it is not what grants access.
    """
    from quainex.integrations.telegram import _senders

    candidates = _senders(
        [
            {"message": {"from": {"id": 111, "first_name": "Sam", "username": "sam"}}},
            # Same person again: one entry, not two.
            {"message": {"from": {"id": 111, "first_name": "Sam"}}},
            {"callback_query": {"from": {"id": 222, "first_name": "Alex", "last_name": "Lee"}}},
            {"edited_message": {"from": {"id": 333}}},
            # Malformed and anonymous updates must not produce entries.
            {"message": {}},
            {"channel_post": {"text": "hi"}},
            {"message": {"from": {"id": "not-an-int"}}},
        ]
    )

    assert [entry["user_id"] for entry in candidates] == [111, 222, 333]
    assert candidates[0]["username"] == "sam"
    assert candidates[1]["name"] == "Alex Lee"
    assert candidates[2]["name"] == ""


def test_bridge_is_configured_with_both(tmp_path):
    bridge = _bridge(tmp_path, telegram_bot_token="123:abc", telegram_allowed_users=[ALLOWED_USER])
    assert bridge.is_configured is True


async def test_running_unconfigured_is_refused(tmp_path):
    with pytest.raises(RuntimeError, match="not configured"):
        await _bridge(tmp_path).run()


def test_status_reports_what_is_blocked(tmp_path):
    status = _bridge(tmp_path).status()
    assert status["configured"] is False
    assert "clipboard" in status["blocked_intents"]
    assert "look_at_screen" in status["blocked_intents"]


# -- failures must be visible on the phone ---------------------------------


class SendRecorder:
    """Captures what the bridge tried to send, instead of sending it.

    Attributes:
        sent: The text of every outbound message.
    """

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def post(self, url: str, **kwargs: object) -> object:
        json_body = kwargs.get("json")
        if isinstance(json_body, dict):
            self.sent.append(str(json_body.get("text", "")))
        return _OkResponse()


class _OkResponse:
    """Minimal stand-in for a successful Telegram response."""

    status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {"ok": True, "result": {}}


async def test_a_provider_failure_is_reported_to_the_chat_not_only_the_log(tmp_path):
    """The bug that made a working bridge look dead.

    Every provider was out of quota, the exception was logged, and *nothing* was
    sent back. From the phone that is indistinguishable from a crashed server —
    and the log is on a machine the user is not looking at, which is the whole
    reason they are on Telegram.
    """
    bridge = _bridge(tmp_path, telegram_bot_token="123:abc", telegram_allowed_users=[ALLOWED_USER])
    recorder = SendRecorder()

    # The Brain raises exactly as it did in the real failure.
    async def exhausted(*_args: object, **_kwargs: object) -> None:
        raise ProviderError("Every AI provider failed. groq: 429; gemini: 429")

    bridge._brain.interpret = exhausted  # type: ignore[method-assign]

    await bridge._dispatch(
        recorder,  # type: ignore[arg-type]
        _parse_update(
            {
                "update_id": 1,
                "message": {"chat": {"id": 7}, "from": {"id": ALLOWED_USER}, "text": "hello"},
            }
        ),
    )

    assert len(recorder.sent) == 1
    reply = recorder.sent[0]
    # Says it is a quota rather than a fault, because that changes what the user
    # should do about it.
    assert "quota" in reply.lower()
    # And does not paste three nested vendor JSON blobs into a phone screen.
    assert "429" not in reply


async def test_an_unexpected_failure_stays_generic(tmp_path):
    """A traceback is not something to put in a chat transcript."""
    bridge = _bridge(tmp_path, telegram_bot_token="123:abc", telegram_allowed_users=[ALLOWED_USER])
    recorder = SendRecorder()

    async def boom(*_args: object, **_kwargs: object) -> None:
        raise ValueError("internal detail nobody should see")

    bridge._brain.interpret = boom  # type: ignore[method-assign]

    await bridge._dispatch(
        recorder,  # type: ignore[arg-type]
        _parse_update(
            {
                "update_id": 1,
                "message": {"chat": {"id": 7}, "from": {"id": ALLOWED_USER}, "text": "hello"},
            }
        ),
    )

    assert len(recorder.sent) == 1
    assert "internal detail nobody should see" not in recorder.sent[0]
    assert "went wrong" in recorder.sent[0]


async def test_a_stranger_still_gets_no_reply_at_all(tmp_path):
    """Error reporting must not become an oracle.

    Replying to an unknown sender — even to refuse — confirms the bot exists and
    is listening, which is information they have not earned.
    """
    bridge = _bridge(tmp_path, telegram_bot_token="123:abc", telegram_allowed_users=[ALLOWED_USER])
    recorder = SendRecorder()

    await bridge._dispatch(
        recorder,  # type: ignore[arg-type]
        _parse_update(
            {
                "update_id": 1,
                "message": {"chat": {"id": 7}, "from": {"id": STRANGER}, "text": "hello"},
            }
        ),
    )

    assert recorder.sent == []


# -- privacy: what must not leave the machine -----------------------------


def test_output_revealing_intents_are_blocked():
    # These are not dangerous actions — the danger is that their *output* would
    # land in a third-party chat log.
    assert IntentType.CLIPBOARD in TELEGRAM_BLOCKED_INTENTS
    assert IntentType.LOOK_AT_SCREEN in TELEGRAM_BLOCKED_INTENTS
    assert IntentType.READ_DOCUMENT in TELEGRAM_BLOCKED_INTENTS


def test_ordinary_actions_are_not_blocked():
    for intent in (IntentType.OPEN_APPLICATION, IntentType.SYSTEM_INFO, IntentType.LIST_WINDOWS):
        assert intent not in TELEGRAM_BLOCKED_INTENTS


# -- update parsing --------------------------------------------------------


def test_text_message_is_parsed():
    update = _parse_update(
        {
            "update_id": 7,
            "message": {
                "chat": {"id": 111},
                "from": {"id": ALLOWED_USER},
                "text": "open vs code",
            },
        }
    )
    assert update.update_id == 7
    assert update.chat_id == 111
    assert update.user_id == ALLOWED_USER
    assert update.text == "open vs code"


def test_voice_note_is_parsed():
    update = _parse_update(
        {
            "update_id": 8,
            "message": {
                "chat": {"id": 111},
                "from": {"id": ALLOWED_USER},
                "voice": {"file_id": "AwACAgQAA"},
            },
        }
    )
    assert update.voice_file_id == "AwACAgQAA"
    assert update.text is None


def test_button_tap_is_parsed():
    update = _parse_update(
        {
            "update_id": 9,
            "callback_query": {
                "id": "cb1",
                "from": {"id": ALLOWED_USER},
                "data": "yes:c0shutdown",
                "message": {"chat": {"id": 111}},
            },
        }
    )
    assert update.callback_data == "yes:c0shutdown"
    assert update.callback_id == "cb1"
    assert update.chat_id == 111


def test_an_empty_update_does_not_crash_the_parser():
    # Telegram sends update kinds the bridge does not handle; they must parse to
    # something inert rather than raising inside the poll loop.
    update = _parse_update({"update_id": 10})
    assert update.chat_id is None
    assert update.user_id is None


# -- message limits --------------------------------------------------------


def test_short_messages_pass_through():
    assert _truncate("hello") == "hello"


def test_long_messages_are_trimmed_to_telegrams_limit():
    trimmed = _truncate("x" * 10_000)
    assert len(trimmed) < 4096, "Telegram rejects messages over 4096 characters"
    assert trimmed.endswith("…(truncated)")


# -- built-in commands answer locally, without the model ------------------


@pytest.mark.parametrize("command", ["/start", "/help"])
def test_help_is_answered_without_calling_the_model(tmp_path, command):
    reply = _bridge(tmp_path)._builtin(command)
    assert "Quainex" in reply
    assert "voice note" in reply


def test_status_command_names_what_is_disabled(tmp_path):
    reply = _bridge(tmp_path)._builtin("/status")
    assert "clipboard" in reply
    assert "commands available" in reply


def test_unknown_slash_commands_are_reported(tmp_path):
    assert "Unknown command" in _bridge(tmp_path)._builtin("/launch_missiles")
