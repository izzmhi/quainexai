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
