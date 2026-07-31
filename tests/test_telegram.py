"""Tests for the Telegram phone bridge.

Nothing here talks to Telegram. What is verified is the logic that matters when
a third party sits in the middle of your commands: who is obeyed, what is
refused, and that a confirmation still has to be a real one.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import httpx
import pytest

from quainex.config.settings import Settings
from quainex.core.brain import Brain, Intent, IntentClassification, IntentType
from quainex.core.commands import CommandStatus, build_executor
from quainex.core.commands.base import CommandResult
from quainex.core.exceptions import ProviderError
from quainex.integrations.telegram import (
    TELEGRAM_BLOCKED_INTENTS,
    TelegramBridge,
    _esc,
    _parse_update,
    _plain,
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


# -- diagnostics must not break the thing they diagnose --------------------


class _OkResponse:
    """Minimal stand-in for a successful Telegram response."""

    status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {"ok": True, "result": {}}


async def test_diagnose_does_not_poll_while_the_bridge_is_running(tmp_path, monkeypatch):
    """Telegram allows one ``getUpdates`` per bot and 409s the loser.

    So a "check my bot" button that polls terminates the bridge's own long poll —
    which is exactly what happened: pressing it produced a run of
    ``telegram_poll_failed`` entries and made a healthy bridge look broken.
    """
    bridge = _bridge(tmp_path, telegram_bot_token="123:abc", telegram_allowed_users=[ALLOWED_USER])
    bridge._running = True
    bridge._seen_senders = {ALLOWED_USER: {"user_id": ALLOWED_USER, "name": "Sam", "username": ""}}

    requested: list[str] = []

    class SpyClient:
        async def __aenter__(self) -> SpyClient:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def get(self, url: str, **_: object) -> _OkResponse:
            requested.append(url.rsplit("/", 1)[-1])
            return _OkResponse()

    monkeypatch.setattr("quainex.integrations.telegram.httpx.AsyncClient", lambda **_: SpyClient())

    result = await bridge.diagnose()

    assert "getMe" in requested
    assert "getUpdates" not in requested
    # Candidates still come back — from what the bridge has already seen, which is
    # better information than one pending update anyway.
    assert result["candidates"] == [{"user_id": ALLOWED_USER, "name": "Sam", "username": ""}]


async def test_a_conflict_stops_the_loop_instead_of_retrying_forever(tmp_path, monkeypatch):
    """409 means another instance is polling, and retrying cannot fix that.

    The two simply terminate each other's long poll forever: every poll fails,
    messages arrive erratically or not at all, and the bridge still reports itself
    as running. That is exactly how one leftover process became hours of
    debugging — the log filled with `telegram_poll_failed` and nothing said why.

    Retrying a conflict is not resilience. It is a busy loop hiding a
    configuration error, so this one status is terminal and says what to do.
    """
    bridge = _bridge(tmp_path, telegram_bot_token="123:abc", telegram_allowed_users=[ALLOWED_USER])

    class ConflictClient:
        def __init__(self) -> None:
            self.polls = 0

        async def __aenter__(self) -> ConflictClient:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def get(self, url: str, **_: object) -> httpx.Response:
            self.polls += 1
            request = httpx.Request("GET", url)
            response = httpx.Response(409, request=request, text="Conflict")
            raise httpx.HTTPStatusError("conflict", request=request, response=response)

        async def post(self, url: str, **_: object) -> _OkResponse:
            # Startup registers a command menu and sends an "online" ping before
            # polling; both post, and both are best-effort.
            return _OkResponse()

    client = ConflictClient()
    monkeypatch.setattr("quainex.integrations.telegram.httpx.AsyncClient", lambda **_: client)

    # Completes rather than hanging: the conflict ends the loop.
    await asyncio.wait_for(bridge.run(), timeout=5)

    assert bridge.is_running is False
    # Once, not in a retry loop.
    assert client.polls == 1


async def test_starting_a_bridge_twice_is_refused(tmp_path):
    """Two loops on one bridge would 409 each other exactly as two processes do."""
    bridge = _bridge(tmp_path, telegram_bot_token="123:abc", telegram_allowed_users=[ALLOWED_USER])
    bridge._running = True

    with pytest.raises(RuntimeError, match="already polling"):
        await bridge.run()


def test_status_reports_observed_liveness_not_only_a_flag(tmp_path):
    """A stalled loop reports ``running: true`` forever.

    Without a real timestamp there is no way to tell that apart from a healthy
    bridge, which is how "it says it is running but nothing arrives" became
    guesswork.
    """
    bridge = _bridge(tmp_path, telegram_bot_token="123:abc", telegram_allowed_users=[ALLOWED_USER])

    assert bridge.status()["last_poll_seconds_ago"] is None

    bridge._last_poll_at = 100.0
    observed = bridge.status()["last_poll_seconds_ago"]
    assert isinstance(observed, float)


def test_remembered_senders_are_bounded(tmp_path):
    """An unauthorised sender must not grow this by messaging repeatedly."""
    from quainex.integrations.telegram import _MAX_SEEN_SENDERS

    bridge = _bridge(tmp_path)
    for user_id in range(_MAX_SEEN_SENDERS + 10):
        bridge._remember_sender({"update_id": user_id, "message": {"from": {"id": 1000 + user_id}}})

    assert len(bridge._seen_senders) == _MAX_SEEN_SENDERS


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


class PhotoRecorder(SendRecorder):
    """Also records photo uploads, so a test can prove what actually left.

    Attributes:
        photos: One entry per uploaded image: ``(filename, bytes)``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.photos: list[tuple[str, int]] = []

    async def post(self, url: str, **kwargs: object) -> object:
        if url.endswith("/sendPhoto"):
            files = kwargs.get("files")
            if isinstance(files, dict) and "photo" in files:
                name, payload, *_ = files["photo"]
                self.photos.append((str(name), len(payload)))
            return _OkResponse()
        return await super().post(url, **kwargs)


def _screenshot_result(path: Path) -> CommandResult:
    """A successful screenshot result carrying a saved path."""
    return CommandResult(
        status=CommandStatus.SUCCEEDED,
        intent="screenshot",
        message=f"Saved a screenshot to {path}.",
        executed=True,
        data={"path": str(path)},
    )


def _screenshot_intent() -> Intent:
    """An intent asking for a screenshot."""
    return Intent(
        intent=IntentType.SCREENSHOT,
        target=None,
        confidence=1.0,
        reasoning="test",
        requires_confirmation=False,
    )


async def test_the_image_is_not_sent_unless_uploading_is_enabled(tmp_path):
    """Default behaviour discloses nothing: the reply is a path, not a picture."""
    image = tmp_path / "shot.png"
    image.write_bytes(b"pretend png")
    bridge = _bridge(tmp_path, telegram_bot_token="123:abc", telegram_allowed_users=[ALLOWED_USER])
    recorder = PhotoRecorder()

    await bridge._maybe_send_image(
        recorder,  # type: ignore[arg-type]
        7,
        _screenshot_intent(),
        _screenshot_result(image),
    )

    assert recorder.photos == []


async def test_the_image_is_uploaded_when_enabled(tmp_path):
    """The owner asked for it, so it works — and the picture is what is sent."""
    image = tmp_path / "shot.png"
    image.write_bytes(b"pretend png bytes")
    bridge = _bridge(
        tmp_path,
        telegram_bot_token="123:abc",
        telegram_allowed_users=[ALLOWED_USER],
        telegram_send_screenshots=True,
    )
    recorder = PhotoRecorder()

    await bridge._maybe_send_image(
        recorder,  # type: ignore[arg-type]
        7,
        _screenshot_intent(),
        _screenshot_result(image),
    )

    assert recorder.photos == [("shot.png", len(b"pretend png bytes"))]


async def test_an_oversized_screenshot_says_so_rather_than_failing_silently(tmp_path):
    """Telegram rejects photos over 10 MB, and a silent rejection looks like a bug."""
    from quainex.integrations.telegram import _MAX_PHOTO_BYTES

    image = tmp_path / "huge.png"
    image.write_bytes(b"x" * (_MAX_PHOTO_BYTES + 1))
    bridge = _bridge(
        tmp_path,
        telegram_bot_token="123:abc",
        telegram_allowed_users=[ALLOWED_USER],
        telegram_send_screenshots=True,
    )
    recorder = PhotoRecorder()

    await bridge._maybe_send_image(
        recorder,  # type: ignore[arg-type]
        7,
        _screenshot_intent(),
        _screenshot_result(image),
    )

    assert recorder.photos == []
    assert "limit" in recorder.sent[0]


async def test_only_screenshots_are_uploaded(tmp_path):
    """The switch is for screenshots, not for any command that happens to have a path."""
    image = tmp_path / "shot.png"
    image.write_bytes(b"pretend png")
    bridge = _bridge(
        tmp_path,
        telegram_bot_token="123:abc",
        telegram_allowed_users=[ALLOWED_USER],
        telegram_send_screenshots=True,
    )
    recorder = PhotoRecorder()
    other = Intent(
        intent=IntentType.SEARCH_FILES,
        target="invoice",
        confidence=1.0,
        reasoning="test",
        requires_confirmation=False,
    )

    await bridge._maybe_send_image(
        recorder,  # type: ignore[arg-type]
        7,
        other,
        _screenshot_result(image),
    )

    assert recorder.photos == []


def test_the_block_is_about_the_reply_not_the_action():
    """A screenshot discloses nothing over Telegram, so it is not blocked.

    It writes a PNG to disk and replies with a file path. Nothing from the screen
    travels. Blocking it conflated "touches the screen" with "reveals the screen" —
    and the refusal asserted that its output would leave the machine, which was
    false. A security message that misstates its own reason is worse than no
    message: it teaches the user that the refusals are noise.
    """
    assert IntentType.SCREENSHOT not in TELEGRAM_BLOCKED_INTENTS
    # The three that remain each put machine contents into the reply itself.
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
async def test_help_is_answered_without_calling_the_model(tmp_path, command):
    reply = await _bridge(tmp_path)._builtin(command)
    assert "Quainex" in reply
    assert "voice note" in reply


async def test_status_is_a_rich_local_snapshot(tmp_path):
    """Composed from local sources — no model, and it says so."""
    reply = await _bridge(tmp_path)._builtin("/status")

    # Built from the fake controller's metrics and window list.
    assert "CPU" in reply
    assert "TestNet" in reply  # the fake's Wi-Fi
    assert "Chrome" in reply  # the fake's running apps
    assert "0 tokens" in reply
    # Still names what is kept off Telegram.
    assert "clipboard" in reply


async def test_unknown_slash_commands_are_reported(tmp_path):
    assert "Unknown command" in await _bridge(tmp_path)._builtin("/launch_missiles")


async def test_help_is_a_categorised_catalogue(tmp_path):
    """The full menu, grouped by area rather than an alphabetical dump."""
    reply = await _bridge(tmp_path)._builtin("/help")

    # Every major area a person reaches for is represented.
    for phrase in ("Browser", "browse", "panic", "screenshot", "volume", "brightness"):
        assert phrase in reply, f"help should mention {phrase!r}"


async def test_builtin_replies_are_valid_telegram_html(tmp_path):
    """The bridge opts its own messages into HTML, so they must be well-formed.

    A malformed tag or a bare ``&`` is exactly what made Markdown replies vanish
    with a 400. These messages are static and under the bridge's control, so the
    guarantee is simply that they parse.
    """
    bridge = _bridge(tmp_path)
    for command in ("/start", "/help", "/status"):
        html = await bridge._builtin(command)
        # Tags are balanced and drawn only from Telegram's allowed set.
        stack: list[str] = []
        for match in re.finditer(r"<(/?)([a-zA-Z]+)[^>]*>", html):
            closing, tag = match.group(1), match.group(2).lower()
            assert tag in {"b", "i", "u", "s", "code", "pre", "a", "blockquote"}
            if closing:
                assert stack and stack.pop() == tag, f"{command}: mismatched </{tag}>"
            else:
                stack.append(tag)
        assert not stack, f"{command}: unclosed {stack}"
        # No bare ampersand — every & must open a valid entity.
        assert not re.search(r"&(?!amp;|lt;|gt;|quot;|#\d+;)", html), f"{command}: bare &"


def test_escaping_and_the_plain_text_fallback_round_trip():
    """Dynamic values are escaped going in, and recoverable coming back out.

    ``_plain`` is the safety net: if Telegram ever rejects the HTML, the bridge
    resends the same message stripped to text — a styling failure must never cost
    the message itself.
    """
    assert _esc("Rock & <Roll>") == "Rock &amp; &lt;Roll&gt;"
    assert _plain("<b>Up</b> 2h &amp; ticking") == "Up 2h & ticking"


# -- receiving files: save what is sent, where it is told ------------------


class _FileResponse:
    """A successful Telegram response carrying either JSON or file bytes."""

    status_code = 200

    def __init__(self, *, json_body: dict[str, object] | None = None, content: bytes = b"") -> None:
        self._json = json_body or {"ok": True, "result": {}}
        self.content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._json


class IncomingFileClient(SendRecorder):
    """Serves getFile + the file download, and records outbound messages."""

    def __init__(self, payload: bytes = b"the bytes") -> None:
        super().__init__()
        self._payload = payload

    async def get(self, url: str, **_: object) -> _FileResponse:
        if "/getFile" in url:
            return _FileResponse(json_body={"ok": True, "result": {"file_path": "docs/x.bin"}})
        return _FileResponse(content=self._payload)


def _document_update(caption: str | None, *, size: int = 12, name: str = "note.txt") -> object:
    return _parse_update(
        {
            "update_id": 1,
            "message": {
                "chat": {"id": 7},
                "from": {"id": ALLOWED_USER},
                "caption": caption,
                "document": {"file_id": "F1", "file_name": name, "file_size": size},
            },
        }
    )


@pytest.mark.parametrize(
    ("caption", "location", "rename"),
    [
        (None, None, None),
        ("save to downloads", "downloads", None),
        ("documents/reports", "documents/reports", None),
        ("please save this in documents as budget.xlsx", "documents", "budget.xlsx"),
        ("save as report.pdf", None, "report.pdf"),
        ("keep it under desktop/work", "desktop/work", None),
    ],
)
def test_a_save_caption_is_read_into_a_location_and_a_name(caption, location, rename):
    from quainex.integrations.telegram import _parse_save_caption

    assert _parse_save_caption(caption) == (location, rename)


def test_an_attachment_is_recognised_whatever_its_kind():
    from quainex.integrations.telegram import _attachment

    doc = _attachment({"document": {"file_id": "D", "file_name": "a.pdf", "file_size": 3}})
    assert doc == {"file_id": "D", "file_name": "a.pdf", "file_size": 3}

    photo = _attachment({"photo": [{"file_id": "small"}, {"file_id": "big", "file_size": 9}]})
    assert photo["file_id"] == "big"  # the largest size, not the first
    assert photo["file_name"].endswith(".jpg")

    assert _attachment({"text": "hello"}) == {}


async def test_a_sent_file_is_saved_where_the_caption_says(tmp_path):
    """The whole feature: attach a file, name a place, it lands there."""
    desktop = FakeDesktopController()
    bridge = _bridge(
        tmp_path,
        desktop=desktop,
        telegram_bot_token="123:abc",
        telegram_allowed_users=[ALLOWED_USER],
    )
    client = IncomingFileClient(payload=b"real bytes")

    await bridge._dispatch(
        client,  # type: ignore[arg-type]
        _document_update("save to documents as report.pdf"),
    )

    # The controller was asked to save the downloaded bytes at the named place.
    saves = [c for c in desktop.calls if c[0] == "save_incoming_file"]
    assert saves == [("save_incoming_file", ("report.pdf", "documents", len(b"real bytes")))]
    # And the chat is told where it went.
    assert any("Saved" in message for message in client.sent)


async def test_receiving_files_can_be_turned_off(tmp_path):
    desktop = FakeDesktopController()
    bridge = _bridge(
        tmp_path,
        desktop=desktop,
        telegram_bot_token="123:abc",
        telegram_allowed_users=[ALLOWED_USER],
        telegram_receive_files=False,
    )
    client = IncomingFileClient()

    await bridge._dispatch(client, _document_update("save to downloads"))  # type: ignore[arg-type]

    assert not any(c[0] == "save_incoming_file" for c in desktop.calls)
    assert any("turned off" in message for message in client.sent)


async def test_a_file_too_big_to_download_is_reported_not_attempted(tmp_path):
    """Telegram only lets a bot download up to 20 MB; a bigger one is explained."""
    from quainex.integrations.telegram import _MAX_INCOMING_BYTES

    desktop = FakeDesktopController()
    bridge = _bridge(
        tmp_path,
        desktop=desktop,
        telegram_bot_token="123:abc",
        telegram_allowed_users=[ALLOWED_USER],
    )
    client = IncomingFileClient()

    await bridge._dispatch(  # type: ignore[arg-type]
        client,
        _document_update("save to downloads", size=_MAX_INCOMING_BYTES + 1),
    )

    assert not any(c[0] == "save_incoming_file" for c in desktop.calls)
    assert any("MB" in message for message in client.sent)


# -- quick-action menu and the startup heartbeat ---------------------------


class MenuClient(SendRecorder):
    """Records outbound posts, including the inline keyboard of a menu."""

    def __init__(self) -> None:
        super().__init__()
        self.keyboards: list[object] = []

    async def post(self, url: str, **kwargs: object) -> object:
        json_body = kwargs.get("json")
        if isinstance(json_body, dict) and "reply_markup" in json_body:
            self.keyboards.append(json_body["reply_markup"])
        return await super().post(url, **kwargs)


def _button_update(data: str) -> object:
    return _parse_update(
        {
            "update_id": 1,
            "callback_query": {
                "id": "cb1",
                "from": {"id": ALLOWED_USER},
                "data": data,
                "message": {"chat": {"id": 7}},
            },
        }
    )


async def test_menu_sends_tappable_quick_actions(tmp_path):
    bridge = _bridge(tmp_path, telegram_bot_token="123:abc", telegram_allowed_users=[ALLOWED_USER])
    client = MenuClient()

    await bridge._send_menu(client, 7)  # type: ignore[arg-type]

    assert client.keyboards, "the menu must carry an inline keyboard"
    # Every button carries a "do:" action.
    buttons = [b for row in client.keyboards[0]["inline_keyboard"] for b in row]  # type: ignore[index]
    assert all(b["callback_data"].startswith("do:") for b in buttons)
    assert any(b["callback_data"] == "do:screenshot" for b in buttons)


async def test_a_menu_button_runs_the_action(tmp_path):
    """Tapping ⏯ Pause runs the media intent through the ordinary executor."""
    desktop = FakeDesktopController()
    bridge = _bridge(
        tmp_path,
        desktop=desktop,
        telegram_bot_token="123:abc",
        telegram_allowed_users=[ALLOWED_USER],
    )
    client = MenuClient()

    await bridge._dispatch(client, _button_update("do:pause"))  # type: ignore[arg-type]

    assert any(c == ("media_control", "pause") for c in desktop.calls)


async def test_a_status_menu_button_reports_without_an_intent(tmp_path):
    bridge = _bridge(tmp_path, telegram_bot_token="123:abc", telegram_allowed_users=[ALLOWED_USER])
    client = MenuClient()

    await bridge._dispatch(client, _button_update("do:status"))  # type: ignore[arg-type]

    assert any("CPU" in message for message in client.sent)


async def test_the_startup_ping_greets_the_allowed_user(tmp_path):
    bridge = _bridge(tmp_path, telegram_bot_token="123:abc", telegram_allowed_users=[ALLOWED_USER])
    client = SendRecorder()

    await bridge._announce_online(client)  # type: ignore[arg-type]

    assert any("online" in message.lower() for message in client.sent)


async def test_the_startup_ping_can_be_silenced(tmp_path):
    bridge = _bridge(
        tmp_path,
        telegram_bot_token="123:abc",
        telegram_allowed_users=[ALLOWED_USER],
        telegram_startup_ping=False,
    )
    client = SendRecorder()

    await bridge._announce_online(client)  # type: ignore[arg-type]

    assert client.sent == []


# -- the file browser: navigate with buttons, tap to receive --------------


class BrowseClient(MenuClient):
    """Records keyboards *and* uploaded documents, for the file browser."""

    def __init__(self) -> None:
        super().__init__()
        self.documents: list[tuple[str, int]] = []

    async def post(self, url: str, **kwargs: object) -> object:
        if url.endswith("/sendDocument"):
            files = kwargs.get("files")
            if isinstance(files, dict) and "document" in files:
                name, payload, *_ = files["document"]
                self.documents.append((str(name), len(payload)))
            return _OkResponse()
        return await super().post(url, **kwargs)


async def test_files_opens_the_root_browser(tmp_path):
    desktop = FakeDesktopController()
    bridge = _bridge(
        tmp_path,
        desktop=desktop,
        telegram_bot_token="123:abc",
        telegram_allowed_users=[ALLOWED_USER],
    )
    client = BrowseClient()

    await bridge._send_files_root(client, 7)  # type: ignore[arg-type]

    assert any(c[0] == "browse_roots" for c in desktop.calls)
    buttons = [b for row in client.keyboards[0]["inline_keyboard"] for b in row]  # type: ignore[index]
    assert buttons and all(b["callback_data"].startswith("nav:") for b in buttons)


async def test_tapping_a_folder_lists_its_contents(tmp_path):
    desktop = FakeDesktopController()
    bridge = _bridge(
        tmp_path,
        desktop=desktop,
        telegram_bot_token="123:abc",
        telegram_allowed_users=[ALLOWED_USER],
    )
    client = BrowseClient()

    await bridge._send_files_root(client, 7)  # type: ignore[arg-type]
    root_button = client.keyboards[0]["inline_keyboard"][0][0]  # type: ignore[index]
    await bridge._dispatch(client, _button_update(root_button["callback_data"]))  # type: ignore[arg-type]

    # The fake lists a sub-folder and a file; both appear as buttons.
    labels = [b["text"] for row in client.keyboards[-1]["inline_keyboard"] for b in row]  # type: ignore[index]
    assert any("sub" in label for label in labels)
    assert any("a.txt" in label for label in labels)


async def test_tapping_a_file_uploads_it(tmp_path):
    bridge = _bridge(
        tmp_path,
        telegram_bot_token="123:abc",
        telegram_allowed_users=[ALLOWED_USER],
    )
    client = BrowseClient()
    real = tmp_path / "grab-me.bin"
    real.write_bytes(b"payload")
    token = bridge._browse_token(str(real))

    await bridge._handle_browse_button(client, 7, "get", token)  # type: ignore[arg-type]

    assert client.documents == [("grab-me.bin", len(b"payload"))]


async def test_send_this_folder_button_zips_and_uploads(tmp_path):
    desktop = FakeDesktopController()
    bridge = _bridge(
        tmp_path,
        desktop=desktop,
        telegram_bot_token="123:abc",
        telegram_allowed_users=[ALLOWED_USER],
    )
    client = BrowseClient()
    token = bridge._browse_token("work")

    await bridge._handle_browse_button(client, 7, "zipdir", token)  # type: ignore[arg-type]

    assert any(c == ("zip_folder", "work") for c in desktop.calls)
    assert client.documents and client.documents[0][0].endswith(".zip")


async def test_a_receive_with_files_disabled_is_declined(tmp_path):
    bridge = _bridge(
        tmp_path,
        telegram_bot_token="123:abc",
        telegram_allowed_users=[ALLOWED_USER],
        telegram_send_files=False,
    )
    client = BrowseClient()
    token = bridge._browse_token(str(tmp_path / "x"))

    await bridge._handle_browse_button(client, 7, "get", token)  # type: ignore[arg-type]

    assert client.documents == []
    assert any("turned off" in message for message in client.sent)


async def test_an_expired_browse_token_says_so(tmp_path):
    bridge = _bridge(
        tmp_path,
        telegram_bot_token="123:abc",
        telegram_allowed_users=[ALLOWED_USER],
    )
    client = BrowseClient()

    await bridge._handle_browse_button(client, 7, "nav", "p999999")  # type: ignore[arg-type]

    assert any("expired" in message for message in client.sent)
