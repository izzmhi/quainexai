"""Telegram bridge — control Quainex from a phone.

Purpose:
    Give Quainex a phone interface that works from anywhere, without exposing
    this machine to the internet.

Why Telegram rather than the REST API plus a web dashboard:
    The REST API is the right *local* interface, and it stays. But reaching it
    from a phone outside the house needs three things this project does not have:
    a TLS certificate, a port forwarded through the router, and a UI to talk to
    it. Each is a job, and the middle one means accepting inbound connections
    from the internet on a personal machine.

    Telegram removes all three:

    * **Outbound only.** The bridge *polls* Telegram; nothing connects inward.
      The router is untouched. This is the important one.
    * **Transport is Telegram's problem.** No certificate to obtain or renew.
    * **The UI already exists**, including inline buttons — which map onto
      confirmation better than anything I would have built. The button carries
      the signed token, so tapping "Yes" is a real approval rather than a client
      asserting one.

**The tradeoff, stated plainly:** messages pass through Telegram's servers, and
the Bot API is not end-to-end encrypted. For a system that calls itself
privacy-conscious that matters, so:

    * commands whose *output* is sensitive (clipboard contents, screen contents)
      are refused over Telegram by default — see ``TELEGRAM_BLOCKED_INTENTS``;
    * only allowlisted user IDs are obeyed, and everything else is ignored
      silently rather than answered.

    Anything you would not want a third party to hold should stay on the local
    API. The bridge is deliberately not a full replacement for it.

Architecture:
    Telegram  --long poll-->  TelegramBridge
                                 |-- /start, /help, /status   local answers
                                 |-- free text  -> Brain -> CommandExecutor
                                 |     requires confirmation? -> inline buttons
                                 |-- button tap  -> confirmation token -> execute
                                 +-- voice note  -> download -> Whisper -> as above

Dependencies:
    httpx (already present), quainex.core.{brain,commands,voice}

Future improvements:
    * Webhook mode, for lower latency than polling.
    * Per-chat conversation sessions, so two people get separate memory.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import BaseModel

from quainex.core.brain import IntentType
from quainex.core.commands import CommandStatus
from quainex.core.commands.base import CommandResult
from quainex.core.exceptions import ProviderError, QuainexError
from quainex.core.logging import get_logger

if TYPE_CHECKING:
    from quainex.config.settings import Settings
    from quainex.core.brain import Brain, Intent
    from quainex.core.commands import CommandExecutor
    from quainex.core.memory import MemoryManager
    from quainex.core.voice import VoiceSession

_log = get_logger(__name__)

_API_ROOT = "https://api.telegram.org"

#: Intents whose *reply* would carry private data into a third-party chat. These
#: stay on the local API. The action is not dangerous — the disclosure is.
#:
#: The test is narrow and it is about the reply, not the action: does the text sent
#: back to Telegram contain something that was on this machine?
#:
#: ``SCREENSHOT`` was on this list and should not have been. It saves a PNG locally
#: and replies with a *file path* — the image never leaves the machine, so there is
#: nothing to disclose. Worse, the refusal it produced said "its output would leave
#: your machine", which was simply untrue, and a security message that misstates
#: the reason teaches the user to distrust the ones that are right.
TELEGRAM_BLOCKED_INTENTS: frozenset[IntentType] = frozenset(
    {
        IntentType.CLIPBOARD,  # the reply contains the clipboard; those hold passwords
        IntentType.LOOK_AT_SCREEN,  # the reply describes everything on screen
        IntentType.READ_DOCUMENT,  # the reply contains the document's contents
    }
)

#: Telegram rejects messages longer than 4096 characters.
_MAX_MESSAGE_CHARS = 3900

#: How long to hold a long-poll open. Telegram returns early when something
#: arrives, so a long timeout costs nothing and avoids a busy loop.
_POLL_TIMEOUT_SECONDS = 25

#: Ceiling on remembered senders. Bounded so that an unauthorised sender cannot
#: grow this without limit simply by messaging repeatedly.
_MAX_SEEN_SENDERS = 20

#: Telegram's response when a second instance polls the same bot.
_HTTP_CONFLICT = 409

#: Telegram rejects photo uploads above 10 MB.
_MAX_PHOTO_BYTES = 10 * 1024 * 1024


class TelegramUpdate(BaseModel):
    """One update from Telegram, reduced to what the bridge uses.

    Attributes:
        update_id: Monotonic id, used to acknowledge and avoid re-delivery.
        chat_id: Who sent it.
        user_id: The sender's Telegram user id, checked against the allowlist.
        text: Message text, when it was a text message.
        voice_file_id: File id of a voice note, when it was one.
        callback_data: Data from a tapped inline button.
        callback_id: Id needed to acknowledge a button tap.
    """

    update_id: int
    chat_id: int | None = None
    user_id: int | None = None
    text: str | None = None
    voice_file_id: str | None = None
    callback_data: str | None = None
    callback_id: str | None = None


class TelegramBridge:
    """Polls Telegram and drives Quainex on behalf of allowlisted users."""

    def __init__(
        self,
        settings: Settings,
        *,
        brain: Brain,
        commands: CommandExecutor,
        voice: VoiceSession | None = None,
        memory: MemoryManager | None = None,
    ) -> None:
        """Construct the bridge.

        Args:
            settings: Configuration supplying the bot token and allowlist.
            brain: Classifier for incoming messages.
            commands: Executor carrying out intents.
            voice: Optional voice session, for transcribing voice notes.
            memory: Optional memory, for conversation continuity.
        """
        self._settings = settings
        self._brain = brain
        self._commands = commands
        self._voice = voice
        self._memory = memory
        self._offset = 0
        self._running = False
        #: Confirmation tokens keyed by a short id, because Telegram limits
        #: callback_data to 64 bytes and a signed token is far longer.
        self._pending: dict[str, tuple[Intent, str]] = {}
        #: When the last poll returned. Observed rather than declared — see
        #: ``status``, where a boolean flag alone turned out to be a liability.
        self._last_poll_at: float | None = None
        #: Everyone who has messaged the bot this session, authorised or not.
        #:
        #: Recorded here so that setup can offer candidate user ids *without*
        #: calling ``getUpdates`` — see ``diagnose`` for why that call is unsafe
        #: while the bridge is polling. Bounded, because an unauthorised sender
        #: must not be able to grow this without limit by messaging repeatedly.
        self._seen_senders: dict[int, dict[str, object]] = {}

    @property
    def is_configured(self) -> bool:
        """Whether a bot token and at least one allowed user are set."""
        return bool(self._settings.telegram_bot_token and self._settings.telegram_allowed_users)

    @property
    def is_running(self) -> bool:
        """Whether the polling loop is active."""
        return self._running

    def status(self) -> dict[str, object]:
        """Report bridge state for the API and for diagnostics.

        ``running`` is a flag somebody set; ``last_poll_seconds_ago`` is something
        that actually happened. The distinction is not academic — a poll loop can
        stall or die while the flag still says ``True``, and chasing "it says it is
        running but nothing arrives" without a real timestamp is guesswork. A value
        under about 30 seconds means the loop is genuinely alive.

        Returns:
            Configuration and run state.
        """
        return {
            "configured": self.is_configured,
            "running": self._running,
            "last_poll_seconds_ago": (
                None
                if self._last_poll_at is None
                else round(time.monotonic() - self._last_poll_at, 1)
            ),
            "allowed_users": len(self._settings.telegram_allowed_users),
            "blocked_intents": sorted(i.value for i in TELEGRAM_BLOCKED_INTENTS),
        }

    async def run(self) -> None:
        """Poll Telegram until stopped.

        Raises:
            RuntimeError: The bridge is not configured, or already polling.
        """
        if not self.is_configured:
            raise RuntimeError(
                "Telegram is not configured. Set QUAINEX_TELEGRAM_BOT_TOKEN and "
                "QUAINEX_TELEGRAM_ALLOWED_USERS."
            )
        if self._running:
            # Two loops on one bridge would 409 each other exactly as two processes
            # do. Refusing here makes a double-start a caught mistake rather than a
            # mystery.
            raise RuntimeError("This bridge is already polling.")

        self._running = True
        _log.info(
            "telegram_bridge_started",
            allowed_users=len(self._settings.telegram_allowed_users),
        )

        async with httpx.AsyncClient(timeout=_POLL_TIMEOUT_SECONDS + 10) as client:
            while self._running:
                try:
                    for update in await self._poll(client):
                        await self._dispatch(client, update)
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code == _HTTP_CONFLICT:
                        # 409 means another instance of this bot is polling, and
                        # retrying cannot fix that — the two simply terminate each
                        # other's long poll forever. Every poll fails, so messages
                        # arrive erratically or not at all, while the bridge still
                        # reports itself as running.
                        #
                        # That is precisely how a leftover process turned into hours
                        # of debugging. Retrying a conflict is not resilience; it is
                        # a busy loop that hides a configuration error. So this one
                        # status stops the loop and says what to do about it.
                        _log.error(
                            "telegram_conflict",
                            detail=(
                                "Another Quainex (or another program using this bot "
                                "token) is already polling Telegram. Telegram allows "
                                "only one. Stopping this bridge. Check for a leftover "
                                "process: Get-Process python*, or "
                                "Stop-ScheduledTask -TaskName 'Quainex Server'."
                            ),
                        )
                        self._running = False
                        break
                    _log.warning("telegram_poll_failed", error=str(exc))
                    await asyncio.sleep(5)
                except httpx.HTTPError as exc:
                    # A network blip must not end the bridge; back off and retry.
                    _log.warning("telegram_poll_failed", error=str(exc))
                    await asyncio.sleep(5)
                except Exception:
                    _log.exception("telegram_dispatch_failed")
                    await asyncio.sleep(1)

        _log.info("telegram_bridge_stopped")

    def stop(self) -> None:
        """Ask the polling loop to finish."""
        self._running = False

    async def diagnose(self) -> dict[str, object]:
        """Check the token against Telegram and report who has messaged the bot.

        Two questions this answers that ``status()`` cannot, because both need a
        network call: *is this token real*, and *what is my user id*.

        The second exists to remove a genuine dead end. Setup otherwise requires
        finding your numeric id via a third-party bot and typing it in correctly —
        friction that is enough to make people conclude the feature is broken.
        Anyone who has messaged the bot appears here as a **candidate**.

        Candidates are reported, never trusted. Auto-allowing whoever messaged
        first would mean a stranger who found the bot before you could grant
        themselves control of your machine. The list is shown so a human picks
        from it; nothing is authorised until they do.

        **Does not call ``getUpdates`` while the bridge is polling.** Telegram
        permits exactly one ``getUpdates`` per bot and terminates the loser with
        *409 Conflict* — so a "check my bot" button that polls would kill the very
        thing it is reporting on. That is not hypothetical: it is what produced a
        run of ``telegram_poll_failed`` entries and made a healthy bridge look
        broken every time the button was pressed.

        So while polling, candidates come from senders the bridge has already seen
        (``_seen_senders``), which is strictly better information anyway — it is
        drawn from the whole session rather than one pending update. Only when the
        bridge is *stopped* does this call ``getUpdates`` directly, with
        ``offset=-1`` so a pending message is still delivered later.

        ``getMe`` is always safe: it does not touch the update queue.

        Returns:
            ``ok``, the bot's ``username``, ``candidates``, and ``error`` on
            failure. Never raises: this is a diagnostic, and an exception here
            would tell the user less than a message does.
        """
        if not self._settings.telegram_bot_token:
            return {"ok": False, "error": "No bot token is configured.", "candidates": []}

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                me = await client.get(f"{self._url()}/getMe")
                if me.status_code == 401:
                    return {
                        "ok": False,
                        "error": (
                            "Telegram rejected this bot token. Check it against the "
                            "one @BotFather gave you, or use /revoke to issue a new one."
                        ),
                        "candidates": [],
                    }
                me.raise_for_status()
                username = str((me.json().get("result") or {}).get("username") or "")

                if self._running:
                    # Never poll here: it would terminate the bridge's own poll.
                    candidates = list(self._seen_senders.values())
                else:
                    # `offset=-1` returns only the most recent update and leaves
                    # the queue intact, so nothing is swallowed.
                    updates = await client.get(
                        f"{self._url()}/getUpdates", params={"offset": -1, "timeout": 0}
                    )
                    candidates = _senders(updates.json().get("result", []))
        except httpx.HTTPError as exc:
            return {"ok": False, "error": f"Could not reach Telegram: {exc}", "candidates": []}

        _log.info("telegram_diagnosed", username=username, candidates=len(candidates))
        return {"ok": True, "username": username, "candidates": candidates}

    # -- polling ----------------------------------------------------------

    async def _poll(self, client: httpx.AsyncClient) -> list[TelegramUpdate]:
        """Fetch pending updates.

        Args:
            client: HTTP client.

        Returns:
            Updates, already acknowledged by advancing the offset.
        """
        response = await client.get(
            f"{self._url()}/getUpdates",
            params={"offset": self._offset, "timeout": _POLL_TIMEOUT_SECONDS},
        )
        response.raise_for_status()
        payload = response.json()
        self._last_poll_at = time.monotonic()

        updates: list[TelegramUpdate] = []
        for raw in payload.get("result", []):
            self._remember_sender(raw)
            update = _parse_update(raw)
            self._offset = max(self._offset, update.update_id + 1)
            updates.append(update)
        return updates

    def _remember_sender(self, raw: dict[str, Any]) -> None:
        """Note who sent an update, for the setup screen.

        Recorded before authorisation is checked, deliberately: the whole point is
        to show an unconfigured user their own id, and at that moment they are not
        on the allowlist yet. Nothing is granted by appearing here.

        Args:
            raw: The update as Telegram sent it.
        """
        for candidate in _senders([raw]):
            user_id = candidate["user_id"]
            if isinstance(user_id, int) and len(self._seen_senders) < _MAX_SEEN_SENDERS:
                self._seen_senders.setdefault(user_id, candidate)

    async def _dispatch(self, client: httpx.AsyncClient, update: TelegramUpdate) -> None:
        """Route one update.

        Args:
            client: HTTP client.
            update: The update to handle.
        """
        if update.user_id is None or update.user_id not in self._settings.telegram_allowed_users:
            # Silent: replying would confirm the bot exists and is listening,
            # which is information an unknown sender has not earned.
            _log.warning("telegram_unauthorised", user_id=update.user_id)
            return

        if update.chat_id is None:
            return

        # Every handler is wrapped, because the alternative is silence.
        #
        # Before this, a failure inside a handler propagated to the polling loop,
        # which logged it and carried on — correct for the bridge's uptime, and
        # useless for the person holding the phone. They saw nothing at all and
        # concluded the whole system was dead, when in fact every provider was
        # simply out of quota. A remote interface that can fail invisibly is worse
        # than one that fails loudly: the log is on a machine they are not looking
        # at, which is the entire reason they are using Telegram.
        try:
            if update.callback_data:
                await self._handle_button(client, update)
            elif update.voice_file_id:
                await self._handle_voice(client, update)
            elif update.text:
                await self._handle_text(client, update.chat_id, update.text)
        except QuainexError as exc:
            _log.warning("telegram_request_failed", reason=exc.message)
            await self._send(client, update.chat_id, _explain(exc))
        except Exception:
            # Unexpected, so the message stays generic — an internal traceback is
            # not something to put in a chat transcript. The log has the detail.
            _log.exception("telegram_dispatch_failed")
            await self._send(
                client,
                update.chat_id,
                "Something went wrong handling that. The details are in the Quainex log.",
            )

    # -- handlers ----------------------------------------------------------

    async def _handle_text(self, client: httpx.AsyncClient, chat_id: int, text: str) -> None:
        """Handle a text message.

        Args:
            client: HTTP client.
            chat_id: Where to reply.
            text: What was said.
        """
        command = text.strip()
        if command.startswith("/"):
            await self._send(client, chat_id, self._builtin(command))
            return

        history = await self._memory.conversation_context() if self._memory else None
        intent = await self._brain.interpret(command, history=history)

        if intent.intent in TELEGRAM_BLOCKED_INTENTS:
            await self._send(
                client,
                chat_id,
                f"`{intent.intent.value}` is disabled over Telegram because its output "
                "would leave your machine. Use the local API for that one.",
            )
            return

        result = await self._commands.execute(intent)

        if self._memory is not None:
            await self._memory.remember_exchange(command, intent, result)

        if result.status is CommandStatus.REQUIRES_CONFIRMATION and result.confirmation_token:
            key = f"c{len(self._pending)}{intent.intent.value[:8]}"[:60]
            self._pending[key] = (intent, result.confirmation_token)
            await self._send_confirmation(client, chat_id, result.message, key)
            return

        await self._send(client, chat_id, result.message)
        await self._maybe_send_image(client, chat_id, intent, result)

    async def _maybe_send_image(
        self,
        client: httpx.AsyncClient,
        chat_id: int,
        intent: Intent,
        result: CommandResult,
    ) -> None:
        """Upload the captured image when one was produced and uploading is enabled.

        Covers screenshots and webcam photos — both put a picture of, or around, the
        machine into a third-party chat that is not end-to-end encrypted, so both are
        governed by the same ``telegram_send_screenshots`` switch. With it off the
        reply is a file path, which discloses nothing.

        A webcam photo answered over Telegram is worth a word on the tradeoff, since
        it is the whole point of the feature: the image reaches Telegram's servers
        and lives in the chat history until deleted. The owner turned it on.

        Args:
            client: HTTP client.
            chat_id: Where to send it.
            intent: What was asked for.
            result: What happened, carrying the saved path.
        """
        if intent.intent not in {IntentType.SCREENSHOT, IntentType.WEBCAM} or not result.ok:
            return
        if not self._settings.telegram_send_screenshots:
            return

        path_value = (result.data or {}).get("path")
        if not isinstance(path_value, str):
            return

        image = Path(path_value)
        if not image.is_file():
            _log.warning("telegram_image_missing", path=path_value)
            return

        try:
            payload = image.read_bytes()
        except OSError as exc:
            _log.warning("telegram_image_unreadable", error=str(exc))
            return

        if len(payload) > _MAX_PHOTO_BYTES:
            # Telegram rejects oversized photos outright, and a silent rejection
            # would look like the feature simply not working.
            await self._send(
                client,
                chat_id,
                f"The image is {len(payload) // 1024 // 1024} MB, over Telegram's "
                f"{_MAX_PHOTO_BYTES // 1024 // 1024} MB photo limit. It is saved at "
                f"{image.name} on the machine.",
            )
            return

        mime = "image/jpeg" if image.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
        try:
            response = await client.post(
                f"{self._url()}/sendPhoto",
                data={"chat_id": str(chat_id)},
                files={"photo": (image.name, payload, mime)},
                timeout=120,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            _log.warning("telegram_image_upload_failed", error=str(exc))
            await self._send(client, chat_id, "The image was saved but could not be sent.")
            return

        _log.info("telegram_image_sent", intent=intent.intent.value, bytes=len(payload))

    async def _handle_button(self, client: httpx.AsyncClient, update: TelegramUpdate) -> None:
        """Handle a tapped inline button.

        Args:
            client: HTTP client.
            update: The callback update.
        """
        assert update.chat_id is not None  # noqa: S101 - checked by the caller
        data = update.callback_data or ""

        if update.callback_id:
            await client.post(
                f"{self._url()}/answerCallbackQuery",
                json={"callback_query_id": update.callback_id},
            )

        if data.startswith("no:"):
            self._pending.pop(data[3:], None)
            await self._send(client, update.chat_id, "Cancelled.")
            return

        key = data.removeprefix("yes:")
        pending = self._pending.pop(key, None)
        if pending is None:
            # Tokens expire in two minutes; a stale button is normal, not an error.
            await self._send(client, update.chat_id, "That confirmation has expired. Ask again.")
            return

        intent, token = pending
        result = await self._commands.execute(intent, confirmation_token=token)
        await self._send(client, update.chat_id, result.message)

    async def _handle_voice(self, client: httpx.AsyncClient, update: TelegramUpdate) -> None:
        """Transcribe a voice note and act on it.

        Args:
            client: HTTP client.
            update: The voice update.
        """
        assert update.chat_id is not None  # noqa: S101 - checked by the caller
        if self._voice is None or not self._voice.status()["speech_to_text"]:
            await self._send(
                client, update.chat_id, 'Voice is not installed. Run: pip install -e ".[voice]"'
            )
            return

        file_response = await client.get(
            f"{self._url()}/getFile", params={"file_id": update.voice_file_id}
        )
        file_response.raise_for_status()
        remote_path = file_response.json()["result"]["file_path"]

        audio = await client.get(f"{_API_ROOT}/file/bot{self._token()}/{remote_path}", timeout=60)
        audio.raise_for_status()

        with tempfile.TemporaryDirectory(prefix="quainex-tg-") as tmp:
            local = Path(tmp) / "voice.ogg"
            local.write_bytes(audio.content)
            transcript = await self._voice.transcribe(local)

        if not transcript.text.strip():
            await self._send(client, update.chat_id, "I could not make that out.")
            return

        await self._send(client, update.chat_id, f'Heard: "{transcript.text}"')
        # Voice notes are addressed by being sent, so the wake word is redundant.
        await self._handle_text(client, update.chat_id, transcript.text)

    def _builtin(self, command: str) -> str:
        """Answer a slash command locally, without the model.

        Args:
            command: The slash command.

        Returns:
            The reply text.
        """
        verb = command.split()[0].lower()
        if verb in {"/start", "/help"}:
            return (
                "*Quainex*\n\n"
                "Send me a request in plain language — 'open vs code', "
                "'run the tests', 'what's my battery' — or record a voice note.\n\n"
                "Anything risky comes back with Yes/No buttons first.\n\n"
                "/status — what I can currently do"
            )
        if verb == "/status":
            blocked = ", ".join(sorted(i.value for i in TELEGRAM_BLOCKED_INTENTS))
            return (
                f"Connected. {len(self._commands.catalogue)} commands available.\n\n"
                f"Disabled over Telegram (output would leave your machine): {blocked}"
            )
        return f"Unknown command `{verb}`. Try /help."

    # -- transport ---------------------------------------------------------

    def _token(self) -> str:
        """Return the bot token.

        Returns:
            The configured token.
        """
        token = self._settings.telegram_bot_token
        return token.get_secret_value() if token else ""

    def _url(self) -> str:
        """Return the bot API base URL.

        Returns:
            The base URL including the token.
        """
        return f"{_API_ROOT}/bot{self._token()}"

    async def _send(self, client: httpx.AsyncClient, chat_id: int, text: str) -> None:
        """Send a plain message.

        Args:
            client: HTTP client.
            chat_id: Where to send.
            text: What to send, truncated to Telegram's limit.
        """
        await client.post(
            f"{self._url()}/sendMessage",
            json={"chat_id": chat_id, "text": _truncate(text), "parse_mode": "Markdown"},
        )

    async def _send_confirmation(
        self, client: httpx.AsyncClient, chat_id: int, question: str, key: str
    ) -> None:
        """Send a question with Yes/No buttons.

        The button carries a short key, not the token itself: Telegram limits
        callback data to 64 bytes, and a signed token is longer than that. The
        token stays on this machine and is looked up when the button is tapped.

        Args:
            client: HTTP client.
            chat_id: Where to send.
            question: The confirmation prompt.
            key: Lookup key for the held token.
        """
        await client.post(
            f"{self._url()}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": _truncate(question),
                "reply_markup": {
                    "inline_keyboard": [
                        [
                            {"text": "✅ Yes", "callback_data": f"yes:{key}"},
                            {"text": "❌ No", "callback_data": f"no:{key}"},
                        ]
                    ]
                },
            },
        )


def _truncate(text: str) -> str:
    """Fit text within Telegram's message limit.

    Args:
        text: The message.

    Returns:
        The message, trimmed if necessary.
    """
    if len(text) <= _MAX_MESSAGE_CHARS:
        return text
    return text[:_MAX_MESSAGE_CHARS] + "\n\n…(truncated)"


def _explain(error: QuainexError) -> str:
    """Turn an internal failure into something useful on a phone.

    The generic path is to relay the message, which is already written for a
    person. Provider exhaustion gets its own wording because the raw text is three
    nested vendor JSON blobs — accurate, unreadable on a phone, and it buries the
    one thing that matters: this is a quota, not a fault, and it will come back.

    Args:
        error: The failure.

    Returns:
        Text to send to the chat.
    """
    if isinstance(error, ProviderError):
        return (
            "⚠️ No AI provider could answer — every configured one is out of quota "
            "or credit right now.\n\n"
            "This is a limit, not a breakage: free tiers reset, so it will start "
            "working again on its own. Add another key in the dashboard's Settings "
            "panel if you would rather not wait.\n\n"
            "Commands that do not need a model are unaffected."
        )
    return f"⚠️ {error.message}"


def _senders(raw_updates: list[dict[str, Any]]) -> list[dict[str, object]]:
    """Extract who sent each pending update, for the setup screen.

    Args:
        raw_updates: Updates as Telegram sent them.

    Returns:
        One entry per distinct sender: ``user_id``, ``name`` and ``username``.
        The name is for recognition only — the id is what authorises, and a
        display name is chosen by its owner and can be anything at all.
    """
    seen: dict[int, dict[str, object]] = {}
    for raw in raw_updates:
        sender = (
            (raw.get("callback_query") or {}).get("from")
            or (raw.get("message") or {}).get("from")
            or (raw.get("edited_message") or {}).get("from")
            or {}
        )
        user_id = sender.get("id")
        if not isinstance(user_id, int) or user_id in seen:
            continue
        seen[user_id] = {
            "user_id": user_id,
            "name": " ".join(
                part
                for part in (sender.get("first_name"), sender.get("last_name"))
                if isinstance(part, str)
            ).strip(),
            "username": str(sender.get("username") or ""),
        }
    return list(seen.values())


def _parse_update(raw: dict[str, Any]) -> TelegramUpdate:
    """Reduce a raw Telegram update to the fields the bridge uses.

    Args:
        raw: The update as Telegram sent it.

    Returns:
        The parsed update.
    """
    update_id = int(raw.get("update_id", 0))

    if callback := raw.get("callback_query"):
        message = callback.get("message", {})
        return TelegramUpdate(
            update_id=update_id,
            chat_id=message.get("chat", {}).get("id"),
            user_id=callback.get("from", {}).get("id"),
            callback_data=callback.get("data"),
            callback_id=callback.get("id"),
        )

    message = raw.get("message") or raw.get("edited_message") or {}
    voice = message.get("voice") or {}
    return TelegramUpdate(
        update_id=update_id,
        chat_id=message.get("chat", {}).get("id"),
        user_id=message.get("from", {}).get("id"),
        text=message.get("text"),
        voice_file_id=voice.get("file_id"),
    )
