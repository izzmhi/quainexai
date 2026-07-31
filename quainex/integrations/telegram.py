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
import contextlib
import re
import socket
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import BaseModel

from quainex.core.brain import Intent, IntentType
from quainex.core.commands import CommandStatus
from quainex.core.commands.base import CommandResult
from quainex.core.exceptions import ProviderError, QuainexError
from quainex.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from quainex.config.settings import Settings
    from quainex.core.brain import Brain
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

#: Intents whose reply *always* discloses something from the machine, with no safe
#: variant. Clipboard is deliberately not here: writing to it reveals nothing, and
#: reading it is a per-owner choice (see ``_blocked_reason``), so its enforcement is
#: dynamic rather than a flat membership test.
_OUTPUT_REVEALING: frozenset[IntentType] = frozenset(
    {IntentType.LOOK_AT_SCREEN, IntentType.READ_DOCUMENT}
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

#: The Bot API caps ``sendDocument`` at 50 MB.
_MAX_DOCUMENT_BYTES = 50 * 1024 * 1024

#: The Bot API only lets a bot *download* files up to 20 MB via ``getFile``. A file
#: larger than this cannot be fetched at all, so it is reported rather than attempted.
_MAX_INCOMING_BYTES = 20 * 1024 * 1024

#: How often to re-send the "typing…" action. Telegram's lasts about five seconds;
#: four keeps it continuous without hammering the API.
_TYPING_REFRESH_SECONDS = 4.0

#: Browser intents each produce a screenshot of the page.
_BROWSER_IMAGE_INTENTS: frozenset[IntentType] = frozenset(
    {
        IntentType.BROWSE,
        IntentType.BROWSER_SCROLL,
        IntentType.BROWSER_CLICK,
        IntentType.BROWSER_TYPE,
        IntentType.BROWSER_BACK,
    }
)

#: Intents that always send their image, regardless of the send-images switch —
#: the user asked to see this specific thing (a page, a panic photo).
_ALWAYS_SEND_IMAGE: frozenset[IntentType] = frozenset({IntentType.PANIC}) | _BROWSER_IMAGE_INTENTS

#: Every intent whose successful result carries an image to upload.
_IMAGE_INTENTS: frozenset[IntentType] = (
    frozenset({IntentType.SCREENSHOT, IntentType.WEBCAM}) | _ALWAYS_SEND_IMAGE
)

#: Quick-action buttons for /menu that map to a one-shot intent. The two report
#: actions (status, wifi) are handled separately because they render text directly
#: rather than running an intent. ``callback_data`` carries "do:<key>".
_MENU_ACTION_INTENTS: dict[str, tuple[IntentType, str | None]] = {
    "screenshot": (IntentType.SCREENSHOT, None),
    "lock": (IntentType.LOCK_SCREEN, None),
    "pause": (IntentType.MEDIA_CONTROL, "pause"),
    "next": (IntentType.MEDIA_CONTROL, "next"),
}

#: The /menu keyboard layout, as rows of (label, action-key).
_MENU_LAYOUT: tuple[tuple[tuple[str, str], ...], ...] = (
    (("📸 Screenshot", "screenshot"), ("🔒 Lock", "lock")),
    (("⏯ Pause", "pause"), ("⏭ Next", "next")),
    (("📊 Status", "status"), ("📶 Wi-Fi", "wifi")),
)

#: Intents whose successful result carries a file to upload as a document.
_DOCUMENT_INTENTS: frozenset[IntentType] = frozenset(
    {IntentType.SEND_FILE, IntentType.SEND_FOLDER}
)

#: Cap on remembered file-browser paths. Each folder listing mints short tokens for
#: its entries (Telegram limits callback data to 64 bytes, too little for a full
#: path), and this bounds how many are kept so browsing cannot grow memory without
#: limit. Old tokens fall off the front; a tap on a long-stale button just re-opens
#: the browser rather than acting.
_MAX_BROWSE_TOKENS = 2000

#: How many entries a single folder listing shows, so a large directory does not
#: build a keyboard Telegram will reject.
_BROWSE_PAGE = 40


class TelegramUpdate(BaseModel):
    """One update from Telegram, reduced to what the bridge uses.

    Attributes:
        update_id: Monotonic id, used to acknowledge and avoid re-delivery.
        chat_id: Who sent it.
        user_id: The sender's Telegram user id, checked against the allowlist.
        text: Message text, when it was a text message.
        voice_file_id: File id of a voice note, when it was one.
        file_id: File id of an attached document/photo/video/audio to save.
        file_name: The sender's name for that attachment, when known.
        file_size: The attachment's size in bytes, when known.
        caption: The text sent alongside an attachment — the save instruction.
        callback_data: Data from a tapped inline button.
        callback_id: Id needed to acknowledge a button tap.
    """

    update_id: int
    chat_id: int | None = None
    user_id: int | None = None
    text: str | None = None
    voice_file_id: str | None = None
    file_id: str | None = None
    file_name: str | None = None
    file_size: int | None = None
    caption: str | None = None
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
        #: File-browser tokens → absolute paths, for the same 64-byte reason. A
        #: monotonic counter names them; the map is bounded (see ``_MAX_BROWSE_TOKENS``).
        self._browse_paths: dict[str, str] = {}
        self._browse_seq = 0
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
            await self._register_commands(client)
            await self._announce_online(client)
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

    async def _register_commands(self, client: httpx.AsyncClient) -> None:
        """Publish the slash-command menu, so Telegram offers "/" suggestions.

        This is what puts ``/help``, ``/status`` and ``/start`` in the blue menu
        beside the message box — the small thing that makes the bot feel finished
        rather than improvised. Best-effort: the bot works identically without it,
        so a failure here is logged and shrugged off, never fatal.

        Args:
            client: HTTP client.
        """
        try:
            await client.post(
                f"{self._url()}/setMyCommands",
                json={
                    "commands": [
                        {"command": "menu", "description": "Quick-action buttons"},
                        {"command": "files", "description": "Browse folders and grab a file"},
                        {"command": "help", "description": "Everything Quainex can do"},
                        {"command": "status", "description": "A live snapshot of this machine"},
                        {"command": "start", "description": "Welcome and a few examples"},
                    ]
                },
                timeout=10,
            )
        except Exception as exc:
            # Registering the menu must never jeopardise the poll loop; any failure
            # here is cosmetic, so every exception type is caught, not just HTTP ones.
            _log.warning("telegram_setmycommands_failed", error=str(exc))

    async def _announce_online(self, client: httpx.AsyncClient) -> None:
        """Ping the allowed users once, to say the machine is back.

        The point of a phone bridge is that the machine is somewhere you are not, so
        it coming back — after a reboot, a crash, or a restart — is worth exactly one
        line. In a private chat the chat id is the user id, so each allowed user is
        messaged directly. Best-effort, like every other courtesy here.

        Args:
            client: HTTP client.
        """
        if not self._settings.telegram_startup_ping:
            return
        host = _esc(socket.gethostname())
        text = f"🟢 <b>Quainex is online</b>\n{host} · {time.strftime('%H:%M')}"
        for user_id in self._settings.telegram_allowed_users:
            await self._send(client, user_id, text, parse_mode="HTML")

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
            elif update.file_id:
                await self._handle_incoming_file(client, update)
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

    def _blocked_reason(self, intent: Intent) -> str | None:
        """Why an intent must not run over Telegram, or ``None`` if it may.

        The test is about the *reply*: does the text sent back carry something that
        was on the machine? Reading the clipboard, the screen, or a document does;
        copying *to* the clipboard does not. So:

        * a clipboard **write** is always allowed — nothing leaves the machine;
        * a clipboard **read** is allowed only when the owner has opted in, because
          the clipboard often holds a password just copied and Telegram is not
          end-to-end encrypted;
        * the two genuinely output-revealing intents stay off entirely.

        Args:
            intent: The classified intent.

        Returns:
            A message explaining the refusal, or ``None`` to proceed.
        """
        if intent.intent is IntentType.CLIPBOARD:
            action = intent.parameters_as_dict().get("action", "read").lower()
            if action == "write" or self._settings.telegram_allow_clipboard_read:
                return None
            return (
                "Reading the clipboard is kept off Telegram by default — it can hold a "
                "password you just copied, and Telegram is not end-to-end encrypted.\n\n"
                "To copy something TO your PC, say:\n  copy this to my PC: your text\n\n"
                "To allow reading it here, set QUAINEX_TELEGRAM_ALLOW_CLIPBOARD_READ=true."
            )
        if intent.intent in _OUTPUT_REVEALING:
            return (
                f"{intent.intent.value} is disabled over Telegram because its output "
                "would leave your machine. Use the local API for that one."
            )
        return None

    async def _handle_text(self, client: httpx.AsyncClient, chat_id: int, text: str) -> None:
        """Handle a text message.

        Args:
            client: HTTP client.
            chat_id: Where to reply.
            text: What was said.
        """
        command = text.strip()
        if command.startswith("/"):
            verb = command.split()[0].lower()
            if verb == "/menu":
                await self._send_menu(client, chat_id)
                return
            if verb == "/files":
                await self._send_files_root(client, chat_id)
                return
            await self._send(client, chat_id, await self._builtin(command), parse_mode="HTML")
            return

        # A "typing…" indicator from the first moment, kept alive for the whole
        # turn. A local match finishes before it is even visible; a request that has
        # to fall through several rate-limited providers can take ten seconds, and
        # without this the chat looks frozen. Telegram's typing action expires after
        # ~5s, so it is re-sent on a timer rather than issued once.
        async with self._typing(client, chat_id):
            history = await self._memory.conversation_context() if self._memory else None
            intent = await self._brain.interpret(command, history=history)

            reason = self._blocked_reason(intent)
            if reason is not None:
                await self._send(client, chat_id, reason)
                return

            result = await self._commands.execute(intent)

            if self._memory is not None:
                await self._memory.remember_exchange(command, intent, result)

        await self._deliver(client, chat_id, intent, result)

    async def _deliver(
        self,
        client: httpx.AsyncClient,
        chat_id: int,
        intent: Intent,
        result: CommandResult,
    ) -> None:
        """Send the outcome of an intent: a confirmation, or the reply and any media.

        Shared by typed requests and quick-action buttons so both honour the same
        confirmation gate — a button that skipped "are you sure?" would be a quiet
        way to bypass it.

        Args:
            client: HTTP client.
            chat_id: Where to reply.
            intent: The intent that ran.
            result: Its outcome.
        """
        if result.status is CommandStatus.REQUIRES_CONFIRMATION and result.confirmation_token:
            key = f"c{len(self._pending)}{intent.intent.value[:8]}"[:60]
            self._pending[key] = (intent, result.confirmation_token)
            await self._send_confirmation(client, chat_id, result.message, key)
            return

        await self._send(client, chat_id, result.message)
        await self._maybe_send_image(client, chat_id, intent, result)
        await self._maybe_send_document(client, chat_id, intent, result)

    async def _send_menu(self, client: httpx.AsyncClient, chat_id: int) -> None:
        """Send the quick-action button menu.

        The things reached for daily — a screenshot, a lock, pause — one tap away,
        so the common case needs no typing. Typing still works for everything else.

        Args:
            client: HTTP client.
            chat_id: Where to send.
        """
        keyboard = [
            [{"text": label, "callback_data": f"do:{action}"} for label, action in row]
            for row in _MENU_LAYOUT
        ]
        await client.post(
            f"{self._url()}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": "⚡ <b>Quick actions</b>\nTap one — or just type a request.",
                "parse_mode": "HTML",
                "reply_markup": {"inline_keyboard": keyboard},
            },
        )

    async def _handle_menu_action(
        self, client: httpx.AsyncClient, chat_id: int, action: str
    ) -> None:
        """Run a quick-action button.

        Reports (status, Wi-Fi) render text directly; the rest run a one-shot intent
        through the ordinary executor, so their confirmation policy still applies.

        Args:
            client: HTTP client.
            chat_id: Where to reply.
            action: The action key from the button.
        """
        if action == "status":
            await self._send(client, chat_id, await self._status_report(), parse_mode="HTML")
            return
        if action == "wifi":
            state = self._commands.context.desktop.wifi_status()
            await self._send(client, chat_id, f"📶 {_esc(state)}", parse_mode="HTML")
            return

        spec = _MENU_ACTION_INTENTS.get(action)
        if spec is None:
            await self._send(client, chat_id, "That action is no longer available.")
            return
        intent_type, target = spec
        intent = Intent(
            intent=intent_type,
            target=target,
            confidence=1.0,
            reasoning="Quick-action button.",
            requires_confirmation=False,
            utterance=f"/menu {action}",
        )
        result = await self._commands.execute(intent)
        await self._deliver(client, chat_id, intent, result)

    # -- file browser ------------------------------------------------------

    def _browse_token(self, path: str) -> str:
        """Mint a short token for a path, so it fits in Telegram's callback data.

        A full path is far longer than the 64 bytes callback data allows, so each
        listed entry is referenced by a tiny id that maps back to its path here.
        Bounded: the oldest token falls off once the map is full, and a tap on one
        that has aged out just says so rather than acting on a wrong path.

        Args:
            path: The absolute path to remember.

        Returns:
            The token to put in a button.
        """
        self._browse_seq += 1
        token = f"p{self._browse_seq}"
        self._browse_paths[token] = path
        if len(self._browse_paths) > _MAX_BROWSE_TOKENS:
            del self._browse_paths[next(iter(self._browse_paths))]
        return token

    async def _send_files_root(self, client: httpx.AsyncClient, chat_id: int) -> None:
        """Open the file browser at the permitted roots.

        Args:
            client: HTTP client.
            chat_id: Where to send.
        """
        roots = self._commands.context.desktop.browse_roots()
        if not roots:
            await self._send(client, chat_id, "No browsable folders are configured.")
            return
        rows = [
            [
                {
                    "text": f"📁 {_esc(str(root))}",
                    "callback_data": f"nav:{self._browse_token(str(root))}",
                }
            ]
            for root in roots
        ]
        await client.post(
            f"{self._url()}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": "📂 <b>Your folders</b>\nTap to open. Tap a file to receive it.",
                "parse_mode": "HTML",
                "reply_markup": {"inline_keyboard": rows},
            },
        )

    async def _send_directory(self, client: httpx.AsyncClient, chat_id: int, path: str) -> None:
        """List a folder as tappable buttons: sub-folders to open, files to receive.

        Args:
            client: HTTP client.
            chat_id: Where to send.
            path: The folder to list.
        """
        listing = self._commands.context.desktop.list_directory(path, limit=_BROWSE_PAGE)

        rows: list[list[dict[str, str]]] = []
        top: list[dict[str, str]] = []
        if listing.parent:
            top.append(
                {"text": "⬆️ Up", "callback_data": f"nav:{self._browse_token(listing.parent)}"}
            )
        if self._settings.telegram_send_files:
            top.append(
                {
                    "text": "📦 Send this folder",
                    "callback_data": f"zipdir:{self._browse_token(listing.path)}",
                }
            )
        if top:
            rows.append(top)

        for entry in listing.entries:
            if entry.is_dir:
                label = f"📁 {entry.name}"
                data = f"nav:{self._browse_token(entry.path)}"
            else:
                size = "" if entry.size_bytes is None else f"  ·  {_human_size(entry.size_bytes)}"
                label = f"📄 {entry.name}{size}"
                data = f"get:{self._browse_token(entry.path)}"
            rows.append([{"text": label[:60], "callback_data": data}])

        header = f"📂 <b>{_esc(listing.path)}</b>"
        if not listing.entries:
            header += "\n<i>(empty)</i>"
        if listing.truncated:
            header += f"\n<i>Showing the first {_BROWSE_PAGE}.</i>"
        await client.post(
            f"{self._url()}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": header,
                "parse_mode": "HTML",
                "reply_markup": {"inline_keyboard": rows},
            },
        )

    async def _handle_browse_button(
        self, client: httpx.AsyncClient, chat_id: int, kind: str, token: str
    ) -> None:
        """Act on a file-browser button: navigate, receive a file, or zip a folder.

        Args:
            client: HTTP client.
            chat_id: Where to reply.
            kind: ``nav``, ``get`` or ``zipdir``.
            token: The path token from the button.
        """
        path = self._browse_paths.get(token)
        if path is None:
            await self._send(client, chat_id, "That listing has expired — open /files again.")
            return

        if kind == "nav":
            await self._send_directory(client, chat_id, path)
            return

        if not self._settings.telegram_send_files:
            await self._send(
                client,
                chat_id,
                "Sending files over Telegram is turned off (QUAINEX_TELEGRAM_SEND_FILES=false).",
            )
            return

        async with self._typing(client, chat_id):
            if kind == "zipdir":
                archive = self._commands.context.desktop.zip_folder(path)
                await self._upload_document(client, chat_id, archive)
            else:  # get
                await self._upload_document(client, chat_id, Path(path))

    @contextlib.asynccontextmanager
    async def _typing(self, client: httpx.AsyncClient, chat_id: int) -> AsyncIterator[None]:
        """Show a "typing…" indicator in the chat for the duration of a block.

        Telegram's typing action lasts about five seconds, so for anything slower a
        single call would flicker off mid-thought. A background task re-sends it on
        a timer, and is cancelled the instant the block finishes — the indicator
        disappears exactly when the reply arrives, which is the behaviour that reads
        as "it was working, and now it's done".

        Args:
            client: HTTP client.
            chat_id: The chat to show typing in.

        Yields:
            Control, while typing is shown.
        """

        async def keep_alive() -> None:
            while True:
                await self._send_chat_action(client, chat_id, "typing")
                await asyncio.sleep(_TYPING_REFRESH_SECONDS)

        task = asyncio.create_task(keep_alive())
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _send_chat_action(self, client: httpx.AsyncClient, chat_id: int, action: str) -> None:
        """Send a chat action (typing, uploading a photo, …).

        Best-effort: a failed indicator must never affect the actual work, so any
        error is swallowed. The point is feedback, not correctness.

        Args:
            client: HTTP client.
            chat_id: The chat.
            action: A Telegram chat action, e.g. ``"typing"``.
        """
        try:
            await client.post(
                f"{self._url()}/sendChatAction",
                json={"chat_id": chat_id, "action": action},
                timeout=10,
            )
        except httpx.HTTPError:
            pass

    async def _maybe_send_document(
        self,
        client: httpx.AsyncClient,
        chat_id: int,
        intent: Intent,
        result: CommandResult,
    ) -> None:
        """Upload a requested file or zipped folder as a Telegram document.

        Fires for ``SEND_FILE`` and ``SEND_FOLDER``. Unlike an image this keeps the
        bytes exactly — any type, not re-encoded — which is the point of "send me
        the file". The disclosure is explicit (the user named it) and the source is
        confined to the search roots by the controller, so there is nothing to gate
        beyond the on/off switch.

        Args:
            client: HTTP client.
            chat_id: Where to send it.
            intent: What was asked for.
            result: What happened, carrying the resolved path.
        """
        if intent.intent not in _DOCUMENT_INTENTS or not result.ok:
            return
        if not self._settings.telegram_send_files:
            await self._send(
                client,
                chat_id,
                "Sending files over Telegram is turned off (QUAINEX_TELEGRAM_SEND_FILES=false).",
            )
            return

        path_value = (result.data or {}).get("path")
        if isinstance(path_value, str):
            await self._upload_document(client, chat_id, Path(path_value))

    async def _upload_document(
        self, client: httpx.AsyncClient, chat_id: int, document: Path
    ) -> None:
        """Upload one file to the chat as a document, with a size guard.

        Shared by "send me <file>", the zipped-folder send, and the file browser's
        tap-to-receive. Keeps the file byte-for-byte and reports rather than fails
        silently when it is missing, unreadable, or over Telegram's limit.

        Args:
            client: HTTP client.
            chat_id: Where to send it.
            document: The file to upload.
        """
        if not document.is_file():
            _log.warning("telegram_document_missing", path=str(document))
            await self._send(client, chat_id, f"{document.name} is no longer there.")
            return

        try:
            payload = document.read_bytes()
        except OSError as exc:
            _log.warning("telegram_document_unreadable", error=str(exc))
            await self._send(client, chat_id, f"Could not read {document.name}.")
            return

        if len(payload) > _MAX_DOCUMENT_BYTES:
            await self._send(
                client,
                chat_id,
                f"{document.name} is {len(payload) // 1024 // 1024} MB, over Telegram's "
                f"{_MAX_DOCUMENT_BYTES // 1024 // 1024} MB bot upload limit.",
            )
            return

        try:
            response = await client.post(
                f"{self._url()}/sendDocument",
                data={"chat_id": str(chat_id)},
                files={"document": (document.name, payload, "application/octet-stream")},
                timeout=180,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            _log.warning("telegram_document_upload_failed", error=str(exc))
            await self._send(client, chat_id, f"{document.name} could not be sent.")
            return

        _log.info("telegram_document_sent", name=document.name, bytes=len(payload))

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
        if intent.intent not in _IMAGE_INTENTS or not result.ok:
            return
        # These always carry a screenshot the user explicitly asked to see — a
        # browsed page, or a panic photo — so gating them behind the send-images
        # switch would make the feature silently do nothing. Screenshots and
        # ordinary webcam shots still respect the switch.
        if intent.intent not in _ALWAYS_SEND_IMAGE and not self._settings.telegram_send_screenshots:
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

        if data.startswith("do:"):
            await self._handle_menu_action(client, update.chat_id, data[3:])
            return

        for kind in ("nav", "get", "zipdir"):
            prefix = f"{kind}:"
            if data.startswith(prefix):
                await self._handle_browse_button(
                    client, update.chat_id, kind, data[len(prefix) :]
                )
                return

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

        # Transcription downloads the clip and runs Whisper — several seconds. Show
        # activity throughout, so a voice note does not sit in silence.
        async with self._typing(client, update.chat_id):
            file_response = await client.get(
                f"{self._url()}/getFile", params={"file_id": update.voice_file_id}
            )
            file_response.raise_for_status()
            remote_path = file_response.json()["result"]["file_path"]

            audio = await client.get(
                f"{_API_ROOT}/file/bot{self._token()}/{remote_path}", timeout=60
            )
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

    async def _handle_incoming_file(
        self, client: httpx.AsyncClient, update: TelegramUpdate
    ) -> None:
        """Save a file the user attached, at the location their caption names.

        The caption is the instruction: "save to downloads", "documents/reports as
        budget.xlsx", or nothing at all — in which case the file lands in a tidy
        ``Downloads/Quainex`` inbox. All the safety lives one layer down, in the
        controller: the destination is confined to the permitted roots, the name is
        sanitised, and an existing file is never overwritten. The bridge's job is
        only to fetch the bytes and report where they went.

        Args:
            client: HTTP client.
            update: The update carrying the attachment.
        """
        assert update.chat_id is not None  # noqa: S101 - checked by the caller
        if not self._settings.telegram_receive_files:
            await self._send(
                client,
                update.chat_id,
                "Saving files sent over Telegram is turned off "
                "(QUAINEX_TELEGRAM_RECEIVE_FILES=false).",
            )
            return

        size = update.file_size or 0
        if size > _MAX_INCOMING_BYTES:
            await self._send(
                client,
                update.chat_id,
                f"That file is {size // 1024 // 1024} MB. Telegram only lets a bot "
                f"download files up to {_MAX_INCOMING_BYTES // 1024 // 1024} MB, so I "
                "cannot save it. Send it split up, or copy it over another way.",
            )
            return

        location, rename = _parse_save_caption(update.caption)
        suggested = rename or update.file_name or f"file-{time.strftime('%Y%m%d-%H%M%S')}"

        # Downloading and writing can take a moment for a larger file; show activity.
        async with self._typing(client, update.chat_id):
            try:
                file_response = await client.get(
                    f"{self._url()}/getFile", params={"file_id": update.file_id}
                )
                file_response.raise_for_status()
                remote_path = file_response.json()["result"]["file_path"]

                download = await client.get(
                    f"{_API_ROOT}/file/bot{self._token()}/{remote_path}", timeout=180
                )
                download.raise_for_status()
            except httpx.HTTPError as exc:
                _log.warning("telegram_incoming_download_failed", error=str(exc))
                await self._send(
                    client, update.chat_id, "I could not download that file from Telegram."
                )
                return

            saved = self._commands.context.desktop.save_incoming_file(
                download.content, suggested_name=suggested, location=location
            )

        renamed = (
            ""
            if saved.name == suggested
            else f"\n(named {saved.name} — the folder already had one)"
        )
        await self._send(
            client,
            update.chat_id,
            f"💾 <b>Saved</b>\n<code>{_esc(str(saved))}</code>{_esc(renamed)}",
            parse_mode="HTML",
        )

    async def _builtin(self, command: str) -> str:
        """Answer a slash command locally, without the model.

        Returns HTML — the slash commands are the bridge's own "chrome", fully under
        its control, so they can be styled richly and safely. Arbitrary command
        *output* stays plain text, where an unescaped ``<`` in a file name or page
        title cannot break a message.

        Args:
            command: The slash command.

        Returns:
            The reply text, as Telegram-flavoured HTML.
        """
        verb = command.split()[0].lower()
        if verb == "/start":
            return _WELCOME
        if verb == "/help":
            return _HELP
        if verb == "/status":
            return await self._status_report()
        return f"Unknown command <code>{_esc(verb)}</code>. Try /help."

    async def _status_report(self) -> str:
        """Build a rich, live snapshot of the machine.

        Composed entirely from local, token-free sources — system metrics, Wi-Fi,
        the process list, the voice/listener state. Each piece is guarded on its
        own, so one unavailable subsystem degrades to a line rather than failing
        the whole report.

        Returns:
            A formatted multi-line status.
        """
        desktop = self._commands.context.desktop
        lines = ["📟 <b>Quainex — status</b>", ""]

        try:
            snap = desktop.system_info()
            battery = (
                f" · 🔋 {snap.battery_percent:.0f}%" if snap.battery_percent is not None else ""
            )
            uptime = _format_uptime(snap.uptime_seconds)
            lines.append(
                f"🖥 <b>CPU</b> {snap.cpu_percent:.0f}% · <b>RAM</b> {snap.memory_percent:.0f}% · "
                f"<b>Disk</b> {snap.disk_percent:.0f}%{battery}"
            )
            lines.append(f"⏱ Up {uptime}")
        except Exception as exc:
            lines.append(f"🖥 System metrics unavailable ({_esc(str(exc))}).")

        try:
            lines.append(f"📶 {_esc(desktop.wifi_status())}")
        except Exception:
            lines.append("📶 Wi-Fi state unavailable.")

        try:
            apps = desktop.list_running_apps(8)
            if apps:
                lines.append("🪟 Running: " + _esc(", ".join(apps)))
        except Exception:
            # The process list is a nicety; its absence just drops one line.
            _log.debug("status_running_apps_unavailable")

        voice = "on" if self._voice and self._voice.is_available else "off"
        lines.append(f"🎙 Voice notes: {voice}")

        lines.append("")
        lines.append(
            f"⚙️ {len(self._commands.catalogue)} commands · everything above cost 0 tokens"
        )
        blocked = ", ".join(sorted(i.value for i in TELEGRAM_BLOCKED_INTENTS))
        lines.append(f"🔒 Kept off Telegram: {blocked}")
        return "\n".join(lines)

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

    async def _send(
        self,
        client: httpx.AsyncClient,
        chat_id: int,
        text: str,
        *,
        parse_mode: str | None = None,
    ) -> None:
        """Send a message, optionally formatted, always delivered.

        ``parse_mode`` defaults to ``None`` — plain text — because arbitrary command
        *output* (a file name like ``twilio_2FA.txt``, a page title with an ``&``)
        must never be reinterpreted as markup. This is the scar from the Markdown
        era, when every reply carrying an underscore or asterisk was rejected with a
        400 and silently never arrived. So the bridge only opts specific,
        fully-controlled messages into ``"HTML"`` — ``/help``, ``/status``,
        confirmations — where it has escaped every dynamic value itself.

        And even those degrade rather than disappear: if Telegram rejects the
        formatting with a 400, the message is re-sent as plain text with the tags
        stripped. A styling mistake can cost the styling; it can never cost the
        message. Every other failure is logged, so nothing vanishes without a trace.

        Args:
            client: HTTP client.
            chat_id: Where to send.
            text: What to send, truncated to Telegram's limit.
            parse_mode: ``"HTML"`` for bridge-composed messages, else plain text.
        """
        body = _truncate(text)
        payload: dict[str, object] = {"chat_id": chat_id, "text": body}
        if parse_mode:
            payload["parse_mode"] = parse_mode
        try:
            response = await client.post(f"{self._url()}/sendMessage", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if parse_mode and exc.response.status_code == 400:
                # Formatting was rejected. The message itself must still arrive, so
                # retry once as plain text — the hard-won rule that a reply never
                # silently vanishes for the sake of how it looks.
                _log.warning("telegram_formatting_rejected", error=str(exc))
                await self._send(client, chat_id, _plain(body))
                return
            _log.warning("telegram_send_failed", error=str(exc))
        except httpx.HTTPError as exc:
            _log.warning("telegram_send_failed", error=str(exc))

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
                "text": _truncate(f"⚠️ <b>Confirm</b>\n\n{_esc(question)}"),
                "parse_mode": "HTML",
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


def _format_uptime(seconds: float) -> str:
    """Render an uptime in seconds as a short human string.

    Args:
        seconds: Uptime in seconds.

    Returns:
        Something like ``2d 3h`` or ``15m``.
    """
    total = int(seconds)
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


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


def _human_size(size: int | None) -> str:
    """Render a byte count as a short human string.

    Args:
        size: The size in bytes, or ``None``.

    Returns:
        Something like ``4 KB`` or ``2 MB``; empty when unknown.
    """
    if size is None:
        return ""
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024:
            return f"{value:.0f} {unit}"
        value /= 1024
    return f"{value:.0f} TB"


def _esc(text: str) -> str:
    """Escape the three characters that carry meaning in Telegram HTML.

    Applied to every dynamic value that goes into an HTML message — a Wi-Fi SSID, a
    process name, an error string — so a stray ``<`` or ``&`` styles nothing and,
    more importantly, does not make Telegram reject the whole message.

    Args:
        text: Untrusted text.

    Returns:
        The text, safe to drop between HTML tags.
    """
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _plain(html: str) -> str:
    """Reduce a piece of Telegram HTML back to readable plain text.

    The fallback path when Telegram rejects formatting: strip the tags and undo the
    escaping, so the reader gets the words even when the styling could not be shown.

    Args:
        html: The HTML that was rejected.

    Returns:
        Tag-free, unescaped text.
    """
    stripped = re.sub(r"<[^>]+>", "", html)
    return stripped.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")


#: The /start welcome. Short and inviting — the first thing a new user sees.
_WELCOME = (
    "🟢 <b>Quainex</b>\n"
    "<i>Your machine, in your pocket.</i>\n\n"
    "Send me a request in plain language — or a voice note — and I'll carry it out "
    "and reply. For example:\n"
    "• <code>open chrome</code>\n"
    "• <code>take a screenshot</code>\n"
    "• <code>send me my latest download</code>\n\n"
    "Anything risky asks first, with <b>Yes</b> / <b>No</b> buttons.\n\n"
    "<b>/menu</b> — quick-action buttons\n"
    "<b>/files</b> — browse folders and grab a file\n"
    "<b>/help</b> — everything I can do\n"
    "<b>/status</b> — a live snapshot of this machine"
)

#: The /help catalogue. Grouped and scannable rather than an alphabetical dump —
#: a person reaches for a capability by area ("sound", "files"), not by name.
_HELP = (
    "📖 <b>Quainex — what I can do</b>\n"
    "Talk to me in plain language, or send a <b>voice note</b>.\n\n"
    "🌐 <b>Browser</b> — I steer it and send a screenshot each step\n"
    "• <code>browse github.com</code> — open a site\n"
    "• <code>search best laptops 2026</code> — search the web\n"
    "• <code>scroll down</code> · <code>click Sign in</code> · <code>go back</code>\n"
    "• <code>close the browser</code>\n\n"
    "🖥 <b>Apps &amp; windows</b>\n"
    "• <code>open spotify</code> · <code>close spotify</code>\n"
    "• <code>what's running</code> · <code>list windows</code>\n"
    "• <code>minimise this window</code> · <code>maximise</code>\n\n"
    "🔊 <b>Sound &amp; media</b>\n"
    "• <code>set volume to 30</code> · <code>mute</code>\n"
    "• <code>play</code> · <code>pause</code> · <code>next</code>\n\n"
    "📸 <b>See &amp; capture</b>\n"
    "• <code>take a screenshot</code>\n"
    "• <code>take a webcam photo</code>\n\n"
    "📁 <b>Files &amp; folders</b>\n"
    "• <b>/files</b> — browse folders, tap a file to receive it\n"
    "• <code>open downloads</code> · <code>open desktop</code>\n"
    "• <code>create a folder called work</code>\n"
    "• <code>find report.pdf</code>\n"
    "• <code>send me my latest download</code>\n"
    "• <code>send me my work folder</code> — zips and sends it\n"
    "• <b>Attach any file</b> with a caption like "
    "<code>save to documents as report.pdf</code> — I'll save it here\n\n"
    "🛡 <b>Security &amp; location</b>\n"
    "• <code>panic</code> — lock, photo &amp; locate, sent to you\n"
    "• <code>where is my laptop</code>\n"
    "• <code>lock the screen</code>\n"
    "• <code>wifi off</code> · <code>wifi status</code>\n\n"
    "💡 <b>Display &amp; keyboard</b>\n"
    "• <code>set brightness to 50</code>\n"
    "• <code>keyboard light on</code> · <code>off</code>\n\n"
    "⌨️ <b>Type &amp; transfer to the PC</b>\n"
    "• <code>copy this to my PC: …</code> — set the PC clipboard\n"
    "• <code>type …</code> — type into the active window\n"
    "• <code>download &lt;link&gt; to downloads</code> — save a link onto the PC\n\n"
    "💬 <b>Ask &amp; code</b>\n"
    "• Ask me anything, or <code>write code to…</code>\n"
    "• <code>explain this file…</code> · <code>review…</code>\n"
    "• <code>read this PDF and…</code>\n\n"
    "⚡ <b>Power</b>\n"
    "• <code>sleep</code> · <code>shut down</code> · <code>restart</code>\n\n"
    "<b>/menu</b> — quick-action buttons · <b>/status</b> — a live snapshot\n\n"
    "Anything risky asks first, with <b>Yes</b> / <b>No</b> buttons. A few things "
    "stay on this machine for privacy: clipboard, screen reading, documents."
)


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
    attachment = _attachment(message)
    return TelegramUpdate(
        update_id=update_id,
        chat_id=message.get("chat", {}).get("id"),
        user_id=message.get("from", {}).get("id"),
        text=message.get("text"),
        voice_file_id=voice.get("file_id"),
        file_id=attachment.get("file_id"),
        file_name=attachment.get("file_name"),
        file_size=attachment.get("file_size"),
        caption=message.get("caption"),
    )


def _attachment(message: dict[str, Any]) -> dict[str, Any]:
    """Extract a savable attachment from a message, whatever kind it is.

    Telegram carries a document, a photo, a video and an audio clip in different
    shapes: a document is an object with a name, a photo is an array of sizes with
    none. This reduces all of them to ``file_id`` / ``file_name`` / ``file_size``,
    inventing a sensible name and extension for the kinds that arrive without one so
    a photo saves as a ``.jpg`` rather than a mystery blob.

    A voice note is deliberately excluded: those are transcribed as commands, not
    saved as files.

    Args:
        message: The Telegram message object.

    Returns:
        The attachment fields, or an empty dict when there is nothing to save.
    """
    if document := message.get("document"):
        return {
            "file_id": document.get("file_id"),
            "file_name": document.get("file_name"),
            "file_size": document.get("file_size"),
        }
    if photos := message.get("photo"):
        # An array of the same image at increasing sizes; the last is the largest.
        largest = photos[-1]
        return {
            "file_id": largest.get("file_id"),
            "file_name": f"photo-{time.strftime('%Y%m%d-%H%M%S')}.jpg",
            "file_size": largest.get("file_size"),
        }
    for kind, extension in (("video", ".mp4"), ("audio", ".mp3")):
        if media := message.get(kind):
            return {
                "file_id": media.get("file_id"),
                "file_name": media.get("file_name")
                or f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}{extension}",
                "file_size": media.get("file_size"),
            }
    return {}


def _parse_save_caption(caption: str | None) -> tuple[str | None, str | None]:
    """Read a save instruction from an attachment's caption.

    Understands "save to downloads", "documents/reports", "keep this in documents as
    budget.xlsx", or nothing. Returns the location (or ``None`` for the default
    inbox) and an optional new name from an "as <name>" clause.

    Args:
        caption: The caption sent with the file.

    Returns:
        ``(location, rename)``.
    """
    text = (caption or "").strip()
    if not text:
        return None, None

    rename: str | None = None
    if match := re.search(r"\bas\s+(.+)$", text, re.IGNORECASE):
        rename = match.group(1).strip().strip("\"'") or None
        text = text[: match.start()].strip()

    # Peel off the leading verbs and prepositions, so "please save this to
    # downloads" and "downloads" both reduce to the location "downloads".
    text = re.sub(r"^(please\s+)?(save|store|put|keep|download|drop)\b", "", text, flags=re.I)
    text = re.sub(r"^\s*(it|this|that|the file|here)\b", "", text.strip(), flags=re.I)
    text = re.sub(r"^\s*(to|in|into|inside|on|at|under)\b", "", text.strip(), flags=re.I)
    location = text.strip().strip("/\\").strip() or None
    return location, rename
