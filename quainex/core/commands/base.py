"""Command types: what a command is, and what executing one produces.

Purpose:
    Give every action a uniform shape so the executor can enforce one policy
    across all of them, and callers get one result type to handle.

Why commands are data, not classes:
    A ``Command`` is a handler plus its metadata. Fifteen subclasses that each
    override one method would be fifteen files of ceremony around fifteen
    function bodies. The dataclass keeps the interesting part — the handler —
    adjacent to the policy flags that govern it.

Why a status enum rather than exceptions for everything:
    "Refused pending confirmation" and "blocked by configuration" are not errors;
    they are *outcomes the caller must render*. Modelling them as statuses means
    the API returns 200 with an explicit state rather than an error the client
    has to reverse-engineer. Genuine faults still raise.

Architecture:
    Intent -> CommandExecutor -> Command.handler(desktop, intent) -> str
                                                                  -> CommandResult

Dependencies:
    pydantic, quainex.core.automation.desktop, quainex.core.brain

Future improvements:
    * Add an ``undo`` handler so reversible commands can be rolled back.
    * Add per-command cooldowns to blunt runaway loops in Phase 10.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from quainex.core.automation.desktop import DesktopController
    from quainex.core.brain import Intent, IntentType
    from quainex.core.browser import BrowserSession
    from quainex.core.conversation import Conversationalist
    from quainex.core.devtools.assistant import CodeAssistant
    from quainex.core.devtools.runner import DevRunner
    from quainex.vision.screen import ScreenAnalyst


class CommandStatus(StrEnum):
    """The outcome of an execution attempt.

    Only ``SUCCEEDED`` and ``FAILED`` imply the action was attempted. The other
    three are refusals recorded *before* any side effect.
    """

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REQUIRES_CONFIRMATION = "requires_confirmation"
    BLOCKED = "blocked"
    UNSUPPORTED = "unsupported"


class CommandResult(BaseModel):
    """What happened when Quainex tried to act on an intent.

    Attributes:
        status: The outcome.
        intent: The intent that was dispatched.
        message: Human-readable description, suitable for speaking aloud.
        executed: Whether any side effect occurred. Explicit rather than derived,
            so an auditor never has to infer it from the status.
        data: Optional structured payload (search hits, system metrics).
        confirmation_token: Present only on ``REQUIRES_CONFIRMATION``. Bound to
            this exact action; present it back to execute the command. A caller
            cannot mint one, which is what stops a remote client from simply
            declaring that the user agreed.
    """

    status: CommandStatus
    intent: str
    message: str
    executed: bool
    data: dict[str, object] | None = None
    confirmation_token: str | None = None

    @property
    def ok(self) -> bool:
        """Whether the command completed successfully."""
        return self.status is CommandStatus.SUCCEEDED


@dataclass(frozen=True, slots=True)
class CommandContext:
    """Everything a command handler is allowed to reach.

    Phases 1-6 needed only the desktop controller, so handlers took one. Phases 7
    and 8 added commands that call a model or run a build tool, and passing three
    more positional arguments to fifteen handlers — most of which need none of
    them — would have been the wrong shape. A context object means a handler
    reaches for what it needs and ignores the rest, and adding a collaborator in
    Phase 9 does not touch a single existing handler.

    Attributes:
        desktop: OS-level actions.
        dev: Development operations, when configured.
        code: AI-backed code assistance, when configured.
        vision: Screen and document understanding, when configured.
        conversation: Replies for the intents that are not machine actions, when
            configured.
        browser: A steerable web browser, when configured.
    """

    desktop: DesktopController
    dev: DevRunner | None = None
    code: CodeAssistant | None = None
    vision: ScreenAnalyst | None = None
    conversation: Conversationalist | None = None
    browser: BrowserSession | None = None


#: A command implementation: given the context and the intent, do the thing and
#: describe it. Raises ``CommandExecutionError`` / ``CommandNotAllowedError``.
#:
#: Async because half the commands now await a model or a subprocess. Making the
#: sync handlers async too keeps one dispatch path rather than two.
CommandHandler = Callable[["CommandContext", "Intent"], Awaitable["CommandOutcome"]]


class CommandOutcome(BaseModel):
    """What a handler returns.

    Attributes:
        message: Human-readable description of what happened.
        data: Optional structured payload to pass through to the caller.
    """

    message: str
    data: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class Command:
    """One executable action, bound to the intent that triggers it.

    Attributes:
        intent: The intent this command handles.
        summary: One-line description, surfaced by the catalogue endpoint.
        handler: The implementation.
        requires_target: Whether ``Intent.target`` must be present and non-empty.
        destructive: Whether this action is gated behind
            ``allow_destructive_commands``. Set independently of the Brain's
            confirmation policy: confirmation asks the user, this asks the
            operator, and both must pass.
        has_side_effect: Whether running this changes anything outside Quainex.
            Defaults to ``True`` because almost every command does; the exceptions
            are the ones that only produce text. It becomes ``CommandResult.
            executed``, which the audit trail records, so a command that merely
            answers a question must not claim the machine was touched.
    """

    intent: IntentType
    summary: str
    handler: CommandHandler
    requires_target: bool = False
    destructive: bool = False
    has_side_effect: bool = True
