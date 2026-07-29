"""Command registry and executor — where safety policy is enforced.

Purpose:
    Decide whether an intent may be acted on, dispatch it if so, and record what
    happened either way.

Why every gate lives here rather than in each command:
    Fifteen handlers enforcing their own confirmation checks is fifteen chances
    to forget one, and the one that gets forgotten will be the dangerous one.
    Every path to a side effect runs through ``execute()``, so the gates are
    written once and cannot be bypassed by adding a new command.

The gate order, and why it is this order:

    1. **Registered?**       An intent with no command cannot act.
    2. **Confirmed?**        The Brain flagged it; the user has not said yes.
    3. **Operator allows?**  Destructive actions need ``allow_destructive_commands``.
    4. **Target present?**   Reject incomplete requests before touching the OS.
    5. **Execute.**

    Cheap, certain refusals come before expensive, fallible work, and — more
    importantly — every gate that can refuse *without a side effect* runs before
    the one that causes them.

Architecture:
    Intent + confirmed flag
        -> CommandRegistry.get(intent)
        -> four gates
        -> Command.handler(desktop, intent)
        -> audit log -> CommandResult

Dependencies:
    quainex.config.settings, quainex.core.automation, quainex.core.brain,
    quainex.core.commands, quainex.core.exceptions

Future improvements:
    * Rate-limit executions per intent to blunt runaway loops in Phase 10.
    * Record results to the Phase 5 memory store so "what did you just do" works.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from quainex.core.brain import IntentType
from quainex.core.commands.base import Command, CommandResult, CommandStatus
from quainex.core.commands.builtin import build_commands
from quainex.core.exceptions import CommandExecutionError, CommandNotAllowedError
from quainex.core.logging import get_logger

if TYPE_CHECKING:
    from quainex.config.settings import Settings
    from quainex.core.automation.desktop import DesktopController
    from quainex.core.brain import Intent

_log = get_logger(__name__)


class CommandRegistry:
    """Maps intents onto the commands that carry them out."""

    def __init__(self, commands: list[Command]) -> None:
        """Build the registry.

        Args:
            commands: Commands to register.

        Raises:
            ValueError: Two commands claim the same intent, which would make
                dispatch depend on registration order.
        """
        self._commands: dict[IntentType, Command] = {}
        for command in commands:
            if command.intent in self._commands:
                raise ValueError(f"Duplicate command registered for intent '{command.intent}'")
            self._commands[command.intent] = command

    def get(self, intent: IntentType) -> Command | None:
        """Return the command for an intent, if one is registered.

        Args:
            intent: The intent to look up.

        Returns:
            The command, or ``None`` when the intent is not executable.
        """
        return self._commands.get(intent)

    def catalogue(self) -> dict[str, str]:
        """Describe every registered command.

        Returns:
            Intent value mapped to its one-line summary.
        """
        return {intent.value: command.summary for intent, command in self._commands.items()}


class CommandExecutor:
    """Applies safety policy, then dispatches intents to commands."""

    def __init__(
        self,
        registry: CommandRegistry,
        desktop: DesktopController,
        settings: Settings,
    ) -> None:
        """Construct the executor.

        Args:
            registry: Commands available for dispatch.
            desktop: The controller commands act through.
            settings: Configuration supplying the destructive-action switch.
        """
        self._registry = registry
        self._desktop = desktop
        self._settings = settings

    @property
    def catalogue(self) -> dict[str, str]:
        """Every executable intent and its summary."""
        return self._registry.catalogue()

    def execute(self, intent: Intent, *, confirmed: bool = False) -> CommandResult:
        """Run the command for an intent, subject to policy.

        Args:
            intent: The classified intent to act on.
            confirmed: Whether the user has explicitly approved this action.
                Only meaningful when ``intent.requires_confirmation`` is set.

        Returns:
            The outcome, including whether any side effect occurred.
        """
        command = self._registry.get(intent.intent)

        if command is None:
            return self._refuse(
                intent,
                CommandStatus.UNSUPPORTED,
                f"'{intent.intent.value}' is not something Quainex can execute.",
            )

        if intent.requires_confirmation and not confirmed:
            return self._refuse(
                intent,
                CommandStatus.REQUIRES_CONFIRMATION,
                self._confirmation_prompt(intent),
            )

        if command.destructive and not self._settings.allow_destructive_commands:
            return self._refuse(
                intent,
                CommandStatus.BLOCKED,
                (
                    f"'{intent.intent.value}' is disabled. "
                    f"Set QUAINEX_ALLOW_DESTRUCTIVE_COMMANDS=true to enable power actions."
                ),
            )

        if command.requires_target and not (intent.target or "").strip():
            return self._refuse(
                intent,
                CommandStatus.FAILED,
                f"'{intent.intent.value}' needs a target, but none was identified.",
            )

        return self._dispatch(command, intent)

    # -- internals --------------------------------------------------------

    def _dispatch(self, command: Command, intent: Intent) -> CommandResult:
        """Run a command that has cleared every gate.

        Args:
            command: The command to run.
            intent: The intent being acted on.

        Returns:
            The outcome of the attempt.
        """
        try:
            outcome = command.handler(self._desktop, intent)
        except CommandNotAllowedError as exc:
            # Refused inside the controller (unknown app, path outside roots).
            # No side effect occurred, so this is a refusal, not a failure.
            _log.warning(
                "command_refused",
                intent=intent.intent.value,
                target=intent.target,
                reason=exc.message,
            )
            return CommandResult(
                status=CommandStatus.BLOCKED,
                intent=intent.intent.value,
                message=exc.message,
                executed=False,
            )
        except CommandExecutionError as exc:
            # Logged at error level without a traceback: the message already
            # states the cause, and the stack is our own dispatch frames.
            _log.error(
                "command_failed",
                intent=intent.intent.value,
                target=intent.target,
                reason=exc.message,
            )
            return CommandResult(
                status=CommandStatus.FAILED,
                intent=intent.intent.value,
                message=exc.message,
                executed=False,
            )

        _log.info(
            "command_executed",
            intent=intent.intent.value,
            target=intent.target,
            confirmed=intent.requires_confirmation,
        )
        return CommandResult(
            status=CommandStatus.SUCCEEDED,
            intent=intent.intent.value,
            message=outcome.message,
            executed=True,
            data=outcome.data,
        )

    @staticmethod
    def _confirmation_prompt(intent: Intent) -> str:
        """Phrase the question the user must answer before this action runs.

        Args:
            intent: The intent awaiting confirmation.

        Returns:
            A question naming the action and its target.
        """
        action = intent.intent.value.replace("_", " ")
        if intent.target:
            return f"Confirm: {action} '{intent.target}'?"
        return f"Confirm: {action}?"

    @staticmethod
    def _refuse(intent: Intent, status: CommandStatus, message: str) -> CommandResult:
        """Record a refusal that occurred before any side effect.

        Args:
            intent: The intent that was refused.
            status: The refusal status.
            message: Explanation for the caller.

        Returns:
            A result with ``executed=False``.
        """
        _log.info(
            "command_not_executed",
            intent=intent.intent.value,
            target=intent.target,
            status=status.value,
        )
        return CommandResult(
            status=status,
            intent=intent.intent.value,
            message=message,
            executed=False,
        )


def build_executor(desktop: DesktopController, settings: Settings) -> CommandExecutor:
    """Assemble a registry of built-in commands and an executor over it.

    Args:
        desktop: The controller commands act through.
        settings: Application configuration.

    Returns:
        A ready-to-use executor.
    """
    return CommandExecutor(
        registry=CommandRegistry(build_commands(settings)),
        desktop=desktop,
        settings=settings,
    )
