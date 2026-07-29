"""Command registry and dispatch.

Phase 3. Maps Brain intents onto executable, permission-checked commands.
"""

from quainex.core.commands.base import (
    Command,
    CommandOutcome,
    CommandResult,
    CommandStatus,
)
from quainex.core.commands.executor import CommandExecutor, CommandRegistry, build_executor

__all__ = [
    "Command",
    "CommandExecutor",
    "CommandOutcome",
    "CommandRegistry",
    "CommandResult",
    "CommandStatus",
    "build_executor",
]
