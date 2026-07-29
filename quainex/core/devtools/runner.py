"""Developer operation runner.

Purpose:
    Execute catalogue operations inside a project directory, safely, and return
    what they printed.

Three guarantees:
    1. **The argv comes from the catalogue, never from the model.** Only declared
       slots are substituted, and each is validated for shape first.
    2. **The working directory is inside a permitted root.** Resolved to its
       canonical form before the containment check, same rule as Phase 3.
    3. **Output is bounded.** A test suite can print megabytes; the tail is what
       matters, so that is what is kept.

Why the tail rather than the head:
    A failing pytest run puts the summary at the end, a linter puts the count at
    the end, and git puts the result at the end. Truncating from the front would
    reliably discard the part a user actually asked for.

Dependencies:
    quainex.core.devtools.operations, quainex.core.exceptions

Future improvements:
    * Stream output over the WebSocket, so a long test run reports progress.
    * Detect the project's toolchain (poetry, uv, npm) rather than assuming the
      current interpreter's module invocations.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from quainex.core.devtools.operations import (
    MAX_MESSAGE_CHARS,
    operation_catalogue,
    resolve_operation,
)
from quainex.core.exceptions import CommandExecutionError, CommandNotAllowedError
from quainex.core.logging import get_logger

if TYPE_CHECKING:
    from quainex.config.settings import Settings

_log = get_logger(__name__)

#: Characters of output retained. Enough for a full pytest summary.
MAX_OUTPUT_CHARS = 8000

#: A test suite can legitimately take minutes; anything beyond this is wedged.
_TIMEOUT_SECONDS = 300

#: Control characters are stripped from a commit message: they cannot help, and
#: they make log output ambiguous.
_MESSAGE_FORBIDDEN = frozenset({"\x00", "\r"})


class DevResult(BaseModel):
    """The outcome of running a development operation.

    Attributes:
        operation: The operation key that ran.
        command: The command as executed, for the audit trail.
        directory: Where it ran.
        exit_code: The process exit code.
        succeeded: Whether the exit code was one the operation tolerates.
        output: Captured stdout and stderr, tail-truncated.
        truncated: Whether output was cut.
    """

    operation: str
    command: str
    directory: str
    exit_code: int
    succeeded: bool
    output: str
    truncated: bool = False


class DevRunner:
    """Runs catalogue operations inside permitted project directories."""

    def __init__(self, settings: Settings) -> None:
        """Construct the runner.

        Args:
            settings: Configuration supplying the permitted roots.
        """
        self._settings = settings

    @property
    def catalogue(self) -> dict[str, str]:
        """Every available operation and its summary."""
        return operation_catalogue()

    def run(
        self,
        operation_key: str,
        directory: str | None = None,
        message: str | None = None,
    ) -> DevResult:
        """Run a catalogue operation.

        Args:
            operation_key: Which operation, e.g. ``git.status``.
            directory: Project directory; defaults to the first permitted root.
            message: Message argument, for operations that take one.

        Returns:
            The captured result.

        Raises:
            CommandNotAllowedError: Unknown operation, bad message, or a
                directory outside the permitted roots.
            CommandExecutionError: The tool is not installed, or the run failed.
        """
        operation = resolve_operation(operation_key)
        if operation is None:
            available = ", ".join(sorted(self.catalogue))
            raise CommandNotAllowedError(
                f"'{operation_key}' is not an available operation. Available: {available}."
            )

        workdir = self._resolve_directory(directory)
        argv = self._build_argv(operation.argv, operation.needs_message, message)

        # Resolve the executable to an absolute path so PATH cannot be used to
        # substitute a different `git` or `docker`.
        resolved = shutil.which(argv[0])
        if resolved is None:
            raise CommandExecutionError(
                f"'{argv[0]}' is not installed or not on PATH, so '{operation.key}' cannot run."
            )
        argv = [resolved, *argv[1:]]

        try:
            completed = subprocess.run(  # noqa: S603 - catalogue argv, absolute exe, no shell
                argv,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise CommandExecutionError(
                f"'{operation.key}' did not finish within {_TIMEOUT_SECONDS} seconds."
            ) from exc
        except OSError as exc:
            raise CommandExecutionError(f"Could not run '{operation.key}': {exc}") from exc

        output, truncated = _tail(f"{completed.stdout}{completed.stderr}".strip())
        succeeded = completed.returncode in operation.tolerated_exit_codes

        _log.info(
            "dev_operation_ran",
            operation=operation.key,
            directory=str(workdir),
            exit_code=completed.returncode,
            succeeded=succeeded,
            mutating=operation.mutating,
        )
        return DevResult(
            operation=operation.key,
            # The message is included as an argument here, not interpolated into
            # a shell string — this is a record of what ran, not a runnable line.
            command=" ".join(argv[1:]) if len(argv) > 1 else argv[0],
            directory=str(workdir),
            exit_code=completed.returncode,
            succeeded=succeeded,
            output=output or "(no output)",
            truncated=truncated,
        )

    # -- internals --------------------------------------------------------

    def _resolve_directory(self, directory: str | None) -> Path:
        """Resolve and validate the working directory.

        Args:
            directory: Requested directory, or ``None`` for the default root.

        Returns:
            An existing directory inside a permitted root.

        Raises:
            CommandNotAllowedError: The path escapes the permitted roots.
            CommandExecutionError: The path does not exist or is not a directory.
        """
        roots = self._settings.resolved_search_roots
        if not roots:
            raise CommandNotAllowedError("No permitted project roots are configured.")

        if directory is None or not directory.strip():
            return roots[0]

        raw = Path(directory.strip()).expanduser()
        resolved = (raw if raw.is_absolute() else roots[0] / raw).resolve()

        if not any(resolved.is_relative_to(root) for root in roots):
            allowed = ", ".join(str(root) for root in roots)
            raise CommandNotAllowedError(
                f"'{resolved}' is outside the folders Quainex may work in. Allowed: {allowed}."
            )
        if not resolved.is_dir():
            raise CommandExecutionError(f"'{resolved}' is not a directory.")
        return resolved

    @staticmethod
    def _build_argv(
        template: tuple[str, ...], needs_message: bool, message: str | None
    ) -> list[str]:
        """Fill the operation's argv template.

        Args:
            template: The catalogue argv, possibly containing ``{message}``.
            needs_message: Whether a message is required.
            message: The supplied message.

        Returns:
            The concrete argument list.

        Raises:
            CommandNotAllowedError: A required message is missing or unusable.
        """
        if not needs_message:
            return list(template)

        cleaned = (message or "").strip()
        if not cleaned:
            raise CommandNotAllowedError("This operation needs a message, but none was given.")
        if len(cleaned) > MAX_MESSAGE_CHARS:
            raise CommandNotAllowedError(
                f"The message is {len(cleaned)} characters; the limit is {MAX_MESSAGE_CHARS}."
            )
        if any(character in cleaned for character in _MESSAGE_FORBIDDEN):
            raise CommandNotAllowedError("The message contains characters that are not allowed.")

        # Substituted as a whole argv element. Because there is no shell, the
        # message cannot become anything other than one argument, whatever it
        # contains — quotes, semicolons, newlines and all.
        return [cleaned if part == "{message}" else part for part in template]


def _tail(text: str) -> tuple[str, bool]:
    """Keep the end of long output.

    Args:
        text: Full captured output.

    Returns:
        The retained text and whether it was truncated.
    """
    if len(text) <= MAX_OUTPUT_CHARS:
        return text, False
    return "…(earlier output trimmed)…\n" + text[-MAX_OUTPUT_CHARS:], True
