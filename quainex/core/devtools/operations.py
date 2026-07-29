"""Developer operation catalogue.

Purpose:
    Define exactly which development commands Quainex may run.

Why complete argument vectors rather than a command allowlist:
    The obvious design is "allow the ``git`` executable, plus a list of permitted
    subcommands". It is not enough. ``git`` alone reaches arbitrary code
    execution through several doors — ``git -c core.pager=<anything> log``,
    ``git submodule``, aliases defined in a repo's own config. Allowing an
    executable means allowing everything that executable can be talked into.

    So an operation here is a **fixed argv list**. Nothing about it is assembled
    from model output except values that land in explicitly declared slots, each
    validated for shape. ``git.commit`` can only ever run
    ``git commit -m <message>`` — there is no arrangement of words the Brain can
    produce that turns it into something else.

Why "run this Python file" is not in the catalogue:
    It would be arbitrary code execution wearing a lanyard. Running a project's
    *test suite* is a fixed, recognisable operation; running an arbitrary script
    the model picked is not, and no amount of path validation changes what the
    script does once started. Phase 10 may revisit this with a sandbox.

Architecture:
    Intent(RUN_DEV_COMMAND, target="git.status")
        -> DevOperation lookup
        -> DevRunner: validate slots, resolve cwd inside permitted roots, run
        -> captured stdout/stderr, truncated

Dependencies:
    Standard library only.

Future improvements:
    * Per-project operation sets, so a Node project offers ``npm test`` and a
      Python one offers ``pytest``, detected from the manifest present.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Longest accepted commit message. Long enough for a real message, short enough
#: that a runaway model cannot build a megabyte-sized argv.
MAX_MESSAGE_CHARS = 500


@dataclass(frozen=True, slots=True)
class DevOperation:
    """One development command Quainex is permitted to run.

    Attributes:
        key: Stable identifier, e.g. ``git.status``.
        summary: One-line description for the catalogue and the Brain's prompt.
        argv: The complete command, with ``{message}`` as the only substitutable
            slot. Nothing else is ever interpolated.
        mutating: Whether this changes repository or system state, and therefore
            requires user confirmation before it runs.
        needs_message: Whether the operation requires a message argument.
    """

    key: str
    summary: str
    argv: tuple[str, ...]
    mutating: bool = False
    needs_message: bool = False
    #: Exit codes that are not failures. `git diff --quiet` returns 1 to mean
    #: "there are differences", and pytest returns 1 for "tests failed" — both
    #: are useful answers, not errors.
    tolerated_exit_codes: frozenset[int] = field(default_factory=lambda: frozenset({0}))


#: Every development operation Quainex can perform. Anything absent is refused.
DEV_OPERATIONS: tuple[DevOperation, ...] = (
    # --- git: inspection ---
    DevOperation(
        key="git.status",
        summary="Show which files have changed.",
        argv=("git", "status", "--short", "--branch"),
    ),
    DevOperation(
        key="git.log",
        summary="Show the last 20 commits.",
        argv=("git", "log", "--oneline", "-20"),
    ),
    DevOperation(
        key="git.diff",
        summary="Show unstaged changes as a summary.",
        argv=("git", "diff", "--stat"),
        tolerated_exit_codes=frozenset({0, 1}),
    ),
    DevOperation(
        key="git.diff.staged",
        summary="Show staged changes as a summary.",
        argv=("git", "diff", "--cached", "--stat"),
        tolerated_exit_codes=frozenset({0, 1}),
    ),
    DevOperation(
        key="git.branch",
        summary="List branches.",
        argv=("git", "branch", "--list"),
    ),
    # --- git: mutating ---
    DevOperation(
        key="git.add",
        summary="Stage every change in the repository.",
        argv=("git", "add", "--all"),
        mutating=True,
    ),
    DevOperation(
        key="git.commit",
        summary="Commit staged changes with a message.",
        argv=("git", "commit", "-m", "{message}"),
        mutating=True,
        needs_message=True,
    ),
    DevOperation(
        key="git.push",
        summary="Push the current branch to its remote.",
        argv=("git", "push"),
        mutating=True,
    ),
    DevOperation(
        key="git.pull",
        summary="Pull the current branch from its remote.",
        argv=("git", "pull", "--ff-only"),
        mutating=True,
    ),
    # --- quality gates ---
    DevOperation(
        key="tests.run",
        summary="Run the test suite.",
        argv=("python", "-m", "pytest", "-q"),
        # 1 means tests failed, 5 means none were collected. Both are results.
        tolerated_exit_codes=frozenset({0, 1, 5}),
    ),
    DevOperation(
        key="lint.run",
        summary="Run the linter.",
        argv=("python", "-m", "ruff", "check", "."),
        tolerated_exit_codes=frozenset({0, 1}),
    ),
    DevOperation(
        key="format.check",
        summary="Check formatting without changing anything.",
        argv=("python", "-m", "ruff", "format", "--check", "."),
        tolerated_exit_codes=frozenset({0, 1}),
    ),
    DevOperation(
        key="types.check",
        summary="Run the type checker.",
        argv=("python", "-m", "mypy", "."),
        tolerated_exit_codes=frozenset({0, 1}),
    ),
    # --- docker: inspection only ---
    DevOperation(
        key="docker.ps",
        summary="List running containers.",
        argv=("docker", "ps"),
    ),
    DevOperation(
        key="docker.images",
        summary="List local images.",
        argv=("docker", "images"),
    ),
)

_BY_KEY: dict[str, DevOperation] = {operation.key: operation for operation in DEV_OPERATIONS}


def resolve_operation(key: str) -> DevOperation | None:
    """Find an operation by key.

    Matching is exact after normalisation. Fuzzy matching is deliberately absent:
    "git.push" and "git.pull" differ by one character and do very different
    things, and a near-miss here would be acted on.

    Args:
        key: The operation identifier, e.g. ``git.status``.

    Returns:
        The operation, or ``None`` when it is not in the catalogue.
    """
    return _BY_KEY.get(key.strip().lower().replace(" ", "."))


def operation_catalogue() -> dict[str, str]:
    """Describe every available operation.

    Returns:
        Operation key mapped to its summary.
    """
    return {operation.key: operation.summary for operation in DEV_OPERATIONS}
