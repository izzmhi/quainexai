"""AI-backed code assistance.

Purpose:
    Explain, review and generate code — the parts of the developer assistant that
    need a model rather than a subprocess.

Why generated code is returned, never written:
    ``generate`` hands back text. It does not create files, and it deliberately
    has no path parameter. Writing model output to disk on a spoken request is a
    different risk class from running ``git status``: it is silent, it can
    overwrite work, and the mistake is only visible later. The user pastes it, or
    a future phase adds an explicit write-with-confirmation flow.

Why review returns a structured verdict:
    A prose review reads well and cannot be acted on. A list of typed findings
    can be counted, filtered by severity, and — in Phase 10 — gated on.

Architecture:
    file path -> read (size-capped, inside permitted roots)
              -> AIProvider.complete()  for explain
              -> AIProvider.parse()     for review -> CodeReview
              -> text / CodeReview

Dependencies:
    quainex.services.ai, quainex.core.exceptions

Future improvements:
    * Feed the surrounding module, not just the file, so review sees callers.
    * Review a diff rather than a whole file, so a large codebase is affordable.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from quainex.core.exceptions import CommandExecutionError, CommandNotAllowedError
from quainex.core.logging import get_logger
from quainex.services.ai.provider import ChatMessage

if TYPE_CHECKING:
    from quainex.config.settings import Settings
    from quainex.services.ai.provider import AIProvider

_log = get_logger(__name__)

#: Largest file that will be sent for explanation or review.
MAX_SOURCE_CHARS = 60_000

#: Extensions treated as source. Excludes binaries, which would be meaningless,
#: and `.env`-style files, which should never be sent to a model at all.
_SOURCE_SUFFIXES = frozenset(
    {
        ".py",
        ".pyi",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".go",
        ".rs",
        ".java",
        ".kt",
        ".c",
        ".h",
        ".cpp",
        ".hpp",
        ".cs",
        ".rb",
        ".php",
        ".swift",
        ".dart",
        ".sh",
        ".ps1",
        ".sql",
        ".html",
        ".css",
        ".scss",
        ".toml",
        ".yaml",
        ".yml",
        ".json",
        ".md",
        ".txt",
    }
)

#: Never read these, whatever their extension. A model does not need your keys.
_FORBIDDEN_NAMES = frozenset({".env", ".env.local", "id_rsa", "credentials", ".npmrc"})

_EXPLAIN_SYSTEM = """
You are explaining code to the developer who owns it.

Lead with what the code is for, then how it works, then anything surprising.
Assume competence: do not explain what a for-loop is. Be concrete about this
code rather than general about the language. If something looks like a bug, say
so plainly.
""".strip()

_REVIEW_SYSTEM = """
You are reviewing code for the developer who wrote it.

Report every issue you find, including ones you are uncertain about. Do not
filter for importance — record a severity and let the reader decide. It is better
to surface a finding that gets dismissed than to silently drop a real bug.

For each finding give the line number if you can identify one, a one-sentence
description of the defect, and why it matters. Prefer correctness and security
issues over style.
""".strip()


class Severity(StrEnum):
    """How much a finding matters."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class Finding(BaseModel):
    """One issue identified during review.

    Attributes:
        severity: How much it matters.
        line: Line number, when one can be identified.
        summary: One-sentence statement of the defect.
        detail: Why it matters and what to do about it.
    """

    severity: Severity
    line: int | None = None
    summary: str
    detail: str


class CodeReview(BaseModel):
    """The result of reviewing a file.

    Attributes:
        verdict: One-line overall assessment.
        findings: Individual issues, most severe first.
    """

    verdict: str
    findings: list[Finding] = Field(default_factory=list)


class CodeAssistant:
    """Explains, reviews and generates code."""

    def __init__(self, provider: AIProvider, settings: Settings) -> None:
        """Construct the assistant.

        Args:
            provider: The language model backend.
            settings: Configuration supplying the permitted roots.
        """
        self._provider = provider
        self._settings = settings

    async def explain(self, path: str) -> str:
        """Explain what a source file does.

        Args:
            path: File to explain.

        Returns:
            The explanation.
        """
        source, resolved = self._read_source(path)
        _log.info("code_explained", path=str(resolved), characters=len(source))
        return await self._provider.complete(
            messages=[
                ChatMessage(
                    role="user",
                    content=f"Explain this file, `{resolved.name}`:\n\n```\n{source}\n```",
                )
            ],
            system=_EXPLAIN_SYSTEM,
        )

    async def review(self, path: str) -> CodeReview:
        """Review a source file for defects.

        Args:
            path: File to review.

        Returns:
            The structured review.
        """
        source, resolved = self._read_source(path)
        numbered = "\n".join(
            f"{number:>4} | {line}" for number, line in enumerate(source.splitlines(), start=1)
        )
        review = await self._provider.parse(
            messages=[
                ChatMessage(
                    role="user",
                    content=f"Review `{resolved.name}`:\n\n```\n{numbered}\n```",
                )
            ],
            output_model=CodeReview,
            system=_REVIEW_SYSTEM,
        )
        _log.info("code_reviewed", path=str(resolved), findings=len(review.findings))
        return review

    async def generate(self, request: str) -> str:
        """Generate code from a description.

        Returns text. Nothing is written to disk — see the module docstring.

        Args:
            request: What to write.

        Returns:
            The generated code, with whatever explanation the model attached.
        """
        _log.info("code_generated", characters=len(request))
        return await self._provider.complete(
            messages=[ChatMessage(role="user", content=request)],
            system=(
                "You write code for an experienced developer. Produce the code "
                "requested with brief commentary only where a choice is not "
                "obvious. Match common conventions for the language."
            ),
        )

    # -- internals --------------------------------------------------------

    def _read_source(self, path: str) -> tuple[str, Path]:
        """Read a source file, subject to the same containment rules as Phase 3.

        Args:
            path: Requested file path.

        Returns:
            The file contents and its resolved path.

        Raises:
            CommandNotAllowedError: Outside the permitted roots, a forbidden
                name, or not a recognised source type.
            CommandExecutionError: Missing, unreadable, or too large.
        """
        roots = self._settings.resolved_search_roots
        raw = Path(path.strip()).expanduser()
        base = roots[0] if roots else Path.cwd()
        resolved = (raw if raw.is_absolute() else base / raw).resolve()

        if not any(resolved.is_relative_to(root) for root in roots):
            raise CommandNotAllowedError(f"'{resolved}' is outside the folders Quainex may read.")
        if resolved.name.lower() in _FORBIDDEN_NAMES:
            # Sending a secrets file to a model is not a mistake worth being
            # clever about; it is simply refused.
            raise CommandNotAllowedError(f"'{resolved.name}' will not be sent to a model.")
        if resolved.suffix.lower() not in _SOURCE_SUFFIXES:
            raise CommandNotAllowedError(
                f"'{resolved.suffix or resolved.name}' is not a recognised source file type."
            )
        if not resolved.is_file():
            raise CommandExecutionError(f"No file at '{resolved}'.")

        try:
            source = resolved.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise CommandExecutionError(f"Could not read '{resolved}': {exc}") from exc

        if len(source) > MAX_SOURCE_CHARS:
            raise CommandExecutionError(
                f"'{resolved.name}' is {len(source)} characters; the limit is "
                f"{MAX_SOURCE_CHARS}. Point at a smaller file."
            )
        return source, resolved
