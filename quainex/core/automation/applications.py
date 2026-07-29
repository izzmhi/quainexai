"""Application allowlist and name resolution.

Purpose:
    Map a friendly name the user spoke ("my editor", "vs code") onto a specific
    executable that Quainex is permitted to launch or terminate.

Why an allowlist rather than "just run what the user said":
    ``Intent.target`` is model output derived from speech. Passing it to the
    shell — or even resolving it as a path — turns every misheard word into
    arbitrary code execution, and turns prompt injection into remote code
    execution once Phase 6 exposes the API to a phone. The catalogue below is the
    boundary: a name that does not resolve is refused, and the refusal names what
    was asked for so the user can see the miss.

    This is defence in depth alongside two other rules the launcher follows:
    arguments are never taken from model output, and no subprocess is ever
    started with ``shell=True``.

Architecture:
    Intent.target ("vs code")
        -> normalise (case, punctuation, filler words)
        -> match against key / display name / aliases
        -> ApplicationSpec (executables to try, process names to kill)
        -> WindowsDesktopController launches or terminates it

Dependencies:
    Standard library only.

Future improvements:
    * Let users register their own applications from a config file, with the same
      spec shape, so the catalogue is extensible without a code change.
    * Learn aliases from corrections in Phase 5 memory ("no, I meant Spotify").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ApplicationSpec:
    """One application Quainex is permitted to control.

    Attributes:
        key: Stable canonical identifier.
        display: Human-readable name used in responses.
        executables: Candidate executables, tried in order via ``shutil.which``.
        process_names: Process image names used when terminating the app.
        uri: Optional protocol URI used when no executable resolves (Store apps).
        aliases: Additional spoken forms that should resolve to this app.
    """

    key: str
    display: str
    executables: tuple[str, ...] = ()
    process_names: tuple[str, ...] = ()
    uri: str | None = None
    aliases: tuple[str, ...] = field(default_factory=tuple)


#: The applications Quainex may launch or close. Anything absent is refused.
APPLICATION_CATALOGUE: tuple[ApplicationSpec, ...] = (
    ApplicationSpec(
        key="vscode",
        display="Visual Studio Code",
        executables=("code.cmd", "code.exe", "code"),
        process_names=("Code.exe",),
        aliases=("vs code", "vscode", "visual studio code", "code editor", "my editor"),
    ),
    ApplicationSpec(
        key="notepad",
        display="Notepad",
        executables=("notepad.exe",),
        process_names=("notepad.exe",),
        aliases=("note pad", "text editor"),
    ),
    ApplicationSpec(
        key="calculator",
        display="Calculator",
        executables=("calc.exe",),
        process_names=("CalculatorApp.exe", "Calculator.exe"),
        aliases=("calc",),
    ),
    ApplicationSpec(
        key="explorer",
        display="File Explorer",
        executables=("explorer.exe",),
        process_names=("explorer.exe",),
        aliases=("file explorer", "files", "windows explorer", "file manager"),
    ),
    ApplicationSpec(
        key="terminal",
        display="Windows Terminal",
        executables=("wt.exe", "powershell.exe"),
        process_names=("WindowsTerminal.exe",),
        aliases=("windows terminal", "console", "command line", "shell"),
    ),
    ApplicationSpec(
        key="powershell",
        display="PowerShell",
        executables=("powershell.exe",),
        process_names=("powershell.exe",),
        aliases=("power shell",),
    ),
    ApplicationSpec(
        key="chrome",
        display="Google Chrome",
        executables=("chrome.exe",),
        process_names=("chrome.exe",),
        aliases=("google chrome",),
    ),
    ApplicationSpec(
        key="edge",
        display="Microsoft Edge",
        executables=("msedge.exe",),
        process_names=("msedge.exe",),
        aliases=("microsoft edge",),
    ),
    ApplicationSpec(
        key="firefox",
        display="Firefox",
        executables=("firefox.exe",),
        process_names=("firefox.exe",),
        aliases=("mozilla firefox",),
    ),
    ApplicationSpec(
        key="spotify",
        display="Spotify",
        executables=("spotify.exe",),
        process_names=("Spotify.exe",),
        uri="spotify:",
        aliases=("music",),
    ),
    ApplicationSpec(
        key="discord",
        display="Discord",
        executables=("discord.exe", "Update.exe"),
        process_names=("Discord.exe",),
    ),
    ApplicationSpec(
        key="paint",
        display="Paint",
        executables=("mspaint.exe",),
        process_names=("mspaint.exe",),
        aliases=("ms paint",),
    ),
    ApplicationSpec(
        key="task_manager",
        display="Task Manager",
        executables=("taskmgr.exe",),
        process_names=("Taskmgr.exe",),
        aliases=("task manager",),
    ),
    ApplicationSpec(
        key="settings",
        display="Windows Settings",
        executables=(),
        uri="ms-settings:",
        aliases=("windows settings", "system settings", "control panel"),
    ),
)

#: Filler words stripped before matching, so "open up my vs code please" resolves.
_FILLER = frozenset({"the", "my", "a", "an", "app", "application", "program", "please", "up"})

_PUNCTUATION = re.compile(r"[^\w\s]")


def _normalise(name: str) -> str:
    """Reduce a spoken name to a comparable form.

    Args:
        name: Raw name as extracted from the utterance.

    Returns:
        Lower-cased text with punctuation and filler words removed.
    """
    cleaned = _PUNCTUATION.sub(" ", name.lower())
    words = [w for w in cleaned.split() if w not in _FILLER]
    return " ".join(words)


def resolve_application(name: str) -> ApplicationSpec | None:
    """Find the catalogue entry matching a spoken application name.

    Matching runs strictest-first — exact match on key, display name or alias,
    then a containment check — so "code" prefers Visual Studio Code over any
    entry that merely contains the word.

    Args:
        name: The name the user used.

    Returns:
        The matching spec, or ``None`` if the application is not allowlisted.
    """
    target = _normalise(name)
    if not target:
        return None

    for spec in APPLICATION_CATALOGUE:
        candidates = {_normalise(spec.key), _normalise(spec.display)}
        candidates.update(_normalise(alias) for alias in spec.aliases)
        if target in candidates:
            return spec

    for spec in APPLICATION_CATALOGUE:
        candidates = {_normalise(spec.key), _normalise(spec.display)}
        candidates.update(_normalise(alias) for alias in spec.aliases)
        if any(target in c or c in target for c in candidates if c):
            return spec

    return None


def known_application_names() -> list[str]:
    """List the display names of every allowlisted application.

    Used in error messages so a refusal tells the user what *is* available.

    Returns:
        Display names in catalogue order.
    """
    return [spec.display for spec in APPLICATION_CATALOGUE]
