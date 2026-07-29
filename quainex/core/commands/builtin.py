"""Built-in command implementations.

Purpose:
    Bind each ``IntentType`` to the work that carries it out.

This module is deliberately thin. Every handler does three things: pull what it
needs off the intent, call one collaborator on the context, and describe the
result. All validation, allowlisting and path containment lives in those
collaborators; all policy lives in the executor. If a handler here grows a
branch, the logic probably belongs in one of those two places instead.

Architecture:
    IntentType -> Command(handler=...) -> CommandContext.{desktop,dev,code,vision}

Dependencies:
    quainex.core.{automation,brain,commands,devtools}, quainex.vision

Future improvements:
    * ``SEARCH_FILES`` should rank hits by recency once memory is consulted.
    * ``NOTIFY`` should also route through the WebSocket, so a notification
      reaches the phone client too.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from quainex.core.brain import IntentType
from quainex.core.commands.base import Command, CommandContext, CommandOutcome
from quainex.core.exceptions import CommandNotAllowedError

if TYPE_CHECKING:
    from quainex.config.settings import Settings
    from quainex.core.automation.desktop import LevelChange
    from quainex.core.brain import Intent


def _target(intent: Intent) -> str:
    """Return the intent's target, guaranteed non-empty.

    The executor rejects a missing target before a handler runs, so this is a
    narrowing helper rather than a validation step.

    Args:
        intent: The dispatched intent.

    Returns:
        The target text.
    """
    return (intent.target or "").strip()


def _require(component: object | None, name: str) -> None:
    """Fail clearly when a command's collaborator is not configured.

    Args:
        component: The collaborator to check.
        name: Human-readable name for the error message.

    Raises:
        CommandNotAllowedError: The collaborator is unavailable.
    """
    if component is None:
        raise CommandNotAllowedError(f"{name} is not available on this instance.")


def build_commands(settings: Settings) -> list[Command]:
    """Construct every built-in command, bound to the given configuration.

    Built as a factory rather than a module-level constant because several
    commands close over settings (search limits, shutdown delay, screenshot
    directory), and a module-level list would freeze whatever configuration
    happened to be loaded at import time.

    Args:
        settings: Configuration supplying limits and destinations.

    Returns:
        Every built-in command.
    """
    # --- desktop (Phase 3) ----------------------------------------------

    async def open_application(ctx: CommandContext, intent: Intent) -> CommandOutcome:
        return CommandOutcome(message=ctx.desktop.open_application(_target(intent)))

    async def close_application(ctx: CommandContext, intent: Intent) -> CommandOutcome:
        return CommandOutcome(message=ctx.desktop.close_application(_target(intent)))

    async def open_website(ctx: CommandContext, intent: Intent) -> CommandOutcome:
        return CommandOutcome(message=ctx.desktop.open_url(_target(intent)))

    async def open_folder(ctx: CommandContext, intent: Intent) -> CommandOutcome:
        return CommandOutcome(message=ctx.desktop.open_folder(_target(intent)))

    async def search_files(ctx: CommandContext, intent: Intent) -> CommandOutcome:
        hits = ctx.desktop.search_files(_target(intent), settings.command_search_max_results)
        message = (
            f"Found {len(hits)} file(s) matching '{_target(intent)}'."
            if hits
            else f"No files matched '{_target(intent)}'."
        )
        return CommandOutcome(message=message, data={"results": [hit.model_dump() for hit in hits]})

    async def lock_screen(ctx: CommandContext, _intent: Intent) -> CommandOutcome:
        return CommandOutcome(message=ctx.desktop.lock_screen())

    async def sleep(ctx: CommandContext, _intent: Intent) -> CommandOutcome:
        return CommandOutcome(message=ctx.desktop.sleep())

    async def restart(ctx: CommandContext, _intent: Intent) -> CommandOutcome:
        return CommandOutcome(message=ctx.desktop.restart(settings.shutdown_delay_seconds))

    async def shutdown(ctx: CommandContext, _intent: Intent) -> CommandOutcome:
        return CommandOutcome(message=ctx.desktop.shutdown(settings.shutdown_delay_seconds))

    async def set_volume(ctx: CommandContext, intent: Intent) -> CommandOutcome:
        return CommandOutcome(message=ctx.desktop.set_volume(_coerce_level(_target(intent))))

    async def set_brightness(ctx: CommandContext, intent: Intent) -> CommandOutcome:
        return CommandOutcome(message=ctx.desktop.set_brightness(_coerce_level(_target(intent))))

    async def screenshot(ctx: CommandContext, _intent: Intent) -> CommandOutcome:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        destination = settings.screenshot_dir / f"quainex-{stamp}.png"
        return CommandOutcome(
            message=ctx.desktop.screenshot(destination), data={"path": str(destination)}
        )

    async def clipboard(ctx: CommandContext, intent: Intent) -> CommandOutcome:
        parameters = intent.parameters_as_dict()
        action = parameters.get("action", "read").lower()
        if action == "write":
            text = parameters.get("text") or _target(intent)
            return CommandOutcome(message=ctx.desktop.write_clipboard(text))
        contents = ctx.desktop.read_clipboard()
        return CommandOutcome(message=contents, data={"text": contents})

    async def notify(ctx: CommandContext, intent: Intent) -> CommandOutcome:
        return CommandOutcome(message=ctx.desktop.notify(_target(intent), title="Quainex"))

    async def system_info(ctx: CommandContext, _intent: Intent) -> CommandOutcome:
        snapshot = ctx.desktop.system_info()
        message = (
            f"CPU {snapshot.cpu_percent:.0f}%, "
            f"memory {snapshot.memory_percent:.0f}%, "
            f"disk {snapshot.disk_percent:.0f}%"
        )
        if snapshot.battery_percent is not None:
            message += f", battery {snapshot.battery_percent:.0f}%"
        return CommandOutcome(message=message, data=snapshot.model_dump())

    # --- developer assistant (Phase 7) ----------------------------------

    async def run_dev_command(ctx: CommandContext, intent: Intent) -> CommandOutcome:
        _require(ctx.dev, "Development tooling")
        assert ctx.dev is not None  # noqa: S101 - narrowed by _require
        parameters = intent.parameters_as_dict()
        result = ctx.dev.run(
            _target(intent),
            directory=parameters.get("directory"),
            message=parameters.get("message"),
        )
        headline = "succeeded" if result.succeeded else f"exited with {result.exit_code}"
        return CommandOutcome(
            message=f"{result.operation} {headline}.\n\n{result.output}",
            data=result.model_dump(),
        )

    async def explain_code(ctx: CommandContext, intent: Intent) -> CommandOutcome:
        _require(ctx.code, "Code assistance")
        assert ctx.code is not None  # noqa: S101
        return CommandOutcome(message=await ctx.code.explain(_target(intent)))

    async def review_code(ctx: CommandContext, intent: Intent) -> CommandOutcome:
        _require(ctx.code, "Code assistance")
        assert ctx.code is not None  # noqa: S101
        review = await ctx.code.review(_target(intent))
        counts = f"{len(review.findings)} finding(s)"
        return CommandOutcome(message=f"{review.verdict} ({counts})", data=review.model_dump())

    async def generate_code(ctx: CommandContext, intent: Intent) -> CommandOutcome:
        _require(ctx.code, "Code assistance")
        assert ctx.code is not None  # noqa: S101
        return CommandOutcome(message=await ctx.code.generate(_target(intent)))

    # --- vision (Phase 8) -----------------------------------------------

    async def look_at_screen(ctx: CommandContext, intent: Intent) -> CommandOutcome:
        _require(ctx.vision, "Vision")
        assert ctx.vision is not None  # noqa: S101
        question = _target(intent) or "What is on the screen right now?"
        return CommandOutcome(message=await ctx.vision.look_at_screen(question))

    async def read_document(ctx: CommandContext, intent: Intent) -> CommandOutcome:
        _require(ctx.vision, "Vision")
        assert ctx.vision is not None  # noqa: S101
        parameters = intent.parameters_as_dict()
        question = parameters.get("question") or "Summarise this document."
        return CommandOutcome(message=await ctx.vision.read_document(_target(intent), question))

    async def list_windows(ctx: CommandContext, _intent: Intent) -> CommandOutcome:
        _require(ctx.vision, "Vision")
        assert ctx.vision is not None  # noqa: S101
        windows = ctx.vision.list_windows()
        titles = ", ".join(window.title for window in windows[:10]) or "none"
        return CommandOutcome(
            message=f"{len(windows)} window(s) open: {titles}",
            data={"windows": [window.model_dump() for window in windows]},
        )

    return [
        Command(
            intent=IntentType.OPEN_APPLICATION,
            summary="Launch an allowlisted application.",
            handler=open_application,
            requires_target=True,
        ),
        Command(
            intent=IntentType.CLOSE_APPLICATION,
            summary="Terminate a running allowlisted application.",
            handler=close_application,
            requires_target=True,
        ),
        Command(
            intent=IntentType.OPEN_WEBSITE,
            summary="Open an http(s) URL in the default browser.",
            handler=open_website,
            requires_target=True,
        ),
        Command(
            intent=IntentType.OPEN_FOLDER,
            summary="Reveal a folder inside the permitted roots.",
            handler=open_folder,
            requires_target=True,
        ),
        Command(
            intent=IntentType.SEARCH_FILES,
            summary="Find files by name inside the permitted roots.",
            handler=search_files,
            requires_target=True,
        ),
        Command(
            intent=IntentType.LOCK_SCREEN,
            summary="Lock the workstation.",
            handler=lock_screen,
        ),
        Command(
            intent=IntentType.SLEEP,
            summary="Put the machine to sleep.",
            handler=sleep,
            destructive=True,
        ),
        Command(
            intent=IntentType.RESTART,
            summary="Restart the machine after a grace period.",
            handler=restart,
            destructive=True,
        ),
        Command(
            intent=IntentType.SHUTDOWN,
            summary="Shut the machine down after a grace period.",
            handler=shutdown,
            destructive=True,
        ),
        Command(
            intent=IntentType.SET_VOLUME,
            summary="Raise, lower or mute the system volume.",
            handler=set_volume,
            requires_target=True,
        ),
        Command(
            intent=IntentType.SET_BRIGHTNESS,
            summary="Raise, lower or set display brightness.",
            handler=set_brightness,
            requires_target=True,
        ),
        Command(
            intent=IntentType.SCREENSHOT,
            summary="Capture the screen to a PNG file.",
            handler=screenshot,
        ),
        Command(
            intent=IntentType.CLIPBOARD,
            summary="Read or write the clipboard.",
            handler=clipboard,
        ),
        Command(
            intent=IntentType.NOTIFY,
            summary="Show a desktop notification.",
            handler=notify,
            requires_target=True,
        ),
        Command(
            intent=IntentType.SYSTEM_INFO,
            summary="Report CPU, memory, disk and battery.",
            handler=system_info,
        ),
        Command(
            intent=IntentType.RUN_DEV_COMMAND,
            summary="Run an allowlisted development command (git, tests, lint, docker).",
            handler=run_dev_command,
            requires_target=True,
        ),
        Command(
            intent=IntentType.EXPLAIN_CODE,
            summary="Explain what a source file does.",
            handler=explain_code,
            requires_target=True,
        ),
        Command(
            intent=IntentType.REVIEW_CODE,
            summary="Review a source file for defects.",
            handler=review_code,
            requires_target=True,
        ),
        Command(
            intent=IntentType.GENERATE_CODE,
            summary="Write code from a description. Returns text; writes nothing.",
            handler=generate_code,
            requires_target=True,
        ),
        Command(
            intent=IntentType.LOOK_AT_SCREEN,
            summary="Answer a question about what is on screen.",
            handler=look_at_screen,
        ),
        Command(
            intent=IntentType.READ_DOCUMENT,
            summary="Answer a question about a PDF.",
            handler=read_document,
            requires_target=True,
        ),
        Command(
            intent=IntentType.LIST_WINDOWS,
            summary="List open windows. Local; no model call.",
            handler=list_windows,
        ),
    ]


def _coerce_level(target: str) -> LevelChange:
    """Interpret a level target as a direction or a numeric level.

    Rejecting unrecognised text here — rather than passing it down and letting
    the controller decide — means an unusable value never reaches an OS call,
    and the refusal is identical across platforms.

    Args:
        target: Raw target text, e.g. ``"up"``, ``"40"`` or ``"40%"``.

    Returns:
        A direction, or an integer level between 0 and 100.

    Raises:
        CommandNotAllowedError: The target is neither a direction nor a level.
    """
    cleaned = target.strip().lower().removesuffix("%").strip()

    if cleaned.isdigit():
        level = int(cleaned)
        if not 0 <= level <= 100:
            raise CommandNotAllowedError(f"'{target}' is outside the range 0-100.")
        return level

    # Compared one at a time so the return type narrows to the literal.
    if cleaned == "up":
        return "up"
    if cleaned == "down":
        return "down"
    if cleaned in {"mute", "muted", "silence"}:
        return "mute"

    raise CommandNotAllowedError(
        f"Cannot interpret '{target}' as a level; use 'up', 'down', 'mute' or 0-100."
    )
