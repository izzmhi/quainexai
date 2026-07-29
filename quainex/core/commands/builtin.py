"""Built-in command implementations.

Purpose:
    Bind each ``IntentType`` to the desktop action that carries it out.

This module is deliberately thin. Every handler does three things: pull what it
needs off the intent, call one ``DesktopController`` method, and describe the
result. All validation, allowlisting and path containment lives in the
controller; all policy lives in the executor. If a handler here grows a branch,
the logic probably belongs in one of those two places instead.

Architecture:
    IntentType -> Command(handler=...) -> DesktopController method

Dependencies:
    quainex.core.automation, quainex.core.brain, quainex.core.commands.base

Future improvements:
    * ``SEARCH_FILES`` should rank hits by recency once memory exists (Phase 5).
    * ``NOTIFY`` should route through the WebSocket as well, so a notification
      reaches the phone client too (Phase 6).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from quainex.core.brain import IntentType
from quainex.core.commands.base import Command, CommandOutcome
from quainex.core.exceptions import CommandNotAllowedError

if TYPE_CHECKING:
    from quainex.config.settings import Settings
    from quainex.core.automation.desktop import DesktopController, LevelChange
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

    def open_application(desktop: DesktopController, intent: Intent) -> CommandOutcome:
        return CommandOutcome(message=desktop.open_application(_target(intent)))

    def close_application(desktop: DesktopController, intent: Intent) -> CommandOutcome:
        return CommandOutcome(message=desktop.close_application(_target(intent)))

    def open_website(desktop: DesktopController, intent: Intent) -> CommandOutcome:
        return CommandOutcome(message=desktop.open_url(_target(intent)))

    def open_folder(desktop: DesktopController, intent: Intent) -> CommandOutcome:
        return CommandOutcome(message=desktop.open_folder(_target(intent)))

    def search_files(desktop: DesktopController, intent: Intent) -> CommandOutcome:
        hits = desktop.search_files(_target(intent), settings.command_search_max_results)
        message = (
            f"Found {len(hits)} file(s) matching '{_target(intent)}'."
            if hits
            else f"No files matched '{_target(intent)}'."
        )
        return CommandOutcome(
            message=message,
            data={"results": [hit.model_dump() for hit in hits]},
        )

    def lock_screen(desktop: DesktopController, _intent: Intent) -> CommandOutcome:
        return CommandOutcome(message=desktop.lock_screen())

    def sleep(desktop: DesktopController, _intent: Intent) -> CommandOutcome:
        return CommandOutcome(message=desktop.sleep())

    def restart(desktop: DesktopController, _intent: Intent) -> CommandOutcome:
        return CommandOutcome(message=desktop.restart(settings.shutdown_delay_seconds))

    def shutdown(desktop: DesktopController, _intent: Intent) -> CommandOutcome:
        return CommandOutcome(message=desktop.shutdown(settings.shutdown_delay_seconds))

    def set_volume(desktop: DesktopController, intent: Intent) -> CommandOutcome:
        return CommandOutcome(message=desktop.set_volume(_coerce_level(_target(intent))))

    def set_brightness(desktop: DesktopController, intent: Intent) -> CommandOutcome:
        return CommandOutcome(message=desktop.set_brightness(_coerce_level(_target(intent))))

    def screenshot(desktop: DesktopController, _intent: Intent) -> CommandOutcome:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        destination = settings.screenshot_dir / f"quainex-{stamp}.png"
        return CommandOutcome(
            message=desktop.screenshot(destination),
            data={"path": str(destination)},
        )

    def clipboard(desktop: DesktopController, intent: Intent) -> CommandOutcome:
        parameters = intent.parameters_as_dict()
        action = parameters.get("action", "read").lower()
        if action == "write":
            text = parameters.get("text") or _target(intent)
            return CommandOutcome(message=desktop.write_clipboard(text))
        contents = desktop.read_clipboard()
        return CommandOutcome(message=contents, data={"text": contents})

    def notify(desktop: DesktopController, intent: Intent) -> CommandOutcome:
        return CommandOutcome(message=desktop.notify(_target(intent), title="Quainex"))

    def system_info(desktop: DesktopController, _intent: Intent) -> CommandOutcome:
        snapshot = desktop.system_info()
        message = (
            f"CPU {snapshot.cpu_percent:.0f}%, "
            f"memory {snapshot.memory_percent:.0f}%, "
            f"disk {snapshot.disk_percent:.0f}%"
        )
        if snapshot.battery_percent is not None:
            message += f", battery {snapshot.battery_percent:.0f}%"
        return CommandOutcome(message=message, data=snapshot.model_dump())

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
