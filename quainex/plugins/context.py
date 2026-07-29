"""The capability object handed to a plugin.

Purpose:
    Give a plugin exactly the abilities its manifest declared, and nothing else.

How the gating works:
    Every method checks the manifest before delegating. A plugin that did not
    declare ``network`` and calls ``ctx.fetch(...)`` gets
    ``PluginPermissionError`` naming the permission it is missing — not a silent
    failure, which would look like a bug in the plugin rather than a
    configuration problem.

    The context is also where scoping happens. A plugin's memory is namespaced to
    it, and its writable directory is its own. Two plugins cannot see each
    other's state by accident, and neither can reach the user's files by
    accident.

    "By accident" is doing real work in that sentence — see ``manifest.py`` for
    what this system does and does not defend against.

Architecture:
    PluginContext(manifest, ...)
        |-- run_command()  -> CommandExecutor (all Phase 3 gates still apply)
        |-- ask()          -> AIProvider
        |-- remember()     -> MemoryManager, namespaced "plugin:<name>:<key>"
        |-- notify()       -> DesktopController
        |-- read_file()    -> permitted roots only
        +-- write_file()   -> the plugin's own directory only

Dependencies:
    quainex.core.{brain,commands,memory}, quainex.plugins.manifest

Future improvements:
    * Per-plugin rate limits, so one misbehaving plugin cannot exhaust the API
      budget for everything else.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from quainex.core.exceptions import PluginPermissionError
from quainex.core.logging import get_logger
from quainex.plugins.manifest import Permission
from quainex.services.ai.provider import ChatMessage

if TYPE_CHECKING:
    from quainex.config.settings import Settings
    from quainex.core.automation.desktop import DesktopController
    from quainex.core.brain import Intent
    from quainex.core.commands import CommandExecutor, CommandResult
    from quainex.core.memory import MemoryManager
    from quainex.plugins.manifest import PluginManifest
    from quainex.services.ai.provider import AIProvider

_log = get_logger(__name__)

#: Largest file a plugin may read or write in one call.
MAX_PLUGIN_FILE_BYTES = 2 * 1024 * 1024


class PluginContext:
    """The capabilities granted to one plugin."""

    def __init__(
        self,
        manifest: PluginManifest,
        *,
        settings: Settings,
        data_dir: Path,
        commands: CommandExecutor | None = None,
        provider: AIProvider | None = None,
        memory: MemoryManager | None = None,
        desktop: DesktopController | None = None,
    ) -> None:
        """Construct the context.

        Args:
            manifest: The plugin's declared permissions.
            settings: Application configuration.
            data_dir: Directory this plugin may write to.
            commands: Command executor, for the ``commands`` permission.
            provider: Model backend, for the ``ai`` permission.
            memory: Memory manager, for the ``memory`` permission.
            desktop: Desktop controller, for the ``notify`` permission.
        """
        self._manifest = manifest
        self._settings = settings
        self._data_dir = data_dir
        self._commands = commands
        self._provider = provider
        self._memory = memory
        self._desktop = desktop

    @property
    def name(self) -> str:
        """The plugin's name."""
        return self._manifest.name

    @property
    def data_dir(self) -> Path:
        """The directory this plugin may write to."""
        self._data_dir.mkdir(parents=True, exist_ok=True)
        return self._data_dir

    # -- capability gate ---------------------------------------------------

    def _require(self, permission: Permission, component: object | None = None) -> None:
        """Check a permission before granting a capability.

        Args:
            permission: The capability being used.
            component: The collaborator needed, when one is.

        Raises:
            PluginPermissionError: The permission was not declared, or the
                capability is unavailable on this instance.
        """
        if not self._manifest.grants(permission):
            _log.warning(
                "plugin_permission_denied",
                plugin=self._manifest.name,
                permission=permission.value,
            )
            raise PluginPermissionError(
                f"Plugin '{self._manifest.name}' did not declare the "
                f"'{permission.value}' permission."
            )
        if component is None and permission is not Permission.FILES_WRITE:
            raise PluginPermissionError(f"'{permission.value}' is not available on this instance.")

    # -- capabilities ------------------------------------------------------

    async def run_command(self, intent: Intent) -> CommandResult:
        """Execute a Quainex command.

        Every Phase 3 gate still applies — a plugin cannot confirm on the user's
        behalf, and destructive actions remain blocked unless the operator
        enabled them. A plugin is a caller, not a privileged one.

        Args:
            intent: The intent to execute.

        Returns:
            The outcome.
        """
        self._require(Permission.COMMANDS, self._commands)
        assert self._commands is not None  # noqa: S101 - narrowed by _require
        return await self._commands.execute(intent)

    async def ask(self, question: str, system: str | None = None) -> str:
        """Ask the language model a question.

        Args:
            question: What to ask.
            system: Optional system prompt.

        Returns:
            The model's answer.
        """
        self._require(Permission.AI, self._provider)
        assert self._provider is not None  # noqa: S101
        return await self._provider.complete(
            messages=[ChatMessage(role="user", content=question)], system=system
        )

    async def remember(self, key: str, value: str) -> None:
        """Store a value in plugin-scoped memory.

        Args:
            key: Identifier within this plugin's namespace.
            value: What to store.
        """
        self._require(Permission.MEMORY, self._memory)
        assert self._memory is not None  # noqa: S101
        # Namespaced by plugin, so two plugins using the key "token" do not
        # collide and cannot read each other's state.
        await self._memory.remember(f"plugin:{self._manifest.name}", key, value)

    async def recall(self, key: str) -> str | None:
        """Read a value from plugin-scoped memory.

        Args:
            key: Identifier within this plugin's namespace.

        Returns:
            The stored value, or ``None``.
        """
        self._require(Permission.MEMORY, self._memory)
        assert self._memory is not None  # noqa: S101
        return await self._memory.recall(f"plugin:{self._manifest.name}", key)

    def notify(self, message: str) -> str:
        """Show a desktop notification.

        Args:
            message: What to show.

        Returns:
            Confirmation text.
        """
        self._require(Permission.NOTIFY, self._desktop)
        assert self._desktop is not None  # noqa: S101
        return self._desktop.notify(message, title=f"Quainex · {self._manifest.name}")

    def read_file(self, path: str) -> str:
        """Read a file from inside the permitted roots.

        Args:
            path: File to read.

        Returns:
            The file contents.

        Raises:
            PluginPermissionError: Not permitted, or outside the roots.
        """
        self._require(Permission.FILES_READ, self._settings)
        resolved = Path(path).expanduser().resolve()
        roots = self._settings.resolved_search_roots

        if not any(resolved.is_relative_to(root) for root in roots):
            raise PluginPermissionError(
                f"Plugin '{self._manifest.name}' may not read outside the permitted folders."
            )
        if not resolved.is_file():
            raise PluginPermissionError(f"No file at '{resolved}'.")
        if resolved.stat().st_size > MAX_PLUGIN_FILE_BYTES:
            raise PluginPermissionError(f"'{resolved.name}' is too large to read.")

        return resolved.read_text(encoding="utf-8", errors="replace")

    def write_file(self, filename: str, content: str) -> Path:
        """Write a file into this plugin's own directory.

        The filename is reduced to its basename, so a plugin cannot write outside
        its directory by supplying a path — with or without meaning to.

        Args:
            filename: Name of the file. Directory components are discarded.
            content: What to write.

        Returns:
            The path written.

        Raises:
            PluginPermissionError: Not permitted, or the content is too large.
        """
        self._require(Permission.FILES_WRITE)
        if len(content.encode("utf-8")) > MAX_PLUGIN_FILE_BYTES:
            raise PluginPermissionError("The content is too large to write.")

        safe_name = Path(filename).name
        if not safe_name or safe_name in {".", ".."}:
            raise PluginPermissionError(f"'{filename}' is not a usable file name.")

        destination = self.data_dir / safe_name
        destination.write_text(content, encoding="utf-8")
        _log.info("plugin_wrote_file", plugin=self._manifest.name, file=safe_name)
        return destination

    def granted(self) -> list[str]:
        """List the permissions this plugin holds.

        Returns:
            Permission values.
        """
        return [permission.value for permission in self._manifest.permissions]

    def __repr__(self) -> str:
        """Return a debug representation naming the plugin and its permissions."""
        return f"<PluginContext {self._manifest.name} permissions={self.granted()}>"
