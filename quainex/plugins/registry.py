"""Plugin discovery, loading and lifecycle.

Purpose:
    Find plugins on disk, load the ones the user enabled, and route requests to
    them.

Why plugins are opt-in per plugin:
    Discovery and loading are separate steps. A plugin dropped into the folder is
    *found* — its manifest and requested permissions are readable — but it is not
    *loaded* until enabled. That ordering means the user sees what a plugin wants
    before any of its code runs, which is the only moment the permission list is
    worth anything.

Architecture:
    plugins/<name>/plugin.toml   manifest, read without executing anything
    plugins/<name>/__init__.py   defines `plugin` (a Plugin instance)

    discover()  -> read manifests only            no code executed
    enable(n)   -> import module, build context   code runs from here
    invoke(...) -> plugin.handle(ctx, request)

Dependencies:
    quainex.plugins.{manifest,context}, tomllib

Future improvements:
    * Install from a URL or registry, with a signature check.
    * Subprocess isolation, making the permission boundary enforced rather than
      cooperative.
"""

from __future__ import annotations

import importlib.util
import sys
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, Field

from quainex.core.exceptions import PluginError
from quainex.core.logging import get_logger
from quainex.plugins.context import PluginContext
from quainex.plugins.manifest import PluginManifest

if TYPE_CHECKING:
    from quainex.config.settings import Settings
    from quainex.core.automation.desktop import DesktopController
    from quainex.core.commands import CommandExecutor
    from quainex.core.memory import MemoryManager
    from quainex.services.ai.provider import AIProvider

_log = get_logger(__name__)

MANIFEST_FILENAME = "plugin.toml"


class PluginRequest(BaseModel):
    """A call into a plugin.

    Attributes:
        action: Which of the plugin's declared actions to run.
        arguments: Action arguments.
    """

    action: str
    arguments: dict[str, str] = Field(default_factory=dict)


class PluginResponse(BaseModel):
    """What a plugin returned.

    Attributes:
        message: Human-readable result.
        data: Optional structured payload.
    """

    message: str
    data: dict[str, Any] | None = None


class Plugin(Protocol):
    """What a plugin module must provide."""

    async def handle(self, ctx: PluginContext, request: PluginRequest) -> PluginResponse:
        """Handle one request."""
        ...


class DiscoveredPlugin(BaseModel):
    """A plugin found on disk, whether or not it is loaded.

    Attributes:
        manifest: Its declared metadata and permissions.
        path: Where it lives.
        enabled: Whether its code has been loaded.
        error: Why it failed to load, when it did.
    """

    manifest: PluginManifest
    path: str
    enabled: bool = False
    error: str | None = None


class PluginRegistry:
    """Finds, loads and invokes plugins."""

    def __init__(
        self,
        settings: Settings,
        *,
        commands: CommandExecutor | None = None,
        provider: AIProvider | None = None,
        memory: MemoryManager | None = None,
        desktop: DesktopController | None = None,
    ) -> None:
        """Construct the registry.

        Args:
            settings: Configuration supplying the plugin directory.
            commands: Command executor granted to plugins with that permission.
            provider: Model backend granted to plugins with that permission.
            memory: Memory manager granted to plugins with that permission.
            desktop: Desktop controller granted to plugins with that permission.
        """
        self._settings = settings
        self._commands = commands
        self._provider = provider
        self._memory = memory
        self._desktop = desktop
        self._discovered: dict[str, DiscoveredPlugin] = {}
        self._loaded: dict[str, tuple[Plugin, PluginContext]] = {}

    @property
    def directory(self) -> Path:
        """Where plugins are looked for."""
        return self._settings.plugin_dir

    def discover(self) -> list[DiscoveredPlugin]:
        """Read every plugin manifest without executing any plugin code.

        Returns:
            Every plugin found, in name order.
        """
        self._discovered.clear()
        root = self.directory
        if not root.is_dir():
            return []

        for entry in sorted(root.iterdir()):
            manifest_path = entry / MANIFEST_FILENAME
            if not entry.is_dir() or not manifest_path.is_file():
                continue
            try:
                manifest = self._read_manifest(manifest_path)
            except PluginError as exc:
                _log.warning("plugin_manifest_invalid", path=str(entry), reason=exc.message)
                continue

            self._discovered[manifest.name] = DiscoveredPlugin(
                manifest=manifest,
                path=str(entry),
                enabled=manifest.name in self._loaded,
            )

        _log.info("plugins_discovered", count=len(self._discovered))
        return list(self._discovered.values())

    def enable(self, name: str) -> DiscoveredPlugin:
        """Load a plugin's code and build its capability context.

        This is the point at which plugin code first runs, which is why it is a
        separate, explicit step from discovery.

        Args:
            name: The plugin to enable.

        Returns:
            Its updated record.

        Raises:
            PluginError: Not found, or it failed to load.
        """
        if not self._discovered:
            self.discover()

        record = self._discovered.get(name)
        if record is None:
            raise PluginError(f"No plugin named '{name}' was found in {self.directory}.")

        try:
            plugin = self._load_module(Path(record.path), record.manifest.name)
        except PluginError as exc:
            record.error = exc.message
            record.enabled = False
            raise

        context = PluginContext(
            record.manifest,
            settings=self._settings,
            data_dir=self.directory / record.manifest.name / "data",
            commands=self._commands,
            provider=self._provider,
            memory=self._memory,
            desktop=self._desktop,
        )
        self._loaded[name] = (plugin, context)
        record.enabled = True
        record.error = None

        _log.info(
            "plugin_enabled",
            plugin=name,
            permissions=[p.value for p in record.manifest.permissions],
        )
        return record

    def disable(self, name: str) -> bool:
        """Unload a plugin.

        The module object stays in ``sys.modules`` — Python cannot truly unload
        code — but the plugin no longer receives requests and its context is
        dropped. A full unload needs the process isolation noted as future work.

        Args:
            name: The plugin to disable.

        Returns:
            Whether it had been enabled.
        """
        removed = self._loaded.pop(name, None) is not None
        if removed:
            if record := self._discovered.get(name):
                record.enabled = False
            _log.info("plugin_disabled", plugin=name)
        return removed

    async def invoke(self, name: str, request: PluginRequest) -> PluginResponse:
        """Send a request to an enabled plugin.

        Args:
            name: Which plugin.
            request: The action and its arguments.

        Returns:
            The plugin's response.

        Raises:
            PluginError: Not enabled, unknown action, or the plugin raised.
        """
        entry = self._loaded.get(name)
        if entry is None:
            raise PluginError(f"Plugin '{name}' is not enabled.")

        plugin, context = entry
        record = self._discovered[name]
        if request.action not in record.manifest.actions:
            available = ", ".join(record.manifest.actions) or "none"
            raise PluginError(
                f"'{request.action}' is not an action of '{name}'. Available: {available}."
            )

        try:
            return await plugin.handle(context, request)
        except PluginError:
            raise
        except Exception as exc:
            # A plugin is third-party code; its failure must not take down the
            # process or leak a traceback to the caller.
            _log.exception("plugin_failed", plugin=name, action=request.action)
            raise PluginError(f"Plugin '{name}' failed handling '{request.action}': {exc}") from exc

    def enabled_names(self) -> list[str]:
        """List currently enabled plugins.

        Returns:
            Plugin names.
        """
        return sorted(self._loaded)

    # -- internals --------------------------------------------------------

    @staticmethod
    def _read_manifest(path: Path) -> PluginManifest:
        """Parse and validate a manifest file.

        Args:
            path: The ``plugin.toml`` to read.

        Returns:
            The validated manifest.

        Raises:
            PluginError: The file is unreadable or invalid.
        """
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise PluginError(f"Could not read manifest '{path}': {exc}") from exc

        # Strip a UTF-8 BOM before parsing. tomllib rejects one outright, and on
        # Windows a BOM is what you get from Notepad and from PowerShell's
        # `Set-Content -Encoding utf8` — including, on first attempt, from this
        # project's own example manifest. Refusing a file the user's default
        # editor produces would be a poor first experience for plugin authors.
        if payload.startswith(b"\xef\xbb\xbf"):
            payload = payload[3:]

        try:
            raw = tomllib.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise PluginError(f"Could not read manifest '{path}': {exc}") from exc

        try:
            return PluginManifest.model_validate(raw.get("plugin", raw))
        except ValueError as exc:
            raise PluginError(f"Manifest '{path}' is not valid: {exc}") from exc

    @staticmethod
    def _load_module(directory: Path, name: str) -> Plugin:
        """Import a plugin package and return its ``plugin`` object.

        Args:
            directory: The plugin's directory.
            name: Its manifest name.

        Returns:
            The plugin instance.

        Raises:
            PluginError: The module is missing, will not import, or does not
                expose a usable ``plugin``.
        """
        entry = directory / "__init__.py"
        if not entry.is_file():
            raise PluginError(f"Plugin '{name}' has no __init__.py.")

        module_name = f"quainex_plugin_{name}"
        spec = importlib.util.spec_from_file_location(module_name, entry)
        if spec is None or spec.loader is None:
            raise PluginError(f"Plugin '{name}' could not be prepared for import.")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop(module_name, None)
            raise PluginError(f"Plugin '{name}' failed to import: {exc}") from exc

        plugin = getattr(module, "plugin", None)
        if plugin is None or not hasattr(plugin, "handle"):
            raise PluginError(
                f"Plugin '{name}' must define a module-level `plugin` object with a "
                "`handle(ctx, request)` coroutine."
            )
        return plugin  # type: ignore[no-any-return]
