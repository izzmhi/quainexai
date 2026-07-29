"""Tests for the plugin system.

The important tests here are the ones about *when code runs* and *what a plugin
can reach*: discovery must not execute anything, and a capability not declared in
the manifest must not be usable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from quainex.config.settings import Settings
from quainex.core.exceptions import PluginError, PluginPermissionError
from quainex.plugins import (
    Permission,
    PluginContext,
    PluginManifest,
    PluginRegistry,
    PluginRequest,
)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        log_dir=tmp_path / "logs",
        database_path=tmp_path / "t.db",
        command_search_roots=[tmp_path],
        plugin_dir=tmp_path / "plugins",
    )


def _write_plugin(
    root: Path,
    name: str,
    permissions: str = "",
    body: str | None = None,
    actions: str = 'ping = "Return pong."',
) -> Path:
    """Create a plugin on disk and return its directory."""
    directory = root / "plugins" / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "plugin.toml").write_text(
        f'[plugin]\nname = "{name}"\npermissions = [{permissions}]\n\n'
        f"[plugin.actions]\n{actions}\n",
        encoding="utf-8",
    )
    (directory / "__init__.py").write_text(
        body
        or (
            "from quainex.plugins import PluginResponse\n"
            "class P:\n"
            "    async def handle(self, ctx, request):\n"
            "        return PluginResponse(message='pong')\n"
            "plugin = P()\n"
        ),
        encoding="utf-8",
    )
    return directory


# -- manifests -------------------------------------------------------------


@pytest.mark.parametrize("name", ["spotify", "my-plugin", "a_b_c", "github2"])
def test_valid_names_are_accepted(name):
    assert PluginManifest(name=name).name == name


@pytest.mark.parametrize(
    "name", ["", "A", "1plugin", "has space", "../escape", "x" * 60, "plugin/../.."]
)
def test_unsafe_names_are_rejected(name):
    # The name becomes a URL segment and a directory name, so traversal and
    # whitespace are refused rather than sanitised.
    with pytest.raises(ValueError, match="not a valid plugin name"):
        PluginManifest(name=name)


def test_sensitive_permissions_are_singled_out():
    manifest = PluginManifest(name="thing", permissions=[Permission.NETWORK, Permission.MEMORY])
    assert Permission.NETWORK in manifest.sensitive_permissions
    assert Permission.MEMORY not in manifest.sensitive_permissions


# -- discovery does not execute plugin code -------------------------------


def test_discovery_reads_manifests_without_importing(tmp_path):
    # The plugin's module raises on import. Discovery must still succeed,
    # because it is exactly the moment the user has not yet agreed to run it.
    _write_plugin(tmp_path, "landmine", body="raise RuntimeError('should not run')\n")

    found = PluginRegistry(_settings(tmp_path)).discover()

    assert [p.manifest.name for p in found] == ["landmine"]
    assert found[0].enabled is False


def test_enabling_a_broken_plugin_reports_rather_than_crashes(tmp_path):
    _write_plugin(tmp_path, "landmine", body="raise RuntimeError('boom')\n")
    registry = PluginRegistry(_settings(tmp_path))

    with pytest.raises(PluginError, match="failed to import"):
        registry.enable("landmine")


def test_plugin_without_a_plugin_object_is_rejected(tmp_path):
    _write_plugin(tmp_path, "empty", body="x = 1\n")
    with pytest.raises(PluginError, match="must define a module-level"):
        PluginRegistry(_settings(tmp_path)).enable("empty")


def test_invalid_manifests_are_skipped_not_fatal(tmp_path):
    directory = tmp_path / "plugins" / "broken"
    directory.mkdir(parents=True)
    (directory / "plugin.toml").write_text("this is not toml [[[", encoding="utf-8")
    _write_plugin(tmp_path, "good")

    found = PluginRegistry(_settings(tmp_path)).discover()

    # One bad plugin must not stop the others being usable.
    assert [p.manifest.name for p in found] == ["good"]


def test_missing_plugin_directory_is_not_an_error(tmp_path):
    assert PluginRegistry(_settings(tmp_path)).discover() == []


def test_a_manifest_with_a_utf8_bom_still_parses(tmp_path):
    # Notepad and PowerShell's `Set-Content -Encoding utf8` both write a BOM,
    # which tomllib rejects outright. This project's own example manifest was
    # written that way on the first attempt, so plugin authors will hit it too.
    _write_plugin(tmp_path, "bommed")
    manifest = tmp_path / "plugins" / "bommed" / "plugin.toml"
    manifest.write_bytes(b"\xef\xbb\xbf" + manifest.read_bytes())

    found = PluginRegistry(_settings(tmp_path)).discover()
    assert [p.manifest.name for p in found] == ["bommed"]


# -- lifecycle -------------------------------------------------------------


async def test_enable_invoke_disable(tmp_path):
    _write_plugin(tmp_path, "pinger")
    registry = PluginRegistry(_settings(tmp_path))

    registry.enable("pinger")
    assert registry.enabled_names() == ["pinger"]

    response = await registry.invoke("pinger", PluginRequest(action="ping"))
    assert response.message == "pong"

    assert registry.disable("pinger") is True
    assert registry.enabled_names() == []


async def test_invoking_a_disabled_plugin_is_refused(tmp_path):
    _write_plugin(tmp_path, "pinger")
    registry = PluginRegistry(_settings(tmp_path))
    registry.discover()

    with pytest.raises(PluginError, match="is not enabled"):
        await registry.invoke("pinger", PluginRequest(action="ping"))


async def test_undeclared_actions_are_refused(tmp_path):
    _write_plugin(tmp_path, "pinger")
    registry = PluginRegistry(_settings(tmp_path))
    registry.enable("pinger")

    with pytest.raises(PluginError, match="is not an action"):
        await registry.invoke("pinger", PluginRequest(action="launch_missiles"))


async def test_a_plugin_that_raises_does_not_take_down_the_process(tmp_path):
    _write_plugin(
        tmp_path,
        "crasher",
        body=(
            "class P:\n"
            "    async def handle(self, ctx, request):\n"
            "        raise ValueError('plugin bug')\n"
            "plugin = P()\n"
        ),
    )
    registry = PluginRegistry(_settings(tmp_path))
    registry.enable("crasher")

    with pytest.raises(PluginError, match="failed handling"):
        await registry.invoke("crasher", PluginRequest(action="ping"))


def test_enabling_an_unknown_plugin_is_refused(tmp_path):
    with pytest.raises(PluginError, match="No plugin named"):
        PluginRegistry(_settings(tmp_path)).enable("ghost")


# -- capability gating -----------------------------------------------------


def _context(tmp_path: Path, *permissions: Permission) -> PluginContext:
    return PluginContext(
        PluginManifest(name="tester", permissions=list(permissions)),
        settings=_settings(tmp_path),
        data_dir=tmp_path / "plugin-data",
    )


async def test_undeclared_capabilities_raise(tmp_path):
    # Raising rather than no-op: a missing permission is a configuration
    # problem, and should not look like a bug in the plugin's own logic.
    context = _context(tmp_path)

    with pytest.raises(PluginPermissionError, match="did not declare"):
        await context.remember("k", "v")
    with pytest.raises(PluginPermissionError, match="did not declare"):
        await context.ask("anything")
    with pytest.raises(PluginPermissionError, match="did not declare"):
        context.read_file(str(tmp_path / "x.txt"))
    with pytest.raises(PluginPermissionError, match="did not declare"):
        context.write_file("x.txt", "data")


def test_declared_write_permission_works(tmp_path):
    context = _context(tmp_path, Permission.FILES_WRITE)
    written = context.write_file("notes.txt", "hello")

    assert written.read_text(encoding="utf-8") == "hello"
    assert written.parent == tmp_path / "plugin-data"


@pytest.mark.parametrize("filename", ["../escape.txt", "..\\escape.txt", "/etc/passwd", ".."])
def test_writes_cannot_escape_the_plugin_directory(tmp_path, filename):
    # The filename is reduced to its basename, so a path cannot walk out.
    context = _context(tmp_path, Permission.FILES_WRITE)
    try:
        written = context.write_file(filename, "data")
    except PluginPermissionError:
        return  # refused outright is also correct
    assert written.parent == tmp_path / "plugin-data"


def test_reads_are_confined_to_permitted_roots(tmp_path):
    context = _context(tmp_path, Permission.FILES_READ)
    with pytest.raises(PluginPermissionError, match="may not read outside"):
        context.read_file("C:\\Windows\\System32\\drivers\\etc\\hosts")


def test_reads_inside_the_roots_work(tmp_path):
    target = tmp_path / "note.txt"
    target.write_text("visible", encoding="utf-8")

    assert _context(tmp_path, Permission.FILES_READ).read_file(str(target)) == "visible"


def test_granted_lists_declared_permissions(tmp_path):
    context = _context(tmp_path, Permission.MEMORY, Permission.AI)
    assert set(context.granted()) == {"memory", "ai"}


# -- the shipped example plugin -------------------------------------------


def test_the_example_plugin_manifest_is_valid():
    from quainex.config.settings import REPO_ROOT

    example = REPO_ROOT / "plugins_installed" / "hello"
    if not (example / "plugin.toml").is_file():
        pytest.skip("example plugin not present")

    registry = PluginRegistry(Settings(_env_file=None, plugin_dir=example.parent))  # type: ignore[call-arg]
    names = [p.manifest.name for p in registry.discover()]
    assert "hello" in names
