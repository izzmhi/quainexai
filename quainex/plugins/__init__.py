"""Plugin system: manifests, capability contexts, discovery and dispatch.

Phase 9. Plugins declare permissions in a manifest; the context hands them only
what they declared. See manifest.py for what that does and does not defend
against.
"""

from quainex.plugins.context import PluginContext
from quainex.plugins.manifest import SENSITIVE_PERMISSIONS, Permission, PluginManifest
from quainex.plugins.registry import (
    DiscoveredPlugin,
    Plugin,
    PluginRegistry,
    PluginRequest,
    PluginResponse,
)

__all__ = [
    "SENSITIVE_PERMISSIONS",
    "DiscoveredPlugin",
    "Permission",
    "Plugin",
    "PluginContext",
    "PluginManifest",
    "PluginRegistry",
    "PluginRequest",
    "PluginResponse",
]
