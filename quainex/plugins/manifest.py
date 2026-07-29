"""Plugin manifests and the permission vocabulary.

Purpose:
    Describe what a plugin is and what it is allowed to touch.

**What the permission system is, and what it is not.** This is the most important
paragraph in the plugin system, so it is stated plainly rather than buried:

    A Python plugin runs in this process. It can ``import os``. Nothing in this
    module — or anywhere else in Quainex — prevents that. **This is not a
    security sandbox, and calling it one would be a lie.** Real isolation needs a
    separate process with OS-level restrictions, or a different language for
    plugins entirely; both are noted as future work.

    What this *is* is a capability system. A plugin receives a ``PluginContext``
    carrying only the capabilities its manifest declares and the user approved.
    That defends against the failure that actually happens in practice: a plugin
    that means well, and quietly grows past what you thought you installed. A
    Spotify plugin that declares ``network`` and later starts reading files gets
    an error, not silence — and the manifest it shipped with is the thing you
    agreed to.

    Trust the code you install. The permissions tell you what to look for.

Architecture:
    plugin.toml manifest -> PluginManifest (validated)
        -> user approves permissions
        -> PluginContext built with only those capabilities
        -> plugin.handle(ctx, request)

Dependencies:
    pydantic

Future improvements:
    * Run plugins in a subprocess with a restricted token, making the permission
      boundary real rather than cooperative.
    * Signed plugins, so a manifest cannot be edited after review.
"""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

#: Plugin names are used in URLs and on disk, so the character set is narrow.
_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{1,48}$")


class Permission(StrEnum):
    """A capability a plugin may request.

    Each maps to a method group on ``PluginContext``. A plugin that did not
    declare a permission does not receive the corresponding capability, and
    calling it raises rather than silently doing nothing — a silent no-op would
    make a misconfigured plugin look like a broken one.
    """

    #: Read files inside the permitted roots.
    FILES_READ = "files.read"
    #: Write files inside a directory private to the plugin.
    FILES_WRITE = "files.write"
    #: Make outbound HTTP requests.
    NETWORK = "network"
    #: Run allowlisted Quainex commands (subject to every Phase 3 gate).
    COMMANDS = "commands"
    #: Read and write plugin-scoped memory.
    MEMORY = "memory"
    #: Show desktop notifications.
    NOTIFY = "notify"
    #: Ask the language model.
    AI = "ai"


#: Permissions that let a plugin affect the world outside Quainex. Requesting one
#: of these is worth showing prominently when a user is deciding whether to
#: install something.
SENSITIVE_PERMISSIONS: frozenset[Permission] = frozenset(
    {Permission.COMMANDS, Permission.NETWORK, Permission.FILES_READ}
)


class PluginManifest(BaseModel):
    """What a plugin declares about itself.

    Attributes:
        name: Unique identifier, lower-case, used in URLs and paths.
        version: Semantic-ish version string.
        description: One line explaining what the plugin does.
        author: Who wrote it.
        permissions: Capabilities the plugin requires.
        actions: Named actions the plugin exposes, mapped to descriptions.
        homepage: Optional link for the user to check before installing.
    """

    name: str
    version: str = "0.1.0"
    description: str = ""
    author: str = "unknown"
    permissions: list[Permission] = Field(default_factory=list)
    actions: dict[str, str] = Field(default_factory=dict)
    homepage: str | None = None

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        """Ensure the name is safe for use in a URL and a directory name.

        Args:
            value: The proposed name.

        Returns:
            The validated name.

        Raises:
            ValueError: The name would be unsafe or ambiguous.
        """
        if not _NAME_PATTERN.match(value):
            raise ValueError(
                f"'{value}' is not a valid plugin name. Use 2-49 lower-case "
                "letters, digits, hyphens or underscores, starting with a letter."
            )
        return value

    @property
    def sensitive_permissions(self) -> list[Permission]:
        """Permissions worth highlighting before installation."""
        return [p for p in self.permissions if p in SENSITIVE_PERMISSIONS]

    def grants(self, permission: Permission) -> bool:
        """Whether this manifest requests a permission.

        Args:
            permission: The capability to check.

        Returns:
            Whether it was declared.
        """
        return permission in self.permissions
