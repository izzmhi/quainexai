"""Desktop automation primitives.

Phase 3. Application control, navigation, power, media and utilities, behind a
platform-neutral ``DesktopController`` contract.
"""

from quainex.core.automation.desktop import (
    DesktopController,
    FileHit,
    LevelChange,
    SystemSnapshot,
)
from quainex.core.automation.windows import WindowsDesktopController

__all__ = [
    "DesktopController",
    "FileHit",
    "LevelChange",
    "SystemSnapshot",
    "WindowsDesktopController",
]
