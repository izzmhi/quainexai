"""Screen and document understanding.

Purpose:
    Let Quainex answer questions about what is on screen, and about documents on
    disk.

Why there is no OCR library here:
    The obvious Phase 8 shopping list is Tesseract plus OpenCV plus template
    matching. That returns *characters*, and the questions people actually ask
    are not about characters — "which button do I press", "what is this error
    telling me", "is the build finished". A vision-capable model answers those
    directly, and reading text is the easy subset it gets for free.

    It also removes a native binary from the install. On a machine where large
    downloads are unreliable, "pip install and it works" is worth a great deal.

    The trade is real and worth stating: this sends a screenshot off the machine,
    and it costs a request. Window enumeration below is local and free, so cheap
    questions ("is VS Code open") never need the model.

Architecture:
    look_at_screen()  -> DesktopController.screenshot() -> temp PNG
                      -> AIProvider.look(image, question)
                      -> answer, screenshot deleted

    read_document()   -> AIProvider.read_document(pdf, question)

    list_windows()    -> ctypes EnumWindows — local, no model, no cost

Dependencies:
    quainex.core.automation, quainex.services.ai

Future improvements:
    * Return click coordinates for a named control, so Phase 10 can act on what
      it sees rather than only describe it.
    * Cache the last screenshot briefly, so several questions about one screen
      cost one request.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from quainex.core.logging import get_logger

if TYPE_CHECKING:
    from quainex.config.settings import Settings
    from quainex.core.automation.desktop import DesktopController
    from quainex.services.ai.provider import AIProvider

_log = get_logger(__name__)

_SCREEN_SYSTEM = """
You are looking at a screenshot of the user's own computer, taken at their
request, to answer their question about it.

Answer the question directly and concretely. Name what you can see — window
titles, button labels, error text — rather than describing the screen in
general. If the answer is not visible, say so instead of guessing.
""".strip()


class WindowInfo(BaseModel):
    """One open, visible window.

    Attributes:
        title: The window title.
        process_id: The owning process id.
    """

    title: str
    process_id: int


class ScreenAnalyst:
    """Answers questions about the screen and about documents."""

    def __init__(
        self,
        provider: AIProvider,
        desktop: DesktopController,
        settings: Settings,
    ) -> None:
        """Construct the analyst.

        Args:
            provider: Vision-capable model backend.
            desktop: Controller used to capture the screen.
            settings: Application configuration.
        """
        self._provider = provider
        self._desktop = desktop
        self._settings = settings

    async def look_at_screen(self, question: str) -> str:
        """Capture the screen and answer a question about it.

        The screenshot is written to a temporary directory and deleted when the
        answer returns. A picture of the user's whole desktop is not something to
        leave lying on disk.

        Args:
            question: What to ask about the screen.

        Returns:
            The answer.
        """
        with tempfile.TemporaryDirectory(prefix="quainex-vision-") as tmp:
            shot = Path(tmp) / "screen.png"
            self._desktop.screenshot(shot)
            _log.info("screen_captured_for_analysis", question_chars=len(question))
            return await self._provider.look(
                image_paths=[shot],
                question=question,
                system=_SCREEN_SYSTEM,
            )

    async def look_at_image(self, image_path: str, question: str) -> str:
        """Answer a question about an image file.

        Args:
            image_path: Image to examine.
            question: What to ask about it.

        Returns:
            The answer.
        """
        resolved = self._resolve(image_path)
        _log.info("image_analysed", path=str(resolved))
        return await self._provider.look(
            image_paths=[resolved], question=question, system=_SCREEN_SYSTEM
        )

    async def read_document(self, document_path: str, question: str) -> str:
        """Answer a question about a PDF.

        Args:
            document_path: The PDF to read.
            question: What to ask about it.

        Returns:
            The answer.
        """
        resolved = self._resolve(document_path)
        _log.info("document_read", path=str(resolved))
        return await self._provider.read_document(
            document_path=resolved,
            question=question,
            system=(
                "Answer the user's question about this document directly, "
                "citing the part of the document you drew the answer from."
            ),
        )

    def list_windows(self) -> list[WindowInfo]:
        """List open, visible, titled windows.

        Local and free — no model call. Answers cheap questions like "is VS Code
        open" without a screenshot or a request.

        Returns:
            Visible windows with titles.
        """
        return _enumerate_windows()

    def _resolve(self, path: str) -> Path:
        """Resolve a path and confirm it is inside a permitted root.

        Args:
            path: Requested file path.

        Returns:
            The resolved path.

        Raises:
            CommandNotAllowedError: The path escapes the permitted roots.
        """
        from quainex.core.exceptions import CommandNotAllowedError

        roots = self._settings.resolved_search_roots
        raw = Path(path.strip()).expanduser()
        resolved = (raw if raw.is_absolute() else roots[0] / raw).resolve() if roots else raw

        if not any(resolved.is_relative_to(root) for root in roots):
            raise CommandNotAllowedError(f"'{resolved}' is outside the folders Quainex may read.")
        return resolved


def _enumerate_windows() -> list[WindowInfo]:
    """Enumerate visible, titled top-level windows.

    Uses the Win32 API directly through ctypes rather than a wrapper package:
    it is roughly thirty lines, and a dependency for thirty lines of stable
    API is a poor trade.

    Returns:
        Visible windows, or an empty list off Windows.
    """
    import ctypes
    from ctypes import wintypes

    try:
        user32 = ctypes.windll.user32
    except AttributeError:  # pragma: no cover - non-Windows platform
        return []

    windows: list[WindowInfo] = []

    # WINFUNCTYPE, not CFUNCTYPE: EnumWindows expects the stdcall convention.
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

    def _collect(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True

        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            # Untitled windows are tooltips, shims and invisible helpers.
            return True

        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)

        process_id = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))

        windows.append(WindowInfo(title=buffer.value, process_id=int(process_id.value)))
        return True

    user32.EnumWindows(callback_type(_collect), 0)
    return windows
