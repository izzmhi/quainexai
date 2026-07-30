"""A controllable web browser, driven from Telegram.

Purpose:
    Let you say "browse to a site", "scroll down", "click the login link", and see
    a screenshot of the page after each step — a browser you steer from your phone,
    which is the "search and navigate and show me" the request asked for.

Why Playwright over the installed Edge, not a bundled Chromium:
    Playwright normally downloads its own ~150 MB Chromium, which this machine's
    unreliable connection would not survive. Pointed at the ``msedge`` channel it
    drives the Edge that is already installed — nothing to download — and Edge is
    Chromium underneath, so the automation is identical.

Why the *sync* API in a dedicated thread, not the async one:
    Playwright launches a helper subprocess, and on Windows that needs the Proactor
    event loop. Uvicorn does not always run on it, so the async API failed inside
    the server with a bare ``NotImplementedError`` — while working perfectly from a
    standalone script. Rather than depend on the server's loop policy, every browser
    call runs on a single dedicated worker thread that owns a sync Playwright: that
    thread has no asyncio loop to conflict with, and being single-worker it keeps
    Playwright's thread-affine objects on the one thread they were created on. The
    async methods here just marshal to it and await the result.

Why one persistent page:
    Navigation is stateful. "Scroll down" only means something relative to where
    "click search" left you, so a single browser and page live for the session.

Headless, and screenshots as the interface:
    No visible window — it should not fight the user's desktop for focus — and each
    action returns a screenshot of the *viewport* (not the full page), precisely
    because scrolling is a command a full-page capture would render invisible.

What it does not do:
    Fill password fields or persist a login. This is for looking things up, not for
    automating an authenticated account — a different feature, a different threat
    model.

Dependencies:
    playwright (the "browser" extra), plus an installed Edge or Chrome

Future improvements:
    * Click by numbered overlay ("click 3") for pages where text is ambiguous.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, TypeVar
from urllib.parse import quote_plus, urlparse

from quainex.core.exceptions import CommandExecutionError
from quainex.core.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from quainex.config.settings import Settings

_log = get_logger(__name__)

_T = TypeVar("_T")

#: Browser channels to try, in order. Edge first (every modern Windows has it),
#: Chrome as a fallback.
_CHANNELS = ("msedge", "chrome")

#: Viewport size. 720p-ish: readable in a Telegram photo without a huge upload.
_VIEWPORT = {"width": 1280, "height": 800}

#: How far one "scroll" moves, in pixels — a bit less than a screen, so context
#: carries over between steps.
_SCROLL_STEP = 650

#: Per-action timeout (ms). A page that will not load in fifteen seconds is
#: reported, not waited on forever.
_ACTION_TIMEOUT_MS = 15000


class BrowserSession:
    """A single, persistent, phone-steerable browser page."""

    def __init__(self, settings: Settings) -> None:
        """Construct the session without launching anything.

        The browser starts on first use, so constructing this never opens Edge — a
        session that never browses pays nothing.

        Args:
            settings: Application configuration.
        """
        self._settings = settings
        # One worker: Playwright objects are thread-affine, so every call must land
        # on the same thread. A single-worker pool guarantees exactly that, and
        # serialises actions for free.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="quainex-browser")
        self._lock = asyncio.Lock()
        # These live on the worker thread; the main thread only ever hands work to
        # it, never touches them directly.
        self._playwright: Any = None
        self._browser: Any = None
        self._page: Any = None

    @staticmethod
    def is_available() -> bool:
        """Whether browser control can run.

        Returns:
            Whether Playwright imports. A missing browser binary surfaces at launch
            time with a clear message, not here.
        """
        import importlib.util

        return importlib.util.find_spec("playwright") is not None

    @property
    def is_open(self) -> bool:
        """Whether a page is currently live."""
        return self._page is not None

    async def open(self, target: str) -> tuple[str, str]:
        """Navigate to a URL, or search the web for a phrase.

        A target that parses as a domain or URL is opened directly; anything else
        becomes a search — "browse github.com" goes straight there, "browse best
        laptops 2026" searches.

        Args:
            target: A URL, bare domain, or search phrase.

        Returns:
            The resulting ``(url, title)``.

        Raises:
            CommandExecutionError: The page could not be loaded.
        """
        url = _to_url(target)
        return await self._run(lambda: self._sync_open(url))

    async def scroll(self, direction: str) -> tuple[str, str]:
        """Scroll the page.

        Args:
            direction: ``up``, ``down``, ``top`` or ``bottom``.

        Returns:
            The current ``(url, title)``.
        """
        return await self._run(lambda: self._sync_scroll(direction))

    async def click(self, text: str) -> tuple[str, str]:
        """Click the first link or button whose visible text matches.

        Args:
            text: The visible text to click.

        Returns:
            The ``(url, title)`` after the click.

        Raises:
            CommandExecutionError: Nothing matched, or no page is open.
        """
        return await self._run(lambda: self._sync_click(text))

    async def type_text(self, text: str) -> tuple[str, str]:
        """Type into the focused field, or the first text box, and press Enter.

        Args:
            text: The text to type.

        Returns:
            The ``(url, title)`` after typing.

        Raises:
            CommandExecutionError: There is nowhere to type, or no page is open.
        """
        return await self._run(lambda: self._sync_type(text))

    async def back(self) -> tuple[str, str]:
        """Go back one page in history.

        Returns:
            The ``(url, title)`` after going back.
        """
        return await self._run(self._sync_back)

    async def screenshot(self, destination: Path) -> Path:
        """Capture the current viewport to a PNG.

        Args:
            destination: Where to write the image.

        Returns:
            ``destination``.

        Raises:
            CommandExecutionError: No page is open.
        """
        return await self._run(lambda: self._sync_screenshot(destination))

    async def close(self) -> None:
        """Close the browser and release everything. Idempotent."""
        async with self._lock:
            if self._browser is not None or self._playwright is not None:
                await asyncio.get_running_loop().run_in_executor(self._executor, self._sync_close)
        self._executor.shutdown(wait=False)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="quainex-browser")

    # -- marshalling ------------------------------------------------------

    async def _run(self, work: Callable[[], _T]) -> _T:
        """Run a synchronous browser operation on the worker thread.

        Args:
            work: The operation to run.

        Returns:
            Its result.
        """
        async with self._lock:
            return await asyncio.get_running_loop().run_in_executor(self._executor, work)

    # -- synchronous operations (worker thread only) ----------------------

    def _sync_ensure_page(self) -> Any:
        """Launch the browser on first use and return the live page.

        Runs on the worker thread, where a sync Playwright is safe to start.

        Returns:
            The Playwright page.

        Raises:
            CommandExecutionError: No usable browser is installed.
        """
        if self._page is not None:
            return self._page

        if not self.is_available():
            raise CommandExecutionError(
                'Browser control is not installed. Run: pip install -e ".[browser]".'
            )

        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()
        last_error = "no channel"
        for channel in _CHANNELS:
            try:
                self._browser = self._playwright.chromium.launch(channel=channel, headless=True)
                break
            except Exception as exc:
                last_error = _brief(exc)
        if self._browser is None:
            self._sync_close()
            raise CommandExecutionError(
                f"Could not start a browser (tried Edge and Chrome): {last_error}. "
                f"Install Microsoft Edge, or run: playwright install chromium."
            )

        context = self._browser.new_context(viewport=_VIEWPORT)
        self._page = context.new_page()
        _log.info("browser_launched")
        return self._page

    def _sync_require_page(self) -> Any:
        """Return the live page or explain that none is open.

        Returns:
            The current page.

        Raises:
            CommandExecutionError: No page is open.
        """
        if self._page is None:
            raise CommandExecutionError("No page is open. Say 'browse <site>' first.")
        return self._page

    def _sync_open(self, url: str) -> tuple[str, str]:
        page = self._sync_ensure_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=_ACTION_TIMEOUT_MS)
        except Exception as exc:
            raise CommandExecutionError(f"Could not open {url}: {_brief(exc)}") from exc
        return self._sync_where()

    def _sync_scroll(self, direction: str) -> tuple[str, str]:
        page = self._sync_require_page()
        delta = {"down": _SCROLL_STEP, "up": -_SCROLL_STEP, "bottom": 1_000_000, "top": -1_000_000}
        page.mouse.wheel(0, delta.get(direction, _SCROLL_STEP))
        page.wait_for_timeout(400)
        return self._sync_where()

    def _sync_click(self, text: str) -> tuple[str, str]:
        page = self._sync_require_page()
        for locator in (
            page.get_by_role("link", name=text, exact=False),
            page.get_by_role("button", name=text, exact=False),
            page.get_by_text(text, exact=False),
        ):
            try:
                locator.first.click(timeout=3000)
            except Exception:  # noqa: S112 - not here; try the next locator
                continue
            page.wait_for_timeout(800)
            return self._sync_where()
        raise CommandExecutionError(f"Nothing clickable matching '{text}' on this page.")

    def _sync_type(self, text: str) -> tuple[str, str]:
        page = self._sync_require_page()
        box = page.get_by_role("searchbox").or_(
            page.locator("input[type=text], input:not([type]), textarea")
        )
        try:
            box.first.fill(text, timeout=3000)
            box.first.press("Enter")
        except Exception as exc:
            raise CommandExecutionError(f"Could not type here: {_brief(exc)}") from exc
        page.wait_for_timeout(800)
        return self._sync_where()

    def _sync_back(self) -> tuple[str, str]:
        page = self._sync_require_page()
        page.go_back(wait_until="domcontentloaded", timeout=_ACTION_TIMEOUT_MS)
        return self._sync_where()

    def _sync_screenshot(self, destination: Path) -> Path:
        page = self._sync_require_page()
        destination.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(destination))
        return destination

    def _sync_where(self) -> tuple[str, str]:
        page = self._page
        if page is None:
            return "", ""
        try:
            return page.url, page.title()
        except Exception:
            return page.url, ""

    def _sync_close(self) -> None:
        for closer, method in ((self._browser, "close"), (self._playwright, "stop")):
            if closer is None:
                continue
            try:
                getattr(closer, method)()
            except Exception as exc:
                _log.warning("browser_close_error", error=_brief(exc))
        self._page = None
        self._browser = None
        self._playwright = None


def _to_url(target: str) -> str:
    """Turn a target into a URL — direct for a domain, a search otherwise.

    Args:
        target: A URL, bare domain, or search phrase.

    Returns:
        An ``http(s)`` URL.
    """
    cleaned = target.strip()
    parsed = urlparse(cleaned)
    if parsed.scheme in {"http", "https"}:
        return cleaned
    if " " not in cleaned and "." in cleaned and not cleaned.startswith("."):
        return f"https://{cleaned}"
    return f"https://duckduckgo.com/?q={quote_plus(cleaned)}"


def _brief(exc: Exception) -> str:
    """Shorten a Playwright exception to one readable line.

    Args:
        exc: The exception.

    Returns:
        A short description.
    """
    return str(exc).splitlines()[0][:160]
