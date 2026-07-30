"""A phone-steerable web browser.

Playwright driving the installed Edge, kept alive across commands so navigation is
stateful. Optional — without the browser extra, ``BrowserSession.is_available()``
reports false and the rest of the system is unaffected.
"""

from quainex.core.browser.session import BrowserSession

__all__ = ["BrowserSession"]
