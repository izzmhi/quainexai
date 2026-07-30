"""Static checks on the dashboard's HTML, CSS and JavaScript.

There is no browser in this suite, so these do not test behaviour. They defend
the three invariants the dashboard's design rests on, each of which has exactly
one failure mode that is invisible until someone is looking at the page:

1. ``hidden`` controls visibility, so no CSS rule may outrank it.
2. ``data-bind`` is the only coupling between the script and the markup, so every
   name the script reaches for must exist.
3. Nothing is loaded from another host, so the page works offline.

A linter would catch none of these. Each one has already broken once, or would
have gone unnoticed if it had.
"""

from __future__ import annotations

import re

import pytest

from quainex.config.settings import REPO_ROOT

DASHBOARD = REPO_ROOT / "dashboard"


@pytest.fixture(scope="module")
def html() -> str:
    """The dashboard document."""
    return (DASHBOARD / "index.html").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def css() -> str:
    """The dashboard stylesheet."""
    return (DASHBOARD / "app.css").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def script() -> str:
    """The dashboard controller."""
    return (DASHBOARD / "app.js").read_text(encoding="utf-8")


def test_the_hidden_attribute_outranks_every_component_rule(css: str):
    """The bug this exists to prevent, stated concretely.

    ``[hidden] { display: none }`` is a *user-agent* rule, so any author rule that
    sets ``display`` beats it. ``.modal-backdrop { display: grid }`` therefore left
    the confirmation dialog visible on page load, and ``element.hidden = true`` in
    app.js was a silent no-op — so its buttons looked dead when in fact closing it
    did nothing.

    Deleting this rule as redundant tidying would bring all of that straight back.
    """
    normalised = re.sub(r"\s+", " ", css)

    assert re.search(r"\[hidden\]\s*\{[^}]*display:\s*none\s*!important", normalised), (
        "app.css must force [hidden] to display: none !important — see the comment "
        "in the reset section for why the browser's own rule is not enough."
    )


def test_every_element_the_script_binds_to_exists_in_the_markup(html: str, script: str):
    """``data-bind`` is the entire contract between app.js and index.html.

    A typo, or a rename on one side only, produces a panel that quietly never
    populates: the script's helper returns ``null`` and every call site tolerates
    that by design, so nothing throws and nothing appears.
    """
    referenced = set(re.findall(r'bind\("([^"]+)"\)', script))
    present = set(re.findall(r'data-bind="([^"]+)"', html))

    assert referenced <= present, (
        f"app.js binds to names absent from index.html: {referenced - present}"
    )


def test_every_action_the_script_wires_up_exists_in_the_markup(html: str, script: str):
    """Same contract, for the buttons that are found by ``data-action``.

    This is the check that would have caught a dead button, had the cause been a
    missing element rather than CSS.
    """
    referenced = set(re.findall(r'\[data-action="([^"]+)"\]', script))
    present = set(re.findall(r'data-action="([^"]+)"', html))

    assert referenced <= present, (
        f"app.js wires actions absent from index.html: {referenced - present}"
    )


def test_the_confirmation_dialog_starts_hidden(html: str):
    """It must be invisible until an action actually needs approving."""
    match = re.search(r'<div class="modal-backdrop"[^>]*>', html)

    assert match is not None
    assert "hidden" in match.group(0)


def test_the_core_starts_in_a_state_the_stylesheet_knows(html: str, css: str):
    """The core's whole appearance derives from one attribute.

    An unstyled initial state would render the centrepiece of the interface as an
    unadorned circle until the first interaction.
    """
    match = re.search(r'data-bind="core"[^>]*data-state="([a-z]+)"', html)

    assert match is not None
    assert f'.core[data-state="{match.group(1)}"]' in css or ".core {" in css


@pytest.mark.parametrize("state", ["idle", "listening", "thinking", "speaking", "error"])
def test_every_state_the_script_can_set_is_a_state_the_script_captions(state: str, script: str):
    """A state with no label would leave the caption stale and wrong."""
    labels = script[script.index("const labels = {") : script.index("const hints = {")]
    hints = script[script.index("const hints = {") :][:600]

    assert f"{state}:" in labels
    assert f"{state}:" in hints


def test_nothing_is_loaded_from_another_host(html: str):
    """The page must work with the network unplugged.

    This is the promise that justifies having no build step at all, so it is worth
    an assertion rather than a line in a README. A single CDN ``<script>`` would
    make a local application depend on someone else's uptime.
    """
    for attribute in re.findall(r'(?:src|href)="([^"]+)"', html):
        assert not attribute.startswith(("http://", "https://", "//")), (
            f"index.html loads {attribute} from another host; the dashboard must be self-contained."
        )


def test_the_document_loads_the_stylesheet_and_the_script(html: str):
    assert 'href="./app.css"' in html
    assert 'src="./app.js"' in html
