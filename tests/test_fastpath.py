"""Tests for local, token-free classification.

Two halves, and the second matters more. The first checks that common commands
are matched without a model. The second checks everything the fast path must
*refuse* — because a pattern that guesses is worse than no pattern at all: it
produces a confident wrong answer where the fallback would have produced a
correct one.
"""

from __future__ import annotations

import pytest

from quainex.core.brain import IntentType
from quainex.core.brain.fastpath import classify_locally

# -- what it handles -------------------------------------------------------


@pytest.mark.parametrize(
    ("said", "intent"),
    [
        ("take a screenshot", IntentType.SCREENSHOT),
        ("screenshot", IntentType.SCREENSHOT),
        ("grab a screenshot", IntentType.SCREENSHOT),
        ("capture the screen", IntentType.SCREENSHOT),
        ("lock the screen", IntentType.LOCK_SCREEN),
        ("lock my computer", IntentType.LOCK_SCREEN),
        ("lock", IntentType.LOCK_SCREEN),
        ("system info", IntentType.SYSTEM_INFO),
        ("what's my system status", IntentType.SYSTEM_INFO),
        ("battery", IntentType.SYSTEM_INFO),
        ("how's my cpu", IntentType.SYSTEM_INFO),
        ("list windows", IntentType.LIST_WINDOWS),
        ("what's open", IntentType.LIST_WINDOWS),
    ],
)
def test_targetless_commands_need_no_model(said: str, intent: IntentType):
    result = classify_locally(said)

    assert result is not None
    assert result.intent is intent
    assert result.target is None


@pytest.mark.parametrize(
    ("said", "intent", "target"),
    [
        ("open notepad", IntentType.OPEN_APPLICATION, "notepad"),
        ("launch vs code", IntentType.OPEN_APPLICATION, "vs code"),
        ("start visual studio code", IntentType.OPEN_APPLICATION, "visual studio code"),
        ("close spotify", IntentType.CLOSE_APPLICATION, "spotify"),
        ("quit chrome", IntentType.CLOSE_APPLICATION, "chrome"),
    ],
)
def test_targeted_commands_need_no_model(said: str, intent: IntentType, target: str):
    result = classify_locally(said)

    assert result is not None
    assert result.intent is intent
    assert result.target == target


@pytest.mark.parametrize(
    ("said", "target"),
    [
        ("open github.com", "github.com"),
        ("go to youtube.com", "youtube.com"),
        ("visit https://example.com/page", "https://example.com/page"),
    ],
)
def test_a_site_is_not_mistaken_for_an_application(said: str, target: str):
    """Checked before the application patterns, which would otherwise claim it."""
    result = classify_locally(said)

    assert result is not None
    assert result.intent is IntentType.OPEN_WEBSITE
    assert result.target == target


@pytest.mark.parametrize(
    ("said", "target"),
    [
        ("volume up", "up"),
        ("turn the volume down", "down"),
        ("set volume to 50", "50"),
        ("mute", "mute"),
        ("brightness up", "up"),
        ("set brightness to 80", "80"),
    ],
)
def test_level_controls_need_no_model(said: str, target: str):
    result = classify_locally(said)

    assert result is not None
    assert result.target == target


@pytest.mark.parametrize(
    "filler",
    ["hey ", "please ", "can you ", "okay ", "i want to ", "just "],
)
def test_politeness_does_not_defeat_the_match(filler: str):
    """People do not issue commands like a command line."""
    result = classify_locally(f"{filler}take a screenshot")

    assert result is not None
    assert result.intent is IntentType.SCREENSHOT


def test_trailing_punctuation_does_not_defeat_the_match():
    """Speech recognition adds it, so it must not matter."""
    assert classify_locally("Take a screenshot.") is not None
    assert classify_locally("lock the screen!") is not None


def test_a_local_match_is_fully_confident():
    """The pattern either matched or it did not; there is no uncertainty.

    This also keeps the Brain's low-confidence confirmation rule from firing on a
    local match, which would mean asking the user to confirm "take a screenshot".
    """
    result = classify_locally("take a screenshot")

    assert result is not None
    assert result.confidence == 1.0


def test_a_local_match_says_so_in_its_reasoning():
    """So the audit trail distinguishes a free classification from a paid one."""
    result = classify_locally("lock the screen")

    assert result is not None
    assert "no model call" in result.reasoning


# -- what it must refuse ---------------------------------------------------


@pytest.mark.parametrize(
    "said",
    [
        # Destructive: always the model's decision, never a regex's. "restart the
        # router" is not about this computer, and a pattern cannot tell.
        "shut down",
        "shutdown the computer",
        "restart",
        "restart my pc",
        "restart the router",
        "go to sleep",
        "sleep",
    ],
)
def test_destructive_requests_always_reach_the_model(said: str):
    """Confirmation would probably catch a misfire.

    "Probably caught downstream" is not the standard for powering off a machine,
    and the saving on these is a rounding error — they are rare.
    """
    assert classify_locally(said) is None


@pytest.mark.parametrize(
    "said",
    [
        # Structure a pattern cannot see. Each looks like "open X" to a regex and
        # means something else entirely.
        "open a new tab in chrome",
        "open the file i was working on",
        "close everything except spotify",
        "start a timer for ten minutes",
    ],
)
def test_structured_requests_fall_through(said: str):
    assert classify_locally(said) is None


@pytest.mark.parametrize(
    "said",
    [
        "how are you doing",
        "what is the capital of japan",
        "explain what this error means",
        "why is my computer slow today",
        "can you review the file i changed",
    ],
)
def test_conversation_and_questions_fall_through(said: str):
    """These need a model by definition; matching them would be a wrong answer."""
    assert classify_locally(said) is None


def test_a_long_utterance_falls_through():
    """Once someone is explaining themselves, the meaning is in the explanation."""
    assert classify_locally("please open notepad because i need to write something down") is None


@pytest.mark.parametrize("said", ["", "   ", "\n\t"])
def test_empty_input_falls_through_rather_than_matching(said: str):
    assert classify_locally(said) is None


def test_a_bare_verb_with_no_target_falls_through():
    """A bare verb is a request for clarification, not a command."""
    assert classify_locally("open") is None
    assert classify_locally("close") is None
