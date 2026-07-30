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


# -- developer commands ----------------------------------------------------


@pytest.mark.parametrize(
    ("said", "operation"),
    [
        ("git status", "git.status"),
        ("git st", "git.status"),
        ("git log", "git.log"),
        ("git diff", "git.diff"),
        ("run the tests", "tests.run"),
        ("tests", "tests.run"),
        ("pytest", "tests.run"),
        ("run the linter", "lint.run"),
        ("lint", "lint.run"),
        ("check types", "types.check"),
        ("mypy", "types.check"),
        ("format check", "format.check"),
        ("docker ps", "docker.ps"),
    ],
)
def test_read_only_dev_commands_need_no_model(said: str, operation: str):
    """The executor takes exact keys, so these map straight through."""
    result = classify_locally(said)

    assert result is not None
    assert result.intent is IntentType.RUN_DEV_COMMAND
    assert result.target == operation


@pytest.mark.parametrize(
    "said",
    ["git commit", "git push", "git pull", "git add", "commit my changes"],
)
def test_dev_commands_that_change_something_reach_the_model(said: str):
    """Only the ones that *report* are local.

    ``push`` and ``pull`` touch a remote, ``add`` and ``commit`` change a
    repository, and a pattern cannot judge whether now is the moment. ``commit``
    could not be local anyway — it needs a message extracted from the sentence,
    which is exactly the work a model is for.
    """
    assert classify_locally(said) is None


def test_run_no_longer_swallows_a_dev_command_as_an_application():
    """The misclassification that measuring exposed.

    "run" was one of the generic application verbs, so "run the tests" came back as
    ``open_application`` targeting "the tests" — a confidently wrong answer where
    falling through would have been right. The allowlist would have refused it, but
    the classification was still nonsense.
    """
    result = classify_locally("run the tests")

    assert result is not None
    assert result.intent is IntentType.RUN_DEV_COMMAND


# -- folders, search and the screen ----------------------------------------


@pytest.mark.parametrize(
    ("said", "target"),
    [
        ("open documents", "documents"),
        ("open my downloads folder", "downloads"),
        ("open the desktop folder", "desktop"),
        ("go to pictures", "pictures"),
    ],
)
def test_named_folders_are_not_mistaken_for_applications(said: str, target: str):
    """A folder name and a program name are indistinguishable to a pattern.

    Only the name distinguishes them, so the folder rule is limited to names that
    are unambiguously directories.
    """
    result = classify_locally(said)

    assert result is not None
    assert result.intent is IntentType.OPEN_FOLDER
    assert result.target == target


@pytest.mark.parametrize("said", ["open notepad", "open chrome", "launch spotify"])
def test_applications_are_still_applications(said: str):
    """The folder rule must not claim these."""
    result = classify_locally(said)

    assert result is not None
    assert result.intent is IntentType.OPEN_APPLICATION


@pytest.mark.parametrize(
    ("said", "target"),
    [
        ("find my invoice pdf", "invoice pdf"),
        ("search for budget", "budget"),
        ("look for the meeting notes", "meeting notes"),
    ],
)
def test_file_search_needs_no_model(said: str, target: str):
    result = classify_locally(said)

    assert result is not None
    assert result.intent is IntentType.SEARCH_FILES
    assert result.target == target


@pytest.mark.parametrize(
    "said",
    [
        "what's on my screen",
        "what is on the screen",
        "read the error on screen",
        "what am i looking at",
    ],
)
def test_screen_questions_are_classified_locally(said: str):
    """Answering still needs a vision model; *classifying* does not.

    So the classification tokens are saved on every one of these, which is most of
    the cost of a short question.
    """
    result = classify_locally(said)

    assert result is not None
    assert result.intent is IntentType.LOOK_AT_SCREEN
    # The whole question is the target: the analyst needs the words, not a label.
    assert result.target


# -- webcam, wifi, web search ----------------------------------------------


@pytest.mark.parametrize(
    "said",
    [
        "take a webcam picture",
        "webcam",
        "take a selfie",
        "who is there",
        "show me the camera",
        "snap a photo with the webcam",
    ],
)
def test_webcam_requests_are_local(said: str):
    """A camera photo needs no model to classify, and is distinct from a screenshot."""
    result = classify_locally(said)

    assert result is not None
    assert result.intent is IntentType.WEBCAM


def test_a_screenshot_is_not_mistaken_for_a_webcam_photo():
    """A photo means the camera; a screenshot means the screen."""
    assert classify_locally("take a screenshot").intent is IntentType.SCREENSHOT
    assert classify_locally("take a webcam photo").intent is IntentType.WEBCAM


@pytest.mark.parametrize(
    ("said", "target"),
    [
        ("turn on wifi", "on"),
        ("turn off wifi", "off"),
        ("wifi on", "on"),
        ("wi-fi off", "off"),
        ("wifi status", "status"),
        ("is wifi on", "status"),
        ("what's my wifi", "status"),
    ],
)
def test_wifi_control_is_local(said: str, target: str):
    result = classify_locally(said)

    assert result is not None
    assert result.intent is IntentType.WIFI
    assert result.target == target


@pytest.mark.parametrize(
    ("said", "target"),
    [
        ("google python enumerate", "python enumerate"),
        ("search the web for weather in lagos", "weather in lagos"),
        ("look up bitcoin price online", "bitcoin price"),
        ("search the web for the best pizza places near me", "the best pizza places near me"),
    ],
)
def test_web_search_is_local(said: str, target: str):
    """The browser open costs no tokens.

    Only a factual summary might, and that uses a keyless free API, not the model.
    """
    result = classify_locally(said)

    assert result is not None
    assert result.intent is IntentType.WEB_SEARCH
    assert result.target == target


def test_a_plain_search_is_still_a_file_search():
    """Only an explicit web marker routes to the web; "search for X" means files."""
    result = classify_locally("search for budget")

    assert result is not None
    assert result.intent is IntentType.SEARCH_FILES


# -- folders and files -----------------------------------------------------


@pytest.mark.parametrize(
    "said",
    ["open desktop", "open downloads", "open documents", "open my pictures folder", "go to music"],
)
def test_known_folders_open_locally(said: str):
    """The classification is free; the controller resolves the real path.

    A folder word goes through the known-folder API, so OneDrive redirection is
    handled downstream — but deciding it is a folder costs no tokens.
    """
    result = classify_locally(said)

    assert result is not None
    assert result.intent is IntentType.OPEN_FOLDER


@pytest.mark.parametrize(
    ("said", "name"),
    [
        ("create a folder called projects", "projects"),
        ("make a new folder reports", "reports"),
        ("create tax folder", "tax"),
        ("make a folder named invoices", "invoices"),
    ],
)
def test_create_folder_is_local(said: str, name: str):
    result = classify_locally(said)

    assert result is not None
    assert result.intent is IntentType.CREATE_FOLDER
    assert result.target == name


@pytest.mark.parametrize(
    ("said", "target"),
    [
        ("send me report.pdf", "report.pdf"),
        ("send me the file budget.xlsx", "budget.xlsx"),
        ("send me my latest download", "latest download"),
        ("send me latest", "latest"),
    ],
)
def test_send_file_is_local(said: str, target: str):
    result = classify_locally(said)

    assert result is not None
    assert result.intent is IntentType.SEND_FILE
    assert result.target == target


# -- the steerable browser -------------------------------------------------


@pytest.mark.parametrize(
    ("said", "intent", "target"),
    [
        ("browse github.com", IntentType.BROWSE, "github.com"),
        ("browse best laptops 2026", IntentType.BROWSE, "best laptops 2026"),
        ("open youtube.com in the browser", IntentType.BROWSE, "youtube.com"),
        ("scroll down", IntentType.BROWSER_SCROLL, "down"),
        ("scroll to bottom", IntentType.BROWSER_SCROLL, "bottom"),
        ("click the login button", IntentType.BROWSER_CLICK, "login button"),
        ("type hello world", IntentType.BROWSER_TYPE, "hello world"),
    ],
)
def test_browser_control_is_local(said: str, intent: IntentType, target: str):
    """Steering the browser costs no tokens.

    Only the page content does, and that is fetched by the browser, not a model.
    """
    result = classify_locally(said)

    assert result is not None
    assert result.intent is intent
    assert result.target == target


def test_browser_back_and_close_take_no_target():
    assert classify_locally("go back").intent is IntentType.BROWSER_BACK
    assert classify_locally("close the browser").intent is IntentType.BROWSER_CLOSE


def test_the_controlled_browser_is_distinct_from_a_plain_launch():
    """The browse verb steers the headless browser; open launches real Edge."""
    assert classify_locally("browse github.com").intent is IntentType.BROWSE
    assert classify_locally("open github.com").intent is IntentType.OPEN_WEBSITE


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
