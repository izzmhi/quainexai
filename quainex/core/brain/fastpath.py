"""Deterministic classification for commands that need no model.

Purpose:
    Answer the most common requests without spending a single API token.

Why this is the highest-value optimisation available:
    Every model-backed classification pays roughly 1,650 tokens of prompt before
    the user's words are counted — the intent catalogue, the output schema, the
    instructions. On a free tier metered per day that is the difference between a
    few dozen commands and a few hundred.

    And most of what people actually say to a machine assistant is formulaic.
    "take a screenshot", "lock the screen", "open notepad", "volume up" are not
    ambiguous, do not need reasoning, and do not benefit from a 120-billion
    parameter model. Sending them to one costs money, adds a second of latency,
    and can still get them wrong.

    So they are matched here, locally, in microseconds and for nothing.

What is deliberately *not* here:
    **Nothing destructive.** ``SHUTDOWN``, ``RESTART`` and ``SLEEP`` always go to
    the model, even though they are confirmation-gated and a pattern would match
    them easily. A regex has no idea that "restart the router" is not about this
    computer; a model does. The confirmation gate would probably catch the mistake,
    but "probably caught downstream" is not the standard for powering off a
    machine. Cheap and safe beat cheap.

    Anything long or hedged also falls through. A formulaic request is short; once
    someone is explaining themselves, the meaning is in the explanation.

Why patterns and not a small local model:
    A local model is another dependency, another download, and another thing that
    can be subtly wrong. These patterns are auditable: you can read them and know
    exactly what they will and will not match, which is not true of a classifier.
    The fallback to a real model is right there for everything else.

Architecture:
    Brain.interpret(utterance)
        -> classify_locally(utterance)      <-- this module, 0 tokens
             |-- matched   -> IntentClassification (confidence 1.0)
             +-- no match  -> None -> provider.parse()  (the expensive path)

Dependencies:
    re, quainex.core.brain.schemas

Example:
    >>> classify_locally("take a screenshot").intent
    <IntentType.SCREENSHOT: 'screenshot'>
    >>> classify_locally("what do you think about the weather") is None
    True

Future improvements:
    * Learn from the model's answers: when it classifies the same phrasing the
      same way repeatedly, that phrasing could earn a pattern.
    * Per-user aliases ("open my editor") once memory resolution lands.
"""

from __future__ import annotations

import re

from quainex.core.brain.schemas import IntentClassification, IntentParameter, IntentType

#: Longest utterance the fast path will consider.
#:
#: A formulaic command is short — but a *search query* legitimately is not:
#: "search the web for the best pizza near me" is eleven words and entirely
#: unambiguous. So the cap is generous, and the guard against a rambling sentence
#: being force-matched lives where it belongs instead: ``_target_is_simple``
#: refuses a greedy "open X" whose target runs long or contains a preposition, and
#: the anchored search/dev/wifi patterns only match their own explicit prefixes.
_MAX_WORDS = 12

#: Words that signal a captured target is not a plain name.
#:
#: "open notepad" is a command. "open a new tab in chrome" looks identical to a
#: regex and means something quite different, and the difference lives entirely in
#: the preposition. When one of these appears in a captured target, the utterance
#: goes to the model.
#:
#: Three kinds, all found the same way — by a test asserting a phrase must *not*
#: match, and failing:
#:   prepositions   join a name to a context the pattern cannot use
#:   exclusions     "close everything except spotify" is a set operation
#:   quantifiers    "everything" names no single thing
_STRUCTURE_WORDS = frozenset(
    {
        # prepositions and conjunctions
        "in", "on", "at", "for", "with", "to", "from", "about", "and", "then", "of",
        # exclusions
        "except", "but", "besides", "without", "unless", "apart",
        # quantifiers
        "everything", "anything", "something", "all", "every", "each", "any",
    }
)  # fmt: skip

#: Filler that can be stripped without changing meaning.
_LEADING_FILLER = re.compile(
    r"^(?:hey |ok |okay |please |can you |could you |would you |i want to |i need to |just )+",
    re.IGNORECASE,
)

#: Targetless commands: the phrasing *is* the whole request.
#:
#: Ordered longest-first within each intent so that a specific phrase is not
#: shadowed by a looser one.
_EXACT_PATTERNS: tuple[tuple[re.Pattern[str], IntentType], ...] = (
    (re.compile(r"^(?:take |grab |capture )?(?:a )?screen ?shot$"), IntentType.SCREENSHOT),
    (re.compile(r"^capture (?:the )?screen$"), IntentType.SCREENSHOT),
    (re.compile(r"^lock (?:the |my )?(?:screen|computer|pc|workstation)$"), IntentType.LOCK_SCREEN),
    (re.compile(r"^lock$"), IntentType.LOCK_SCREEN),
    (
        re.compile(r"^(?:what(?:'s| is) my )?system (?:info|information|status)$"),
        IntentType.SYSTEM_INFO,
    ),
    (
        re.compile(r"^(?:how(?:'s| is) my )?(?:cpu|memory|ram|battery|disk)(?: usage)?$"),
        IntentType.SYSTEM_INFO,
    ),
    (
        re.compile(r"^(?:list |show |what(?:'s| is) )?(?:the )?(?:open )?windows$"),
        IntentType.LIST_WINDOWS,
    ),
    (re.compile(r"^what(?:'s| is) open$"), IntentType.LIST_WINDOWS),
    # Webcam. Kept distinct from screenshot: "photo/picture" means the camera,
    # "screenshot" means the screen. "who is there" is the anti-theft phrasing.
    (
        re.compile(
            r"^(?:take |grab |capture |snap )?(?:a |the )?"
            r"(?:webcam|web cam|camera|selfie)(?: (?:photo|picture|pic|shot|image|snap))?$"
        ),
        IntentType.WEBCAM,
    ),
    (
        re.compile(
            r"^(?:take|grab|snap) (?:a |the )?(?:photo|picture|pic|selfie)"
            r"(?: (?:with|from|on|using) (?:the |my )?(?:webcam|web cam|camera))?$"
        ),
        IntentType.WEBCAM,
    ),
    (re.compile(r"^who(?:'s| is)(?: (?:there|in front of (?:me|the camera)))$"), IntentType.WEBCAM),
    (re.compile(r"^show me (?:the )?(?:webcam|camera)$"), IntentType.WEBCAM),
)

#: Read-only development commands, mapped to the executor's exact keys.
#:
#: Only the ones that *report*. ``git.add``, ``git.commit``, ``git.push`` and
#: ``git.pull`` are deliberately absent: they change a repository or a remote, and
#: the same rule that keeps ``shutdown`` off the fast path applies — a pattern
#: cannot judge whether now is the moment to push. They still work, via the model.
#:
#: ``git.commit`` could not be here anyway; it needs a message extracted from the
#: sentence, which is exactly the work a model is for.
_DEV_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^git (?:status|st)$"), "git.status"),
    (re.compile(r"^git log$"), "git.log"),
    (re.compile(r"^git diff(?: staged)?$"), "git.diff"),
    (re.compile(r"^git branch(?:es)?$"), "git.branch"),
    (re.compile(r"^(?:run |run the )?(?:tests|test suite|pytest)$"), "tests.run"),
    (re.compile(r"^run the tests$"), "tests.run"),
    (re.compile(r"^(?:run (?:the )?)?(?:lint|linter|ruff)$"), "lint.run"),
    (re.compile(r"^(?:check |run )?(?:the )?(?:types|type check|typecheck|mypy)$"), "types.check"),
    (re.compile(r"^(?:check |run )?format(?: check)?$"), "format.check"),
    (re.compile(r"^docker ps$"), "docker.ps"),
    (re.compile(r"^docker images$"), "docker.images"),
)

#: Wi-Fi, with the direction or "status" as the target.
_WIFI_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^(?:turn |switch )?wi[\s-]?fi (?P<t>on|off)$"), ""),
    (re.compile(r"^(?:turn |switch )(?P<t>on|off) (?:the )?wi[\s-]?fi$"), ""),
    (re.compile(r"^(?:what(?:'s| is) (?:my )?)?wi[\s-]?fi(?: status)?$"), "status"),
    (re.compile(r"^is wi[\s-]?fi (?:on|connected)$"), "status"),
)

#: Web search. The target is the query. Distinguished from file search by an
#: explicit web marker — "google", "search the web", "look up … online" — so plain
#: "search for X" still means files.
_WEB_SEARCH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(?:google|bing) (?P<target>.+)$"),
    re.compile(
        r"^(?:web search|search the (?:web|internet)|search online) (?:for )?(?P<target>.+)$"
    ),
    re.compile(r"^search google (?:for )?(?P<target>.+)$"),
    re.compile(r"^look up (?P<target>.+?) (?:online|on the web|on google)$"),
)

#: Folders people name rather than describe. Matched before applications, because
#: "open documents" is a directory and "open notepad" is a program, and only the
#: name distinguishes them. Aliases included, since the controller accepts them and
#: the fast path should not be pickier than what it calls.
_KNOWN_FOLDERS = frozenset(
    {
        "downloads", "download", "documents", "document", "docs", "desktop",
        "pictures", "picture", "photos", "images", "music", "videos", "video",
        "movies", "home",
    }
)  # fmt: skip

#: "create a folder called X", "make a new folder X in downloads".
_CREATE_FOLDER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^(?:create|make|new|add) (?:a |an )?(?:new )?(?:folder|directory) (?P<target>.+)$"
    ),
    re.compile(r"^(?:create|make) (?P<target>.+?) (?:folder|directory)$"),
)

#: Leading noise a folder name picks up: "called reports", "named X in downloads".
_CREATE_FOLDER_NOISE = re.compile(r"^(?:called |named |titled )")

#: "send me report.pdf", "send me my latest download".
_SEND_FILE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^send (?:me )?(?:my |the )?(?P<target>latest(?: download| file)?|last download)$"),
    re.compile(r"^send (?:me )?(?:the file |file )?(?P<target>[\w .()\-]+\.\w{1,8})$"),
    re.compile(r"^(?:send|share|upload) (?:me )?(?:the )?file (?P<target>.+)$"),
)

#: Explicitly-named folders: "open the downloads folder", "open my documents".
_FOLDER_PATTERN = re.compile(
    r"^(?:open|show|go to) (?:my |the )?(?P<target>[\w\s-]+?)(?: folder| directory)?$"
)

#: File search. Distinct from opening, and the target is a query rather than a name.
_SEARCH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(?:find|search for|look for) (?:my |the )?(?P<target>.+)$"),
    re.compile(r"^(?:find|search) files? (?:called |named |matching )?(?P<target>.+)$"),
)

#: Questions about what is currently on screen. The *classification* is local even
#: though answering still needs a vision model — which saves the classification
#: tokens on every one of these.
_SCREEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^what(?:'s| is)(?: on)? (?:my |the )?screen(?: right now| say(?:ing)?)?$"),
    re.compile(r"^(?:read|what does) (?:the )?(?P<target>.*(?:error|message|dialog|screen).*)$"),
    re.compile(r"^what am i looking at$"),
)

#: Commands whose target is captured from the phrasing.
#:
#: Checked *last*, because these verbs are the greediest in the language. "run" once
#: matched "run the tests" as an application named "the tests" — a confidently wrong
#: answer where falling through to the model would have been right.
_TARGET_PATTERNS: tuple[tuple[re.Pattern[str], IntentType], ...] = (
    (re.compile(r"^(?:open|launch|start) (?P<target>.+)$"), IntentType.OPEN_APPLICATION),
    (re.compile(r"^(?:close|quit|exit|kill) (?P<target>.+)$"), IntentType.CLOSE_APPLICATION),
)

#: Sites and URLs, which take precedence over "open <application>".
_WEBSITE_PATTERN = re.compile(
    r"^(?:open|go to|visit|browse to) (?P<target>(?:https?://\S+)|(?:[\w-]+\.[a-z]{2,}\S*))$",
    re.IGNORECASE,
)

#: Level controls. Kept separate because the target is a direction or a number.
_LEVEL_PATTERNS: tuple[tuple[re.Pattern[str], IntentType], ...] = (
    (
        re.compile(r"^(?:set |turn )?volume (?:to )?(?P<target>up|down|mute|unmute|\d{1,3})$"),
        IntentType.SET_VOLUME,
    ),
    (
        re.compile(r"^(?:turn |set )?(?:the )?volume (?P<target>up|down)$"),
        IntentType.SET_VOLUME,
    ),
    (re.compile(r"^(?P<target>mute|unmute)$"), IntentType.SET_VOLUME),
    (
        re.compile(r"^(?:set |turn )?brightness (?:to )?(?P<target>up|down|\d{1,3})$"),
        IntentType.SET_BRIGHTNESS,
    ),
)

#: Media transport. Play and pause map to a toggle key downstream, but the intent
#: still records which was meant.
_MEDIA_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"^(?:play|resume)(?: (?:the )?(?:music|song|track|audio)| my music| it)?$"),
        "play",
    ),
    (re.compile(r"^(?:play|resume) (?:the )?music on spotify$"), "play"),
    (
        re.compile(r"^pause(?: (?:the )?(?:music|song|track|playback|audio)| it| spotify)?$"),
        "pause",
    ),
    (re.compile(r"^(?:next|skip)(?: (?:track|song))?$"), "next"),
    (re.compile(r"^(?:previous|prev|back)(?: (?:track|song))?$"), "previous"),
    (re.compile(r"^stop(?: (?:the )?(?:music|song|playback|audio))?$"), "stop"),
)

#: Window control. The name is captured; the action rides in parameters so one
#: intent covers all of them.
_WINDOW_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^(?:minimize|minimise) (?:all|everything|all windows)$"), "minimize_all"),
    (re.compile(r"^(?:show|go to) (?:the )?desktop$"), "minimize_all"),
    (re.compile(r"^(?:minimize|minimise) (?P<name>.+)$"), "minimize"),
    (re.compile(r"^(?:maximize|maximise) (?P<name>.+)$"), "maximize"),
    (re.compile(r"^(?:restore|unminimize) (?P<name>.+)$"), "restore"),
)

#: "what's running", "list running apps".
_RUNNING_APPS_PATTERN = re.compile(
    r"^(?:what(?:'s| is) running|what apps are (?:open|running)|running apps|"
    r"list (?:running )?apps|what programs are (?:open|running)|show running apps)$"
)

#: Force-close by name. Distinct from "close X", which asks the app to quit.
_KILL_PATTERN = re.compile(
    r"^(?:kill|force close|force quit|force kill|end) (?:the )?(?P<target>.+)$"
)

#: "where is my laptop", "what's my ip".
_LOCATE_PATTERN = re.compile(
    r"^(?:where(?:'s| is) (?:my |this )?(?:laptop|computer|pc|device)|"
    r"locate (?:my )?(?:laptop|computer|device)|"
    r"what(?:'s|s| is) my (?:public )?ip|public ip|my location|where am i)$"
)

#: Anti-theft trigger.
_PANIC_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(?:panic|panic mode|anti[\s-]?theft)$"),
    re.compile(
        r"^(?:someone (?:took|has|stole|grabbed) (?:my )?(?:laptop|computer|pc)|"
        r"my (?:laptop|computer) (?:was|is|got) stolen)$"
    ),
)

#: Keyboard backlight, direction captured.
_KEYBOARD_LIGHT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(?:turn|switch) (?P<t>on|off) (?:the )?keyboard (?:light|backlight|leds?)$"),
    re.compile(r"^(?:turn|switch) (?:the )?keyboard (?:light|backlight|leds?) (?P<t>on|off)$"),
    re.compile(r"^keyboard (?:light|backlight|leds?) (?P<t>on|off)$"),
)

#: Reasoning recorded on a local match, so the audit trail distinguishes a
#: classification nobody paid for from one a model produced.
_REASON = "Matched a built-in pattern locally; no model call was needed."


def _normalise(utterance: str) -> str:
    """Reduce an utterance to a comparable form.

    Args:
        utterance: What the user said.

    Returns:
        Lower-cased, filler-stripped, without trailing punctuation.
    """
    text = utterance.strip().lower()
    text = _LEADING_FILLER.sub("", text)
    return text.strip().rstrip(".!?,").strip()


def _target_is_simple(target: str) -> bool:
    """Whether a captured target is a plain name rather than a phrase.

    Args:
        target: The captured text.

    Returns:
        ``True`` when it looks like something nameable.
    """
    words = target.split()
    if not words or len(words) > 4:
        return False
    # "open a new tab in chrome" captures "a new tab in chrome", which a pattern
    # cannot interpret. The preposition is the tell.
    return not any(word in _STRUCTURE_WORDS for word in words)


def classify_locally(utterance: str) -> IntentClassification | None:
    """Classify an utterance without calling a model.

    Args:
        utterance: What the user said.

    Returns:
        A classification when the utterance matches a built-in pattern
        unambiguously, otherwise ``None`` so the caller falls back to the model.
    """
    text = _normalise(utterance)
    if not text or len(text.split()) > _MAX_WORDS:
        return None

    # Order is the whole design here: every rule below is checked before the
    # greedy verb patterns, because "open" and "find" will otherwise claim a
    # request that a more specific rule reads correctly.
    for pattern, intent in _EXACT_PATTERNS:
        if pattern.match(text):
            return _classification(intent, None)

    for pattern, operation in _DEV_PATTERNS:
        if pattern.match(text):
            return _classification(IntentType.RUN_DEV_COMMAND, operation)

    for pattern, fixed in _WIFI_PATTERNS:
        if match := pattern.match(text):
            # A pattern with a fixed target ("status") uses it; the on/off ones
            # capture the direction from the group.
            return _classification(IntentType.WIFI, fixed or match.group("t"))

    for pattern, action in _MEDIA_PATTERNS:
        if pattern.match(text):
            return _classification(IntentType.MEDIA_CONTROL, action)

    for pattern, action in _WINDOW_PATTERNS:
        if match := pattern.match(text):
            name = match.groupdict().get("name")
            return _classification(
                IntentType.WINDOW_CONTROL,
                (name or "all").strip(),
                parameters=[IntentParameter(key="action", value=action)],
            )

    if _RUNNING_APPS_PATTERN.match(text):
        return _classification(IntentType.RUNNING_APPS, None)

    if match := _KILL_PATTERN.match(text):
        return _classification(IntentType.CLOSE_PROCESS, match.group("target").strip())

    if _LOCATE_PATTERN.match(text):
        return _classification(IntentType.LOCATE_DEVICE, None)

    for pattern in _PANIC_PATTERNS:
        if pattern.match(text):
            return _classification(IntentType.PANIC, None)

    for pattern in _KEYBOARD_LIGHT_PATTERNS:
        if match := pattern.match(text):
            return _classification(IntentType.KEYBOARD_LIGHT, match.group("t"))

    # Before file search, so "google X" is a web search rather than a file query.
    for pattern in _WEB_SEARCH_PATTERNS:
        if match := pattern.match(text):
            return _classification(IntentType.WEB_SEARCH, match.group("target").strip())

    for pattern in _CREATE_FOLDER_PATTERNS:
        if match := pattern.match(text):
            name = _CREATE_FOLDER_NOISE.sub("", match.group("target").strip()).strip()
            if name:
                return _classification(IntentType.CREATE_FOLDER, name)

    # Before file search, since "send me report.pdf" is a retrieval, not a query.
    for pattern in _SEND_FILE_PATTERNS:
        if match := pattern.match(text):
            return _classification(IntentType.SEND_FILE, match.group("target").strip())

    # Before the application patterns, which would otherwise claim
    # "open github.com" as an application named "github.com".
    if match := _WEBSITE_PATTERN.match(text):
        return _classification(IntentType.OPEN_WEBSITE, match.group("target"))

    for pattern in _SCREEN_PATTERNS:
        if match := pattern.match(text):
            # The question itself is the target; the analyst needs it, not a label.
            return _classification(IntentType.LOOK_AT_SCREEN, text)

    for pattern, intent in _LEVEL_PATTERNS:
        if match := pattern.match(text):
            return _classification(intent, match.group("target"))

    if match := _FOLDER_PATTERN.match(text):
        target = match.group("target").strip()
        # Only names that are unambiguously directories. Everything else — an
        # application, a document, a project — reads identically to a pattern, and
        # guessing wrong sends "open notepad" to the file explorer.
        if target in _KNOWN_FOLDERS or text.rstrip(".").endswith(("folder", "directory")):
            return _classification(IntentType.OPEN_FOLDER, target)

    for pattern in _SEARCH_PATTERNS:
        if match := pattern.match(text):
            return _classification(IntentType.SEARCH_FILES, match.group("target").strip())

    for pattern, intent in _TARGET_PATTERNS:
        if match := pattern.match(text):
            target = match.group("target").strip()
            if _target_is_simple(target):
                return _classification(intent, target)
            # A structured phrase: fall through to the model rather than guessing.
            return None

    return None


def _classification(
    intent: IntentType, target: str | None, parameters: list[IntentParameter] | None = None
) -> IntentClassification:
    """Build a classification for a local match.

    Confidence is 1.0 and that is not flattery: the pattern either matched or it
    did not, so there is no uncertainty to express. It also means the Brain's
    low-confidence confirmation rule never fires on a local match — correct,
    because the alternative would be asking the user to confirm "take a
    screenshot".

    Args:
        intent: The matched intent.
        target: The captured target, if any.
        parameters: Extra details, if any.

    Returns:
        The classification.
    """
    return IntentClassification(
        intent=intent,
        target=target,
        parameters=parameters or [],
        confidence=1.0,
        reasoning=_REASON,
    )
