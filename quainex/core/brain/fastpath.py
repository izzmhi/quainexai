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
#: A formulaic command is short. Past this length the request is being explained
#: rather than issued, and explanation is exactly what a model is for.
_MAX_WORDS = 6

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
)

#: Commands whose target is captured from the phrasing.
_TARGET_PATTERNS: tuple[tuple[re.Pattern[str], IntentType], ...] = (
    (re.compile(r"^(?:open|launch|start|run) (?P<target>.+)$"), IntentType.OPEN_APPLICATION),
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

    for pattern, intent in _EXACT_PATTERNS:
        if pattern.match(text):
            return _classification(intent, None)

    # Checked before the application patterns, which would otherwise claim
    # "open github.com" as an application named "github.com".
    if match := _WEBSITE_PATTERN.match(text):
        return _classification(IntentType.OPEN_WEBSITE, match.group("target"))

    for pattern, intent in _LEVEL_PATTERNS:
        if match := pattern.match(text):
            return _classification(intent, match.group("target"))

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
