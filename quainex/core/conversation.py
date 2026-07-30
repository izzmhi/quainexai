"""Conversational replies for intents that are not machine actions.

Purpose:
    Answer questions, return a greeting, and respond honestly when a request was
    not understood — so that talking to Quainex works even when nothing on the
    machine needs to happen.

Why this module exists at all:
    The Brain has always been able to classify conversation: ``ANSWER_QUESTION``,
    ``SMALL_TALK`` and ``UNKNOWN`` are in ``IntentType``, and ``is_actionable``
    exists precisely to name them. What was missing was the other half — no
    command was ever registered against them, so the executor's first gate
    rejected them with *"'small_talk' is not something Quainex can execute"*.

    That message was technically accurate and completely wrong as a product: an
    AI operating system that cannot answer "how are you doing?" is broken in the
    first thirty seconds of use. The gate was right; the registry was incomplete.

Why replies are short:
    Every one of these can be spoken aloud through the voice loop or delivered to
    a phone over Telegram. A four-paragraph answer is unusable in both. Brevity is
    requested in the prompt rather than enforced with a small token cap, because
    on current models the budget covers internal reasoning as well as the visible
    answer — starving it truncates mid-sentence instead of producing a short reply.

What it is *not* allowed to do:
    Claim to have acted. These handlers have no access to the desktop controller,
    which is not an oversight but the design: a conversational reply that says
    "I've opened that for you" when nothing was opened is worse than an error, and
    the only reliable way to prevent it is to make the action unreachable from
    here. Anything with a side effect goes through the command registry and its
    gates.

Architecture:
    Intent (ANSWER_QUESTION | SMALL_TALK | UNKNOWN)
        -> Conversationalist.reply()
             |-- system prompt: identity + what Quainex can actually do
             |                  (derived from INTENT_DESCRIPTIONS, so the list
             |                   cannot drift from the Brain's own vocabulary)
             |-- memory: recent turns, so follow-ups resolve
             |-- memory: remembered facts, so "where is my project" works
             +-- provider.complete() -> one short reply

Dependencies:
    quainex.core.brain (intent vocabulary), quainex.core.memory,
    quainex.services.ai

Example:
    >>> reply = await conversation.reply(message="how are you?", kind=IntentType.SMALL_TALK)
    ...  # doctest: +SKIP
    'Running fine — 22 commands loaded and nothing on fire. What do you need?'

Future improvements:
    * Stream the reply so long answers appear as they are produced.
    * Let the model ask a clarifying question and carry the pending intent
      forward, rather than answering a half-understood request.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from quainex.core.brain import INTENT_DESCRIPTIONS, NON_ACTIONABLE, IntentType
from quainex.core.logging import get_logger
from quainex.services.ai.provider import ChatMessage

if TYPE_CHECKING:
    from quainex.config.settings import Settings
    from quainex.core.memory import MemoryManager
    from quainex.services.ai.provider import AIProvider

_log = get_logger(__name__)

#: How many remembered facts to put in front of the model. Bounded because this
#: cost is paid on every conversational turn, and a personal store grows forever.
_MAX_FACTS = 25

#: Per-intent guidance. Kept separate from the identity prompt so that the shared
#: part is written once and each case only states what makes it different.
_GUIDANCE: dict[IntentType, str] = {
    IntentType.ANSWER_QUESTION: (
        "Answer the question directly. Two or three sentences unless it genuinely "
        "needs more. If you do not know, say so instead of guessing."
    ),
    IntentType.SMALL_TALK: (
        "This is small talk. Reply in one or two sentences, warmly and briefly, "
        "and do not launch into an explanation of your capabilities unless asked."
    ),
    IntentType.UNKNOWN: (
        "You could not tell what action was being asked for. Say so plainly in one "
        "sentence. Then either answer it as a question if that is what it turned "
        "out to be, or name the closest thing you can actually do. Never imply "
        "that you have done anything."
    ),
}


def _capability_summary() -> str:
    """Describe what Quainex can actually do, for the system prompt.

    Derived from ``INTENT_DESCRIPTIONS`` rather than written out here, so that
    adding a command cannot leave this list stale. The same dictionary builds the
    Brain's classification prompt, which means the assistant's description of
    itself and its actual vocabulary come from one place.

    Returns:
        A newline-separated list of the actionable intents.
    """
    lines = [
        f"- {intent.value}: {description}"
        for intent, description in INTENT_DESCRIPTIONS.items()
        if intent not in NON_ACTIONABLE
    ]
    return "\n".join(lines)


class Conversationalist:
    """Produces spoken-length replies for non-actionable intents."""

    def __init__(
        self,
        provider: AIProvider,
        settings: Settings,
        memory: MemoryManager | None = None,
    ) -> None:
        """Construct the conversationalist.

        Args:
            provider: The model backing replies.
            settings: Application configuration.
            memory: Optional store supplying recent turns and remembered facts.
                Without it replies still work; they simply have no continuity.
        """
        self._provider = provider
        self._settings = settings
        self._memory = memory

    @property
    def is_available(self) -> bool:
        """Whether a provider is configured to answer."""
        return self._provider.is_available

    async def reply(self, *, message: str, kind: IntentType) -> str:
        """Produce a conversational reply.

        Args:
            message: What the user said. For ``ANSWER_QUESTION`` this is the
                question; otherwise it is the utterance itself.
            kind: Which of the three non-actionable intents this is.

        Returns:
            A short reply, suitable for speaking aloud.

        Raises:
            ProviderError: No provider could answer.
        """
        history = await self._history()
        messages = [*history, ChatMessage(role="user", content=message)]

        reply = await self._provider.complete(
            messages=messages,
            system=await self._system_prompt(kind),
            # No cap passed: each provider applies its own, and only the provider
            # knows what its own limits are. Free tiers meter *requested* tokens
            # against a per-minute budget, so a caller-supplied number that suits
            # Claude gets a 429 out of Groq. Reply length is controlled by the
            # instruction in the prompt, which works on every provider.
        )

        _log.info("conversational_reply", kind=kind.value, characters=len(reply))
        return reply.strip()

    # -- internals --------------------------------------------------------

    async def _history(self) -> list[ChatMessage]:
        """Fetch recent turns, so a follow-up resolves against what came before.

        Returns:
            Recent conversation turns, oldest first. Empty when there is no
            memory, or when reading it failed — losing continuity is a far better
            outcome than failing the reply.
        """
        if self._memory is None:
            return []
        try:
            return await self._memory.conversation_context()
        except Exception as exc:
            _log.warning("conversation_context_unavailable", error=str(exc))
            return []

    async def _system_prompt(self, kind: IntentType) -> str:
        """Build the system prompt for one reply.

        Args:
            kind: Which non-actionable intent is being answered.

        Returns:
            The full system prompt.
        """
        parts = [
            f"You are {self._settings.app_name}, a personal AI operating system "
            f"running on the user's own Windows machine. You are speaking to the "
            f"person who owns it.",
            "Your replies may be read aloud by a speech synthesiser or delivered "
            "to a phone, so keep them short and plain. No markdown, no bullet "
            "lists, no headings.",
            "Never claim to have performed an action. You are answering, not "
            "acting. If something needs doing on the machine, say what you would "
            "do and let the user ask for it.",
            _GUIDANCE.get(kind, _GUIDANCE[IntentType.UNKNOWN]),
            # The identifiers are for the model's benefit, not the user's. Without
            # this line it repeats them verbatim — a reply that says "you could try
            # look_at_screen" reads like a leaked internal, because it is one.
            f"For reference, these are the actions you can be asked to perform. The "
            f"identifiers are internal: never repeat one to the user, describe the "
            f"capability in ordinary words instead.\n{_capability_summary()}",
        ]

        if facts := await self._facts():
            parts.append(
                "Things the user has told you to remember. Use them when relevant "
                f"and do not repeat them back unprompted:\n{facts}"
            )

        return "\n\n".join(parts)

    async def _facts(self) -> str:
        """Fetch remembered facts for the prompt.

        Returns:
            A newline-separated list, or an empty string when there is nothing
            stored or memory is unavailable.
        """
        if self._memory is None:
            return ""
        try:
            # An empty query matches everything, so this is "the most recent
            # facts" rather than a search.
            records = await self._memory.search("", limit=_MAX_FACTS)
        except Exception as exc:
            _log.warning("facts_unavailable", error=str(exc))
            return ""

        return "\n".join(f"- {record.category}/{record.key}: {record.value}" for record in records)
