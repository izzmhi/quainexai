"""Tests for conversational replies.

The provider is always a double, so nothing here makes a network call. What is
being tested is the *prompt* and the *context* — that the assistant is told who
it is, what it can actually do, what was said before, and what it must not claim.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from quainex.config.settings import Settings
from quainex.core.brain import IntentType
from quainex.core.conversation import Conversationalist
from quainex.core.exceptions import ProviderError
from quainex.services.ai.provider import ChatMessage


class RecordingProvider:
    """Captures the request instead of sending it.

    Attributes:
        system: The system prompt of the last call.
        messages: The message list of the last call.
        max_tokens: The token cap of the last call.
    """

    def __init__(self, reply: str = "All good.", error: Exception | None = None) -> None:
        self.system: str | None = None
        self.messages: list[ChatMessage] = []
        self.max_tokens: int | None = None
        self._reply = reply
        self._error = error

    @property
    def name(self) -> str:
        return "recording"

    @property
    def is_available(self) -> bool:
        return True

    async def complete(
        self,
        *,
        messages: list[ChatMessage],
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        if self._error is not None:
            raise self._error
        self.system = system
        self.messages = messages
        self.max_tokens = max_tokens
        return self._reply

    async def parse(self, **_: object) -> object:  # pragma: no cover - unused here
        raise NotImplementedError

    async def look(self, **_: object) -> str:  # pragma: no cover - unused here
        raise NotImplementedError

    async def read_document(self, **_: object) -> str:  # pragma: no cover - unused here
        raise NotImplementedError

    async def aclose(self) -> None:
        return None


class FakeMemory:
    """Supplies a fixed history and fact set."""

    def __init__(
        self,
        context: list[ChatMessage] | None = None,
        facts: list[object] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._context = context or []
        self._facts = facts or []
        self._error = error

    async def conversation_context(self, session_id: str = "default") -> list[ChatMessage]:
        if self._error is not None:
            raise self._error
        return self._context

    async def search(self, query: str, limit: int = 20) -> list[object]:
        if self._error is not None:
            raise self._error
        return self._facts[:limit]


class FactRow:
    """Minimal stand-in for a stored fact."""

    def __init__(self, category: str, key: str, value: str) -> None:
        self.category = category
        self.key = key
        self.value = value


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Isolated settings."""
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        log_dir=tmp_path / "logs",
        database_path=tmp_path / "test.db",
        credentials_path=tmp_path / "credentials.dat",
    )


# -- the reply -------------------------------------------------------------


async def test_a_greeting_gets_a_reply(settings: Settings):
    provider = RecordingProvider("Running fine — what do you need?")
    conversation = Conversationalist(provider, settings)

    reply = await conversation.reply(message="how are you?", kind=IntentType.SMALL_TALK)

    assert reply == "Running fine — what do you need?"
    assert provider.messages[-1].content == "how are you?"


async def test_surrounding_whitespace_is_trimmed(settings: Settings):
    """The reply may be spoken or shown in a transcript; neither wants padding."""
    conversation = Conversationalist(RecordingProvider("\n  Yes.  \n"), settings)

    assert await conversation.reply(message="ok?", kind=IntentType.SMALL_TALK) == "Yes."


async def test_a_provider_failure_propagates_rather_than_being_faked(settings: Settings):
    """No cheerful fallback string.

    A canned "I'm doing well!" when the model never answered would make a broken
    configuration invisible — the user would think Quainex was working.
    """
    conversation = Conversationalist(RecordingProvider(error=ProviderError("no key")), settings)

    with pytest.raises(ProviderError):
        await conversation.reply(message="hi", kind=IntentType.SMALL_TALK)


# -- the prompt ------------------------------------------------------------


async def test_the_assistant_is_told_who_and_where_it_is(settings: Settings):
    provider = RecordingProvider()
    await Conversationalist(provider, settings).reply(
        message="who are you?", kind=IntentType.SMALL_TALK
    )

    assert provider.system is not None
    assert "Quainex" in provider.system
    assert "Windows" in provider.system


async def test_the_assistant_is_forbidden_from_claiming_it_acted(settings: Settings):
    """The single most important line in the prompt.

    A conversational reply has no desktop access at all, so any claim to have
    opened, closed or changed something would be false.
    """
    provider = RecordingProvider()
    await Conversationalist(provider, settings).reply(
        message="open notepad", kind=IntentType.UNKNOWN
    )

    assert provider.system is not None
    assert "Never claim to have performed an action" in provider.system


async def test_replies_are_constrained_to_something_speakable(settings: Settings):
    """These go through a speech synthesiser and a Telegram message."""
    provider = RecordingProvider()
    await Conversationalist(provider, settings).reply(message="hi", kind=IntentType.SMALL_TALK)

    assert provider.system is not None
    assert "read aloud" in provider.system
    assert "No markdown" in provider.system


async def test_no_token_cap_is_imposed_on_the_provider(settings: Settings):
    """Only the provider knows its own limits.

    A number that suits Claude — where the budget also funds hidden reasoning —
    gets a 429 out of a free tier that meters *requested* tokens against a
    per-minute quota. Passing one from here would override the per-provider cap
    that exists precisely to avoid that. Reply length is controlled by the prompt,
    which works everywhere.
    """
    provider = RecordingProvider()
    await Conversationalist(provider, settings).reply(message="hi", kind=IntentType.SMALL_TALK)

    assert provider.max_tokens is None


async def test_the_prompt_lists_what_quainex_can_actually_do(settings: Settings):
    """So that an unrecognised request can name the closest real capability."""
    provider = RecordingProvider()
    await Conversationalist(provider, settings).reply(
        message="make me a sandwich", kind=IntentType.UNKNOWN
    )

    assert provider.system is not None
    assert "open_application" in provider.system
    assert "screenshot" in provider.system


async def test_the_model_is_told_not_to_repeat_internal_identifiers(settings: Settings):
    """Without this it does.

    A reply saying "you could try look_at_screen" is a leaked internal, because
    that is exactly what it is.
    """
    provider = RecordingProvider()
    await Conversationalist(provider, settings).reply(message="???", kind=IntentType.UNKNOWN)

    assert provider.system is not None
    assert "never repeat one to the user" in provider.system


async def test_the_capability_list_excludes_the_conversational_intents(settings: Settings):
    """Listing "small_talk" as an action it can perform would be nonsense."""
    provider = RecordingProvider()
    await Conversationalist(provider, settings).reply(message="hi", kind=IntentType.SMALL_TALK)

    assert provider.system is not None
    capabilities = provider.system.split("actions you can be asked to perform:")[-1]
    assert "small_talk" not in capabilities
    assert "answer_question" not in capabilities


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (IntentType.SMALL_TALK, "small talk"),
        (IntentType.ANSWER_QUESTION, "Answer the question directly"),
        (IntentType.UNKNOWN, "could not tell what action"),
    ],
)
async def test_each_intent_gets_its_own_guidance(
    settings: Settings, kind: IntentType, expected: str
):
    """Three different situations that must not be answered identically."""
    provider = RecordingProvider()
    await Conversationalist(provider, settings).reply(message="anything", kind=kind)

    assert provider.system is not None
    assert expected in provider.system


# -- memory ----------------------------------------------------------------


async def test_recent_turns_are_included_so_follow_ups_resolve(settings: Settings):
    provider = RecordingProvider()
    memory = FakeMemory(
        context=[
            ChatMessage(role="user", content="what is the capital of France?"),
            ChatMessage(role="assistant", content="Paris."),
        ]
    )

    await Conversationalist(provider, settings, memory).reply(  # type: ignore[arg-type]
        message="and its population?", kind=IntentType.ANSWER_QUESTION
    )

    assert [message.content for message in provider.messages] == [
        "what is the capital of France?",
        "Paris.",
        "and its population?",
    ]


async def test_remembered_facts_reach_the_prompt(settings: Settings):
    """So "where is my project?" can be answered from what the user has said."""
    provider = RecordingProvider()
    memory = FakeMemory(facts=[FactRow("project", "quainex", "C:/Users/G8/dev/quainex")])

    await Conversationalist(provider, settings, memory).reply(  # type: ignore[arg-type]
        message="where is my project?", kind=IntentType.ANSWER_QUESTION
    )

    assert provider.system is not None
    assert "C:/Users/G8/dev/quainex" in provider.system


async def test_no_facts_means_no_empty_section_in_the_prompt(settings: Settings):
    provider = RecordingProvider()

    await Conversationalist(provider, settings, FakeMemory()).reply(  # type: ignore[arg-type]
        message="hi", kind=IntentType.SMALL_TALK
    )

    assert provider.system is not None
    assert "told you to remember" not in provider.system


async def test_a_broken_memory_costs_continuity_not_the_reply(settings: Settings):
    """Failing the answer because the history could not be read is the wrong trade.

    The user asked a question; losing the previous turn is a degradation, while
    losing the answer is a failure.
    """
    provider = RecordingProvider("Still here.")
    memory = FakeMemory(error=RuntimeError("database is locked"))

    reply = await Conversationalist(provider, settings, memory).reply(  # type: ignore[arg-type]
        message="hi", kind=IntentType.SMALL_TALK
    )

    assert reply == "Still here."
    assert [message.content for message in provider.messages] == ["hi"]


async def test_availability_follows_the_provider(settings: Settings):
    assert Conversationalist(RecordingProvider(), settings).is_available is True
