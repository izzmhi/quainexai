"""Memory manager — the API the rest of Quainex uses to remember.

Purpose:
    Sit between the store and its callers so that remembering is one call, not a
    sequence every caller has to get right.

Why a manager over the store:
    Recording a completed turn means writing the user's utterance, the
    assistant's reply and an activity record — three writes that must always
    happen together. Left to callers, the voice loop, the HTTP route and Phase
    10's autonomous loop would each implement that sequence, and they would drift.

    It also converts stored turns into the ``ChatMessage`` shape the Brain
    expects, so the Brain depends on its own vocabulary rather than on the
    database's.

Architecture:
    VoiceSession / commands route
        -> MemoryManager.remember_exchange()   one call, three writes
        -> MemoryManager.conversation_context() -> list[ChatMessage] -> Brain

Dependencies:
    quainex.core.memory.store, quainex.services.ai.provider

Future improvements:
    * Summarise a long conversation into a fact when it ends, so the useful part
      outlives the turn window.
    * Surface relevant facts to the Brain as system context, so "open my project"
      resolves against what Quainex already knows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from quainex.core.logging import get_logger
from quainex.services.ai.provider import ChatMessage

if TYPE_CHECKING:
    from quainex.config.settings import Settings
    from quainex.core.brain import Intent
    from quainex.core.commands import CommandResult
    from quainex.core.memory.store import (
        ActivityEntry,
        FactRecord,
        MemoryStore,
        PreferenceRecord,
        TurnRecord,
    )

_log = get_logger(__name__)

#: Conversations that do not name a session share this one.
DEFAULT_SESSION_ID = "default"


class MemoryManager:
    """Short-term conversation context and long-term recall."""

    def __init__(self, store: MemoryStore, settings: Settings) -> None:
        """Construct the manager.

        Args:
            store: Backing store.
            settings: Configuration supplying the context window size.
        """
        self._store = store
        self._settings = settings

    # -- short-term: the conversation window ------------------------------

    async def conversation_context(self, session_id: str = DEFAULT_SESSION_ID) -> list[ChatMessage]:
        """Return recent turns in the shape the Brain expects.

        Args:
            session_id: Conversation to read.

        Returns:
            Recent turns as chat messages, oldest first.
        """
        turns = await self._store.recent_turns(session_id, self._settings.memory_context_turns)
        return [
            ChatMessage(role="user" if turn.role == "user" else "assistant", content=turn.content)
            for turn in turns
        ]

    async def recent_turns(
        self, session_id: str = DEFAULT_SESSION_ID, limit: int | None = None
    ) -> list[TurnRecord]:
        """Return recent turns with their metadata.

        Args:
            session_id: Conversation to read.
            limit: Maximum turns; defaults to the configured window.

        Returns:
            Turns, oldest first.
        """
        return await self._store.recent_turns(
            session_id, limit if limit is not None else self._settings.memory_context_turns
        )

    async def remember_exchange(
        self,
        utterance: str,
        intent: Intent | None,
        result: CommandResult | None,
        session_id: str = DEFAULT_SESSION_ID,
    ) -> None:
        """Record one complete exchange.

        Writes the user's turn, Quainex's reply, and — when something was
        actually attempted — an activity record. Kept as one call because these
        three writes must not drift apart across the callers that produce them.

        Args:
            utterance: What the user said.
            intent: What Quainex understood, when it got that far.
            result: What it did, when it got that far.
            session_id: Conversation to append to.
        """
        await self._store.add_turn(
            session_id=session_id,
            role="user",
            content=utterance,
            intent=intent.intent.value if intent else None,
        )

        if result is not None:
            await self._store.add_turn(
                session_id=session_id,
                role="assistant",
                content=result.message,
                intent=intent.intent.value if intent else None,
            )
            await self._store.record_activity(
                intent=result.intent,
                target=intent.target if intent else None,
                status=result.status.value,
                executed=result.executed,
                detail=result.message,
            )

    async def clear_conversation(self, session_id: str = DEFAULT_SESSION_ID) -> int:
        """Forget a conversation.

        Args:
            session_id: Conversation to clear.

        Returns:
            How many turns were removed.
        """
        return await self._store.clear_conversation(session_id)

    # -- long-term: preferences, facts, activity --------------------------

    async def set_preference(self, key: str, value: str) -> None:
        """Store a preference.

        Args:
            key: Preference identifier.
            value: Value to store.
        """
        await self._store.set_preference(key, value)

    async def get_preference(self, key: str) -> str | None:
        """Read a preference.

        Args:
            key: Preference identifier.

        Returns:
            The value, or ``None``.
        """
        return await self._store.get_preference(key)

    async def list_preferences(self) -> list[PreferenceRecord]:
        """Return every preference.

        Returns:
            Preferences ordered by key.
        """
        return await self._store.list_preferences()

    async def remember(self, category: str, key: str, value: str) -> None:
        """Record a fact.

        Args:
            category: Grouping such as ``project``.
            key: Identifier within the category.
            value: What to remember.
        """
        await self._store.remember_fact(category, key, value)

    async def recall(self, category: str, key: str) -> str | None:
        """Recall a fact.

        Args:
            category: Grouping to look in.
            key: Identifier within the category.

        Returns:
            The value, or ``None``.
        """
        return await self._store.recall_fact(category, key)

    async def search(self, query: str, limit: int = 20) -> list[FactRecord]:
        """Search facts by substring.

        Args:
            query: Text to look for.
            limit: Maximum results.

        Returns:
            Matching facts.
        """
        return await self._store.search_facts(query, limit)

    async def forget(self, category: str, key: str) -> bool:
        """Delete a fact.

        Args:
            category: Grouping to delete from.
            key: Identifier within the category.

        Returns:
            Whether anything was removed.
        """
        return await self._store.forget_fact(category, key)

    async def recent_activity(self, limit: int = 20) -> list[ActivityEntry]:
        """Return what Quainex has been doing.

        Args:
            limit: Maximum results.

        Returns:
            Actions, newest first.
        """
        return await self._store.recent_activity(limit)
