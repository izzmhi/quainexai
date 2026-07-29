"""Memory store contract and its SQLAlchemy implementation.

Purpose:
    Persist what Quainex should remember, and read it back.

What "short-term" and "long-term" mean here:
    Not two copies of the same data in different places. They are different
    *scopes*:

    * **Short-term** — the current conversation's recent turns, scoped to a
      session and read as a bounded window. It gives "close it" a referent.
    * **Long-term** — preferences, facts and activity. Not tied to a
      conversation, survives restarts, and is what makes Quainex feel like it
      knows the user rather than meeting them each time.

    Keeping a separate in-process cache of recent turns was considered and
    rejected: a local SQLite read is well under a millisecond, and two copies of
    conversation state is two things that can disagree.

Architecture:
    MemoryManager
        -> MemoryStore (Protocol)
             |-- SqlAlchemyMemoryStore  (implemented)
             +-- InMemoryMemoryStore    (tests, and a future privacy mode that
                                          keeps nothing on disk)
        -> Database.session() -> SQLite

Dependencies:
    sqlalchemy, pydantic, quainex.database, quainex.models

Future improvements:
    * Embeddings on facts, so "where do I keep the tax stuff" matches a fact
      recorded as "documents/finance".
    * Retention policy: activity older than N months summarised then pruned.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, cast

from pydantic import BaseModel
from sqlalchemy import delete, desc, select
from sqlalchemy.engine import CursorResult

from quainex.core.logging import get_logger
from quainex.models.memory import ActivityRecord, ConversationTurn, Fact, Preference

if TYPE_CHECKING:
    from quainex.database.engine import Database

_log = get_logger(__name__)


class TurnRecord(BaseModel):
    """One remembered conversation turn.

    Attributes:
        role: Who spoke.
        content: What was said.
        intent: The classified intent, when there was one.
        created_at: When it happened.
    """

    role: str
    content: str
    intent: str | None = None
    created_at: datetime


class FactRecord(BaseModel):
    """One remembered fact.

    Attributes:
        category: Grouping such as ``project`` or ``folder``.
        key: Identifier within the category.
        value: The remembered content.
        updated_at: When it was last changed.
    """

    category: str
    key: str
    value: str
    updated_at: datetime


class PreferenceRecord(BaseModel):
    """One stored preference.

    Attributes:
        key: Preference identifier.
        value: Its value.
        updated_at: When it was last set.
    """

    key: str
    value: str
    updated_at: datetime


class ActivityEntry(BaseModel):
    """One recorded action.

    Attributes:
        intent: The intent acted on.
        target: What it acted on.
        status: The command result status.
        executed: Whether a side effect occurred.
        detail: Human-readable outcome.
        created_at: When it happened.
    """

    intent: str
    target: str | None
    status: str
    executed: bool
    detail: str
    created_at: datetime


class MemoryStore(Protocol):
    """Everything Quainex needs to remember and recall."""

    async def add_turn(
        self, session_id: str, role: str, content: str, intent: str | None = None
    ) -> None:
        """Append a conversation turn."""
        ...

    async def recent_turns(self, session_id: str, limit: int) -> list[TurnRecord]:
        """Return the most recent turns of a conversation, oldest first."""
        ...

    async def clear_conversation(self, session_id: str) -> int:
        """Delete a conversation's turns, returning how many were removed."""
        ...

    async def set_preference(self, key: str, value: str) -> None:
        """Set a preference, replacing any existing value."""
        ...

    async def get_preference(self, key: str) -> str | None:
        """Return a preference value, or ``None``."""
        ...

    async def list_preferences(self) -> list[PreferenceRecord]:
        """Return every stored preference."""
        ...

    async def remember_fact(self, category: str, key: str, value: str) -> None:
        """Record a fact, replacing any existing value for that key."""
        ...

    async def recall_fact(self, category: str, key: str) -> str | None:
        """Return a fact's value, or ``None``."""
        ...

    async def search_facts(self, query: str, limit: int) -> list[FactRecord]:
        """Find facts whose key or value contains ``query``."""
        ...

    async def forget_fact(self, category: str, key: str) -> bool:
        """Delete a fact, reporting whether one was removed."""
        ...

    async def record_activity(
        self, intent: str, target: str | None, status: str, executed: bool, detail: str
    ) -> None:
        """Append an activity record."""
        ...

    async def recent_activity(self, limit: int) -> list[ActivityEntry]:
        """Return the most recent actions, newest first."""
        ...


class SqlAlchemyMemoryStore:
    """Memory backed by SQLite through SQLAlchemy."""

    def __init__(self, database: Database) -> None:
        """Construct the store.

        Args:
            database: The database supplying sessions.
        """
        self._database = database

    # -- conversation (short-term) ---------------------------------------

    async def add_turn(
        self, session_id: str, role: str, content: str, intent: str | None = None
    ) -> None:
        """Append a conversation turn.

        Args:
            session_id: Conversation the turn belongs to.
            role: ``user`` or ``assistant``.
            content: What was said.
            intent: Classified intent, when there was one.
        """
        async with self._database.session() as session:
            session.add(
                ConversationTurn(session_id=session_id, role=role, content=content, intent=intent)
            )

    async def recent_turns(self, session_id: str, limit: int) -> list[TurnRecord]:
        """Return the most recent turns of a conversation.

        Args:
            session_id: Conversation to read.
            limit: Maximum number of turns.

        Returns:
            Turns in chronological order, oldest first.
        """
        if limit <= 0:
            return []

        async with self._database.session() as session:
            # Newest-first in SQL to use the index and apply the limit, then
            # reversed in Python: a conversation must be replayed oldest-first.
            result = await session.execute(
                select(ConversationTurn)
                .where(ConversationTurn.session_id == session_id)
                .order_by(desc(ConversationTurn.created_at), desc(ConversationTurn.id))
                .limit(limit)
            )
            rows = list(result.scalars().all())

        return [
            TurnRecord(
                role=row.role,
                content=row.content,
                intent=row.intent,
                created_at=row.created_at,
            )
            for row in reversed(rows)
        ]

    async def clear_conversation(self, session_id: str) -> int:
        """Delete every turn of a conversation.

        Args:
            session_id: Conversation to clear.

        Returns:
            How many turns were removed.
        """
        async with self._database.session() as session:
            # `execute` is typed as returning Result; a DELETE actually returns a
            # CursorResult, which is what carries `rowcount`.
            result = cast(
                "CursorResult[Any]",
                await session.execute(
                    delete(ConversationTurn).where(ConversationTurn.session_id == session_id)
                ),
            )
        removed = int(result.rowcount or 0)
        _log.info("conversation_cleared", session_id=session_id, turns=removed)
        return removed

    # -- preferences (long-term) -----------------------------------------

    async def set_preference(self, key: str, value: str) -> None:
        """Set a preference, replacing any existing value.

        Args:
            key: Preference identifier.
            value: Value to store.
        """
        async with self._database.session() as session:
            existing = await session.scalar(select(Preference).where(Preference.key == key))
            if existing is None:
                session.add(Preference(key=key, value=value))
            else:
                existing.value = value
        _log.info("preference_set", key=key)

    async def get_preference(self, key: str) -> str | None:
        """Return a preference value.

        Args:
            key: Preference identifier.

        Returns:
            The value, or ``None`` when unset.
        """
        async with self._database.session() as session:
            row = await session.scalar(select(Preference).where(Preference.key == key))
            return row.value if row else None

    async def list_preferences(self) -> list[PreferenceRecord]:
        """Return every stored preference.

        Returns:
            Preferences ordered by key.
        """
        async with self._database.session() as session:
            result = await session.execute(select(Preference).order_by(Preference.key))
            return [
                PreferenceRecord(key=row.key, value=row.value, updated_at=row.updated_at)
                for row in result.scalars().all()
            ]

    # -- facts (long-term) ------------------------------------------------

    async def remember_fact(self, category: str, key: str, value: str) -> None:
        """Record a fact, replacing any existing value for that key.

        Args:
            category: Grouping such as ``project``.
            key: Identifier within the category.
            value: What to remember.
        """
        async with self._database.session() as session:
            existing = await session.scalar(
                select(Fact).where(Fact.category == category, Fact.key == key)
            )
            if existing is None:
                session.add(Fact(category=category, key=key, value=value))
            else:
                existing.value = value
        _log.info("fact_remembered", category=category, key=key)

    async def recall_fact(self, category: str, key: str) -> str | None:
        """Return a fact's value.

        Args:
            category: Grouping to look in.
            key: Identifier within the category.

        Returns:
            The value, or ``None``.
        """
        async with self._database.session() as session:
            row = await session.scalar(
                select(Fact).where(Fact.category == category, Fact.key == key)
            )
            return row.value if row else None

    async def search_facts(self, query: str, limit: int) -> list[FactRecord]:
        """Find facts whose key or value contains ``query``.

        Substring matching, deliberately: semantic search needs embeddings and a
        model call, which is the wrong trade for recalling "where is my tax
        folder" on a local machine.

        Args:
            query: Text to look for.
            limit: Maximum results.

        Returns:
            Matching facts, most recently updated first.
        """
        needle = f"%{query.strip()}%"
        async with self._database.session() as session:
            result = await session.execute(
                select(Fact)
                .where(Fact.key.ilike(needle) | Fact.value.ilike(needle))
                .order_by(desc(Fact.updated_at))
                .limit(limit)
            )
            return [
                FactRecord(
                    category=row.category,
                    key=row.key,
                    value=row.value,
                    updated_at=row.updated_at,
                )
                for row in result.scalars().all()
            ]

    async def forget_fact(self, category: str, key: str) -> bool:
        """Delete a fact.

        Args:
            category: Grouping to delete from.
            key: Identifier within the category.

        Returns:
            Whether a fact was removed.
        """
        async with self._database.session() as session:
            result = cast(
                "CursorResult[Any]",
                await session.execute(
                    delete(Fact).where(Fact.category == category, Fact.key == key)
                ),
            )
        removed = bool(result.rowcount)
        if removed:
            _log.info("fact_forgotten", category=category, key=key)
        return removed

    # -- activity (long-term, append-only) --------------------------------

    async def record_activity(
        self, intent: str, target: str | None, status: str, executed: bool, detail: str
    ) -> None:
        """Append an activity record.

        Args:
            intent: The intent acted on.
            target: What it acted on.
            status: Result status.
            executed: Whether a side effect occurred.
            detail: Human-readable outcome.
        """
        async with self._database.session() as session:
            session.add(
                ActivityRecord(
                    intent=intent,
                    target=target,
                    status=status,
                    executed=executed,
                    detail=detail,
                )
            )

    async def recent_activity(self, limit: int) -> list[ActivityEntry]:
        """Return the most recent actions.

        Args:
            limit: Maximum results.

        Returns:
            Actions, newest first.
        """
        async with self._database.session() as session:
            result = await session.execute(
                select(ActivityRecord).order_by(desc(ActivityRecord.created_at)).limit(limit)
            )
            return [
                ActivityEntry(
                    intent=row.intent,
                    target=row.target,
                    status=row.status,
                    executed=row.executed,
                    detail=row.detail,
                    created_at=row.created_at,
                )
                for row in result.scalars().all()
            ]
