"""Persistence models for the memory engine.

Purpose:
    Define the tables that let Quainex remember across restarts.

Why four tables rather than one "memories" table:
    The four things Quainex remembers have genuinely different shapes and
    lifetimes. Conversation turns are append-only and numerous; preferences are
    keyed and overwritten; facts are keyed and searchable; activity is an audit
    trail that must never be edited. Collapsing them into one table with a
    ``kind`` column would mean every query filters on ``kind``, every row carries
    columns irrelevant to it, and the uniqueness rule that makes preferences work
    ("one value per key") could not be expressed at all.

Architecture:
    MemoryStore
        |-- ConversationTurn   append-only dialogue history
        |-- Preference         key -> value, last write wins
        |-- Fact               key -> value + category, searchable
        +-- ActivityRecord     append-only audit of what Quainex did

Dependencies:
    sqlalchemy

Future improvements:
    * Add embeddings to ``Fact`` for semantic recall rather than substring match.
    * Partition ``ActivityRecord`` by month once it grows large.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utc_now() -> datetime:
    """Return the current UTC time.

    Timestamps are stored in UTC without exception: a personal assistant that
    survives a timezone change or a daylight-saving boundary must not have its
    history reorder itself.

    Returns:
        The current time, timezone-aware, in UTC.
    """
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Declarative base for every Quainex table."""


class ConversationTurn(Base):
    """One exchange in the ongoing dialogue.

    Attributes:
        id: Surrogate primary key.
        session_id: Groups turns belonging to one conversation.
        role: Who spoke — ``user`` or ``assistant``.
        content: What was said.
        intent: The classified intent, when the turn produced one.
        created_at: When the turn happened.
    """

    __tablename__ = "conversation_turns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    intent: Mapped[str | None] = mapped_column(String(64), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)

    # History is always read as "the last N turns of this conversation", so the
    # index matches that access pattern rather than either column alone.
    __table_args__ = (Index("ix_turns_session_created", "session_id", "created_at"),)


class Preference(Base):
    """A durable user preference.

    Attributes:
        id: Surrogate primary key.
        key: Stable identifier, e.g. ``preferred_browser``.
        value: The preference value, as text.
        updated_at: When it was last set.
    """

    __tablename__ = "preferences"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Unique: a preference with two values is not a preference. The database
    # enforces this rather than trusting every write path to check first.
    key: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    value: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utc_now, onupdate=_utc_now
    )


class Fact(Base):
    """Something Quainex was told and should recall later.

    Attributes:
        id: Surrogate primary key.
        category: Grouping such as ``project`` or ``folder``.
        key: Identifier, unique within its category.
        value: The remembered content.
        created_at: When it was first recorded.
        updated_at: When it was last changed.
    """

    __tablename__ = "facts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    category: Mapped[str] = mapped_column(String(64), index=True)
    key: Mapped[str] = mapped_column(String(128), index=True)
    value: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utc_now, onupdate=_utc_now
    )

    # Keys are unique per category, so "home" can mean one thing under `folder`
    # and another under `location` without collision.
    __table_args__ = (UniqueConstraint("category", "key", name="uq_fact_category_key"),)


class ActivityRecord(Base):
    """An append-only record of something Quainex did.

    Never updated or deleted by application code. An audit trail that can be
    rewritten is not an audit trail, and Phase 10's autonomous loop makes this
    the record of what the system chose to do unattended.

    Attributes:
        id: Surrogate primary key.
        intent: The intent acted on.
        target: What it acted on, when there was one.
        status: The command result status.
        executed: Whether a side effect actually occurred.
        detail: Human-readable outcome.
        created_at: When it happened.
    """

    __tablename__ = "activity"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    intent: Mapped[str] = mapped_column(String(64), index=True)
    target: Mapped[str | None] = mapped_column(String(512), default=None)
    status: Mapped[str] = mapped_column(String(32))
    executed: Mapped[bool] = mapped_column(default=False)
    detail: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utc_now, index=True
    )
