"""Memory endpoints.

Purpose:
    Let the user inspect and correct what Quainex remembers.

Why "forget" is a first-class endpoint:
    A personal assistant that accumulates facts about someone and offers no way
    to see or delete them is a liability, not a feature. Everything Quainex
    stores is listable and removable through this router, and the audit trail is
    the one exception — deliberately, because a record of what the system did
    that the system can rewrite is worthless.

Architecture:
    GET    /memory/conversation      recent turns
    DELETE /memory/conversation      forget a conversation
    GET    /memory/preferences       list preferences
    PUT    /memory/preferences/{key} set one
    POST   /memory/facts             remember a fact
    GET    /memory/facts             search facts
    DELETE /memory/facts/{cat}/{key} forget one
    GET    /memory/activity          what Quainex has been doing

Dependencies:
    fastapi, quainex.core.memory

Future improvements:
    * Export everything as JSON, so the user can take their data elsewhere.
    * A retention setting that prunes activity older than N months.
"""

from __future__ import annotations

from fastapi import APIRouter, Query, status
from pydantic import BaseModel, Field

from quainex.api.dependencies import ContainerDep
from quainex.core.memory import (
    DEFAULT_SESSION_ID,
    ActivityEntry,
    FactRecord,
    PreferenceRecord,
    TurnRecord,
)

router = APIRouter(prefix="/memory", tags=["memory"])


class PreferenceBody(BaseModel):
    """Body of a preference write.

    Attributes:
        value: The value to store.
    """

    value: str


class FactBody(BaseModel):
    """Body of a fact write.

    Attributes:
        category: Grouping such as ``project`` or ``folder``.
        key: Identifier within the category.
        value: What to remember.
    """

    category: str = Field(min_length=1, examples=["project"])
    key: str = Field(min_length=1, examples=["quainex"])
    value: str = Field(min_length=1, examples=["C:/Users/G8/dev/quainex"])


@router.get("/conversation", summary="Recent conversation turns")
async def conversation(
    container: ContainerDep,
    session_id: str = DEFAULT_SESSION_ID,
    limit: int = Query(default=20, ge=1, le=200),
) -> list[TurnRecord]:
    """Return recent turns of a conversation.

    Args:
        container: Injected application container.
        session_id: Conversation to read.
        limit: Maximum turns to return.

    Returns:
        Turns, oldest first.
    """
    return await container.memory.recent_turns(session_id, limit)


@router.delete("/conversation", summary="Forget a conversation")
async def forget_conversation(
    container: ContainerDep, session_id: str = DEFAULT_SESSION_ID
) -> dict[str, int]:
    """Delete every turn of a conversation.

    Args:
        container: Injected application container.
        session_id: Conversation to clear.

    Returns:
        How many turns were removed.
    """
    return {"removed": await container.memory.clear_conversation(session_id)}


@router.get("/preferences", summary="List stored preferences")
async def preferences(container: ContainerDep) -> list[PreferenceRecord]:
    """Return every stored preference.

    Args:
        container: Injected application container.

    Returns:
        Preferences ordered by key.
    """
    return await container.memory.list_preferences()


@router.put(
    "/preferences/{key}",
    status_code=status.HTTP_200_OK,
    summary="Set a preference",
)
async def set_preference(key: str, body: PreferenceBody, container: ContainerDep) -> dict[str, str]:
    """Store a preference, replacing any existing value.

    Args:
        key: Preference identifier.
        body: The value to store.
        container: Injected application container.

    Returns:
        The stored key and value.
    """
    await container.memory.set_preference(key, body.value)
    return {"key": key, "value": body.value}


@router.post("/facts", status_code=status.HTTP_201_CREATED, summary="Remember a fact")
async def remember_fact(body: FactBody, container: ContainerDep) -> FactBody:
    """Record a fact, replacing any existing value for that key.

    Args:
        body: Category, key and value.
        container: Injected application container.

    Returns:
        What was stored.
    """
    await container.memory.remember(body.category, body.key, body.value)
    return body


@router.get("/facts", summary="Search remembered facts")
async def search_facts(
    container: ContainerDep,
    query: str = Query(default="", description="Substring to match in the key or value."),
    limit: int = Query(default=20, ge=1, le=200),
) -> list[FactRecord]:
    """Find facts matching a substring.

    Args:
        container: Injected application container.
        query: Text to look for; empty matches everything.
        limit: Maximum results.

    Returns:
        Matching facts, most recently updated first.
    """
    return await container.memory.search(query, limit)


@router.delete("/facts/{category}/{key}", summary="Forget a fact")
async def forget_fact(category: str, key: str, container: ContainerDep) -> dict[str, bool]:
    """Delete a fact.

    Args:
        category: Grouping to delete from.
        key: Identifier within the category.
        container: Injected application container.

    Returns:
        Whether anything was removed.
    """
    return {"forgotten": await container.memory.forget(category, key)}


@router.get("/activity", summary="What Quainex has been doing")
async def activity(
    container: ContainerDep,
    limit: int = Query(default=20, ge=1, le=200),
) -> list[ActivityEntry]:
    """Return the recent activity trail.

    Read-only by design: an audit trail the system can edit is not an audit
    trail. There is deliberately no endpoint to delete these.

    Args:
        container: Injected application container.
        limit: Maximum results.

    Returns:
        Actions, newest first.
    """
    return await container.memory.recent_activity(limit)
