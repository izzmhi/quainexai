"""Tests for the memory engine: persistence, scoping, and the endpoints.

Every test runs against a real SQLite database in a temporary directory, not a
fake. The store is thin enough that mocking SQLAlchemy would test the mock; what
is worth testing is that the schema, the upserts and the ordering actually behave
— and that only exercises against a real database.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from quainex.config.settings import Settings
from quainex.core.brain import Intent, IntentType
from quainex.core.commands import CommandResult, CommandStatus
from quainex.core.memory import MemoryManager, SqlAlchemyMemoryStore
from quainex.database.engine import Database

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fastapi.testclient import TestClient


@pytest.fixture
async def memory(tmp_path: Path) -> AsyncIterator[MemoryManager]:
    """A memory manager backed by a fresh database."""
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        log_dir=tmp_path / "logs",
        database_path=tmp_path / "memory.db",
        memory_context_turns=6,
    )
    database = Database.create(settings)
    await database.create_schema()
    try:
        yield MemoryManager(SqlAlchemyMemoryStore(database), settings)
    finally:
        await database.aclose()


def _intent(
    intent: IntentType = IntentType.OPEN_APPLICATION, target: str | None = "VS Code"
) -> Intent:
    return Intent(
        intent=intent,
        target=target,
        confidence=0.95,
        reasoning="test",
        requires_confirmation=False,
    )


def _result(message: str = "Opened Visual Studio Code.") -> CommandResult:
    return CommandResult(
        status=CommandStatus.SUCCEEDED,
        intent="open_application",
        message=message,
        executed=True,
    )


# -- conversation (short-term) --------------------------------------------


async def test_turns_are_returned_oldest_first(memory: MemoryManager):
    for index in range(3):
        await memory.remember_exchange(f"utterance {index}", _intent(), _result(f"reply {index}"))

    turns = await memory.recent_turns(limit=10)

    # A conversation replayed newest-first would read as nonsense to the model.
    assert [t.content for t in turns] == [
        "utterance 0",
        "reply 0",
        "utterance 1",
        "reply 1",
        "utterance 2",
        "reply 2",
    ]


async def test_context_window_is_bounded(memory: MemoryManager):
    for index in range(20):
        await memory.remember_exchange(f"utterance {index}", _intent(), _result())

    context = await memory.conversation_context()

    # Bounded so token cost stays flat however long a session runs.
    assert len(context) == 6
    assert context[-1].content == _result().message


async def test_context_uses_chat_roles(memory: MemoryManager):
    await memory.remember_exchange("open vs code", _intent(), _result())
    context = await memory.conversation_context()

    assert [m.role for m in context] == ["user", "assistant"]


async def test_conversations_are_isolated_by_session(memory: MemoryManager):
    await memory.remember_exchange("work thing", _intent(), _result(), session_id="work")
    await memory.remember_exchange("home thing", _intent(), _result(), session_id="home")

    work = await memory.recent_turns("work", limit=10)
    home = await memory.recent_turns("home", limit=10)

    assert "work thing" in [t.content for t in work]
    assert "work thing" not in [t.content for t in home]


async def test_unclassified_utterance_is_still_recorded(memory: MemoryManager):
    # A turn Quainex failed to understand is exactly the one worth keeping.
    await memory.remember_exchange("mumble mumble", intent=None, result=None)

    turns = await memory.recent_turns(limit=10)
    assert len(turns) == 1
    assert turns[0].content == "mumble mumble"
    assert turns[0].intent is None


async def test_clearing_a_conversation_removes_only_that_one(memory: MemoryManager):
    await memory.remember_exchange("keep me", _intent(), _result(), session_id="keep")
    await memory.remember_exchange("drop me", _intent(), _result(), session_id="drop")

    removed = await memory.clear_conversation("drop")

    assert removed == 2
    assert await memory.recent_turns("drop", limit=10) == []
    assert len(await memory.recent_turns("keep", limit=10)) == 2


# -- preferences (long-term) ----------------------------------------------


async def test_preferences_round_trip(memory: MemoryManager):
    await memory.set_preference("preferred_browser", "firefox")
    assert await memory.get_preference("preferred_browser") == "firefox"


async def test_setting_a_preference_twice_overwrites(memory: MemoryManager):
    await memory.set_preference("preferred_browser", "firefox")
    await memory.set_preference("preferred_browser", "chrome")

    assert await memory.get_preference("preferred_browser") == "chrome"
    # A preference with two values is not a preference.
    assert len(await memory.list_preferences()) == 1


async def test_missing_preference_is_none(memory: MemoryManager):
    assert await memory.get_preference("never_set") is None


# -- facts (long-term) ----------------------------------------------------


async def test_facts_round_trip(memory: MemoryManager):
    await memory.remember("project", "quainex", "C:/Users/G8/dev/quainex")
    assert await memory.recall("project", "quainex") == "C:/Users/G8/dev/quainex"


async def test_the_same_key_in_two_categories_does_not_collide(memory: MemoryManager):
    await memory.remember("folder", "home", "C:/Users/G8")
    await memory.remember("location", "home", "London")

    assert await memory.recall("folder", "home") == "C:/Users/G8"
    assert await memory.recall("location", "home") == "London"


async def test_remembering_the_same_fact_twice_updates_it(memory: MemoryManager):
    await memory.remember("project", "quainex", "old path")
    await memory.remember("project", "quainex", "new path")

    assert await memory.recall("project", "quainex") == "new path"
    assert len(await memory.search("quainex")) == 1


async def test_facts_are_searchable_by_key_and_value(memory: MemoryManager):
    await memory.remember("project", "quainex", "an AI operating system")
    await memory.remember("project", "myapp", "a flutter application")

    assert len(await memory.search("quainex")) == 1
    assert len(await memory.search("flutter")) == 1
    assert len(await memory.search("nothing matches this")) == 0


async def test_forgetting_a_fact(memory: MemoryManager):
    await memory.remember("project", "quainex", "path")

    assert await memory.forget("project", "quainex") is True
    assert await memory.recall("project", "quainex") is None
    # Forgetting something already gone is not an error.
    assert await memory.forget("project", "quainex") is False


# -- activity (long-term, append-only) ------------------------------------


async def test_activity_is_recorded_for_executed_commands(memory: MemoryManager):
    await memory.remember_exchange("open vs code", _intent(), _result())

    activity = await memory.recent_activity()
    assert len(activity) == 1
    assert activity[0].intent == "open_application"
    assert activity[0].executed is True


async def test_refusals_are_recorded_too(memory: MemoryManager):
    # What Quainex declined to do is as much part of the trail as what it did.
    refused = CommandResult(
        status=CommandStatus.REQUIRES_CONFIRMATION,
        intent="shutdown",
        message="Confirm: shutdown?",
        executed=False,
    )
    await memory.remember_exchange("shut down", _intent(IntentType.SHUTDOWN, None), refused)

    activity = await memory.recent_activity()
    assert activity[0].status == "requires_confirmation"
    assert activity[0].executed is False


async def test_activity_is_newest_first(memory: MemoryManager):
    for index in range(3):
        await memory.remember_exchange(f"turn {index}", _intent(), _result(f"reply {index}"))

    activity = await memory.recent_activity()
    assert activity[0].detail == "reply 2"


# -- persistence across restarts ------------------------------------------


async def test_memories_survive_a_restart(tmp_path: Path):
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        log_dir=tmp_path / "logs",
        database_path=tmp_path / "persist.db",
    )

    first = Database.create(settings)
    await first.create_schema()
    await MemoryManager(SqlAlchemyMemoryStore(first), settings).remember(
        "project", "quainex", "still here"
    )
    await first.aclose()

    # A new engine over the same file: this is what a restart looks like.
    second = Database.create(settings)
    await second.create_schema()
    recalled = await MemoryManager(SqlAlchemyMemoryStore(second), settings).recall(
        "project", "quainex"
    )
    await second.aclose()

    assert recalled == "still here"


# -- HTTP endpoints --------------------------------------------------------


def test_preference_endpoints(client: TestClient):
    assert client.put("/memory/preferences/theme", json={"value": "dark"}).status_code == 200

    body = client.get("/memory/preferences").json()
    assert body[0]["key"] == "theme"
    assert body[0]["value"] == "dark"


def test_fact_endpoints(client: TestClient):
    created = client.post(
        "/memory/facts",
        json={"category": "project", "key": "quainex", "value": "an AI OS"},
    )
    assert created.status_code == 201

    found = client.get("/memory/facts", params={"query": "AI OS"}).json()
    assert len(found) == 1
    assert found[0]["key"] == "quainex"

    deleted = client.delete("/memory/facts/project/quainex").json()
    assert deleted["forgotten"] is True
    assert client.get("/memory/facts", params={"query": "AI OS"}).json() == []


def test_conversation_endpoints(client: TestClient):
    assert client.get("/memory/conversation").json() == []
    assert client.delete("/memory/conversation").json() == {"removed": 0}


def test_activity_endpoint_is_read_only(client: TestClient):
    assert client.get("/memory/activity").status_code == 200
    # There is deliberately no way to delete the audit trail.
    assert client.delete("/memory/activity").status_code == 405
