# Phase 5 — Memory Engine

## Goal

Let Quainex remember: what was just said, what the user prefers, what it has been
told, and what it has done.

## Architecture

```
   VoiceSession / routes
        │
        ▼
   MemoryManager ──── remember_exchange()      one call, three writes
        │        └─── conversation_context()   -> list[ChatMessage] -> Brain
        ▼
   MemoryStore (Protocol)
        └── SqlAlchemyMemoryStore
                 ▼
   Database.session()  ──▶ SQLite (async, aiosqlite)
        ├── conversation_turns   append-only    short-term
        ├── preferences          key -> value   long-term
        ├── facts                keyed, searchable
        └── activity             append-only audit
```

## What "short-term" and "long-term" actually mean here

Not two copies of the same data. Different **scopes**:

- **Short-term** — the current conversation's recent turns, scoped by session and
  read as a bounded window. This is what gives "close it" a referent.
- **Long-term** — preferences, facts and activity. Not tied to a conversation,
  survives restarts, and is what makes Quainex feel like it knows the user.

An in-process cache of recent turns was considered and rejected: a local SQLite
read is well under a millisecond, and two copies of conversation state is two
things that can disagree.

## Why four tables, not one `memories` table

The four things Quainex remembers have genuinely different shapes and lifetimes.
A single table with a `kind` column would mean every query filters on `kind`,
every row carries irrelevant columns, and — decisively — the rule that makes
preferences work ("one value per key") could not be expressed as a constraint at
all. It is enforced by the database here, not by hoping every write path checks
first.

`facts` is unique on `(category, key)`, so `home` can mean one thing under
`folder` and another under `location` without collision.

## Bounded context, flat cost

`conversation_context()` returns at most `QUAINEX_MEMORY_CONTEXT_TURNS` (default
6) turns. Token cost per request therefore stays flat however long a session
runs — a conversation that grows unboundedly would otherwise get slower and more
expensive with every exchange.

Turns are queried newest-first (to use the index and apply the limit) then
reversed in Python, because a conversation replayed newest-first reads as
nonsense to a model.

## `remember_exchange` is one call on purpose

Recording a turn means three writes: the user's utterance, the assistant's reply,
and an activity record. Left to callers, the voice loop, the HTTP route and Phase
10's autonomous loop would each implement that sequence, and they would drift.

## The user can see and delete everything

Every category is listable and removable through `/memory` — with one deliberate
exception. **Activity has no delete endpoint.** An audit trail the system can
rewrite is worthless, and Phase 10 makes this the record of what Quainex chose to
do unattended.

## API

| Endpoint | Purpose |
|---|---|
| `GET/DELETE /memory/conversation` | Read or forget a conversation |
| `GET /memory/preferences`, `PUT /memory/preferences/{key}` | Preferences |
| `POST/GET /memory/facts`, `DELETE /memory/facts/{cat}/{key}` | Facts |
| `GET /memory/activity` | What Quainex has been doing (read-only) |

## Verification

| Check | Result |
|---|---|
| `ruff` / `mypy` (strict) | Clean, 56 files |
| `pytest` | **22 memory tests**, 186 total, all passing |

These run against a **real SQLite database** in a temp directory, not a mock. The
store is thin enough that mocking SQLAlchemy would only test the mock; what is
worth verifying is that the schema, the upserts and the ordering behave — and one
test restarts the engine over the same file to prove memories actually survive.

## Known gaps

1. **No migrations.** `create_all` is additive only — it creates missing tables
   and cannot alter existing columns. Alembic is needed before the schema
   changes in an incompatible way.
2. **Fact search is substring matching.** "Where do I keep the tax stuff" will
   not match a fact recorded as `documents/finance`; that needs embeddings.
3. **Facts are not yet surfaced to the Brain.** Memory can answer "what is my
   project path" but the Brain does not consult facts while classifying, so
   "open my project" does not resolve against what Quainex knows. This is the
   highest-value next step.
4. **No retention policy.** Activity grows without bound.
5. **The `/commands/ask` route does not record to memory** — only the voice loop
   does. That asymmetry should be closed.
