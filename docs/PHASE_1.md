# Phase 1 — Project Foundation

## Goal

A clean, production-shaped foundation that Phases 2–10 extend without rework.
Nothing user-facing ships in this phase; the deliverable is the ground the rest
of Quainex stands on.

## Architecture

```
                        ┌──────────────────────────────┐
   HTTP / WebSocket ────▶  quainex/api                  │
                        │   ├── middleware.py           │  correlation IDs
                        │   ├── errors.py               │  envelope + leak guard
                        │   ├── dependencies.py         │  Depends bridge
                        │   └── routes/{health,ws}.py   │
                        └───────────────┬───────────────┘
                                        │ Depends(get_container)
                        ┌───────────────▼───────────────┐
                        │  quainex/core/container.py    │  composition root
                        └───────────────┬───────────────┘
                ┌───────────────────────┼───────────────────────┐
                ▼                       ▼                       ▼
   config/settings.py          core/logging.py       services/ai/provider.py
   (pydantic-settings)         (structlog)           (Protocol)
                │                       │                       │
                │                       │                       ▼
             .env                 logs/quainex.log    anthropic_provider.py
        (gitignored)              (rotating JSON)     (AsyncAnthropic)
```

Startup and shutdown are driven by the FastAPI lifespan: it builds the
`Container`, records the start time, and closes the container on exit.

## What was built

| Area | Delivered |
|---|---|
| Configuration | `Settings` with typed enums, range-validated port, `SecretStr` API key, unknown-variable rejection, production invariants |
| Logging | structlog → readable console (dev) + rotating JSON file (always); credential redaction; uvicorn routed through the same pipeline |
| Errors | `QuainexError` hierarchy; one JSON envelope for known failures, `HTTPException`s and unexpected crashes |
| DI | `Container` composition root, exposed to routes via `ContainerDep` |
| HTTP | `GET /health` reporting version, uptime and subsystem state |
| WebSocket | `WS /ws` with a `ConnectionManager`, typed frame protocol, size guard |
| Observability | Correlation ID per request, bound into every log record and echoed in the response header |
| AI | `AIProvider` protocol + Anthropic implementation with refusal handling and error translation |
| Tests | 44 tests covering settings, health, WebSocket frames, error handling and the provider |

## Verification results

| Check | Result |
|---|---|
| `ruff check .` | All checks passed |
| `ruff format --check .` | 41 files already formatted |
| `mypy quainex main.py` | Success — no issues in 35 source files (strict mode) |
| `pytest` | 43 passed, 1 skipped (live API test, opt-in) |
| Server boot | `GET /health` → 200 with correct body; `/docs` → 200 |
| Log format | 8/8 lines valid JSON; zero credential-shaped strings |

## Decisions and trade-offs

**Application factory over a module-level `app`.**
Cost: one extra function call, and `uvicorn` needs `--factory`.
Benefit: importing the package has no side effects, and tests can build an app
with isolated settings. Worth it — the alternative makes test isolation
impossible.

**No DI framework.**
Considered `dependency-injector`. Rejected: FastAPI's `Depends` already covers
request-scoped injection, and a `Container` dataclass covers process-scoped
objects. Adding a framework would introduce a second lifecycle to reconcile for
no capability gained.

**No database in Phase 1.**
The folders exist; the code does not. An ORM with no entities is dead weight that
still has to be maintained and upgraded. It lands in Phase 5, where memory
actually needs persistence.

**Structured logs before there is anything to audit.**
Costs a dependency and slightly noisier call sites now. Retrofitting later means
rewriting every logging call in the codebase, and the earliest incidents — the
ones on unfamiliar code — would be the ones without usable traces.

**Health reports subsystem state instead of failing on it.**
A 503 from a missing API key would cause restart loops without fixing anything.
Liveness and dependency-readiness are different questions; Phase 6 adds a
separate `/ready` for the second one.

## Deliberately out of scope

Intent parsing (Phase 2), desktop automation (Phase 3), voice (Phase 4),
persistence (Phase 5), authentication (Phase 6), Docker, the React dashboard and
the Tauri shell. Folders are created; implementations are not.

## Known gaps to address later

1. **The WebSocket endpoint is unauthenticated.** Acceptable only while bound to
   `127.0.0.1`. Phase 6 must add a handshake token *before* the bind address
   changes — this is the single largest security risk in the current design.
2. **`ConnectionManager` is a module-level singleton.** Fine for one process;
   it must move into the `Container` if Quainex ever runs multi-process.
3. **No rate limiting** on either HTTP or WebSocket.
4. **No `/ready` endpoint** distinguishing liveness from dependency readiness.
5. **`starlette.testclient` deprecation warning** — the pinned Starlette suggests
   `httpx2`. Harmless today; revisit when it becomes an error.

## Recommended next steps

Phase 2 (Brain) should need no structural change:

- Add `quainex/core/brain/schemas.py` with the `Intent` model
  (`intent`, `target`, `confidence`, `parameters`).
- Add `quainex/core/brain/brain.py` taking an `AIProvider` in its constructor and
  calling `provider.parse(output_model=Intent, ...)` — the schema-constrained
  path already implemented and tested here.
- Register `brain` as a `Container` field.
- Add `POST /brain/interpret` returning the parsed intent.
- Consider `QUAINEX_AI_EFFORT=low` for routing: intent classification is a small,
  well-scoped task and effort is the main latency/cost lever.
