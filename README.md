# Quainex

**Your Personal AI Operating System.**

Quainex is not a chatbot with a microphone attached. The goal is a system that
understands natural language, automates workflows, controls the machine it runs
on, and eventually manages a digital life — while staying privacy-conscious and
under the user's control.

> **Status: Phases 1–5 of 10 complete.**
> Quainex understands a spoken or typed request, classifies it into a typed
> intent, executes it on this machine behind two independent safety gates,
> answers out loud, and remembers the exchange. Remote control from a phone is
> Phase 6.

## Optional: voice

```powershell
pip install -e ".[voice]"      # Whisper + microphone capture
```

Kept optional deliberately — Whisper weights are a large first-run download, and
Quainex runs fine by text without them. Speech *output* needs no download at all,
so it works either way. `GET /voice/status` reports each component separately.

## What it can do today

```powershell
# Understand a request without acting on it
curl -X POST http://127.0.0.1:8000/brain/interpret `
  -H "Content-Type: application/json" `
  -d '{\"utterance\": \"Open VS Code\"}'

# Understand and act, in one call
curl -X POST http://127.0.0.1:8000/commands/ask `
  -H "Content-Type: application/json" `
  -d '{\"utterance\": \"take a screenshot\"}'
```

```powershell
# Speak to it (needs the voice extra), or post recognised text
curl -X POST http://127.0.0.1:8000/voice/turn `
  -H "Content-Type: application/json" `
  -d '{\"text\": \"Quainex, take a screenshot\"}'

# See what it remembers
curl http://127.0.0.1:8000/memory/activity
```

15 commands: open/close applications, open websites and folders, search files,
lock, sleep, restart, shut down, volume, brightness, screenshot, clipboard,
notifications, system info. `GET /commands` lists them.

It also **remembers** — recent turns (so "close it" resolves), your preferences,
facts you tell it, and an append-only record of everything it has done. All of it
is listable and deletable through `/memory`, except the audit trail, which is
deliberately read-only.

**Two switches guard anything dangerous**, and both must pass:

| Switch | Question it answers | Who decides |
|---|---|---|
| `requires_confirmation` | "Are you sure?" | you, per action |
| `QUAINEX_ALLOW_DESTRUCTIVE_COMMANDS` | "May Quainex ever do this?" | you, once, in `.env` |

Out of the box the second is `false`, so **Quainex cannot power off your machine
no matter what it is told**.

---

## Requirements

| Tool | Version | Notes |
|---|---|---|
| Python | 3.12+ | `winget install Python.Python.3.12` |
| Git | any recent | |
| Anthropic API key | optional | Only needed for AI features; Quainex boots without one |

---

## Setup

```powershell
git clone <your-repo-url> quainex
cd quainex

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # PowerShell
# .\.venv\Scripts\activate.bat        # cmd

pip install -e ".[dev]"

Copy-Item .env.example .env           # then edit .env
```

If PowerShell refuses to run the activation script, allow local scripts for
your user only:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Activating the venv is optional — every command below also works by calling
`.\.venv\Scripts\python.exe` directly.

---

## Running

```powershell
python main.py
```

Then:

- Health: <http://127.0.0.1:8000/health>
- API docs: <http://127.0.0.1:8000/docs> (development only)

With auto-reload during development:

```powershell
uvicorn quainex.api.app:create_app --factory --reload
```

### Trying the WebSocket

Paste this into any browser console with the server running:

```js
const ws = new WebSocket("ws://127.0.0.1:8000/ws");
ws.onmessage = (e) => console.log("<-", e.data);
ws.onopen = () => {
  ws.send(JSON.stringify({ type: "ping" }));
  ws.send(JSON.stringify({ type: "echo", data: "hello Quainex" }));
};
```

Frame protocol (Phase 1):

| Send | Receive |
|---|---|
| `{"type":"ping"}` | `{"type":"pong"}` |
| `{"type":"echo","data":X}` | `{"type":"echo","data":X}` |
| malformed / unknown | `{"type":"error","error":"..."}` |

---

## Development

```powershell
.\.venv\Scripts\python.exe -m pytest         # tests
.\.venv\Scripts\python.exe -m ruff check .   # lint
.\.venv\Scripts\python.exe -m ruff format .  # format
.\.venv\Scripts\python.exe -m mypy quainex main.py   # type check
```

All four must pass before a phase is considered complete.

The provider smoke test that calls the real API is skipped unless a key is
present, so a fresh clone runs green and free:

```powershell
$env:QUAINEX_ANTHROPIC_API_KEY = "sk-ant-..."
.\.venv\Scripts\python.exe -m pytest tests/test_provider.py
```

---

## Project layout

```
quainex/
├── main.py                  entrypoint (thin launcher)
├── pyproject.toml           deps + ruff/mypy/pytest config
├── .env.example             documented configuration template
│
├── quainex/
│   ├── config/settings.py   validated settings (pydantic-settings)
│   ├── core/
│   │   ├── container.py     DI composition root
│   │   ├── exceptions.py    error hierarchy
│   │   ├── logging.py       structlog -> console + rotating JSON
│   │   ├── brain/           Phase 2 — intent detection
│   │   ├── commands/        Phase 3 — command registry
│   │   ├── automation/      Phase 3 — desktop control
│   │   ├── voice/ speech/   Phase 4 — STT / TTS
│   │   └── memory/          Phase 5 — short & long-term memory
│   ├── api/
│   │   ├── app.py           application factory
│   │   ├── middleware.py    correlation IDs
│   │   ├── errors.py        error envelope + leak guard
│   │   ├── dependencies.py  FastAPI DI bridge
│   │   └── routes/          health, ws
│   ├── services/ai/         provider protocol + Anthropic implementation
│   ├── auth/ security/      Phase 6 — JWT, permissions, audit
│   ├── database/ models/    Phase 5 — persistence
│   ├── vision/              Phase 8 — OCR, screen understanding
│   ├── plugins/             Phase 9 — plugin system
│   ├── scheduler/           Phase 10 — task scheduling
│   └── integrations/ utils/
│
├── desktop/ dashboard/      Phase 6 — Tauri shell, React dashboard
├── docs/ scripts/ logs/ tests/
```

Empty packages are intentional: each carries a docstring naming its purpose and
target phase, so the structure documents the roadmap.

---

## Design decisions

**Application factory, not a module-level `app`.** A module-level application is
built on import, so importing anything would configure logging and construct an
API client as a side effect. `create_app(settings)` makes construction explicit
and lets tests build an app with isolated settings.

**Hand-rolled DI container, not a DI framework.** FastAPI already has `Depends`.
A second container library would add a framework to learn for no capability we
lack. `Container` is a dataclass that builds collaborators and closes them.

**Structured logging from day one.** Phase 6 exposes Quainex to a phone and
Phase 10 lets it act autonomously; both require an auditable record. Retrofitting
structure means rewriting every call site.

**A Protocol for AI providers, not an ABC.** The roadmap calls for multiple
providers including local models. A `Protocol` lets an implementation qualify by
shape alone, which also makes test doubles trivial.

**Health never fails on a dependency.** A missing API key reports
`ai.available: false` in the body rather than returning 503 — otherwise a
supervisor would restart the process in a loop over something restarting cannot
fix.

---

## Security notes

- `.env` is gitignored. The API key is held as a `SecretStr`, so it does not
  appear in logs, reprs, or tracebacks. The log pipeline additionally redacts any
  field whose key looks credential-shaped.
- **The default bind address is `127.0.0.1` and must stay that way until Phase 6
  adds authentication.** The WebSocket endpoint is currently unauthenticated;
  binding to `0.0.0.0` would expose it to the whole network.
- Error responses never contain tracebacks or internal paths. Each carries a
  correlation ID that ties it to the full record in `logs/quainex.log`.
- Interactive API docs are disabled when `QUAINEX_ENVIRONMENT=prod`.

---

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | Project foundation | **Complete** |
| 2 | Brain — intent detection | **Complete** |
| 3 | Desktop automation | **Complete** |
| 4 | Voice assistant | **Complete** |
| 5 | Memory engine | **Complete** |
| 6 | Phone remote control | Next |
| 7 | Developer assistant | Planned |
| 8 | Vision | Planned |
| 9 | Plugin marketplace | Planned |
| 10 | Autonomous agent | Planned |

See [docs/PHASE_1.md](docs/PHASE_1.md) for what Phase 1 delivered and what it
deliberately left out.
