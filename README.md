# Quainex

**Your Personal AI Operating System.**

Quainex is not a chatbot with a microphone attached. The goal is a system that
understands natural language, automates workflows, controls the machine it runs
on, and eventually manages a digital life — while staying privacy-conscious and
under the user's control.

> **Status: Phases 1–10 of 10 complete.**
> Quainex understands a spoken or typed request, classifies it into a typed
> intent, executes it on this machine behind two independent safety gates,
> answers out loud, and remembers the exchange. It has a browser interface, takes
> orders from a phone over Telegram, runs goals autonomously within a budget, and
> loads capability-gated plugins.

## The interface

```powershell
python main.py
```

Then open <http://127.0.0.1:8000/ui/>. Four panels:

| Panel | What it is for |
|---|---|
| **Console** | Type or hold the mic. A reactive core shows what Quainex is doing, and anything dangerous opens a confirmation gate. |
| **History** | Every turn, plus the append-only activity trail. |
| **Commands** | The executor's own registry — if it is not listed, no phrasing will run it. |
| **Settings** | Paste API keys, see which provider is answering, and prove a key works. |

**You never have to edit a config file to get started.** Paste a free Groq or
Gemini key into Settings and it takes effect on the next request — no restart. The
key is encrypted with Windows DPAPI before it touches the disk, and nothing in
Quainex will ever read it back to you.

No build step, no `npm install`, no CDN. See [`dashboard/README.md`](dashboard/README.md).

## AI providers: free first

Quainex tries providers in order and the first one holding a key answers:

| Order | Provider | Cost | Notes |
|---|---|---|---|
| 1 | **Groq** | free | Fastest. Structured output is prompt-guided, not enforced. |
| 2 | **Gemini** | free | Native schema enforcement, and vision (screen + PDF questions). |
| 3 | **Anthropic** | paid | Strongest reasoning. The backstop, not the entry fee. |
| 4 | **Local** | free | Any OpenAI-compatible server — Ollama, LM Studio. Fully offline. |

Only *provider* failures fall through — a bad key, a rate limit, a missing
capability. **A refusal does not.** If a model declines a request, asking a
different model the same thing is shopping for a yes, not error recovery.

Reorder with `QUAINEX_AI_PROVIDERS`. An entry with no key is skipped, so an
unused slot costs nothing.

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

**Developer tools** (Phase 7): `git.status`, `git.commit`, `tests.run`,
`lint.run`, `types.check`, `docker.ps` and more — an allowlist of *complete
commands*, never a shell. Plus explain, review and generate for source files.

**Vision** (Phase 8): ask about what's on screen, or about a PDF. Window
enumeration is local and free, so "is VS Code open?" costs nothing.

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
| An AI key | optional | A **free** [Groq](https://console.groq.com/keys) or [Gemini](https://aistudio.google.com/apikey) key is enough. Quainex boots without any. |

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

- **Interface: <http://127.0.0.1:8000/ui/>** — a bare `/` redirects here
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
- **You cannot expose Quainex without a password.** Authentication is *derived*
  from the bind address rather than configured separately, so "reachable from the
  network with auth off" is not a configuration that exists — binding to anything
  other than loopback turns auth on, and startup fails without credentials.
  Run `python scripts/hash_password.py` to set them up.
- **Confirmation must be proved, not asserted.** An action needing approval comes
  back with a signed, single-use token bound to that exact action. A client
  cannot mint one, and a token issued for "close Spotify" will not shut the
  machine down.
- **No TLS yet.** Tokens cross the LAN in the clear — put a reverse proxy with a
  certificate in front before this leaves a trusted network, and do not
  port-forward it to the internet as it stands.
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
| 6 | Remote access and auth | **Complete** |
| 7 | Developer assistant | **Complete** |
| 8 | Vision | **Complete** |
| 9 | Plugin marketplace | Next |
| 10 | Autonomous agent | Planned |

See [docs/PHASE_1.md](docs/PHASE_1.md) for what Phase 1 delivered and what it
deliberately left out.
