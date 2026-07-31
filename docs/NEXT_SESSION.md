# Quainex — session handoff (continue here)

**Last updated:** 2026-07-31, after the clipboard fix (`ca220c6`).
**Repo:** `C:\Users\G8\dev\quainex` · GitHub `izzmhi/quainexai` (public) · branch `main`.

This file is the continuation point. The user paused (weekly usage limit) mid-way
through a planned 5-batch program. **Batches 1–3 are done and shipped; Batch 4 is
next; Batch 5 follows.** Read this, then continue with Batch 4.

---

## Where things stand

Quainex is a **Telegram-first personal AI OS for Windows**. The server is FastAPI +
uvicorn; commands classify **token-free** in `quainex/core/brain/fastpath.py` when
possible, else fall through a multi-provider AI chain (groq → gemini → openrouter →
anthropic → local). Everything is driven from Telegram (`quainex/integrations/telegram.py`).

- **~762 tests pass**, ruff + mypy clean across ~90 source files. Run:
  `./.venv/Scripts/python.exe -m pytest tests/ -q` (≈2 min),
  `./.venv/Scripts/python.exe -m ruff check quainex/ tests/`,
  `./.venv/Scripts/python.exe -m mypy quainex`.
- **47 commands** registered. Telegram has `/menu`, `/files`, `/help`, `/status`, `/start`.

### Done this program
- **Batch 1** (`8986a62`): single-instance lock (`quainex/core/single_instance.py`),
  startup heartbeat ("🟢 Quainex is online" ping), quick-action `/menu` (buttons).
- **Batch 2** (`6df8553`): phone→PC input — `SET_CLIPBOARD` (copy to PC), `TYPE_TEXT`
  (SendInput unicode, no final Enter), `DOWNLOAD_URL` (streamed, 50 MB cap, contained).
- **Batch 3** (`1a54c6b`): `/files` button-driven folder browser (navigate, tap a file
  to receive), `SEND_FOLDER` (zip-and-send). Browse uses a bounded token map
  (`_browse_paths`) because Telegram callback data is capped at 64 bytes.
- **Clipboard fix** (`ca220c6`): clipboard *writes* over Telegram are never blocked
  (only *reads* are, and only by default — see `QUAINEX_TELEGRAM_ALLOW_CLIPBOARD_READ`).
  Enforcement is now `TelegramBridge._blocked_reason()`, not a flat membership test.

### Earlier foundation (before this program)
Production autostart with `reload` off by default (`QUAINEX_RELOAD` to opt in),
inbound file saving (`save_incoming_file`), rich `/status`, typing indicator,
HTML `/help`, browser control (Playwright/Edge, sync API on a worker thread).

---

## Working agreement (follow these)

- **Per-batch delivery:** one solid, tested, committed batch at a time. Get approval
  before starting the *next* batch. Each batch: what → why → architecture → code →
  tests → how-to-run → recommended improvements. See [[quainex-phase-workflow]].
- **Ship quality:** ruff + mypy clean and tests green before every commit. Commit
  messages end with the `Co-Authored-By: Claude Opus 4.8` trailer. Push after commit.
- **Security posture (must persist):** only the allowlisted Telegram user id is obeyed;
  every filesystem path is canonicalised then **contained to `command_search_roots`**
  (`WindowsDesktopController._contain`); no `shell=True`, model output is never a shell
  arg; screenshots/webcam/file-sends are opt-in disclosures to a non-e2e third party.
- **Safe server restart** (learned the hard way — see [[quainex-server-ops]]):
  kill by matching `main.py` (NOT `quainex`, which also kills pytest/the venv), delete
  `%TEMP%\quainex-8000.lock`, **wait ~27 s** for Telegram to release the long-poll
  (409 is terminal), then `nohup ./.venv/Scripts/python.exe main.py`. Every restart
  sends the user a heartbeat ping — expected.
- **Telegram formatting:** bridge-composed messages use HTML with `_esc()` on dynamic
  values and a plain-text fallback in `_send`; arbitrary command output stays plain.
- **Token-free classification:** add fast-path patterns in `fastpath.py`. Content-
  bearing commands (clipboard/type/url) match the **raw** utterance via
  `_classify_carried_content` so payload case/punctuation/length survive.

---

## Batch 4 — Memory: remember preferences & facts  (DO THIS NEXT)

**Goal:** "remember my documents go to D:\Files", "remember my work laptop is …" —
persistent personal facts that survive restarts and shape future answers.

There is already a `MemoryManager` (`quainex/core/memory/`) used for conversation
context and `remember_exchange`; build on it, don't replace it. Check what it exposes
and how it persists (SQLite via `quainex/database/`).

**Suggested design**
- New intents: `REMEMBER_FACT` ("remember that/my X is Y", "note that …") and
  `RECALL_FACTS` ("what do you remember", "what are my notes"), plus maybe
  `FORGET_FACT`. Add to `schemas.py` `IntentType` + `INTENT_DESCRIPTIONS`, and
  fast-path patterns in `fastpath.py` (content-bearing → match raw utterance).
- Persistence: a small `facts` table (or reuse memory storage) — `key`/`value`/
  `created_at`, per nothing-fancy. A fact is freeform text; optionally parse
  "X is Y" into key/value but storing the whole sentence is fine and robust.
- Handlers in `builtin.py`: `remember_fact` stores; `recall_facts` returns the list;
  `forget_fact` deletes by match. `has_side_effect=True` for remember/forget.
- **Feed facts into the brain**: the valuable part. Inject remembered facts into the
  system prompt / context the AI provider sees (via `Brain`/`MemoryManager`), so
  "where do my documents go" is answered from memory. Keep it bounded (cap count /
  chars) and token-aware.
- Telegram: not blocked (facts are the user's own, going to their own chat). Add to
  `/help`. Consider a `/remember` or surfacing in `/status`.
- Tests: classification, store/recall/forget round-trip against a fake store, and
  that a remembered fact reaches the brain's context.

**Watch:** don't store secrets the user wouldn't want in the DB unprompted; this is
explicit ("remember …") so it's fine, but don't auto-harvest. Keep the fact store
inside the repo/data dir, not world-readable surprises.

---

## Batch 5 — Scheduling & triggers  (after Batch 4)

**Goal:** timed + recurring actions and condition triggers — the headline feature.
"every night at 11 lock the screen", "remind me at 3pm to …", "in 20 minutes shut
down", "when battery < 20% tell me".

This is a **subsystem**, so give it its own batch:
- A persistent schedule store (SQLite) surviving restarts; load on startup.
- A background scheduler loop (asyncio task started in the container lifespan, like
  the Telegram autostart) that wakes on due jobs and fires the intent through the
  executor, delivering results/notifications to Telegram.
- Intents: `SCHEDULE_TASK` (one-shot + recurring), `LIST_SCHEDULES`, `CANCEL_SCHEDULE`.
  Parse time expressions ("in 20 min", "at 3pm", "every night at 11"). A small parser
  or a lightweight dep; keep it token-free where feasible, model fallback for fuzzy.
- Condition triggers (battery/disk/new-file) are a stretch — a periodic evaluator that
  checks `system_info()` / a watched folder and fires when a threshold crosses.
- Telegram: `/schedule` or natural language; results pushed proactively (reuse the
  outbound `_send`). Respect the single-instance model (one scheduler).
- Tests: parsing, persistence + reload, a due job fires exactly once, cancel works.

---

## Two loose ends the user raised

1. **Clipboard** — fixed (`ca220c6`). Writes work; reads need
   `QUAINEX_TELEGRAM_ALLOW_CLIPBOARD_READ=true` to enable.
2. **"Voice input button turned to camera" in Telegram** — this is the **Telegram
   client's own** mic⇄video toggle, not something the bot controls. Tell the user to
   **tap the round button** (currently a camera) once to switch it back to the
   microphone, then press-and-hold to record a voice note. Quainex already transcribes
   voice notes (`_handle_voice` → Whisper). No code change possible/needed bot-side.

---

## Quick orientation for the next session
- Bridge: `quainex/integrations/telegram.py` (handlers, `/menu`, `/files`, block logic).
- Intents/patterns: `quainex/core/brain/schemas.py`, `quainex/core/brain/fastpath.py`.
- Commands: `quainex/core/commands/builtin.py` (handlers + registration).
- OS actions + containment: `quainex/core/automation/windows.py` (+ `desktop.py` Protocol).
- Settings: `quainex/config/settings.py`. Tests mirror each module under `tests/`.
