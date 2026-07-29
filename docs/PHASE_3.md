# Phase 3 — Desktop Automation

## Goal

Give Quainex the ability to act on the machine — and make it hard for it to act
wrongly. This is the first phase with real side effects, so most of the design
effort went into the paths that *refuse*.

## Architecture

```
   Intent (from Phase 2)
        │
        ▼
   CommandExecutor.execute(intent, confirmed=?)
        │
        ├── gate 1  registered?        no → UNSUPPORTED   executed=false
        ├── gate 2  confirmed?         no → REQUIRES_CONFIRMATION
        ├── gate 3  operator allows?   no → BLOCKED
        ├── gate 4  target present?    no → FAILED
        │
        ▼  all gates passed
   Command.handler(desktop, intent)
        │
        ▼
   DesktopController  ◄── Protocol boundary
        ├── WindowsDesktopController   (real: ctypes / psutil / Pillow / subprocess)
        └── FakeDesktopController      (tests: records, never acts)
```

## Two independent safety switches

They answer different questions and both must pass:

| Switch | Question | Who answers | Where |
|---|---|---|---|
| `requires_confirmation` | "Are you sure?" | the **user**, per action | set by the Brain (Phase 2), enforced here |
| `allow_destructive_commands` | "May Quainex ever do this?" | the **operator**, once | `.env`, default **off** |

So powering off the machine requires the Brain to flag it, the user to confirm
it, *and* the operator to have enabled power actions at all. The default
configuration cannot shut your machine down no matter what anyone says to it.

## Why the gates live in one place

Fifteen handlers each remembering to check confirmation is fifteen chances to
forget, and the one that gets forgotten will be the dangerous one. Every path to
a side effect runs through `execute()`. A new command added in Phase 7 inherits
all four gates without its author doing anything.

The **order** matters too: every gate that can refuse without a side effect runs
before the one that causes them, and the user-facing gate (confirmation) is
checked before the operator gate, so the message is the actionable "Confirm:
shutdown?" rather than a configuration lecture.

## Why `DesktopController` is a Protocol

Not primarily for macOS later — for testing now. Every command in this phase is
a real side effect. Without a substitutable boundary, the test suite would kill
processes and power off the machine that runs it. `FakeDesktopController` records
what it was asked to do; **122 tests run with zero side effects.**

## Where untrusted input is stopped

`Intent.target` is model output derived from speech. Three places convert it into
something the OS acts on, and each is closed:

**Application names → allowlist.** `applications.py` maps spoken names onto a
fixed catalogue of executables. `"cmd /c del /f /s /q C:\"` does not resolve, so
it is refused — a name that looks like a shell command is just a name that is not
in the catalogue. The refusal lists what *is* available.

**Paths → canonicalise, then contain.** `_resolve_within_roots` calls `resolve()`
*before* the containment check. In the other order, `~/../../Windows/System32`
passes a prefix test.

**URLs → scheme allowlist, parsed before rewriting.** See the bug below.

Plus two invariants across the module: **no `shell=True` anywhere**, and system
executables resolve to absolute paths under `%SystemRoot%\System32` rather than
being found on `PATH` — otherwise anything that can write an earlier `PATH` entry
can shadow `powershell.exe`.

## Bug found by the tests

The first version of `open_url` upgraded bare domains like this:

```python
if "://" not in candidate:
    candidate = f"https://{candidate}"  # then check the scheme
```

`javascript:alert(1)` contains no `://`. It became `https://javascript:alert(1)`,
which parses with scheme `https` and a non-empty netloc — so it **passed** the
scheme check and was handed to the browser. `ms-settings:privacy` and the string
`not a url` did the same.

The convenience rewrite ran *before* validation and laundered the input. Fixed by
parsing first: an explicit scheme must be `http`/`https`, and only a string that
is already a syntactically valid hostname gets upgraded. Three tests now pin this.

This is a good argument for writing the hostile-input tests before trusting the
implementation — the happy path worked perfectly the whole time.

## What Quainex can now do

15 commands: open/close applications, open websites and folders, search files,
lock, sleep, restart, shut down, volume, brightness, screenshot, clipboard
read/write, notifications, and system info.

## API

| Endpoint | Purpose |
|---|---|
| `GET /commands` | What Quainex can do |
| `POST /commands/execute` | Act on an already-classified intent |
| `POST /commands/ask` | Interpret and execute in one call |

A refusal is a **200 with `executed: false`** and an explanation, not an error —
the caller asked a legitimate question and got a definite answer. Only genuine
faults produce error responses.

`/execute` and `/ask` both exist so the convenient path isn't the only path: a
voice loop that wants to confirm mid-flow interprets first, shows the user, then
executes.

## Verification

| Check | Result |
|---|---|
| `ruff check .` | All checks passed |
| `mypy quainex main.py` (strict) | No issues, 46 files |
| `pytest` | **122 passed, 1 skipped** (+57 from Phase 2) |

## Known gaps

1. **`confirmed: true` is asserted by the caller.** Nothing proves the user was
   actually asked. Fine on localhost; Phase 6 must issue signed confirmation
   tokens before this API is reachable from a phone. **This is the most important
   thing to fix before the bind address changes.**
2. **Exact volume levels are unsupported.** Media-key events can only nudge;
   setting "volume to 40" needs the Core Audio API (pycaw).
3. **Brightness needs WMI support.** Many desktop monitors do not expose it, and
   the command reports that rather than silently doing nothing.
4. **File search is a filesystem walk.** Fine for a home directory, slow across a
   large disk. The Windows Search index would be much faster.
5. **No rate limiting.** Phase 10's autonomous loop could execute in a tight
   cycle; per-intent cooldowns should land before then.
6. **The app catalogue is hardcoded.** Users cannot register their own
   applications without editing source.
