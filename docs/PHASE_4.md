# Phase 4 — Voice Assistant

## Goal

Say something, have it understood, have it done, hear the answer — without voice
becoming a prerequisite for Quainex working at all.

## Architecture

```
   microphone ──▶ MicrophoneRecorder     stop on trailing silence
              ──▶ FasterWhisperSTT       lazy model, offline once cached
              ──▶ wake-word gate  ──── not addressed? STOP HERE
              ──▶ Brain.interpret()     (Phase 2)
              ──▶ CommandExecutor       (Phase 3, all four gates intact)
              ──▶ WindowsSapiTTS        speak the result
              ──▶ VoiceTurn             the whole trace
```

## The decision that shaped this phase

**Voice is optional, and its absence is not a startup failure.**

Whisper weights are hundreds of megabytes fetched on first use, and this machine
has a documented history of stalling large CDN transfers. So:

- Voice dependencies are an **extra** (`pip install -e ".[voice]"`), not a core
  requirement.
- The model loads **lazily** — constructing the engine downloads nothing.
- `is_available` reports per component. Speech *output* needs no download at all,
  so Quainex can talk even when it cannot listen.

`GET /voice/status` reports each component separately rather than one flag,
because that is genuinely the state of the system.

## Why Windows SAPI for speech output

`pyttsx3` and similar wrap the same Windows engine PowerShell can reach directly.
The package buys an extra dependency and a COM-initialisation quirk in async
contexts, and nothing else. SAPI ships with the OS, works offline, and speaks
immediately.

Spoken text is model output and is **never interpolated into the PowerShell
script**. A fixed script reads the text from a temp file we control, so no
quantity of quotes or `$(...)` in a response can escape into the shell.

## Why faster-whisper over openai-whisper

Same model, CTranslate2 instead of PyTorch: a ~40 MB dependency instead of
~2.5 GB, and roughly 4x faster on CPU. On a machine that has to stay responsive
while the user works, that is the difference between usable and not.

## The wake word: what the tests taught

The gate stops unaddressed speech **before the Brain is called** — no API spend,
and nothing the room happens to say can reach the command executor.

Matching is fuzzy because recognisers mangle unusual proper nouns. The
interesting part was calibrating it:

| Word | Score vs "quainex" | What it is |
|---|---|---|
| `kwainex` | 0.714 | a genuine mishearing ("kw" and "qu" are the same sound) |
| `equinox` | **0.714** | an ordinary English word |

**Identical scores.** Lowering the threshold to accept the first accepts the
second — a false-positive test caught this within a minute of trying it. Edit
distance simply cannot separate these two cases, so no threshold value is
correct.

The fix was phonetic folding (`kw` → `qu`) applied before comparison. `kwainex`
becomes an exact match; `equinox` stays exactly where it was, below the bar. The
threshold stayed at 0.75. Eleven near-neighbour words are pinned by tests.

## Voice cannot bypass safety

A spoken "shut down the PC" goes through the identical Phase 3 gates. There is no
voice-specific execution path — `VoiceSession` calls the same
`CommandExecutor.execute()`, so confirmation and the destructive-command switch
both still apply. A test asserts this directly.

## Privacy

Recordings are written to a temp directory and deleted after transcription unless
`QUAINEX_KEEP_RECORDINGS=true`. Audio of a room is not something to accumulate
on disk without being asked. Transcript *text* is never written to the log
either — only its length.

## API

| Endpoint | Purpose |
|---|---|
| `GET /voice/status` | Per-component availability |
| `POST /voice/transcribe` | Audio file → transcript |
| `POST /voice/say` | Speak text aloud |
| `POST /voice/turn` | Recognised text → full pipeline |
| `POST /voice/listen` | Host microphone → full pipeline |

`/listen` and `/turn` both exist because `/listen` uses *this machine's*
microphone, which is useless to the Phase 6 phone client. `/turn` runs the same
pipeline over text recognised elsewhere.

## Verification

| Check | Result |
|---|---|
| `ruff` / `mypy` (strict) | Clean, 56 files |
| `pytest` | **42 voice tests**, part of 186 total |

No microphone is opened, no model downloaded, nothing spoken aloud. Every
component sits behind a Protocol precisely so the suite runs on a machine with no
audio hardware.

## Known gaps

1. **No continuous background listener.** The wake word only works within a
   request; there is no always-on hotword thread yet.
2. **Transcript-based wake detection is CPU-heavy** compared to a dedicated
   engine like Porcupine, because it transcribes first and matches second.
3. **No barge-in** — Quainex will not stop talking when you start.
4. **Energy-gate VAD is a constant**, not calibrated from ambient noise. A loud
   room ends recordings late.
5. **Exact volume levels** still unsupported (carried over from Phase 3).
