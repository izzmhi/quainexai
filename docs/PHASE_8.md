# Phase 8 — Vision

## Goal

Let Quainex answer questions about what is on screen and about documents on disk.

## The decision this phase turns on

The obvious Phase 8 shopping list is **Tesseract + OpenCV + template matching**.
I did not use any of it.

OCR returns *characters*. The questions people actually ask are not about
characters:

- "What does this error say?"
- "Which button do I press?"
- "Has the build finished?"

A vision-capable model answers those directly, and reading text is the easy
subset it gets for free. Template matching would additionally need a reference
image of every button you might ever ask about.

It also removes a native binary from the install. On a machine with a documented
history of stalling large downloads, "pip install and it works" is worth a great
deal.

**The trade is real and worth stating plainly:** this sends a screenshot off the
machine, and each question costs a request. That is why window enumeration exists
alongside it.

## Window enumeration is local and free

```
"Is VS Code open?"        -> list_windows()   ctypes, no model, no cost, no upload
"What does this say?"     -> look_at_screen()  screenshot -> model
```

Roughly thirty lines of `ctypes` against `EnumWindows`. A dependency for thirty
lines of stable API is a poor trade, and this way the cheap question never
becomes an expensive one.

## Screenshots do not linger

`look_at_screen()` writes to a temporary directory and deletes on the way out. A
picture of the user's entire desktop — password managers, private messages,
whatever was open — is not something to leave on disk because a question was
asked about it.

A test asserts both halves: the file exists while being read, and is gone
afterwards. To make that assertion meaningful I had to make the fake desktop
controller actually write a file; a fake that only recorded the call would have
made the test vacuous while still passing.

## PDFs are sent as documents, not extracted text

A PDF-to-text library discards layout, tables and figures — the parts a question
is most often actually about. The API takes PDFs natively, so the model sees the
document as it is.

## Capabilities added

| Intent | What it does | Cost |
|---|---|---|
| `look_at_screen` | Answer a question about the current screen | 1 request + upload |
| `read_document` | Answer a question about a PDF | 1 request + upload |
| `list_windows` | List open windows | free, local |

## Guards

- **Path containment** — images and PDFs must resolve inside the permitted roots,
  canonicalised before the check, same rule as Phases 3 and 7.
- **Size ceilings** — 5 MB per image, 25 MB per document, 8 images per request.
  These bound a mistake (a whole photo library, a 200 MB scan), not normal use.
- **Type allowlist** — PNG, JPEG, GIF, WebP. Typed as a literal union, so a typo
  is a type error rather than a 400 at runtime.

## Verification

| Check | Result |
|---|---|
| `ruff` / `mypy` (strict) | Clean, 67 files |
| `pytest` | **295 passed**, 1 skipped |

Nothing in the suite captures a real screen or calls a real model — both are
faked, so what is verified is Quainex's own behaviour: screenshot lifetime, path
containment, and payload construction.

## Known gaps

1. **No coordinates returned.** Quainex can say "the Save button is top-left" but
   cannot yet say *where* to click. Phase 10 needs that to act on what it sees.
2. **No screenshot caching.** Three questions about one screen cost three
   captures and three requests.
3. **Window enumeration is Windows-only.** Returns an empty list elsewhere rather
   than failing, so the calling code is already portable.
4. **No region capture.** Always the full virtual desktop, which on a multi-monitor
   setup uploads more than the question needs.
5. **Privacy is per-request, not policy.** There is no "never screenshot while
   this app is focused" rule. Worth having before any autonomous phase can
   trigger captures on its own.
