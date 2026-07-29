# Phase 6 — Remote Access and Authentication

## Goal

Make Quainex safely reachable from a phone — and close the security gap Phases 3
and 4 shipped with, which was only ever defensible while everything was bound to
localhost.

## Architecture

```
   phone ──▶ POST /auth/token  {password}
              └─ scrypt verify ──▶ signed bearer token (1h)
                                        │
   phone ──▶ any protected route  Authorization: Bearer …
              ├─ rate limit        429 if over
              ├─ token verify      401 if bad/expired
              └─ handler

   phone ──▶ POST /commands/execute  {intent}
              └─ requires confirmation
                   └─▶ 200, executed=false, + confirmation_token   ← bound to THIS action
   user taps "yes"
   phone ──▶ POST /commands/execute  {intent, confirmation_token}
              └─▶ executed
```

## The gap this phase closes

Phase 3 shipped `POST /commands/execute` with `confirmed: true`. On localhost
that was fine — the only caller was the user sitting at the machine. The moment
the API is reachable from a phone it stops being fine: **any client holding a
valid token could send `{"intent": {...shutdown...}, "confirmed": true}` without
ever having shown a human anything.**

Authentication proves *who* is calling. It says nothing about whether a person
was actually asked. Those are different questions and they need different
answers.

So `confirmed` is gone from the HTTP surface. A refusal now hands back a
**signed, single-use, action-bound token**, and executing requires presenting it:

| Property | Why it is there |
|---|---|
| **Signed** (HMAC-SHA256) | A caller cannot mint one |
| **Bound** to intent + target | A "close Spotify" approval cannot power off the machine |
| **Single-use** (nonce) | Captured tokens cannot be replayed |
| **Expiring** (2 min) | A token found in a log is almost always already dead |

The binding matters as much as the signature. A token that merely proved "the
user confirmed *something*" would be a skeleton key for whatever the caller asked
next. Three tests cover exactly this.

The in-process `confirmed=True` flag still exists, because the voice loop
genuinely did ask: it spoke the prompt aloud and heard the answer. It is never
set from an HTTP request.

## Authentication is derived, not configured

The single most important design decision in this phase:

```python
@property
def auth_required(self) -> bool:
    if self.require_auth is not None:
        return self.require_auth
    return not self.is_loopback  # ← derived from the bind address
```

A separate `enable_auth` flag would make **"exposed to the network with
authentication off"** a reachable configuration — and that configuration is
precisely the disaster. Here it is unreachable: binding to anything other than
loopback turns authentication on, and settings validation then **refuses to
start** without a password and secret configured.

Failing at startup is deliberate. Booting anyway with a warning means the window
between "listening on the network" and "someone reads the warning" is a window
with no front door on the machine.

```
ValueError: Authentication is required (host='0.0.0.0' is not loopback...),
but QUAINEX_AUTH_SECRET and QUAINEX_AUTH_PASSWORD_HASH are not configured.
```

## Setup

```powershell
python scripts/hash_password.py     # prints the two .env lines
# then set QUAINEX_HOST=0.0.0.0
```

The password itself is never written anywhere — only its scrypt hash, which
cannot be reversed. A plaintext password in `.env` is a plaintext password in
backups and in any screen-share of that file.

## Other decisions

**stdlib scrypt over bcrypt/argon2.** Memory-hard, in the standard library, no
compiled dependency. What matters far more than the choice among the three is
that the password is never stored in plaintext and never compared with `==` —
`hmac.compare_digest` prevents timing analysis.

**PyJWT rather than hand-rolled auth tokens.** Quainex hand-rolls the HMAC
*confirmation* tokens quite happily — they are short-lived, single-purpose and
never leave the system. Authentication tokens are different: they are what an
attacker attacks, and unverified `alg` headers have historically been
catastrophic. The algorithm is pinned explicitly rather than read from the token.

**Auth as a router dependency, not middleware.** Middleware would need a path
allowlist — a string-matching list that silently stops protecting a route the day
someone renames it. Attached to the router, a route added later is protected by
construction.

**WebSockets authenticate in the handshake.** HTTP dependencies do not apply to a
socket upgrade, and browsers cannot set headers on one — so the token may come
from the query string, with the header honoured for native clients. The socket is
closed with code 1008 *before* `accept()`, so an unauthenticated client never
reaches the message loop.

**Rate limiting only when auth is required.** A localhost-only Quainex has one
caller who can already do anything directly; throttling them is friction with no
benefit. When exposed, it slows password guessing and caps what a looping client
— such as Phase 10's autonomous agent — can do.

## Bug found by the tests

`hashlib.scrypt` at n=2¹⁵, r=8 needs exactly 32 MiB, which is exactly OpenSSL's
default `maxmem` ceiling — so every hash raised
`ValueError: memory limit exceeded`. Fixed by deriving the allowance from the
work factors (`128 * n * r * 2`) rather than lowering the cost. Deriving it means
raising the factors later cannot silently reintroduce the failure.

Without a test, this would have surfaced as "login is broken" the first time
anyone tried to use their phone.

## Verification

| Check | Result |
|---|---|
| `ruff` / `mypy` (strict) | Clean, 62 files |
| `pytest` | **239 passed**, 1 skipped (53 new) |

The load-bearing tests assert things *cannot* happen: `confirmed: true` over HTTP
is inert, a token for one action will not execute another, a replayed token
fails, and binding to `0.0.0.0` without credentials refuses to start.

## Known gaps

1. **Single user, single password.** No accounts, no roles. Fine for a personal
   system; revisit if Quainex is ever shared.
2. **No token revocation.** A lost phone's token stays valid until it expires;
   the mitigation today is the one-hour lifetime.
3. **Spent confirmation nonces are in-process.** A restart clears them, so the
   widest replay window a restart can open is the 2-minute token TTL.
4. **No TLS.** Tokens cross the LAN in the clear. A reverse proxy with a
   certificate is the right answer before this leaves a trusted network — **do
   not port-forward this to the internet as it stands.**
5. **No dashboard UI yet.** The API is complete; the React front end is not
   built. `/docs` is serviceable in the meantime.
6. **No remote terminal.** The roadmap lists a restricted terminal; it is
   deliberately deferred, since arbitrary command execution is the one thing the
   Phase 3 allowlist exists to prevent.
