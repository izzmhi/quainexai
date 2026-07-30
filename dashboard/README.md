# Quainex dashboard

The browser interface. Served by the Python process itself at
<http://127.0.0.1:8000/ui/> — start `python main.py` and open it. A bare visit to
`/` redirects here.

It lives at `/ui` rather than `/` deliberately: a static mount is a catch-all for
everything beneath its path, so mounting at the root would silently shadow every
API route registered after it. See `_mount_dashboard` in `quainex/api/app.py`.

## Purpose

Three jobs, in the order people actually need them:

1. **Talk to Quainex** — type or hold the mic, watch what it decided to do, and
   approve anything it refused to do unasked.
2. **See what happened** — the conversation and the append-only activity trail.
3. **Configure it** — paste API keys without opening a text editor, and prove
   they work before relying on them.

## Files

| File | What it is |
|---|---|
| `index.html` | The whole document. Every panel exists up front; the router hides all but one. |
| `app.css` | All presentation *and all animation*. Nothing here is set from JavaScript. |
| `app.js` | Routing, API calls, the core's state machine, and the settings screen. |

## Why there is no build step

No framework, no bundler, no CDN, no `npm install`.

Quainex is a local application talking to exactly one origin. A build step would
mean the interface only opens after a dependency install succeeded — on a machine
where large downloads are already unreliable — to render a page that a browser
can render natively. The page works from a cold start with the network unplugged.

The cost is real: no components, no reactive bindings, manual DOM construction.
Two rules keep that from turning into mud:

- **Every element the script touches is reached through `data-bind`.** No
  positional selectors, no class-name coupling, so restyling cannot break
  behaviour.
- **All animation is CSS keyed off one attribute.** `[data-bind="core"]` carries
  `data-state="idle | listening | thinking | speaking | error"`, and every ring,
  glow and timing derives from it. `app.js` never sets a style property, so the
  two files cannot drift.

**When this trade stops paying:** the moment two people are editing the interface
at once, or a panel needs genuine list virtualisation, or the same widget is
wanted in three places. At that point port it to a framework — the API underneath
is unaffected, which is the point of keeping the two apart.

## The core

The circle on the console panel is not a logo. `data-state` is its single input,
and the waveform inside it is driven by the *actual* microphone signal while
recording — via a Web Audio analyser — rather than a canned animation. When it is
not recording, the wave is two beating sines, cheap enough to run for hours.

## Security notes

These are properties of the page, not decoration:

- **The access token lives in `sessionStorage`.** A bearer token that outlives
  the tab, in a store any script on the origin can read, is a worse trade than
  asking for the password again. On loopback there is no token at all.
- **API keys are write-only.** The server has no endpoint that returns a stored
  key, and this page has no code path that would display one. You can replace a
  key; you cannot read it back. Saved values are cleared out of the input
  immediately, because a key sitting in a DOM node is a key in a screenshot.
- **Confirmations are replayed, never granted.** When the server refuses an
  action pending confirmation it returns a signed token bound to that one
  intent and target. "Yes, do it" sends that token back. The page cannot approve
  anything on its own — the same rule the autonomous agent lives under.
- **The static files are served unauthenticated; everything they display is
  not.** The page is how you reach the login prompt, so gating it would be
  circular. Every byte of state in it comes from a guarded router.

## Keyboard

| Key | Action |
|---|---|
| `Enter` | Send the composer |
| `Escape` | Cancel a pending confirmation |

## Future improvements

- Stream assistant replies over the WebSocket instead of awaiting the POST, so
  long answers appear as they are produced. The socket is already open and
  heartbeating; today it only proves liveness.
- Reorder the provider chain from the settings panel (currently `.env` only).
- A always-on wake-word listener, so the console reacts to "hey Quainex" without
  the mic being held. This needs a persistent audio pipeline, which is a
  deliberate decision to make explicitly rather than acquire by accident.
