/*
 * Quainex dashboard controller.
 *
 * Purpose:
 *   Drive the whole interface: routing between panels, talking to the Quainex
 *   HTTP API, the reactive core animation, and the settings screen that stores
 *   API keys.
 *
 * Why vanilla, and why one file:
 *   Quainex is a local application. A build step would mean the dashboard can
 *   only be opened after `npm install` succeeded, on a machine where large
 *   downloads are already unreliable, to render a page that talks to exactly one
 *   origin. So: no framework, no bundler, no CDN. The page works from a cold
 *   `python main.py` with the network unplugged.
 *
 *   The cost is honest — no components, no reactivity, manual DOM. It is paid
 *   back by two rules kept throughout: every element the script touches is
 *   reached through `data-bind`, and all animation lives in CSS keyed off a
 *   single `data-state` attribute. This file never sets a style property.
 *
 * Security posture:
 *   - The access token is held in `sessionStorage`, not `localStorage`: a bearer
 *     token that outlives the tab, in a store any script on the origin can read,
 *     is a worse trade than asking for the password again.
 *   - API keys are write-only here. The server never returns one, and this file
 *     has no code path that would display one if it did.
 *   - Confirmations are replayed, not granted. When the server refuses an action
 *     pending confirmation it returns a signed token bound to that one action;
 *     "Yes, do it" sends that token back. The UI cannot approve anything by
 *     itself, which is the same rule the autonomous agent lives under.
 *
 * Architecture:
 *   boot()
 *     |-- router          data-route buttons  -> show one [data-view]
 *     |-- api()           fetch + error envelope + bearer token + 401 handling
 *     |-- core            setState() -> [data-bind=core][data-state] -> CSS
 *     |-- console         ask -> render turn -> maybe confirm -> execute
 *     |-- mic             MediaRecorder -> /voice/transcribe -> ask
 *     |-- history         /memory/conversation + /memory/activity
 *     |-- commands        /commands
 *     |-- settings        /settings/providers  (read, store, clear, test)
 *     +-- link            WebSocket ping/pong -> connection pill
 *
 * Dependencies:
 *   None. Browser APIs only (fetch, WebSocket, MediaRecorder, Web Audio).
 *
 * Future improvements:
 *   - Stream assistant replies over the WebSocket instead of awaiting the POST,
 *     so long answers appear as they are produced.
 *   - Move the transcript to a virtualised list once sessions run to thousands
 *     of turns.
 */

"use strict";

/* ------------------------------------------------------------------ helpers */

/** @returns {HTMLElement|null} The element carrying this data-bind name. */
const bind = (name) => document.querySelector(`[data-bind="${name}"]`);

/**
 * Build an element.
 *
 * @param {string} tag Tag name.
 * @param {object} [attrs] Attributes to set. `class` and `text` are special.
 * @param {(Node|string)[]} [children] Children to append.
 * @returns {HTMLElement} The new element.
 */
function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else node.setAttribute(key, String(value));
  }
  for (const child of children) {
    node.append(child);
  }
  return node;
}

/**
 * Format a timestamp for display.
 *
 * @param {string|null} iso An ISO-8601 timestamp.
 * @returns {string} Local time, or an em dash when absent or unparseable.
 */
function when(iso) {
  if (!iso) return "—";
  const parsed = new Date(iso);
  if (Number.isNaN(parsed.getTime())) return "—";
  return parsed.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/**
 * Turn a snake_case identifier into something readable.
 *
 * @param {string} value The identifier.
 * @returns {string} Title-cased words.
 */
function humanise(value) {
  return String(value)
    .replaceAll("_", " ")
    .replace(/^\w/, (c) => c.toUpperCase());
}

/* -------------------------------------------------------------------- state */

const state = {
  /** @type {string} The visible panel. */
  route: "console",
  /** @type {{role: string, content: string}[]} History sent to the Brain. */
  history: [],
  /** @type {{intent: object, token: string}|null} Action awaiting a yes. */
  pending: null,
  /** @type {boolean} Whether a request is in flight. */
  busy: false,
  /** @type {object|null} Last /health payload. */
  health: null,
  /** @type {MediaRecorder|null} Live recorder while the mic is held. */
  recorder: null,
};

/** Where the bearer token lives. Session-scoped on purpose — see the header. */
const TOKEN_KEY = "quainex.token";

/* ---------------------------------------------------------------- api layer */

/** Error carrying the server's structured envelope, so callers can inspect it. */
class ApiError extends Error {
  /**
   * @param {string} message Human-readable failure.
   * @param {number} status HTTP status code.
   * @param {object} body Decoded response body, when there was one.
   */
  constructor(message, status, body) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

/**
 * Call the Quainex API.
 *
 * @param {string} path Path beginning with a slash.
 * @param {object} [options] `method`, `body` (auto-JSON), `form` (FormData).
 * @returns {Promise<any>} The decoded response body.
 * @throws {ApiError} The request failed, or the server returned 4xx/5xx.
 */
async function api(path, options = {}) {
  const { method = "GET", body, form } = options;
  const headers = {};
  const token = sessionStorage.getItem(TOKEN_KEY);
  if (token) headers.Authorization = `Bearer ${token}`;
  if (body !== undefined) headers["Content-Type"] = "application/json";

  let response;
  try {
    response = await fetch(path, {
      method,
      headers,
      body: form ?? (body !== undefined ? JSON.stringify(body) : undefined),
    });
  } catch {
    // A network-level failure here almost always means the Python process is
    // gone, so say that rather than surfacing "Failed to fetch".
    throw new ApiError("Quainex is not responding. Is the server still running?", 0, {});
  }

  const text = await response.text();
  let payload = {};
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = { detail: text.slice(0, 400) };
    }
  }

  if (response.ok) return payload;

  if (response.status === 401) {
    sessionStorage.removeItem(TOKEN_KEY);
    await promptForPassword();
  }

  // The API's error envelope nests the message under `error`; validation errors
  // and raw HTTPExceptions use `detail`. Read both rather than showing "[object
  // Object]" for one of them.
  const message =
    payload?.error?.message ??
    payload?.error ??
    (typeof payload?.detail === "string" ? payload.detail : null) ??
    `Request failed (${response.status}).`;
  throw new ApiError(String(message), response.status, payload);
}

/**
 * Ask for the remote-access password and exchange it for a token.
 *
 * Only reachable when the server requires authentication, which only happens
 * when it is bound to something other than loopback.
 *
 * @returns {Promise<boolean>} Whether a token was obtained.
 */
async function promptForPassword() {
  const password = window.prompt("Quainex requires authentication. Enter the remote password:");
  if (!password) return false;
  try {
    const response = await fetch("/auth/token", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password }),
    });
    if (!response.ok) {
      toast("That password was not accepted.", "bad");
      return false;
    }
    const { access_token: accessToken } = await response.json();
    sessionStorage.setItem(TOKEN_KEY, accessToken);
    toast("Authenticated.", "good");
    return true;
  } catch {
    toast("Could not reach the login endpoint.", "bad");
    return false;
  }
}

/* ------------------------------------------------------------------- toasts */

/**
 * Show a transient message.
 *
 * @param {string} message What to say.
 * @param {"info"|"good"|"bad"} [kind] Visual treatment.
 */
function toast(message, kind = "info") {
  const host = bind("toasts");
  if (!host) return;
  const node = el("div", { class: "toast", "data-kind": kind, text: message });
  host.append(node);
  // Long enough to read a provider error, short enough not to stack up.
  setTimeout(() => node.remove(), kind === "bad" ? 8000 : 4000);
}

/* --------------------------------------------------------------- the core */

/**
 * Reactive core: one attribute drives every ring, glow and timing in app.css.
 *
 * The wave is the only thing scripted, because it carries real information —
 * live microphone amplitude while recording — rather than decoration.
 */
const core = {
  /** @type {AnalyserNode|null} */
  analyser: null,
  /** @type {AudioContext|null} */
  audio: null,
  /** @type {number|null} */
  frame: null,
  phase: 0,

  /**
   * Set the core's state and its captions.
   *
   * @param {"idle"|"listening"|"thinking"|"speaking"|"error"} value New state.
   * @param {string} [hint] Optional replacement for the hint line.
   */
  set(value, hint) {
    const node = bind("core");
    if (node) node.setAttribute("data-state", value);

    const labels = {
      idle: "Standing by",
      listening: "Listening",
      thinking: "Thinking",
      speaking: "Responding",
      error: "Something went wrong",
    };
    const hints = {
      idle: "Type below, or press and hold the mic.",
      listening: "Release the mic when you have finished speaking.",
      thinking: "Working out what you meant, then doing it.",
      speaking: "Reading the answer back on this machine's speakers.",
      error: "The transcript below has the details.",
    };
    const label = bind("stateLabel");
    const hintNode = bind("stateHint");
    if (label) label.textContent = labels[value] ?? value;
    if (hintNode) hintNode.textContent = hint ?? hints[value] ?? "";
  },

  /** Start the wave animation loop. Idempotent. */
  start() {
    if (core.frame !== null) return;
    const draw = () => {
      core.phase += 0.06;
      core.render();
      core.frame = requestAnimationFrame(draw);
    };
    core.frame = requestAnimationFrame(draw);
  },

  /** Write one frame of the waveform. */
  render() {
    const line = bind("wave");
    if (!line) return;
    const stateName = bind("core")?.getAttribute("data-state") ?? "idle";
    const points = [];
    const count = 64;

    let samples = null;
    if (core.analyser) {
      samples = new Uint8Array(core.analyser.frequencyBinCount);
      core.analyser.getByteTimeDomainData(samples);
    }

    // Amplitudes are tuned per state so the core reads at a glance: flat when
    // idle, agitated when thinking, generous when speaking.
    const gain = { idle: 1.5, listening: 22, thinking: 9, speaking: 14, error: 3 }[stateName] ?? 2;

    for (let i = 0; i < count; i += 1) {
      const x = (i / (count - 1)) * 200;
      let y;
      if (samples) {
        // Real microphone data, centred on 128.
        const sample = samples[Math.floor((i / count) * samples.length)] ?? 128;
        y = 30 + ((sample - 128) / 128) * gain * 1.6;
      } else {
        // Two beating sines: enough motion to look alive, cheap enough to run
        // for hours without touching the battery.
        const t = core.phase + i * 0.28;
        y = 30 + (Math.sin(t) * 0.7 + Math.sin(t * 0.43) * 0.3) * gain;
      }
      points.push(`${x.toFixed(1)},${y.toFixed(1)}`);
    }
    line.setAttribute("points", points.join(" "));
  },

  /**
   * Attach a live microphone stream so the wave shows real amplitude.
   *
   * @param {MediaStream} stream The stream being recorded.
   */
  listen(stream) {
    try {
      core.audio = core.audio ?? new AudioContext();
      const source = core.audio.createMediaStreamSource(stream);
      core.analyser = core.audio.createAnalyser();
      core.analyser.fftSize = 512;
      source.connect(core.analyser);
    } catch {
      // Web Audio is a nicety here; losing it costs a prettier waveform and
      // nothing else, so recording continues either way.
      core.analyser = null;
    }
  },

  /** Detach the microphone stream. */
  deafen() {
    core.analyser = null;
  },
};

/* ------------------------------------------------------------------ router */

/**
 * Show one panel and mark its tab current.
 *
 * @param {string} route The `data-view` to show.
 */
function navigate(route) {
  state.route = route;
  for (const view of document.querySelectorAll("[data-view]")) {
    view.hidden = view.dataset.view !== route;
  }
  for (const tab of document.querySelectorAll(".tab")) {
    if (tab.dataset.route === route) tab.setAttribute("aria-current", "page");
    else tab.removeAttribute("aria-current");
  }
  // Panels load on show rather than on boot: three extra requests at startup to
  // populate tabs nobody has opened is three requests too many.
  if (route === "history") void loadHistory();
  if (route === "commands") void loadCommands();
  if (route === "settings") void loadSettings();
}

/* ----------------------------------------------------------------- console */

/**
 * Append a turn to the transcript.
 *
 * @param {object} turn `{ role, text, meta, data, error }`.
 */
function addTurn({ role, text, meta, data, error }) {
  const host = bind("transcript");
  if (!host) return;

  const classes = ["turn"];
  if (role === "user") classes.push("turn--user");
  if (error) classes.push("turn--error");

  const children = [];
  if (meta) children.push(el("div", { class: "turn-meta", text: meta }));
  children.push(el("div", { class: "turn-body", text }));
  if (data && Object.keys(data).length > 0) {
    children.push(el("pre", { class: "turn-data", text: JSON.stringify(data, null, 2) }));
  }

  host.append(el("div", { class: classes.join(" ") }, children));
  host.scrollTop = host.scrollHeight;
}

/**
 * Enable or disable the composer while a request is in flight.
 *
 * @param {boolean} busy Whether Quainex is working.
 */
function setBusy(busy) {
  state.busy = busy;
  const send = bind("send");
  const input = bind("utterance");
  if (send) send.disabled = busy;
  if (input) input.disabled = busy;
}

/**
 * Send an utterance through interpretation and execution.
 *
 * @param {string} utterance What the user said or typed.
 */
async function ask(utterance) {
  const trimmed = utterance.trim();
  if (!trimmed || state.busy) return;

  addTurn({ role: "user", text: trimmed });
  state.history.push({ role: "user", content: trimmed });
  setBusy(true);
  core.set("thinking");

  try {
    const { intent, result } = await api("/commands/ask", {
      method: "POST",
      // Bounded rather than unbounded: the server keeps its own memory, and
      // sending an entire session back on every turn pays for the same tokens
      // repeatedly. Twelve entries covers "close it" after "open Spotify".
      body: { utterance: trimmed, history: state.history.slice(-12) },
    });

    renderResult(intent, result);
  } catch (error) {
    core.set("error");
    addTurn({
      role: "assistant",
      text: error instanceof ApiError ? error.message : String(error),
      meta: "failed",
      error: true,
    });
    toast(error.message, "bad");
  } finally {
    setBusy(false);
    const input = bind("utterance");
    if (input) input.focus();
  }
}

/**
 * Render one command result, opening the confirmation gate when required.
 *
 * @param {object} intent The classified intent.
 * @param {object} result The executor's verdict.
 */
function renderResult(intent, result) {
  const confidence = Math.round((intent.confidence ?? 0) * 100);
  const meta = `${humanise(intent.intent)}${intent.target ? ` → ${intent.target}` : ""} · ${confidence}% confident · ${result.status}`;

  addTurn({
    role: "assistant",
    text: result.message,
    meta,
    data: result.data,
    error: result.status === "failed" || result.status === "blocked",
  });
  state.history.push({ role: "assistant", content: result.message });

  if (result.status === "requires_confirmation" && result.confirmation_token) {
    state.pending = { intent, token: result.confirmation_token };
    openConfirm(result.message);
    core.set("idle", "Waiting for you to confirm.");
    // Spoken too: a confirmation you have not noticed is a request that appears to
    // have been ignored, and the mic path means you may not be looking at the tab.
    void speak(result.message);
    return;
  }

  core.set(result.status === "succeeded" ? "idle" : "error");
  void speak(result.message);
}

/**
 * Open the confirmation gate.
 *
 * @param {string} message What the server said it wants confirmed.
 */
function openConfirm(message) {
  const backdrop = bind("confirmBackdrop");
  const text = bind("confirmMessage");
  if (text) text.textContent = message;
  if (backdrop) backdrop.hidden = false;
  document.querySelector('[data-action="confirm-accept"]')?.focus();
}

/** Close the confirmation gate and drop the pending action. */
function closeConfirm() {
  const backdrop = bind("confirmBackdrop");
  if (backdrop) backdrop.hidden = true;
  state.pending = null;
}

/** Replay the signed confirmation token, executing the action it is bound to. */
async function acceptConfirm() {
  const pending = state.pending;
  closeConfirm();
  if (!pending) return;

  setBusy(true);
  core.set("thinking");
  try {
    const result = await api("/commands/execute", {
      method: "POST",
      // The token is sent back exactly as issued. It is signed, single-use, and
      // bound to this intent and target, so it cannot be reused for anything
      // else — and the dashboard could not forge one if it tried.
      body: { intent: pending.intent, confirmation_token: pending.token },
    });
    addTurn({
      role: "assistant",
      text: result.message,
      meta: `confirmed · ${result.status}`,
      data: result.data,
      error: result.status !== "succeeded",
    });
    core.set(result.status === "succeeded" ? "idle" : "error");
    void speak(result.message);
  } catch (error) {
    addTurn({ role: "assistant", text: error.message, meta: "failed", error: true });
    core.set("error");
    toast(error.message, "bad");
  } finally {
    setBusy(false);
  }
}

/* --------------------------------------------------------------------- mic */

/**
 * Start recording from the browser's microphone.
 *
 * Recording happens in the page rather than through `/voice/listen` on purpose:
 * press-and-hold should capture the microphone of whatever device you are
 * holding, and the server endpoint always records the host machine's.
 */
async function startRecording() {
  if (state.recorder) return;
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch {
    toast("Microphone access was refused.", "bad");
    return;
  }

  const chunks = [];
  const recorder = new MediaRecorder(stream);
  recorder.ondataavailable = (event) => {
    if (event.data.size > 0) chunks.push(event.data);
  };
  recorder.onstop = async () => {
    stream.getTracks().forEach((track) => track.stop());
    core.deafen();
    bind("mic")?.setAttribute("data-recording", "false");
    state.recorder = null;

    if (chunks.length === 0) {
      core.set("idle");
      return;
    }

    core.set("thinking", "Transcribing what you said.");
    const form = new FormData();
    form.append("audio", new Blob(chunks, { type: recorder.mimeType }), "recording.webm");
    try {
      const transcript = await api("/voice/transcribe", { method: "POST", form });
      const text = (transcript.text ?? "").trim();
      if (!text) {
        core.set("idle", "Nothing was recognised in that recording.");
        return;
      }
      await ask(text);
    } catch (error) {
      core.set("error");
      toast(error.message, "bad");
    }
  };

  state.recorder = recorder;
  core.listen(stream);
  core.set("listening");
  bind("mic")?.setAttribute("data-recording", "true");
  recorder.start();
}

/** Stop recording and hand the audio to transcription. */
function stopRecording() {
  if (state.recorder && state.recorder.state !== "inactive") state.recorder.stop();
}

/* ----------------------------------------------------------------- history */

/** Load and render the conversation and activity trails. */
async function loadHistory() {
  const turns = bind("historyTurns");
  const activity = bind("historyActivity");
  if (turns) turns.replaceChildren(el("li", { class: "empty", text: "Loading…" }));
  if (activity) activity.replaceChildren(el("li", { class: "empty", text: "Loading…" }));

  try {
    const [conversation, trail] = await Promise.all([
      api("/memory/conversation?limit=50"),
      api("/memory/activity?limit=50"),
    ]);

    if (turns) {
      turns.replaceChildren(
        ...(conversation.length
          ? conversation.map((turn) =>
              el("li", { "data-ok": "true" }, [
                el("div", {
                  class: "timeline-meta",
                  text: `${humanise(turn.role)} · ${when(turn.created_at)}${turn.intent ? ` · ${humanise(turn.intent)}` : ""}`,
                }),
                el("div", { class: "timeline-body", text: turn.content }),
              ]),
            )
          : [el("li", { class: "empty", text: "Nothing said yet." })]),
      );
    }

    if (activity) {
      activity.replaceChildren(
        ...(trail.length
          ? trail.map((entry) =>
              el("li", { "data-ok": String(entry.status === "succeeded") }, [
                el("div", {
                  class: "timeline-meta",
                  text: `${humanise(entry.intent)}${entry.target ? ` → ${entry.target}` : ""} · ${entry.status} · ${when(entry.created_at)}`,
                }),
                el("div", { class: "timeline-body", text: entry.detail ?? "" }),
              ]),
            )
          : [el("li", { class: "empty", text: "Nothing done yet." })]),
      );
    }
  } catch (error) {
    toast(error.message, "bad");
  }
}

/** Forget the current conversation, after asking. */
async function clearConversation() {
  if (!window.confirm("Forget the stored conversation? The activity trail is kept.")) return;
  try {
    await api("/memory/conversation", { method: "DELETE" });
    state.history = [];
    bind("transcript")?.replaceChildren();
    toast("Conversation forgotten.", "good");
    await loadHistory();
  } catch (error) {
    toast(error.message, "bad");
  }
}

/* ---------------------------------------------------------------- commands */

/** @type {Record<string, string>} Cached capability catalogue. */
let capabilities = {};

/** Load the executor's registry and render it. */
async function loadCommands() {
  if (Object.keys(capabilities).length === 0) {
    try {
      capabilities = await api("/commands");
    } catch (error) {
      toast(error.message, "bad");
      return;
    }
  }
  renderCommands();
}

/** Render the capability grid, honouring the filter box. */
function renderCommands() {
  const grid = bind("commandGrid");
  if (!grid) return;
  const needle = (bind("commandFilter")?.value ?? "").trim().toLowerCase();

  const entries = Object.entries(capabilities).filter(
    ([name, detail]) =>
      !needle || name.toLowerCase().includes(needle) || detail.toLowerCase().includes(needle),
  );

  grid.replaceChildren(
    ...(entries.length
      ? entries.map(([name, detail]) =>
          el("button", { class: "chip", type: "button", "data-command": name }, [
            el("strong", { text: humanise(name) }),
            el("span", { text: detail }),
          ]),
        )
      : [el("p", { class: "empty", text: "No capability matches that." })]),
  );
}

/* ---------------------------------------------------------------- settings */

/** Load provider configuration and render the settings panel. */
async function loadSettings() {
  try {
    renderSettings(await api("/settings/providers"));
  } catch (error) {
    toast(error.message, "bad");
  }
  await loadTelegram();
}

/* ------------------------------------------------------- phone control */

/** Load and render the Telegram setup card. */
async function loadTelegram() {
  try {
    renderTelegram(await api("/settings/telegram"));
  } catch (error) {
    toast(error.message, "bad");
  }
}

/**
 * Render the phone-control status.
 *
 * @param {object} state The `/settings/telegram` payload.
 */
function renderTelegram(state) {
  const input = bind("telegramUsers");
  // Only populated when empty, so a half-typed list is never overwritten by a
  // background refresh.
  if (input && !input.value) input.value = state.allowed_users.join(", ");

  const host = bind("telegramStatus");
  if (!host) return;

  // The status says *why* it is off, not just that it is. "Not configured" with
  // no explanation is the reason this feature went unused.
  // `running` is a flag; `last_poll_seconds_ago` is evidence. A stalled loop
  // reports running forever, so a stale timestamp is reported as a stall rather
  // than shown as healthy.
  const stalled = state.running && state.last_poll_seconds_ago > 60;
  const [tag, text] = stalled
    ? [
        "tag--bad",
        `Says it is polling, but the last poll was ${Math.round(state.last_poll_seconds_ago)}s ago. Press Stop then Start.`,
      ]
    : state.running
      ? ["tag--ok", "Polling. Message your bot from your phone."]
      : state.configured
        ? ["tag--warn", "Ready, but not polling. Press Start."]
        : ["tag--bad", `Off — still needs ${state.missing.join(" and ")}.`];

  host.replaceChildren(
    el("span", { class: `tag ${tag}`, text: state.running ? "live" : "off" }),
    el("span", { text }),
    el("span", {
      class: "muted",
      text: `Token ${state.token_configured ? "saved" : "missing"} · ${state.allowed_users.length} allowed account(s) · blocks ${state.blocked_intents.join(", ")}`,
    }),
  );
}

/** Save the allowlist, replacing whatever was there. */
async function saveTelegramUsers() {
  const input = bind("telegramUsers");
  if (!input) return;

  // Tolerant of commas, spaces and newlines: this is typed by hand, and being
  // strict about separators would only produce a puzzle.
  const ids = input.value
    .split(/[\s,]+/)
    .filter(Boolean)
    .map(Number);

  if (ids.some((id) => !Number.isInteger(id) || id <= 0)) {
    toast("Telegram user ids are positive whole numbers.", "bad");
    return;
  }

  try {
    const state = await api("/settings/telegram/allowed-users", {
      method: "PUT",
      body: { user_ids: ids },
    });
    renderTelegram(state);
    toast(
      ids.length
        ? `${ids.length} account(s) allowed. ${state.configured ? "Press Start to begin polling." : ""}`
        : "Allowlist cleared — phone control is off.",
      "good",
    );
  } catch (error) {
    toast(error.message, "bad");
  }
}

/**
 * Check the token with Telegram and offer any user ids it can see.
 *
 * Removes the one genuinely awkward step in setup — finding your numeric id via
 * a third-party bot and typing it correctly.
 */
async function testTelegram() {
  toast("Asking Telegram about this bot…");
  let result;
  try {
    result = await api("/settings/telegram/test", { method: "POST" });
  } catch (error) {
    toast(error.message, "bad");
    return;
  }

  if (!result.ok) {
    toast(result.error ?? "Telegram did not accept the token.", "bad");
    return;
  }

  toast(`Token works — the bot is @${result.username}.`, "good");

  const host = bind("telegramCandidates");
  if (!host) return;

  const candidates = result.candidates ?? [];
  if (candidates.length === 0) {
    host.hidden = false;
    host.replaceChildren(
      el("p", {
        class: "muted",
        text: `Send any message to @${result.username} from your phone, then press "Check bot" again — your account will appear here.`,
      }),
    );
    return;
  }

  host.hidden = false;
  host.replaceChildren(
    el("p", {
      class: "muted",
      text: "Accounts that have messaged this bot. Only add one you recognise — messaging the bot does not grant access, you do.",
    }),
    ...candidates.map((candidate) => {
      const label = [candidate.name, candidate.username && `@${candidate.username}`]
        .filter(Boolean)
        .join(" ");
      const button = el("button", {
        class: "chip",
        type: "button",
        title: "Add this id to the allowlist field",
      });
      button.append(
        el("strong", { text: String(candidate.user_id) }),
        el("span", { text: label || "no display name" }),
      );
      // Fills the field rather than saving: adding an account to the list that
      // controls this machine stays a deliberate act, not one click.
      button.addEventListener("click", () => {
        const input = bind("telegramUsers");
        if (!input) return;
        const existing = input.value
          .split(/[\s,]+/)
          .filter(Boolean)
          .map(String);
        if (!existing.includes(String(candidate.user_id))) {
          existing.push(String(candidate.user_id));
        }
        input.value = existing.join(", ");
        input.focus();
        toast("Added to the field. Press Save to apply it.");
      });
      return button;
    }),
  );
}

/**
 * Start or stop polling.
 *
 * @param {"start"|"stop"} action Which to do.
 */
async function toggleTelegram(action) {
  try {
    await api(`/telegram/${action}`, { method: "POST" });
    await loadTelegram();
    toast(action === "start" ? "Polling Telegram." : "Stopped polling.", "good");
  } catch (error) {
    toast(error.message, "bad");
  }
}

/**
 * Render the settings panel from a configuration snapshot.
 *
 * @param {object} snapshot The `/settings/providers` payload.
 */
function renderSettings(snapshot) {
  const pathNode = bind("vaultPath");
  if (pathNode) pathNode.textContent = `Vault: ${snapshot.vault_path}`;

  const notice = bind("vaultNotice");
  if (notice) {
    if (snapshot.vault_writable) {
      notice.hidden = true;
    } else {
      notice.hidden = false;
      notice.textContent =
        "Credentials cannot be encrypted on this platform, so saving is disabled here. " +
        "Set the keys in .env instead.";
    }
  }

  const chain = bind("chain");
  if (chain) {
    chain.replaceChildren(
      ...snapshot.chain.map((node, index) =>
        el(
          "div",
          {
            class: "chain-node",
            "data-active": String(Boolean(node.available)),
            "data-first": String(index === 0 && Boolean(node.available)),
          },
          [
            el("b", { text: String(node.name).split("/")[0] }),
            el("span", { text: node.available ? "ready" : "no key" }),
          ],
        ),
      ),
    );
  }

  const keys = bind("keys");
  if (keys) {
    keys.replaceChildren(...snapshot.secrets.map((secret) => keyRow(secret, snapshot)));
  }

  renderSystem();
}

/**
 * Build one credential row.
 *
 * @param {object} secret A `SecretState` entry.
 * @param {object} snapshot The whole snapshot, for the writable flag.
 * @returns {HTMLElement} The row.
 */
function keyRow(secret, snapshot) {
  const labelChildren = [el("b", { text: secret.label }), el("span", { text: secret.detail })];
  if (secret.url) {
    labelChildren.push(
      el("a", { href: secret.url, target: "_blank", rel: "noreferrer noopener" }, [
        document.createTextNode("Get a key ↗"),
      ]),
    );
  }

  const input = el("input", {
    type: "password",
    placeholder: secret.configured ? "Replace this key…" : "Paste a key…",
    autocomplete: "off",
    spellcheck: "false",
    "aria-label": `${secret.label} API key`,
  });
  if (!snapshot.vault_writable) input.disabled = true;

  const save = el("button", { class: "primary", type: "button", text: "Save" });
  save.disabled = !snapshot.vault_writable;
  save.addEventListener("click", async () => {
    const value = input.value.trim();
    if (value.length < 8) {
      toast("That does not look like an API key.", "bad");
      return;
    }
    save.disabled = true;
    try {
      const updated = await api(`/settings/providers/${secret.name}`, {
        method: "PUT",
        body: { value },
      });
      // Cleared immediately: the value is stored, and a key sitting in a DOM
      // node is a key in a screenshot.
      input.value = "";
      toast(`${secret.label} saved. The chain reloaded — no restart needed.`, "good");
      renderSettings(updated);
      void refreshHealth();
    } catch (error) {
      toast(error.message, "bad");
      save.disabled = false;
    }
  });

  const children = [
    el("div", { class: "key-label" }, labelChildren),
    el("div", { class: "key-input" }, [input, save]),
    el("div", { class: "key-state" }, [
      el("span", {
        class: `tag ${secret.source === "vault" ? "tag--ok" : secret.source === "env" ? "tag--warn" : "tag--bad"}`,
        text: secret.source === "vault" ? "saved here" : secret.source === "env" ? ".env" : "not set",
      }),
      el("span", { class: "muted", text: secret.hint }),
    ]),
  ];

  if (secret.source === "vault") {
    const clear = el("button", { class: "danger-ghost", type: "button", text: "Clear" });
    clear.addEventListener("click", async () => {
      if (!window.confirm(`Remove the stored ${secret.label} key?`)) return;
      try {
        renderSettings(await api(`/settings/providers/${secret.name}`, { method: "DELETE" }));
        toast(`${secret.label} key removed.`, "good");
        void refreshHealth();
      } catch (error) {
        toast(error.message, "bad");
      }
    });
    children[2].append(clear);
  }

  return el("div", { class: "key-row", "data-configured": String(secret.configured) }, children);
}

/** Send one real prompt through the chain and report what answered. */
async function testProviders() {
  toast("Asking the chain to answer one prompt…");
  try {
    const result = await api("/settings/providers/test", { method: "POST" });
    if (result.ok) toast(`${result.provider} replied: ${result.reply}`, "good");
    else toast(result.error ?? "No provider answered.", "bad");
  } catch (error) {
    toast(error.message, "bad");
  }
}

/** Render the system panel from the last health payload. */
function renderSystem() {
  const grid = bind("systemGrid");
  if (!grid || !state.health) return;
  const health = state.health;

  const stats = [
    ["Version", `${health.app} ${health.version}`],
    ["Environment", health.environment],
    ["Uptime", `${Math.round(health.uptime_seconds)}s`],
    ["Model", health.ai?.model ?? "—"],
    ["Chain", health.ai?.available ? "ready" : "no provider configured"],
    ["Providers", String(health.ai?.providers?.length ?? 0)],
  ];

  grid.replaceChildren(
    ...stats.map(([label, value]) =>
      el("div", { class: "stat" }, [el("span", { text: label }), el("b", { text: String(value) })]),
    ),
  );
}

/* ------------------------------------------------------------------ status */

/** Poll `/health` and update the provider pill. */
async function refreshHealth() {
  try {
    state.health = await api("/health");
  } catch {
    // /health needs no token and never fails on its own, so a failure here means
    // the process is down — which the link pill already reports.
    setPill("providerPill", "providerName", "offline", "bad");
    return;
  }

  const ai = state.health.ai ?? {};
  const configured = (ai.providers ?? []).filter((provider) => provider.available);
  if (!ai.available) {
    setPill("providerPill", "providerName", "no API key", "warn");
  } else {
    const leader = String(configured[0]?.name ?? ai.provider).split("/")[0];
    setPill("providerPill", "providerName", `${leader} +${Math.max(configured.length - 1, 0)}`, "good");
  }
  renderSystem();
}

/**
 * Update one status pill.
 *
 * @param {string} pill data-bind of the pill.
 * @param {string} label data-bind of its text node.
 * @param {string} text What to show.
 * @param {"good"|"warn"|"bad"|"unknown"} status Dot colour.
 */
function setPill(pill, label, text, status) {
  const host = bind(pill);
  const textNode = bind(label);
  if (textNode) textNode.textContent = text;
  host?.querySelector(".dot")?.setAttribute("data-state", status);
}

/**
 * Keep a WebSocket open purely as a liveness signal.
 *
 * The socket is an echo endpoint today, so this is honest about what it proves:
 * the process is up and accepting connections. When streaming replies land, this
 * is the channel they arrive on.
 */
function openLink() {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  let socket;
  try {
    socket = new WebSocket(`${scheme}://${location.host}/ws`);
  } catch {
    setPill("linkPill", "linkName", "link down", "bad");
    return;
  }

  socket.addEventListener("open", () => setPill("linkPill", "linkName", "live", "good"));
  socket.addEventListener("close", () => {
    setPill("linkPill", "linkName", "reconnecting", "warn");
    // Fixed backoff rather than exponential: the only realistic cause is a
    // restarted dev server, and five seconds is how long that takes.
    setTimeout(openLink, 5000);
  });
  socket.addEventListener("error", () => setPill("linkPill", "linkName", "link down", "bad"));

  const heartbeat = setInterval(() => {
    if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: "ping" }));
    else clearInterval(heartbeat);
  }, 20000);
}

/* ------------------------------------------------------------------ speech */

/** Where the preference lives. Persistent, unlike the access token. */
const SPEECH_KEY = "quainex.speech";

/** @returns {boolean} Whether replies should be spoken. */
const speechEnabled = () => localStorage.getItem(SPEECH_KEY) === "on";

/**
 * Speak a reply on the host machine's speakers.
 *
 * Server-side rather than the browser's `speechSynthesis`, deliberately. Quainex
 * runs on *this* machine and answers about it; the voice should come out of its
 * speakers, the same ones the voice loop and the wake word already use. Browser
 * synthesis would also make the assistant silent whenever the tab was closed,
 * which is precisely when a spoken answer is most useful.
 *
 * @param {string} text What to say.
 */
async function speak(text) {
  if (!speechEnabled() || !text.trim()) return;
  core.set("speaking");
  try {
    await api("/voice/say", { method: "POST", body: { text } });
  } catch {
    // Speech is an enhancement. A failure here must not make a successful command
    // look failed, so it is swallowed rather than surfaced as an error turn.
  } finally {
    core.set("idle");
  }
}

/** Turn speech output on or off, remembering the choice. */
function toggleSpeech() {
  const next = speechEnabled() ? "off" : "on";
  localStorage.setItem(SPEECH_KEY, next);
  renderSpeechToggle();
  toast(next === "on" ? "Quainex will speak its replies." : "Speech output off.", "good");
  if (next === "on") void speak("Speech output is on.");
}

/** Reflect the speech preference in the toolbar. */
function renderSpeechToggle() {
  const button = bind("speechToggle");
  if (!button) return;
  const on = speechEnabled();
  button.dataset.on = String(on);
  button.setAttribute("aria-pressed", String(on));
  button.title = on ? "Speech on - click to mute" : "Speak replies aloud on this machine";
}

/* ------------------------------------------------------------------- theme */

/** Toggle between the dark and light palettes, remembering the choice. */
function toggleTheme() {
  const root = document.documentElement;
  const next = root.dataset.theme === "dark" ? "light" : "dark";
  root.dataset.theme = next;
  localStorage.setItem("quainex.theme", next);
}

/* -------------------------------------------------------------------- boot */

/** Wire every listener and start the interface. */
function boot() {
  document.documentElement.dataset.theme = localStorage.getItem("quainex.theme") ?? "dark";

  for (const tab of document.querySelectorAll(".tab")) {
    tab.addEventListener("click", () => navigate(tab.dataset.route));
  }

  bind("composer")?.addEventListener("submit", (event) => {
    event.preventDefault();
    const input = bind("utterance");
    if (!input) return;
    const value = input.value;
    input.value = "";
    void ask(value);
  });

  const mic = bind("mic");
  if (mic) {
    // Pointer events rather than mouse+touch pairs: one code path covers mouse,
    // pen and finger, and `pointercancel` means a drag off the button releases
    // the microphone instead of leaving it open.
    mic.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      void startRecording();
    });
    for (const name of ["pointerup", "pointerleave", "pointercancel"]) {
      mic.addEventListener(name, stopRecording);
    }
  }

  document.querySelector('[data-action="speech"]')?.addEventListener("click", toggleSpeech);
  document.querySelector('[data-action="theme"]')?.addEventListener("click", toggleTheme);
  document.querySelector('[data-action="refresh-history"]')?.addEventListener("click", loadHistory);
  document
    .querySelector('[data-action="clear-conversation"]')
    ?.addEventListener("click", clearConversation);
  document.querySelector('[data-action="test-providers"]')?.addEventListener("click", testProviders);
  document
    .querySelector('[data-action="telegram-save"]')
    ?.addEventListener("click", saveTelegramUsers);
  document.querySelector('[data-action="telegram-test"]')?.addEventListener("click", testTelegram);
  document
    .querySelector('[data-action="telegram-start"]')
    ?.addEventListener("click", () => toggleTelegram("start"));
  document
    .querySelector('[data-action="telegram-stop"]')
    ?.addEventListener("click", () => toggleTelegram("stop"));
  document.querySelector('[data-action="confirm-cancel"]')?.addEventListener("click", () => {
    closeConfirm();
    addTurn({ role: "assistant", text: "Cancelled. Nothing was done.", meta: "cancelled" });
    core.set("idle");
  });
  document.querySelector('[data-action="confirm-accept"]')?.addEventListener("click", acceptConfirm);

  bind("commandFilter")?.addEventListener("input", renderCommands);

  // Clicking a capability drops a usable phrasing into the composer rather than
  // executing it: the catalogue is a reference, not a set of trigger buttons.
  bind("commandGrid")?.addEventListener("click", (event) => {
    const chip = event.target.closest("[data-command]");
    if (!chip) return;
    navigate("console");
    const input = bind("utterance");
    if (input) {
      input.value = `${humanise(chip.dataset.command).toLowerCase()} `;
      input.focus();
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !bind("confirmBackdrop")?.hidden) closeConfirm();
  });

  renderSpeechToggle();
  core.set("idle");
  core.start();
  navigate("console");
  void refreshHealth();
  openLink();
  // Slow poll: uptime and provider availability change on the scale of minutes,
  // and the settings panel refreshes health directly after any change.
  setInterval(refreshHealth, 30000);
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
else boot();
