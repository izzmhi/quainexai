"""Brain data model: the intent catalogue and the structures around it.

Purpose:
    Define the closed vocabulary the Brain may classify an utterance into, and
    the two structures that carry a classification through the system.

Why two models, not one:
    ``IntentClassification`` is what the *language model* produces — perception.
    ``Intent`` is what the *Brain* returns — perception plus policy. The model is
    never asked whether an action needs confirmation, because that is a security
    decision belonging to Quainex, not to a probabilistic classifier. Keeping
    them separate means a prompt-injected utterance cannot talk its way out of a
    confirmation prompt: the flag is computed locally from the intent type.

Why a closed enum, not free text:
    Phase 3 dispatches on this value. A free-text intent would mean matching on
    model-authored strings that drift between runs and model versions. An enum
    makes the command registry exhaustive and the failure mode explicit — an
    unrecognised request becomes ``UNKNOWN`` instead of a plausible-looking slug
    that silently matches nothing.

Architecture:
    utterance -> AIProvider.parse(output_model=IntentClassification)
              -> Brain applies policy
              -> Intent (adds requires_confirmation)
              -> Phase 3 command registry dispatches on `.intent`

Dependencies:
    pydantic

Future improvements:
    * Add ``SET_REMINDER`` / ``SCHEDULE_TASK`` when Phase 10 lands the scheduler.
    * Add per-intent parameter schemas once commands declare their own inputs.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class IntentType(StrEnum):
    """Every action Quainex can be asked to take.

    Values are stable identifiers: Phase 3 commands register against them, so
    renaming one is a breaking change to the command registry.
    """

    # --- Applications ---
    OPEN_APPLICATION = "open_application"
    CLOSE_APPLICATION = "close_application"

    # --- Navigation ---
    OPEN_WEBSITE = "open_website"
    OPEN_FOLDER = "open_folder"
    CREATE_FOLDER = "create_folder"
    SEARCH_FILES = "search_files"
    SEND_FILE = "send_file"
    SEND_FOLDER = "send_folder"

    # --- System control ---
    LOCK_SCREEN = "lock_screen"
    SLEEP = "sleep"
    RESTART = "restart"
    SHUTDOWN = "shutdown"
    SET_VOLUME = "set_volume"
    SET_BRIGHTNESS = "set_brightness"
    KEYBOARD_LIGHT = "keyboard_light"
    MEDIA_CONTROL = "media_control"
    WINDOW_CONTROL = "window_control"
    RUNNING_APPS = "running_apps"
    CLOSE_PROCESS = "close_process"
    LOCATE_DEVICE = "locate_device"
    PANIC = "panic"

    # --- Utilities ---
    SCREENSHOT = "screenshot"
    WEBCAM = "webcam"
    CLIPBOARD = "clipboard"
    SET_CLIPBOARD = "set_clipboard"
    TYPE_TEXT = "type_text"
    DOWNLOAD_URL = "download_url"
    NOTIFY = "notify"
    SYSTEM_INFO = "system_info"
    WEB_SEARCH = "web_search"
    WIFI = "wifi"

    # --- Developer assistant (Phase 7) ---
    RUN_DEV_COMMAND = "run_dev_command"
    EXPLAIN_CODE = "explain_code"
    REVIEW_CODE = "review_code"
    GENERATE_CODE = "generate_code"

    # --- Vision (Phase 8) ---
    LOOK_AT_SCREEN = "look_at_screen"
    READ_DOCUMENT = "read_document"
    LIST_WINDOWS = "list_windows"

    # --- Controlled browser ---
    BROWSE = "browse"
    BROWSER_SCROLL = "browser_scroll"
    BROWSER_CLICK = "browser_click"
    BROWSER_TYPE = "browser_type"
    BROWSER_BACK = "browser_back"
    BROWSER_CLOSE = "browser_close"

    # --- Conversation ---
    ANSWER_QUESTION = "answer_question"
    SMALL_TALK = "small_talk"
    UNKNOWN = "unknown"


#: Human-readable description of each intent, used to build the system prompt.
#: Keeping descriptions here rather than in the prompt file means adding an
#: intent updates the model's instructions automatically — the enum and the
#: prompt cannot drift apart.
INTENT_DESCRIPTIONS: dict[IntentType, str] = {
    IntentType.OPEN_APPLICATION: "Launch a program. target = application name.",
    IntentType.CLOSE_APPLICATION: "Quit a running program. target = application name.",
    IntentType.OPEN_WEBSITE: "Open a URL or site in the browser. target = URL or site name.",
    IntentType.OPEN_FOLDER: (
        "Open a directory in the file explorer. target = a folder word (desktop, "
        "downloads, documents, pictures, music, videos) or a path."
    ),
    IntentType.CREATE_FOLDER: (
        "Make a new folder and open it. target = the folder name, optionally "
        "prefixed with a location, e.g. 'projects' or 'documents/tax 2026'."
    ),
    IntentType.SEARCH_FILES: "Find files on disk. target = the search query.",
    IntentType.SEND_FILE: (
        "Send a file from this machine to the person (over Telegram). target = the "
        "file name, or 'latest' for the most recent download."
    ),
    IntentType.SEND_FOLDER: (
        "Zip a whole folder and send it (over Telegram). Use for 'send me my work "
        "folder', 'zip and send downloads'. target = the folder name or path."
    ),
    IntentType.LOCK_SCREEN: "Lock the workstation. No target.",
    IntentType.SLEEP: "Put the machine to sleep. No target.",
    IntentType.RESTART: "Reboot the machine. No target.",
    IntentType.SHUTDOWN: "Power the machine off. No target.",
    IntentType.SET_VOLUME: (
        "Change audio volume. target = 'up', 'down', 'mute', or a number 0-100."
    ),
    IntentType.SET_BRIGHTNESS: (
        "Change screen brightness. target = 'up', 'down', or a number 0-100."
    ),
    IntentType.KEYBOARD_LIGHT: ("Turn the keyboard backlight on or off. target = 'on' or 'off'."),
    IntentType.MEDIA_CONTROL: (
        "Control media playback (any player, including Spotify). target = 'play', "
        "'pause', 'next', 'previous' or 'stop'."
    ),
    IntentType.WINDOW_CONTROL: (
        "Minimise, maximise or restore a window, or minimise everything. target = "
        "the window/app name (or 'all'). Put the action in parameters as key "
        "'action' = minimize|maximize|restore|minimize_all."
    ),
    IntentType.RUNNING_APPS: "List the applications currently running. No target.",
    IntentType.CLOSE_PROCESS: (
        "Force-close a running program by name (any process, not just allowlisted). "
        "target = the program name."
    ),
    IntentType.LOCATE_DEVICE: (
        "Report where this machine is: Wi-Fi network, public IP and rough location. No target."
    ),
    IntentType.PANIC: (
        "Anti-theft: lock the screen, take a webcam photo, and report the network "
        "and location — all sent to the phone. No target."
    ),
    IntentType.SCREENSHOT: "Capture the screen. No target.",
    IntentType.WEBCAM: (
        "Take a photo from the webcam. Use for 'take a webcam picture', 'who is "
        "there', 'show me the camera'. No target."
    ),
    IntentType.WEB_SEARCH: (
        "Search the web and open the results in the browser. target = the search "
        "query, e.g. 'weather in Lagos' or 'python enumerate'."
    ),
    IntentType.WIFI: (
        "Turn Wi-Fi on or off, or report its state. target must be 'on', 'off', or 'status'."
    ),
    IntentType.CLIPBOARD: (
        "Read or write the clipboard. parameters may carry action=read|write and text=..."
    ),
    IntentType.SET_CLIPBOARD: (
        "Put text onto the machine's clipboard from the phone. Use for 'copy this to "
        "my PC', 'set my clipboard to ...', 'paste this on my computer'. target = the "
        "text to copy. This only writes; it never reads the clipboard."
    ),
    IntentType.TYPE_TEXT: (
        "Type text into whatever window is focused on the machine, as if from the "
        "keyboard. Use for 'type ...', 'type this for me', 'write ... into the active "
        "window'. target = the text to type. Does not press Enter."
    ),
    IntentType.DOWNLOAD_URL: (
        "Download a file from a web link straight onto the machine. Use for 'download "
        "this link', 'save this url to downloads'. target = the http(s) URL. An "
        "optional destination folder goes in parameters as key 'location'."
    ),
    IntentType.NOTIFY: "Show a desktop notification. target = the message.",
    IntentType.SYSTEM_INFO: "Report system status such as CPU, memory or battery. No target.",
    IntentType.RUN_DEV_COMMAND: (
        "Run a development command. target must be one of these exact keys: "
        "git.status, git.log, git.diff, git.diff.staged, git.branch, git.add, "
        "git.commit, git.push, git.pull, tests.run, lint.run, format.check, "
        "types.check, docker.ps, docker.images. "
        "For git.commit put the commit message in parameters as key 'message'. "
        "Put the project folder in parameters as key 'directory' if the user named one."
    ),
    IntentType.EXPLAIN_CODE: "Explain what a source file does. target = the file path.",
    IntentType.REVIEW_CODE: "Review a source file for bugs. target = the file path.",
    IntentType.GENERATE_CODE: (
        "Write new code from a description. target = the full description of what to write."
    ),
    IntentType.LOOK_AT_SCREEN: (
        "Answer a question about what is currently on screen. target = the question. "
        "Use this for 'what does this error say', 'what is on my screen', "
        "'which button should I click'."
    ),
    IntentType.READ_DOCUMENT: (
        "Answer a question about a PDF file. target = the file path. "
        "Put the question in parameters as key 'question'."
    ),
    IntentType.LIST_WINDOWS: "List the windows that are currently open. No target.",
    IntentType.BROWSE: (
        "Open a page in the steerable browser and screenshot it. target = a URL, a "
        "site, or a search phrase. Use for 'browse to X', 'open X in the browser'."
    ),
    IntentType.BROWSER_SCROLL: (
        "Scroll the browser page and screenshot it. target = 'up', 'down', 'top' or 'bottom'."
    ),
    IntentType.BROWSER_CLICK: (
        "Click a link or button in the browser by its text, then screenshot. "
        "target = the visible text to click."
    ),
    IntentType.BROWSER_TYPE: (
        "Type into the browser's focused field and press Enter, then screenshot. "
        "target = the text to type."
    ),
    IntentType.BROWSER_BACK: "Go back one page in the browser, then screenshot. No target.",
    IntentType.BROWSER_CLOSE: "Close the steerable browser. No target.",
    IntentType.ANSWER_QUESTION: (
        "The user asked a question needing knowledge, not a machine action. target = the question."
    ),
    IntentType.SMALL_TALK: "Greeting or conversational filler with no action. No target.",
    IntentType.UNKNOWN: (
        "The request does not match any other intent, or is too ambiguous to classify. "
        "Prefer this over guessing."
    ),
}

#: Intents that produce a reply rather than an effect on the machine.
#:
#: One definition, used in four places: ``Intent.is_actionable``, the confirmation
#: policy (which exempts them, since there is nothing to approve), the capability
#: list the assistant is given about itself, and the command registry. Written out
#: separately in each, they would eventually disagree — and a conversational intent
#: that one site thinks is actionable is exactly how a "reply" acquires a side
#: effect.
NON_ACTIONABLE: frozenset[IntentType] = frozenset(
    {
        IntentType.ANSWER_QUESTION,
        IntentType.SMALL_TALK,
        IntentType.UNKNOWN,
    }
)

#: Intents that are disruptive or hard to reverse, and must be confirmed by the
#: user before Phase 3 executes them — regardless of how confident the model is.
#: This directly implements the "never execute dangerous actions without
#: confirmation" requirement, and it is enforced in code rather than by prompt.
CONFIRMATION_REQUIRED: frozenset[IntentType] = frozenset(
    {
        IntentType.SHUTDOWN,
        IntentType.RESTART,
        IntentType.SLEEP,
        IntentType.CLOSE_APPLICATION,
    }
)


class IntentParameter(BaseModel):
    """One extra key/value detail extracted from the utterance.

    Modelled as a list of pairs rather than a free-form ``dict`` because
    structured outputs require ``additionalProperties: false`` on every object;
    an open-ended mapping is not expressible in that schema. A list of fixed
    objects is, and it round-trips to a dict via ``Intent.parameters_as_dict``.

    Attributes:
        key: Parameter name, lower_snake_case.
        value: Parameter value as text.
    """

    key: str
    value: str


class IntentClassification(BaseModel):
    """What the language model returns for one utterance.

    This is the schema handed to the provider, so every field must be something
    a classifier can legitimately decide. Security policy is deliberately absent.

    Attributes:
        intent: The classified action.
        target: Primary object of the action, or ``None`` if the intent takes none.
        parameters: Additional extracted details.
        confidence: The model's confidence, 0.0 to 1.0.
        reasoning: One short sentence explaining the classification.
    """

    intent: IntentType
    target: str | None = None
    parameters: list[IntentParameter] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class Intent(IntentClassification):
    """A classification with Quainex's policy decision attached.

    Attributes:
        requires_confirmation: Whether Phase 3 must ask the user before acting.
        utterance: What the user actually said. Carried here rather than left
            behind because the conversational intents need it: ``SMALL_TALK`` has
            no ``target`` by definition, so without this a handler for "how are
            you?" would receive nothing to reply to. It also makes the audit trail
            answer "what was said" and not only "what was decided".

            Set by the Brain, never by the model — it is deliberately absent from
            ``IntentClassification`` so that a classifier cannot rewrite the
            request it was given.
    """

    requires_confirmation: bool
    utterance: str = ""

    @property
    def subject(self) -> str:
        """The text a conversational handler should respond to.

        Returns:
            The target when the classifier extracted one, otherwise the original
            utterance.
        """
        return (self.target or "").strip() or self.utterance.strip()

    def parameters_as_dict(self) -> dict[str, str]:
        """Return the parameter list as a mapping.

        Later keys win on duplicates, which the model should not produce but
        which must not raise if it does.

        Returns:
            Parameters keyed by name.
        """
        return {p.key: p.value for p in self.parameters}

    @property
    def is_actionable(self) -> bool:
        """Whether this intent maps to a machine action rather than conversation."""
        return self.intent not in NON_ACTIONABLE
