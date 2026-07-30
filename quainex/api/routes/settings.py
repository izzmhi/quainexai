r"""Credential and provider configuration endpoints.

Purpose:
    Let the dashboard read which AI providers are configured, save an API key,
    clear one, and prove a key works — without the user opening a text editor.

Security posture (the part that matters here):
    This router hands out the ability to write credentials, so it is the most
    sensitive surface in Quainex after command execution. Four rules:

    1. **No key is ever returned.** ``GET`` reports ``configured: true`` and a
       masked hint derived from the length. There is no endpoint that reveals a
       stored value, so a stolen access token cannot be turned into a stolen API
       key — it can only overwrite one.
    2. **The name is an allowlist**, enforced in the vault, not here. A caller
       cannot invent a secret name and use Quainex as an arbitrary key/value
       store on the user's disk.
    3. **Writes are refused when not bound to loopback unless authenticated** —
       inherited automatically, because the router is registered behind the same
       ``require_auth`` dependency as everything else.
    4. **Testing a key is a real call to the provider**, with the key that is
       already stored. There is no "test this value" endpoint that would accept a
       key in a request body and echo back what it learned.

Why hot-reload rather than "restart to apply":
    A settings page that saves successfully and then does nothing until the user
    finds and restarts the process is a bug wearing a success message. Saving
    calls ``Container.reload_ai_providers()``, which swaps the chain in place, so
    the very next request uses the new key.

Architecture:
    GET    /settings/providers   -> vault status + live chain state
    PUT    /settings/providers/{name}  -> store -> reload chain -> new status
    DELETE /settings/providers/{name}  -> clear -> reload chain -> new status
    POST   /settings/providers/test    -> one real completion through the chain

Dependencies:
    fastapi, pydantic, quainex.security.vault, quainex.core.container

Example:
    $ curl -X PUT http://127.0.0.1:8000/settings/providers/gemini_api_key \\
        -H 'content-type: application/json' -d '{"value":"AIza..."}'

Future improvements:
    * Reorder the provider chain from the dashboard (currently ``.env`` only).
    * Per-provider quota and spend counters, so "which key am I burning" is
      answerable without opening a vendor console.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from quainex.api.dependencies import ContainerDep
from quainex.core.exceptions import ProviderError
from quainex.integrations.telegram import TELEGRAM_BLOCKED_INTENTS
from quainex.security.vault import STORABLE_SECRETS, CredentialVault
from quainex.services.ai.provider import ChatMessage

if TYPE_CHECKING:
    from quainex.config.settings import Settings
    from quainex.core.container import Container

router = APIRouter(prefix="/settings", tags=["settings"])

#: What each storable secret is for, and where to get one. The dashboard renders
#: these, so the user is never asked to paste a key without being told which page
#: to get it from — the most common reason a settings screen goes unused.
_SECRET_GUIDE: dict[str, dict[str, str]] = {
    "groq_api_key": {
        "label": "Groq",
        "detail": "Free tier, fastest responses. Tried first.",
        "url": "https://console.groq.com/keys",
        "prefix": "gsk_",
    },
    "gemini_api_key": {
        "label": "Google Gemini",
        "detail": "Free tier. Handles screen and PDF questions.",
        "url": "https://aistudio.google.com/apikey",
        "prefix": "AIza",
    },
    "openrouter_api_key": {
        "label": "OpenRouter",
        "detail": "Free tier. One key reaches many vendors.",
        "url": "https://openrouter.ai/keys",
        "prefix": "sk-or-",
    },
    "anthropic_api_key": {
        "label": "Anthropic Claude",
        "detail": "Paid. Strongest reasoning; used as the fallback.",
        "url": "https://console.anthropic.com/settings/keys",
        "prefix": "sk-ant-",
    },
    "local_api_key": {
        "label": "Local / self-hosted",
        # Stated plainly because a key alone does nothing here: this slot points
        # at a server only the user knows about, so it needs a URL in .env too.
        # Without saying so, a pasted key looks accepted and silently never runs.
        "detail": "Ollama or LM Studio. Also needs QUAINEX_LOCAL_BASE_URL in .env.",
        "url": "",
        "prefix": "",
    },
    "telegram_bot_token": {
        "label": "Telegram bot",
        "detail": "Control Quainex from your phone. Get one from @BotFather.",
        "url": "https://t.me/BotFather",
        "prefix": "",
    },
}


class SecretState(BaseModel):
    """Whether one credential is set, and where it came from.

    Attributes:
        name: The credential's settings name.
        label: Human-readable provider name.
        detail: One line on what it does.
        url: Where to obtain a key. Empty when there is nowhere to send the user.
        configured: Whether a value is available from any source.
        source: ``vault`` when saved here, ``env`` when from ``.env``, ``none``
            when unset. Shown so the vault-wins precedence is never a mystery.
        hint: A masked stand-in, never any part of the real value.
    """

    name: str
    label: str
    detail: str
    url: str
    configured: bool
    source: str
    hint: str


class SettingsResponse(BaseModel):
    """The dashboard's view of provider configuration.

    Attributes:
        secrets: Every storable credential, configured or not.
        chain: Provider chain in preference order, as ``/health`` reports it.
        chain_available: Whether any provider can currently serve a request.
        vault_writable: Whether this platform can store credentials. ``False``
            means the dashboard must tell the user to edit ``.env`` instead.
        vault_path: Where the encrypted file lives, so it can be found or deleted.
    """

    secrets: list[SecretState]
    chain: list[dict[str, object]]
    chain_available: bool
    vault_writable: bool
    vault_path: str


class SecretUpdate(BaseModel):
    """A credential being stored.

    Attributes:
        value: The plaintext key. Bounded to reject a pasted document rather than
            a key, and to keep an oversized body from reaching the vault.
    """

    value: str = Field(min_length=8, max_length=512)


class TestResult(BaseModel):
    """Outcome of a live call through the provider chain.

    Attributes:
        ok: Whether a provider answered.
        provider: The chain that was used.
        reply: The model's answer, truncated. Present only on success.
        error: What went wrong, naming every provider tried. Present on failure.
    """

    ok: bool
    provider: str
    reply: str | None = None
    error: str | None = None


def _mask(length: int) -> str:
    """Build a masked hint for a value of the given length.

    Deliberately reveals nothing — not even the first characters, which for most
    vendors identify the account tier and are enough to confirm a guessed key.

    Args:
        length: Number of characters in the stored value.

    Returns:
        A fixed-width mask carrying only the length.
    """
    return f"{'•' * 12} ({length} characters)"


def _env_source(settings: Settings, name: str) -> bool:
    """Whether ``.env`` or the process environment supplies this credential.

    Args:
        settings: Settings *before* vault credentials were overlaid.
        name: The credential name.

    Returns:
        Whether a non-empty value is present.
    """
    value = getattr(settings, name, None)
    if value is None:
        return False
    secret = getattr(value, "get_secret_value", None)
    return bool(secret().strip()) if callable(secret) else bool(str(value).strip())


def _snapshot(container: Container) -> SettingsResponse:
    """Build the current configuration view.

    Args:
        container: The application container.

    Returns:
        Everything the dashboard's settings panel needs.
    """
    vault = CredentialVault(container.base_settings.credentials_path)
    stored = vault.status()

    secrets: list[SecretState] = []
    # Iterated in guide order, not sorted: the sequence is the *preference* order
    # the chain uses, and a settings page that lists the paid provider first
    # implies the wrong thing about what Quainex reaches for.
    for name, guide in _SECRET_GUIDE.items():
        if name not in STORABLE_SECRETS:  # pragma: no cover - kept in step below
            continue
        in_vault = name in stored
        in_env = _env_source(container.base_settings, name)
        secrets.append(
            SecretState(
                name=name,
                label=guide["label"],
                detail=guide["detail"],
                url=guide["url"],
                configured=in_vault or in_env,
                source="vault" if in_vault else "env" if in_env else "none",
                hint=_mask(stored[name]) if in_vault else "set in .env" if in_env else "",
            )
        )

    describe = getattr(container.ai_provider, "describe", None)
    chain: list[dict[str, object]] = describe() if callable(describe) else []

    return SettingsResponse(
        secrets=secrets,
        chain=chain,
        chain_available=container.ai_provider.is_available,
        vault_writable=vault.is_writable,
        vault_path=str(vault.path),
    )


async def _apply_stored_credential(container: Container, name: str) -> None:
    """Make a just-changed credential take effect.

    Which subsystem to rebuild depends on the credential, and getting this wrong
    is invisible: the save returns 200 and nothing happens. The Telegram token had
    exactly that bug — it reloaded the AI provider chain, which does not hold it.

    Args:
        container: The application container.
        name: The credential that changed.
    """
    if name == "telegram_bot_token":
        await container.reload_telegram()
    else:
        await container.reload_ai_providers()


@router.get("/providers", response_model=SettingsResponse, summary="Read provider configuration")
async def read_providers(container: ContainerDep) -> SettingsResponse:
    """Report which credentials are set and how the chain is ordered.

    Args:
        container: Injected application container.

    Returns:
        The configuration snapshot. Never includes a credential value.
    """
    return _snapshot(container)


@router.put(
    "/providers/{name}",
    response_model=SettingsResponse,
    summary="Store a credential and reload the chain",
)
async def store_credential(
    name: str, payload: SecretUpdate, container: ContainerDep
) -> SettingsResponse:
    """Encrypt and store one credential, then rebuild the provider chain.

    Args:
        name: The credential to set. Must be on the vault's allowlist.
        payload: The value to store.
        container: Injected application container.

    Returns:
        The configuration snapshot after the change, so the dashboard can render
        the new state without a second request.

    Raises:
        UnknownSecretError: ``name`` is not storable.
        VaultUnavailableError: This platform cannot encrypt credentials.
    """
    vault = CredentialVault(container.base_settings.credentials_path)
    vault.set(name, payload.value)
    await _apply_stored_credential(container, name)
    return _snapshot(container)


@router.delete(
    "/providers/{name}",
    response_model=SettingsResponse,
    summary="Clear a credential and reload the chain",
)
async def clear_credential(name: str, container: ContainerDep) -> SettingsResponse:
    """Remove one stored credential, then rebuild the provider chain.

    Clearing a credential that is also set in ``.env`` reveals the ``.env`` value
    again rather than disabling the provider — the snapshot's ``source`` field
    shows this, so the outcome is visible.

    Args:
        name: The credential to remove.
        container: Injected application container.

    Returns:
        The configuration snapshot after the change.

    Raises:
        UnknownSecretError: ``name`` is not storable.
        VaultUnavailableError: This platform cannot encrypt credentials.
    """
    vault = CredentialVault(container.base_settings.credentials_path)
    vault.delete(name)
    await _apply_stored_credential(container, name)
    return _snapshot(container)


class TelegramState(BaseModel):
    """Everything the dashboard needs to finish, verify or stop phone control.

    Attributes:
        token_configured: Whether a bot token is set. The token itself is never
            returned.
        allowed_users: The user ids permitted to drive this machine. Returned in
            full, unlike the token — an authorisation list you cannot inspect is
            not one you can trust.
        configured: Whether both halves are present. Polling is refused otherwise.
        running: Whether the bridge is currently polling.
        blocked_intents: Commands the bridge refuses regardless of who asks.
        missing: What is still needed, in plain words. Present so the dashboard
            never has to show "not configured" without saying why.
    """

    token_configured: bool
    allowed_users: list[int]
    configured: bool
    running: bool
    blocked_intents: list[str]
    missing: list[str]


class TelegramAllowlist(BaseModel):
    """The set of Telegram accounts permitted to control this machine.

    Attributes:
        user_ids: Numeric Telegram user ids. An empty list clears the allowlist,
            which disables the bridge — the honest way to turn phone control off
            without deleting the bot.
    """

    # Bounded: this is a personal machine, and a list of a thousand ids is a
    # configuration mistake rather than a use case.
    user_ids: list[int] = Field(default_factory=list, max_length=32)


def _telegram_state(container: Container) -> TelegramState:
    """Build the phone-control status view.

    Args:
        container: The application container.

    Returns:
        Current state, including what is still missing.
    """
    bridge = container.telegram
    settings = container.settings
    allowed = list(settings.telegram_allowed_users)
    has_token = bool(settings.telegram_bot_token)

    missing: list[str] = []
    if not has_token:
        missing.append("a bot token from @BotFather")
    if not allowed:
        missing.append("at least one user id from @userinfobot")

    return TelegramState(
        token_configured=has_token,
        allowed_users=allowed,
        configured=bridge.is_configured,
        running=bridge.is_running,
        # Read from the constant rather than out of `bridge.status()`, whose values
        # are typed `object` — the type would have to be asserted away, and the
        # constant is the actual source of truth.
        blocked_intents=sorted(intent.value for intent in TELEGRAM_BLOCKED_INTENTS),
        missing=missing,
    )


@router.get("/telegram", response_model=TelegramState, summary="Read phone-control setup")
async def read_telegram(container: ContainerDep) -> TelegramState:
    """Report whether phone control is set up, and what is missing if not.

    Args:
        container: Injected application container.

    Returns:
        The current state.
    """
    return _telegram_state(container)


@router.post(
    "/telegram/test",
    summary="Check the bot token and find your Telegram user id",
)
async def test_telegram(container: ContainerDep) -> dict[str, object]:
    """Verify the token with Telegram and list anyone who has messaged the bot.

    Returns 200 with ``ok: false`` on failure rather than an error status, for the
    same reason as the provider test: "your token is wrong" and "Telegram is
    unreachable" need different responses from the user, and an error envelope
    flattens them into one red box.

    The candidate list is **not** an allowlist. Nothing is authorised until a
    human picks from it — auto-allowing whoever messaged first would let a
    stranger who found the bot grant themselves control of the machine.

    Args:
        container: Injected application container.

    Returns:
        Whether the token works, the bot's username, and candidate user ids.
    """
    return await container.telegram.diagnose()


@router.put(
    "/telegram/allowed-users",
    response_model=TelegramState,
    summary="Set who may control this machine from Telegram",
)
async def set_telegram_allowlist(
    payload: TelegramAllowlist, container: ContainerDep
) -> TelegramState:
    """Replace the Telegram allowlist and rebuild the bridge.

    Replaces rather than appends, deliberately: revoking access must be one
    request, not a delete endpoint someone forgets to call. The bridge is rebuilt
    immediately, so a removed id stops being obeyed now rather than at the next
    restart — for an authorisation list, "eventually" is not good enough.

    Args:
        payload: The ids to permit. An empty list disables the bridge.
        container: Injected application container.

    Returns:
        The state after the change.

    Raises:
        VaultUnavailableError: This platform cannot store the list.
    """
    vault = CredentialVault(container.base_settings.credentials_path)
    if payload.user_ids:
        vault.set("telegram_allowed_users", ",".join(str(uid) for uid in payload.user_ids))
    else:
        vault.delete("telegram_allowed_users")

    await container.reload_telegram()
    return _telegram_state(container)


@router.post(
    "/providers/test",
    response_model=TestResult,
    status_code=status.HTTP_200_OK,
    summary="Prove the configured chain can reach a model",
)
async def test_providers(container: ContainerDep) -> TestResult:
    """Send one trivial prompt through the chain and report what happened.

    Returns 200 with ``ok: false`` rather than an error status. A failed
    *provider* is not a failed *request* — the dashboard needs the diagnostic
    text to show the user, and an error envelope would flatten "Groq rejected the
    key" and "no provider is configured" into the same red box.

    Args:
        container: Injected application container.

    Returns:
        Success with the model's reply, or failure with the reason.
    """
    try:
        reply = await container.ai_provider.complete(
            messages=[ChatMessage(role="user", content="Reply with exactly: ready")],
            system="You are being health-checked. Answer in one word.",
            max_tokens=32,
        )
    except ProviderError as exc:
        return TestResult(ok=False, provider=container.ai_provider.name, error=exc.message)

    return TestResult(ok=True, provider=container.ai_provider.name, reply=reply[:200])
