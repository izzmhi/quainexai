"""Tests for authentication, confirmation tokens and rate limiting.

The load-bearing tests in this file are the ones asserting that things *cannot*
happen: that a caller cannot declare its own confirmation, that a token issued
for one action cannot authorise another, and that Quainex refuses to start
exposed to the network without a password.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from quainex.api.app import create_app
from quainex.auth import TokenService, hash_password, verify_password
from quainex.config.settings import Settings
from quainex.core.brain import Intent, IntentType
from quainex.core.commands import CommandStatus, build_executor
from quainex.core.exceptions import AuthenticationError
from quainex.security import ConfirmationService, RateLimiter
from tests.test_commands import FakeDesktopController

if TYPE_CHECKING:
    from collections.abc import Iterator

PASSWORD = "a-sufficiently-long-password"
SECRET = "x" * 48


def _intent(
    intent: IntentType = IntentType.SHUTDOWN,
    target: str | None = None,
    *,
    requires_confirmation: bool = True,
) -> Intent:
    return Intent(
        intent=intent,
        target=target,
        confidence=0.99,
        reasoning="test",
        requires_confirmation=requires_confirmation,
    )


# -- password hashing ------------------------------------------------------


def test_password_round_trip():
    stored = hash_password(PASSWORD)
    assert verify_password(PASSWORD, stored) is True
    assert verify_password("not the password", stored) is False


def test_hash_does_not_contain_the_password():
    assert PASSWORD not in hash_password(PASSWORD)


def test_same_password_hashes_differently_each_time():
    # Distinct salts: identical hashes would reveal that two accounts share a
    # password, and would make a precomputed table worth building.
    assert hash_password(PASSWORD) != hash_password(PASSWORD)


@pytest.mark.parametrize("corrupt", ["", "nonsense", "scrypt$bad$format", "argon2$1$2$3$4"])
def test_corrupt_hashes_fail_closed(corrupt):
    assert verify_password(PASSWORD, corrupt) is False


def test_empty_password_is_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        hash_password("")


# -- access tokens ---------------------------------------------------------


def test_token_round_trip():
    service = TokenService(SECRET, ttl_minutes=60)
    token, expires_in = service.issue()

    assert service.verify(token) == "owner"
    assert expires_in == 3600


def test_token_signed_with_another_secret_is_rejected():
    issued = TokenService(SECRET, 60).issue()[0]
    other = TokenService("y" * 48, 60)

    with pytest.raises(AuthenticationError):
        other.verify(issued)


@pytest.mark.parametrize("bad", ["", "not-a-token", "a.b.c"])
def test_malformed_tokens_are_rejected(bad):
    with pytest.raises(AuthenticationError):
        TokenService(SECRET, 60).verify(bad)


def test_expired_token_is_rejected():
    service = TokenService(SECRET, ttl_minutes=1)
    token, _ = service.issue()

    # Rebuild with a negative lifetime to mint an already-dead token.
    expired = TokenService(SECRET, ttl_minutes=1)
    expired._ttl = -abs(expired._ttl)
    dead, _ = expired.issue()

    assert service.verify(token) == "owner"
    with pytest.raises(AuthenticationError, match="expired"):
        service.verify(dead)


def test_short_secrets_are_refused():
    # A guessable signing secret makes every token forgeable.
    with pytest.raises(ValueError, match="at least 32"):
        TokenService("short", 60)


# -- confirmation tokens: the Phase 3 hole, closed ------------------------


def test_confirmation_token_authorises_its_own_action():
    service = ConfirmationService(SECRET)
    intent = _intent()

    token = service.issue(intent)
    assert service.verify(token, intent) is True


def test_confirmation_token_cannot_authorise_a_different_action():
    # The binding is the point. A token that merely proved "the user confirmed
    # something" would be a skeleton key for whatever was asked next.
    service = ConfirmationService(SECRET)
    token = service.issue(_intent(IntentType.CLOSE_APPLICATION, "Spotify"))

    assert service.verify(token, _intent(IntentType.SHUTDOWN)) is False


def test_confirmation_token_is_bound_to_its_target():
    service = ConfirmationService(SECRET)
    token = service.issue(_intent(IntentType.CLOSE_APPLICATION, "Spotify"))

    assert service.verify(token, _intent(IntentType.CLOSE_APPLICATION, "Chrome")) is False


def test_confirmation_token_is_single_use():
    service = ConfirmationService(SECRET)
    intent = _intent()
    token = service.issue(intent)

    assert service.verify(token, intent) is True
    assert service.verify(token, intent) is False, "a replayed token must not work"


def test_confirmation_token_from_another_secret_is_rejected():
    intent = _intent()
    forged = ConfirmationService("z" * 48).issue(intent)

    assert ConfirmationService(SECRET).verify(forged, intent) is False


@pytest.mark.parametrize("bad", ["", "garbage", "a.b", "...."])
def test_malformed_confirmation_tokens_are_rejected(bad):
    assert ConfirmationService(SECRET).verify(bad, _intent()) is False


def test_expired_confirmation_token_is_rejected():
    service = ConfirmationService(SECRET, ttl_seconds=-1)
    intent = _intent()

    assert service.verify(service.issue(intent), intent) is False


# -- executor integration --------------------------------------------------


def _executor(tmp_path: Path, desktop: FakeDesktopController):
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        log_dir=tmp_path / "logs",
        database_path=tmp_path / "t.db",
        command_search_roots=[tmp_path],
        allow_destructive_commands=True,
    )
    return build_executor(desktop, settings, ConfirmationService(SECRET))


async def test_refusal_hands_back_a_usable_token(tmp_path):
    desktop = FakeDesktopController()
    executor = _executor(tmp_path, desktop)
    intent = _intent()

    refused = await executor.execute(intent)
    assert refused.status is CommandStatus.REQUIRES_CONFIRMATION
    assert refused.confirmation_token, "the caller needs a way to proceed"
    assert desktop.calls == []

    allowed = await executor.execute(intent, confirmation_token=refused.confirmation_token)
    assert allowed.status is CommandStatus.SUCCEEDED
    assert desktop.actions == ["shutdown"]


async def test_a_forged_token_does_not_execute(tmp_path):
    desktop = FakeDesktopController()
    executor = _executor(tmp_path, desktop)

    result = await executor.execute(_intent(), confirmation_token="obviously-made-up")

    assert result.status is CommandStatus.REQUIRES_CONFIRMATION
    assert desktop.calls == []


async def test_a_token_for_one_action_does_not_execute_another(tmp_path):
    desktop = FakeDesktopController()
    executor = _executor(tmp_path, desktop)

    close_token = (
        await executor.execute(_intent(IntentType.CLOSE_APPLICATION, "Spotify"))
    ).confirmation_token
    assert close_token is not None

    result = await executor.execute(_intent(IntentType.SHUTDOWN), confirmation_token=close_token)

    assert result.status is CommandStatus.REQUIRES_CONFIRMATION
    assert desktop.calls == [], "a close-Spotify approval must never power off the machine"


# -- the API surface -------------------------------------------------------


def test_execute_endpoint_no_longer_accepts_a_confirmed_flag(client: TestClient):
    # The Phase 3 regression guard: `confirmed: true` was a caller asserting the
    # user agreed. It must now be inert.
    desktop = FakeDesktopController()
    container = client.app.state.container
    container.desktop = desktop
    container.commands = build_executor(desktop, container.settings, container.confirmations)

    response = client.post(
        "/commands/execute",
        json={
            "intent": _intent(IntentType.CLOSE_APPLICATION, "Spotify").model_dump(mode="json"),
            "confirmed": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "requires_confirmation"
    assert body["executed"] is False
    assert desktop.calls == [], "asserting confirmation must not be enough"


def test_execute_endpoint_accepts_the_issued_token(client: TestClient):
    desktop = FakeDesktopController()
    container = client.app.state.container
    container.desktop = desktop
    container.commands = build_executor(desktop, container.settings, container.confirmations)

    intent = _intent(IntentType.CLOSE_APPLICATION, "Spotify").model_dump(mode="json")
    refused = client.post("/commands/execute", json={"intent": intent}).json()

    executed = client.post(
        "/commands/execute",
        json={"intent": intent, "confirmation_token": refused["confirmation_token"]},
    ).json()

    assert executed["status"] == "succeeded"
    assert desktop.actions == ["close_application"]


# -- auth is derived from the bind address --------------------------------


def test_loopback_does_not_require_auth():
    settings = Settings(_env_file=None, host="127.0.0.1")  # type: ignore[call-arg]
    assert settings.is_loopback is True
    assert settings.auth_required is False


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.50", "10.0.0.4"])  # noqa: S104
def test_exposing_without_credentials_refuses_to_start(host):
    # The whole design: "exposed with auth off" is not a reachable configuration.
    with pytest.raises(ValidationError, match="Authentication is required"):
        Settings(_env_file=None, host=host)  # type: ignore[call-arg]


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.50"])  # noqa: S104
def test_exposing_with_credentials_is_allowed(host):
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        host=host,
        auth_secret=SECRET,
        auth_password_hash=hash_password(PASSWORD),
    )
    assert settings.auth_required is True


def test_auth_can_be_forced_on_for_loopback():
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        host="127.0.0.1",
        require_auth=True,
        auth_secret=SECRET,
        auth_password_hash=hash_password(PASSWORD),
    )
    assert settings.auth_required is True


def test_forcing_auth_on_without_credentials_also_refuses():
    with pytest.raises(ValidationError, match="Authentication is required"):
        Settings(_env_file=None, host="127.0.0.1", require_auth=True)  # type: ignore[call-arg]


# -- the guarded API -------------------------------------------------------


@pytest.fixture
def secured_client(tmp_path: Path) -> Iterator[TestClient]:
    """A client for an instance that enforces authentication."""
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        host="127.0.0.1",
        require_auth=True,
        auth_secret=SECRET,
        auth_password_hash=hash_password(PASSWORD),
        log_dir=tmp_path / "logs",
        database_path=tmp_path / "secured.db",
    )
    with TestClient(create_app(settings), raise_server_exceptions=False) as client:
        yield client


def test_health_stays_open(secured_client: TestClient):
    # A supervisor must be able to check liveness without a credential.
    assert secured_client.get("/health").status_code == 200


@pytest.mark.parametrize(
    "path", ["/commands", "/memory/preferences", "/memory/activity", "/voice/status"]
)
def test_protected_routes_reject_anonymous_callers(secured_client: TestClient, path):
    assert secured_client.get(path).status_code == 401


def test_wrong_password_is_rejected(secured_client: TestClient):
    response = secured_client.post("/auth/token", json={"password": "wrong"})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "authentication_failed"


def test_correct_password_yields_a_working_token(secured_client: TestClient):
    token = secured_client.post("/auth/token", json={"password": PASSWORD}).json()["access_token"]

    response = secured_client.get("/commands", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200


def test_garbage_token_is_rejected(secured_client: TestClient):
    response = secured_client.get("/commands", headers={"Authorization": "Bearer nonsense"})
    assert response.status_code == 401


def test_session_endpoint_reports_the_subject(secured_client: TestClient):
    token = secured_client.post("/auth/token", json={"password": PASSWORD}).json()["access_token"]
    body = secured_client.get("/auth/session", headers={"Authorization": f"Bearer {token}"}).json()

    assert body["subject"] == "owner"
    assert body["auth_required"] is True


def test_websocket_rejects_unauthenticated_clients(secured_client: TestClient):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect), secured_client.websocket_connect("/ws") as socket:
        socket.receive_json()


def test_websocket_accepts_a_token_in_the_query_string(secured_client: TestClient):
    # Browsers cannot set headers on a WebSocket, so the query string is the
    # only option available to a web dashboard.
    token = secured_client.post("/auth/token", json={"password": PASSWORD}).json()["access_token"]

    with secured_client.websocket_connect(f"/ws?token={token}") as socket:
        assert socket.receive_json()["type"] == "welcome"


def test_local_instances_are_not_guarded(client: TestClient):
    # Loopback with no credentials configured: usable without ceremony.
    assert client.get("/commands").status_code == 200
    assert client.get("/auth/session").json()["auth_required"] is False


# -- rate limiting ---------------------------------------------------------


def test_rate_limiter_allows_up_to_the_limit():
    limiter = RateLimiter(limit_per_minute=3)
    assert [limiter.check("client") for _ in range(3)] == [True, True, True]
    assert limiter.check("client") is False


def test_rate_limiter_tracks_clients_separately():
    limiter = RateLimiter(limit_per_minute=1)
    assert limiter.check("phone") is True
    assert limiter.check("laptop") is True
    assert limiter.check("phone") is False


def test_rate_limiter_reports_remaining():
    limiter = RateLimiter(limit_per_minute=5)
    limiter.check("client")
    assert limiter.remaining("client") == 4
