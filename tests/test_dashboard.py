"""Tests for the dashboard mount and the provider-settings endpoints.

Two things are being defended here. The first is boring but was already a real
bug: a static mount must not shadow the API. The second is not boring at all —
this router can *write credentials*, so the tests assert the properties that make
that safe rather than merely that it works.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from quainex.api.app import create_app
from quainex.config.settings import Settings

#: Writing a credential goes through the platform's real cipher, which only
#: exists on Windows. The read-only assertions run everywhere.
windows_only = pytest.mark.skipif(
    sys.platform != "win32", reason="Credential storage requires Windows DPAPI."
)


# -- serving the interface -------------------------------------------------


def test_the_root_redirects_to_the_interface(client: TestClient):
    response = client.get("/", follow_redirects=False)

    assert response.status_code in {307, 308}
    # Trailing slash matters: without it the page's relative asset links would
    # resolve against the root rather than /ui/.
    assert response.headers["location"] == "/ui/"


def test_the_interface_is_served(client: TestClient):
    response = client.get("/ui/")

    assert response.status_code == 200
    assert "QUAINEX" in response.text
    assert "app.js" in response.text


@pytest.mark.parametrize("asset", ["app.css", "app.js"])
def test_the_assets_are_served(client: TestClient, asset: str):
    assert client.get(f"/ui/{asset}").status_code == 200


def test_the_interface_needs_no_token(client: TestClient):
    """The page is how you reach the login prompt, so gating it is circular.

    Safe because the files are static and carry no data — every byte of state
    they display comes from a router that *is* guarded.
    """
    assert client.get("/ui/").status_code == 200


def test_the_mount_does_not_shadow_the_api(client: TestClient):
    """The reason the mount is at /ui and not /.

    A static mount is a catch-all for everything beneath its path and Starlette
    matches in registration order, so a root mount silently turns every route
    registered after it into a 404 from the file handler. This asserts the
    ordering that prevents that.
    """
    assert client.get("/health").status_code == 200
    assert client.get("/commands").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_a_missing_asset_is_a_404_not_the_index(client: TestClient):
    """`html=True` must not turn every typo into a 200."""
    assert client.get("/ui/does-not-exist.js").status_code == 404


def test_the_api_still_works_with_no_dashboard_on_disk(tmp_path: Path, settings: Settings):
    """An operator who deletes the directory gets an API, not a crash."""
    headless = settings.model_copy(update={"dashboard_dir": tmp_path / "absent"})

    with TestClient(create_app(headless)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ui/").status_code == 404


def test_serving_can_be_switched_off(settings: Settings):
    """A deployment that wants API-only should need no code change."""
    api_only = settings.model_copy(update={"serve_dashboard": False})

    with TestClient(create_app(api_only)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ui/").status_code == 404


# -- reading provider configuration ---------------------------------------


def test_configuration_lists_every_credential_including_unset_ones(client: TestClient):
    """An input for a key you have not set yet is the whole point of the page."""
    body = client.get("/settings/providers").json()

    names = [secret["name"] for secret in body["secrets"]]
    assert "groq_api_key" in names
    assert "gemini_api_key" in names
    assert "anthropic_api_key" in names
    assert all(secret["configured"] is False for secret in body["secrets"])


def test_the_listed_order_is_the_preference_order(client: TestClient):
    """A settings page that lists the paid provider first implies the wrong thing."""
    body = client.get("/settings/providers").json()
    names = [secret["name"] for secret in body["secrets"]]

    assert names.index("groq_api_key") < names.index("anthropic_api_key")
    assert names.index("gemini_api_key") < names.index("anthropic_api_key")


def test_every_credential_says_where_to_get_one(client: TestClient):
    """Being asked to paste a key with no idea where from is why these go unused."""
    body = client.get("/settings/providers").json()

    for secret in body["secrets"]:
        assert secret["label"]
        assert secret["detail"]
        if secret["name"] in {"groq_api_key", "gemini_api_key", "anthropic_api_key"}:
            assert secret["url"].startswith("https://")


def test_configuration_reports_the_chain_with_unavailable_providers_included(client: TestClient):
    body = client.get("/settings/providers").json()

    assert body["chain_available"] is False
    assert [entry["order"] for entry in body["chain"]] == list(range(len(body["chain"])))
    assert all(entry["available"] is False for entry in body["chain"])


def test_an_env_supplied_key_is_reported_as_coming_from_env(settings: Settings):
    """Vault-wins precedence must never be a mystery."""
    # `SecretStr`, not a bare string: `model_copy` does not coerce, and a plain
    # string in a `SecretStr` field would break every reader of it.
    with_env = settings.model_copy(update={"groq_api_key": SecretStr("gsk_from_dot_env")})

    with TestClient(create_app(with_env)) as client:
        body = client.get("/settings/providers").json()
        groq = next(s for s in body["secrets"] if s["name"] == "groq_api_key")

        assert groq["configured"] is True
        assert groq["source"] == "env"
        assert "gsk_from_dot_env" not in str(body)


# -- writing credentials ---------------------------------------------------


@windows_only
def test_saving_a_key_configures_the_provider_without_a_restart(client: TestClient):
    """A settings page that saves and then does nothing is a bug wearing a tick."""
    before = client.get("/settings/providers").json()
    assert before["chain_available"] is False

    after = client.put(
        "/settings/providers/gemini_api_key", json={"value": "AIza_test_value_here"}
    ).json()

    gemini = next(s for s in after["secrets"] if s["name"] == "gemini_api_key")
    assert gemini["configured"] is True
    assert gemini["source"] == "vault"
    # The chain was rebuilt in place, so the very next request would use the key.
    assert after["chain_available"] is True


@windows_only
def test_no_endpoint_ever_returns_a_stored_key(client: TestClient):
    """The property that makes a stolen access token less than a stolen API key.

    Asserted against the whole serialised response rather than a named field, so
    a future field that leaks the value fails this test rather than sliding past
    a check on `hint`.
    """
    sentinel = "AIza_unique_sentinel_0123456789"
    client.put("/settings/providers/gemini_api_key", json={"value": sentinel})

    for response in (
        client.get("/settings/providers"),
        client.get("/health"),
        client.get("/openapi.json"),
    ):
        assert sentinel not in response.text


@windows_only
def test_the_hint_carries_only_a_length(client: TestClient):
    body = client.put(
        "/settings/providers/gemini_api_key", json={"value": "AIza_twenty_four_chars_x"}
    ).json()
    gemini = next(s for s in body["secrets"] if s["name"] == "gemini_api_key")

    assert "24 characters" in gemini["hint"]
    # Not even the first characters, which for most vendors identify the account
    # tier and are enough to confirm a guessed key.
    assert "AIza" not in gemini["hint"]


@windows_only
def test_clearing_a_key_disables_the_provider_again(client: TestClient):
    client.put("/settings/providers/gemini_api_key", json={"value": "AIza_test_value_here"})

    after = client.delete("/settings/providers/gemini_api_key").json()

    gemini = next(s for s in after["secrets"] if s["name"] == "gemini_api_key")
    assert gemini["configured"] is False
    assert after["chain_available"] is False


@windows_only
def test_clearing_a_vault_key_reveals_the_env_value_rather_than_disabling_it(settings: Settings):
    """The outcome of a delete must be visible, not surprising."""
    # `SecretStr`, not a bare string: `model_copy` does not coerce, and a plain
    # string in a `SecretStr` field would break every reader of it.
    with_env = settings.model_copy(update={"groq_api_key": SecretStr("gsk_from_dot_env")})

    with TestClient(create_app(with_env)) as client:
        client.put("/settings/providers/groq_api_key", json={"value": "gsk_from_the_vault"})
        after = client.delete("/settings/providers/groq_api_key").json()

        groq = next(s for s in after["secrets"] if s["name"] == "groq_api_key")
        assert groq["configured"] is True
        assert groq["source"] == "env"


def test_a_name_off_the_allowlist_is_refused(client: TestClient):
    """The name comes from the URL.

    This is the difference between a settings page and an HTTP-reachable write
    primitive on the user's disk.
    """
    response = client.put(
        "/settings/providers/aws_secret_access_key", json={"value": "AKIAEXAMPLE12345"}
    )

    assert response.status_code == 400
    assert client.delete("/settings/providers/id_rsa").status_code == 400


def test_a_value_too_short_to_be_a_key_is_rejected(client: TestClient):
    assert (
        client.put("/settings/providers/groq_api_key", json={"value": "short"}).status_code == 422
    )


def test_a_pasted_document_is_rejected_rather_than_stored(client: TestClient):
    response = client.put("/settings/providers/groq_api_key", json={"value": "x" * 600})

    assert response.status_code == 422


# -- testing the chain -----------------------------------------------------


def test_testing_an_unconfigured_chain_reports_the_reason_not_an_error(client: TestClient):
    """200 with ok:false, on purpose.

    A failed provider is not a failed request, and an error envelope would flatten
    "your key was rejected" and "you have no key" into the same red box.
    """
    response = client.post("/settings/providers/test")

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert body["error"]
    assert body["reply"] is None
