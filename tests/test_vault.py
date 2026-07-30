"""Tests for encrypted credential storage.

A fake cipher is injected throughout, so these tests assert the *vault's*
behaviour — allowlisting, write-through, masking, tolerance of a broken file —
without depending on a real DPAPI key. One test does exercise the real Windows
cipher, because "we encrypt credentials" is a claim worth proving once.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from quainex.security.vault import (
    STORABLE_SECRETS,
    CredentialVault,
    UnknownSecretError,
    VaultError,
    VaultUnavailableError,
)


class ReversingCipher:
    """Stand-in cipher: reversible, obviously not secure, entirely predictable.

    Deliberately *not* a no-op. Reversing the bytes means a test can assert the
    file on disk does not contain the plaintext, which is the property that
    matters, without pretending to test cryptography.
    """

    is_available = True

    def encrypt(self, plaintext: bytes) -> bytes:
        """Reverse the bytes."""
        return plaintext[::-1]

    def decrypt(self, ciphertext: bytes) -> bytes:
        """Reverse them back."""
        return ciphertext[::-1]


class BrokenCipher:
    """Cipher that cannot decrypt, standing in for a file from another machine."""

    is_available = True

    def encrypt(self, plaintext: bytes) -> bytes:
        """Pass through."""
        return plaintext

    def decrypt(self, ciphertext: bytes) -> bytes:
        """Fail the way DPAPI fails for another user's blob."""
        raise VaultError("CryptUnprotectData failed (Windows error 13)")


class UnavailableCipher:
    """Cipher for a platform with no protection API."""

    is_available = False

    def encrypt(self, plaintext: bytes) -> bytes:
        """Never called."""
        raise VaultUnavailableError

    def decrypt(self, ciphertext: bytes) -> bytes:
        """Never called."""
        raise VaultUnavailableError


@pytest.fixture
def vault(tmp_path: Path) -> CredentialVault:
    """A vault with a predictable cipher, in a throwaway directory."""
    return CredentialVault(tmp_path / "credentials.dat", cipher=ReversingCipher())


def test_a_vault_for_a_path_that_does_not_exist_reads_as_empty(vault: CredentialVault):
    """First run is not an error state."""
    assert vault.load() == {}
    assert vault.status() == {}
    assert vault.get("groq_api_key") is None


def test_a_stored_secret_survives_a_new_vault_instance(vault: CredentialVault):
    vault.set("groq_api_key", "gsk_example_key_value")

    reopened = CredentialVault(vault.path, cipher=ReversingCipher())
    assert reopened.get("groq_api_key") == "gsk_example_key_value"


def test_the_plaintext_is_not_present_in_the_file(vault: CredentialVault):
    """The whole justification for this module in one assertion."""
    vault.set("gemini_api_key", "AIzaSyUniqueSentinelValue")

    on_disk = vault.path.read_bytes()
    assert b"AIzaSyUniqueSentinelValue" not in on_disk


def test_whitespace_around_a_pasted_key_is_stripped(vault: CredentialVault):
    """Keys copied from a web page routinely carry a trailing newline."""
    vault.set("groq_api_key", "  gsk_padded_key_value\n")

    assert vault.get("groq_api_key") == "gsk_padded_key_value"


def test_an_empty_value_is_refused(vault: CredentialVault):
    with pytest.raises(VaultError):
        vault.set("groq_api_key", "   ")


def test_a_name_off_the_allowlist_is_refused(vault: CredentialVault):
    """The vault must not become an HTTP-reachable key/value store.

    ``/settings/providers/{name}`` takes the name from the URL, so without this
    the endpoint would write arbitrary attacker-chosen data to the user's disk.
    """
    with pytest.raises(UnknownSecretError):
        vault.set("aws_secret_access_key", "AKIAEXAMPLEVALUE")

    with pytest.raises(UnknownSecretError):
        vault.delete("../../etc/passwd")

    assert not vault.path.exists()


def test_deleting_reports_whether_anything_was_there(vault: CredentialVault):
    assert vault.delete("groq_api_key") is False

    vault.set("groq_api_key", "gsk_example_key_value")
    assert vault.delete("groq_api_key") is True
    assert vault.get("groq_api_key") is None


def test_status_reports_lengths_and_never_values(vault: CredentialVault):
    vault.set("groq_api_key", "gsk_twelve!!")

    status = vault.status()
    assert status == {"groq_api_key": 12}
    # Belt and braces: the whole point is that no value can leak through here.
    assert "gsk_twelve!!" not in str(status)


def test_secrets_are_independent(vault: CredentialVault):
    vault.set("groq_api_key", "gsk_example_key_value")
    vault.set("gemini_api_key", "AIza_example_key_value")
    vault.delete("groq_api_key")

    assert vault.get("gemini_api_key") == "AIza_example_key_value"


def test_an_undecryptable_file_reads_as_empty_rather_than_crashing(tmp_path: Path):
    """A vault written by another Windows user must not brick the application.

    Raising here would strand the user with no dashboard to fix it from, so the
    failure is logged and treated as "nothing stored" — the file is still there
    to be deleted.
    """
    path = tmp_path / "credentials.dat"
    path.write_bytes(b"not decryptable by us")

    assert CredentialVault(path, cipher=BrokenCipher()).load() == {}


def test_a_file_containing_the_wrong_shape_reads_as_empty(tmp_path: Path):
    path = tmp_path / "credentials.dat"
    # A JSON array, not an object — what a partial write or a different version
    # might leave behind.
    path.write_bytes(ReversingCipher().encrypt(b'["not", "a", "mapping"]'))

    assert CredentialVault(path, cipher=ReversingCipher()).load() == {}


def test_unknown_names_in_a_decrypted_file_are_dropped(tmp_path: Path):
    """The allowlist is enforced on read as well as write.

    Otherwise a file edited to contain an extra name would still be honoured.
    """
    path = tmp_path / "credentials.dat"
    payload = b'{"groq_api_key": "gsk_ok", "surprise": "value", "gemini_api_key": 42}'
    path.write_bytes(ReversingCipher().encrypt(payload))

    assert CredentialVault(path, cipher=ReversingCipher()).load() == {"groq_api_key": "gsk_ok"}


def test_an_unsupported_platform_refuses_to_write_rather_than_storing_plaintext(tmp_path: Path):
    """Better an honest "not here yet" than a padlock icon over base64."""
    vault = CredentialVault(tmp_path / "credentials.dat", cipher=UnavailableCipher())

    assert vault.is_writable is False
    with pytest.raises(VaultUnavailableError):
        vault.set("groq_api_key", "gsk_example_key_value")
    assert not (tmp_path / "credentials.dat").exists()


def test_an_existing_file_is_unreadable_but_harmless_without_a_cipher(tmp_path: Path):
    path = tmp_path / "credentials.dat"
    path.write_bytes(b"ciphertext from a machine that could encrypt")

    assert CredentialVault(path, cipher=UnavailableCipher()).load() == {}


def test_every_guide_entry_matches_the_allowlist():
    """The settings page and the vault must not drift apart.

    A guide entry with no allowlist entry renders an input that always 400s; an
    allowlist entry with no guide is a credential the dashboard cannot set.
    """
    from quainex.api.routes.settings import _SECRET_GUIDE

    assert set(_SECRET_GUIDE) == set(STORABLE_SECRETS)


@pytest.mark.skipif(sys.platform != "win32", reason="DPAPI is a Windows API.")
def test_the_real_windows_cipher_round_trips(tmp_path: Path):
    """Prove the actual encryption path works, not just the fake one.

    Without this, every test above would pass on a machine where DPAPI is broken
    and the feature would be entirely non-functional.
    """
    vault = CredentialVault(tmp_path / "credentials.dat")
    assert vault.is_writable is True

    vault.set("groq_api_key", "gsk_real_dpapi_round_trip")

    assert vault.path.read_bytes() != b""
    assert b"gsk_real_dpapi_round_trip" not in vault.path.read_bytes()
    assert CredentialVault(vault.path).get("groq_api_key") == "gsk_real_dpapi_round_trip"
