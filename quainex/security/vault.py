"""Encrypted credential storage.

Purpose:
    Let API keys be entered in the dashboard and survive a restart, without ever
    sitting on disk in plaintext. The brief's requirement is literal — "Encrypt
    stored credentials" — and a settings page that writes keys into a text file
    would violate it while looking finished.

Why Windows DPAPI rather than a crypto library:
    Encryption is only half the problem; the hard half is *where the key lives*.
    A password-derived key means asking for a password on every boot, which
    defeats the point of a personal OS that starts with the machine. A key file
    beside the ciphertext is theatre — anyone who can read one can read both.

    DPAPI solves exactly this: Windows holds the master key, derived from the
    user's logon credentials, and never hands it out. Ciphertext written here is
    undecryptable by another user on the same machine and undecryptable on any
    other machine — even with the file and the source code. No key management,
    no new dependency.

    An app-specific entropy string is mixed in so that a blob stolen from this
    file cannot be decrypted by a different program running as the same user
    without also knowing the constant.

Why other platforms refuse rather than degrade:
    ``macOS``/``Linux`` get ``VaultUnavailableError`` and the dashboard tells the
    user to put the key in ``.env`` instead. The alternative — base64, or a key
    file next to the data — would present a padlock icon over no security, and
    that is worse than an honest "not here yet". Keyring support is the fix; see
    Future improvements.

Threat model, stated plainly:
    This protects credentials **at rest** against another user on the machine,
    against the file being copied elsewhere, and against casual inspection of the
    repo. It does **not** protect against malware already running as this user —
    such code can simply ask DPAPI to decrypt, exactly as Quainex does. Nothing
    that stores a usable key locally can defend against that, and claiming
    otherwise would be a lie.

Architecture:
    CredentialVault(path)
        |-- load()          decrypt -> dict[str, str]
        |-- get(name)       one secret, or None
        |-- set(name, val)  write-through: mutate -> encrypt -> replace file
        |-- delete(name)    same, minus the entry
        +-- status()        which names are set, never the values

    encrypt/decrypt -> _WindowsDpapi (ctypes -> crypt32.dll)
                       _Unavailable  (every other platform)

Dependencies:
    ctypes (Windows only), json, pathlib

Example:
    >>> vault = CredentialVault(Path.home() / ".quainex" / "credentials.dat")
    >>> vault.set("gemini_api_key", "AIza...")  # doctest: +SKIP
    >>> sorted(vault.status())                  # doctest: +SKIP
    ['gemini_api_key']

Future improvements:
    * ``keyring`` backend so macOS Keychain and Secret Service work identically.
    * Per-secret metadata (when set, last used) for an audit view.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from http import HTTPStatus
from pathlib import Path
from typing import Protocol

from quainex.core.exceptions import QuainexError
from quainex.core.logging import get_logger

_log = get_logger(__name__)

#: Mixed into every blob. Not a secret — it scopes ciphertext to Quainex so that
#: another program running as this user cannot decrypt the file by accident.
_ENTROPY = b"quainex.credential.vault.v1"

#: Secret names the vault will store. An allowlist, so a caller cannot use the
#: vault as a general-purpose key/value store reachable over HTTP.
STORABLE_SECRETS: frozenset[str] = frozenset(
    {
        "groq_api_key",
        "gemini_api_key",
        "anthropic_api_key",
        "local_api_key",
        "telegram_bot_token",
    }
)


class VaultError(QuainexError):
    """A credential could not be stored or retrieved."""

    error_code = "vault_error"
    http_status = HTTPStatus.INTERNAL_SERVER_ERROR


class VaultUnavailableError(VaultError):
    """Encryption is not available on this platform.

    501 rather than 500: the request was well-formed and the server simply does
    not implement it here. A client can distinguish "try again" from "this will
    never work on this machine".
    """

    error_code = "vault_unavailable"
    http_status = HTTPStatus.NOT_IMPLEMENTED

    def __init__(self) -> None:
        """Explain the situation and the way around it."""
        super().__init__(
            f"Encrypted credential storage is only implemented on Windows "
            f"(this is {sys.platform}). Set keys in the .env file instead — "
            f"the dashboard will read them, it just cannot write them here."
        )


class UnknownSecretError(VaultError):
    """A caller asked to store a name that is not on the allowlist.

    400: the caller's fault, and the message names what is permitted so a
    legitimate client can correct itself.
    """

    error_code = "unknown_secret"
    http_status = HTTPStatus.BAD_REQUEST

    def __init__(self, name: str) -> None:
        """Name what was refused and what is permitted.

        Args:
            name: The rejected secret name.
        """
        super().__init__(
            f"'{name}' is not a storable credential. Permitted: "
            f"{', '.join(sorted(STORABLE_SECRETS))}."
        )


class _Cipher(Protocol):
    """Symmetric protection bound to the current user."""

    @property
    def is_available(self) -> bool:
        """Whether this cipher can be used on this machine."""
        ...

    def encrypt(self, plaintext: bytes) -> bytes:
        """Protect bytes so only this user on this machine can read them."""
        ...

    def decrypt(self, ciphertext: bytes) -> bytes:
        """Recover bytes previously protected by ``encrypt``."""
        ...


class _Unavailable:
    """Cipher stand-in for platforms with no implementation."""

    @property
    def is_available(self) -> bool:
        """Always ``False``."""
        return False

    def encrypt(self, plaintext: bytes) -> bytes:
        """Refuse to encrypt.

        Args:
            plaintext: Ignored.

        Raises:
            VaultUnavailableError: Always.
        """
        raise VaultUnavailableError

    def decrypt(self, ciphertext: bytes) -> bytes:
        """Refuse to decrypt.

        Args:
            ciphertext: Ignored.

        Raises:
            VaultUnavailableError: Always.
        """
        raise VaultUnavailableError


class _WindowsDpapi:
    """Cipher backed by ``CryptProtectData`` / ``CryptUnprotectData``."""

    #: Never show a UI prompt. This runs inside a request handler, where a modal
    #: dialog on the server's desktop would hang the call until someone noticed.
    _UI_FORBIDDEN = 0x01

    def __init__(self) -> None:
        """Bind the crypt32 entry points."""
        import ctypes
        from ctypes import wintypes

        class _Blob(ctypes.Structure):
            """The ``DATA_BLOB`` struct DPAPI passes data in and out through."""

            _fields_ = (("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char)))

        self._ctypes = ctypes
        self._blob = _Blob
        self._crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    @property
    def is_available(self) -> bool:
        """Always ``True`` — construction would have failed otherwise."""
        return True

    def encrypt(self, plaintext: bytes) -> bytes:
        """Protect bytes with the current user's DPAPI master key.

        Args:
            plaintext: Bytes to protect.

        Returns:
            The opaque protected blob.

        Raises:
            VaultError: The Windows call failed.
        """
        return self._call(self._crypt32.CryptProtectData, plaintext, "CryptProtectData")

    def decrypt(self, ciphertext: bytes) -> bytes:
        """Recover bytes protected by ``encrypt``.

        Args:
            ciphertext: A blob produced by ``encrypt``.

        Returns:
            The original bytes.

        Raises:
            VaultError: The blob is corrupt, or belongs to another user or
                machine. These are indistinguishable to DPAPI and the message
                says so rather than guessing.
        """
        return self._call(self._crypt32.CryptUnprotectData, ciphertext, "CryptUnprotectData")

    def _call(self, function: object, payload: bytes, label: str) -> bytes:
        """Invoke a DPAPI function and copy its result out.

        Args:
            function: ``CryptProtectData`` or ``CryptUnprotectData``.
            payload: Input bytes.
            label: Function name, for the error message.

        Returns:
            The output bytes.

        Raises:
            VaultError: The call returned failure.
        """
        ctypes = self._ctypes
        source = self._blob(len(payload), ctypes.cast(payload, ctypes.POINTER(ctypes.c_char)))
        entropy = self._blob(len(_ENTROPY), ctypes.cast(_ENTROPY, ctypes.POINTER(ctypes.c_char)))
        result = self._blob()

        ok = function(  # type: ignore[operator]
            ctypes.byref(source),
            None,
            ctypes.byref(entropy),
            None,
            None,
            self._UI_FORBIDDEN,
            ctypes.byref(result),
        )
        if not ok:
            code = ctypes.get_last_error()
            raise VaultError(
                f"{label} failed (Windows error {code}). If this is a decrypt, the "
                f"credential file was most likely written by a different Windows "
                f"user or on a different machine, and cannot be recovered — delete "
                f"it and re-enter your keys."
            )

        try:
            # The buffer belongs to Windows; copy before freeing it.
            return bytes(ctypes.string_at(result.pbData, result.cbData))
        finally:
            self._kernel32.LocalFree(result.pbData)


def _build_cipher() -> _Cipher:
    """Select the cipher for this platform.

    Returns:
        A DPAPI cipher on Windows, otherwise an unavailable stand-in.
    """
    if sys.platform != "win32":
        return _Unavailable()
    try:
        return _WindowsDpapi()
    except OSError as exc:  # pragma: no cover - crypt32 is always present
        _log.warning("dpapi_unavailable", error=str(exc))
        return _Unavailable()


class CredentialVault:
    """Stores a small set of named secrets, encrypted at rest."""

    def __init__(self, path: Path, cipher: _Cipher | None = None) -> None:
        """Construct the vault.

        Does not touch the filesystem — a vault for a path that does not exist
        yet is valid and simply reads as empty.

        Args:
            path: Where the encrypted blob lives.
            cipher: Protection implementation. Defaults to the platform's; tests
                inject a fake so they never depend on a real DPAPI key.
        """
        self._path = path
        self._cipher = cipher if cipher is not None else _build_cipher()

    @property
    def path(self) -> Path:
        """Where the encrypted file lives."""
        return self._path

    @property
    def is_writable(self) -> bool:
        """Whether new credentials can be stored on this platform."""
        return self._cipher.is_available

    def load(self) -> dict[str, str]:
        """Read and decrypt every stored secret.

        A missing file is not an error — it is the normal state on first run.

        Returns:
            Secret name to value. Empty when nothing is stored, when the platform
            has no cipher, or when the file cannot be decrypted.
        """
        if not self._path.is_file():
            return {}
        if not self._cipher.is_available:
            # A file exists but this platform cannot read it. Surfacing this as
            # empty (rather than raising) keeps the app bootable.
            _log.warning("vault_unreadable_on_this_platform", path=str(self._path))
            return {}

        try:
            decrypted = self._cipher.decrypt(self._path.read_bytes())
            loaded = json.loads(decrypted.decode("utf-8"))
        except (OSError, VaultError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            # Refusing to boot over an unreadable credential file would strand
            # the user with no way to reach the dashboard and fix it.
            _log.error("vault_load_failed", path=str(self._path), error=str(exc))
            return {}

        if not isinstance(loaded, dict):
            _log.error("vault_unexpected_shape", path=str(self._path))
            return {}

        return {
            str(name): str(value)
            for name, value in loaded.items()
            if name in STORABLE_SECRETS and isinstance(value, str)
        }

    def get(self, name: str) -> str | None:
        """Return one stored secret.

        Args:
            name: The secret to read.

        Returns:
            The value, or ``None`` when it is not stored.
        """
        return self.load().get(name)

    def set(self, name: str, value: str) -> None:
        """Store or replace one secret.

        Args:
            name: The secret to write. Must be on the allowlist.
            value: The plaintext value. Surrounding whitespace is stripped,
                because a key pasted from a web page usually carries some.

        Raises:
            UnknownSecretError: ``name`` is not storable.
            VaultUnavailableError: This platform cannot encrypt.
            VaultError: The value is empty, or the file could not be written.
        """
        self._check(name)
        cleaned = value.strip()
        if not cleaned:
            raise VaultError(f"Refusing to store an empty value for '{name}'.")

        secrets = self.load()
        secrets[name] = cleaned
        self._write(secrets)
        # Length only. Logging any part of a key would defeat the file it is in.
        _log.info("credential_stored", secret=name, characters=len(cleaned))

    def delete(self, name: str) -> bool:
        """Remove one secret.

        Args:
            name: The secret to remove.

        Returns:
            Whether it was present.

        Raises:
            UnknownSecretError: ``name`` is not storable.
            VaultUnavailableError: This platform cannot encrypt.
            VaultError: The file could not be written.
        """
        self._check(name)
        secrets = self.load()
        if name not in secrets:
            return False

        del secrets[name]
        self._write(secrets)
        _log.info("credential_deleted", secret=name)
        return True

    def status(self) -> dict[str, int]:
        """Report which secrets are stored, and how long each is.

        Never returns values. The dashboard needs to show "configured" and a
        masked hint; the length is enough for both and leaks nothing useful.

        Returns:
            Secret name to character count.
        """
        return {name: len(value) for name, value in self.load().items()}

    # -- internals --------------------------------------------------------

    @staticmethod
    def _check(name: str) -> None:
        """Reject names outside the allowlist.

        Args:
            name: The candidate secret name.

        Raises:
            UnknownSecretError: ``name`` is not storable.
        """
        if name not in STORABLE_SECRETS:
            raise UnknownSecretError(name)

    def _write(self, secrets: dict[str, str]) -> None:
        """Encrypt and atomically replace the credential file.

        Written to a temporary file in the same directory and then renamed, so an
        interrupted write cannot leave a half-file where every stored key used to
        be. ``os.replace`` is atomic on Windows and POSIX alike.

        Args:
            secrets: The complete set to persist.

        Raises:
            VaultUnavailableError: This platform cannot encrypt.
            VaultError: The file could not be written.
        """
        blob = self._cipher.encrypt(json.dumps(secrets).encode("utf-8"))

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            handle, temporary = tempfile.mkstemp(
                dir=self._path.parent, prefix=".credentials-", suffix=".tmp"
            )
            try:
                with os.fdopen(handle, "wb") as stream:
                    stream.write(blob)
                    # Durability before the rename, so the atomic swap cannot
                    # publish a file whose contents are still in a buffer.
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self._path)
            except OSError:
                Path(temporary).unlink(missing_ok=True)
                raise
        except OSError as exc:
            raise VaultError(f"Could not write '{self._path}': {exc}") from exc
