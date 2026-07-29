"""Password hashing and verification.

Purpose:
    Store the remote-access password in a form that is useless to anyone who
    reads the configuration file.

Why stdlib scrypt rather than bcrypt or argon2:
    ``hashlib.scrypt`` is in the standard library, is memory-hard, and is a
    perfectly respectable KDF. bcrypt and argon2 are both fine choices too, but
    each adds a compiled dependency for a capability Python already ships. Fewer
    binary wheels also means fewer things to break on a machine where large
    downloads are unreliable.

    What matters far more than the choice among these three is that the password
    is never stored in plaintext and never compared with ``==``.

Format:
    ``scrypt$n$r$p$<salt-b64>$<hash-b64>``

    Self-describing on purpose: the work factors travel with the hash, so raising
    them later does not invalidate existing passwords — an old hash still
    verifies with the parameters it was created under.

Dependencies:
    Standard library only.

Example:
    >>> stored = hash_password("correct horse battery staple")
    >>> verify_password("correct horse battery staple", stored)
    True

Future improvements:
    * Re-hash on successful login when the stored work factors are below current
      defaults, so passwords strengthen over time without user action.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

#: scrypt cost parameters. n=2**15 takes roughly 100 ms on a modern desktop —
#: unnoticeable for a login, punishing for an offline guessing attack.
_N = 2**15
_R = 8
_P = 1
_DKLEN = 32
_SALT_BYTES = 16
_PREFIX = "scrypt"


def _maxmem(n: int, r: int) -> int:
    """Memory allowance to pass to scrypt for the given work factors.

    scrypt needs ``128 * n * r`` bytes, and OpenSSL defaults to a 32 MiB ceiling
    — which n=2**15, r=8 hits exactly, failing with "memory limit exceeded".
    Deriving the allowance from the parameters means raising the work factors
    later cannot silently reintroduce that failure.

    Args:
        n: CPU/memory cost parameter.
        r: Block size parameter.

    Returns:
        A byte allowance with headroom over the true requirement.
    """
    return 128 * n * r * 2


def hash_password(password: str) -> str:
    """Hash a password for storage.

    Args:
        password: The plaintext password.

    Returns:
        A self-describing hash string safe to write to configuration.

    Raises:
        ValueError: The password is empty.
    """
    if not password:
        raise ValueError("Password must not be empty")

    salt = secrets.token_bytes(_SALT_BYTES)
    derived = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_N,
        r=_R,
        p=_P,
        dklen=_DKLEN,
        maxmem=_maxmem(_N, _R),
    )
    return "$".join(
        (
            _PREFIX,
            str(_N),
            str(_R),
            str(_P),
            base64.b64encode(salt).decode("ascii"),
            base64.b64encode(derived).decode("ascii"),
        )
    )


def verify_password(password: str, stored: str) -> bool:
    """Check a password against a stored hash.

    Returns ``False`` rather than raising on a malformed hash: a corrupt entry in
    the configuration must fail closed, not crash the login route and leak the
    difference between "bad password" and "bad configuration".

    Args:
        password: The plaintext password to check.
        stored: The stored hash, as produced by ``hash_password``.

    Returns:
        Whether the password matches.
    """
    try:
        prefix, n_raw, r_raw, p_raw, salt_b64, hash_b64 = stored.split("$")
        if prefix != _PREFIX:
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
        n, r, p = int(n_raw), int(r_raw), int(p_raw)
        derived = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=n,
            r=r,
            p=p,
            dklen=len(expected),
            # Derived from the *stored* parameters, so a hash created under
            # different work factors still verifies.
            maxmem=_maxmem(n, r),
        )
    except (ValueError, TypeError, MemoryError):
        return False

    # Constant-time: a plain `==` leaks how many leading bytes matched, which is
    # enough to reconstruct a hash one byte at a time given enough attempts.
    return hmac.compare_digest(derived, expected)
