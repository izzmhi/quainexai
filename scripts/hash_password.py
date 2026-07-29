"""Generate the credentials needed for remote access.

Run this before exposing Quainex beyond localhost:

    python scripts/hash_password.py

It prints the two lines to paste into ``.env``. The password itself is never
written anywhere — only its scrypt hash, which cannot be reversed.

Why a script rather than a settings field for the plaintext password:
    A plaintext password in ``.env`` is a plaintext password on disk, in
    backups, and in any screen-share of the file. Hashing it here means the
    configuration file holds nothing worth stealing.
"""

from __future__ import annotations

import getpass
import secrets
import sys

from quainex.auth import hash_password

MIN_LENGTH = 12


def main() -> int:
    """Prompt for a password and print the configuration lines.

    Returns:
        Process exit code.
    """
    print("Quainex remote access setup")
    print("=" * 40)
    print("This password is what you will type from your phone to get a token.\n")

    try:
        password = getpass.getpass("Password: ")
        again = getpass.getpass("Confirm:  ")
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        return 1

    if password != again:
        print("\nThose do not match. Nothing was changed.", file=sys.stderr)
        return 1

    if len(password) < MIN_LENGTH:
        print(
            f"\nUse at least {MIN_LENGTH} characters. This password is the only "
            "thing between your machine and anyone who can reach the port.",
            file=sys.stderr,
        )
        return 1

    print("\nAdd these two lines to your .env file:\n")
    print(f"QUAINEX_AUTH_PASSWORD_HASH={hash_password(password)}")
    print(f"QUAINEX_AUTH_SECRET={secrets.token_urlsafe(48)}")
    print(
        "\nThen set the bind address to expose Quainex on your network:\n"
        "\n    QUAINEX_HOST=0.0.0.0\n"
        "\nQuainex refuses to start on a non-loopback address without these,\n"
        "so there is no way to expose it accidentally with no password set."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
