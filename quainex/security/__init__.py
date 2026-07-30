"""Security primitives: confirmation tokens, rate limiting, credential storage.

Cross-cutting; hardened in Phase 6.
"""

from quainex.security.confirmations import ConfirmationService
from quainex.security.ratelimit import RateLimiter
from quainex.security.vault import (
    STORABLE_SECRETS,
    CredentialVault,
    UnknownSecretError,
    VaultError,
    VaultUnavailableError,
)

__all__ = [
    "STORABLE_SECRETS",
    "ConfirmationService",
    "CredentialVault",
    "RateLimiter",
    "UnknownSecretError",
    "VaultError",
    "VaultUnavailableError",
]
