"""Authentication and authorization.

Phase 6. Password hashing, bearer-token issue and verification, and the
dependency that guards every route reachable from outside this machine.
"""

from quainex.auth.dependencies import LOCAL_SUBJECT, AuthDep, require_auth
from quainex.auth.passwords import hash_password, verify_password
from quainex.auth.tokens import TokenService

__all__ = [
    "LOCAL_SUBJECT",
    "AuthDep",
    "TokenService",
    "hash_password",
    "require_auth",
    "verify_password",
]
