"""Quainex Brain - natural language understanding and intent detection.

Phase 2. Consumes `quainex.services.ai.AIProvider`; emits structured Intent
objects for the Phase 3 command registry to dispatch on.
"""

from quainex.core.brain.brain import Brain
from quainex.core.brain.schemas import (
    CONFIRMATION_REQUIRED,
    INTENT_DESCRIPTIONS,
    NON_ACTIONABLE,
    Intent,
    IntentClassification,
    IntentParameter,
    IntentType,
)

__all__ = [
    "CONFIRMATION_REQUIRED",
    "INTENT_DESCRIPTIONS",
    "NON_ACTIONABLE",
    "Brain",
    "Intent",
    "IntentClassification",
    "IntentParameter",
    "IntentType",
]
