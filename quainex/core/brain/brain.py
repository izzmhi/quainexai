"""The Quainex Brain — natural language in, structured intent out.

Purpose:
    Turn "Open VS Code" into a typed, validated ``Intent`` that Phase 3 can
    dispatch on, without any component downstream having to parse prose.

Architecture:
    utterance (+ optional history)
        -> validate locally         reject empty/oversized before spending a call
        -> AIProvider.parse(...)    schema-constrained classification
        -> apply policy             compute requires_confirmation
        -> audit log                every interpretation is recorded
        -> Intent

Two design points worth stating plainly:

**Perception and policy are separate.** The model decides *what was asked*. The
Brain decides *whether it is safe to act without asking*. If the model chose the
confirmation flag, a crafted utterance could argue its way past it; because the
flag is computed from the intent type in code, it cannot.

**Low confidence downgrades safety, not correctness.** A classification below the
configured threshold still returns its best guess — it is simply marked as
needing confirmation. Discarding it would lose information the user could
resolve with one "yes".

Dependencies:
    quainex.config.settings, quainex.core.brain.{prompts,schemas},
    quainex.core.exceptions, quainex.core.logging, quainex.services.ai

Example:
    >>> intent = await brain.interpret("Open VS Code")
    >>> intent.intent, intent.target, intent.requires_confirmation
    (<IntentType.OPEN_APPLICATION: 'open_application'>, 'VS Code', False)

Future improvements:
    * Cache classifications of repeated utterances (Phase 5 memory).
    * Fall back to a deterministic keyword matcher when the provider is offline,
      so core commands keep working without network access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from quainex.core.brain.prompts import SYSTEM_PROMPT
from quainex.core.brain.schemas import (
    CONFIRMATION_REQUIRED,
    NON_ACTIONABLE,
    Intent,
    IntentClassification,
)
from quainex.core.exceptions import InvalidUtteranceError
from quainex.core.logging import get_logger
from quainex.services.ai.provider import ChatMessage

if TYPE_CHECKING:
    from quainex.config.settings import Settings
    from quainex.services.ai.provider import AIProvider

#: Longest utterance accepted. Well above any spoken command, low enough that a
#: pasted document cannot be used to run up token spend.
MAX_UTTERANCE_CHARS = 4000

#: How many prior turns to include as context. Enough to resolve "close it" after
#: "open Spotify", bounded so cost stays predictable.
MAX_HISTORY_TURNS = 6


class Brain:
    """Classifies natural language into structured, actionable intents."""

    def __init__(self, provider: AIProvider, settings: Settings) -> None:
        """Construct the Brain.

        Args:
            provider: The language model backend to classify with.
            settings: Configuration supplying the confidence threshold.
        """
        self._provider = provider
        self._settings = settings
        self._log = get_logger(__name__, provider=provider.name)

    @property
    def is_available(self) -> bool:
        """Whether the underlying provider can currently serve requests."""
        return self._provider.is_available

    async def interpret(
        self,
        utterance: str,
        history: list[ChatMessage] | None = None,
    ) -> Intent:
        """Classify an utterance into a structured intent.

        Args:
            utterance: What the user said or typed.
            history: Optional prior conversation turns, oldest first. Only the
                most recent ``MAX_HISTORY_TURNS`` are sent.

        Returns:
            The classified intent, with Quainex's confirmation policy applied.

        Raises:
            InvalidUtteranceError: The utterance is empty or too long.
            quainex.core.exceptions.ProviderNotConfiguredError: No credentials.
            quainex.core.exceptions.ProviderError: The upstream call failed.
        """
        cleaned = self._validate(utterance)

        messages = [
            *(history or [])[-MAX_HISTORY_TURNS:],
            ChatMessage(role="user", content=cleaned),
        ]

        classification = await self._provider.parse(
            messages=messages,
            output_model=IntentClassification,
            system=SYSTEM_PROMPT,
        )

        intent = Intent(
            **classification.model_dump(),
            requires_confirmation=self._needs_confirmation(classification),
            # The validated utterance, not the raw one, and not whatever the model
            # may have echoed back: a conversational handler replies to this, so it
            # must be what the user actually said.
            utterance=cleaned,
        )

        # Audit record: this is the decision that Phase 3 will act on, so it is
        # logged whether or not the action is ultimately executed.
        self._log.info(
            "intent_classified",
            intent=intent.intent.value,
            target=intent.target,
            confidence=round(intent.confidence, 3),
            requires_confirmation=intent.requires_confirmation,
            actionable=intent.is_actionable,
            utterance_chars=len(cleaned),
        )
        return intent

    # -- internals --------------------------------------------------------

    @staticmethod
    def _validate(utterance: str) -> str:
        """Check and normalise the raw utterance.

        Args:
            utterance: Raw user input.

        Returns:
            The trimmed utterance.

        Raises:
            InvalidUtteranceError: The input is empty or exceeds the size limit.
        """
        cleaned = utterance.strip()
        if not cleaned:
            raise InvalidUtteranceError("Utterance is empty")
        if len(cleaned) > MAX_UTTERANCE_CHARS:
            raise InvalidUtteranceError(
                f"Utterance is {len(cleaned)} characters; the limit is {MAX_UTTERANCE_CHARS}"
            )
        return cleaned

    def _needs_confirmation(self, classification: IntentClassification) -> bool:
        """Decide whether the user must confirm before this intent is executed.

        Two independent triggers:

        1. The intent is inherently disruptive or hard to reverse.
        2. The classification is not confident enough to act on unattended.

        One exemption, and it is a security decision rather than a convenience:
        **conversational intents are never gated.** There is nothing to approve —
        the reply has no side effect and cannot be made to have one, because the
        handler has no access to the desktop at all.

        Left in, the low-confidence rule made gibberish produce *"Confirm:
        unknown?"* — a prompt asking permission for nothing. That is worse than
        noise. Confirmation only protects anything if the user reads it, and a
        system that asks "are you sure?" about a greeting teaches them to click
        yes without looking. The one time it matters — shutting the machine down —
        they would click straight through. Cheapening the prompt is how the
        mechanism stops working.

        Args:
            classification: The model's classification.

        Returns:
            ``True`` if Phase 3 must ask before acting.
        """
        if classification.intent in CONFIRMATION_REQUIRED:
            return True
        if classification.intent in NON_ACTIONABLE:
            return False
        return classification.confidence < self._settings.brain_confidence_threshold
