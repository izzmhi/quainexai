"""Tests for the Brain: classification, confirmation policy, and the endpoint.

Every test runs against a fake provider, so the suite exercises Quainex's own
logic — validation, policy, history handling — without network access, cost, or
dependence on what a real model happens to answer on a given day.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import BaseModel

from quainex.config.settings import Settings
from quainex.core.brain import Brain, IntentClassification, IntentParameter, IntentType
from quainex.core.brain.brain import MAX_HISTORY_TURNS, MAX_UTTERANCE_CHARS
from quainex.core.exceptions import InvalidUtteranceError, ProviderError
from quainex.services.ai.provider import AIProvider, ChatMessage

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

#: An utterance the local fast path deliberately declines, so the model path is
#: the one under test.
#:
#: Needed once ``classify_locally`` existed: these tests were written against
#: "Open VS Code", which is now answered locally for nothing — so the fake provider
#: was never reached and five tests failed while the code was working correctly.
#: A test of the expensive path has to ask for something that genuinely needs it.
NEEDS_MODEL = "bring up my editor"


class FakeProvider:
    """An ``AIProvider`` that returns a canned classification and records inputs."""

    def __init__(
        self,
        classification: IntentClassification | None = None,
        error: Exception | None = None,
        available: bool = True,
    ) -> None:
        self._classification = classification
        self._error = error
        self._available = available
        self.calls = 0
        self.last_messages: list[ChatMessage] = []
        self.last_system: str | None = None
        self.last_max_tokens: int | None = None

    @property
    def name(self) -> str:
        return "fake"

    @property
    def is_available(self) -> bool:
        return self._available

    async def complete(
        self,
        *,
        messages: list[ChatMessage],
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        return "unused"

    async def parse(
        self,
        *,
        messages: list[ChatMessage],
        output_model: type[BaseModel],
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> BaseModel:
        self.calls += 1
        self.last_messages = messages
        self.last_system = system
        self.last_max_tokens = max_tokens
        if self._error is not None:
            raise self._error
        assert self._classification is not None
        return self._classification

    async def aclose(self) -> None:
        return None


def _classification(
    intent: IntentType = IntentType.OPEN_APPLICATION,
    target: str | None = "VS Code",
    confidence: float = 0.99,
    parameters: list[IntentParameter] | None = None,
) -> IntentClassification:
    return IntentClassification(
        intent=intent,
        target=target,
        confidence=confidence,
        reasoning="test fixture",
        parameters=parameters or [],
    )


def _brain(provider: FakeProvider, tmp_path, threshold: float = 0.6) -> Brain:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        log_dir=tmp_path / "logs",
        brain_confidence_threshold=threshold,
    )
    return Brain(provider=provider, settings=settings)


# -- protocol conformance --------------------------------------------------


def test_fake_provider_satisfies_the_protocol():
    provider: AIProvider = FakeProvider()
    assert provider.name == "fake"


# -- happy path ------------------------------------------------------------


async def test_clear_command_is_classified_and_needs_no_confirmation(tmp_path):
    provider = FakeProvider(_classification())
    intent = await _brain(provider, tmp_path).interpret(NEEDS_MODEL)

    assert intent.intent is IntentType.OPEN_APPLICATION
    assert intent.target == "VS Code"
    assert intent.confidence == pytest.approx(0.99)
    assert intent.requires_confirmation is False
    assert intent.is_actionable is True


async def test_system_prompt_and_utterance_are_sent(tmp_path):
    provider = FakeProvider(_classification())
    await _brain(provider, tmp_path).interpret(f"  {NEEDS_MODEL}  ")

    assert provider.last_system is not None
    assert "Intent catalogue" in provider.last_system
    # The utterance is trimmed before it is sent.
    assert provider.last_messages[-1].content == NEEDS_MODEL
    assert provider.last_messages[-1].role == "user"


async def test_parameters_round_trip_to_a_dict(tmp_path):
    provider = FakeProvider(
        _classification(
            intent=IntentType.CLIPBOARD,
            target=None,
            parameters=[
                IntentParameter(key="action", value="write"),
                IntentParameter(key="text", value="hello"),
            ],
        )
    )
    intent = await _brain(provider, tmp_path).interpret("Copy hello to my clipboard")
    assert intent.parameters_as_dict() == {"action": "write", "text": "hello"}


# -- the local fast path ---------------------------------------------------


async def test_a_common_command_never_reaches_the_provider(tmp_path):
    """The point of the fast path, asserted on the provider rather than the result.

    Getting the right answer is not the claim being made here; getting it without
    spending roughly 1,650 prompt tokens is.
    """
    provider = FakeProvider(_classification())

    intent = await _brain(provider, tmp_path).interpret("take a screenshot")

    assert provider.calls == 0
    assert intent.intent is IntentType.SCREENSHOT
    assert intent.confidence == 1.0
    assert "no model call" in intent.reasoning


async def test_the_utterance_survives_a_local_match(tmp_path):
    """Conversational handlers and the audit trail both read it."""
    intent = await _brain(FakeProvider(_classification()), tmp_path).interpret("  Lock the screen ")

    assert intent.utterance == "Lock the screen"


async def test_a_local_match_is_not_gated_by_the_confidence_threshold(tmp_path):
    """A pattern match is certain, so even a strict threshold must not gate it.

    Otherwise raising the threshold would start asking the user to confirm "take a
    screenshot".
    """
    intent = await _brain(FakeProvider(_classification()), tmp_path, threshold=0.99).interpret(
        "take a screenshot"
    )

    assert intent.requires_confirmation is False


async def test_history_sends_the_request_to_the_model_even_when_it_would_match(tmp_path):
    """A follow-up is only meaningful in context, and a pattern cannot see context.

    So the presence of history disables the fast path entirely rather than risking
    a locally-matched answer that ignores what came before.
    """
    provider = FakeProvider(_classification())

    await _brain(provider, tmp_path).interpret(
        "close spotify", history=[ChatMessage(role="user", content="open spotify")]
    )

    assert provider.calls == 1


async def test_classification_asks_for_a_smaller_token_budget_than_prose(tmp_path):
    """Free tiers meter *requested* tokens, so an oversized cap costs real quota.

    A five-field JSON object does not need the budget that a paragraph does.
    """
    provider = FakeProvider(_classification())
    settings = Settings(_env_file=None, log_dir=tmp_path / "logs")  # type: ignore[call-arg]

    await Brain(provider=provider, settings=settings).interpret(NEEDS_MODEL)

    assert provider.last_max_tokens == settings.ai_max_tokens_classification
    assert settings.ai_max_tokens_classification < settings.ai_max_tokens


# -- confirmation policy ---------------------------------------------------


@pytest.mark.parametrize(
    "intent_type",
    [IntentType.SHUTDOWN, IntentType.RESTART, IntentType.SLEEP, IntentType.CLOSE_APPLICATION],
)
async def test_disruptive_intents_always_require_confirmation(intent_type, tmp_path):
    # Even at maximum confidence: the policy is not the model's to decide.
    provider = FakeProvider(_classification(intent=intent_type, target=None, confidence=1.0))
    intent = await _brain(provider, tmp_path).interpret("do the thing")
    assert intent.requires_confirmation is True


async def test_low_confidence_requires_confirmation(tmp_path):
    provider = FakeProvider(_classification(confidence=0.3))
    intent = await _brain(provider, tmp_path).interpret("uh, open the code thing")

    assert intent.requires_confirmation is True
    # The best guess is preserved rather than discarded — one "yes" resolves it.
    assert intent.intent is IntentType.OPEN_APPLICATION
    assert intent.target == "VS Code"


async def test_confidence_threshold_is_configurable(tmp_path):
    classification = _classification(confidence=0.5)

    strict = await _brain(FakeProvider(classification), tmp_path, threshold=0.9).interpret("go")
    lenient = await _brain(FakeProvider(classification), tmp_path, threshold=0.1).interpret("go")

    assert strict.requires_confirmation is True
    assert lenient.requires_confirmation is False


async def test_conversational_intents_are_not_actionable(tmp_path):
    provider = FakeProvider(
        _classification(intent=IntentType.ANSWER_QUESTION, target="what is 2+2", confidence=0.95)
    )
    intent = await _brain(provider, tmp_path).interpret("what is 2+2")
    assert intent.is_actionable is False


@pytest.mark.parametrize(
    "intent_type",
    [IntentType.SMALL_TALK, IntentType.ANSWER_QUESTION, IntentType.UNKNOWN],
)
async def test_a_conversational_intent_is_never_gated_however_unsure_the_model_is(
    tmp_path, intent_type: IntentType
):
    """Confirmation exists to guard side effects. A reply has none.

    Before this exemption, gibberish classified as ``unknown`` with 0.0 confidence
    produced *"Confirm: unknown?"* — a prompt asking permission for nothing.

    That is not merely untidy. Confirmation only protects anything if the user
    reads it, and a system that asks "are you sure?" about a greeting teaches them
    to click yes without looking. The one time it matters — powering the machine
    off — they would click straight through.
    """
    provider = FakeProvider(_classification(intent=intent_type, target=None, confidence=0.0))

    intent = await _brain(provider, tmp_path, threshold=0.9).interpret("asdkjh qwe zxcv")

    assert intent.requires_confirmation is False


async def test_the_exemption_does_not_extend_to_actions(tmp_path):
    """The low-confidence gate must still hold for anything with an effect.

    Stated as its own test so that widening ``NON_ACTIONABLE`` by accident — the
    one change that would quietly disarm the gate — fails here.
    """
    provider = FakeProvider(
        _classification(intent=IntentType.OPEN_APPLICATION, target="something", confidence=0.1)
    )

    intent = await _brain(provider, tmp_path, threshold=0.9).interpret("uh, that thing")

    assert intent.requires_confirmation is True


async def test_the_utterance_is_carried_through_for_a_handler_to_reply_to(tmp_path):
    """``small_talk`` has no target, so the utterance is all a handler has."""
    provider = FakeProvider(
        _classification(intent=IntentType.SMALL_TALK, target=None, confidence=0.9)
    )

    intent = await _brain(provider, tmp_path).interpret("  how are you doing?  ")

    assert intent.utterance == "how are you doing?"
    assert intent.subject == "how are you doing?"


# -- input validation ------------------------------------------------------


@pytest.mark.parametrize("utterance", ["", "   ", "\n\t "])
async def test_empty_utterance_is_rejected_without_calling_the_model(utterance, tmp_path):
    provider = FakeProvider(_classification())
    with pytest.raises(InvalidUtteranceError):
        await _brain(provider, tmp_path).interpret(utterance)
    assert provider.calls == 0, "must not pay for a model call to classify whitespace"


async def test_oversized_utterance_is_rejected(tmp_path):
    provider = FakeProvider(_classification())
    with pytest.raises(InvalidUtteranceError, match="limit is"):
        await _brain(provider, tmp_path).interpret("x" * (MAX_UTTERANCE_CHARS + 1))
    assert provider.calls == 0


async def test_history_is_truncated_to_the_configured_window(tmp_path):
    provider = FakeProvider(_classification())
    history = [ChatMessage(role="user", content=f"turn {i}") for i in range(20)]

    await _brain(provider, tmp_path).interpret("close it", history=history)

    # Truncated history plus the new utterance.
    assert len(provider.last_messages) == MAX_HISTORY_TURNS + 1
    assert provider.last_messages[0].content == f"turn {20 - MAX_HISTORY_TURNS}"
    assert provider.last_messages[-1].content == "close it"


# -- failure propagation ---------------------------------------------------


async def test_provider_errors_propagate(tmp_path):
    provider = FakeProvider(error=ProviderError("upstream exploded"))
    with pytest.raises(ProviderError, match="upstream exploded"):
        await _brain(provider, tmp_path).interpret(NEEDS_MODEL)


# -- HTTP endpoint ---------------------------------------------------------


def _install_fake_brain(client: TestClient, provider: FakeProvider) -> None:
    container = client.app.state.container
    container.brain = Brain(provider=provider, settings=container.settings)


def test_interpret_endpoint_returns_the_intent(client: TestClient):
    _install_fake_brain(client, FakeProvider(_classification()))

    response = client.post("/brain/interpret", json={"utterance": NEEDS_MODEL})
    assert response.status_code == 200

    body = response.json()
    assert body["intent"] == "open_application"
    assert body["target"] == "VS Code"
    assert body["requires_confirmation"] is False
    assert body["reasoning"]


def test_interpret_endpoint_flags_dangerous_intents(client: TestClient):
    _install_fake_brain(
        client,
        FakeProvider(_classification(intent=IntentType.SHUTDOWN, target=None, confidence=1.0)),
    )
    body = client.post("/brain/interpret", json={"utterance": "shut down"}).json()
    assert body["requires_confirmation"] is True


def test_whitespace_utterance_returns_400_envelope(client: TestClient):
    _install_fake_brain(client, FakeProvider(_classification()))

    response = client.post("/brain/interpret", json={"utterance": "   "})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_utterance"


def test_malformed_body_uses_the_same_envelope(client: TestClient):
    # FastAPI's default 422 shape would be the one response clients need a
    # second parser for; it is normalised into the standard envelope.
    response = client.post("/brain/interpret", json={"wrong_field": "x"})
    assert response.status_code == 422

    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert error["correlation_id"]
    assert error["fields"], "per-field detail is preserved for the caller"


def test_interpret_without_credentials_returns_503(client: TestClient):
    # The default test container has no API key, so the real provider is used.
    response = client.post("/brain/interpret", json={"utterance": NEEDS_MODEL})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "provider_not_configured"
