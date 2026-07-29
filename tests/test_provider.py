"""Tests for the Anthropic provider.

Everything here runs offline. The one test that would call the real API is
skipped unless a key is present in the ambient environment, so the suite stays
green — and free — on a fresh clone.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from pydantic import BaseModel

from quainex.config.settings import Settings
from quainex.core.exceptions import ProviderError, ProviderNotConfiguredError
from quainex.services.ai.anthropic_provider import AnthropicProvider
from quainex.services.ai.provider import AIProvider, ChatMessage

if TYPE_CHECKING:
    from pathlib import Path


class Intent(BaseModel):
    """Minimal stand-in for the Phase 2 Brain output schema."""

    intent: str
    confidence: float


@pytest.fixture
def unconfigured(tmp_path: Path) -> AnthropicProvider:
    return AnthropicProvider(
        Settings(
            _env_file=None,  # type: ignore[call-arg]
            log_dir=tmp_path / "logs",
            anthropic_api_key=None,
        )
    )


@pytest.fixture
def configured(tmp_path: Path) -> AnthropicProvider:
    # A syntactically valid but fake key: enough to build a client, never called.
    return AnthropicProvider(
        Settings(
            _env_file=None,  # type: ignore[call-arg]
            log_dir=tmp_path / "logs",
            anthropic_api_key="sk-ant-fake-key-for-tests",
        )
    )


# -- contract --------------------------------------------------------------


def test_provider_satisfies_the_protocol(configured: AnthropicProvider):
    # Structural check: assignment fails type-checking if the shape drifts.
    provider: AIProvider = configured
    assert provider.name == "anthropic"


# -- graceful degradation without credentials ------------------------------


def test_missing_key_does_not_raise_at_construction(unconfigured: AnthropicProvider):
    # Quainex must boot on a machine with no API key.
    assert unconfigured.is_available is False


def test_present_key_marks_provider_available(configured: AnthropicProvider):
    assert configured.is_available is True


async def test_complete_without_key_raises_actionable_error(unconfigured: AnthropicProvider):
    with pytest.raises(ProviderNotConfiguredError) as exc_info:
        await unconfigured.complete(messages=[ChatMessage(role="user", content="hi")])
    assert "anthropic" in str(exc_info.value)


async def test_parse_without_key_raises_actionable_error(unconfigured: AnthropicProvider):
    with pytest.raises(ProviderNotConfiguredError):
        await unconfigured.parse(
            messages=[ChatMessage(role="user", content="hi")],
            output_model=Intent,
        )


async def test_aclose_is_safe_when_never_configured(unconfigured: AnthropicProvider):
    await unconfigured.aclose()  # must not raise


# -- response handling -----------------------------------------------------


def test_empty_conversation_is_rejected(configured: AnthropicProvider):
    with pytest.raises(ProviderError, match="empty conversation"):
        configured._to_sdk_messages([])


def test_text_is_extracted_past_non_text_blocks(configured: AnthropicProvider):
    # Reasoning blocks can precede the answer, so blocks are filtered by type
    # rather than read positionally.
    message = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="..."),
            SimpleNamespace(type="text", text="Opening "),
            SimpleNamespace(type="text", text="VS Code."),
        ]
    )
    assert configured._extract_text(message) == "Opening VS Code."  # type: ignore[arg-type]


def test_response_without_text_is_an_error(configured: AnthropicProvider):
    message = SimpleNamespace(content=[SimpleNamespace(type="thinking", thinking="...")])
    with pytest.raises(ProviderError, match="no text"):
        configured._extract_text(message)  # type: ignore[arg-type]


def test_refusal_is_detected_before_content_is_read(configured: AnthropicProvider):
    # Refusals arrive as a successful HTTP response with an empty body.
    message = SimpleNamespace(
        stop_reason="refusal",
        stop_details=SimpleNamespace(category="cyber"),
        content=[],
    )
    with pytest.raises(ProviderError, match="declined"):
        configured._guard_refusal(message)  # type: ignore[arg-type]


def test_truncated_response_is_warned_about_but_not_fatal(configured: AnthropicProvider):
    message = SimpleNamespace(stop_reason="max_tokens", stop_details=None, content=[])
    configured._guard_refusal(message)  # type: ignore[arg-type]


# -- live smoke test (opt-in) ---------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("QUAINEX_ANTHROPIC_API_KEY"),
    reason="Set QUAINEX_ANTHROPIC_API_KEY to run the live provider smoke test.",
)
async def test_live_structured_parse(tmp_path: Path):
    provider = AnthropicProvider(
        Settings(_env_file=None, log_dir=tmp_path / "logs")  # type: ignore[call-arg]
    )
    result = await provider.parse(
        messages=[ChatMessage(role="user", content="Open VS Code")],
        output_model=Intent,
        system="Classify the user's request into a short intent slug.",
    )
    try:
        assert isinstance(result, Intent)
        assert 0.0 <= result.confidence <= 1.0
    finally:
        await provider.aclose()
