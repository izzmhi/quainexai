"""Tests for the multi-provider fallback chain and its schema helpers.

The point of these tests is the *policy*, not the plumbing: which failures cause
the chain to move on, which do not, and what happens to a schema on the way to a
provider that cannot read Pydantic.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from quainex.config.settings import AIProviderName, Settings
from quainex.core.container import Container
from quainex.core.exceptions import ProviderError, ProviderNotConfiguredError
from quainex.services.ai.fallback import FallbackProvider
from quainex.services.ai.openai_compatible import OpenAICompatibleProvider
from quainex.services.ai.provider import ChatMessage
from quainex.services.ai.schemas import (
    flatten_schema,
    gemini_schema,
    parse_model,
    schema_instruction,
)


class Leaf(BaseModel):
    """Nested model, so the flattener has a ``$ref`` to resolve."""

    key: str
    value: str


class Branch(BaseModel):
    """Outer model referencing ``Leaf``."""

    name: str
    items: list[Leaf]


class FakeProvider:
    """Scriptable ``AIProvider`` double.

    Attributes:
        calls: How many times ``complete`` was entered, so a test can prove the
            chain stopped rather than merely returned the right value.
    """

    def __init__(
        self,
        name: str,
        *,
        available: bool = True,
        raises: Exception | None = None,
        reply: str = "ok",
    ) -> None:
        """Construct the double.

        Args:
            name: Identifier reported to the chain.
            available: What ``is_available`` reports.
            raises: Raised by every method when set.
            reply: Returned by every method otherwise.
        """
        self._name = name
        self._available = available
        self._raises = raises
        self._reply = reply
        self.calls = 0

    @property
    def name(self) -> str:
        """Identifier."""
        return self._name

    @property
    def is_available(self) -> bool:
        """Whether the chain should try this provider."""
        return self._available

    async def complete(self, **_: object) -> str:
        """Return the scripted reply or raise the scripted error."""
        self.calls += 1
        if self._raises:
            raise self._raises
        return self._reply

    async def parse(self, *, output_model: type[BaseModel], **_: object) -> BaseModel:
        """Return a scripted instance or raise the scripted error."""
        self.calls += 1
        if self._raises:
            raise self._raises
        return output_model.model_construct()

    async def look(self, **_: object) -> str:
        """Return the scripted reply or raise the scripted error."""
        self.calls += 1
        if self._raises:
            raise self._raises
        return self._reply

    async def read_document(self, **_: object) -> str:
        """Return the scripted reply or raise the scripted error."""
        self.calls += 1
        if self._raises:
            raise self._raises
        return self._reply

    async def aclose(self) -> None:
        """Nothing to release."""


# -- fallback policy -------------------------------------------------------


async def test_first_available_provider_answers_and_the_rest_are_untouched():
    first = FakeProvider("first", reply="from first")
    second = FakeProvider("second", reply="from second")
    chain = FallbackProvider([first, second])

    assert await chain.complete(messages=[ChatMessage(role="user", content="hi")]) == "from first"
    assert second.calls == 0


async def test_unconfigured_providers_are_skipped_not_tried():
    absent = FakeProvider("absent", available=False)
    present = FakeProvider("present", reply="answered")
    chain = FallbackProvider([absent, present])

    assert await chain.complete(messages=[]) == "answered"
    assert absent.calls == 0


async def test_a_provider_failure_falls_through_to_the_next():
    broken = FakeProvider("broken", raises=ProviderError("429 rate limited"))
    backup = FakeProvider("backup", reply="rescued")
    chain = FallbackProvider([broken, backup])

    assert await chain.complete(messages=[]) == "rescued"
    assert broken.calls == 1
    assert backup.calls == 1


async def test_a_missing_capability_falls_through_so_vision_still_works():
    """A text-only provider must not make screen questions impossible.

    This is why ``look()`` on the OpenAI-compatible provider raises rather than
    returning an apology string: the chain can only route past a failure it can
    see.
    """
    text_only = FakeProvider("groq", raises=ProviderError("no image input"))
    seeing = FakeProvider("gemini", reply="a code editor")
    chain = FallbackProvider([text_only, seeing])

    answer = await chain.look(image_paths=[Path("shot.png")], question="what is this?")
    assert answer == "a code editor"


async def test_exhausting_the_chain_names_every_provider_that_was_tried():
    chain = FallbackProvider(
        [
            FakeProvider("groq", raises=ProviderError("bad key")),
            FakeProvider("gemini", raises=ProviderError("quota exceeded")),
        ]
    )

    with pytest.raises(ProviderError) as caught:
        await chain.complete(messages=[])

    message = caught.value.message
    # Both names *and* both reasons: "everything failed" without saying which key
    # is wrong sends the user to guess.
    assert "groq" in message
    assert "gemini" in message
    assert "bad key" in message
    assert "quota exceeded" in message


async def test_an_empty_chain_reports_that_nothing_is_configured():
    chain = FallbackProvider([FakeProvider("groq", available=False)])

    with pytest.raises(ProviderNotConfiguredError) as caught:
        await chain.complete(messages=[])

    # The error names where to fix it, because "no provider configured" is only
    # actionable if you know what to configure.
    message = caught.value.message
    assert "Settings panel" in message
    assert "QUAINEX_GROQ_API_KEY" in message


async def test_a_refusal_is_not_a_reason_to_ask_a_different_model():
    """Refusals must propagate, not trigger fallback.

    A model declining a request is an *answer*. Retrying it against another
    provider is shopping for a yes, and it would mean a policy Quainex does not
    control could be bypassed by adding a second API key.
    """

    class Refusal(Exception):
        """Stands in for a refusal, which is not a ``ProviderError``."""

    refuser = FakeProvider("gemini", raises=Refusal("I can't help with that"))
    eager = FakeProvider("anthropic", reply="sure, here you go")
    chain = FallbackProvider([refuser, eager])

    with pytest.raises(Refusal):
        await chain.complete(messages=[])
    assert eager.calls == 0


async def test_the_chain_name_lists_only_configured_providers_in_order():
    chain = FallbackProvider(
        [
            FakeProvider("groq/llama", available=False),
            FakeProvider("gemini/flash"),
            FakeProvider("anthropic/opus"),
        ]
    )

    assert chain.name == "chain(gemini/flash -> anthropic/opus)"
    assert chain.is_available is True


async def test_describe_reports_every_provider_including_unavailable_ones():
    """The settings panel needs to show a provider that has no key yet."""
    chain = FallbackProvider([FakeProvider("groq", available=False), FakeProvider("gemini")])

    assert chain.describe() == [
        {"name": "groq", "available": False, "order": 0},
        {"name": "gemini", "available": True, "order": 1},
    ]


async def test_closing_the_chain_closes_every_provider():
    closed: list[str] = []

    class Recorder(FakeProvider):
        async def aclose(self) -> None:
            closed.append(self.name)

    await FallbackProvider([Recorder("a"), Recorder("b")]).aclose()
    assert closed == ["a", "b"]


# -- schema handling -------------------------------------------------------


def _has_key(node: object, needle: str) -> bool:
    """Whether a key appears anywhere in a nested structure.

    Walks the structure rather than searching its ``repr``: a model's own
    docstring becomes the schema's ``description``, so a substring search would
    match prose that merely mentions the keyword.

    Args:
        node: The fragment to search.
        needle: The key to look for.

    Returns:
        Whether any mapping in the tree carries that key.
    """
    if isinstance(node, dict):
        return needle in node or any(_has_key(value, needle) for value in node.values())
    if isinstance(node, list):
        return any(_has_key(item, needle) for item in node)
    return False


def test_flattening_inlines_nested_definitions():
    """Gemini does not follow ``$ref``, so a nested model must be inlined.

    Without this the nested type arrives as an empty object and structured output
    degrades silently — the response validates against nothing useful.
    """
    schema = flatten_schema(Branch)

    assert not _has_key(schema, "$defs")
    assert not _has_key(schema, "$ref")
    assert schema["properties"]["items"]["items"]["properties"].keys() == {"key", "value"}


def test_gemini_schema_strips_keywords_gemini_rejects():
    schema = gemini_schema(Branch)

    for rejected in ("additionalProperties", "$schema", "default", "const"):
        assert not _has_key(schema, rejected)


def test_the_prompt_instruction_carries_the_schema_for_providers_with_no_field():
    instruction = schema_instruction(Leaf)

    assert "JSON" in instruction
    assert '"key"' in instruction


@pytest.mark.parametrize(
    "raw",
    [
        '{"key": "a", "value": "b"}',
        '```json\n{"key": "a", "value": "b"}\n```',
        '```\n{"key": "a", "value": "b"}\n```',
        'Here you go: {"key": "a", "value": "b"}',
        '{"key": "a", "value": "b"}\n\nLet me know if you need anything else.',
    ],
)
def test_parsing_recovers_json_from_the_wrappers_models_actually_use(raw: str):
    """Tolerance here is not laxity — it is the difference between working and not.

    A provider asked for JSON may fence it, preface it, or add a sign-off. Failing
    on any of those would make structured output unreliable for reasons that have
    nothing to do with the model's answer.
    """
    parsed = parse_model(raw, Leaf)

    assert parsed is not None
    assert parsed.key == "a"


def test_parsing_returns_none_rather_than_guessing():
    assert parse_model("I'd rather not answer that.", Leaf) is None
    assert parse_model('{"key": "a"}', Leaf) is None  # `value` is required


# -- construction from settings -------------------------------------------


def test_the_container_builds_the_chain_in_the_configured_order(tmp_path: Path):
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        log_dir=tmp_path / "logs",
        database_path=tmp_path / "test.db",
        credentials_path=tmp_path / "credentials.dat",
        ai_providers=[AIProviderName.GEMINI, AIProviderName.ANTHROPIC],
        gemini_api_key="test-gemini-key",
    )

    chain = Container._build_ai_provider(settings)

    assert isinstance(chain, FallbackProvider)
    described = chain.describe()
    assert [entry["name"] for entry in described] == ["gemini/gemini-2.0-flash", "anthropic"]
    # Gemini has a key and Anthropic does not, so only the first is usable — and
    # the order came from settings, not from the enum's declaration order.
    assert [entry["available"] for entry in described] == [True, False]


def test_free_providers_lead_by_default(tmp_path: Path):
    """The default order is a product decision, so it is asserted, not assumed."""
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        log_dir=tmp_path / "logs",
        database_path=tmp_path / "test.db",
        credentials_path=tmp_path / "credentials.dat",
    )

    order = [str(name) for name in settings.ai_providers]

    assert order == ["groq", "gemini", "openrouter", "anthropic", "local"]
    # The property that matters, asserted separately from the exact list so that
    # adding a provider does not require re-stating the whole intent: the paid
    # one is a backstop, never the first thing reached for.
    assert order.index("anthropic") > order.index("groq")
    assert order.index("anthropic") > order.index("gemini")


def test_a_local_endpoint_counts_as_configured_without_any_key(tmp_path: Path):
    """Offline mode must not require a secret it has no use for."""
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        log_dir=tmp_path / "logs",
        database_path=tmp_path / "test.db",
        credentials_path=tmp_path / "credentials.dat",
        local_base_url="http://127.0.0.1:11434/v1",
    )

    assert settings.has_ai_credentials is True


def test_free_tiers_get_a_smaller_token_cap_than_the_paid_provider(tmp_path: Path):
    """A single request must not spend a whole minute's quota.

    Free tiers meter *requested* tokens, and ``max_tokens`` is a request whether
    or not the model uses it. Groq's free tier allows 12000 tokens per minute, so
    the 8192 default meant roughly one call per minute before a 429 — which
    presents as an outage rather than a quota. These models also have no hidden
    reasoning to fund, which is the only reason the large default exists.
    """
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        log_dir=tmp_path / "logs",
        database_path=tmp_path / "test.db",
        credentials_path=tmp_path / "credentials.dat",
        groq_api_key="gsk_test_key_value",
        openrouter_api_key="sk-or-test_key_value",
    )

    assert settings.ai_max_tokens_free_tier < settings.ai_max_tokens
    # Small enough that several calls fit inside Groq's 12000-per-minute budget.
    assert settings.ai_max_tokens_free_tier * 4 < 12000

    chain = Container._build_ai_provider(settings)
    assert isinstance(chain, FallbackProvider)
    caps = {
        provider.name.split("/")[0]: provider._max_tokens
        for provider in chain.providers
        if isinstance(provider, OpenAICompatibleProvider)
    }

    assert caps["groq"] == settings.ai_max_tokens_free_tier
    assert caps["openrouter"] == settings.ai_max_tokens_free_tier
    # Your own machine has no per-minute quota to fit inside.
    assert caps["local"] == settings.ai_max_tokens


async def test_an_openai_compatible_provider_without_a_url_is_inert():
    """A provider with nowhere to call must report itself unavailable, not 401."""
    provider = OpenAICompatibleProvider(
        name="groq", base_url="", model="llama-3.3-70b-versatile", api_key=None, max_tokens=128
    )

    assert provider.is_available is False
    with pytest.raises(ProviderNotConfiguredError):
        await provider.complete(messages=[])
