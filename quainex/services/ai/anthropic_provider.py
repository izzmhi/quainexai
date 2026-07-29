"""Anthropic Claude implementation of the ``AIProvider`` contract.

Purpose:
    Translate Quainex's provider-neutral interface into Anthropic SDK calls, and
    translate the SDK's failures back into Quainex exceptions — so nothing
    outside this module imports ``anthropic``.

Architecture:
    ``AIProvider`` protocol
        -> ``AnthropicProvider``
             -> ``anthropic.AsyncAnthropic``
                  -> ``messages.create()``  free text  -> ``complete()``
                  -> ``messages.parse()``   schema     -> ``parse()``

    Two behaviours here matter more than they look:

    1. **Absent credentials do not raise at construction.** Quainex must boot
       and serve ``/health`` on a machine with no API key; only an actual call
       fails, and it fails with an actionable error.
    2. **Refusals are checked before content is read.** Current models can
       decline a request and return HTTP 200 with an empty or partial body.
       Code that reads ``content[0]`` unconditionally breaks on that path.

Dependencies:
    anthropic, pydantic, quainex.config.settings, quainex.core

Example:
    >>> provider = AnthropicProvider(get_settings())
    >>> await provider.complete(messages=[ChatMessage(role="user", content="Hi")])
    'Hello!'

Future improvements:
    * Add ``stream()`` for token-by-token voice output (Phase 4).
    * Add server-side fallback to a second model so a refusal degrades instead
      of failing outright.
    * Add a retry/backoff budget distinct from the SDK's built-in retries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import anthropic
from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, OutputConfigParam

from quainex.core.exceptions import ProviderError, ProviderNotConfiguredError
from quainex.core.logging import get_logger
from quainex.services.ai.provider import ChatMessage, ResponseModelT

if TYPE_CHECKING:
    from anthropic.types import Message, ParsedMessage

    from quainex.config.settings import Settings

_PROVIDER_NAME = "anthropic"


class AnthropicProvider:
    """``AIProvider`` backed by the Anthropic Claude API."""

    def __init__(self, settings: Settings) -> None:
        """Construct the provider.

        Does not raise when credentials are missing — see ``is_available``.

        Args:
            settings: Validated configuration supplying the API key and model.
        """
        self._settings = settings
        self._log = get_logger(__name__, provider=_PROVIDER_NAME, model=settings.ai_model)
        self._client: AsyncAnthropic | None = None

        if settings.has_ai_credentials and settings.anthropic_api_key is not None:
            self._client = AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())
        else:
            self._log.warning(
                "ai_provider_unconfigured",
                detail=(
                    "No API key set; AI features are disabled until "
                    "QUAINEX_ANTHROPIC_API_KEY is provided."
                ),
            )

    @property
    def name(self) -> str:
        """Short identifier for this provider."""
        return _PROVIDER_NAME

    @property
    def is_available(self) -> bool:
        """Whether an API client was successfully constructed."""
        return self._client is not None

    async def complete(
        self,
        *,
        messages: list[ChatMessage],
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a free-text response.

        Args:
            messages: Conversation history, oldest first.
            system: Optional system prompt.
            max_tokens: Optional token cap; falls back to the configured default.

        Returns:
            The concatenated text of the response.

        Raises:
            ProviderNotConfiguredError: No API key is configured.
            ProviderError: The upstream call failed, was refused, or was empty.
        """
        client = self._require_client()
        try:
            message = await client.messages.create(
                model=self._settings.ai_model,
                max_tokens=max_tokens or self._settings.ai_max_tokens,
                messages=self._to_sdk_messages(messages),
                system=system if system is not None else anthropic.omit,
                output_config=self._output_config(),
            )
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
            raise self._wrap(exc) from exc

        self._guard_refusal(message)
        return self._extract_text(message)

    async def parse(
        self,
        *,
        messages: list[ChatMessage],
        output_model: type[ResponseModelT],
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> ResponseModelT:
        """Generate a response validated against a Pydantic model.

        Args:
            messages: Conversation history, oldest first.
            output_model: The schema the response must conform to.
            system: Optional system prompt.
            max_tokens: Optional token cap; falls back to the configured default.

        Returns:
            A validated instance of ``output_model``.

        Raises:
            ProviderNotConfiguredError: No API key is configured.
            ProviderError: The call failed, was refused, or produced no valid object.
        """
        client = self._require_client()
        try:
            parsed = await client.messages.parse(
                model=self._settings.ai_model,
                max_tokens=max_tokens or self._settings.ai_max_tokens,
                messages=self._to_sdk_messages(messages),
                system=system if system is not None else anthropic.omit,
                output_config=self._output_config(),
                output_format=output_model,
            )
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
            raise self._wrap(exc) from exc

        self._guard_refusal(parsed)

        result = parsed.parsed_output
        if result is None:
            raise ProviderError(
                f"Model returned no object matching schema '{output_model.__name__}'"
            )
        return result

    async def aclose(self) -> None:
        """Close the underlying HTTP client, if one was created."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    # -- internals --------------------------------------------------------

    def _require_client(self) -> AsyncAnthropic:
        """Return the API client or fail with an actionable error.

        Returns:
            The constructed async client.

        Raises:
            ProviderNotConfiguredError: No API key is configured.
        """
        if self._client is None:
            raise ProviderNotConfiguredError(_PROVIDER_NAME)
        return self._client

    def _output_config(self) -> OutputConfigParam:
        """Build the per-request output configuration.

        Returns:
            Config carrying the configured reasoning effort level.
        """
        return OutputConfigParam(effort=self._settings.ai_effort.value)

    @staticmethod
    def _to_sdk_messages(messages: list[ChatMessage]) -> list[MessageParam]:
        """Convert Quainex chat turns into the SDK's message shape.

        Args:
            messages: Provider-neutral conversation turns.

        Returns:
            Turns in the SDK's expected format.

        Raises:
            ProviderError: The conversation is empty.
        """
        if not messages:
            raise ProviderError("Cannot send an empty conversation to the model")
        return [MessageParam(role=m.role, content=m.content) for m in messages]

    def _guard_refusal(self, message: Message | ParsedMessage) -> None:
        """Raise if the model declined the request.

        Safety classifiers return a successful HTTP response with
        ``stop_reason == "refusal"``, so this must be checked before reading
        content — the body may be empty or only partially generated.

        Args:
            message: The response to inspect.

        Raises:
            ProviderError: The request was refused.
        """
        if message.stop_reason == "refusal":
            category = getattr(message.stop_details, "category", None)
            self._log.warning("ai_request_refused", category=category)
            raise ProviderError(
                f"The model declined this request (category: {category or 'unspecified'})"
            )

        if message.stop_reason == "max_tokens":
            # Not fatal, but the answer is cut short — surface it in the audit log
            # so a truncated command can be traced back to a too-small budget.
            self._log.warning(
                "ai_response_truncated",
                max_tokens=self._settings.ai_max_tokens,
                detail="Response hit the token ceiling; consider raising QUAINEX_AI_MAX_TOKENS.",
            )

    @staticmethod
    def _extract_text(message: Message) -> str:
        """Concatenate the text blocks of a response.

        Responses may contain non-text blocks (reasoning, tool use) before the
        answer, so blocks are filtered by type rather than indexed positionally.

        Args:
            message: The response to read.

        Returns:
            The joined text content.

        Raises:
            ProviderError: The response contained no text.
        """
        parts = [block.text for block in message.content if block.type == "text"]
        text = "".join(parts).strip()
        if not text:
            raise ProviderError("Model returned a response containing no text")
        return text

    @staticmethod
    def _wrap(exc: anthropic.APIStatusError | anthropic.APIConnectionError) -> ProviderError:
        """Convert an SDK exception into a Quainex ``ProviderError``.

        Args:
            exc: The SDK exception to translate.

        Returns:
            An equivalent Quainex error carrying a readable message.
        """
        if isinstance(exc, anthropic.APIConnectionError):
            return ProviderError(f"Could not reach the Anthropic API: {exc}")
        return ProviderError(f"Anthropic API returned {exc.status_code}: {exc.message}")
