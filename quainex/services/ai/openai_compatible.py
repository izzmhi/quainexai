"""Provider for any OpenAI-compatible chat endpoint.

Purpose:
    Cover Groq, Ollama, OpenRouter, LM Studio and anything else that speaks the
    ``/chat/completions`` shape — with one implementation rather than one each.

Why one class instead of a ``GroqProvider``:
    The differences between these services are a base URL, a model name and a
    key. Writing a class per vendor would be four copies of the same file
    diverging over time. Groq is the default here because its free tier is fast
    enough to make voice feel immediate — but the same class points at a local
    Ollama with two settings changed, which is the offline mode the roadmap wants.

Structured output, honestly:
    Support ranges from strict JSON-schema enforcement to nothing at all, and it
    varies by *model* as well as by vendor. Rather than detect that, ``parse()``
    asks for JSON mode, states the schema in the system prompt, and validates
    what comes back — retrying once with the validation error fed in.

    That is less reliable than Anthropic's or Gemini's native schema support, and
    it is why those two rank above this one in the default fallback order. It is
    good enough for intent classification and it costs nothing.

Architecture:
    AIProvider (Protocol)
        -> OpenAICompatibleProvider(base_url, model, key)
             -> POST {base_url}/chat/completions
                  parse(): response_format=json_object + schema in prompt
                           -> validate -> retry once with the error -> give up

Dependencies:
    httpx, pydantic, quainex.services.ai.schemas

Future improvements:
    * Use ``response_format: json_schema`` where the model supports it.
    * Vision for the endpoints that support image content parts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx

from quainex.core.exceptions import ProviderError, ProviderNotConfiguredError
from quainex.core.logging import get_logger
from quainex.services.ai.provider import ChatMessage, ResponseModelT
from quainex.services.ai.schemas import parse_model, schema_instruction

if TYPE_CHECKING:
    from pathlib import Path

_log = get_logger(__name__)

_TIMEOUT_SECONDS = 120


class OpenAICompatibleProvider:
    """``AIProvider`` for any endpoint speaking the OpenAI chat API."""

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        model: str,
        api_key: str | None,
        max_tokens: int,
    ) -> None:
        """Construct the provider.

        Args:
            name: Short identifier, e.g. ``groq``.
            base_url: API root, e.g. ``https://api.groq.com/openai/v1``.
            model: Model identifier.
            api_key: Bearer key. ``None`` for local servers that need none.
            max_tokens: Default token cap.
        """
        self._name = name
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._max_tokens = max_tokens
        # A local endpoint legitimately needs no key, so availability is about
        # having a URL, not about holding a secret.
        self._client: httpx.AsyncClient | None = (
            httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) if base_url else None
        )

    @property
    def name(self) -> str:
        """Short identifier for this provider."""
        return f"{self._name}/{self._model}"

    @property
    def is_available(self) -> bool:
        """Whether the provider is configured enough to try."""
        return self._client is not None and bool(self._model)

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
            max_tokens: Optional token cap.

        Returns:
            The generated text.

        Raises:
            ProviderNotConfiguredError: Not configured.
            ProviderError: The call failed.
        """
        return await self._chat(self._messages(messages, system), max_tokens)

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
            output_model: The schema the response must satisfy.
            system: Optional system prompt.
            max_tokens: Optional token cap.

        Returns:
            A validated instance.

        Raises:
            ProviderNotConfiguredError: Not configured.
            ProviderError: The call failed, or the reply never validated.
        """
        instruction = schema_instruction(output_model)
        combined = f"{system}\n\n{instruction}" if system else instruction
        payload = self._messages(messages, combined)

        text = await self._chat(payload, max_tokens, json_mode=True)
        if parsed := parse_model(text, output_model):
            return parsed

        # One retry, with the failure shown. Models that miss a required field
        # usually fix it when told which one; a second failure means the schema
        # is beyond this model and pretending otherwise wastes another call.
        _log.info("structured_output_retry", provider=self._name)
        payload.append({"role": "assistant", "content": text})
        payload.append(
            {
                "role": "user",
                "content": (
                    "That did not match the schema. Reply again with only the JSON "
                    "object, including every required field."
                ),
            }
        )
        retry = await self._chat(payload, max_tokens, json_mode=True)
        if parsed := parse_model(retry, output_model):
            return parsed

        raise ProviderError(
            f"{self._name} did not produce valid '{output_model.__name__}' JSON "
            f"after a retry. First 200 characters: {retry[:200]}"
        )

    async def look(
        self,
        *,
        image_paths: list[Path],
        question: str,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Vision is not implemented for this provider.

        Args:
            image_paths: Ignored.
            question: Ignored.
            system: Ignored.
            max_tokens: Ignored.

        Raises:
            ProviderError: Always. Vision support varies too much between
                OpenAI-compatible endpoints to promise here; the fallback chain
                routes vision to a provider that has it.
        """
        raise ProviderError(f"{self._name} does not support image input in Quainex yet.")

    async def read_document(
        self,
        *,
        document_path: Path,
        question: str,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Document reading is not implemented for this provider.

        Args:
            document_path: Ignored.
            question: Ignored.
            system: Ignored.
            max_tokens: Ignored.

        Raises:
            ProviderError: Always. See ``look``.
        """
        raise ProviderError(f"{self._name} does not support PDF input in Quainex yet.")

    async def aclose(self) -> None:
        """Close the HTTP client, if one was created."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # -- internals --------------------------------------------------------

    @staticmethod
    def _messages(messages: list[ChatMessage], system: str | None) -> list[dict[str, str]]:
        """Build the message array.

        Args:
            messages: Conversation turns.
            system: Optional system prompt.

        Returns:
            Messages in OpenAI format.
        """
        payload: list[dict[str, str]] = []
        if system:
            payload.append({"role": "system", "content": system})
        payload.extend({"role": m.role, "content": m.content} for m in messages)
        return payload

    async def _chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None,
        json_mode: bool = False,
    ) -> str:
        """Send a chat completion request.

        Args:
            messages: The message array.
            max_tokens: Optional token cap.
            json_mode: Whether to request a JSON object response.

        Returns:
            The reply text.

        Raises:
            ProviderNotConfiguredError: Not configured.
            ProviderError: The request failed or returned no text.
        """
        if self._client is None:
            raise ProviderNotConfiguredError(self._name)

        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens or self._max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

        try:
            response = await self._client.post(
                f"{self._base_url}/chat/completions", headers=headers, json=body
            )
        except httpx.HTTPError as exc:
            raise ProviderError(f"Could not reach {self._name}: {exc}") from exc

        if response.status_code >= 400:
            raise ProviderError(
                f"{self._name} returned {response.status_code}: {response.text[:300]}"
            )

        try:
            data = response.json()
            text = str(data["choices"][0]["message"]["content"] or "").strip()
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                f"{self._name} returned an unexpected response shape: {exc}"
            ) from exc

        if not text:
            # Name the usual cause. On a reasoning model the token budget funds the
            # thinking as well as the answer, so too small a cap produces a
            # perfectly successful call with nothing visible in it — which reads as
            # a broken model unless the message says otherwise.
            budget = max_tokens or self._max_tokens
            raise ProviderError(
                f"{self._name} returned an empty reply. If this is a reasoning "
                f"model, {budget} tokens may all have gone on reasoning before any "
                f"answer was produced — try raising the cap."
            )

        _log.info("chat_response", provider=self._name, characters=len(text))
        return text
