"""Google Gemini implementation of the ``AIProvider`` contract.

Purpose:
    Give Quainex a capable provider with a genuinely usable free tier, so the
    system is worth running before anyone has paid for anything.

Why raw HTTP rather than the ``google-genai`` SDK:
    The Gemini REST surface Quainex needs is one endpoint — ``generateContent``.
    Against that, an SDK adds a dependency, its own transitive tree, and another
    thing to break on a machine where large installs are unreliable. ``httpx``
    is already here for the Telegram bridge.

    That trade would be wrong for authentication (see ``auth/tokens.py``, where
    PyJWT is used precisely because hand-rolling is dangerous). It is right here:
    a wrong request shape produces a clear 400, not a silent security hole.

Structured output:
    Gemini takes ``responseSchema`` natively, which is why ``parse()`` is
    reliable rather than hopeful. The schema still needs flattening first — see
    ``schemas.py``, since Gemini does not follow ``$ref``.

Architecture:
    AIProvider (Protocol)
        -> GeminiProvider
             -> POST /v1beta/models/{model}:generateContent
                  |-- contents[]        conversation turns
                  |-- systemInstruction system prompt
                  +-- generationConfig  responseSchema for parse()

Dependencies:
    httpx, pydantic, quainex.services.ai.schemas

Future improvements:
    * Streaming via ``streamGenerateContent`` for voice responses.
    * Gemini's own vision path rather than falling back to text-only.
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any

import httpx

from quainex.core.exceptions import ProviderError, ProviderNotConfiguredError
from quainex.core.logging import get_logger
from quainex.services.ai.provider import ChatMessage, ResponseModelT
from quainex.services.ai.schemas import gemini_schema, parse_model

if TYPE_CHECKING:
    from pathlib import Path

    from quainex.config.settings import Settings

_log = get_logger(__name__)

_API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
_TIMEOUT_SECONDS = 120

_IMAGE_MIME_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".pdf": "application/pdf",
}


class GeminiProvider:
    """``AIProvider`` backed by Google's Gemini API."""

    def __init__(self, settings: Settings) -> None:
        """Construct the provider.

        Does not raise without credentials — Quainex must boot regardless, and
        ``is_available`` reports the truth.

        Args:
            settings: Configuration supplying the API key and model.
        """
        self._settings = settings
        self._client: httpx.AsyncClient | None = None
        if settings.gemini_api_key:
            self._client = httpx.AsyncClient(timeout=_TIMEOUT_SECONDS)

    @property
    def name(self) -> str:
        """Short identifier for this provider."""
        return f"gemini/{self._settings.gemini_model}"

    @property
    def is_available(self) -> bool:
        """Whether an API key is configured."""
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
            max_tokens: Optional token cap.

        Returns:
            The generated text.

        Raises:
            ProviderNotConfiguredError: No API key.
            ProviderError: The call failed or returned nothing usable.
        """
        payload = self._base_payload(messages, system, max_tokens)
        return self._text_of(await self._post(payload))

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
            ProviderNotConfiguredError: No API key.
            ProviderError: The call failed, or the reply did not validate.
        """
        payload = self._base_payload(messages, system, max_tokens)
        payload["generationConfig"]["responseMimeType"] = "application/json"
        payload["generationConfig"]["responseSchema"] = gemini_schema(output_model)

        text = self._text_of(await self._post(payload))
        parsed = parse_model(text, output_model)
        if parsed is None:
            raise ProviderError(
                f"Gemini returned a reply that does not match "
                f"'{output_model.__name__}'. First 200 characters: {text[:200]}"
            )
        return parsed

    async def look(
        self,
        *,
        image_paths: list[Path],
        question: str,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Answer a question about one or more images.

        Args:
            image_paths: Images to examine.
            question: What to ask.
            system: Optional system prompt.
            max_tokens: Optional token cap.

        Returns:
            The answer.

        Raises:
            ProviderError: The call failed or an image was unreadable.
        """
        parts: list[dict[str, Any]] = [self._inline_file(path) for path in image_paths[:8]]
        parts.append({"text": question})

        payload = self._base_payload([], system, max_tokens)
        payload["contents"] = [{"role": "user", "parts": parts}]
        return self._text_of(await self._post(payload))

    async def read_document(
        self,
        *,
        document_path: Path,
        question: str,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Answer a question about a PDF.

        Args:
            document_path: The PDF to read.
            question: What to ask.
            system: Optional system prompt.
            max_tokens: Optional token cap.

        Returns:
            The answer.
        """
        payload = self._base_payload([], system, max_tokens)
        payload["contents"] = [
            {"role": "user", "parts": [self._inline_file(document_path), {"text": question}]}
        ]
        return self._text_of(await self._post(payload))

    async def aclose(self) -> None:
        """Close the HTTP client, if one was created."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # -- internals --------------------------------------------------------

    def _base_payload(
        self, messages: list[ChatMessage], system: str | None, max_tokens: int | None
    ) -> dict[str, Any]:
        """Build the common request body.

        Args:
            messages: Conversation turns.
            system: Optional system prompt.
            max_tokens: Optional token cap.

        Returns:
            The request payload.
        """
        payload: dict[str, Any] = {
            # Gemini calls the assistant role "model"; everything else is "user".
            "contents": [
                {
                    "role": "model" if message.role == "assistant" else "user",
                    "parts": [{"text": message.content}],
                }
                for message in messages
            ],
            "generationConfig": {
                "maxOutputTokens": max_tokens or self._settings.ai_max_tokens,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        return payload

    def _inline_file(self, path: Path) -> dict[str, Any]:
        """Encode a file as an inline data part.

        Args:
            path: The file to encode.

        Returns:
            The content part.

        Raises:
            ProviderError: Unsupported type, missing, or unreadable.
        """
        mime = _IMAGE_MIME_TYPES.get(path.suffix.lower())
        if mime is None:
            supported = ", ".join(sorted(_IMAGE_MIME_TYPES))
            raise ProviderError(f"'{path.suffix}' is not supported. Supported: {supported}.")
        if not path.is_file():
            raise ProviderError(f"No file at '{path}'.")

        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise ProviderError(f"Could not read '{path}': {exc}") from exc

        return {
            "inlineData": {
                "mimeType": mime,
                "data": base64.standard_b64encode(payload).decode("ascii"),
            }
        }

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a generateContent request.

        Args:
            payload: The request body.

        Returns:
            The decoded response.

        Raises:
            ProviderNotConfiguredError: No API key.
            ProviderError: The request failed.
        """
        if self._client is None or self._settings.gemini_api_key is None:
            raise ProviderNotConfiguredError("gemini")

        url = f"{_API_ROOT}/models/{self._settings.gemini_model}:generateContent"
        try:
            response = await self._client.post(
                url,
                # Header rather than a query parameter: a key in a URL lands in
                # access logs and proxy history.
                headers={"x-goog-api-key": self._settings.gemini_api_key.get_secret_value()},
                json=payload,
            )
        except httpx.HTTPError as exc:
            raise ProviderError(f"Could not reach the Gemini API: {exc}") from exc

        if response.status_code >= 400:
            detail = response.text[:300]
            raise ProviderError(f"Gemini returned {response.status_code}: {detail}")

        try:
            return dict(response.json())
        except ValueError as exc:
            raise ProviderError(f"Gemini returned a non-JSON response: {exc}") from exc

    @staticmethod
    def _text_of(response: dict[str, Any]) -> str:
        """Extract the reply text from a response.

        Args:
            response: The decoded response body.

        Returns:
            The concatenated text parts.

        Raises:
            ProviderError: The request was blocked or produced no text.
        """
        candidates = response.get("candidates") or []
        if not candidates:
            # A blocked prompt returns no candidates and a reason worth showing.
            reason = (response.get("promptFeedback") or {}).get("blockReason")
            raise ProviderError(
                f"Gemini declined the request ({reason})."
                if reason
                else "Gemini returned no reply."
            )

        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts).strip()
        if not text:
            finish = candidates[0].get("finishReason", "unknown")
            raise ProviderError(f"Gemini returned an empty reply (finishReason: {finish}).")

        _log.info("gemini_response", characters=len(text))
        return text
