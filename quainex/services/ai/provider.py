"""AI provider contract.

Purpose:
    Define what Quainex needs from *any* language model backend, independent of
    which vendor supplies it.

Why a Protocol (not an ABC):
    The roadmap requires "support multiple AI providers", including local models
    for offline mode. A ``Protocol`` lets an implementation satisfy this contract
    by shape alone — no inheritance from a Quainex base class — so a thin wrapper
    around a local runtime is as easy to plug in as a cloud SDK. It also makes
    test doubles trivial: a small class with the right methods is a valid provider.

Architecture:
    Phase 2 Brain
        -> ``AIProvider`` (this contract)
             |-- ``AnthropicProvider``  (implemented, Phase 1)
             |-- ``OpenAIProvider``     (future)
             +-- ``LocalProvider``      (future, offline mode)

    ``complete()`` returns free text. ``parse()`` returns a validated Pydantic
    model — that second method is what makes the Brain reliable, because intent
    routing needs a guaranteed shape, not prose that happens to look like JSON.

Dependencies:
    pydantic, typing

Example:
    >>> class Echo:
    ...     name = "echo"
    ...     is_available = True
    ...     async def complete(self, *, messages, system=None, max_tokens=None):
    ...         return messages[-1].content
    >>> # `Echo` satisfies AIProvider without importing it.

Future improvements:
    * Add ``stream()`` yielding tokens, once Phase 4 wires voice responses.
    * Add tool/function-calling once Phase 3 lets the model invoke commands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from pathlib import Path

#: A Pydantic model describing the structure a ``parse()`` call must return.
ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)


class ChatMessage(BaseModel):
    """One turn in a conversation sent to a provider.

    Attributes:
        role: Who produced the turn.
        content: The turn's text.
    """

    role: Literal["user", "assistant"]
    content: str


class AIProvider(Protocol):
    """The capabilities Quainex requires of a language model backend."""

    @property
    def name(self) -> str:
        """Short identifier for this provider, used in logs and diagnostics."""
        ...

    @property
    def is_available(self) -> bool:
        """Whether the provider is configured and usable right now.

        Implementations must report ``False`` rather than raising when
        credentials are absent, so callers can degrade gracefully.
        """
        ...

    async def complete(
        self,
        *,
        messages: list[ChatMessage],
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a free-text response.

        Args:
            messages: Conversation history, oldest first. Must end with a user turn.
            system: Optional system prompt setting behaviour and constraints.
            max_tokens: Optional cap on generated tokens; provider default if omitted.

        Returns:
            The generated text.

        Raises:
            quainex.core.exceptions.ProviderError: The upstream call failed.
            quainex.core.exceptions.ProviderNotConfiguredError: No credentials.
        """
        ...

    async def parse(
        self,
        *,
        messages: list[ChatMessage],
        output_model: type[ResponseModelT],
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> ResponseModelT:
        """Generate a response validated against a Pydantic model.

        This is the method the Brain uses. Constraining the model to a schema
        turns "probably parseable" into "guaranteed to validate", which is the
        difference between an intent router that works and one that works most
        of the time.

        Args:
            messages: Conversation history, oldest first. Must end with a user turn.
            output_model: The Pydantic model the response must conform to.
            system: Optional system prompt setting behaviour and constraints.
            max_tokens: Optional cap on generated tokens; provider default if omitted.

        Returns:
            A validated instance of ``output_model``.

        Raises:
            quainex.core.exceptions.ProviderError: The call failed or did not validate.
            quainex.core.exceptions.ProviderNotConfiguredError: No credentials.
        """
        ...

    async def look(
        self,
        *,
        image_paths: list[Path],
        question: str,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Answer a question about one or more images.

        Phase 8 uses this for screen understanding. Vision lives on the provider
        rather than in a separate OCR component because a model that can read
        text in an image can also say what the image *is* — which button to press,
        which window has focus, what an error dialog means. Bolting on a
        text-extraction library would return characters and lose all of that.

        Args:
            image_paths: Images to examine (PNG, JPEG, GIF or WebP).
            question: What to ask about them.
            system: Optional system prompt.
            max_tokens: Optional cap on generated tokens.

        Returns:
            The answer.

        Raises:
            quainex.core.exceptions.ProviderError: The call failed or the image
                could not be read.
            quainex.core.exceptions.ProviderNotConfiguredError: No credentials.
        """
        ...

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
            question: What to ask about it.
            system: Optional system prompt.
            max_tokens: Optional cap on generated tokens.

        Returns:
            The answer.

        Raises:
            quainex.core.exceptions.ProviderError: The call failed or the file
                could not be read.
            quainex.core.exceptions.ProviderNotConfiguredError: No credentials.
        """
        ...

    async def aclose(self) -> None:
        """Release any held resources (HTTP connections, sockets, sessions)."""
        ...
