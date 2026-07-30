"""Provider fallback chain.

Purpose:
    Try providers in order, so Quainex uses the free ones first and only reaches
    a paid API when it has to.

Why this is a provider rather than logic inside the Brain:
    ``FallbackProvider`` satisfies ``AIProvider`` itself, so everything upstream
    — Brain, agent, plugins, vision — is unchanged and unaware. Fallback is a
    composition concern, and putting it in the Brain would mean re-implementing
    it in the agent and again in every plugin.

    This is the payoff for defining the contract as a Protocol back in Phase 1.

What counts as a reason to move on:
    Only *provider* failures: no credentials, a network error, a bad status, an
    empty reply. A refusal is **not** a fallback trigger — if a model declines a
    request, asking a different model the same thing is shopping for a yes, not
    error recovery. Refusals propagate.

    Capability gaps do fall through: an OpenAI-compatible endpoint with no vision
    raises ``ProviderError`` for ``look()``, and the chain moves to one that has
    it. That is how a Groq-first setup still answers "what's on my screen".

Architecture:
    FallbackProvider([groq, gemini, anthropic])
        -> complete/parse/look/read_document
             for each available provider:
                 try   -> return
                 catch ProviderError -> log, try the next
             all failed -> raise the last error, naming everything tried

Dependencies:
    quainex.core.exceptions, quainex.services.ai.provider

Future improvements:
    * Remember which provider last worked and start there.
    * Briefly bench a provider that just failed, instead of retrying it first
      on every request.
"""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

from quainex.core.exceptions import ProviderError, ProviderNotConfiguredError
from quainex.core.logging import get_logger
from quainex.services.ai.provider import ResponseModelT

if TYPE_CHECKING:
    from pathlib import Path

    from quainex.services.ai.provider import AIProvider, ChatMessage

_log = get_logger(__name__)


class FallbackProvider:
    """Tries several providers in order until one answers."""

    def __init__(self, providers: list[AIProvider]) -> None:
        """Construct the chain.

        Args:
            providers: Providers in preference order. Unavailable ones are
                skipped at call time rather than filtered here, so configuring a
                key later starts working without a restart.
        """
        self._providers = providers

    @property
    def name(self) -> str:
        """Identifier naming the chain in preference order."""
        available = [p.name for p in self._providers if p.is_available]
        return f"chain({' -> '.join(available)})" if available else "chain(none available)"

    @property
    def is_available(self) -> bool:
        """Whether any provider in the chain is usable."""
        return any(provider.is_available for provider in self._providers)

    @property
    def providers(self) -> list[AIProvider]:
        """The chain, in preference order.

        Returns a copy: handing out the live list would let a caller reorder the
        chain behind ``replace``'s back.
        """
        return list(self._providers)

    async def replace(self, providers: list[AIProvider]) -> None:
        """Swap the whole chain in place, closing the providers being discarded.

        In place, deliberately. The Brain, agent, plugins and vision analyst each
        hold a reference to *this* object, captured at construction. Rebuilding
        the chain by assigning a new ``FallbackProvider`` onto the container would
        leave every one of them pointing at the old one, so a key saved in the
        dashboard would appear to work and change nothing. Mutating the list
        behind a stable identity is what makes hot-reload actually reach the
        components that matter.

        Args:
            providers: The replacement chain, in preference order.
        """
        previous = self._providers
        self._providers = providers
        for provider in previous:
            # A failure closing a discarded client must not abort the swap: the
            # new chain is already live and the old socket will be collected.
            with suppress(Exception):
                await provider.aclose()
        _log.info("provider_chain_replaced", chain=self.name)

    def describe(self) -> list[dict[str, object]]:
        """Report each provider and whether it is usable.

        Returns:
            One entry per provider, in preference order.
        """
        return [
            {"name": provider.name, "available": provider.is_available, "order": index}
            for index, provider in enumerate(self._providers)
        ]

    async def complete(
        self,
        *,
        messages: list[ChatMessage],
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a free-text response from the first provider that answers.

        Args:
            messages: Conversation history, oldest first.
            system: Optional system prompt.
            max_tokens: Optional token cap.

        Returns:
            The generated text.

        Raises:
            ProviderNotConfiguredError: No provider is configured.
            ProviderError: Every configured provider failed.
        """
        errors: list[str] = []
        for provider in self._usable():
            try:
                return await provider.complete(
                    messages=messages, system=system, max_tokens=max_tokens
                )
            except ProviderError as exc:
                errors.append(self._note(provider, exc))
        raise self._exhausted(errors)

    async def parse(
        self,
        *,
        messages: list[ChatMessage],
        output_model: type[ResponseModelT],
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> ResponseModelT:
        """Generate a validated response from the first provider that answers.

        Args:
            messages: Conversation history, oldest first.
            output_model: The schema the response must satisfy.
            system: Optional system prompt.
            max_tokens: Optional token cap.

        Returns:
            A validated instance.

        Raises:
            ProviderNotConfiguredError: No provider is configured.
            ProviderError: Every configured provider failed.
        """
        errors: list[str] = []
        for provider in self._usable():
            try:
                return await provider.parse(
                    messages=messages,
                    output_model=output_model,
                    system=system,
                    max_tokens=max_tokens,
                )
            except ProviderError as exc:
                errors.append(self._note(provider, exc))
        raise self._exhausted(errors)

    async def look(
        self,
        *,
        image_paths: list[Path],
        question: str,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Answer a question about images, from the first provider that can.

        Providers without vision raise, and the chain moves on — so a
        Groq-first setup still answers screen questions through Gemini.

        Args:
            image_paths: Images to examine.
            question: What to ask.
            system: Optional system prompt.
            max_tokens: Optional token cap.

        Returns:
            The answer.

        Raises:
            ProviderNotConfiguredError: No provider is configured.
            ProviderError: No provider could answer.
        """
        errors: list[str] = []
        for provider in self._usable():
            try:
                return await provider.look(
                    image_paths=image_paths,
                    question=question,
                    system=system,
                    max_tokens=max_tokens,
                )
            except ProviderError as exc:
                errors.append(self._note(provider, exc))
        raise self._exhausted(errors)

    async def read_document(
        self,
        *,
        document_path: Path,
        question: str,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Answer a question about a PDF, from the first provider that can.

        Args:
            document_path: The PDF to read.
            question: What to ask.
            system: Optional system prompt.
            max_tokens: Optional token cap.

        Returns:
            The answer.

        Raises:
            ProviderNotConfiguredError: No provider is configured.
            ProviderError: No provider could answer.
        """
        errors: list[str] = []
        for provider in self._usable():
            try:
                return await provider.read_document(
                    document_path=document_path,
                    question=question,
                    system=system,
                    max_tokens=max_tokens,
                )
            except ProviderError as exc:
                errors.append(self._note(provider, exc))
        raise self._exhausted(errors)

    async def aclose(self) -> None:
        """Close every provider in the chain."""
        for provider in self._providers:
            await provider.aclose()

    # -- internals --------------------------------------------------------

    def _usable(self) -> list[AIProvider]:
        """Return the configured providers, in preference order.

        Returns:
            Providers reporting themselves available.
        """
        return [provider for provider in self._providers if provider.is_available]

    @staticmethod
    def _note(provider: AIProvider, exc: ProviderError) -> str:
        """Record a provider failure and move on.

        Args:
            provider: The provider that failed.
            exc: What went wrong.

        Returns:
            A short description for the combined error.
        """
        _log.warning("provider_fell_through", provider=provider.name, reason=exc.message)
        return f"{provider.name}: {exc.message}"

    def _exhausted(self, errors: list[str]) -> ProviderError:
        """Build the error raised when no provider answered.

        Args:
            errors: What each provider said.

        Returns:
            The error to raise, naming everything that was tried.
        """
        if not errors:
            return ProviderNotConfiguredError(
                "none",
                remedy=(
                    "No provider in the chain has credentials. Add a free key in "
                    "the dashboard's Settings panel, or set QUAINEX_GROQ_API_KEY "
                    "or QUAINEX_GEMINI_API_KEY in .env."
                ),
            )
        return ProviderError("Every AI provider failed. " + "; ".join(errors))
