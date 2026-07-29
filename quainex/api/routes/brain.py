"""Brain endpoint — natural language interpretation.

Purpose:
    Expose intent classification over HTTP so the future dashboard, phone client
    and voice loop all consume one implementation.

Design note — why this endpoint does not execute anything:
    ``POST /brain/interpret`` is deliberately read-only. Separating "work out
    what was meant" from "do it" means the interpretation can be shown to the
    user, logged, or rejected before anything touches the machine. Phase 3 adds
    ``/commands/execute``, which takes an already-classified intent and honours
    ``requires_confirmation``.

Architecture:
    POST /brain/interpret
        -> Depends(get_container) -> container.brain
        -> Brain.interpret()
        -> Intent (200) | InvalidUtteranceError (400) | ProviderNotConfigured (503)

Dependencies:
    fastapi, pydantic, quainex.api.dependencies, quainex.core.brain

Example:
    $ curl -X POST http://127.0.0.1:8000/brain/interpret \
        -H "Content-Type: application/json" \
        -d '{"utterance": "Open VS Code"}'

Future improvements:
    * Accept an audio blob and run Phase 4 transcription before classifying.
    * Return candidate alternatives when confidence is low, so the UI can offer
      a choice instead of a yes/no confirmation.
"""

from __future__ import annotations

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from quainex.api.dependencies import ContainerDep
from quainex.core.brain import Intent
from quainex.services.ai.provider import ChatMessage

router = APIRouter(prefix="/brain", tags=["brain"])


class InterpretRequest(BaseModel):
    """Body of an interpretation request.

    Attributes:
        utterance: What the user said or typed.
        history: Optional prior turns, oldest first, for resolving references
            such as "close it".
    """

    utterance: str = Field(min_length=1, examples=["Open VS Code"])
    history: list[ChatMessage] = Field(default_factory=list)


@router.post(
    "/interpret",
    response_model=Intent,
    status_code=status.HTTP_200_OK,
    summary="Classify natural language into a structured intent",
    responses={
        400: {"description": "The utterance was empty or too long."},
        503: {"description": "No AI provider is configured."},
    },
)
async def interpret(request: InterpretRequest, container: ContainerDep) -> Intent:
    """Classify an utterance without executing anything.

    Args:
        request: The utterance and optional conversation history.
        container: Injected application container.

    Returns:
        The classified intent, including whether it needs user confirmation.
    """
    return await container.brain.interpret(
        utterance=request.utterance,
        history=request.history,
    )
