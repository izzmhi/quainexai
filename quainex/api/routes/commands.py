"""Command endpoints — interpret, confirm, execute.

Purpose:
    Expose the command layer over HTTP, keeping the two dangerous transitions —
    deciding what was meant, and doing it — visible and separately auditable.

The three endpoints and why there are three:

    ``GET  /commands``        what Quainex can do. Useful to a UI, and to a user
                              wondering why a request was refused.
    ``POST /commands/execute`` act on an already-classified intent. The caller
                              supplies the intent, so a client can show it, edit
                              it, or drop it before anything happens.
    ``POST /commands/ask``     the convenience path: classify then execute in one
                              call. Still cannot bypass confirmation — an intent
                              needing approval comes back unexecuted with the
                              question to ask.

    Splitting them means the "one call does everything" path exists for
    convenience without becoming the only path. A voice loop wanting to confirm
    with the user mid-flow uses the first two.

Architecture:
    POST /commands/ask -> Brain.interpret() -> CommandExecutor.execute()
                                                    |-- gates -> refusal (200, executed=false)
                                                    +-- dispatch -> CommandResult

Dependencies:
    fastapi, pydantic, quainex.api.dependencies, quainex.core.{brain,commands}

Future improvements:
    * Stream results over the WebSocket so long actions report progress.
    * Signed confirmation tokens, so "confirmed" cannot be asserted by a caller
      that never showed the user anything (Phase 6).
"""

from __future__ import annotations

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from quainex.api.dependencies import ContainerDep
from quainex.core.brain import Intent
from quainex.core.commands import CommandResult
from quainex.services.ai.provider import ChatMessage

router = APIRouter(prefix="/commands", tags=["commands"])


class ExecuteRequest(BaseModel):
    """Body of an execution request.

    Attributes:
        intent: A previously classified intent, typically from ``/brain/interpret``.
        confirmed: Whether the user has explicitly approved this action. Only
            consulted when the intent is flagged as needing confirmation.
    """

    intent: Intent
    confirmed: bool = False


class AskRequest(BaseModel):
    """Body of a combined interpret-and-execute request.

    Attributes:
        utterance: What the user said or typed.
        history: Optional prior turns for resolving references.
        confirmed: Blanket approval for this utterance. Use with care — it
            pre-approves whatever the utterance turns out to mean.
    """

    utterance: str = Field(min_length=1, examples=["Open VS Code"])
    history: list[ChatMessage] = Field(default_factory=list)
    confirmed: bool = False


class AskResponse(BaseModel):
    """Result of a combined interpret-and-execute request.

    Attributes:
        intent: What Quainex understood.
        result: What it did about it.
    """

    intent: Intent
    result: CommandResult


@router.get("", summary="List every executable command")
async def list_commands(container: ContainerDep) -> dict[str, str]:
    """Describe the commands Quainex can execute.

    Args:
        container: Injected application container.

    Returns:
        Intent value mapped to a one-line summary.
    """
    return container.commands.catalogue


@router.post(
    "/execute",
    response_model=CommandResult,
    status_code=status.HTTP_200_OK,
    summary="Execute an already-classified intent",
)
async def execute(request: ExecuteRequest, container: ContainerDep) -> CommandResult:
    """Act on an intent, subject to confirmation and configuration policy.

    A refusal is a 200 with ``executed: false`` and an explanatory message, not
    an error — the caller asked a legitimate question and got a definite answer.

    Args:
        request: The intent to execute and whether it has been confirmed.
        container: Injected application container.

    Returns:
        The outcome, including whether any side effect occurred.
    """
    return container.commands.execute(request.intent, confirmed=request.confirmed)


@router.post(
    "/ask",
    response_model=AskResponse,
    status_code=status.HTTP_200_OK,
    summary="Interpret an utterance and execute it",
    responses={
        400: {"description": "The utterance was empty or too long."},
        503: {"description": "No AI provider is configured."},
    },
)
async def ask(request: AskRequest, container: ContainerDep) -> AskResponse:
    """Classify an utterance and act on it in one call.

    Args:
        request: The utterance, optional history, and blanket approval flag.
        container: Injected application container.

    Returns:
        Both what was understood and what was done.
    """
    intent = await container.brain.interpret(
        utterance=request.utterance,
        history=request.history,
    )
    result = container.commands.execute(intent, confirmed=request.confirmed)
    return AskResponse(intent=intent, result=result)
