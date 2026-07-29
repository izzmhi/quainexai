"""Autonomous agent and plugin endpoints.

Purpose:
    Expose goal execution and the plugin system over HTTP.

Why an agent run returns 200 even when it stops early:
    ``awaiting_confirmation`` and ``budget_exhausted`` are outcomes, not errors.
    The run did what it was designed to do — stop rather than proceed without a
    human, or stop rather than exceed its ceiling. Reporting those as failures
    would train a client to treat the safety machinery as noise.

Architecture:
    POST /agent/run        goal -> plan -> bounded execution -> AgentRun
    GET  /plugins          discovered plugins and requested permissions
    POST /plugins/{n}/enable   load a plugin's code (first time it runs)
    POST /plugins/{n}/disable
    POST /plugins/{n}/invoke   call one of its declared actions

Dependencies:
    fastapi, quainex.core.agent, quainex.plugins

Future improvements:
    * Stream agent steps over the WebSocket as they happen.
    * Resume a paused run once the user confirms, rather than starting over.
"""

from __future__ import annotations

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from quainex.api.dependencies import ContainerDep
from quainex.core.agent import AgentBudget, AgentRun
from quainex.plugins import DiscoveredPlugin, PluginRequest, PluginResponse

agent_router = APIRouter(prefix="/agent", tags=["agent"])
plugin_router = APIRouter(prefix="/plugins", tags=["plugins"])


class RunRequest(BaseModel):
    """Body of an autonomous run request.

    Attributes:
        goal: What to achieve.
        max_steps: Optional override of the step ceiling.
        max_actions: Optional override of the action ceiling.
        max_seconds: Optional override of the time ceiling.
    """

    goal: str = Field(min_length=1, examples=["run the tests and tell me if they pass"])
    max_steps: int | None = Field(default=None, ge=1, le=100)
    max_actions: int | None = Field(default=None, ge=1, le=200)
    max_seconds: float | None = Field(default=None, gt=0, le=3600)


@agent_router.post(
    "/run",
    response_model=AgentRun,
    status_code=status.HTTP_200_OK,
    summary="Plan and carry out a goal autonomously",
    responses={503: {"description": "No AI provider is configured."}},
)
async def run_goal(request: RunRequest, container: ContainerDep) -> AgentRun:
    """Plan a goal and execute it within a budget.

    The agent cannot approve its own confirmations. A goal implying a
    confirmation-gated action returns ``awaiting_confirmation`` with the action
    unexecuted, for a human to decide on.

    Args:
        request: The goal and any budget overrides.
        container: Injected application container.

    Returns:
        The complete run record.
    """
    settings = container.settings
    budget = AgentBudget(
        max_steps=request.max_steps or settings.agent_max_steps,
        max_seconds=request.max_seconds or settings.agent_max_seconds,
        max_actions=request.max_actions or settings.agent_max_actions,
        max_repeats_per_action=settings.agent_max_repeats,
    )
    return await container.agent.run(request.goal, budget)


@plugin_router.get("", summary="List discovered plugins")
async def list_plugins(container: ContainerDep) -> list[DiscoveredPlugin]:
    """Report every plugin found on disk.

    Discovery reads manifests only — no plugin code runs here, which is what
    makes the reported permission list meaningful before you enable anything.

    Args:
        container: Injected application container.

    Returns:
        Discovered plugins, with their declared permissions.
    """
    return container.plugins.discover()


@plugin_router.post("/{name}/enable", summary="Enable a plugin")
async def enable_plugin(name: str, container: ContainerDep) -> DiscoveredPlugin:
    """Load a plugin's code and grant it the capabilities it declared.

    This is the first time the plugin's code runs.

    Args:
        name: The plugin to enable.
        container: Injected application container.

    Returns:
        Its updated record.
    """
    return container.plugins.enable(name)


@plugin_router.post("/{name}/disable", summary="Disable a plugin")
async def disable_plugin(name: str, container: ContainerDep) -> dict[str, bool]:
    """Stop routing requests to a plugin.

    Args:
        name: The plugin to disable.
        container: Injected application container.

    Returns:
        Whether it had been enabled.
    """
    return {"disabled": container.plugins.disable(name)}


@plugin_router.post("/{name}/invoke", summary="Call a plugin action")
async def invoke_plugin(
    name: str, request: PluginRequest, container: ContainerDep
) -> PluginResponse:
    """Invoke one of a plugin's declared actions.

    Args:
        name: The plugin to call.
        request: The action and its arguments.
        container: Injected application container.

    Returns:
        The plugin's response.
    """
    return await container.plugins.invoke(name, request)
