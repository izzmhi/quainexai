"""Health and readiness endpoint.

Purpose:
    Report whether the process is alive and what state its subsystems are in.

Design note — liveness vs readiness:
    ``/health`` must never fail because a *dependency* is unhealthy. If a missing
    API key made this endpoint return 503, a supervisor would restart Quainex in
    a loop over a problem restarting cannot fix. So subsystem state is *reported*
    in the body (``ai.available``) rather than encoded in the status code.

Architecture:
    GET /health -> Depends(get_container) -> read settings + provider state
                -> HealthResponse (typed, appears in the OpenAPI schema)

Dependencies:
    fastapi, pydantic, quainex.api.dependencies

Example:
    $ curl http://127.0.0.1:8000/health
    {"status":"ok","app":"Quainex","version":"0.1.0","environment":"dev",
     "uptime_seconds":12.4,"ai":{"provider":"anthropic","available":false}}

Future improvements:
    * Add a separate ``/ready`` that *does* gate on dependencies, for Phase 6
      when a load balancer needs to know whether to route traffic here.
"""

from __future__ import annotations

import time
from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from quainex.api.dependencies import ContainerDep

router = APIRouter(tags=["system"])


class AIStatus(BaseModel):
    """State of the configured AI provider.

    Attributes:
        provider: Name of the active provider implementation.
        available: Whether it holds the credentials needed to serve requests.
        model: The model identifier it will call.
    """

    provider: str
    available: bool
    model: str


class HealthResponse(BaseModel):
    """Payload returned by the health endpoint.

    Attributes:
        status: Always ``"ok"`` — reaching this handler proves liveness.
        app: Configured application name.
        version: Running application version.
        environment: Active environment (``dev`` or ``prod``).
        uptime_seconds: Seconds since the application finished starting.
        ai: State of the AI subsystem.
    """

    status: Literal["ok"] = "ok"
    app: str
    version: str
    environment: str
    uptime_seconds: float = Field(ge=0)
    ai: AIStatus


@router.get("/health", response_model=HealthResponse, summary="Liveness and subsystem status")
async def health(request: Request, container: ContainerDep) -> HealthResponse:
    """Report process liveness and the state of each subsystem.

    Args:
        request: The incoming request, used to read the recorded start time.
        container: Injected application container.

    Returns:
        The current health snapshot.
    """
    settings = container.settings
    started_at = getattr(request.app.state, "started_at", None)
    uptime = time.monotonic() - started_at if isinstance(started_at, float) else 0.0

    return HealthResponse(
        app=settings.app_name,
        version=settings.version,
        environment=settings.environment.value,
        uptime_seconds=round(uptime, 3),
        ai=AIStatus(
            provider=container.ai_provider.name,
            available=container.ai_provider.is_available,
            model=settings.ai_model,
        ),
    )
