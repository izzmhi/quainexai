"""Autonomous goal execution.

Phase 10. Plans a goal into steps and carries them out inside a budget, with no
means of approving its own confirmations.
"""

from quainex.core.agent.budget import AgentBudget, BudgetOutcome, explain
from quainex.core.agent.loop import (
    AgentRun,
    AutonomousAgent,
    Plan,
    PlanStep,
    RunStatus,
    StepRecord,
)

__all__ = [
    "AgentBudget",
    "AgentRun",
    "AutonomousAgent",
    "BudgetOutcome",
    "Plan",
    "PlanStep",
    "RunStatus",
    "StepRecord",
    "explain",
]
