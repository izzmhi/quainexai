"""Budgets and action throttling for autonomous runs.

Purpose:
    Bound what an unattended agent can do, in every dimension that can run away.

Why this module exists before the loop that uses it:
    I flagged at the end of Phase 8 that Phase 10 is "a program that issues
    commands on its own, and a bug in it currently hits no ceiling". A budget
    added *after* the loop works is a budget fitted around whatever the loop
    happened to do. Written first, it is a constraint the loop is built inside.

Four independent ceilings, because runaway behaviour takes four shapes:

    * **Steps** — a plan that never concludes.
    * **Wall clock** — a step that hangs rather than loops.
    * **Repeats of one action** — the classic stuck agent, retrying the same
      failing command forever. Per-intent rather than global, because ten
      different actions is progress and ten identical ones is a loop.
    * **Total actions** — the backstop for everything the first three miss.

    Any one being hit stops the run. A budget you can talk your way past is
    decoration, so exhaustion is terminal rather than a warning.

Dependencies:
    Standard library only.

Future improvements:
    * A token budget, so cost is bounded directly rather than by proxy.
    * Exponential backoff between repeats of the same action.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum


class BudgetOutcome(StrEnum):
    """Why a run may not continue."""

    OK = "ok"
    STEPS_EXHAUSTED = "steps_exhausted"
    TIME_EXHAUSTED = "time_exhausted"
    ACTIONS_EXHAUSTED = "actions_exhausted"
    REPEATED_ACTION = "repeated_action"


@dataclass(slots=True)
class AgentBudget:
    """Limits for one autonomous run, and the tally against them.

    Attributes:
        max_steps: Reasoning steps allowed.
        max_seconds: Wall-clock seconds allowed.
        max_actions: Total command executions allowed.
        max_repeats_per_action: How often one intent may be executed before the
            run is treated as stuck.
    """

    max_steps: int = 12
    max_seconds: float = 300.0
    max_actions: int = 20
    max_repeats_per_action: int = 3

    steps_used: int = 0
    actions_used: int = 0
    _started: float = field(default_factory=time.monotonic)
    _per_action: Counter[str] = field(default_factory=Counter)

    @property
    def elapsed_seconds(self) -> float:
        """Seconds since the run began."""
        return time.monotonic() - self._started

    @property
    def steps_remaining(self) -> int:
        """Reasoning steps left."""
        return max(0, self.max_steps - self.steps_used)

    def check(self) -> BudgetOutcome:
        """Test whether another step may begin.

        Returns:
            ``OK``, or the ceiling that has been reached.
        """
        if self.steps_used >= self.max_steps:
            return BudgetOutcome.STEPS_EXHAUSTED
        if self.elapsed_seconds >= self.max_seconds:
            return BudgetOutcome.TIME_EXHAUSTED
        if self.actions_used >= self.max_actions:
            return BudgetOutcome.ACTIONS_EXHAUSTED
        return BudgetOutcome.OK

    def record_step(self) -> None:
        """Count a reasoning step."""
        self.steps_used += 1

    def record_action(self, intent_key: str) -> BudgetOutcome:
        """Count an executed action and test the repeat ceiling.

        Args:
            intent_key: Identifier for the action, typically ``intent:target``.

        Returns:
            ``OK``, or ``REPEATED_ACTION`` when this action has run too often.
        """
        self.actions_used += 1
        self._per_action[intent_key] += 1
        if self._per_action[intent_key] > self.max_repeats_per_action:
            return BudgetOutcome.REPEATED_ACTION
        return BudgetOutcome.OK

    def repeats_of(self, intent_key: str) -> int:
        """How many times an action has run this run.

        Args:
            intent_key: The action identifier.

        Returns:
            The count.
        """
        return self._per_action[intent_key]

    def summary(self) -> dict[str, float | int]:
        """Report consumption, for the run record and the audit log.

        Returns:
            Usage against each ceiling.
        """
        return {
            "steps_used": self.steps_used,
            "max_steps": self.max_steps,
            "actions_used": self.actions_used,
            "max_actions": self.max_actions,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "max_seconds": self.max_seconds,
        }


def explain(outcome: BudgetOutcome) -> str:
    """Render a budget outcome as a sentence for the user.

    Args:
        outcome: The ceiling reached.

    Returns:
        An explanation naming what ran out.
    """
    return {
        BudgetOutcome.OK: "Within budget.",
        BudgetOutcome.STEPS_EXHAUSTED: "Stopped: the step budget was used up.",
        BudgetOutcome.TIME_EXHAUSTED: "Stopped: the time budget was used up.",
        BudgetOutcome.ACTIONS_EXHAUSTED: "Stopped: the action budget was used up.",
        BudgetOutcome.REPEATED_ACTION: (
            "Stopped: the same action was attempted repeatedly, which usually "
            "means the run is stuck rather than making progress."
        ),
    }[outcome]
