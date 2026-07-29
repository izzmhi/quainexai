"""The autonomous agent loop.

Purpose:
    Take a goal, break it into steps, carry them out, and stop — safely, and
    provably.

The safety property this whole phase rests on:
    **The agent cannot confirm on the user's behalf.** It calls the same
    ``CommandExecutor`` as everything else, and it never passes ``confirmed=True``
    or a confirmation token. So an action requiring approval comes back
    ``REQUIRES_CONFIRMATION`` with ``executed=False``, and the run *pauses* and
    surfaces the question.

    That is not a policy the loop chooses to follow — it is the consequence of
    the loop having no way to satisfy the gate. The Phase 6 design, where
    confirmation is a signed token minted only by a refusal, is what makes this
    true rather than aspirational. Two tests assert an unattended run cannot
    shut the machine down.

Architecture:
    goal -> plan (model, structured)      -> list[PlanStep]
         -> for each step, within budget:
              Brain.interpret(step)       -> Intent
              budget.record_action()      -> stuck? stop
              CommandExecutor.execute()   -> no confirmation available
                 |-- requires_confirmation -> PAUSE, surface, stop
                 +-- executed / refused    -> record, continue
         -> AgentRun (every step, the budget spent, why it stopped)

Dependencies:
    quainex.core.{agent,brain,commands,memory}

Future improvements:
    * Replan when a step fails, rather than continuing to the next one.
    * Resume a paused run once the user answers, instead of starting over.
    * Learn from completed runs, so a repeated goal skips the planning call.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from quainex.core.agent.budget import AgentBudget, BudgetOutcome, explain
from quainex.core.commands import CommandResult, CommandStatus
from quainex.core.logging import get_logger

if TYPE_CHECKING:
    from quainex.config.settings import Settings
    from quainex.core.brain import Brain, Intent
    from quainex.core.commands import CommandExecutor
    from quainex.core.memory import MemoryManager
    from quainex.services.ai.provider import AIProvider

_log = get_logger(__name__)

_PLANNER_SYSTEM = """
You are planning how a desktop assistant should carry out a goal on the user's
computer.

Break the goal into the smallest number of concrete steps that achieve it. Each
step must be a single instruction the assistant can act on directly, phrased the
way a person would say it — "open VS Code", "run the tests", "take a screenshot".

Rules:
- Prefer fewer steps. A goal needing one action is a one-step plan.
- Do not include steps for thinking, checking or deciding; only actions.
- Do not invent steps the goal did not ask for.
- If the goal cannot be achieved by controlling a computer, return no steps and
  say why in the reasoning.
""".strip()


class RunStatus(StrEnum):
    """How an autonomous run ended."""

    COMPLETED = "completed"
    #: Stopped awaiting a human decision. Not a failure — the design working.
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    BUDGET_EXHAUSTED = "budget_exhausted"
    FAILED = "failed"
    #: No plan could be produced for the goal.
    NOT_ACTIONABLE = "not_actionable"


class PlanStep(BaseModel):
    """One step of a plan.

    Attributes:
        instruction: The action, phrased as the user would say it.
    """

    instruction: str


class Plan(BaseModel):
    """A decomposed goal.

    Attributes:
        reasoning: One sentence on the approach, or why the goal is not actionable.
        steps: The steps to carry out, in order.
    """

    reasoning: str
    steps: list[PlanStep] = Field(default_factory=list)


class StepRecord(BaseModel):
    """What happened on one step.

    Attributes:
        instruction: What was attempted.
        intent: The classified intent, when classification succeeded.
        result: The command outcome, when one was reached.
        error: Why the step could not be attempted, when that happened.
    """

    instruction: str
    intent: str | None = None
    result: CommandResult | None = None
    error: str | None = None


class AgentRun(BaseModel):
    """The complete record of one autonomous run.

    Attributes:
        goal: What was asked for.
        status: How it ended.
        summary: One-line explanation of the outcome.
        plan: The plan that was produced.
        steps: What happened, step by step.
        pending_confirmation: The action awaiting approval, when paused.
        budget: Consumption against each ceiling.
    """

    goal: str
    status: RunStatus
    summary: str
    plan: Plan | None = None
    steps: list[StepRecord] = Field(default_factory=list)
    pending_confirmation: CommandResult | None = None
    budget: dict[str, float | int] = Field(default_factory=dict)


class AutonomousAgent:
    """Plans and carries out goals, within a budget, without self-approval."""

    def __init__(
        self,
        *,
        provider: AIProvider,
        brain: Brain,
        commands: CommandExecutor,
        settings: Settings,
        memory: MemoryManager | None = None,
    ) -> None:
        """Construct the agent.

        Args:
            provider: Model backend used for planning.
            brain: Classifier turning each step into an intent.
            commands: Executor carrying out intents.
            settings: Configuration supplying default budgets.
            memory: Optional memory, for recording what was done.
        """
        self._provider = provider
        self._brain = brain
        self._commands = commands
        self._settings = settings
        self._memory = memory

    async def run(self, goal: str, budget: AgentBudget | None = None) -> AgentRun:
        """Plan a goal and carry it out.

        Args:
            goal: What to achieve.
            budget: Limits for this run; defaults to the configured ones.

        Returns:
            The complete record of what happened.
        """
        limits = budget or AgentBudget(
            max_steps=self._settings.agent_max_steps,
            max_seconds=self._settings.agent_max_seconds,
            max_actions=self._settings.agent_max_actions,
            max_repeats_per_action=self._settings.agent_max_repeats,
        )

        cleaned = goal.strip()
        if not cleaned:
            return AgentRun(
                goal=goal,
                status=RunStatus.NOT_ACTIONABLE,
                summary="No goal was given.",
                budget=limits.summary(),
            )

        plan = await self._plan(cleaned)
        if not plan.steps:
            _log.info("agent_goal_not_actionable", goal_chars=len(cleaned))
            return AgentRun(
                goal=cleaned,
                status=RunStatus.NOT_ACTIONABLE,
                summary=plan.reasoning or "No actions could be planned for this goal.",
                plan=plan,
                budget=limits.summary(),
            )

        _log.info("agent_run_started", steps_planned=len(plan.steps), goal_chars=len(cleaned))
        return await self._execute_plan(cleaned, plan, limits)

    # -- internals --------------------------------------------------------

    async def _plan(self, goal: str) -> Plan:
        """Decompose a goal into steps.

        Args:
            goal: What to achieve.

        Returns:
            The plan. An empty step list means the goal is not actionable.
        """
        from quainex.services.ai.provider import ChatMessage

        return await self._provider.parse(
            messages=[ChatMessage(role="user", content=goal)],
            output_model=Plan,
            system=_PLANNER_SYSTEM,
        )

    async def _execute_plan(self, goal: str, plan: Plan, budget: AgentBudget) -> AgentRun:
        """Carry out a plan step by step, within budget.

        Args:
            goal: The original goal.
            plan: The steps to run.
            budget: Limits for this run.

        Returns:
            The complete run record.
        """
        records: list[StepRecord] = []

        for step in plan.steps:
            outcome = budget.check()
            if outcome is not BudgetOutcome.OK:
                return self._finish(
                    goal, plan, records, RunStatus.BUDGET_EXHAUSTED, explain(outcome), budget
                )

            budget.record_step()
            record, pending = await self._run_step(step, budget)
            records.append(record)

            if pending is not None:
                # The design working, not a failure: the agent has no way to
                # approve this, so the human is asked.
                _log.info(
                    "agent_paused_for_confirmation",
                    intent=pending.intent,
                    step=step.instruction,
                )
                return self._finish(
                    goal,
                    plan,
                    records,
                    RunStatus.AWAITING_CONFIRMATION,
                    f"Paused: {pending.message}",
                    budget,
                    pending=pending,
                )

            if record.error is not None:
                return self._finish(goal, plan, records, RunStatus.FAILED, record.error, budget)

            if record.result is not None and record.result.status is CommandStatus.BLOCKED:
                return self._finish(
                    goal, plan, records, RunStatus.FAILED, record.result.message, budget
                )

        executed = sum(1 for r in records if r.result and r.result.executed)
        return self._finish(
            goal,
            plan,
            records,
            RunStatus.COMPLETED,
            f"Completed {executed} of {len(plan.steps)} planned action(s).",
            budget,
        )

    async def _run_step(
        self, step: PlanStep, budget: AgentBudget
    ) -> tuple[StepRecord, CommandResult | None]:
        """Classify and execute one step.

        Args:
            step: The step to run.
            budget: Limits, updated as actions are taken.

        Returns:
            The step record, and the pending result when confirmation is needed.
        """
        from quainex.core.exceptions import QuainexError

        try:
            intent: Intent = await self._brain.interpret(step.instruction)
        except QuainexError as exc:
            return StepRecord(instruction=step.instruction, error=exc.message), None

        key = f"{intent.intent.value}:{(intent.target or '').strip().lower()}"
        if budget.record_action(key) is BudgetOutcome.REPEATED_ACTION:
            return (
                StepRecord(
                    instruction=step.instruction,
                    intent=intent.intent.value,
                    error=explain(BudgetOutcome.REPEATED_ACTION),
                ),
                None,
            )

        # Note what is *not* passed: no `confirmed`, no `confirmation_token`.
        # The agent has no means to satisfy the confirmation gate.
        result = await self._commands.execute(intent)

        if self._memory is not None:
            await self._memory.remember_exchange(step.instruction, intent, result)

        record = StepRecord(instruction=step.instruction, intent=intent.intent.value, result=result)
        if result.status is CommandStatus.REQUIRES_CONFIRMATION:
            return record, result
        return record, None

    @staticmethod
    def _finish(
        goal: str,
        plan: Plan,
        records: list[StepRecord],
        status: RunStatus,
        summary: str,
        budget: AgentBudget,
        pending: CommandResult | None = None,
    ) -> AgentRun:
        """Assemble the run record.

        Args:
            goal: The original goal.
            plan: The plan produced.
            records: What happened per step.
            status: How the run ended.
            summary: One-line explanation.
            budget: Consumption so far.
            pending: The action awaiting approval, when paused.

        Returns:
            The run record.
        """
        _log.info(
            "agent_run_finished",
            status=status.value,
            steps_run=len(records),
            **budget.summary(),
        )
        return AgentRun(
            goal=goal,
            status=status,
            summary=summary,
            plan=plan,
            steps=records,
            pending_confirmation=pending,
            budget=budget.summary(),
        )
