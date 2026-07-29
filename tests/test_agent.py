"""Tests for the autonomous agent.

The load-bearing tests are the ones proving an unattended run *cannot* do certain
things: it cannot approve its own confirmations, and it cannot exceed its budget
in any of the four dimensions runaway behaviour takes.
"""

from __future__ import annotations

import time
from pathlib import Path

from quainex.config.settings import Settings
from quainex.core.agent import AgentBudget, AutonomousAgent, BudgetOutcome, RunStatus
from quainex.core.agent.budget import explain
from quainex.core.agent.loop import Plan, PlanStep
from quainex.core.brain import Brain, IntentClassification, IntentType
from quainex.core.commands import build_executor
from quainex.core.exceptions import ProviderError
from quainex.security import ConfirmationService
from tests.test_brain import FakeProvider
from tests.test_commands import FakeDesktopController


class PlanningProvider(FakeProvider):
    """Returns a canned plan, then canned classifications."""

    def __init__(self, plan: Plan, classification: IntentClassification) -> None:
        super().__init__(classification)
        self._plan = plan

    async def parse(self, *, messages, output_model, system=None, max_tokens=None):
        self.calls += 1
        self.last_messages = messages
        self.last_system = system
        # The planner asks for a Plan; the Brain asks for an IntentClassification.
        if output_model is Plan:
            return self._plan
        return await super().parse(
            messages=messages, output_model=output_model, system=system, max_tokens=max_tokens
        )


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    base: dict[str, object] = {
        "_env_file": None,
        "log_dir": tmp_path / "logs",
        "database_path": tmp_path / "t.db",
        "command_search_roots": [tmp_path],
        "screenshot_dir": tmp_path / "shots",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _agent(
    tmp_path: Path,
    plan: Plan,
    intent: IntentType = IntentType.OPEN_APPLICATION,
    desktop: FakeDesktopController | None = None,
    **setting_overrides: object,
) -> AutonomousAgent:
    settings = _settings(tmp_path, **setting_overrides)
    provider = PlanningProvider(
        plan,
        IntentClassification(intent=intent, target="VS Code", confidence=0.97, reasoning="test"),
    )
    return AutonomousAgent(
        provider=provider,
        brain=Brain(provider=provider, settings=settings),
        commands=build_executor(
            desktop or FakeDesktopController(), settings, ConfirmationService("k" * 48)
        ),
        settings=settings,
    )


# -- budgets ---------------------------------------------------------------


def test_budget_starts_ok():
    assert AgentBudget().check() is BudgetOutcome.OK


def test_step_ceiling():
    budget = AgentBudget(max_steps=2)
    budget.record_step()
    budget.record_step()
    assert budget.check() is BudgetOutcome.STEPS_EXHAUSTED


def test_action_ceiling():
    budget = AgentBudget(max_actions=2)
    budget.record_action("a")
    budget.record_action("b")
    assert budget.check() is BudgetOutcome.ACTIONS_EXHAUSTED


def test_time_ceiling():
    budget = AgentBudget(max_seconds=0.01)
    time.sleep(0.02)
    assert budget.check() is BudgetOutcome.TIME_EXHAUSTED


def test_repeating_one_action_is_detected_but_variety_is_not():
    # Ten different actions is progress; ten identical ones is a stuck loop.
    budget = AgentBudget(max_repeats_per_action=2, max_actions=100)

    assert budget.record_action("open:code") is BudgetOutcome.OK
    assert budget.record_action("open:code") is BudgetOutcome.OK
    assert budget.record_action("open:code") is BudgetOutcome.REPEATED_ACTION

    fresh = AgentBudget(max_repeats_per_action=2, max_actions=100)
    for index in range(10):
        assert fresh.record_action(f"open:app{index}") is BudgetOutcome.OK


def test_every_outcome_has_an_explanation():
    for outcome in BudgetOutcome:
        assert explain(outcome).strip()


def test_budget_summary_reports_both_use_and_limit():
    budget = AgentBudget(max_steps=5)
    budget.record_step()
    summary = budget.summary()

    assert summary["steps_used"] == 1
    assert summary["max_steps"] == 5


# -- the safety property ---------------------------------------------------


async def test_an_unattended_run_cannot_shut_the_machine_down(tmp_path):
    # The agent has no way to satisfy the confirmation gate: it passes neither
    # `confirmed` nor a token. This is the central claim of Phase 10.
    desktop = FakeDesktopController()
    agent = _agent(
        tmp_path,
        Plan(reasoning="do it", steps=[PlanStep(instruction="shut down the pc")]),
        intent=IntentType.SHUTDOWN,
        desktop=desktop,
        allow_destructive_commands=True,
    )

    run = await agent.run("shut down the computer")

    assert run.status is RunStatus.AWAITING_CONFIRMATION
    assert run.pending_confirmation is not None
    assert run.pending_confirmation.executed is False
    assert desktop.calls == [], "nothing may happen without a human decision"


async def test_a_pause_surfaces_the_question_and_a_token(tmp_path):
    agent = _agent(
        tmp_path,
        Plan(reasoning="x", steps=[PlanStep(instruction="close spotify")]),
        intent=IntentType.CLOSE_APPLICATION,
    )
    run = await agent.run("close spotify")

    assert run.status is RunStatus.AWAITING_CONFIRMATION
    assert run.pending_confirmation is not None
    assert run.pending_confirmation.confirmation_token, "the user needs a way to say yes"
    assert "Confirm" in run.summary


async def test_the_run_stops_at_the_pause_rather_than_continuing(tmp_path):
    desktop = FakeDesktopController()
    agent = _agent(
        tmp_path,
        Plan(
            reasoning="x",
            steps=[
                PlanStep(instruction="close spotify"),
                PlanStep(instruction="open vs code"),
            ],
        ),
        intent=IntentType.CLOSE_APPLICATION,
        desktop=desktop,
    )

    run = await agent.run("do two things")

    assert run.status is RunStatus.AWAITING_CONFIRMATION
    assert len(run.steps) == 1, "later steps must not run past an unanswered question"


# -- normal execution ------------------------------------------------------


async def test_a_simple_goal_completes(tmp_path):
    desktop = FakeDesktopController()
    agent = _agent(
        tmp_path,
        Plan(reasoning="open it", steps=[PlanStep(instruction="open vs code")]),
        desktop=desktop,
    )

    run = await agent.run("open my editor")

    assert run.status is RunStatus.COMPLETED
    assert desktop.actions == ["open_application"]
    assert len(run.steps) == 1
    assert run.steps[0].result is not None
    assert run.steps[0].result.executed is True


async def test_multiple_steps_all_run(tmp_path):
    desktop = FakeDesktopController()
    agent = _agent(
        tmp_path,
        Plan(
            reasoning="x",
            steps=[PlanStep(instruction=f"step {n}") for n in range(3)],
        ),
        desktop=desktop,
    )

    run = await agent.run("do three things")

    assert run.status is RunStatus.COMPLETED
    assert len(run.steps) == 3
    assert len(desktop.calls) == 3


async def test_an_empty_goal_is_not_actionable(tmp_path):
    run = await _agent(tmp_path, Plan(reasoning="", steps=[])).run("   ")
    assert run.status is RunStatus.NOT_ACTIONABLE


async def test_an_unplannable_goal_reports_the_reason(tmp_path):
    agent = _agent(tmp_path, Plan(reasoning="This cannot be done on a computer.", steps=[]))
    run = await agent.run("make me a sandwich")

    assert run.status is RunStatus.NOT_ACTIONABLE
    assert "cannot be done" in run.summary


# -- budgets bind the real loop -------------------------------------------


async def test_the_step_ceiling_stops_a_long_plan(tmp_path):
    desktop = FakeDesktopController()
    agent = _agent(
        tmp_path,
        Plan(reasoning="x", steps=[PlanStep(instruction=f"step {n}") for n in range(10)]),
        desktop=desktop,
    )

    run = await agent.run("do ten things", AgentBudget(max_steps=3, max_actions=100))

    assert run.status is RunStatus.BUDGET_EXHAUSTED
    assert "step budget" in run.summary
    assert len(run.steps) == 3


async def test_the_action_ceiling_stops_a_long_plan(tmp_path):
    agent = _agent(
        tmp_path,
        Plan(reasoning="x", steps=[PlanStep(instruction=f"step {n}") for n in range(10)]),
    )
    run = await agent.run("do ten things", AgentBudget(max_steps=100, max_actions=2))

    assert run.status is RunStatus.BUDGET_EXHAUSTED
    assert "action budget" in run.summary


async def test_a_stuck_run_is_detected(tmp_path):
    # Every step classifies to the same intent and target: the definition of a
    # loop that is not making progress.
    desktop = FakeDesktopController()
    agent = _agent(
        tmp_path,
        Plan(reasoning="x", steps=[PlanStep(instruction="open vs code") for _ in range(6)]),
        desktop=desktop,
    )

    run = await agent.run("keep opening it", AgentBudget(max_repeats_per_action=2))

    assert run.status is RunStatus.FAILED
    assert "repeatedly" in run.summary
    assert len(desktop.calls) <= 3, "a stuck run must not keep acting"


async def test_the_budget_is_reported_whatever_the_outcome(tmp_path):
    run = await _agent(
        tmp_path, Plan(reasoning="x", steps=[PlanStep(instruction="open vs code")])
    ).run("open it")

    assert run.budget["steps_used"] == 1
    assert run.budget["max_steps"] > 0
    assert "elapsed_seconds" in run.budget


# -- failure handling ------------------------------------------------------


async def test_a_classification_failure_ends_the_run_cleanly(tmp_path):
    settings = _settings(tmp_path)
    provider = PlanningProvider(
        Plan(reasoning="x", steps=[PlanStep(instruction="do a thing")]),
        IntentClassification(
            intent=IntentType.OPEN_APPLICATION, target="x", confidence=0.9, reasoning="t"
        ),
    )
    agent = AutonomousAgent(
        provider=provider,
        brain=Brain(provider=FakeProvider(error=ProviderError("upstream down")), settings=settings),
        commands=build_executor(FakeDesktopController(), settings),
        settings=settings,
    )

    run = await agent.run("do a thing")

    assert run.status is RunStatus.FAILED
    assert "upstream down" in run.summary
