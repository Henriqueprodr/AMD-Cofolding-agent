"""Tests for reinvent4_agent/supervisor.py -- the headless worker loop.

The campaign run is injected into ``Supervisor``, so everything around it --
leases, failures, deadlines, shutdown, and reading the achieved profile out of
a results CSV -- is testable with no LLM, no MCP server and no GPU.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from reinvent4_agent.supervisor import (
    DRIVER_DENIED_TOOLS,
    STEP_SECONDS_HINT,
    CampaignStoreClient,
    LeaseLost,
    StoreUnavailable,
    Supervisor,
    TaskOutcome,
    _deadline_seconds,
    _overdue_seconds,
    apply_overrides,
    build_goal,
    job_for_ttp,
    objective_profile,
    prepare_ttp_variant,
)

BASE_TTP = {
    "id": "brd4-bd1-lead-opt",
    "name": "BRD4 BD1 lead optimisation",
    "version": 3,
    "pre_scoring": {
        "components": [
            {"name": "QED", "type": "native", "component": "QED", "weight": 0.5},
        ]
    },
    "scoring": {
        "type": "geometric_mean",
        "objectives": [
            {"name": "BoltzAffinity", "type": "external", "service": "boltz",
             "weight": 1.0},
            {"name": "logD", "type": "external", "service": "admet", "weight": 0.7},
        ],
    },
}


def _task(**spec: Any) -> dict[str, Any]:
    return {
        "task_id": "a1b2c3d4",
        "campaign_id": "feed0001",
        "state": "claimed",
        "spec": {"weight_overrides": {}, "fidelity_tier": "exploration", **spec},
    }


class FakeStore:
    """A campaign store that hands out a fixed list of tasks."""

    def __init__(self, tasks: list[dict[str, Any]] | None = None) -> None:
        self.base_url = "http://store"
        self.owner = "supervisor-0"
        self.queue = list(tasks or [])
        self.results: list[tuple[str, TaskOutcome]] = []
        self.failures: list[tuple[str, str, bool]] = []
        self.heartbeats: list[str] = []
        self.reported_jobs: list[tuple[str, str]] = []
        self.presence: list[tuple[str, Any]] = []
        self.left_the_pool = False
        self.claims = 0

    def claim(self, campaign_id=None, lease_seconds=None):
        self.claims += 1
        return self.queue.pop(0) if self.queue else None

    def report_presence(self, state, campaign_id=None, task_id=None,
                        tasks_completed=0, tasks_failed=0, detail=None):
        self.presence.append((state, task_id))
        return {"owner": self.owner, "state": state}

    def drop_presence(self):
        self.left_the_pool = True

    def heartbeat(self, task_id, lease_seconds=None, job_id=None):
        self.heartbeats.append(task_id)
        if job_id:
            self.reported_jobs.append((task_id, job_id))

    def post_result(self, task_id, outcome):
        self.results.append((task_id, outcome))
        return {"task_id": task_id}

    def fail(self, task_id, error, requeue=True):
        self.failures.append((task_id, error, requeue))
        return {"task_id": task_id, "state": "queued" if requeue else "failed"}


async def _succeeds(task: dict[str, Any]) -> TaskOutcome:
    return TaskOutcome(
        objective_profile={"BoltzAffinity": 0.8, "QED": 0.7},
        candidates=[{"smiles": "CCO"}],
        job_id="job1",
    )


# ---------------------------------------------------------------------------
# Weight overrides
# ---------------------------------------------------------------------------


class TestApplyOverrides:

    def test_overrides_a_scoring_objective(self):
        updated = apply_overrides(BASE_TTP, {"BoltzAffinity": 2.5})
        weights = {
            o["name"]: o["weight"] for o in updated["scoring"]["objectives"]
        }

        assert weights["BoltzAffinity"] == 2.5
        assert weights["logD"] == 0.7

    def test_overrides_a_pre_scoring_component(self):
        """A sweep may well want to lean on drug-likeness, not just affinity."""
        updated = apply_overrides(BASE_TTP, {"QED": 1.5})

        assert updated["pre_scoring"]["components"][0]["weight"] == 1.5

    def test_overrides_a_transform(self):
        updated = apply_overrides(
            BASE_TTP, transform_overrides={"logD": {"type": "sigmoid", "k": 0.4}}
        )
        logd = next(
            o for o in updated["scoring"]["objectives"] if o["name"] == "logD"
        )

        assert logd["transform"] == {"type": "sigmoid", "k": 0.4}

    def test_the_base_ttp_is_not_mutated(self):
        """N supervisors share the TTP store; the base must survive untouched."""
        apply_overrides(BASE_TTP, {"BoltzAffinity": 9.0})

        original = next(
            o for o in BASE_TTP["scoring"]["objectives"]
            if o["name"] == "BoltzAffinity"
        )
        assert original["weight"] == 1.0

    def test_an_unknown_objective_is_an_error(self):
        """A typo must not become a no-op.

        Silently ignoring it would have the task report a profile for a weight
        vector it never ran, and the conductor would draw conclusions from a
        point that was never tried.
        """
        with pytest.raises(ValueError) as exc:
            apply_overrides(BASE_TTP, {"BoltzAfinity": 2.0})

        assert "BoltzAfinity" in str(exc.value)
        assert "BoltzAffinity" in str(exc.value), "should list what is available"

    def test_no_overrides_is_a_faithful_copy(self):
        assert apply_overrides(BASE_TTP) == BASE_TTP

    def test_a_ttp_without_pre_scoring_is_fine(self):
        bare = {"id": "x", "name": "x", "scoring": {"objectives": [
            {"name": "BoltzAffinity", "type": "external", "weight": 1.0},
        ]}}

        assert apply_overrides(bare, {"BoltzAffinity": 2.0})["scoring"][
            "objectives"
        ][0]["weight"] == 2.0


# ---------------------------------------------------------------------------
# Objective profile
# ---------------------------------------------------------------------------


class TestObjectiveProfile:

    def test_the_profile_is_one_molecules_scores(self):
        """Not the best value per objective, which would mix molecules.

        A per-objective maximum describes a compound nobody made.
        """
        candidates = [
            {
                "smiles": "best", "total_score": 0.9,
                "components": {
                    "BoltzAffinity": {"transformed": 0.85},
                    "QED": {"transformed": 0.60},
                },
            },
            {
                "smiles": "spikier", "total_score": 0.4,
                "components": {
                    "BoltzAffinity": {"transformed": 0.10},
                    "QED": {"transformed": 0.99},
                },
            },
        ]

        assert objective_profile(candidates) == {
            "BoltzAffinity": 0.85, "QED": 0.60,
        }

    def test_objectives_can_be_restricted_to_the_campaigns(self, ):
        candidates = [{
            "total_score": 0.9,
            "components": {
                "BoltzAffinity": {"transformed": 0.8},
                "MolecularWeight": {"transformed": 0.5},
            },
        }]

        assert objective_profile(candidates, ["BoltzAffinity"]) == {
            "BoltzAffinity": 0.8
        }

    def test_no_candidates_is_an_empty_profile(self):
        assert objective_profile([]) == {}

    def test_candidates_without_a_total_score_are_skipped(self):
        assert objective_profile([{"smiles": "CCO"}]) == {}

    def test_a_nan_component_is_left_out(self):
        """A failed scorer must not become a number in the profile."""
        candidates = [{
            "total_score": 0.5,
            "components": {
                "BoltzAffinity": {"transformed": None},
                "QED": {"transformed": 0.7},
            },
        }]

        assert objective_profile(candidates) == {"QED": 0.7}


# ---------------------------------------------------------------------------
# The goal handed to the Driver
# ---------------------------------------------------------------------------


class TestBuildGoal:

    def test_the_goal_names_the_prepared_ttp(self):
        """One source of truth for the weight vector: the TTP, not prose."""
        goal = build_goal(_task(weight_overrides={"QED": 2.0}), "brd4-a1b2c3d4")

        assert "brd4-a1b2c3d4" in goal
        assert "do not modify" in goal
        assert "2.0" not in goal, "restating the weights invites paraphrasing"

    def test_a_task_with_no_base_ttp_is_told_to_write_one(self):
        """The usual case. The agent is given the id, not asked to invent it:
        that id is how its run is found afterwards."""

        goal = build_goal(_task(), "ttp-t1", authored=True)

        assert "ttp-t1" in goal
        assert "create_ttp" in goal, "naming the tool is the whole instruction"
        assert "do not modify" not in goal, "there is nothing yet to modify"

    def test_it_names_the_tools_that_would_skip_writing_the_ttp(self):
        """Observed on LUMI: told only to "write one", a sub-agent called
        create_generation_campaign thirty-six times and run_optimization ten,
        never once create_ttp, and failed the task. Both of those start a run
        with no TTP behind it, so they look like progress and are not."""

        goal = build_goal(_task(), "ttp-t1", authored=True)

        assert "run_optimization" in goal
        assert "create_generation_campaign" in goal

    def test_it_names_the_starter_a_scaffold_or_linker_run_needs(self):
        """run_optimization cannot build either one and refuses them.

        Told only to "start the run", a Driver reaches for run_optimization,
        which builds de novo and Mol2Mol TOML only -- so a task asking for a
        scaffold or linker run would stall on a refusal it has no answer to.
        """

        goal = build_goal(_task(), "ttp-t1", authored=True)

        assert "run_generation_campaign" in goal
        assert "libinvent" in goal
        assert "linkinvent" in goal

    def test_a_generation_campaign_is_a_way_of_writing_the_ttp(self):
        """It persists a TTP under the id it is given, so forbidding it
        outright withheld the only tool that writes a scaffold profile."""

        goal = build_goal(_task(), "ttp-t1", authored=True)

        assert "campaign_id='ttp-t1'" in goal

    def test_an_authored_task_is_given_the_campaign_goal(self):
        """It has to choose objectives, and cannot do that from the task
        alone -- the task says what this experiment is for, the campaign
        says what the experiment is part of."""

        goal = build_goal(
            _task(), "ttp-t1",
            campaign_goal="Improve BRD4 BD1 selectivity over BD2",
            authored=True,
        )

        assert "BRD4 BD1 selectivity" in goal

    def test_a_task_with_a_base_ttp_is_not_told_to_write_one(self):
        goal = build_goal(
            _task(), "brd4-a1b2c3d4",
            campaign_goal="Improve BRD4 BD1 selectivity", authored=False,
        )

        assert "do not modify" in goal
        assert "no therapeutic target profile" not in goal

    def test_the_step_budget_is_passed_on_as_a_ceiling(self):
        goal = build_goal(_task(max_steps=120), "ttp-1")

        assert "120 RL steps" in goal
        assert "do not raise it" in goal

    def test_a_validation_task_is_told_not_to_use_the_cache(self):
        goal = build_goal(_task(fidelity_tier="validation"), "ttp-1")

        assert "validation-tier" in goal

    def test_an_exploration_task_is_not(self):
        assert "validation-tier" not in build_goal(_task(), "ttp-1")

    def test_the_tasks_intent_reaches_the_agent_verbatim(self):
        """The channel for what the structured fields cannot say.

        Holding a chemotype is a choice of generative prior, and the agent can
        look the reference compound up and pick a model. It ran de novo not
        because it could not do better but because nothing asked it to.
        """
        intent = "Hold the JQ1 methyl-triazolo-benzodiazepine core fixed."

        goal = build_goal(_task(intent=intent), "ttp-1")

        assert intent in goal

    def test_an_intent_invites_the_agent_to_choose_the_prior(self):
        goal = build_goal(_task(intent="stay on the I-BET151 chemotype"), "ttp-1")

        assert "choose the model" in goal
        assert "look the reference" in goal
        assert "why before starting" in goal

    def test_what_the_campaign_knows_reaches_the_driver_writing_a_profile(self):
        """What the campaign learnt about the target is what decides which
        objectives are worth scoring."""

        goal = build_goal(
            _task(workstream="design"), "ttp-1", authored=True,
            findings=[{
                "finding_id": "f1", "confidence": "high",
                "claim": "hERG is the series liability above 400 Da",
                "evidence": ["job-7"],
            }],
        )

        assert "hERG is the series liability above 400 Da" in goal
        assert "job-7" in goal
        assert "atlas_finding" in goal
        assert "'design'" in goal

    def test_a_task_without_an_intent_says_nothing_about_priors(self):
        """Silence keeps the default de novo behaviour for ordinary sweeps."""
        goal = build_goal(_task(), "ttp-1")

        assert "What this run is for" not in goal
        assert "de novo" not in goal


class TestTheDeNovoFallback:
    """When starting the run ourselves is still the task's own experiment.

    The supervisor starts the run when its agent finishes without doing so,
    using only the TTP and step count -- so from the default de novo prior.
    """

    def test_an_ordinary_task_can_be_started_for_the_agent(self):
        from reinvent4_agent.supervisor import default_run_substitutes_for

        assert default_run_substitutes_for({"weight_overrides": {"QED": 2.0}})

    def test_a_task_with_an_intent_cannot(self):
        """A substituted run reports a result for an experiment nobody asked for."""
        from reinvent4_agent.supervisor import default_run_substitutes_for

        assert not default_run_substitutes_for({"intent": "hold the JQ1 core"})

    def test_an_absent_spec_is_treated_as_ordinary(self):
        from reinvent4_agent.supervisor import default_run_substitutes_for

        assert default_run_substitutes_for({})
        assert default_run_substitutes_for(None)

    def test_nothing_substitutes_for_a_task_that_writes_its_own_ttp(self):
        """The substitute run would name a TTP that does not exist, because
        writing it was exactly the step the agent skipped."""

        from reinvent4_agent.supervisor import default_run_substitutes_for

        assert not default_run_substitutes_for({}, authored=True)
        assert not default_run_substitutes_for(None, authored=True)

    def test_the_driver_is_told_how_long_a_step_takes(self):
        """Without a number it cannot tell a slow step from a broken one.

        The first live run had the Driver poll thirteen times in six minutes,
        decide the job had stalled, and stop it inside its first step -- where
        REINVENT has nothing to checkpoint and no candidates exist yet.
        """
        goal = build_goal(_task(), "ttp-1")

        assert STEP_SECONDS_HINT in goal
        assert "is normal" in goal
        assert "Do not try to stop" in goal


class TestDriverToolBoundary:
    """A supervisor's Driver runs the task it was given, and cannot redefine it."""

    def test_stopping_the_run_is_not_the_drivers_to_do(self):
        # The supervisor owns the deadline and stops with a checkpoint. The
        # Driver stopping on impatience loses the work.
        assert "stop_optimization" in DRIVER_DENIED_TOOLS

    def test_it_cannot_edit_or_replace_a_ttp(self):
        # The whole point of the variant TTP is that the weight vector is
        # fixed in advance; a Driver that can write TTPs can quietly run
        # something other than the point the campaign is measuring.
        assert {"create_ttp", "update_ttp", "delete_ttp"} <= DRIVER_DENIED_TOOLS

    def test_a_driver_writing_its_own_ttp_may_create_one(self):
        """Otherwise the task is impossible.

        Observed on LUMI: told to write a profile and start a run against it,
        two sub-agents in a row called create_generation_campaign and
        run_optimization in a loop instead -- the only TTP-less tools left to
        them -- and both tasks failed. The denial is right for a sweep, where
        the weights are the experiment; it cannot apply when writing the
        profile is the experiment.
        """

        from reinvent4_agent.supervisor import driver_denied_tools

        assert "create_ttp" not in driver_denied_tools(authored=True)

    def test_it_still_cannot_rewrite_or_remove_one(self):
        """Writing a profile is the task. Rewriting one that already has a
        run behind it, or deleting someone else's, is not."""

        from reinvent4_agent.supervisor import driver_denied_tools

        denied = driver_denied_tools(authored=True)
        assert {"update_ttp", "delete_ttp"} <= denied
        assert "stop_optimization" in denied

    def test_a_driver_varying_a_base_ttp_still_cannot_write_one(self):
        from reinvent4_agent.supervisor import driver_denied_tools

        assert "create_ttp" in driver_denied_tools(authored=False)
        assert driver_denied_tools(authored=False) == DRIVER_DENIED_TOOLS

    def test_it_cannot_steer_the_fleet(self):
        """Asserted against the fleet set, not a list written out by hand.

        Written out by hand it held enqueue_task and not its plural twin
        enqueue_tasks, so a Driver could queue work for the whole campaign --
        and rewrite the plan, and claim proposals -- while the denial's own
        reason said only the conducting agent divides the work. Phrased this
        way a thirteenth fleet tool cannot slip past.
        """
        from reinvent4_agent.supervisor import driver_denied_tools
        from reinvent4_agent.web.sessions import FLEET_TOOLS

        assert FLEET_TOOLS <= DRIVER_DENIED_TOOLS
        for authored in (True, False):
            for kind in ("design", "research"):
                assert FLEET_TOOLS <= driver_denied_tools(authored, kind)

    def test_every_withheld_tool_says_what_to_do_instead(self):
        """A refusal with no reason is retried, or the work is abandoned.

        That is what DENIAL_REASONS is for, so a tool added to either denial
        set without one quietly loses the sub-agent its way round.
        """
        from reinvent4_agent.supervisor import DENIAL_REASONS, GENERAL_DENIED_TOOLS

        for tool in sorted(DRIVER_DENIED_TOOLS | GENERAL_DENIED_TOOLS):
            assert DENIAL_REASONS.get(tool), f"{tool} is withheld with no reason"

    def test_reading_is_still_allowed(self):
        # The denial is about mutation, not curiosity -- it still has to be
        # able to watch its own job.
        for tool in (
            "run_optimization", "get_optimization_status", "get_top_candidates",
            "get_ttp", "get_campaign_state", "get_supervisor_result",
        ):
            assert tool not in DRIVER_DENIED_TOOLS

    def test_the_allow_list_is_the_live_tools_minus_the_denied(self):
        """Computed, not hardcoded, so a tool added later is allowed by default."""
        live = {
            "run_optimization", "get_optimization_status", "stop_optimization",
            "update_ttp", "some_future_tool",
        }

        allowed = live - DRIVER_DENIED_TOOLS

        assert allowed == {
            "run_optimization", "get_optimization_status", "some_future_tool",
        }


class TestSubAgentToolBoundary:
    """A task that is not a design run must not start one.

    The supervisor's deadline, monitoring and harvest are built around a
    single RL job. A research task that starts one has nobody watching it and
    nobody stopping it at the deadline, so it would hold a GPU until the
    runner was restarted.
    """

    def test_it_cannot_start_a_run(self):
        from reinvent4_agent.supervisor import driver_denied_tools

        denied = driver_denied_tools(authored=False, kind="research")

        assert {
            "run_optimization", "run_transfer_learning",
            "run_generation_campaign", "create_generation_campaign",
        } <= denied

    def test_it_can_still_write_a_profile(self):
        """Preparing the TTP a later design task runs is a reasonable job,
        and create_ttp cannot overwrite one that already exists."""
        from reinvent4_agent.supervisor import driver_denied_tools

        assert "create_ttp" not in driver_denied_tools(authored=False, kind="research")

    def test_reading_and_scoring_are_untouched(self):
        from reinvent4_agent.supervisor import driver_denied_tools

        denied = driver_denied_tools(authored=False, kind="retrosynthesis")
        for tool in (
            "score_molecules", "predict_admet", "plan_synthesis",
            "fetch_target_info", "get_top_candidates", "list_jobs",
        ):
            assert tool not in denied

    def test_a_design_task_keeps_its_run(self):
        from reinvent4_agent.supervisor import driver_denied_tools

        assert "run_optimization" not in driver_denied_tools(authored=False, kind="design")

    def test_no_sub_agent_can_steer_the_fleet(self):
        from reinvent4_agent.supervisor import driver_denied_tools

        for kind in ("design", "research", "question"):
            assert {"conduct_campaign", "enqueue_task", "start_campaign"} <= (
                driver_denied_tools(authored=False, kind=kind)
            )

    def test_every_refusal_says_what_to_do_instead(self):
        """A bare "not available" gets the same call retried, or the work dropped."""
        from reinvent4_agent.supervisor import denial_reasons, driver_denied_tools

        for kind in ("design", "research"):
            for authored in (True, False):
                reasons = denial_reasons(driver_denied_tools(authored=authored, kind=kind))
                assert all(reasons.values()), (kind, authored, reasons)


class TestSubAgentBrief:
    """What the conductor wrote, and what the sub-agent is told regardless."""

    def test_every_field_the_conductor_filled_in_is_passed_on(self):
        from reinvent4_agent.supervisor import build_brief

        brief = build_brief(
            _task(
                kind="research", title="BD1 vs BD2 selectivity",
                context="The series loses selectivity above 400 Da.",
                intent="Find published selectivity handles.",
                deliverable="A ranked list with references.",
            ),
            campaign_goal="improve BRD4 BD1 affinity",
        )

        assert "BD1 vs BD2 selectivity" in brief
        assert "loses selectivity above 400 Da" in brief
        assert "Find published selectivity handles" in brief
        assert "A ranked list with references" in brief
        assert "improve BRD4 BD1 affinity" in brief

    def test_a_one_line_task_is_still_a_brief(self):
        """The conductor decides how precise to be; a terse task is not a
        broken one, so nothing is required but the intent."""
        from reinvent4_agent.supervisor import build_brief

        brief = build_brief(_task(kind="question", intent="Is PROTAC degradation plausible here?"))

        assert "Is PROTAC degradation plausible here?" in brief
        assert "None" not in brief

    def test_it_is_told_the_report_is_the_only_thing_that_leaves(self):
        """The conductor sees no transcript and no tool calls, so a finding
        left in the transcript did not happen."""
        from reinvent4_agent.supervisor import REPORT_HEADING, build_brief

        brief = build_brief(_task(kind="research", intent="look it up"))

        assert REPORT_HEADING in brief
        assert "only thing the conductor sees" in brief

    def test_it_is_told_it_cannot_start_a_run(self):
        """Being refused the tool without being told why reads as a broken
        toolchain; it should report the run it would have started instead."""
        from reinvent4_agent.supervisor import build_brief

        brief = build_brief(_task(kind="research", intent="look it up"))

        assert "cannot start a reinforcement-learning run" in brief

    def test_a_sub_agent_is_told_nobody_will_answer_a_question(self):
        """There is no user in a sub-agent's conversation: a clarifying
        question would end the task with nothing in it."""
        from reinvent4_agent.system_prompt import SUBAGENT_BRIEFING

        assert "Nobody will answer a question" in SUBAGENT_BRIEFING

    def test_it_is_told_which_campaign_and_branch_to_record_against(self):
        """The atlas tools need both, and nothing else in the brief said so."""
        from reinvent4_agent.supervisor import build_brief

        brief = build_brief(_task(kind="research", intent="look it up",
                                  workstream="selectivity"))

        assert "feed0001" in brief
        assert "a1b2c3d4" in brief
        assert "'selectivity'" in brief
        assert "atlas_finding" in brief

    def test_what_the_branch_already_knows_is_in_the_brief(self):
        """The sub-agent that does not know the question was answered will
        not think to ask: that duplicated work is what this prevents."""
        from reinvent4_agent.supervisor import build_brief

        brief = build_brief(
            _task(kind="research", intent="look it up", workstream="selectivity"),
            findings=[{
                "finding_id": "f1", "confidence": "high",
                "claim": "BD1 tolerates a methyl at the ZA channel",
                "evidence": ["10.1021/jm501245q"],
            }],
        )

        assert "BD1 tolerates a methyl at the ZA channel" in brief
        assert "10.1021/jm501245q" in brief
        assert "f1" in brief
        assert "Do not redo this work" in brief

    def test_an_empty_atlas_adds_nothing_to_the_brief(self):
        """A heading over no findings reads as a campaign that knows nothing
        when it is a branch nobody has filed under yet."""
        from reinvent4_agent.supervisor import build_brief

        brief = build_brief(_task(kind="research", intent="look it up"), findings=[])

        assert "already knows" not in brief


class TestDesignReport:
    """A design task reports to the conductor too, not only a profile."""

    @staticmethod
    def _outcome() -> TaskOutcome:
        return TaskOutcome(
            objective_profile={"QED": 0.81, "BoltzAffinity": 0.93},
            candidates=[{"smiles": "CCO"}],
            job_id="job-1",
        )

    def test_it_leads_with_what_the_ledger_already_knows(self):
        from reinvent4_agent.supervisor import design_report

        report = design_report("I ran it.", self._outcome(), "ttp-a1b2c3d4", False)

        assert "ttp-a1b2c3d4" in report
        assert "BoltzAffinity=0.93" in report
        assert "1 molecule(s) reported" in report
        assert "I ran it." in report

    def test_the_reviewer_handoff_block_is_cut(self):
        """It is addressed to the Reviewer. A conductor reading a wall of
        SMILES-as-JSON learns nothing the candidates do not already say."""
        from reinvent4_agent.supervisor import design_report

        fence = "`" * 3
        account = (
            "Two survived the confidence gate.\n\n"
            f"{fence}json\n"
            '{"candidates": ["CCO"], "ready_for_review": true}\n'
            f"{fence}"
        )

        report = design_report(account, self._outcome(), "ttp-a1b2c3d4", False)

        assert "Two survived the confidence gate." in report
        assert "ready_for_review" not in report

    def test_a_deadline_stop_is_stated(self):
        """Partial results read as a weak run unless the reason is there."""
        from reinvent4_agent.supervisor import design_report

        report = design_report("", self._outcome(), "ttp-a1b2c3d4", True)

        assert "deadline" in report

    def test_an_agent_that_said_nothing_still_reports(self):
        from reinvent4_agent.supervisor import design_report

        report = design_report(None, self._outcome(), "ttp-a1b2c3d4", False)

        assert "ttp-a1b2c3d4" in report


class TestJobIdCollection:
    """Every run a task started, not only the one its molecules came from."""

    @staticmethod
    def _collector():
        from reinvent4_agent.supervisor import _JobIdCollector

        return _JobIdCollector()

    def test_it_notes_the_runs_the_agent_started(self):
        collector = self._collector()

        collector.on_tool_result(
            "run_optimization", '{"job_id": "job-1", "state": "queued"}', False
        )
        collector.on_tool_result(
            "run_transfer_learning", '{"job_id": "job-2"}', False
        )

        assert collector.job_ids == ["job-1", "job-2"]

    def test_reading_the_job_table_is_not_starting_a_job(self):
        """list_jobs answers with every job in the pool, including other
        tasks' -- attributing those here would credit this task with them."""
        collector = self._collector()

        collector.on_tool_result(
            "list_jobs", '{"jobs": [{"job_id": "someone-elses"}]}', False
        )

        assert collector.job_ids == []

    def test_a_failed_call_started_nothing(self):
        collector = self._collector()

        collector.on_tool_result(
            "run_optimization", '{"error": "Maximum concurrent jobs (2) reached"}', True
        )

        assert collector.job_ids == []

    def test_the_same_run_is_noted_once(self):
        collector = self._collector()

        for _ in range(3):
            collector.on_tool_result("run_optimization", '{"job_id": "job-1"}', False)

        assert collector.job_ids == ["job-1"]


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


class TestSupervisorLoop:
    """The loop drives its own event loop: the repo has no pytest-asyncio."""

    def test_an_empty_queue_is_not_work(self):
        store = FakeStore()
        supervisor = Supervisor(store, _succeeds)

        assert asyncio.run(supervisor.run_once()) is False
        assert store.results == []

    def test_a_successful_task_posts_its_profile(self):
        store = FakeStore([_task()])
        supervisor = Supervisor(store, _succeeds)

        assert asyncio.run(supervisor.run_once()) is True

        task_id, outcome = store.results[0]
        assert task_id == "a1b2c3d4"
        assert outcome.objective_profile == {"BoltzAffinity": 0.8, "QED": 0.7}
        assert supervisor.tasks_completed == 1

    def test_a_failing_task_is_reported_and_requeued(self):
        async def _explodes(task):
            raise RuntimeError("the runner went away")

        store = FakeStore([_task()])
        supervisor = Supervisor(store, _explodes)

        assert asyncio.run(supervisor.run_once()) is True

        assert store.failures == [("a1b2c3d4", "the runner went away", True)]
        assert store.results == []
        assert supervisor.tasks_failed == 1

    def test_a_lost_lease_mid_task_posts_nothing(self):
        """The new holder's result must not be overwritten by ours."""
        async def _loses_lease(task):
            raise LeaseLost("held by supervisor-9")

        store = FakeStore([_task()])
        supervisor = Supervisor(store, _loses_lease)

        assert asyncio.run(supervisor.run_once()) is True

        assert store.results == []
        assert store.failures == [], "a task we no longer own is not ours to fail"

    def test_a_failure_report_that_cannot_be_delivered_is_survivable(self):
        """A store outage must not take the pod down with it."""
        class Unreachable(FakeStore):
            def fail(self, task_id, error, requeue=True):
                raise StoreUnavailable("store is down")

        async def _explodes(task):
            raise RuntimeError("boom")

        store = Unreachable([_task()])

        assert asyncio.run(Supervisor(store, _explodes).run_once()) is True

    def test_the_lease_is_renewed_while_the_task_runs(self):
        store = FakeStore([_task()])

        async def scenario():
            slow_enough = asyncio.Event()

            async def _slow(task):
                await slow_enough.wait()
                return await _succeeds(task)

            supervisor = Supervisor(store, _slow, heartbeat_seconds=0.01)
            running = asyncio.create_task(supervisor.run_once())
            await asyncio.sleep(0.08)
            slow_enough.set()
            await running

        asyncio.run(scenario())

        assert store.heartbeats, "the lease was never renewed"
        assert set(store.heartbeats) == {"a1b2c3d4"}

    def test_the_run_is_reported_while_it_runs(self):
        """The ledger learned the job id from the result, which is too late.

        For the hours a run was actually going, nothing joined the task to
        the job, so a person watching a working supervisor could not open
        what it was working on.
        """
        store = FakeStore([_task()])

        async def scenario():
            slow_enough = asyncio.Event()

            async def _slow(task):
                await slow_enough.wait()
                return await _succeeds(task)

            supervisor = Supervisor(
                store, _slow, heartbeat_seconds=0.01,
                job_id_source=lambda: "job-42",
            )
            running = asyncio.create_task(supervisor.run_once())
            await asyncio.sleep(0.08)
            slow_enough.set()
            await running

        asyncio.run(scenario())

        assert ("a1b2c3d4", "job-42") in store.reported_jobs

    def test_a_task_with_no_run_yet_still_beats(self):
        """The lease matters more than knowing which job it started."""
        store = FakeStore([_task()])

        async def scenario():
            slow_enough = asyncio.Event()

            async def _slow(task):
                await slow_enough.wait()
                return await _succeeds(task)

            supervisor = Supervisor(
                store, _slow, heartbeat_seconds=0.01,
                job_id_source=lambda: None,
            )
            running = asyncio.create_task(supervisor.run_once())
            await asyncio.sleep(0.08)
            slow_enough.set()
            await running

        asyncio.run(scenario())

        assert store.heartbeats, "the lease was never renewed"
        assert store.reported_jobs == []

    def test_a_job_lookup_that_breaks_does_not_drop_the_lease(self):
        """Losing the lease would requeue a task that is running fine."""
        store = FakeStore([_task()])

        def _explodes():
            raise RuntimeError("the runner is not answering")

        async def scenario():
            slow_enough = asyncio.Event()

            async def _slow(task):
                await slow_enough.wait()
                return await _succeeds(task)

            supervisor = Supervisor(
                store, _slow, heartbeat_seconds=0.01, job_id_source=_explodes,
            )
            running = asyncio.create_task(supervisor.run_once())
            await asyncio.sleep(0.08)
            slow_enough.set()
            await running

        asyncio.run(scenario())

        assert store.heartbeats, "the lease was never renewed"

    def test_heartbeating_stops_when_the_task_does(self):
        store = FakeStore([_task()])

        async def scenario():
            supervisor = Supervisor(store, _succeeds, heartbeat_seconds=0.01)
            await supervisor.run_once()
            before = len(store.heartbeats)
            await asyncio.sleep(0.05)
            return before

        before = asyncio.run(scenario())

        assert len(store.heartbeats) == before

    def test_a_store_outage_does_not_end_a_running_task(self):
        """The work in flight is worth more than the ledger entry."""
        class FlakyHeartbeat(FakeStore):
            def heartbeat(self, task_id, lease_seconds=None, job_id=None):
                super().heartbeat(task_id, lease_seconds, job_id)
                raise StoreUnavailable("store is down")

        store = FlakyHeartbeat([_task()])

        async def scenario():
            done = asyncio.Event()

            async def _slow(task):
                await done.wait()
                return await _succeeds(task)

            supervisor = Supervisor(store, _slow, heartbeat_seconds=0.01)
            running = asyncio.create_task(supervisor.run_once())
            await asyncio.sleep(0.05)
            done.set()
            await running

        asyncio.run(scenario())

        assert store.results, "the task should still have completed"

    def test_run_forever_drains_the_queue_then_stops(self):
        store = FakeStore([_task(), _task()])

        async def scenario():
            supervisor = Supervisor(store, _succeeds, poll_seconds=0.01)

            async def _stop_once_empty():
                while store.queue:
                    await asyncio.sleep(0.01)
                supervisor.stop()

            await asyncio.gather(supervisor.run_forever(), _stop_once_empty())

        asyncio.run(scenario())

        assert len(store.results) == 2

    def test_shutdown_wakes_the_poll_immediately(self):
        """A drain must not wait out a long empty-queue poll."""
        async def scenario():
            supervisor = Supervisor(FakeStore(), _succeeds, poll_seconds=30)

            async def _stop_soon():
                await asyncio.sleep(0.05)
                supervisor.stop()

            await asyncio.wait_for(
                asyncio.gather(supervisor.run_forever(), _stop_soon()), timeout=5
            )

        asyncio.run(scenario())

    def test_an_unreachable_store_backs_off_instead_of_crashing(self):
        class Dead(FakeStore):
            def claim(self, campaign_id=None, lease_seconds=None):
                self.claims += 1
                raise StoreUnavailable("connection refused")

        store = Dead()

        async def scenario():
            supervisor = Supervisor(store, _succeeds, poll_seconds=0.01)

            async def _stop_after_a_few():
                while store.claims < 3:
                    await asyncio.sleep(0.01)
                supervisor.stop()

            await asyncio.wait_for(
                asyncio.gather(supervisor.run_forever(), _stop_after_a_few()),
                timeout=5,
            )

        asyncio.run(scenario())

        assert store.claims >= 3, "it should keep retrying while the store is down"

    def test_it_reports_itself_idle_before_claiming(self):
        """The pass that finds an empty queue is the one worth reporting.

        A supervisor otherwise touches the ledger only when it claims, so the
        conductor could not tell a free one from one that never started.
        """
        store = FakeStore()

        asyncio.run(Supervisor(store, _succeeds).run_once())

        assert store.presence == [("idle", None)]

    def test_it_reports_itself_busy_once_it_holds_a_task(self):
        store = FakeStore([_task()])

        asyncio.run(Supervisor(store, _succeeds).run_once())

        assert ("busy", "a1b2c3d4") in store.presence

    def test_presence_is_renewed_while_a_task_runs(self):
        """A task outlasts the presence TTL many times over."""
        store = FakeStore([_task()])

        async def scenario():
            slow_enough = asyncio.Event()

            async def _slow(task):
                await slow_enough.wait()
                return await _succeeds(task)

            supervisor = Supervisor(
                store, _slow, poll_seconds=0.01, presence_seconds=0.01
            )
            running = asyncio.create_task(supervisor.run_forever())
            await asyncio.sleep(0.1)
            slow_enough.set()
            supervisor.stop()
            await running

        asyncio.run(scenario())

        assert store.presence.count(("busy", "a1b2c3d4")) > 1

    def test_losing_a_lease_does_not_remove_it_from_the_pool(self):
        """Presence is about the supervisor; the lease is about the task.

        They were renewed on the same beat, so a supervisor whose task was
        taken -- by a cancel, or its campaign being finalized -- kept working
        on the job it had already started while vanishing from the pool. Two
        did exactly that: alive, holding a GPU, invisible.
        """
        class LeaseTaken(FakeStore):
            def heartbeat(self, task_id, lease_seconds=None, job_id=None):
                raise LeaseLost("held by someone else")

        store = LeaseTaken([_task()])

        async def scenario():
            still_working = asyncio.Event()

            async def _slow(task):
                await still_working.wait()
                return await _succeeds(task)

            supervisor = Supervisor(
                store, _slow, poll_seconds=0.01,
                heartbeat_seconds=0.01, presence_seconds=0.01,
            )
            running = asyncio.create_task(supervisor.run_forever())
            await asyncio.sleep(0.1)
            # The lease is gone by now; presence must have carried on.
            renewals = store.presence.count(("busy", "a1b2c3d4"))
            still_working.set()
            supervisor.stop()
            await running
            return renewals

        assert asyncio.run(scenario()) > 1

    def test_it_leaves_the_pool_on_shutdown(self):
        """Draining on command should not read as a crash for the next TTL."""
        store = FakeStore()

        async def scenario():
            supervisor = Supervisor(store, _succeeds, poll_seconds=0.01)

            async def _stop_soon():
                await asyncio.sleep(0.03)
                supervisor.stop()

            await asyncio.gather(supervisor.run_forever(), _stop_soon())

        asyncio.run(scenario())

        assert store.left_the_pool

    def test_a_store_that_cannot_take_presence_does_not_lose_the_task(self):
        """Being invisible is better than losing an hour of RL over it."""
        class NoPresence(FakeStore):
            def report_presence(self, *a, **k):
                raise StoreUnavailable("store is down")

        store = NoPresence([_task()])

        assert asyncio.run(Supervisor(store, _succeeds).run_once()) is True
        assert store.results

    def test_claims_can_be_scoped_to_one_campaign(self):
        seen: list[Any] = []

        class Recording(FakeStore):
            def claim(self, campaign_id=None, lease_seconds=None):
                seen.append(campaign_id)
                return None

        asyncio.run(
            Supervisor(Recording(), _succeeds, campaign_id="feed0001").run_once()
        )

        assert seen == ["feed0001"]


# ---------------------------------------------------------------------------
# TTP variants and deadlines
# ---------------------------------------------------------------------------


@pytest.fixture
def ttp_store(tmp_path, monkeypatch):
    """A TTP store in tmp_path, with one simple base TTP in it."""

    from reinvent4_mcp.config import ProcessConfig, reset_config
    from reinvent4_mcp.tools import ttp as ttp_tools

    config = ProcessConfig(ttp={"storage_dir": str(tmp_path / "ttps")})
    monkeypatch.setattr("reinvent4_mcp.config.get_config", lambda: config)
    monkeypatch.setattr("reinvent4_mcp.tools.ttp.get_config", lambda: config)
    ttp_tools.reset_store()

    created = ttp_tools.create_ttp({
        "id": "brd4-bd1-lead-opt",
        "name": "BRD4 BD1 lead optimisation",
        "pre_scoring": {"components": [
            {"name": "QED", "type": "native", "component": "QED", "weight": 0.5},
        ]},
        "scoring": {"objectives": [
            {"name": "SA Score", "type": "native", "component": "SAScore",
             "weight": 1.0},
        ]},
    })
    # Checked, because an unstored base TTP leaves the store empty and every
    # test here then fails on "TTP not found" instead of on what it meant to
    # assert -- which is how eleven of them read as broken for weeks.
    if "error" in created:
        if "REINVENT4" in created["error"]:
            pytest.skip("validating the base TTP needs a REINVENT4 checkout")
        pytest.fail(f"base TTP rejected: {created['error']}")

    yield ttp_tools
    ttp_tools.reset_store()
    reset_config()


class TestPrepareTtpVariant:

    def test_the_variant_is_named_after_the_task(self, ttp_store):
        """So a result traces back to exactly the TTP that produced it."""
        variant_id = prepare_ttp_variant(
            "brd4-bd1-lead-opt", "a1b2c3d4", {"SA Score": 2.0}
        )

        assert variant_id == "brd4-bd1-lead-opt-a1b2c3d4"

    def test_the_variant_carries_the_overridden_weights(self, ttp_store):
        variant_id = prepare_ttp_variant(
            "brd4-bd1-lead-opt", "a1b2c3d4", {"SA Score": 2.0, "QED": 1.5}
        )
        stored = ttp_store.get_ttp(variant_id)

        assert stored["scoring"]["objectives"][0]["weight"] == 2.0
        assert stored["pre_scoring"]["components"][0]["weight"] == 1.5

    def test_the_base_ttp_is_left_alone(self, ttp_store):
        """Every supervisor reads the same base out of a shared store."""
        prepare_ttp_variant("brd4-bd1-lead-opt", "a1b2c3d4", {"SA Score": 9.0})
        base = ttp_store.get_ttp("brd4-bd1-lead-opt")

        assert base["scoring"]["objectives"][0]["weight"] == 1.0

    def test_the_variant_starts_at_version_one(self, ttp_store):
        """It is a new TTP, not a revision of the base's lineage."""
        variant_id = prepare_ttp_variant("brd4-bd1-lead-opt", "a1b2c3d4")

        assert ttp_store.get_ttp(variant_id)["version"] == 1

    def test_two_tasks_do_not_collide(self, ttp_store):
        """N supervisors write into one shared TTP store."""
        first = prepare_ttp_variant("brd4-bd1-lead-opt", "aaaaaaaa", {"SA Score": 1.0})
        second = prepare_ttp_variant("brd4-bd1-lead-opt", "bbbbbbbb", {"SA Score": 2.0})

        assert first != second
        assert ttp_store.get_ttp(second)["scoring"]["objectives"][0][
            "weight"
        ] == 2.0

    def test_a_retried_task_reuses_its_variant(self, ttp_store):
        """A requeued task must still be runnable.

        create_ttp refuses an id that exists, so the first live retry died
        with "TTP ... already exists" before it started -- meaning every
        retry was doomed. The content is a function of the task, so rewriting
        it changes nothing.
        """
        first = prepare_ttp_variant("brd4-bd1-lead-opt", "a1b2c3d4", {"SA Score": 2.0})
        second = prepare_ttp_variant("brd4-bd1-lead-opt", "a1b2c3d4", {"SA Score": 2.0})

        assert first == second
        stored = ttp_store.get_ttp(second)
        assert stored["scoring"]["objectives"][0]["weight"] == 2.0

    def test_a_missing_base_ttp_is_an_error(self, ttp_store):
        with pytest.raises(ValueError, match="nope"):
            prepare_ttp_variant("nope", "a1b2c3d4")

    def test_an_unknown_override_never_reaches_the_store(self, ttp_store):
        """Better to fail the task than to run the wrong weight vector."""
        with pytest.raises(ValueError, match="Affinity"):
            prepare_ttp_variant("brd4-bd1-lead-opt", "a1b2c3d4", {"Affinity": 2.0})

        assert "error" in ttp_store.get_ttp("brd4-bd1-lead-opt-a1b2c3d4")


class TestAuthoredTtpId:
    """The id reserved for a TTP the agent has not written yet.

    Prescribed rather than chosen, for the same reason the variant id is
    derived: it is how the task's job is found afterwards, by a conductor
    whose supervisor is long gone, without trusting a sentence the model
    wrote about what it did.
    """

    def test_it_is_derived_from_the_task(self):
        from reinvent4_agent.supervisor import authored_ttp_id

        assert authored_ttp_id("t1") == "ttp-t1"

    def test_two_tasks_do_not_collide(self):
        from reinvent4_agent.supervisor import authored_ttp_id

        assert authored_ttp_id("t1") != authored_ttp_id("t2")

    def test_a_retried_task_reuses_its_id(self):
        """A retry is the same experiment, so it finds the same run."""
        from reinvent4_agent.supervisor import authored_ttp_id

        assert authored_ttp_id("t1") == authored_ttp_id("t1")


class TestBaseTtpResolution:
    """Where the base TTP comes from.

    It belongs to the *campaign*, not to each task: a sweep varies weights
    against one fixed TTP. Reading it from the task spec instead left it empty
    and failed every task with "TTP '' not found" the first time a supervisor
    ran against a real ledger.
    """

    class _Store:
        def __init__(self, campaign: dict[str, Any]) -> None:
            self._campaign = campaign
            self.asked_for: list[str] = []

        def campaign(self, campaign_id: str) -> dict[str, Any]:
            self.asked_for.append(campaign_id)
            return self._campaign

    def test_it_comes_from_the_campaign(self):
        from reinvent4_agent.supervisor import CoscientistTaskRunner

        store = self._Store({"base_ttp_id": "brd4-bd1-lead-opt"})
        runner = CoscientistTaskRunner(object(), store)

        assert runner._base_ttp_id(_task()) == "brd4-bd1-lead-opt"
        assert store.asked_for == ["feed0001"]

    def test_a_campaign_without_one_means_the_task_writes_its_own(self):
        """Not an error: it is the usual way a campaign runs.

        A campaign names a base TTP only when it is deliberately a sweep of
        one profile. Without one, each task is told what it is for and the
        agent running it decides how that should be scored.
        """

        from reinvent4_agent.supervisor import CoscientistTaskRunner

        runner = CoscientistTaskRunner(object(), self._Store({}))

        assert runner._base_ttp_id(_task()) is None

    def test_an_empty_base_ttp_reads_as_none(self):
        """The ledger stores the absent TTP as "", and nothing downstream
        should have to know that."""

        from reinvent4_agent.supervisor import CoscientistTaskRunner

        runner = CoscientistTaskRunner(object(), self._Store({"base_ttp_id": ""}))

        assert runner._base_ttp_id(_task()) is None


class TestLeadTasks:

    def test_a_lead_conducts_its_workstream_instead_of_running_an_agent(self):
        from reinvent4_agent.supervisor import CoscientistTaskRunner

        store = MagicMock()
        runner = CoscientistTaskRunner(MagicMock(), store)
        led: list[tuple[Any, dict[str, Any]]] = []

        def conduct(ledger, task, proposer):
            led.append((ledger, task))
            return TaskOutcome(summary="led it")

        with patch("reinvent4_agent.leading.conduct", conduct):
            outcome = asyncio.run(runner(_task(kind="lead")))

        assert outcome.summary == "led it"
        assert led[0][0] is store


class TestJobDiscovery:
    """How the supervisor finds the job its Driver started.

    By observation, not by what the model said. The first live run failed a
    task with "the Driver finished without starting an optimization job"
    while that job was running fine -- the session had ended ``incomplete``,
    so there was no structured handoff to read the id out of.
    """

    @staticmethod
    def _jobs(*jobs: dict[str, Any]):
        manager = MagicMock()
        manager.list_jobs.return_value = list(jobs)
        return patch(
            "reinvent4_mcp.jobs.manager.get_job_manager", return_value=manager
        )

    def test_the_job_is_found_by_the_tasks_own_ttp(self):
        with self._jobs(
            {"job_id": "other", "ttp_id": "brd4-bd1-lead-opt-zzzz", "created_at": "1"},
            {"job_id": "mine", "ttp_id": "brd4-bd1-lead-opt-a1b2", "created_at": "2"},
        ):
            assert job_for_ttp("brd4-bd1-lead-opt-a1b2") == "mine"

    def test_no_job_for_the_ttp_is_reported_as_none(self):
        with self._jobs({"job_id": "other", "ttp_id": "something-else"}):
            assert job_for_ttp("brd4-bd1-lead-opt-a1b2") is None

    def test_an_empty_job_table_is_none(self):
        with self._jobs():
            assert job_for_ttp("brd4-bd1-lead-opt-a1b2") is None

    def test_a_retry_picks_the_newest_job(self):
        """A retried task can leave an older job of its own behind."""
        with self._jobs(
            {"job_id": "old", "ttp_id": "ttp-1", "created_at": "2026-10-08T10:00:00"},
            {"job_id": "new", "ttp_id": "ttp-1", "created_at": "2026-10-08T12:00:00"},
        ):
            assert job_for_ttp("ttp-1") == "new"

    def test_jobs_without_a_timestamp_do_not_break_the_choice(self):
        with self._jobs(
            {"job_id": "undated", "ttp_id": "ttp-1"},
            {"job_id": "dated", "ttp_id": "ttp-1", "created_at": "2026-10-08T12:00:00"},
        ):
            assert job_for_ttp("ttp-1") == "dated"


class TestSingleAgentMode:
    """REINVENT4_AGENT_MULTI_AGENT=false runs the Driver with no Reviewer.

    The Reviewer produces no molecules and roughly doubles the token spend,
    but turning it off is not purely a saving: ``monitor_campaign`` *is* the
    Reviewer cycle, so the supervisor needs its own way to wait for the job.
    """

    @staticmethod
    def _runner(monitor_interval: float = 0.01):
        from reinvent4_agent.supervisor import CoscientistTaskRunner

        args = MagicMock()
        args.monitor_interval = monitor_interval
        return CoscientistTaskRunner(args, MagicMock())

    def test_it_waits_for_the_job_to_stop_running(self):
        states = ["running", "running", "completed"]

        with patch(
            "reinvent4_mcp.tools.results.get_optimization_status",
            side_effect=lambda job_id: {"state": states.pop(0)},
        ):
            asyncio.run(self._runner()._watch_job("job-1"))

        assert states == [], "it should have polled until the job finished"

    def test_a_queued_job_is_not_mistaken_for_a_finished_one(self):
        """A job is queued for a moment before it runs.

        Waiting only on "running" exited on the first poll, so the task
        harvested a 0-byte results CSV, failed with "No columns to parse from
        file", and left its job folding molecules as an orphan.
        """
        states = ["queued", "queued", "running", "completed"]

        with patch(
            "reinvent4_mcp.tools.results.get_optimization_status",
            side_effect=lambda job_id: {"state": states.pop(0)},
        ):
            asyncio.run(self._runner()._watch_job("job-1"))

        assert states == [], "queued and running both mean the job owes a result"

    def test_the_wait_tracks_the_managers_own_active_states(self):
        """So a new job state cannot silently become 'finished' here."""
        from reinvent4_mcp.jobs.manager import ACTIVE_JOB_STATES

        assert {s.value for s in ACTIVE_JOB_STATES} == {"queued", "running"}

    def test_a_healthy_job_carries_a_null_error_and_is_still_waited_on(self):
        """The real status dict always has an "error" key, set to None.

        Testing for the key rather than its value was true for every job, so
        the wait returned on the first poll and harvested a 0-byte CSV while
        the job ran on. The earlier tests here passed a bare {"state": ...},
        which is why they never caught it -- so this one uses the shape the
        job manager actually produces.
        """
        states = ["running", "running", "completed"]

        def _status(job_id):
            return {
                "job_id": job_id,
                "state": states.pop(0),
                "error": None,
                "exit_code": None,
                "trajectory": {"current_step": 1},
            }

        with patch(
            "reinvent4_mcp.tools.results.get_optimization_status", side_effect=_status
        ):
            asyncio.run(self._runner()._watch_job("job-1"))

        assert states == [], "a null error must not end the wait"

    def test_a_status_with_no_state_is_not_waited_on(self):
        """A job the runner cannot describe at all."""
        with patch(
            "reinvent4_mcp.tools.results.get_optimization_status",
            return_value={"error": "No job job-1"},
        ):
            asyncio.run(self._runner()._watch_job("job-1"))


class TestFailedTaskReleasesItsJob:
    """A task that fails must not leave its RL job running.

    Nothing waits on the job once the task is failing, but it still holds a
    GPU and one of the runner's concurrency slots. Two abandoned jobs blocked
    every later task in a campaign with "Maximum concurrent jobs (2) already
    active".
    """

    @staticmethod
    def _runner():
        from reinvent4_agent.supervisor import CoscientistTaskRunner

        return CoscientistTaskRunner(MagicMock(), MagicMock())

    def _release(self, runner, harvest, stop):
        return patch.object(
            type(runner), "_harvest", staticmethod(harvest)
        ), patch.object(type(runner), "_stop_job", staticmethod(stop))

    def test_a_good_harvest_leaves_the_job_alone(self):
        runner = self._runner()
        outcome = TaskOutcome(objective_profile={"QED": 0.8})
        stopped: list[str] = []

        harvest, stop = self._release(
            runner, lambda *a: outcome, lambda j: stopped.append(j)
        )
        with harvest, stop:
            got = asyncio.run(
                runner._harvest_or_release(_task(), "job-1", "ok", 1, 0, False)
            )

        assert got is outcome
        assert stopped == []

    def test_a_failed_harvest_stops_the_job(self):
        runner = self._runner()
        stopped: list[str] = []

        def _no_results(*args):
            raise RuntimeError("produced no readable results")

        harvest, stop = self._release(runner, _no_results, lambda j: stopped.append(j))
        with harvest, stop:
            with pytest.raises(RuntimeError, match="no readable results"):
                asyncio.run(
                    runner._harvest_or_release(_task(), "job-1", "ok", 1, 0, False)
                )

        assert stopped == ["job-1"]

    def test_a_stop_that_fails_does_not_mask_the_real_error(self):
        """The task's own failure is the one worth reporting."""
        runner = self._runner()

        def _no_results(*args):
            raise RuntimeError("produced no readable results")

        def _unreachable(job_id):
            raise RuntimeError("runner unreachable")

        harvest, stop = self._release(runner, _no_results, _unreachable)
        with harvest, stop:
            with pytest.raises(RuntimeError, match="no readable results"):
                asyncio.run(
                    runner._harvest_or_release(_task(), "job-1", "ok", 1, 0, False)
                )


class TestStartingTheRunDirectly:
    """What happens when the agent does not start the run itself.

    With a Reviewer the Coordinator took several turns, so a Driver that got
    distracted had more chances; a single pass gets one. Nothing about the
    call is the agent's decision -- the TTP and step count both come from the
    task -- so the supervisor starts it rather than asking again and hoping.
    """

    @staticmethod
    def _runner():
        from reinvent4_agent.supervisor import CoscientistTaskRunner

        return CoscientistTaskRunner(MagicMock(), MagicMock())

    def test_it_starts_the_run_from_the_task(self):
        calls: list[dict[str, Any]] = []

        def _run(**kwargs):
            calls.append(kwargs)
            return {"job_id": "job-9", "error": None}

        with patch("reinvent4_mcp.tools.optimization.run_optimization", _run):
            assert self._runner()._start_job("ttp-1", 5) == "job-9"

        assert calls == [{"ttp_id": "ttp-1", "max_steps": 5, "fidelity_tier": "exploration"}]

    def test_no_step_budget_is_left_to_the_default(self):
        calls: list[dict[str, Any]] = []

        def _run(**kwargs):
            calls.append(kwargs)
            return {"job_id": "job-9"}

        with patch("reinvent4_mcp.tools.optimization.run_optimization", _run):
            self._runner()._start_job("ttp-1", None)

        # max_steps is omitted (None) so run_optimization can apply the
        # Boltz exploration cap; fidelity_tier defaults to exploration.
        assert calls == [{"ttp_id": "ttp-1", "fidelity_tier": "exploration"}]

    def test_a_refused_run_is_an_error(self):
        """e.g. the concurrency cap, which names the jobs to stop."""
        with patch(
            "reinvent4_mcp.tools.optimization.run_optimization",
            return_value={"error": "Maximum concurrent jobs (2) reached"},
        ):
            with pytest.raises(RuntimeError, match="Maximum concurrent jobs"):
                self._runner()._start_job("ttp-1", 5)

    def test_a_refusal_for_want_of_a_slot_is_told_apart(self):
        """Only that one is worth waiting out; it names the runs holding slots."""
        from reinvent4_agent.supervisor import RunRefused

        with patch(
            "reinvent4_mcp.tools.optimization.run_optimization",
            return_value={"error": "The cluster is slow to start runs", "active_jobs": ["a"]},
        ):
            with pytest.raises(RunRefused):
                self._runner()._start_job("ttp-1", 5)

    def test_a_run_refused_for_a_slot_is_started_when_one_frees(self, monkeypatch):
        """Several supervisors can see the same last slot; the losers wait."""
        from reinvent4_agent import supervisor

        monkeypatch.setattr(supervisor, "RUN_ROOM_POLL_SECONDS", 0.0)
        replies = iter([
            {"error": "full", "active_jobs": ["a", "b"]},
            {"error": "full", "active_jobs": ["a", "b"]},
            {"job_id": "job-9"},
        ])
        with patch("reinvent4_mcp.tools.optimization.run_optimization", lambda **k: next(replies)):
            job_id = asyncio.run(self._runner()._start_when_room(_task(), "ttp-1", 5))

        assert job_id == "job-9"

    def test_a_run_refused_at_the_deadline_fails_the_task(self, monkeypatch):
        from reinvent4_agent.supervisor import RunRefused

        task = _task()
        task["spec"]["deadline"] = "2000-01-01T00:00:00+00:00"
        with patch(
            "reinvent4_mcp.tools.optimization.run_optimization",
            return_value={"error": "full", "active_jobs": ["a"]},
        ):
            with pytest.raises(RunRefused):
                asyncio.run(self._runner()._start_when_room(task, "ttp-1", 5))

    def test_a_design_task_waits_for_room_before_its_agents_start(self, monkeypatch):
        from reinvent4_agent import supervisor

        monkeypatch.setattr(supervisor, "RUN_ROOM_POLL_SECONDS", 0.0)
        rooms = iter([{"free": 0, "active": 4, "limit": 4}, {"free": 0, "active": 4, "limit": 4},
                      {"free": 1, "active": 3, "limit": 4}])
        asked: list[int] = []

        def _room():
            asked.append(1)
            return next(rooms)

        with patch("reinvent4_mcp.tools.campaign.run_room", _room):
            asyncio.run(self._runner()._wait_for_run_room(_task()))

        assert len(asked) == 3

    def test_an_unreadable_window_holds_nothing_up(self):
        with patch("reinvent4_mcp.tools.campaign.run_room", return_value=None):
            asyncio.run(self._runner()._wait_for_run_room(_task()))

    def test_a_null_error_is_not_a_failure(self):
        """Same trap as the status dict: the key is always there."""
        with patch(
            "reinvent4_mcp.tools.optimization.run_optimization",
            return_value={"job_id": "job-9", "error": None},
        ):
            assert self._runner()._start_job("ttp-1", 5) == "job-9"

    def test_a_response_without_a_job_id_is_an_error(self):
        with patch(
            "reinvent4_mcp.tools.optimization.run_optimization", return_value={}
        ):
            with pytest.raises(RuntimeError, match="no job id"):
                self._runner()._start_job("ttp-1", 5)

    def test_a_failed_job_also_ends_the_wait(self):
        """Any terminal state, not just success -- otherwise it waits forever."""
        with patch(
            "reinvent4_mcp.tools.results.get_optimization_status",
            return_value={"state": "failed"},
        ):
            asyncio.run(self._runner()._watch_job("job-1"))

    def test_an_undescribable_job_is_not_waited_on(self):
        """A job the runner cannot report on is not one we can wait for."""
        with patch(
            "reinvent4_mcp.tools.results.get_optimization_status",
            return_value={"error": "No job job-1"},
        ):
            asyncio.run(self._runner()._watch_job("job-1"))

    def test_the_deadline_still_stops_the_job(self):
        """The single-agent watch is bounded by the task deadline, as before."""
        stopped: list[str] = []

        async def _never_finishes(job_id):
            await asyncio.sleep(10)

        runner = self._runner()
        with patch.object(runner, "_watch_job", _never_finishes), patch.object(
            type(runner), "_stop_job", staticmethod(lambda job_id: stopped.append(job_id))
        ):
            timed_out = asyncio.run(
                runner._monitor(None, "goal", "job-1", remaining=0.05)
            )

        assert timed_out is True
        assert stopped == ["job-1"]


class TestHarvestedProfileIsScopedToTheCampaign:
    """The profile reports the campaign's objectives, not every scored column.

    A results CSV carries a column per scorer diagnostic -- std,
    confidence_score, affinity_probability_binary and friends -- and reporting
    those as objectives put a dozen keys on every result where three
    belonged, and bloated the stored results.
    """

    @staticmethod
    def _runner(campaign: dict[str, Any] | Exception):
        from reinvent4_agent.supervisor import CoscientistTaskRunner

        store = MagicMock()
        if isinstance(campaign, Exception):
            store.campaign.side_effect = campaign
        else:
            store.campaign.return_value = campaign
        return CoscientistTaskRunner(MagicMock(), store)

    def test_it_reads_the_objectives_off_the_campaign(self):
        runner = self._runner({"objectives": ["BoltzAffinity", "QED"]})

        assert runner._objectives(_task()) == ["BoltzAffinity", "QED"]

    def test_an_unreadable_campaign_falls_back_to_every_column(self):
        """Reporting nothing leaves the molecules unscored; reporting extra
        only clutters."""
        runner = self._runner(RuntimeError("store down"))

        assert runner._objectives(_task()) is None

    def test_a_campaign_that_declares_none_falls_back_the_same_way(self):
        """A campaign whose tasks write their own TTPs declares no
        objectives. Narrowing a result to an empty list keeps no scores at
        all, so the molecules arrive with nothing to say what they scored --
        seen on LUMI, on a run that had already produced 25 molecules."""

        runner = self._runner({"objectives": []})

        assert runner._objectives(_task()) is None

    def test_molecules_without_scores_are_still_reported(self):
        """The molecules are what the run was for. An unreadable score column
        is a note on the result, not a reason to spend the GPU-hour again."""
        runner = self._runner({"objectives": ["BoltzAffinity"]})
        top = {"candidates": [{"smiles": "CCO", "total_score": 0.4, "components": {}}]}

        with patch("reinvent4_mcp.tools.results.get_top_candidates", return_value=top):
            outcome = runner._harvest(_task(), "job-1", "ok", 1, 0, False)

        assert outcome.objective_profile == {}
        assert len(outcome.candidates) == 1
        assert "no per-objective scores" in outcome.summary

    def test_each_molecule_names_the_run_that_scored_it(self):
        """Where the UI finds its complex, also once a lead pools it with others."""
        runner = self._runner({"objectives": ["BoltzAffinity"]})
        top = {"candidates": [{"smiles": "CCO", "total_score": 0.4}, {"smiles": "CCN", "total_score": 0.3}]}

        with patch("reinvent4_mcp.tools.results.get_top_candidates", return_value=top):
            outcome = runner._harvest(_task(), "job-1", "ok", 1, 0, False)

        assert [c["job_id"] for c in outcome.candidates] == ["job-1", "job-1"]
        assert "job_id" not in top["candidates"][0]

    def test_a_run_with_no_molecules_still_fails(self):
        runner = self._runner({"objectives": ["BoltzAffinity"]})

        with patch(
            "reinvent4_mcp.tools.results.get_top_candidates",
            return_value={"candidates": []},
        ):
            with pytest.raises(RuntimeError, match="without a single molecule"):
                runner._harvest(_task(), "job-1", "ok", 1, 0, False)

    def test_the_profile_drops_the_diagnostic_columns(self):
        candidates = [{
            "total_score": 0.9,
            "components": {
                "BoltzAffinity": {"transformed": 0.9},
                "QED": {"transformed": 0.8},
                "confidence_score (BoltzAffinity)": {"transformed": 0.87},
                "std (logD)": {"transformed": 0.18},
            },
        }]

        scoped = objective_profile(candidates, ["BoltzAffinity", "QED"])

        assert set(scoped) == {"BoltzAffinity", "QED"}


class TestDeadlines:

    def test_a_future_deadline_is_the_time_remaining(self):
        when = datetime.now(timezone.utc) + timedelta(seconds=600)
        remaining = _deadline_seconds(_task(deadline=when.isoformat()))

        assert 590 < remaining <= 600

    def test_a_past_deadline_is_zero_not_negative(self):
        """A negative timeout would raise rather than stop the task."""
        when = datetime.now(timezone.utc) - timedelta(seconds=60)

        assert _deadline_seconds(_task(deadline=when.isoformat())) == 0.0

    def test_no_deadline_means_no_limit(self):
        assert _deadline_seconds(_task()) is None

    def test_a_naive_deadline_is_read_as_utc(self):
        """Everything else in the ledger is UTC, so a bare timestamp is too."""
        when = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(seconds=300)
        remaining = _deadline_seconds(_task(deadline=when.isoformat()))

        assert 290 < remaining <= 300

    def test_an_unparseable_deadline_is_ignored_not_fatal(self):
        assert _deadline_seconds(_task(deadline="whenever")) is None


class TestOverdueTasks:
    """A deadline is absolute, so a queued task can outlive its own.

    Two LUMI attempts on one task were claimed two hours after its deadline,
    started, stopped inside their first RL step with nothing to checkpoint,
    and reported as an unreadable results CSV -- the real reason appearing
    nowhere. Zero time left and no deadline at all have to read differently.
    """

    def test_a_task_claimed_after_its_deadline_is_overdue(self):
        when = datetime.now(timezone.utc) - timedelta(minutes=120)

        overdue = _overdue_seconds(_task(deadline=when.isoformat()))

        # Two hours plus however long this test took to get here, so the slack
        # belongs above 7200 and not below: the deadline is read after it is
        # written, never before. Capped at 7200 this passed only where the
        # clock was too coarse to tick between the two reads.
        assert 7200 <= overdue < 7260

    def test_a_task_still_inside_its_deadline_is_not(self):
        when = datetime.now(timezone.utc) + timedelta(seconds=600)

        assert _overdue_seconds(_task(deadline=when.isoformat())) is None

    def test_no_deadline_is_never_overdue(self):
        assert _overdue_seconds(_task()) is None

    def test_an_unparseable_deadline_is_not_treated_as_passed(self):
        """Refusing the task would be worse than running it unbounded."""
        assert _overdue_seconds(_task(deadline="whenever")) is None


# ---------------------------------------------------------------------------
# Store client
# ---------------------------------------------------------------------------


class TestCampaignStoreClient:

    def test_204_means_nothing_to_claim(self, monkeypatch):
        monkeypatch.setattr(
            requests, "request", lambda *a, **k: _response(204, None)
        )

        assert CampaignStoreClient(owner="s0").claim() is None

    def test_a_claim_returns_the_task(self, monkeypatch):
        monkeypatch.setattr(
            requests, "request",
            lambda *a, **k: _response(200, {"task_id": "abc"}),
        )

        assert CampaignStoreClient(owner="s0").claim()["task_id"] == "abc"

    def test_409_is_a_lost_lease_not_an_outage(self, monkeypatch):
        """They are different problems: one means stop, the other means retry."""
        monkeypatch.setattr(
            requests, "request",
            lambda *a, **k: _response(409, {"detail": "held by s9"}),
        )

        with pytest.raises(LeaseLost, match="held by s9"):
            CampaignStoreClient(owner="s0").heartbeat("abc")

    def test_a_connection_error_is_an_outage(self, monkeypatch):
        def _refuse(*a, **k):
            raise requests.ConnectionError("refused")

        monkeypatch.setattr(requests, "request", _refuse)

        with pytest.raises(StoreUnavailable, match="unreachable"):
            CampaignStoreClient(owner="s0").claim()

    def test_a_server_error_is_an_outage(self, monkeypatch):
        monkeypatch.setattr(
            requests, "request", lambda *a, **k: _response(500, {"error": "boom"})
        )

        with pytest.raises(StoreUnavailable, match="500"):
            CampaignStoreClient(owner="s0").claim()

    def test_the_owner_is_sent_on_every_call(self, monkeypatch):
        sent: list[dict[str, Any]] = []

        def _capture(method, url, json=None, timeout=None):
            sent.append(json or {})
            return _response(200, {"task_id": "abc"})

        monkeypatch.setattr(requests, "request", _capture)
        client = CampaignStoreClient(owner="supervisor-3")
        client.claim()
        client.heartbeat("abc")
        client.post_result("abc", TaskOutcome(objective_profile={"QED": 0.5}))
        client.fail("abc", "boom")

        assert [body.get("owner") for body in sent] == ["supervisor-3"] * 4


class TestFollowUps:
    """What a sub-agent proposes at the end of its report."""

    def test_the_proposed_tasks_are_read_off_the_report(self):
        from reinvent4_agent.supervisor import parse_follow_ups

        proposals = parse_follow_ups(
            "## Report\n\nBD1 and BD2 differ at the ZA loop.\n\n"
            "## Next tasks\n"
            "- (design) Hold the triazole: run RL holding the methyl-triazole "
            "and varying the fused ring, since the 0.81 scorer came from it.\n"
            "- (retrosynthesis) Route the top two: check building-block "
            "availability for the two best candidates.\n"
        )

        assert [p["kind"] for p in proposals] == ["design", "retrosynthesis"]
        assert proposals[0]["title"] == "Hold the triazole"
        assert "varying the fused ring" in proposals[0]["intent"]

    def test_a_report_that_proposes_nothing_proposes_nothing(self):
        from reinvent4_agent.supervisor import parse_follow_ups

        assert parse_follow_ups("## Report\n\nNothing conclusive.") == []

    def test_an_unknown_kind_is_research_rather_than_dropped(self):
        """The line is the task; its label is the least of it."""
        from reinvent4_agent.supervisor import parse_follow_ups

        proposals = parse_follow_ups(
            "## Next tasks\n- (wetlab) Assay the series: measure the top five in BD1."
        )

        assert proposals[0]["kind"] == "research"

    def test_a_line_without_a_title_is_titled_by_its_first_words(self):
        from reinvent4_agent.supervisor import parse_follow_ups

        proposals = parse_follow_ups(
            "## Next tasks\n- Counter-screen the best two against BRD2 and BRD3."
        )

        assert proposals[0]["title"] == "Counter-screen the best two against BRD2"
        assert proposals[0]["intent"].startswith("Counter-screen")

    def test_no_more_than_three_are_taken(self):
        """A budget, not a limit on what it may think: the conductor ranks them."""
        from reinvent4_agent.supervisor import MAX_FOLLOW_UPS, parse_follow_ups

        proposals = parse_follow_ups(
            "## Next tasks\n" + "".join(
                f"- (research) Task {i}: do the {i}th thing worth doing.\n" for i in range(6)
            )
        )

        assert len(proposals) == MAX_FOLLOW_UPS

    def test_prose_after_the_list_ends_it(self):
        from reinvent4_agent.supervisor import parse_follow_ups

        proposals = parse_follow_ups(
            "## Next tasks\n"
            "- (research) Check selectivity: counter-screen against BRD2.\n"
            "I would start with the first of these.\n"
            "- (design) Not this one: it comes after the prose.\n"
        )

        assert len(proposals) == 1

    def test_the_brief_asks_for_them(self):
        from reinvent4_agent.supervisor import NEXT_TASKS_HEADING, build_brief

        brief = build_brief(_task(kind="research", intent="Find selectivity handles."))

        assert NEXT_TASKS_HEADING in brief


class _response:
    """Minimal stand-in for requests.Response."""

    def __init__(self, status_code: int, body: Any) -> None:
        self.status_code = status_code
        self._body = body
        self.ok = status_code < 400
        self.content = b"" if body is None else b"{}"
        self.text = str(body)

    def json(self) -> Any:
        if self._body is None:
            raise ValueError("no body")
        return self._body
