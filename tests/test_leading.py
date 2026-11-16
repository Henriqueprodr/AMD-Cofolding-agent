# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A lead conducts a workstream on the parent's pool and reports it up.

Against an in-memory ledger whose sub-agents finish every queued task each
time the lead sleeps, so a whole workstream runs in a handful of ticks.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from reinvent4_agent import leading
from reinvent4_agent.leading import conduct, lead_campaign
from reinvent4_agent.proposing import TaskProposal


class Ledger:
    def __init__(self) -> None:
        self.campaigns_: dict[str, dict[str, Any]] = {
            "parent": {
                "campaign_id": "parent", "goal": "Find a BRD4 BD1-selective lead",
                "base_ttp_id": None, "objectives": ["QED"], "concurrency": 4,
                "state": "dispatching", "parent_task_id": None,
            },
        }
        self.tasks_: list[dict[str, Any]] = []
        self.results_: list[dict[str, Any]] = []
        self.proposals_: list[dict[str, Any]] = []
        self.dropped: list[str] = []
        self.created: list[dict[str, Any]] = []

    # -- campaigns
    def campaigns(self, session_id=None):
        return list(self.campaigns_.values())

    def campaign(self, campaign_id):
        return self.campaigns_[campaign_id]

    def create_campaign(self, goal, **fields):
        campaign_id = f"sub{len(self.created) + 1}"
        campaign = {"campaign_id": campaign_id, "goal": goal, "state": "dispatching", **fields}
        self.campaigns_[campaign_id] = campaign
        self.created.append(campaign)
        return campaign

    def conducted(self, campaign_id):
        return self.campaigns_[campaign_id]

    def close_campaign(self, campaign_id):
        self.campaigns_[campaign_id]["state"] = "draining"

    def finalize_campaign(self, campaign_id):
        self.campaigns_[campaign_id]["state"] = "finalized"

    def status(self, campaign_id):
        counts: dict[str, int] = {}
        for task in self.tasks(campaign_id):
            counts[task["state"]] = counts.get(task["state"], 0) + 1
        return {
            "campaign": self.campaigns_[campaign_id],
            "task_counts": counts,
            "window": {"phase": "dispatching"},
        }

    # -- tasks
    def tasks(self, campaign_id, state=None):
        return [t for t in self.tasks_ if t["campaign_id"] == campaign_id]

    def results(self, campaign_id):
        ids = {t["task_id"] for t in self.tasks(campaign_id)}
        return [r for r in self.results_ if r["task_id"] in ids]

    def enqueue_task(self, campaign_id, **spec):
        task = {"task_id": f"t{len(self.tasks_) + 1}", "campaign_id": campaign_id,
                "state": "queued", "spec": spec}
        self.tasks_.append(task)
        return task

    def proposals(self, campaign_id, state="open"):
        return [p for p in self.proposals_ if p["campaign_id"] == campaign_id]

    def drop_proposals(self, campaign_id, ids):
        self.dropped += ids
        self.proposals_ = [p for p in self.proposals_ if p["proposal_id"] not in ids]

    def take_proposals(self, campaign_id, ids, priority=0):
        return []

    def work(self) -> None:
        """Every queued task is claimed and reports."""
        for task in self.tasks_:
            if task["state"] == "queued":
                task["state"] = "done"
                n = len(self.results_)
                self.results_.append({
                    "task_id": task["task_id"], "summary": f"finding {n}",
                    "objective_profile": {"QED": 0.5 + n / 100},
                    "candidates": [{"smiles": "C" * (n + 1), "score": n}],
                    "job_ids": [f"job-{n}"],
                })


class Proposer:
    def __init__(self) -> None:
        self.asked = 0

    def propose(self, context, count):
        self.asked += 1
        return [TaskProposal(intent=f"question {self.asked}.{i}") for i in range(count)]


LEAD = {
    "task_id": "lead1", "campaign_id": "parent",
    "spec": {"kind": "lead", "title": "Selectivity", "intent": "Settle BD1 over BD2", "concurrency": 2},
}


@pytest.fixture(autouse=True)
def _no_gateway():
    with patch("reinvent4_agent.conductor.queue_pressure", return_value=None):
        yield


def _run(ledger: Ledger, proposer: Proposer | None = None):
    return conduct(ledger, LEAD, proposer or Proposer(), tick_seconds=0,
                   sleep=lambda _: ledger.work(), clock=lambda: 0.0)


class TestTheLeadsCampaign:

    def test_it_is_opened_under_the_parent_with_its_team(self):
        ledger = Ledger()

        campaign = lead_campaign(ledger, LEAD, now=0.0)

        assert campaign["parent_campaign_id"] == "parent"
        assert campaign["parent_task_id"] == "lead1"
        assert campaign["concurrency"] == 2
        assert campaign["objectives"] == ["QED"]
        assert "Settle BD1 over BD2" in campaign["goal"]
        assert "Find a BRD4 BD1-selective lead" in campaign["goal"]

    def test_a_lead_claimed_again_carries_on_with_the_same_one(self):
        ledger = Ledger()
        first = lead_campaign(ledger, LEAD, now=0.0)

        again = lead_campaign(ledger, LEAD, now=0.0)

        assert again["campaign_id"] == first["campaign_id"]
        assert len(ledger.created) == 1


class TestConducting:

    def test_it_dispatches_its_budget_then_drains_and_finalizes(self):
        ledger = Ledger()

        _run(ledger)

        sub = ledger.created[0]["campaign_id"]
        assert ledger.campaigns_[sub]["state"] == "finalized"
        assert len(ledger.tasks(sub)) == 2 * leading.TASKS_PER_MEMBER
        assert all(t["state"] == "done" for t in ledger.tasks(sub))

    def test_it_never_holds_more_than_its_team(self):
        ledger = Ledger()
        held: list[int] = []

        def work(_):
            sub = ledger.created[0]["campaign_id"]
            held.append(sum(t["state"] == "queued" for t in ledger.tasks(sub)))
            ledger.work()

        conduct(ledger, LEAD, Proposer(), tick_seconds=0, sleep=work, clock=lambda: 0.0)

        assert max(held) <= 2

    def test_a_parent_stopped_ends_the_lead(self):
        ledger = Ledger()

        def stop(_):
            ledger.campaigns_[ledger.created[0]["campaign_id"]]["state"] = "finalized"

        outcome = conduct(ledger, LEAD, Proposer(), tick_seconds=0, sleep=stop, clock=lambda: 0.0)

        assert "0 task(s) reported" in outcome.summary


class TestTheReportUp:

    def test_it_carries_every_finding_the_best_scores_and_molecules(self):
        ledger = Ledger()

        outcome = _run(ledger)

        assert outcome.report.startswith("## Report")
        assert outcome.report.count("finding ") == 6
        assert outcome.objective_profile == {"QED": pytest.approx(0.55)}
        assert outcome.candidates[0]["score"] == 5
        assert len(outcome.job_ids) == 6

    def test_what_its_sub_agents_proposed_is_handed_up(self):
        ledger = Ledger()
        sub = lead_campaign(ledger, LEAD, now=0.0)["campaign_id"]
        ledger.proposals_ = [
            {"proposal_id": f"p{i}", "campaign_id": sub,
             "spec": {"kind": "design", "title": f"next {i}", "intent": f"run {i} next"}}
            for i in range(5)
        ]

        outcome = _run(ledger)

        assert [f["title"] for f in outcome.follow_ups] == ["next 0", "next 1", "next 2"]
        assert sorted(ledger.dropped) == [f"p{i}" for i in range(5)]
