"""Tests for services/campaign-store/app.py -- the fleet control-plane ledger.

The two things worth testing hard are the ones that cost real GPU time when
they are wrong: two supervisors must never be handed the same task, and a
supervisor that dies holding one must give it back.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_STORE = Path(__file__).resolve().parents[1] / "services" / "campaign-store" / "app.py"


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A fresh store module bound to a per-test SQLite file.

    The module reads its config at import time, so it is imported per test
    rather than once, which also guarantees no state leaks between tests.
    """

    monkeypatch.setenv("CAMPAIGN_STORE_DB", str(tmp_path / "campaigns.db"))
    monkeypatch.setenv("TASK_MAX_ATTEMPTS", "2")

    spec = importlib.util.spec_from_file_location("campaign_store_under_test", _STORE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    try:
        yield module
    finally:
        sys.modules.pop(spec.name, None)


@pytest.fixture
def client(store):
    with TestClient(store.app) as c:
        yield c


def _campaign(client, **overrides):
    body = {
        "goal": "Improve BRD4 BD1 affinity without losing drug-likeness",
        "base_ttp_id": "brd4-bd1-lead-opt",
        "objectives": ["BoltzAffinity", "QED"],
        "concurrency": 2,
    }
    body.update(overrides)
    response = client.post("/campaigns", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def _task(client, campaign_id, **overrides):
    body = {"weight_overrides": {"BoltzAffinity": 1.0, "QED": 0.3}}
    body.update(overrides)
    response = client.post(f"/campaigns/{campaign_id}/tasks", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def _claim(client, owner, **overrides):
    body = {"owner": owner}
    body.update(overrides)
    return client.post("/tasks/claim", json=body)


# ---------------------------------------------------------------------------
# Campaigns
# ---------------------------------------------------------------------------


class TestCampaigns:

    def test_health_reports_the_db_and_open_work(self, client):
        body = client.get("/health").json()
        assert body["status"] == "healthy"
        assert body["active_campaigns"] == 0
        assert body["open_tasks"] == 0

    def test_create_and_read_back(self, client):
        created = _campaign(client)
        assert created["state"] == "dispatching"
        assert created["objectives"] == ["BoltzAffinity", "QED"]

        fetched = client.get(f"/campaigns/{created['campaign_id']}").json()
        assert fetched == created

    def test_unknown_campaign_is_404(self, client):
        assert client.get("/campaigns/nope").status_code == 404

    def test_a_campaign_needs_neither_a_base_ttp_nor_objectives(self, client):
        """The common case: tasks bring their own TTP, so there is nothing
        to declare up front."""

        response = client.post("/campaigns", json={"goal": "g"})

        assert response.status_code == 201
        campaign = response.json()
        assert campaign["base_ttp_id"] is None
        assert campaign["objectives"] == []

    def test_an_absent_base_ttp_reads_back_as_none_not_empty(self, client):
        """Stored as "" because the column is NOT NULL, which is an artifact
        of the schema and must not leak to anyone reading the campaign."""

        created = client.post("/campaigns", json={"goal": "g"}).json()

        fetched = client.get(f"/campaigns/{created['campaign_id']}").json()
        assert fetched["base_ttp_id"] is None

    def test_deadlines_derive_from_the_dispatch_and_grace_windows(self, client):
        campaign = _campaign(client, dispatch_seconds=600, grace_seconds=300)
        assert campaign["hard_stop_at"] - campaign["dispatch_until"] == pytest.approx(300)

    def test_no_dispatch_deadline_means_no_hard_stop(self, client):
        campaign = _campaign(client)
        assert campaign["dispatch_until"] is None
        assert campaign["hard_stop_at"] is None

    def test_closing_shuts_the_queue_but_not_the_campaign(self, client):
        campaign = _campaign(client)
        cid = campaign["campaign_id"]
        _task(client, cid)

        assert client.post(f"/campaigns/{cid}/close").json()["state"] == "draining"

        refused = client.post(f"/campaigns/{cid}/tasks", json={})
        assert refused.status_code == 409
        assert "closed" in refused.json()["detail"]

        # Already-queued work is still claimable during the drain window.
        assert _claim(client, "supervisor-0").status_code == 200

    def test_concurrency_can_be_changed(self, client):
        """Held in the ledger so it survives a conductor restart."""
        cid = _campaign(client)["campaign_id"]

        updated = client.post(
            f"/campaigns/{cid}/concurrency", json={"concurrency": 5}
        ).json()

        assert updated["concurrency"] == 5
        assert client.get(f"/campaigns/{cid}").json()["concurrency"] == 5

    def test_concurrency_can_be_dropped_to_zero(self, client):
        """Pausing a campaign without closing it."""
        cid = _campaign(client)["campaign_id"]

        assert client.post(
            f"/campaigns/{cid}/concurrency", json={"concurrency": 0}
        ).status_code == 200

    def test_a_negative_concurrency_is_refused(self, client):
        cid = _campaign(client)["campaign_id"]

        assert client.post(
            f"/campaigns/{cid}/concurrency", json={"concurrency": -1}
        ).status_code == 422

    def test_finalize_abandons_outstanding_work(self, client):
        """Finalizing must not leave tasks that nobody will ever run as queued."""
        cid = _campaign(client)["campaign_id"]
        _task(client, cid)
        _task(client, cid)
        _claim(client, "supervisor-0")

        assert client.post(f"/campaigns/{cid}/finalize").json()["state"] == "finalized"

        states = {t["state"] for t in client.get(f"/campaigns/{cid}/tasks").json()}
        assert states == {"abandoned"}
        assert _claim(client, "supervisor-1").status_code == 204


# ---------------------------------------------------------------------------
# Claims and leases
# ---------------------------------------------------------------------------


class TestClaims:

    def test_claim_returns_204_when_idle(self, client):
        _campaign(client)
        assert _claim(client, "supervisor-0").status_code == 204

    def test_claim_hands_out_the_oldest_task_first(self, client):
        cid = _campaign(client)["campaign_id"]
        first = _task(client, cid, weight_overrides={"QED": 0.1})
        second = _task(client, cid, weight_overrides={"QED": 0.9})

        assert _claim(client, "s0").json()["task_id"] == first["task_id"]
        assert _claim(client, "s1").json()["task_id"] == second["task_id"]

    def test_claim_carries_the_spec_the_supervisor_needs(self, client):
        cid = _campaign(client)["campaign_id"]
        _task(
            client, cid,
            weight_overrides={"BoltzAffinity": 2.0},
            fidelity_tier="validation",
            max_steps=120,
        )

        spec = _claim(client, "s0").json()["spec"]
        assert spec["weight_overrides"] == {"BoltzAffinity": 2.0}
        assert spec["fidelity_tier"] == "validation"
        assert spec["max_steps"] == 120

    def test_a_task_is_only_handed_out_once(self, client):
        cid = _campaign(client)["campaign_id"]
        _task(client, cid)

        assert _claim(client, "s0").status_code == 200
        assert _claim(client, "s1").status_code == 204

    def test_concurrent_claims_never_hand_out_the_same_task(self, store, client):
        """The race that costs real GPU time if the claim is not atomic.

        Without BEGIN IMMEDIATE around the select-then-update, two supervisors
        arriving together both read the task as queued and both run the same
        weight vector on separate GPUs.
        """
        cid = _campaign(client)["campaign_id"]
        for _ in range(8):
            _task(client, cid)

        with ThreadPoolExecutor(max_workers=8) as pool:
            responses = list(
                pool.map(
                    lambda owner: _claim(client, owner),
                    [f"supervisor-{i}" for i in range(8)],
                )
            )

        claimed = [r.json()["task_id"] for r in responses if r.status_code == 200]
        assert len(claimed) == 8, "every task should have been claimed"
        assert len(set(claimed)) == 8, f"a task was handed out twice: {claimed}"

    def test_claim_is_scoped_to_one_campaign_when_asked(self, client):
        other = _campaign(client, goal="other")["campaign_id"]
        mine = _campaign(client, goal="mine")["campaign_id"]
        _task(client, other)
        wanted = _task(client, mine)

        claimed = _claim(client, "s0", campaign_id=mine).json()
        assert claimed["task_id"] == wanted["task_id"]

    def test_a_campaign_whose_conductor_went_quiet_gives_its_queue_up(self, store, client, monkeypatch):
        """The next session's pool went to a dead conversation's leftovers.

        A supervisor working any campaign takes the oldest task first, and a
        campaign left open by a session that ended is always the oldest.
        """
        abandoned = _campaign(client, goal="left open")["campaign_id"]
        live = _campaign(client, goal="being conducted")["campaign_id"]
        _task(client, abandoned)
        wanted = _task(client, live)
        assert client.post(f"/campaigns/{abandoned}/conducted").status_code == 200
        monkeypatch.setattr(store.time, "time", lambda real=store.time.time: real() + store.CONDUCTED_WITHIN_SECONDS + 1)
        assert client.post(f"/campaigns/{live}/conducted").json()["conducted_at"] is not None

        assert _claim(client, "s0").json()["task_id"] == wanted["task_id"]
        assert _claim(client, "s1").status_code == 204
        # Still there for a supervisor sent to that campaign by name.
        assert _claim(client, "s2", campaign_id=abandoned).status_code == 200

    def test_a_campaign_that_never_had_a_conductor_is_claimable(self, client):
        cid = _campaign(client)["campaign_id"]
        _task(client, cid)
        assert _claim(client, "s0").status_code == 200

    def test_heartbeat_extends_the_lease(self, client):
        cid = _campaign(client)["campaign_id"]
        _task(client, cid)
        task = _claim(client, "s0", lease_seconds=5).json()

        extended = client.post(
            f"/tasks/{task['task_id']}/heartbeat",
            json={"owner": "s0", "lease_seconds": 600},
        ).json()

        assert extended["lease_expires_at"] > task["lease_expires_at"]

    def test_heartbeat_from_a_stranger_is_refused(self, client):
        cid = _campaign(client)["campaign_id"]
        _task(client, cid)
        task = _claim(client, "s0").json()

        response = client.post(
            f"/tasks/{task['task_id']}/heartbeat", json={"owner": "s1"}
        )
        assert response.status_code == 409
        assert "held by s0" in response.json()["detail"]

    def test_an_expired_lease_is_requeued_for_someone_else(self, client):
        """A supervisor that is OOM-killed has to give its task back."""
        cid = _campaign(client)["campaign_id"]
        _task(client, cid)
        task = _claim(client, "doomed", lease_seconds=0.01).json()
        time.sleep(0.05)

        reclaimed = _claim(client, "survivor")
        assert reclaimed.status_code == 200
        assert reclaimed.json()["task_id"] == task["task_id"]
        assert reclaimed.json()["lease_owner"] == "survivor"
        assert reclaimed.json()["attempts"] == 2

    def test_a_task_that_keeps_losing_its_lease_is_abandoned(self, client):
        """Stops a poison task cycling through every supervisor forever."""
        cid = _campaign(client)["campaign_id"]
        _task(client, cid)

        for owner in ("s0", "s1"):
            assert _claim(client, owner, lease_seconds=0.01).status_code == 200
            time.sleep(0.05)

        assert _claim(client, "s2").status_code == 204
        task = client.get(f"/campaigns/{cid}/tasks").json()[0]
        assert task["state"] == "abandoned"
        assert "lease expired" in task["error"]

    def test_a_task_can_be_cancelled_out_from_under_its_holder(self, client):
        """The conductor overriding a supervisor, not a supervisor reporting.

        So this one is deliberately not owner-checked -- the holder finds out
        on its next heartbeat, which already treats a lost lease as "stop and
        do not post a result".
        """
        cid = _campaign(client)["campaign_id"]
        _task(client, cid)
        task = _claim(client, "s0").json()

        cancelled = client.post(f"/tasks/{task['task_id']}/cancel").json()

        assert cancelled["state"] == "abandoned"
        assert cancelled["lease_owner"] is None
        assert "cancelled" in cancelled["error"]

        refused = client.post(
            f"/tasks/{task['task_id']}/heartbeat", json={"owner": "s0"}
        )
        assert refused.status_code == 409

    def test_a_cancelled_task_is_not_handed_out_again(self, client):
        cid = _campaign(client)["campaign_id"]
        _task(client, cid)
        task = _claim(client, "s0").json()
        client.post(f"/tasks/{task['task_id']}/cancel")

        assert _claim(client, "s1").status_code == 204

    def test_a_queued_task_can_be_cancelled_too(self, client):
        cid = _campaign(client)["campaign_id"]
        task = _task(client, cid)

        assert client.post(
            f"/tasks/{task['task_id']}/cancel"
        ).json()["state"] == "abandoned"

    def test_cancelling_a_finished_task_is_refused(self, client):
        """Its result is already in; abandoning it would lose that."""
        cid = _campaign(client)["campaign_id"]
        _task(client, cid)
        task = _claim(client, "s0").json()
        client.post(
            f"/tasks/{task['task_id']}/result",
            json={"owner": "s0", "objective_profile": {"BoltzAffinity": 0.8}},
        )

        response = client.post(f"/tasks/{task['task_id']}/cancel")

        assert response.status_code == 409
        assert "already finished" in response.json()["detail"]

    def test_cancelling_an_unknown_task_is_404(self, client):
        assert client.post("/tasks/nope/cancel").status_code == 404

    def test_posting_after_losing_the_lease_is_refused(self, client):
        """The new holder's result must not be overwritten by the old one's."""
        cid = _campaign(client)["campaign_id"]
        _task(client, cid)
        task = _claim(client, "slow", lease_seconds=0.01).json()
        time.sleep(0.05)
        _claim(client, "fast")

        response = client.post(
            f"/tasks/{task['task_id']}/result",
            json={"owner": "slow", "objective_profile": {"BoltzAffinity": 0.9, "QED": 0.8}},
        )
        assert response.status_code == 409


# ---------------------------------------------------------------------------
# Results and failures
# ---------------------------------------------------------------------------


class TestResults:

    def _done(self, client, owner, profile, **extra):
        task = _claim(client, owner).json()
        body = {"owner": owner, "objective_profile": profile}
        body.update(extra)
        response = client.post(f"/tasks/{task['task_id']}/result", json=body)
        assert response.status_code == 200, response.text
        return task["task_id"], response.json()

    def test_posting_a_result_closes_the_task(self, client):
        cid = _campaign(client)["campaign_id"]
        _task(client, cid)

        task_id, result = self._done(
            client, "s0", {"BoltzAffinity": 0.81, "QED": 0.74},
            job_id="b02764da", candidates=[{"smiles": "CCO", "score": 0.8}],
        )

        assert result["job_id"] == "b02764da"
        assert result["candidates"][0]["smiles"] == "CCO"

        task = client.get(f"/campaigns/{cid}/tasks").json()[0]
        assert task["state"] == "done"
        assert task["lease_owner"] is None
        assert client.get(f"/campaigns/{cid}/results").json()[0]["task_id"] == task_id

    def test_a_failure_is_requeued_by_default(self, client):
        cid = _campaign(client)["campaign_id"]
        _task(client, cid)
        task = _claim(client, "s0").json()

        response = client.post(
            f"/tasks/{task['task_id']}/fail",
            json={"owner": "s0", "error": "runner unreachable"},
        )
        assert response.json()["state"] == "queued"
        assert _claim(client, "s1").status_code == 200

    def test_a_failure_can_be_terminal(self, client):
        """Some failures are the task's fault and retrying just burns a slot."""
        cid = _campaign(client)["campaign_id"]
        _task(client, cid)
        task = _claim(client, "s0").json()

        response = client.post(
            f"/tasks/{task['task_id']}/fail",
            json={"owner": "s0", "error": "weights invalid", "requeue": False},
        )
        assert response.json()["state"] == "failed"
        assert _claim(client, "s1").status_code == 204

    def test_a_failure_stops_being_requeued_at_the_attempt_limit(self, client):
        cid = _campaign(client)["campaign_id"]
        _task(client, cid)

        for owner in ("s0", "s1"):
            task = _claim(client, owner).json()
            state = client.post(
                f"/tasks/{task['task_id']}/fail",
                json={"owner": owner, "error": "boom"},
            ).json()["state"]

        assert state == "failed"


class TestTaskIntent:
    """Prose on a task, for what the weight vector cannot say.

    Holding a chemotype is a choice of generative prior rather than a weight,
    so it travels as an instruction to the agent that will pick the model.
    """

    def test_an_intent_is_stored_on_the_task(self, client):
        cid = _campaign(client)["campaign_id"]
        intent = "Hold the JQ1 methyl-triazolo-benzodiazepine core fixed."

        task = _task(client, cid, intent=intent)

        assert task["spec"]["intent"] == intent

    def test_it_survives_being_claimed(self, client):
        """The supervisor reads it off the claim, not from the enqueue call."""
        cid = _campaign(client)["campaign_id"]
        _task(client, cid, intent="stay on the I-BET151 chemotype")

        claimed = _claim(client, "s0").json()

        assert claimed["spec"]["intent"] == "stay on the I-BET151 chemotype"

    def test_a_task_without_one_is_still_valid(self, client):
        cid = _campaign(client)["campaign_id"]

        task = _task(client, cid)

        assert task["spec"]["intent"] is None


class TestLineage:
    """Which earlier results led the conductor to this task.

    It always knew -- a proposal was a step away from a specific result --
    and threw it away, which left a finished campaign as a bag of runs with
    no way to see which answer prompted which question.
    """

    def test_a_task_records_what_it_was_built_on(self, client):
        cid = _campaign(client)["campaign_id"]
        first = _task(client, cid)

        second = _task(client, cid, derived_from=[first["task_id"]])

        assert second["spec"]["derived_from"] == [first["task_id"]]

    def test_the_first_tasks_are_built_on_nothing(self, client):
        cid = _campaign(client)["campaign_id"]

        assert _task(client, cid)["spec"]["derived_from"] == []

    def test_a_task_can_come_from_several(self, client):
        """A conductor combining two results is the interesting case."""
        cid = _campaign(client)["campaign_id"]
        a, b = _task(client, cid), _task(client, cid)

        third = _task(client, cid, derived_from=[a["task_id"], b["task_id"]])

        assert third["spec"]["derived_from"] == [a["task_id"], b["task_id"]]

    def test_an_edge_to_a_task_that_does_not_exist_is_refused(self, client):
        """Drawn, believed and wrong is worse than absent."""
        cid = _campaign(client)["campaign_id"]

        response = client.post(
            f"/campaigns/{cid}/tasks", json={"derived_from": ["nosuchid"]}
        )

        assert response.status_code == 400
        assert "nosuchid" in response.text

    def test_an_edge_into_another_campaign_is_refused(self, client):
        """Two campaigns are two experiments, not one graph."""
        mine = _campaign(client)["campaign_id"]
        theirs = _campaign(client)["campaign_id"]
        elsewhere = _task(client, theirs)

        response = client.post(
            f"/campaigns/{mine}/tasks",
            json={"derived_from": [elsewhere["task_id"]]},
        )

        assert response.status_code == 400

    def test_the_rationale_is_kept_for_the_person_watching(self, client):
        cid = _campaign(client)["campaign_id"]

        task = _task(client, cid, rationale="the last two both lost potency")

        assert task["spec"]["rationale"] == "the last two both lost potency"


class TestWhichConversationOwnsACampaign:
    """A campaign belongs to the conversation whose conductor runs it.

    Recorded after the fact because the two facts live apart: the tool that
    opens a campaign runs in a process shared by every conversation, and
    only the chat gateway knows which one asked.
    """

    def test_a_campaign_starts_belonging_to_nobody(self, client):
        assert _campaign(client)["session_id"] is None

    def test_a_conversation_can_claim_one(self, client):
        cid = _campaign(client)["campaign_id"]

        response = client.post(f"/campaigns/{cid}/session", json={"session_id": "s-1"})

        assert response.status_code == 200
        assert response.json()["session_id"] == "s-1"

    def test_claiming_twice_is_not_an_error(self, client):
        """The gateway re-attaches on every campaign tool call."""
        cid = _campaign(client)["campaign_id"]
        client.post(f"/campaigns/{cid}/session", json={"session_id": "s-1"})

        response = client.post(f"/campaigns/{cid}/session", json={"session_id": "s-1"})

        assert response.status_code == 200

    def test_another_conversation_cannot_take_it(self, client):
        """Moving one is a mistake, not an edit."""
        cid = _campaign(client)["campaign_id"]
        client.post(f"/campaigns/{cid}/session", json={"session_id": "s-1"})

        response = client.post(f"/campaigns/{cid}/session", json={"session_id": "s-2"})

        assert response.status_code == 409

    def test_a_conversation_sees_only_its_own(self, client):
        mine = _campaign(client, goal="mine")["campaign_id"]
        _campaign(client, goal="someone else's")
        client.post(f"/campaigns/{mine}/session", json={"session_id": "s-1"})

        listed = client.get("/campaigns?session_id=s-1").json()

        assert [c["campaign_id"] for c in listed] == [mine]

    def test_unclaimed_campaigns_are_not_everybodys(self, client):
        """A filtered list answers 'which are mine', not 'which are free'."""
        _campaign(client)

        assert client.get("/campaigns?session_id=s-1").json() == []

    def test_without_a_filter_every_campaign_is_listed(self, client):
        _campaign(client)
        _campaign(client)

        assert len(client.get("/campaigns").json()) == 2


class TestWhichRunATaskStarted:
    """The job a task is running, while it is running it.

    results.job_id has always had it afterwards, which is no use to somebody
    asking what a supervisor is doing now -- the whole of a long run was the
    part nobody could look inside.
    """

    def test_a_task_starts_with_no_run(self, client):
        cid = _campaign(client)["campaign_id"]

        assert _task(client, cid)["job_id"] is None

    def test_the_beat_reports_the_run(self, client):
        cid = _campaign(client)["campaign_id"]
        _task(client, cid)
        claimed = _claim(client, "s0").json()

        beat = client.post(
            f"/tasks/{claimed['task_id']}/heartbeat",
            json={"owner": "s0", "job_id": "job-42"},
        )

        assert beat.status_code == 200
        assert beat.json()["job_id"] == "job-42"

    def test_it_shows_up_on_the_campaign_task_list(self, client):
        """Which is where anyone watching the campaign would look."""
        cid = _campaign(client)["campaign_id"]
        _task(client, cid)
        claimed = _claim(client, "s0").json()
        client.post(
            f"/tasks/{claimed['task_id']}/heartbeat",
            json={"owner": "s0", "job_id": "job-42"},
        )

        tasks = client.get(f"/campaigns/{cid}/tasks").json()

        assert tasks[0]["job_id"] == "job-42"

    def test_a_beat_without_one_leaves_it_alone(self, client):
        """A run that started is not un-started by the next quiet beat."""
        cid = _campaign(client)["campaign_id"]
        _task(client, cid)
        claimed = _claim(client, "s0").json()
        client.post(
            f"/tasks/{claimed['task_id']}/heartbeat",
            json={"owner": "s0", "job_id": "job-42"},
        )

        beat = client.post(
            f"/tasks/{claimed['task_id']}/heartbeat", json={"owner": "s0"}
        )

        assert beat.json()["job_id"] == "job-42"

    def test_a_second_run_does_not_repoint_the_first(self, client):
        """A retry runs a new job; whoever opened the old one still means it."""
        cid = _campaign(client)["campaign_id"]
        _task(client, cid)
        claimed = _claim(client, "s0").json()
        client.post(
            f"/tasks/{claimed['task_id']}/heartbeat",
            json={"owner": "s0", "job_id": "job-42"},
        )

        beat = client.post(
            f"/tasks/{claimed['task_id']}/heartbeat",
            json={"owner": "s0", "job_id": "job-99"},
        )

        assert beat.json()["job_id"] == "job-42"


class TestSupervisorPresence:
    """Which supervisors exist, as opposed to which hold a task.

    Claiming was the only thing that put a supervisor in the ledger, so an
    idle one -- the only kind a conductor can actually dispatch to -- was
    indistinguishable from one that never started.
    """

    def _present(self, client, owner, state="idle", **extra):
        body = {"state": state}
        body.update(extra)
        response = client.post(f"/supervisors/{owner}/presence", json=body)
        assert response.status_code == 200, response.text
        return response.json()

    def test_an_idle_supervisor_is_visible(self, client):
        self._present(client, "s0")

        pool = client.get("/supervisors").json()

        assert pool["n_alive"] == 1
        assert pool["idle"] == 1
        assert pool["supervisors"][0]["owner"] == "s0"

    def test_the_pool_separates_free_from_working(self, client):
        self._present(client, "s0", state="idle")
        self._present(client, "s1", state="busy", task_id="t1")
        self._present(client, "s2", state="busy", task_id="t2")

        pool = client.get("/supervisors").json()

        assert (pool["idle"], pool["busy"], pool["n_alive"]) == (1, 2, 3)

    def test_a_busy_supervisor_says_what_it_holds(self, client):
        self._present(
            client, "s0", state="busy", task_id="t1",
            tasks_completed=3, tasks_failed=1, detail="running task",
        )

        held = client.get("/supervisors").json()["supervisors"][0]

        assert held["task_id"] == "t1"
        assert (held["tasks_completed"], held["tasks_failed"]) == (3, 1)
        assert held["detail"] == "running task"

    def test_a_supervisor_that_stopped_reporting_is_forgotten(self, client):
        """A stale row read as capacity is what makes a dispatch go nowhere."""
        self._present(client, "s0", ttl_seconds=0.01)
        time.sleep(0.05)

        pool = client.get("/supervisors").json()

        assert pool["n_alive"] == 0

    def test_leaving_on_purpose_is_immediate(self, client):
        """A pool drained on command should not look crashed for two minutes."""
        self._present(client, "s0")

        client.delete("/supervisors/s0")

        assert client.get("/supervisors").json()["n_alive"] == 0

    def test_reporting_again_refreshes_rather_than_duplicates(self, client):
        self._present(client, "s0", state="idle")
        self._present(client, "s0", state="busy", task_id="t1")

        pool = client.get("/supervisors").json()

        assert pool["n_alive"] == 1
        assert pool["supervisors"][0]["state"] == "busy"

    def test_how_long_ago_it_reported_is_what_dates_the_state(self, client):
        self._present(client, "s0")

        held = client.get("/supervisors").json()["supervisors"][0]

        assert held["last_seen_seconds_ago"] < 5

    def test_an_unpinned_supervisor_counts_towards_every_campaign(self, client):
        """It claims from any campaign, so it is capacity for all of them."""
        cid = _campaign(client)["campaign_id"]
        self._present(client, "anyone", campaign_id=None)
        self._present(client, "pinned", campaign_id=cid)
        self._present(client, "elsewhere", campaign_id="other123")

        pool = client.get(f"/supervisors?campaign_id={cid}").json()

        assert {s["owner"] for s in pool["supervisors"]} == {"anyone", "pinned"}

    def test_an_empty_pool_is_not_an_error(self, client):
        pool = client.get("/supervisors").json()

        assert pool == {
            "supervisors": [], "n_alive": 0, "idle": 0, "busy": 0, "by_state": {},
        }

    def test_a_starting_supervisor_is_alive_but_not_idle(self, client):
        """A pod still coming up is not somewhere to send a task."""
        self._present(client, "s0", state="starting")

        pool = client.get("/supervisors").json()

        assert (pool["n_alive"], pool["idle"]) == (1, 0)
        assert pool["by_state"] == {"starting": 1}


class TestCampaignOverview:
    """``/campaigns`` as a triage list rather than bare rows.

    "What ran most recently and did it produce anything" took one call per
    campaign to answer, which for a conductor meant either many calls or a
    guess.
    """

    def test_a_campaign_carries_its_task_and_result_counts(self, client):
        cid = _campaign(client)["campaign_id"]
        for _ in range(3):
            _task(client, cid)
        task = _claim(client, "s0").json()
        client.post(
            f"/tasks/{task['task_id']}/result",
            json={"owner": "s0", "objective_profile": {"BoltzAffinity": 0.9}},
        )

        overview = client.get("/campaigns").json()[0]

        assert overview["task_counts"] == {"done": 1, "queued": 2}
        assert overview["n_tasks"] == 3
        assert overview["n_results"] == 1

    def test_the_latest_activity_is_reported_not_just_the_creation(self, client):
        cid = _campaign(client)["campaign_id"]
        _task(client, cid)
        task = _claim(client, "s0").json()
        client.post(
            f"/tasks/{task['task_id']}/result",
            json={"owner": "s0", "objective_profile": {"BoltzAffinity": 0.9}},
        )

        overview = client.get("/campaigns").json()[0]

        assert overview["last_activity_at"] >= overview["created_at"]
        assert overview["last_activity_seconds_ago"] < 10

    def test_an_untouched_campaign_dates_from_its_creation(self, client):
        _campaign(client)

        overview = client.get("/campaigns").json()[0]

        assert overview["last_activity_at"] == overview["created_at"]
        assert overview["n_tasks"] == 0

    def test_the_phase_follows_the_clock_not_the_stored_state(self, client):
        """A conductor that died past its deadline still reads as draining."""
        _campaign(client, dispatch_seconds=0.01, grace_seconds=600)
        time.sleep(0.05)

        overview = client.get("/campaigns").json()[0]

        assert overview["state"] == "dispatching"
        assert overview["phase"] == "draining"

    def test_campaigns_are_newest_first(self, client):
        first = _campaign(client, goal="older")["campaign_id"]
        second = _campaign(client, goal="newer")["campaign_id"]

        order = [c["campaign_id"] for c in client.get("/campaigns").json()]

        assert order == [second, first]

    def test_counts_do_not_bleed_between_campaigns(self, client):
        busy = _campaign(client)["campaign_id"]
        quiet = _campaign(client)["campaign_id"]
        for _ in range(2):
            _task(client, busy)

        overview = {c["campaign_id"]: c for c in client.get("/campaigns").json()}

        assert overview[busy]["n_tasks"] == 2
        assert overview[quiet]["n_tasks"] == 0


class TestResultsAreNarrowedForAContextWindow:
    """A result's caller is an LLM, and a scorer emits far more than the axes.

    Boltz reports ten per-molecule diagnostics beside the one objective they
    explain, which made a 25-molecule result ~22k characters. Two of those in
    a turn overflowed the conductor's 32768-token context, failing with
    "max_tokens ... got -2623".
    """

    # The column set a real Boltz + QED candidate carries, because the ratio
    # these tests assert on is a property of that shape.
    DIAGNOSTIC_CANDIDATE = {
        "smiles": "CCNc1cc(C(=O)NC(Cc2ccccc2)C(O)CNC2CCN3CCCCC3C2)cc(N2CCCCS2(=O)=O)c1",
        "total_score": 5.71,
        "components": {
            "BoltzAffinity": {"transformed": 1.0, "raw": 0.0092},
            "QED": {"transformed": 0.8, "raw": 0.8},
            "ligand_iptm (BoltzAffinity)": {"transformed": 0.9049},
            "confidence_score (BoltzAffinity)": {"transformed": 0.8692},
            "skip_reason (BoltzAffinity)": {"transformed": None},
            "affinity_pred_value (BoltzAffinity)": {"transformed": 0.0092},
            "affinity_pred_value1 (BoltzAffinity)": {"transformed": -0.8188},
            "affinity_pred_value2 (BoltzAffinity)": {"transformed": 0.8372},
            "affinity_probability_binary (BoltzAffinity)": {"transformed": 0.2148},
            "affinity_probability_binary1 (BoltzAffinity)": {"transformed": 0.1983},
            "affinity_probability_binary2 (BoltzAffinity)": {"transformed": 0.2313},
            "std (QED)": {"transformed": 0.1824},
        },
    }

    def _result(self, client, **extra):
        cid = _campaign(client)["campaign_id"]
        _task(client, cid)
        task = _claim(client, "s0").json()
        body = {
            "owner": "s0",
            "objective_profile": {
                "BoltzAffinity": 0.9, "QED": 0.8,
                "confidence_score (BoltzAffinity)": 0.87,
            },
            "candidates": [dict(self.DIAGNOSTIC_CANDIDATE)],
        }
        body.update(extra)
        client.post(f"/tasks/{task['task_id']}/result", json=body)
        return cid

    def test_the_redundant_half_of_a_scorers_columns_is_dropped(self, client):
        cid = self._result(client)

        candidate = client.get(f"/campaigns/{cid}/results").json()[0]["candidates"][0]

        assert "affinity_pred_value1 (BoltzAffinity)" not in candidate["components"]
        assert "std (QED)" not in candidate["components"]
        assert candidate["components"]["BoltzAffinity"]["raw"] == 0.0092
        assert candidate["smiles"] == self.DIAGNOSTIC_CANDIDATE["smiles"]
        assert candidate["total_score"] == 5.71

    def test_the_signals_that_qualify_an_objective_are_kept(self, client):
        """An affinity with no confidence beside it gets narrated, not read.

        Asked for the best results' molecules with these dropped, the conductor
        asserted a quality gate it could not see, inventing PoseBusters
        numbers for a campaign that never ran PoseBusters.
        """
        cid = self._result(client)

        candidate = client.get(f"/campaigns/{cid}/results").json()[0]["candidates"][0]

        assert candidate["components"]["confidence_score (BoltzAffinity)"] == {
            "transformed": 0.8692
        }
        assert candidate["components"]["ligand_iptm (BoltzAffinity)"] == {
            "transformed": 0.9049
        }

    def test_a_qualifier_with_nothing_to_report_is_left_out(self, client):
        """``skip_reason`` is null for every molecule the scorer did score."""
        cid = self._result(client)

        candidate = client.get(f"/campaigns/{cid}/results").json()[0]["candidates"][0]

        assert "skip_reason (BoltzAffinity)" not in candidate["components"]

    def test_a_skipped_molecule_says_so(self, client):
        """The one case where the objective's value means nothing."""
        cid = self._result(client, candidates=[{
            "smiles": "CCO",
            "components": {
                "BoltzAffinity": {"transformed": 0.0},
                "skip_reason (BoltzAffinity)": {"transformed": "fold failed"},
            },
        }])

        candidate = client.get(f"/campaigns/{cid}/results").json()[0]["candidates"][0]

        assert candidate["components"]["skip_reason (BoltzAffinity)"] == {
            "transformed": "fold failed"
        }

    def test_the_objective_profile_is_narrowed_too(self, client):
        cid = self._result(client)

        result = client.get(f"/campaigns/{cid}/results").json()[0]

        assert set(result["objective_profile"]) == {
            "BoltzAffinity", "QED", "confidence_score (BoltzAffinity)"
        }

    def test_a_result_says_where_the_diagnostics_are(self, client):
        cid = self._result(client)

        result = client.get(f"/campaigns/{cid}/results").json()[0]

        assert "read_results" in result["detail_hint"]

    def test_full_returns_the_stored_record_untouched(self, client):
        """The ledger is not lossy; only its default read is narrow."""
        cid = self._result(client)

        result = client.get(f"/campaigns/{cid}/results?full=1").json()[0]

        assert result["objective_profile"]["confidence_score (BoltzAffinity)"] == 0.87
        assert (
            result["candidates"][0]["components"]
            == self.DIAGNOSTIC_CANDIDATE["components"]
        )
        assert "detail_hint" not in result

    def test_narrowing_keeps_the_run_a_molecule_came_from(self, client):
        """How the UI finds its complex; a molecule posted without one has none."""
        cid = self._result(client, candidates=[{"smiles": "CCO", "job_id": "a1b2c3d4"}, {"smiles": "CCN"}])

        with_run, without = client.get(f"/campaigns/{cid}/results").json()[0]["candidates"]

        assert with_run["job_id"] == "a1b2c3d4"
        assert "job_id" not in without

    def test_narrowing_survives_a_candidate_with_no_components(self, client):
        cid = self._result(client, candidates=[{"smiles": "CCO"}])

        candidate = client.get(f"/campaigns/{cid}/results").json()[0]["candidates"][0]

        assert candidate["components"] == {}
        assert candidate["total_score"] is None

    def test_narrowing_is_worth_doing(self, client):
        """The point of all of this, asserted rather than assumed."""
        cid = self._result(
            client, candidates=[dict(self.DIAGNOSTIC_CANDIDATE)] * 25
        )

        narrow = client.get(f"/campaigns/{cid}/results").text
        full = client.get(f"/campaigns/{cid}/results?full=1").text

        assert len(narrow) < len(full) / 2


# ---------------------------------------------------------------------------
# Events and status
# ---------------------------------------------------------------------------


class TestEventsAndStatus:

    def test_events_record_the_task_lifecycle(self, client):
        cid = _campaign(client)["campaign_id"]
        _task(client, cid)
        task = _claim(client, "s0").json()
        client.post(
            f"/tasks/{task['task_id']}/result",
            json={"owner": "s0", "objective_profile": {"BoltzAffinity": 0.8, "QED": 0.7}},
        )

        kinds = [e["kind"] for e in client.get(f"/campaigns/{cid}/events").json()["events"]]
        assert kinds == ["campaign_created", "task_enqueued", "task_claimed", "task_done"]

    def test_events_can_be_polled_incrementally(self, client):
        """The conductor's completion feed: poll with the last id it saw."""
        cid = _campaign(client)["campaign_id"]
        first = client.get(f"/campaigns/{cid}/events").json()

        _task(client, cid)
        second = client.get(
            f"/campaigns/{cid}/events", params={"since": first["last_event_id"]}
        ).json()

        assert [e["kind"] for e in second["events"]] == ["task_enqueued"]
        assert second["last_event_id"] > first["last_event_id"]

    def test_polling_with_nothing_new_keeps_the_cursor(self, client):
        cid = _campaign(client)["campaign_id"]
        body = client.get(f"/campaigns/{cid}/events").json()
        again = client.get(
            f"/campaigns/{cid}/events", params={"since": body["last_event_id"]}
        ).json()

        assert again["events"] == []
        assert again["last_event_id"] == body["last_event_id"]

    def test_status_reports_counts_in_flight_work_and_results(self, client):
        cid = _campaign(client)["campaign_id"]
        _task(client, cid, weight_overrides={"QED": 0.5})
        _task(client, cid)
        running = _claim(client, "supervisor-0").json()

        status = client.get(f"/campaigns/{cid}/status").json()

        assert status["task_counts"] == {"claimed": 1, "queued": 1}
        assert status["n_results"] == 0
        assert len(status["in_flight"]) == 1
        assert status["in_flight"][0]["owner"] == "supervisor-0"
        assert status["in_flight"][0]["task_id"] == running["task_id"]
        assert status["in_flight"][0]["spec"]["weight_overrides"] == {"QED": 0.5}
        assert status["in_flight"][0]["lease_expires_in"] > 0

    def test_status_sweeps_dead_supervisors_out_of_the_in_flight_list(self, client):
        """In-flight should mean alive, not "claimed by something, once"."""
        cid = _campaign(client)["campaign_id"]
        _task(client, cid)
        _claim(client, "doomed", lease_seconds=0.01)
        time.sleep(0.05)

        status = client.get(f"/campaigns/{cid}/status").json()

        assert status["in_flight"] == []
        assert status["task_counts"] == {"queued": 1}

    def test_status_window_phase_follows_the_clock_not_the_state(self, client):
        """A conductor that missed its own deadline should still see the truth."""
        cid = _campaign(client, dispatch_seconds=0.01, grace_seconds=600)["campaign_id"]
        time.sleep(0.2)

        status = client.get(f"/campaigns/{cid}/status").json()

        assert status["campaign"]["state"] == "dispatching"
        assert status["window"]["phase"] == "draining"
        assert status["window"]["dispatch_remaining"] < 0

    def test_status_window_phase_without_deadlines(self, client):
        cid = _campaign(client)["campaign_id"]
        status = client.get(f"/campaigns/{cid}/status").json()

        assert status["window"]["phase"] == "dispatching"
        assert status["window"]["dispatch_remaining"] is None


# ---------------------------------------------------------------------------
# Durability
# ---------------------------------------------------------------------------


class TestDurability:

    def test_the_ledger_survives_a_restart(self, store, tmp_path):
        """The conductor rebuilds its loop from the ledger, so it has to persist."""
        with TestClient(store.app) as first:
            cid = _campaign(first)["campaign_id"]
            _task(first, cid)
            task = _claim(first, "s0").json()
            first.post(
                f"/tasks/{task['task_id']}/result",
                json={
                    "owner": "s0",
                    "objective_profile": {"BoltzAffinity": 0.8, "QED": 0.7},
                },
            )

        with TestClient(store.app) as second:
            assert second.get(f"/campaigns/{cid}").status_code == 200
            results = second.get(f"/campaigns/{cid}/results").json()
            assert results[0]["objective_profile"]["BoltzAffinity"] == 0.8

    def test_the_spec_round_trips_as_json_not_a_repr(self, store, client):
        """Stored specs are read back by other processes, so they must be JSON."""
        cid = _campaign(client)["campaign_id"]
        _task(client, cid, transform_overrides={"QED": {"type": "sigmoid", "k": 0.5}})

        with store._read() as conn:
            raw = conn.execute("SELECT spec FROM tasks").fetchone()["spec"]

        assert json.loads(raw)["transform_overrides"]["QED"]["k"] == 0.5

    def test_a_ledger_from_before_a_column_existed_still_opens(self, store, tmp_path):
        """CREATE TABLE IF NOT EXISTS does nothing to a table that exists.

        So a store upgraded over a ledger with campaigns in it would never
        get the new columns, and every query naming one would fail -- on the
        one service whose job is to outlive the things around it. Built from
        the previous release's schema rather than by dropping columns from
        this one, because that is the situation being tested.
        """
        old = tmp_path / "before.db"
        conn = sqlite3.connect(old)
        conn.executescript(
            """
            CREATE TABLE campaigns (
                campaign_id TEXT PRIMARY KEY, goal TEXT NOT NULL,
                base_ttp_id TEXT NOT NULL, objectives TEXT NOT NULL,
                concurrency INTEGER NOT NULL, state TEXT NOT NULL,
                created_at TEXT NOT NULL, dispatch_until REAL,
                hard_stop_at REAL, finalized_at TEXT
            );
            CREATE TABLE tasks (
                task_id TEXT PRIMARY KEY,
                campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
                spec TEXT NOT NULL, state TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL,
                lease_owner TEXT, lease_expires_at REAL, claimed_at TEXT,
                finished_at TEXT, error TEXT
            );
            INSERT INTO campaigns VALUES
                ('old1', 'an older goal', 'brd4-bd1-lead-opt', '["QED"]', 1,
                 'dispatching', '2026-01-01T00:00:00+00:00', NULL, NULL, NULL);
            INSERT INTO tasks VALUES
                ('t1', 'old1', '{}', 'queued', 0,
                 '2026-01-01T00:00:00+00:00', NULL, NULL, NULL, NULL, NULL);
            """
        )
        conn.commit()
        conn.close()

        store.DB_PATH = old
        store.init_db()

        with TestClient(store.app) as client:
            assert client.get("/campaigns/old1").json()["session_id"] is None
            assert client.get("/campaigns/old1/tasks").json()[0]["job_id"] is None
            # And the new columns are writable, not merely present.
            claimed = _claim(client, "s0").json()
            beat = client.post(
                f"/tasks/{claimed['task_id']}/heartbeat",
                json={"owner": "s0", "job_id": "job-7"},
            )
            assert beat.json()["job_id"] == "job-7"
            attached = client.post(
                "/campaigns/old1/session", json={"session_id": "s-1"}
            )
            assert attached.json()["session_id"] == "s-1"


# ---------------------------------------------------------------------------
# Follow-ups a sub-agent proposed
# ---------------------------------------------------------------------------


def _reported(client, campaign_id, owner="s0", follow_ups=(), **overrides):
    """One task taken and reported on, with whatever it proposed next."""

    task = _task(client, campaign_id, kind="research",
                 intent="Map the BD1/BD2 selectivity handles", **overrides)
    claimed = _claim(client, owner).json()
    posted = client.post(
        f"/tasks/{claimed['task_id']}/result",
        json={"owner": owner, "summary": "two handles", "report": "## Report\n\nfound them",
              "follow_ups": list(follow_ups)},
    )
    assert posted.status_code == 200, posted.text
    return task


class TestProposals:
    """A sub-agent proposes; the conductor disposes."""

    FOLLOW_UP = {
        "kind": "design",
        "title": "Hold the triazole",
        "intent": "Run RL holding the methyl-triazole and varying the fused ring.",
    }

    def test_what_a_report_proposes_waits_for_the_conductor(self, client):
        campaign = _campaign(client)
        _reported(client, campaign["campaign_id"], follow_ups=[self.FOLLOW_UP])

        open_ones = client.get(f"/campaigns/{campaign['campaign_id']}/proposals").json()

        assert len(open_ones) == 1
        assert open_ones[0]["spec"]["title"] == "Hold the triazole"
        assert open_ones[0]["state"] == "open"
        # Proposed, not queued: nothing may claim it yet.
        assert _claim(client, "s1").status_code == 204

    def test_a_proposal_taken_becomes_a_task_derived_from_its_report(self, client):
        campaign = _campaign(client)
        source = _reported(client, campaign["campaign_id"], follow_ups=[self.FOLLOW_UP])
        proposal = client.get(f"/campaigns/{campaign['campaign_id']}/proposals").json()[0]

        taken = client.post(
            f"/campaigns/{campaign['campaign_id']}/proposals/take",
            json={"proposal_ids": [proposal["proposal_id"]]},
        )

        assert taken.status_code == 201, taken.text
        task = taken.json()[0]
        assert task["spec"]["intent"] == self.FOLLOW_UP["intent"]
        assert task["spec"]["derived_from"] == [source["task_id"]]
        assert _claim(client, "s1").json()["task_id"] == task["task_id"]
        assert client.get(f"/campaigns/{campaign['campaign_id']}/proposals").json() == []

    def test_a_proposal_is_taken_once(self, client):
        campaign = _campaign(client)
        _reported(client, campaign["campaign_id"], follow_ups=[self.FOLLOW_UP])
        proposal = client.get(f"/campaigns/{campaign['campaign_id']}/proposals").json()[0]
        body = {"proposal_ids": [proposal["proposal_id"]]}
        client.post(f"/campaigns/{campaign['campaign_id']}/proposals/take", json=body)

        again = client.post(f"/campaigns/{campaign['campaign_id']}/proposals/take", json=body)

        assert again.status_code == 409

    def test_a_proposal_dropped_is_not_offered_again(self, client):
        campaign = _campaign(client)
        _reported(client, campaign["campaign_id"], follow_ups=[self.FOLLOW_UP])
        proposal = client.get(f"/campaigns/{campaign['campaign_id']}/proposals").json()[0]

        client.post(
            f"/campaigns/{campaign['campaign_id']}/proposals/drop",
            json={"proposal_ids": [proposal["proposal_id"]]},
        )

        assert client.get(f"/campaigns/{campaign['campaign_id']}/proposals").json() == []
        dropped = client.get(
            f"/campaigns/{campaign['campaign_id']}/proposals?state=dropped"
        ).json()
        assert len(dropped) == 1

    def test_a_follow_up_continues_its_tasks_workstream(self, client):
        campaign = _campaign(client)
        _reported(client, campaign["campaign_id"], workstream="selectivity",
                  follow_ups=[self.FOLLOW_UP])

        proposal = client.get(f"/campaigns/{campaign['campaign_id']}/proposals").json()[0]

        assert proposal["spec"]["workstream"] == "selectivity"

    def test_the_report_event_says_how_many_were_proposed(self, client):
        """So the conductor is told to look without reading every report."""
        campaign = _campaign(client)
        _reported(client, campaign["campaign_id"], follow_ups=[self.FOLLOW_UP])

        events = client.get(f"/campaigns/{campaign['campaign_id']}/events").json()["events"]

        done = [e for e in events if e["kind"] == "task_done"]
        assert done[0]["payload"]["n_follow_ups"] == 1

    def test_a_report_proposing_nothing_leaves_no_proposals(self, client):
        campaign = _campaign(client)
        _reported(client, campaign["campaign_id"])

        assert client.get(f"/campaigns/{campaign['campaign_id']}/proposals").json() == []


# ---------------------------------------------------------------------------
# Keeping the team supplied
# ---------------------------------------------------------------------------


class TestReadyQueue:

    def test_a_batch_is_queued_whole(self, client):
        campaign = _campaign(client)

        queued = client.post(
            f"/campaigns/{campaign['campaign_id']}/tasks/batch",
            json={"tasks": [{"intent": f"question {i}", "kind": "question"} for i in range(3)]},
        )

        assert queued.status_code == 201, queued.text
        assert len(queued.json()) == 3
        assert len(client.get(f"/campaigns/{campaign['campaign_id']}/tasks").json()) == 3

    def test_a_batch_with_one_bad_task_queues_none(self, client):
        """Half a batch would leave the conductor guessing which half."""
        campaign = _campaign(client)

        refused = client.post(
            f"/campaigns/{campaign['campaign_id']}/tasks/batch",
            json={"tasks": [
                {"intent": "fine", "kind": "question"},
                {"intent": "derived from nothing", "kind": "question", "derived_from": ["nope"]},
            ]},
        )

        assert refused.status_code == 400
        assert client.get(f"/campaigns/{campaign['campaign_id']}/tasks").json() == []

    def test_higher_priority_is_claimed_first_then_oldest(self, client):
        campaign = _campaign(client)
        first = _task(client, campaign["campaign_id"])
        urgent = _task(client, campaign["campaign_id"], priority=2)
        filler = _task(client, campaign["campaign_id"], priority=-1)
        second = _task(client, campaign["campaign_id"])

        order = [_claim(client, f"s{i}").json()["task_id"] for i in range(4)]

        assert order == [urgent["task_id"], first["task_id"], second["task_id"], filler["task_id"]]

    def test_who_ran_a_task_is_kept_after_it_is_done(self, client):
        campaign = _campaign(client)
        task = _reported(client, campaign["campaign_id"], owner="s7")

        done = [t for t in client.get(f"/campaigns/{campaign['campaign_id']}/tasks").json()
                if t["task_id"] == task["task_id"]][0]

        assert done["lease_owner"] is None
        assert done["claimed_by"] == "s7"

    def test_the_status_says_how_deep_to_keep_the_queue(self, client):
        campaign = _campaign(client, concurrency=5)

        status = client.get(f"/campaigns/{campaign['campaign_id']}/status").json()

        assert status["backlog_target"] == 3


# ---------------------------------------------------------------------------
# The plan and how the work covers it
# ---------------------------------------------------------------------------


PLAN = {"workstreams": [
    {"id": "selectivity", "name": "BD1/BD2 selectivity", "question": "Which handle separates BD1 from BD2?"},
    {"id": "admet", "name": "ADMET liabilities", "question": "What sinks the series in vivo?"},
]}


class TestPlan:

    def test_a_plan_is_kept_on_the_campaign(self, client):
        campaign = _campaign(client)

        put = client.put(f"/campaigns/{campaign['campaign_id']}/plan", json=PLAN)

        assert put.status_code == 200, put.text
        stored = client.get(f"/campaigns/{campaign['campaign_id']}").json()["plan"]
        assert [s["id"] for s in stored["workstreams"]] == ["selectivity", "admet"]
        assert stored["workstreams"][0]["status"] == "open"

    def test_workstream_ids_must_be_unique(self, client):
        campaign = _campaign(client)
        twice = {"workstreams": [PLAN["workstreams"][0], PLAN["workstreams"][0]]}

        assert client.put(f"/campaigns/{campaign['campaign_id']}/plan", json=twice).status_code == 400

    def test_coverage_shows_a_workstream_nobody_is_working_as_a_gap(self, client):
        campaign = _campaign(client)
        client.put(f"/campaigns/{campaign['campaign_id']}/plan", json=PLAN)
        _task(client, campaign["campaign_id"], workstream="selectivity")

        cover = client.get(f"/campaigns/{campaign['campaign_id']}/coverage").json()

        by_id = {s["id"]: s for s in cover["workstreams"]}
        assert by_id["selectivity"]["active"] == 1
        assert cover["gaps"] == ["admet"]

    def test_coverage_carries_the_latest_finding_and_proposals(self, client):
        campaign = _campaign(client)
        client.put(f"/campaigns/{campaign['campaign_id']}/plan", json=PLAN)
        _reported(client, campaign["campaign_id"], workstream="selectivity",
                  follow_ups=[TestProposals.FOLLOW_UP])

        cover = client.get(f"/campaigns/{campaign['campaign_id']}/coverage").json()

        selectivity = {s["id"]: s for s in cover["workstreams"]}["selectivity"]
        assert selectivity["latest"]["summary"] == "two handles"
        assert selectivity["proposals"] == 1
        assert cover["open_proposals"] == 1

    def test_work_outside_the_plan_is_counted_not_lost(self, client):
        campaign = _campaign(client)
        client.put(f"/campaigns/{campaign['campaign_id']}/plan", json=PLAN)
        _task(client, campaign["campaign_id"])

        cover = client.get(f"/campaigns/{campaign['campaign_id']}/coverage").json()

        assert cover["workstreams"][-1]["id"] is None
        assert cover["workstreams"][-1]["active"] == 1


def _finding(client, campaign_id, **overrides):
    body = {
        "branch": "selectivity",
        "claim": "The BD1 ZA channel tolerates a methyl where BD2 does not",
        "evidence": ["10.1021/jm501245q"],
        "author": "s0",
    }
    body.update(overrides)
    response = client.post(f"/campaigns/{campaign_id}/findings", json=body)
    assert response.status_code == 201, response.text
    return response.json()


class TestAtlas:
    """What the campaign knows, as opposed to what it has run."""

    def test_a_finding_is_recorded_with_its_provenance(self, client):
        cid = _campaign(client)["campaign_id"]

        finding = _finding(client, cid, tags=["BD1", "ZA-channel"], confidence="high")

        assert finding["branch"] == "selectivity"
        assert finding["evidence"] == ["10.1021/jm501245q"]
        assert finding["tags"] == ["BD1", "ZA-channel"]
        assert finding["author"] == "s0"
        assert finding["confidence"] == "high"
        assert finding["superseded_by"] is None

    def test_a_finding_needs_a_campaign_that_exists(self, client):
        assert client.post(
            "/campaigns/nope/findings",
            json={"branch": "b", "claim": "c", "author": "s0"},
        ).status_code == 404

    def test_findings_come_back_newest_first_for_one_branch(self, client):
        cid = _campaign(client)["campaign_id"]
        _finding(client, cid, claim="older")
        _finding(client, cid, claim="newer")
        _finding(client, cid, branch="admet", claim="elsewhere")

        listed = client.get(f"/campaigns/{cid}/findings?branch=selectivity").json()

        assert [f["claim"] for f in listed["findings"]] == ["newer", "older"]

    def test_superseding_hides_the_old_one_without_deleting_it(self, client):
        """A campaign that believed something for two hours believed it."""
        cid = _campaign(client)["campaign_id"]
        old = _finding(client, cid, claim="IC50 is 40 nM")

        new = _finding(client, cid, claim="IC50 is 400 nM", supersedes=old["finding_id"])

        current = client.get(f"/campaigns/{cid}/findings").json()["findings"]
        assert [f["claim"] for f in current] == ["IC50 is 400 nM"]
        assert new["supersedes"] == old["finding_id"]

        kept = client.get(f"/campaigns/{cid}/findings?include_superseded=true").json()
        was = {f["finding_id"]: f for f in kept["findings"]}[old["finding_id"]]
        assert was["superseded_by"] == new["finding_id"]

    def test_one_campaign_cannot_supersede_another_campaigns_finding(self, client):
        mine = _campaign(client)["campaign_id"]
        theirs = _campaign(client)["campaign_id"]
        elsewhere = _finding(client, theirs, claim="not yours")

        response = client.post(
            f"/campaigns/{mine}/findings",
            json={"branch": "b", "claim": "c", "author": "s0",
                  "supersedes": elsewhere["finding_id"]},
        )

        assert response.status_code == 404

    def test_a_note_hangs_off_the_finding_it_answers(self, client):
        cid = _campaign(client)["campaign_id"]
        finding = _finding(client, cid)

        noted = client.post(
            f"/campaigns/{cid}/findings/{finding['finding_id']}/notes",
            json={"body": "Our own run says otherwise", "stance": "disputes",
                  "author": "s1"},
        )

        assert noted.status_code == 201, noted.text
        listed = client.get(f"/campaigns/{cid}/findings").json()["findings"]
        assert listed[0]["notes"][0]["stance"] == "disputes"
        assert listed[0]["notes"][0]["author"] == "s1"

    def test_a_note_needs_a_finding_in_this_campaign(self, client):
        cid = _campaign(client)["campaign_id"]

        assert client.post(
            f"/campaigns/{cid}/findings/nope/notes",
            json={"body": "x", "author": "s1"},
        ).status_code == 404

    def test_search_wants_every_word_somewhere_in_the_finding(self, client):
        cid = _campaign(client)["campaign_id"]
        _finding(client, cid, claim="Methyl on the ZA channel costs BD2 affinity",
                 tags=["BD2"])
        _finding(client, cid, claim="Microsomal clearance is the series liability")

        hit = client.get(f"/campaigns/{cid}/findings?q=methyl+bd2").json()["findings"]
        miss = client.get(f"/campaigns/{cid}/findings?q=methyl+clearance").json()["findings"]

        assert len(hit) == 1 and hit[0]["tags"] == ["BD2"]
        assert miss == []

    def test_search_reaches_the_detail_and_the_tags(self, client):
        cid = _campaign(client)["campaign_id"]
        _finding(client, cid, claim="A handle exists", detail="Asn140 is the anchor",
                 tags=["hinge"])

        assert len(client.get(f"/campaigns/{cid}/findings?q=asn140").json()["findings"]) == 1
        assert len(client.get(f"/campaigns/{cid}/findings?tag=hinge").json()["findings"]) == 1
        assert client.get(f"/campaigns/{cid}/findings?tag=hing").json()["findings"] == []

    def test_the_overview_shows_the_branches_the_plan_named_but_nobody_filled(self, client):
        """The empty branch is the point: it is what nobody has settled."""
        cid = _campaign(client)["campaign_id"]
        client.put(f"/campaigns/{cid}/plan", json=PLAN)
        _finding(client, cid, claim="BD1 tolerates it", tags=["BD1"])

        atlas = client.get(f"/campaigns/{cid}/atlas").json()

        by_branch = {b["branch"]: b for b in atlas["branches"]}
        assert by_branch["selectivity"]["findings"] == 1
        assert by_branch["selectivity"]["latest"]["claim"] == "BD1 tolerates it"
        assert by_branch["selectivity"]["tags"] == ["BD1"]
        assert by_branch["admet"]["findings"] == 0
        assert atlas["empty_branches"] == ["admet"]

    def test_the_overview_counts_what_is_disputed(self, client):
        cid = _campaign(client)["campaign_id"]
        finding = _finding(client, cid)
        client.post(
            f"/campaigns/{cid}/findings/{finding['finding_id']}/notes",
            json={"body": "no", "stance": "disputes", "author": "s1"},
        )
        client.post(
            f"/campaigns/{cid}/findings/{finding['finding_id']}/notes",
            json={"body": "also no", "stance": "disputes", "author": "s2"},
        )

        atlas = client.get(f"/campaigns/{cid}/atlas").json()

        branch = {b["branch"]: b for b in atlas["branches"]}["selectivity"]
        assert branch["findings"] == 1
        assert branch["disputed"] == 1

    def test_one_spelling_per_branch(self, client):
        """Free text is how a memory fragments: three spellings, one branch."""
        cid = _campaign(client)["campaign_id"]
        _finding(client, cid, branch="ADMET", claim="first")
        _finding(client, cid, branch=" admet ", claim="second")

        listed = client.get(f"/campaigns/{cid}/findings?branch=Admet").json()

        assert [f["claim"] for f in listed["findings"]] == ["second", "first"]
        atlas = client.get(f"/campaigns/{cid}/atlas").json()
        assert [b["branch"] for b in atlas["branches"]] == ["admet"]

    def test_a_workstream_meets_its_findings_whatever_the_plan_spells(self, client):
        cid = _campaign(client)["campaign_id"]
        client.put(f"/campaigns/{cid}/plan", json={"workstreams": [
            {"id": "ADMET", "name": "ADMET liabilities"},
        ]})
        _finding(client, cid, branch="admet", claim="clearance sinks it")

        atlas = client.get(f"/campaigns/{cid}/atlas").json()

        assert len(atlas["branches"]) == 1
        assert atlas["branches"][0]["in_plan"] is True
        assert atlas["branches"][0]["findings"] == 1
        assert atlas["empty_branches"] == []

    def test_a_branch_outside_the_plan_still_shows(self, client):
        cid = _campaign(client)["campaign_id"]
        client.put(f"/campaigns/{cid}/plan", json=PLAN)
        _finding(client, cid, branch="synthesis")

        atlas = client.get(f"/campaigns/{cid}/atlas").json()

        by_branch = {b["branch"]: b for b in atlas["branches"]}
        assert by_branch["synthesis"]["in_plan"] is False
        assert by_branch["selectivity"]["in_plan"] is True

    def test_recording_a_finding_reaches_the_event_feed(self, client):
        """The watcher drives the UI off events, so a silent write is invisible."""
        cid = _campaign(client)["campaign_id"]

        _finding(client, cid)

        kinds = [e["kind"] for e in client.get(f"/campaigns/{cid}/events").json()["events"]]
        assert "finding_recorded" in kinds


class TestMetrics:

    def test_a_campaign_nobody_has_claimed_from_has_no_utilisation(self, client):
        campaign = _campaign(client, concurrency=2)
        _task(client, campaign["campaign_id"])

        metrics = client.get(f"/campaigns/{campaign['campaign_id']}/metrics").json()

        assert metrics["utilisation"] is None
        assert metrics["queued"] == 1
        assert metrics["free_slots"] == 2

    def test_a_running_task_holds_a_slot(self, client):
        campaign = _campaign(client, concurrency=2)
        _task(client, campaign["campaign_id"])
        _claim(client, "s0")

        metrics = client.get(f"/campaigns/{campaign['campaign_id']}/metrics").json()

        assert metrics["running"] == 1
        assert metrics["free_slots"] == 1

    def test_the_wait_between_tasks_is_timed_per_sub_agent(self, client):
        campaign = _campaign(client)
        _reported(client, campaign["campaign_id"], owner="s0")
        _reported(client, campaign["campaign_id"], owner="s0")

        metrics = client.get(f"/campaigns/{campaign['campaign_id']}/metrics").json()

        assert metrics["handoffs"] == 1
        assert metrics["handoff_seconds"]["median"] >= 0
