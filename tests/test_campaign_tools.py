"""Tests for reinvent4_mcp/tools/campaign.py -- the conductor's own view.

These are the calls an interactive conductor makes to answer "what is the
fleet doing" and "where did it put the output". They read through the ledger,
so what matters here is the shaping on top of it: a dispatch verdict that
names its binding limit, and paths a person can actually open.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from reinvent4_agent.campaign_client import StoreUnavailable
from reinvent4_mcp.tools import campaign as tools


class FakeStore:
    """Enough of CampaignStoreClient for the tools to read through."""

    def __init__(
        self,
        pool: dict[str, Any] | None = None,
        tasks: list[dict[str, Any]] | None = None,
        results: list[dict[str, Any]] | None = None,
        status: dict[str, Any] | None = None,
    ) -> None:
        self._pool = pool or {
            "supervisors": [], "n_alive": 0, "idle": 0, "busy": 0, "by_state": {},
        }
        self._tasks = tasks or []
        self._results = results or []
        self._status = status or {
            "campaign": {"concurrency": 2}, "in_flight": [],
        }

    def supervisors(self, campaign_id=None):
        return self._pool

    def tasks(self, campaign_id, state=None):
        return self._tasks

    def results(self, campaign_id, full=False):
        return self._results

    def status(self, campaign_id):
        return self._status


@pytest.fixture
def store(monkeypatch):
    """Bind the tools to a fake ledger, and return a setter for its contents."""

    holder: dict[str, FakeStore] = {"store": FakeStore()}
    monkeypatch.setattr(tools, "_store", lambda: holder["store"])

    def _use(**kwargs):
        holder["store"] = FakeStore(**kwargs)
        return holder["store"]

    return _use


def _pool(idle=0, busy=0, supervisors=None):
    return {
        "supervisors": supervisors or [],
        "n_alive": idle + busy,
        "idle": idle,
        "busy": busy,
        "by_state": {"idle": idle, "busy": busy},
    }


# ---------------------------------------------------------------------------
# The pool
# ---------------------------------------------------------------------------


class TestListSupervisors:

    def test_the_whole_fleet_needs_no_campaign(self, store):
        """A conductor asks "how many do I have" before it has a campaign."""
        store(pool=_pool(idle=2, busy=1))

        pool = tools.list_supervisors()

        assert (pool["idle"], pool["busy"], pool["n_alive"]) == (2, 1, 3)

    def test_one_campaign_also_gets_its_in_flight_tasks(self, store):
        store(
            pool=_pool(idle=1, busy=1),
            status={
                "campaign": {"concurrency": 3},
                "in_flight": [{"task_id": "t1", "owner": "s1"}],
            },
        )

        pool = tools.list_supervisors("feed0001")

        assert pool["n_in_flight"] == 1
        assert pool["concurrency"] == 3
        assert pool["idle"] == 1

    def test_an_unreachable_store_is_an_error_not_a_zero(self, store, monkeypatch):
        """Reporting "no supervisors" for an outage would read as a drained pool."""
        class Dead(FakeStore):
            def supervisors(self, campaign_id=None):
                raise StoreUnavailable("connection refused")

        monkeypatch.setattr(tools, "_store", Dead)

        assert "error" in tools.list_supervisors()


# ---------------------------------------------------------------------------
# Capacity
# ---------------------------------------------------------------------------


class TestPoolCapacity:
    """Whether to dispatch, and which of the two limits said no.

    Either number alone misleads: an idle supervisor handed work while the
    scorer is saturated only moves the wait, and a quiet scorer is no use
    with nothing alive to dispatch to.
    """

    @staticmethod
    def _pressure(value):
        return patch("reinvent4_agent.conductor.queue_pressure", return_value=value)

    def test_a_free_supervisor_and_a_quiet_scorer_is_a_yes(self, store):
        store(pool=_pool(idle=1))

        with self._pressure(0.5):
            capacity = tools.get_pool_capacity()

        assert capacity["can_dispatch"] == "yes"
        assert capacity["binding_limit"] is None

    def test_a_busy_pool_is_a_no_even_when_the_scorer_is_quiet(self, store):
        store(pool=_pool(idle=0, busy=3))

        with self._pressure(0.0):
            capacity = tools.get_pool_capacity()

        assert capacity["can_dispatch"] == "no"
        assert capacity["binding_limit"] == "every supervisor is busy"

    def test_a_saturated_scorer_is_a_no_even_with_a_free_supervisor(self, store):
        store(pool=_pool(idle=2))

        with self._pressure(5.0):
            capacity = tools.get_pool_capacity()

        assert capacity["can_dispatch"] == "no"
        assert capacity["binding_limit"] == "boltz is saturated"

    def test_an_unreachable_gateway_is_unknown_rather_than_free(self, store):
        store(pool=_pool(idle=1))

        with self._pressure(None):
            capacity = tools.get_pool_capacity()

        assert capacity["can_dispatch"] == "unknown"
        assert capacity["scorer"]["queue_pressure"] is None

    def test_the_two_numbers_are_both_reported(self, store):
        """So a conductor can explain its own verdict."""
        store(pool=_pool(idle=1, busy=2))

        with self._pressure(1.5):
            capacity = tools.get_pool_capacity()

        assert capacity["supervisors"] == {"idle": 1, "busy": 2, "alive": 3}
        assert capacity["scorer"]["queue_pressure"] == 1.5

    def test_a_full_run_window_still_leaves_room_for_other_kinds(self, store, monkeypatch):
        """No design task could start a run; a research task needs none."""
        store(pool=_pool(idle=2))
        monkeypatch.setattr(tools, "run_room", lambda: {
            "limit": 4, "ceiling": 16, "active": 4, "waiting": 2, "free": 0,
            "reason": "a run has waited 20 min for the scheduler to start it",
        })

        with self._pressure(0.5):
            capacity = tools.get_pool_capacity()

        assert capacity["can_dispatch"] == "not design"
        assert "waited 20 min" in capacity["binding_limit"]
        assert capacity["runs"]["free"] == 0


# ---------------------------------------------------------------------------
# Dispatching
# ---------------------------------------------------------------------------


class TestEnqueueTask:

    class Recording(FakeStore):
        def __init__(self) -> None:
            super().__init__()
            self.enqueued: list[dict[str, Any]] = []

        def enqueue_task(self, campaign_id, **spec):
            self.enqueued.append(spec)
            return {"task_id": "t2", "spec": spec}

    def test_the_tasks_it_builds_on_reach_the_ledger(self, monkeypatch):
        """They are the lines between the tasks in the user's view."""
        ledger = self.Recording()
        monkeypatch.setattr(tools, "_store", lambda: ledger)

        tools.enqueue_task("feed0001", "Grow into the ribose pocket", derived_from=["t1"])

        assert ledger.enqueued[0]["derived_from"] == ["t1"]

    def test_a_first_task_carries_no_lineage(self, monkeypatch):
        ledger = self.Recording()
        monkeypatch.setattr(tools, "_store", lambda: ledger)

        tools.enqueue_task("feed0001", "Survey the hinge binders")

        assert "derived_from" not in ledger.enqueued[0]

    def test_its_workstream_and_priority_reach_the_ledger(self, monkeypatch):
        ledger = self.Recording()
        monkeypatch.setattr(tools, "_store", lambda: ledger)

        tools.enqueue_task("feed0001", "Counter-screen BRD2", workstream="selectivity", priority=2)

        assert ledger.enqueued[0]["workstream"] == "selectivity"
        assert ledger.enqueued[0]["priority"] == 2


class TestEnqueueTasks:

    class Recording(FakeStore):
        def __init__(self) -> None:
            super().__init__()
            self.batches: list[list[dict[str, Any]]] = []

        def enqueue_tasks(self, campaign_id, tasks):
            self.batches.append(tasks)
            return [{"task_id": f"t{i}", "spec": spec} for i, spec in enumerate(tasks)]

    def test_a_batch_reaches_the_ledger_in_one_call(self, monkeypatch):
        ledger = self.Recording()
        monkeypatch.setattr(tools, "_store", lambda: ledger)

        answer = tools.enqueue_tasks("feed0001", [
            {"intent": "Survey the hinge binders", "kind": "research"},
            {"intent": "Counter-screen BRD2", "kind": "question", "workstream": "selectivity"},
        ])

        assert answer["n_queued"] == 2
        assert len(ledger.batches) == 1
        assert ledger.batches[0][1]["workstream"] == "selectivity"

    def test_a_task_without_an_intent_refuses_the_batch(self, monkeypatch):
        ledger = self.Recording()
        monkeypatch.setattr(tools, "_store", lambda: ledger)

        answer = tools.enqueue_tasks("feed0001", [{"intent": "fine"}, {"kind": "research"}])

        assert "Task 2" in answer["error"]
        assert ledger.batches == []

    def test_a_field_no_task_has_refuses_the_batch(self, monkeypatch):
        """Silently dropped, it would be an instruction the sub-agent never got."""
        ledger = self.Recording()
        monkeypatch.setattr(tools, "_store", lambda: ledger)

        answer = tools.enqueue_tasks("feed0001", [{"intent": "fine", "urgency": "high"}])

        assert "urgency" in answer["error"]
        assert ledger.batches == []

    @staticmethod
    def _room(free, limit=4, reason=None):
        return {"limit": limit, "ceiling": 16, "active": limit - free,
                "waiting": 0, "free": free, "reason": reason}

    def test_a_fan_out_is_told_how_much_of_it_starts_now(self, monkeypatch):
        """Queued whole, since waiting in the ledger costs nothing, but said.

        A comparison whose branches start an hour apart reads differently
        from one that ran side by side.
        """
        ledger = self.Recording()
        ledger._tasks = [
            {"spec": {"kind": "design"}, "state": "queued"},
            {"spec": {"kind": "design"}, "state": "claimed", "job_id": "j1"},
            {"spec": {"kind": "research"}, "state": "queued"},
        ]
        monkeypatch.setattr(tools, "_store", lambda: ledger)
        monkeypatch.setattr(tools, "run_room", lambda: self._room(free=2, reason="a run waited"))

        answer = tools.enqueue_tasks("feed0001", [
            {"intent": "Branch A"}, {"intent": "Branch B"}, {"intent": "Branch C"},
            {"intent": "Read the SAR", "kind": "research"},
        ])

        assert answer["n_queued"] == 4
        # One design task of the campaign was already waiting for a run.
        assert (answer["runs"]["design_tasks"], answer["runs"]["can_start_now"]) == (3, 1)
        assert "2 of the 3 design tasks will wait" in answer["runs"]["detail"]
        assert "a run waited" in answer["runs"]["detail"]

    def test_a_fan_out_with_room_says_so_without_a_warning(self, monkeypatch):
        ledger = self.Recording()
        monkeypatch.setattr(tools, "_store", lambda: ledger)
        monkeypatch.setattr(tools, "run_room", lambda: self._room(free=4))

        answer = tools.enqueue_tasks("feed0001", [{"intent": "A"}, {"intent": "B"}])

        assert answer["runs"]["can_start_now"] == 2
        assert "detail" not in answer["runs"]

    def test_a_batch_without_design_tasks_does_not_ask_about_runs(self, monkeypatch):
        ledger = self.Recording()
        monkeypatch.setattr(tools, "_store", lambda: ledger)
        monkeypatch.setattr(tools, "run_room", lambda: pytest.fail("asked the runner"))

        answer = tools.enqueue_tasks("feed0001", [{"intent": "Survey", "kind": "research"}])

        assert "runs" not in answer

    def test_an_unreadable_runner_does_not_hold_the_batch_up(self, monkeypatch):
        ledger = self.Recording()
        monkeypatch.setattr(tools, "_store", lambda: ledger)
        monkeypatch.setattr(tools, "run_room", lambda: None)

        answer = tools.enqueue_tasks("feed0001", [{"intent": "A"}])

        assert answer["n_queued"] == 1
        assert "runs" not in answer

    def test_fields_left_empty_are_not_sent(self):
        spec = tools.task_spec("Survey", kind="research", title="", derived_from=[], priority=0)

        assert spec == {"kind": "research", "intent": "Survey", "fidelity_tier": "exploration"}


class TestPlan:

    class Recording(FakeStore):
        def __init__(self) -> None:
            super().__init__()
            self.plans: list[list[dict[str, Any]]] = []

        def set_plan(self, campaign_id, workstreams):
            self.plans.append(workstreams)
            return {"campaign_id": campaign_id, "plan": {"workstreams": workstreams}}

        def coverage(self, campaign_id):
            return {"campaign_id": campaign_id, "workstreams": [], "gaps": ["admet"]}

    def test_the_plan_reaches_the_ledger_whole(self, monkeypatch):
        ledger = self.Recording()
        monkeypatch.setattr(tools, "_store", lambda: ledger)
        streams = [{"id": "selectivity", "name": "Selectivity"}, {"id": "admet", "name": "ADMET"}]

        answer = tools.set_campaign_plan("feed0001", streams)

        assert ledger.plans == [streams]
        assert answer["plan"]["workstreams"] == streams

    def test_coverage_is_the_ledgers_answer(self, monkeypatch):
        monkeypatch.setattr(tools, "_store", lambda: self.Recording())

        assert tools.get_campaign_coverage("feed0001")["gaps"] == ["admet"]


# ---------------------------------------------------------------------------
# What the sub-agents reported
# ---------------------------------------------------------------------------


class TestTaskReports:
    """The report is the only thing that crosses from a sub-agent."""

    REPORT = {
        "task_id": "t1",
        "report": "## Report\n\nTwo handles survive the 400 Da cut.",
        "summary": "Two handles survive the 400 Da cut.",
        "job_ids": ["j1", "j2"],
        "objective_profile": {"QED": 0.8},
        "candidates": [{"smiles": "CCO"}] * 25,
        "posted_at": "2026-11-09T10:00:00+00:00",
    }

    def test_it_returns_what_the_sub_agent_wrote(self, store):
        store(results=[dict(self.REPORT)])

        reports = tools.get_task_reports("feed0001")

        assert reports["n_reports"] == 1
        assert "400 Da cut" in reports["reports"][0]["report"]
        assert reports["reports"][0]["job_ids"] == ["j1", "j2"]

    def test_the_molecules_are_left_out(self, store):
        """A dozen reports with 25 molecules each would not fit in a turn;
        get_supervisor_result has them for one task."""
        store(results=[dict(self.REPORT)])

        report = tools.get_task_reports("feed0001")["reports"][0]

        assert "candidates" not in report
        assert report["n_candidates"] == 25

    def test_one_task_can_be_asked_for(self, store):
        store(results=[dict(self.REPORT), {**self.REPORT, "task_id": "t2"}])

        reports = tools.get_task_reports("feed0001", task_id="t2")

        assert [r["task_id"] for r in reports["reports"]] == ["t2"]

    def test_an_unknown_task_is_said_so(self, store):
        store(results=[dict(self.REPORT)])

        assert "error" in tools.get_task_reports("feed0001", task_id="nope")

    def test_a_result_without_a_report_is_named(self, store):
        """No report is not "found nothing" -- it is an older supervisor or an
        agent that said nothing, and the conductor should not read it as a
        finding."""
        store(results=[{**self.REPORT, "report": None}])

        reports = tools.get_task_reports("feed0001")

        assert reports["without_a_report"] == ["t1"]

    def test_a_store_outage_is_reported_not_raised(self, store, monkeypatch):
        fake = store(results=[])

        def _down(campaign_id, full=False):
            raise StoreUnavailable("the ledger is unreachable")

        monkeypatch.setattr(fake, "results", _down)

        assert "error" in tools.get_task_reports("feed0001")


# ---------------------------------------------------------------------------
# Where the files are
# ---------------------------------------------------------------------------


class TestCampaignFiles:
    """The ledger says what a task achieved; this says where it wrote it."""

    @pytest.fixture
    def jobs_root(self, tmp_path, monkeypatch):
        root = tmp_path / "jobs"
        root.mkdir()

        class _Jobs:
            working_dir = str(root)

        class _Config:
            jobs = _Jobs()

        monkeypatch.setattr("reinvent4_mcp.config.get_config", lambda: _Config())
        return root

    @staticmethod
    def _job(root: Path, job_id: str, **files: str) -> Path:
        job_dir = root / f"job_{job_id}"
        job_dir.mkdir()
        for name, body in files.items():
            # newline="" keeps the bytes on disk the ones written here; left to
            # translate, a "\n" becomes "\r\n" on Windows and the size a test
            # asserts is one byte short per line.
            (job_dir / name.replace("__", ".")).write_text(body, newline="")
        return job_dir

    def test_a_finished_task_points_at_its_directory(self, store, jobs_root):
        self._job(jobs_root, "j1", results_1__csv="SMILES,Score\nCCO,0.5\n")
        store(
            tasks=[{"task_id": "t1", "state": "done", "created_at": "1"}],
            results=[{"task_id": "t1", "job_id": "j1"}],
        )

        files = tools.get_campaign_files("feed0001")

        entry = files["tasks"][0]
        assert entry["directory"] == str(jobs_root / "job_j1")
        assert entry["job_id"] == "j1"

    def test_the_notable_files_are_named_with_their_sizes(self, store, jobs_root):
        self._job(
            jobs_root, "j1",
            results_1__csv="SMILES,Score\n", reinvent__log="step 1\n",
            config__toml="[run]\n", tb_logs_0="noise",
        )
        store(
            tasks=[{"task_id": "t1", "state": "done", "created_at": "1"}],
            results=[{"task_id": "t1", "job_id": "j1"}],
        )

        named = {
            f["name"]: f["bytes"]
            for f in tools.get_campaign_files("feed0001")["tasks"][0]["files"]
        }

        assert set(named) == {"results_1.csv", "reinvent.log", "config.toml"}
        assert named["results_1.csv"] == len("SMILES,Score\n")

    def test_a_task_that_never_started_a_job_says_so(self, store, jobs_root):
        """Rather than pointing at a directory that was never written."""
        store(tasks=[{"task_id": "t1", "state": "failed", "created_at": "1"}])

        entry = tools.get_campaign_files("feed0001")["tasks"][0]

        assert entry["directory"] is None
        assert "did not get as far" in entry["note"]

    def test_a_job_whose_directory_is_gone_is_distinguished(self, store, jobs_root):
        """A cleaned volume is a different problem from a task that never ran."""
        store(
            tasks=[{"task_id": "t1", "state": "done", "created_at": "1"}],
            results=[{"task_id": "t1", "job_id": "vanished"}],
        )

        entry = tools.get_campaign_files("feed0001")["tasks"][0]

        assert entry["directory"].endswith("job_vanished")
        assert "directory is gone" in entry["note"]

    def test_tasks_are_newest_first_and_bounded(self, store, jobs_root):
        store(tasks=[
            {"task_id": f"t{i}", "state": "done", "created_at": f"{i:02d}"}
            for i in range(30)
        ])

        files = tools.get_campaign_files("feed0001", limit=5)

        assert files["n_tasks"] == 30
        assert files["n_listed"] == 5
        assert [t["task_id"] for t in files["tasks"]] == [
            "t29", "t28", "t27", "t26", "t25"
        ]

    def test_it_says_where_the_structures_are_instead(self, store, jobs_root):
        """Cofold outputs are on the shared volume, not in the job directory."""
        store(tasks=[])

        files = tools.get_campaign_files("feed0001")

        assert "list_shared_assets" in files["hint"]
        assert files["jobs_root"] == str(jobs_root)


# ---------------------------------------------------------------------------
# The atlas
# ---------------------------------------------------------------------------


class TestAtlas:
    """Both sides read and write this, so the tools sign what they write."""

    class Recording(FakeStore):
        def __init__(self) -> None:
            super().__init__()
            self.written: list[dict[str, Any]] = []
            self.queries: list[dict[str, Any]] = []

        def atlas(self, campaign_id):
            return {"campaign_id": campaign_id, "branches": [], "empty_branches": ["admet"]}

        def findings(self, campaign_id, branch=None, q=None, tag=None, limit=50):
            self.queries.append(
                {"branch": branch, "q": q, "tag": tag, "limit": limit}
            )
            return {"campaign_id": campaign_id, "findings": []}

        def record_finding(self, campaign_id, author, branch, claim, **fields):
            self.written.append(
                {"author": author, "branch": branch, "claim": claim, **fields}
            )
            return {"finding_id": "f1", "claim": claim}

        def note_finding(self, campaign_id, finding_id, author, body, **fields):
            self.written.append(
                {"finding_id": finding_id, "author": author, "body": body, **fields}
            )
            return {"note_id": 1}

    @pytest.fixture
    def ledger(self, monkeypatch):
        recording = self.Recording()
        monkeypatch.setattr(tools, "_store", lambda: recording)
        return recording

    def test_the_overview_is_the_ledgers_answer(self, ledger):
        assert tools.atlas_overview("feed0001")["empty_branches"] == ["admet"]

    def test_a_branch_is_read_without_a_query(self, ledger):
        tools.atlas_branch("feed0001", "selectivity", limit=10)

        assert ledger.queries == [
            {"branch": "selectivity", "q": None, "tag": None, "limit": 10}
        ]

    def test_a_search_crosses_every_branch(self, ledger):
        tools.atlas_search("feed0001", "hinge donor", tag="BD1", limit=5)

        assert ledger.queries == [
            {"branch": None, "q": "hinge donor", "tag": "BD1", "limit": 5}
        ]

    def test_a_finding_is_signed_by_the_agent_recording_it(self, ledger, monkeypatch):
        """Two sub-agents on one node must not write under one name."""
        monkeypatch.setenv("SUPERVISOR_ID", "sup-7")

        tools.atlas_finding(
            "feed0001", "selectivity", "BD1 tolerates a methyl",
            evidence=["10.1021/x"], confidence="high", tags=["BD1"], task_id="t1",
        )

        assert ledger.written == [{
            "author": "sup-7", "branch": "selectivity",
            "claim": "BD1 tolerates a methyl", "detail": None,
            "evidence": ["10.1021/x"], "confidence": "high", "tags": ["BD1"],
            "task_id": "t1", "supersedes": None,
        }]

    def test_an_omitted_list_reaches_the_ledger_as_a_list(self, ledger):
        tools.atlas_finding("feed0001", "admet", "Clearance is the liability")

        assert ledger.written[0]["evidence"] == []
        assert ledger.written[0]["tags"] == []

    def test_a_note_is_signed_too(self, ledger, monkeypatch):
        monkeypatch.setenv("SUPERVISOR_ID", "sup-9")

        tools.atlas_note("feed0001", "f1", "Our run says otherwise", stance="disputes")

        assert ledger.written == [{
            "finding_id": "f1", "author": "sup-9",
            "body": "Our run says otherwise", "stance": "disputes", "task_id": None,
        }]

    def test_a_store_that_is_down_is_an_error_not_a_lost_turn(self, monkeypatch):
        class Down(FakeStore):
            def atlas(self, campaign_id):
                raise StoreUnavailable("connection refused")

        monkeypatch.setattr(tools, "_store", lambda: Down())

        assert "connection refused" in tools.atlas_overview("feed0001")["error"]
