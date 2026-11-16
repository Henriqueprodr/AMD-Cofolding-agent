# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A REINVENT run as its own Slurm job, through the session's run relay.

The runner cannot reach Slurm from its container, so it hands a run over as
files and reads back what the relay saw. These tests play the relay's part by
writing its state file, and check that the runner's job table -- the one the
supervisors watch -- follows it.
"""

from __future__ import annotations

import json
import shlex
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from reinvent4_mcp.config import ProcessConfig, reset_config
from reinvent4_mcp.jobs import manager as jobs
from reinvent4_mcp.jobs.manager import JobInfo, JobManager, JobState, reset_job_manager


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    reset_config()
    reset_job_manager()
    monkeypatch.setattr(jobs, "RELAY_POLL_SECONDS", 0.01)
    yield
    reset_config()
    reset_job_manager()


@pytest.fixture
def config(tmp_path):
    (tmp_path / "jobs").mkdir()
    return ProcessConfig(
        jobs={"working_dir": str(tmp_path / "jobs"), "stop_grace_seconds": 1},
        ttp={"storage_dir": str(tmp_path / "ttps")},
        logging={"level": "debug", "tb_logdir": str(tmp_path / "tb"), "log_dir": str(tmp_path / "logs")},
    )


@pytest.fixture
def relay(tmp_path, monkeypatch):
    path = tmp_path / "relay"
    monkeypatch.setenv(jobs.RUN_RELAY_ENV, str(path))
    return path


def _job(config, job_id="r1") -> JobInfo:
    job_dir = Path(config.jobs.working_dir) / f"job_{job_id}"
    job_dir.mkdir()
    (job_dir / "config.toml").write_text("")
    return JobInfo(job_id=job_id, ttp_id="t", ttp_version=1, model="m",
                   working_dir=str(job_dir), toml_path=str(job_dir / "config.toml"))


def _started(config, relay):
    manager = JobManager(config)
    job = _job(config)
    manager._jobs[job.job_id] = job
    manager._start_process(job)
    return manager, job


def _relay_saw(job, state, exit_code=None, slurm_id="555"):
    (Path(job.working_dir) / jobs.SLURM_STATE_FILE).write_text(
        json.dumps({"slurm_id": slurm_id, "state": state, "exit_code": exit_code})
    )


def test_the_gpu_worker_sees_drop_in_generation_priors():
    lumi = Path(__file__).resolve().parents[1] / "services" / "lumi"
    script = (lumi / "run.sbatch").read_text()
    session = (lumi / "session.sbatch").read_text()

    assert 'PRIORS_DIR=${PRIORS_DIR:-$REPO/priors}' in script
    assert '--bind "$PRIORS_DIR:/app/REINVENT4/priors"' in script
    assert 'PRIORS_DIR=${PRIORS_DIR:-$REPO/priors}' in session
    assert '--bind "$PRIORS_DIR:/app/REINVENT4/priors"' in session


class TestHandingARunOver:

    def test_the_run_is_left_for_the_relay_not_started_here(self, config, relay):
        _, job = _started(config, relay)

        assert (relay / "submit" / "r1").read_text().strip() == job.working_dir
        assert job.launcher == "slurm"
        assert job.pid is None
        # Waiting in the partition's queue still holds the run's slot.
        assert job.state == JobState.QUEUED

    def test_the_job_runs_the_command_the_runner_would_have(self, config, relay):
        _, job = _started(config, relay)

        script = (Path(job.working_dir) / "worker.sh").read_text()

        assert f"cd {shlex.quote(job.working_dir)}" in script
        assert "_run_worker.py" in script
        assert shlex.quote(job.toml_path) in script
        assert "export PYTHONPATH=" in script

    def test_without_a_relay_the_run_is_a_subprocess_as_before(self, config, monkeypatch):
        monkeypatch.delenv(jobs.RUN_RELAY_ENV, raising=False)
        started = []
        monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **k: started.append(a) or (_ for _ in ()).throw(OSError("no")))

        job = _job(config)
        JobManager(config)._start_process(job)

        assert started
        assert job.launcher == "local"


class TestFollowingTheSlurmJob:

    def test_pending_is_still_queued(self, config, relay):
        manager, job = _started(config, relay)
        _relay_saw(job, "PENDING")

        manager._poll_process(job)

        assert job.state == JobState.QUEUED
        assert job.slurm_job_id == "555"

    def test_running_is_running(self, config, relay):
        manager, job = _started(config, relay)
        _relay_saw(job, "RUNNING")

        manager._poll_process(job)

        assert job.state == JobState.RUNNING
        assert job.started_at

    def test_a_clean_exit_is_completed(self, config, relay):
        manager, job = _started(config, relay)
        _relay_saw(job, "COMPLETED", exit_code=0)

        manager._poll_process(job)

        assert job.state == JobState.COMPLETED
        assert manager.active_job_ids() == []

    def test_a_failure_says_why_from_the_jobs_own_output(self, config, relay):
        manager, job = _started(config, relay)
        (Path(job.working_dir) / "slurm-555.out").write_text("Boltz did not come up on :40211 in 15 min\n")
        _relay_saw(job, "FAILED", exit_code=1)

        manager._poll_process(job)

        assert job.state == JobState.FAILED
        assert "Slurm job 555 ended FAILED" in job.error
        assert "Boltz did not come up" in job.error

    def test_a_refused_submission_fails_the_run(self, config, relay):
        manager, job = _started(config, relay)
        (Path(job.working_dir) / jobs.SLURM_STATE_FILE).write_text(json.dumps(
            {"slurm_id": None, "state": "SUBMIT_FAILED", "error": "sbatch refused the run: QOSMaxSubmitJobPerUserLimit"}
        ))

        manager._poll_process(job)

        assert job.state == JobState.FAILED
        assert "QOSMaxSubmitJobPerUserLimit" in job.error


class TestStopping:

    def test_a_stop_is_asked_of_the_relay_and_reads_as_stopped(self, config, relay):
        manager, job = _started(config, relay)
        _relay_saw(job, "RUNNING")
        manager._poll_process(job)
        _relay_saw(job, "COMPLETED", exit_code=0)

        result = manager.stop_job("r1")

        assert (relay / "stop" / "r1").exists()
        assert job.state == JobState.STOPPED
        assert "error" not in result

    def test_a_run_still_waiting_in_the_queue_can_be_stopped(self, config, relay):
        manager, job = _started(config, relay)
        _relay_saw(job, "CANCELLED")

        manager.stop_job("r1")

        assert job.state == JobState.STOPPED


class TestARunnerRestart:

    def test_a_slurm_run_is_adopted_rather_than_buried(self, config, relay):
        job_dir = Path(config.jobs.working_dir) / "job_live1"
        job_dir.mkdir()
        (job_dir / "job_metadata.json").write_text(json.dumps({
            "job_id": "live1", "ttp_id": "t", "ttp_version": 1, "model": "m",
            "state": "running", "working_dir": str(job_dir),
            "launcher": "slurm", "slurm_job_id": "777",
        }))
        _relay_saw(JobInfo("live1", "t", 1, "m", str(job_dir), ""), "RUNNING", slurm_id="777")

        restored = JobManager(config).list_jobs()[0]

        assert restored["state"] == "running"
        assert restored["slurm_job_id"] == "777"

    def test_listing_jobs_does_not_wait_on_itself(self, config, relay):
        """Listing holds the manager's lock, which applying a Slurm state takes."""
        manager, job = _started(config, relay)
        _relay_saw(job, "RUNNING")
        manager._poll_process(job)

        assert manager.list_jobs()[0]["state"] == "running"


class TestTheRunWindow:
    """How many runs are admitted follows how fast Slurm starts them."""

    @pytest.fixture
    def manager(self, config):
        config.jobs.max_concurrent_jobs = 8
        return JobManager(config)

    @staticmethod
    def _waiting(manager, config, job_id, minutes, launcher="slurm"):
        job = _job(config, job_id)
        job.launcher = launcher
        job.created_at = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()
        manager._jobs[job_id] = job
        return job

    def test_runs_starting_promptly_leave_the_configured_limit(self, manager, config):
        self._waiting(manager, config, "w1", minutes=1)

        assert manager.run_window()["limit"] == 8

    def test_a_run_stuck_in_the_queue_halves_what_is_admitted(self, manager, config):
        for i in range(4):
            self._waiting(manager, config, f"w{i}", minutes=20 if i == 0 else 1)

        refused = manager._refuse_if_full()

        assert manager.run_window()["limit"] == 4
        assert "slow to start runs" in refused["error"]
        assert "waited 20 min" in refused["error"]
        # Raising the configured limit would not help; it is not suggested.
        assert "jobs.max_concurrent_jobs" not in refused["error"]

    def test_a_local_run_is_no_sign_of_a_slow_scheduler(self, manager, config):
        self._waiting(manager, config, "w1", minutes=20, launcher="local")

        assert manager.run_window()["limit"] == 8

    def test_prompt_starts_open_it_again(self, manager, config):
        stuck = self._waiting(manager, config, "w0", minutes=20)
        assert manager.run_window()["limit"] == 4
        stuck.state = JobState.FAILED

        for i in range(5):
            job = self._waiting(manager, config, f"p{i}", minutes=1)
            _relay_saw(job, "RUNNING")
            manager._apply_slurm_state(job)

        assert manager.run_window()["limit"] == 5

    def test_the_window_reaches_the_runner_health(self, manager, config):
        view = manager.run_window()

        assert set(view) >= {"limit", "ceiling", "active", "waiting", "free", "reason"}
        assert (view["ceiling"], view["free"]) == (8, 8)


class TestAPrivateScorer:

    def test_an_override_replaces_the_host_and_keeps_the_path(self, monkeypatch):
        from reinvent_plugins.components.comp_external_rest import override_service_url

        monkeypatch.setenv("SCORER_URL_OVERRIDES", "boltz=http://127.0.0.1:40211")

        assert override_service_url("boltz", "http://nid005518:8091/score") == "http://127.0.0.1:40211/score"
        assert override_service_url("posebusters", "http://nid005518:8082/score") == "http://nid005518:8082/score"

    def test_no_override_leaves_the_url_alone(self, monkeypatch):
        from reinvent_plugins.components.comp_external_rest import override_service_url

        monkeypatch.delenv("SCORER_URL_OVERRIDES", raising=False)

        assert override_service_url("boltz", "http://nid005518:8091/score") == "http://nid005518:8091/score"
