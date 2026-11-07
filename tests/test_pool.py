# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The Slurm scaler and the scheduler detection that picks between backends.

The Kubernetes scaler is covered in test_conductor.py, where it has always
lived; this file is the Slurm half and the choice between them.
"""

from __future__ import annotations

import json
import subprocess
import time

import pytest

from reinvent4_agent.pool import (
    KubernetesPoolScaler,
    NullPoolScaler,
    RelayPoolScaler,
    SlurmPoolScaler,
    make_pool_scaler,
    pool_capacity,
)


def _service_account(path):
    """Make a directory look like a mounted ServiceAccount."""
    (path / "token").write_text("t")
    (path / "namespace").write_text("demo-lux")
    (path / "ca.crt").write_text("c")
    return path


class _Recorder:
    """Stands in for subprocess.run, replaying canned stdout per command."""

    def __init__(self, outputs: dict[str, str], fail: set[str] | None = None):
        self.outputs = outputs
        self.fail = fail or set()
        self.calls: list[list[str]] = []

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        tool = argv[0]
        if tool in self.fail:
            raise subprocess.CalledProcessError(1, argv, stderr="boom")
        return subprocess.CompletedProcess(argv, 0, self.outputs.get(tool, ""), "")

    def argv_for(self, tool: str) -> list[list[str]]:
        return [c for c in self.calls if c[0] == tool]


@pytest.fixture
def slurm(monkeypatch):
    """A SlurmPoolScaler that believes the client tools exist."""
    monkeypatch.setattr(
        "reinvent4_agent.pool.shutil.which", lambda name: f"/usr/bin/{name}"
    )
    return SlurmPoolScaler(job_name="dd-supervisor", sbatch_script="sup.sbatch")


class TestSlurmPoolScaler:

    def test_without_slurm_tools_it_is_a_noop(self, monkeypatch):
        """Same contract as the Kubernetes scaler off-cluster: decline, don't raise."""
        monkeypatch.setattr("reinvent4_agent.pool.shutil.which", lambda name: None)
        scaler = SlurmPoolScaler()

        assert scaler.available is False
        assert scaler.current() is None
        assert scaler.scale(3) is False

    def test_the_count_comes_from_squeue(self, slurm, monkeypatch):
        rec = _Recorder({"squeue": "101 RUNNING\n102 RUNNING\n103 PENDING\n"})
        monkeypatch.setattr(subprocess, "run", rec)

        assert slurm.current() == 3

        argv = rec.argv_for("squeue")[0]
        assert "--name" in argv and "dd-supervisor" in argv
        # A completing job still holds its GPU, so it still holds a slot.
        assert "COMPLETING" in argv[argv.index("--states") + 1]

    def test_scaling_up_submits_the_difference(self, slurm, monkeypatch):
        rec = _Recorder({"squeue": "101 RUNNING\n", "sbatch": "777\n"})
        monkeypatch.setattr(subprocess, "run", rec)

        assert slurm.scale(3) is True

        submits = rec.argv_for("sbatch")
        assert len(submits) == 2
        assert submits[0][-1] == "sup.sbatch"
        assert "--parsable" in submits[0]

    def test_a_working_supervisor_is_asked_to_stop_not_killed(self, slurm, monkeypatch):
        """Cancelling one destroys the RL run it is an hour into.

        It handles SIGTERM by finishing the task and exiting, which takes far
        longer than the grace period a cancel allows -- so the drain it was
        written for would never once have completed.
        """
        rec = _Recorder({"squeue": "101 RUNNING\n102 RUNNING\n103 RUNNING\n"})
        monkeypatch.setattr(subprocess, "run", rec)

        assert slurm.scale(2) is True

        assert rec.argv_for("scancel") == [
            ["scancel", "--signal=TERM", "--full", "103"]
        ]

    def test_the_youngest_running_supervisor_is_the_one_asked(self, slurm, monkeypatch):
        """The oldest is furthest into its task and has the most to lose."""
        rec = _Recorder({"squeue": "101 RUNNING\n102 RUNNING\n103 RUNNING\n104 RUNNING\n"})
        monkeypatch.setattr(subprocess, "run", rec)

        slurm.scale(2)

        assert rec.argv_for("scancel")[0][-2:] == ["103", "104"]

    def test_a_supervisor_that_never_started_is_cancelled_outright(self, slurm, monkeypatch):
        """It holds no work, and the CPU it frees is free now."""
        rec = _Recorder({"squeue": "101 RUNNING\n102 PENDING\n"})
        monkeypatch.setattr(subprocess, "run", rec)

        assert slurm.scale(1) is True

        assert rec.argv_for("scancel") == [["scancel", "102"]]

    def test_queued_supervisors_are_given_back_before_working_ones(self, slurm, monkeypatch):
        """Giving back a running one costs a run; a pending one costs nothing."""
        rec = _Recorder({
            "squeue": "101 RUNNING\n102 RUNNING\n103 PENDING\n104 PENDING\n"
        })
        monkeypatch.setattr(subprocess, "run", rec)

        slurm.scale(2)

        assert rec.argv_for("scancel") == [["scancel", "103", "104"]]

    def test_a_big_shrink_spends_the_queue_then_asks_the_rest(self, slurm, monkeypatch):
        rec = _Recorder({
            "squeue": "101 RUNNING\n102 RUNNING\n103 RUNNING\n104 PENDING\n"
        })
        monkeypatch.setattr(subprocess, "run", rec)

        slurm.scale(1)

        assert rec.argv_for("scancel") == [
            ["scancel", "104"],
            ["scancel", "--signal=TERM", "--full", "102", "103"],
        ]

    def test_scaling_to_the_current_size_touches_nothing(self, slurm, monkeypatch):
        rec = _Recorder({"squeue": "101 RUNNING\n102 RUNNING\n"})
        monkeypatch.setattr(subprocess, "run", rec)

        assert slurm.scale(2) is True

        assert rec.argv_for("sbatch") == []
        assert rec.argv_for("scancel") == []

    def test_two_ticks_in_a_row_submit_once(self, slurm, monkeypatch):
        """The conductor relies on tick() being idempotent."""
        state = {"out": "101 RUNNING\n"}
        rec = _Recorder({"sbatch": "777\n"})

        def _run(argv, **kwargs):
            rec.calls.append(list(argv))
            if argv[0] == "squeue":
                return subprocess.CompletedProcess(argv, 0, state["out"], "")
            if argv[0] == "sbatch":
                state["out"] += "777\n"  # the submitted job now shows up
                return subprocess.CompletedProcess(argv, 0, "777\n", "")
            return subprocess.CompletedProcess(argv, 0, "", "")

        monkeypatch.setattr(subprocess, "run", _run)

        assert slurm.scale(2) is True
        assert slurm.scale(2) is True

        assert len(rec.argv_for("sbatch")) == 1

    def test_a_failed_squeue_is_reported_not_raised(self, slurm, monkeypatch):
        monkeypatch.setattr(subprocess, "run", _Recorder({}, fail={"squeue"}))

        assert slurm.current() is None
        assert slurm.scale(2) is False

    def test_a_failed_sbatch_is_reported_not_raised(self, slurm, monkeypatch):
        monkeypatch.setattr(
            subprocess, "run", _Recorder({"squeue": ""}, fail={"sbatch"})
        )

        assert slurm.scale(2) is False

    def test_account_and_partition_are_passed_when_set(self, monkeypatch):
        monkeypatch.setattr(
            "reinvent4_agent.pool.shutil.which", lambda name: f"/usr/bin/{name}"
        )
        rec = _Recorder({"squeue": "", "sbatch": "1\n"})
        monkeypatch.setattr(subprocess, "run", rec)

        SlurmPoolScaler(account="project_465000000", partition="standard-g").scale(1)

        argv = rec.argv_for("sbatch")[0]
        assert argv[argv.index("--account") + 1] == "project_465000000"
        assert argv[argv.index("--partition") + 1] == "standard-g"


class TestRelayPoolScaler:

    def test_without_a_control_directory_it_is_a_noop(self):
        scaler = RelayPoolScaler("")

        assert scaler.available is False
        assert scaler.current() is None
        assert scaler.scale(3) is False

    def test_the_wanted_count_is_written_for_the_relay(self, tmp_path):
        scaler = RelayPoolScaler(str(tmp_path))

        assert scaler.scale(4) is True

        assert (tmp_path / "pool.want").read_text().strip() == "4"
        # Written by rename, so no partial file is ever left behind.
        assert list(tmp_path.glob("pool.want.*")) == []

    def test_the_count_is_what_the_relay_achieved(self, tmp_path):
        """Not what was asked for: a full queue must show as a smaller pool."""
        scaler = RelayPoolScaler(str(tmp_path))
        scaler.scale(4)
        (tmp_path / "pool.state").write_text(
            json.dumps({"replicas": 2, "jobs": ["101", "102"], "at": time.time()})
        )

        assert scaler.current() == 2

    def test_a_relay_that_stopped_reporting_is_unknown_not_stale(self, tmp_path):
        scaler = RelayPoolScaler(str(tmp_path))
        (tmp_path / "pool.state").write_text(
            json.dumps({"replicas": 3, "at": time.time() - 10_000})
        )

        assert scaler.current() is None

    def test_no_state_yet_is_unknown(self, tmp_path):
        assert RelayPoolScaler(str(tmp_path)).current() is None


class TestCapacity:
    """The ceiling a person is offered before they choose a pool size."""

    def test_slurm_counts_the_partition_s_idle_cpus(self, monkeypatch):
        monkeypatch.setattr(
            "reinvent4_agent.pool.shutil.which", lambda name: f"/usr/bin/{name}"
        )
        # allocated/idle/other/total, and two supervisors already running.
        rec = _Recorder({"sinfo": "120/64/0/184\n", "squeue": "101\n102\n"})
        monkeypatch.setattr(subprocess, "run", rec)

        capacity = SlurmPoolScaler(partition="small").capacity()

        # 64 idle CPUs at 4 each is 16 more, on top of the 2 already there.
        assert capacity.max_supervisors == 18
        assert capacity.detail["idle_cpus"] == 64

    def test_slurm_without_a_partition_says_so_rather_than_guessing(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            "reinvent4_agent.pool.shutil.which", lambda name: f"/usr/bin/{name}"
        )
        monkeypatch.delenv("SLURM_PARTITION", raising=False)

        capacity = SlurmPoolScaler().capacity()

        assert capacity.max_supervisors is None

    def test_the_relay_reports_what_the_host_measured(self, tmp_path):
        (tmp_path / "pool.state").write_text(json.dumps({
            "replicas": 2, "at": time.time(),
            "max_supervisors": 18, "capacity_basis": "idle CPUs in partition small",
            "idle_cpus": 64, "cpus_each": 4,
        }))

        capacity = RelayPoolScaler(str(tmp_path)).capacity()

        assert capacity.max_supervisors == 18
        assert capacity.detail["idle_cpus"] == 64

    def test_a_silent_relay_is_unknown_not_zero(self, tmp_path):
        """Unknown must not read as "no room": that stops a campaign dead."""
        assert RelayPoolScaler(str(tmp_path)).capacity().max_supervisors is None

    def test_kubernetes_divides_the_namespace_quota(self, tmp_path, monkeypatch):
        _service_account(tmp_path)
        scaler = KubernetesPoolScaler(service_account=tmp_path)
        quota = {"items": [{"status": {
            "hard": {"requests.cpu": "40", "requests.memory": "200Gi"},
            "used": {"requests.cpu": "8", "requests.memory": "32Gi"},
        }}]}
        monkeypatch.setattr(
            scaler, "_request",
            lambda method, body=None, path=None: (
                quota if path else {"spec": {"replicas": 2}}
            ),
        )

        capacity = scaler.capacity()

        # 32 free cores at 4 each is 8; 168 GiB at 16 each is 10. CPU binds.
        assert capacity.max_supervisors == 10
        assert capacity.basis == "namespace quota"

    def test_kubernetes_without_a_quota_is_unknown(self, tmp_path, monkeypatch):
        """A cluster that will just leave pods Pending is not a number."""
        _service_account(tmp_path)
        scaler = KubernetesPoolScaler(service_account=tmp_path)
        monkeypatch.setattr(
            scaler, "_request", lambda method, body=None, path=None: {"items": []}
        )

        assert scaler.capacity().max_supervisors is None

    def test_nothing_schedules_here_so_nothing_is_promised(self):
        assert NullPoolScaler().capacity().max_supervisors is None

    def test_the_operator_cap_lowers_a_measured_ceiling(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SUPERVISOR_MAX", "6")
        (tmp_path / "pool.state").write_text(
            json.dumps({"replicas": 0, "at": time.time(), "max_supervisors": 18})
        )

        capacity = pool_capacity(RelayPoolScaler(str(tmp_path)))

        assert capacity.max_supervisors == 6
        assert capacity.detail["measured"] == 18

    def test_the_operator_cap_does_not_raise_a_smaller_one(
        self, tmp_path, monkeypatch
    ):
        """A cap is a ceiling, not a target: the machine still has the say."""
        monkeypatch.setenv("SUPERVISOR_MAX", "20")
        (tmp_path / "pool.state").write_text(
            json.dumps({"replicas": 0, "at": time.time(), "max_supervisors": 3})
        )

        assert pool_capacity(RelayPoolScaler(str(tmp_path))).max_supervisors == 3

    def test_a_supervisor_s_size_is_configurable(self, monkeypatch):
        monkeypatch.setenv("SUPERVISOR_CPUS", "8")
        monkeypatch.setattr(
            "reinvent4_agent.pool.shutil.which", lambda name: f"/usr/bin/{name}"
        )
        monkeypatch.setattr(
            subprocess, "run", _Recorder({"sinfo": "0/64/0/64\n", "squeue": ""})
        )

        assert SlurmPoolScaler(partition="small").capacity().max_supervisors == 8


class TestSchedulerDetection:

    def test_kubernetes_wins_when_a_token_is_mounted(self, tmp_path, monkeypatch):
        """Both could look present; a mounted token is the stronger signal."""
        _service_account(tmp_path)
        monkeypatch.setattr(
            "reinvent4_agent.pool.shutil.which", lambda name: f"/usr/bin/{name}"
        )

        scaler = make_pool_scaler(service_account=tmp_path)

        assert isinstance(scaler, KubernetesPoolScaler)

    def test_slurm_is_used_when_there_is_no_token(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "reinvent4_agent.pool.shutil.which", lambda name: f"/usr/bin/{name}"
        )

        scaler = make_pool_scaler(service_account=tmp_path)

        assert isinstance(scaler, SlurmPoolScaler)

    def test_the_relay_beats_slurm_when_both_look_present(
        self, tmp_path, monkeypatch
    ):
        """In a container sbatch may be on PATH and still be unusable."""
        control = tmp_path / "pool"
        control.mkdir()
        monkeypatch.setattr(
            "reinvent4_agent.pool.shutil.which", lambda name: f"/usr/bin/{name}"
        )
        monkeypatch.setattr(
            "reinvent4_agent.pool.SUPERVISOR_POOL_CONTROL", str(control)
        )

        scaler = make_pool_scaler(service_account=tmp_path)

        assert isinstance(scaler, RelayPoolScaler)

    def test_kubernetes_still_wins_over_the_relay(self, tmp_path, monkeypatch):
        control = tmp_path / "pool"
        control.mkdir()
        _service_account(tmp_path)
        monkeypatch.setattr(
            "reinvent4_agent.pool.SUPERVISOR_POOL_CONTROL", str(control)
        )

        assert isinstance(
            make_pool_scaler(service_account=tmp_path), KubernetesPoolScaler
        )

    def test_neither_gives_a_scaler_that_declines(self, tmp_path, monkeypatch):
        monkeypatch.setattr("reinvent4_agent.pool.shutil.which", lambda name: None)

        scaler = make_pool_scaler(service_account=tmp_path)

        assert isinstance(scaler, NullPoolScaler)
        assert scaler.available is False
        assert scaler.current() is None
        assert scaler.scale(2) is False
