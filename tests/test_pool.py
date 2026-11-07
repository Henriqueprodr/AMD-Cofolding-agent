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
        rec = _Recorder({"squeue": "101\n102\n103\n"})
        monkeypatch.setattr(subprocess, "run", rec)

        assert slurm.current() == 3

        argv = rec.argv_for("squeue")[0]
        assert "--name" in argv and "dd-supervisor" in argv
        # A completing job still holds its GPU, so it still holds a slot.
        assert "COMPLETING" in argv[argv.index("--states") + 1]

    def test_scaling_up_submits_the_difference(self, slurm, monkeypatch):
        rec = _Recorder({"squeue": "101\n", "sbatch": "777\n"})
        monkeypatch.setattr(subprocess, "run", rec)

        assert slurm.scale(3) is True

        submits = rec.argv_for("sbatch")
        assert len(submits) == 2
        assert submits[0][-1] == "sup.sbatch"
        assert "--parsable" in submits[0]

    def test_scaling_down_cancels_the_youngest(self, slurm, monkeypatch):
        """Oldest jobs are closest to a checkpoint, so they are kept."""
        rec = _Recorder({"squeue": "101\n102\n103\n104\n"})
        monkeypatch.setattr(subprocess, "run", rec)

        assert slurm.scale(2) is True

        cancel = rec.argv_for("scancel")[0]
        assert cancel == ["scancel", "103", "104"]

    def test_scaling_to_the_current_size_touches_nothing(self, slurm, monkeypatch):
        rec = _Recorder({"squeue": "101\n102\n"})
        monkeypatch.setattr(subprocess, "run", rec)

        assert slurm.scale(2) is True

        assert rec.argv_for("sbatch") == []
        assert rec.argv_for("scancel") == []

    def test_two_ticks_in_a_row_submit_once(self, slurm, monkeypatch):
        """The conductor relies on tick() being idempotent."""
        state = {"out": "101\n"}
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
