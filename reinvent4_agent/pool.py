"""Sizing the supervisor pool, on Kubernetes or on Slurm.

The conductor decides *how many* supervisors should be running; this module
knows *how* to make that true on whichever scheduler is underneath. Both
implementations answer the same two questions -- how many are there, and make
it be N -- so the conductor never learns which one it has.

On Kubernetes a supervisor is a replica of a Deployment, and scaling is one
PATCH against the scale subresource. On Slurm it is a job: scaling up submits
more, scaling down cancels the most recently submitted, youngest first, so the
work that has run longest is the work least likely to be thrown away.

``make_pool_scaler`` picks between them by looking at the environment rather
than at configuration, because the thing that decides this is where the
process happens to be running.

Env:
    SUPERVISOR_DEPLOYMENT     Deployment to scale on Kubernetes
    POD_NAMESPACE             namespace for the scale call
    SUPERVISOR_JOB_NAME       Slurm job name identifying pool members
    SUPERVISOR_SBATCH         sbatch script submitted to add one supervisor
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

import requests

logger = logging.getLogger(__name__)

SUPERVISOR_DEPLOYMENT = os.getenv("SUPERVISOR_DEPLOYMENT", "dd-agentic-supervisor")
SUPERVISOR_JOB_NAME = os.getenv("SUPERVISOR_JOB_NAME", "dd-supervisor")
SUPERVISOR_SBATCH = os.getenv("SUPERVISOR_SBATCH", "services/slurm/supervisor.sbatch")

_SERVICE_ACCOUNT = Path("/var/run/secrets/kubernetes.io/serviceaccount")

# Slurm states that still hold a place in the pool. A job that is completing
# is still occupying its GPU, so counting it as gone would scale us into
# oversubscription for as long as it takes to exit.
_LIVE_STATES = "PENDING,CONFIGURING,RUNNING,COMPLETING,SUSPENDED"


@runtime_checkable
class PoolScaler(Protocol):
    """What the conductor needs from whatever is running the supervisors."""

    @property
    def available(self) -> bool:
        """False when scaling cannot be done here, so calls are no-ops."""

    def current(self) -> Optional[int]:
        """How many supervisors the pool is sized at, or None if unknown."""

    def scale(self, replicas: int) -> bool:
        """Size the pool. False when it could not be done."""


class KubernetesPoolScaler:
    """Scales the supervisor Deployment through the Kubernetes API.

    Talks to the API directly with the pod's ServiceAccount token rather than
    shelling out to ``kubectl`` (not in the image) or pulling in the Kubernetes
    client (a dependency for two HTTP calls).

    Outside a cluster there is no token, and every call is a logged no-op: a
    conductor run on a laptop should exercise the loop, not fail on the one
    thing that needs a cluster.
    """

    def __init__(
        self,
        deployment: str = SUPERVISOR_DEPLOYMENT,
        namespace: Optional[str] = None,
        service_account: Path = _SERVICE_ACCOUNT,
    ) -> None:
        self.deployment = deployment
        self._sa = service_account
        self.namespace = namespace or os.getenv("POD_NAMESPACE") or self._namespace()

    def _namespace(self) -> str:
        try:
            return (self._sa / "namespace").read_text().strip()
        except OSError:
            return "default"

    @property
    def in_cluster(self) -> bool:
        return (self._sa / "token").is_file()

    @property
    def available(self) -> bool:
        return self.in_cluster

    def _request(self, method: str, body: Optional[dict[str, Any]] = None) -> Any:
        token = (self._sa / "token").read_text().strip()
        url = (
            f"https://kubernetes.default.svc/apis/apps/v1/namespaces/"
            f"{self.namespace}/deployments/{self.deployment}/scale"
        )
        headers = {"Authorization": f"Bearer {token}"}
        if method == "PATCH":
            # Merge patch: the smallest thing that expresses "only replicas".
            headers["Content-Type"] = "application/merge-patch+json"

        response = requests.request(
            method, url, json=body, headers=headers,
            verify=str(self._sa / "ca.crt"), timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def current(self) -> Optional[int]:
        """Replicas the Deployment is set to, or None when not in a cluster."""

        if not self.in_cluster:
            return None
        try:
            return self._request("GET")["spec"].get("replicas", 0)
        except (requests.RequestException, KeyError, OSError) as exc:
            logger.warning("could not read the supervisor replica count: %s", exc)
            return None

    def scale(self, replicas: int) -> bool:
        """Set the replica count. False when it could not be done."""

        if not self.in_cluster:
            logger.info(
                "not in a cluster; would scale %s to %d", self.deployment, replicas
            )
            return False

        if self.current() == replicas:
            return True

        try:
            self._request("PATCH", {"spec": {"replicas": replicas}})
        except (requests.RequestException, OSError) as exc:
            # Most likely the ServiceAccount has no deployments/scale grant.
            logger.error("could not scale %s to %d: %s", self.deployment, replicas, exc)
            return False

        logger.info("scaled %s to %d", self.deployment, replicas)
        return True


class SlurmPoolScaler:
    """Sizes the supervisor pool by submitting and cancelling Slurm jobs.

    There is no replica count to set, so the count is derived: every pool
    member is a job sharing ``job_name``, and ``squeue`` is the source of
    truth. That makes the same idempotence the conductor relies on hold here
    too -- two ticks in a row converge on the same pool rather than submitting
    twice.

    Scaling down cancels the youngest jobs. A supervisor that has been running
    for an hour is holding an RL job that is an hour from its checkpoint, so
    cancelling the newest throws away the least work.
    """

    def __init__(
        self,
        job_name: str = SUPERVISOR_JOB_NAME,
        sbatch_script: str = SUPERVISOR_SBATCH,
        account: Optional[str] = None,
        partition: Optional[str] = None,
    ) -> None:
        self.job_name = job_name
        self.sbatch_script = sbatch_script
        self.account = account or os.getenv("SLURM_ACCOUNT")
        self.partition = partition or os.getenv("SLURM_PARTITION")

    @property
    def in_slurm(self) -> bool:
        """True when Slurm's client tools are usable from here.

        Deliberately not ``SLURM_JOB_ID``: the conductor may well run on a
        login node and submit into the queue, which is a perfectly good way to
        drive a campaign and has no job id of its own.
        """

        return shutil.which("squeue") is not None and shutil.which("sbatch") is not None

    @property
    def available(self) -> bool:
        return self.in_slurm

    def _run(self, *argv: str) -> Optional[str]:
        try:
            done = subprocess.run(
                argv, capture_output=True, text=True, timeout=60, check=True
            )
        except (OSError, subprocess.SubprocessError) as exc:
            stderr = getattr(exc, "stderr", "") or ""
            logger.error("%s failed: %s %s", argv[0], exc, stderr.strip())
            return None
        return done.stdout

    def job_ids(self) -> Optional[list[str]]:
        """Live pool job ids, oldest first, or None if squeue could not say."""

        out = self._run(
            "squeue", "--noheader", "--name", self.job_name,
            "--states", _LIVE_STATES, "--format", "%i",
            "--user", os.getenv("USER", ""), "--sort", "V",
        )
        if out is None:
            return None
        # Array tasks come back as "123_4"; the whole id is what scancel wants.
        return [line.strip() for line in out.splitlines() if line.strip()]

    def current(self) -> Optional[int]:
        ids = self.job_ids() if self.in_slurm else None
        return None if ids is None else len(ids)

    def scale(self, replicas: int) -> bool:
        if not self.in_slurm:
            logger.info(
                "no Slurm client here; would size %s to %d", self.job_name, replicas
            )
            return False

        ids = self.job_ids()
        if ids is None:
            return False

        if len(ids) < replicas:
            return self._submit(replicas - len(ids))
        if len(ids) > replicas:
            # job_ids() is oldest first, so the tail is the youngest work.
            return self._cancel(ids[replicas:])
        return True

    def _submit(self, count: int) -> bool:
        argv = ["sbatch", "--parsable", "--job-name", self.job_name]
        if self.account:
            argv += ["--account", self.account]
        if self.partition:
            argv += ["--partition", self.partition]
        argv.append(self.sbatch_script)

        for _ in range(count):
            out = self._run(*argv)
            if out is None:
                return False
            logger.info("submitted supervisor %s", out.strip())
        return True

    def _cancel(self, ids: list[str]) -> bool:
        if self._run("scancel", *ids) is None:
            return False
        logger.info("cancelled supervisors %s", ", ".join(ids))
        return True


class NullPoolScaler:
    """Stands in where nothing can be scaled, so the loop still runs.

    Returned when neither scheduler is present -- a laptop, a test, a CI box.
    The conductor is written to tolerate a scaler that declines, so this keeps
    that path explicit instead of leaving ``None`` to mean two different
    things.
    """

    available = False

    def current(self) -> Optional[int]:
        return None

    def scale(self, replicas: int) -> bool:
        logger.info("no scheduler here; would size the pool to %d", replicas)
        return False


def make_pool_scaler(**kwargs: Any) -> PoolScaler:
    """Pick the scaler that matches where this process is running.

    Kubernetes is tested first because its signal is the stronger one: a
    ServiceAccount token is only ever mounted by a kubelet, whereas Slurm's
    client tools can be installed on a machine that is not part of a cluster
    at all.
    """

    kubernetes = KubernetesPoolScaler(**kwargs)
    if kubernetes.available:
        return kubernetes

    slurm = SlurmPoolScaler()
    if slurm.available:
        logger.info("scaling the supervisor pool through Slurm")
        return slurm

    return NullPoolScaler()
