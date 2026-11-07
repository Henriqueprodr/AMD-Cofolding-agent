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

A third case sits between the two. On an HPC system the conductor runs inside
a container, and Slurm's client tools do not work in there: they need the
system's munge socket and the host's user database, neither of which crosses
the container boundary. There the conductor writes the number it wants to a
file and a plain bash loop on the host, inside the same allocation, submits
and cancels on its behalf.

Each scaler also answers *how many* could run at all, which is a different
question from how many are running: a person choosing a pool size needs the
ceiling the compute imposes before they choose, not a refusal afterwards. The
ceiling is read from whatever the scheduler will tell us -- free CPUs in the
partition on Slurm, the namespace quota on Kubernetes -- and is ``None`` when
it cannot be known, which is not the same as unlimited.

Env:
    SUPERVISOR_DEPLOYMENT     Deployment to scale on Kubernetes
    POD_NAMESPACE             namespace for the scale call
    SUPERVISOR_JOB_NAME       Slurm job name identifying pool members
    SUPERVISOR_SBATCH         sbatch script submitted to add one supervisor
    SUPERVISOR_POOL_CONTROL   directory shared with a host-side relay
    SUPERVISOR_CPUS           CPUs one supervisor asks for (default 4)
    SUPERVISOR_MEM_GB         memory one supervisor asks for (default 16)
    SUPERVISOR_MAX            hard ceiling, overriding anything measured
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

import requests

logger = logging.getLogger(__name__)

SUPERVISOR_DEPLOYMENT = os.getenv("SUPERVISOR_DEPLOYMENT", "dd-agentic-supervisor")
SUPERVISOR_JOB_NAME = os.getenv("SUPERVISOR_JOB_NAME", "dd-supervisor")
SUPERVISOR_SBATCH = os.getenv("SUPERVISOR_SBATCH", "services/slurm/supervisor.sbatch")
SUPERVISOR_POOL_CONTROL = os.getenv("SUPERVISOR_POOL_CONTROL", "")

_SERVICE_ACCOUNT = Path("/var/run/secrets/kubernetes.io/serviceaccount")

# Slurm states that still hold a place in the pool. A job that is completing
# is still occupying its GPU, so counting it as gone would scale us into
# oversubscription for as long as it takes to exit.
_LIVE_STATES = "PENDING,CONFIGURING,RUNNING,COMPLETING,SUSPENDED"


def _env_number(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


def supervisor_cost() -> tuple[float, float]:
    """CPUs and GiB of memory one supervisor asks its scheduler for."""

    return _env_number("SUPERVISOR_CPUS", 4), _env_number("SUPERVISOR_MEM_GB", 16)


@dataclass
class PoolCapacity:
    """How many supervisors the scheduler would run, and what says so.

    ``max_supervisors`` counts the pool as a whole -- the ones already running
    plus the ones there is room to add -- because that is the number a person
    sets, not the increment.
    """

    max_supervisors: Optional[int]
    basis: str
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _with_override(capacity: PoolCapacity) -> PoolCapacity:
    """Apply SUPERVISOR_MAX, which an operator sets to keep a pool smaller
    than the hardware would allow -- a shared node, a budget."""

    raw = os.getenv("SUPERVISOR_MAX", "").strip()
    if not raw:
        return capacity
    try:
        ceiling = int(raw)
    except ValueError:
        logger.warning("ignoring SUPERVISOR_MAX=%r: not an integer", raw)
        return capacity
    if capacity.max_supervisors is not None and capacity.max_supervisors <= ceiling:
        return capacity
    return PoolCapacity(
        ceiling, "SUPERVISOR_MAX", {**capacity.detail, "measured": capacity.max_supervisors}
    )


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

    def capacity(self) -> PoolCapacity:
        """The largest pool this compute would run, with what decided it."""


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

    def _request(
        self,
        method: str,
        body: Optional[dict[str, Any]] = None,
        path: Optional[str] = None,
    ) -> Any:
        token = (self._sa / "token").read_text().strip()
        url = "https://kubernetes.default.svc" + (path or (
            f"/apis/apps/v1/namespaces/"
            f"{self.namespace}/deployments/{self.deployment}/scale"
        ))
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

    def capacity(self) -> PoolCapacity:
        """How many supervisors the namespace quota leaves room for.

        A quota is the only ceiling a namespace can state about itself. With
        none declared the honest answer is unknown: the cluster's free
        capacity is not ours to read, and a scheduler that will simply leave
        pods Pending is not a number we can offer somebody.
        """

        if not self.in_cluster:
            return PoolCapacity(None, "not in a cluster")

        try:
            quotas = self._request(
                "GET", path=f"/api/v1/namespaces/{self.namespace}/resourcequotas"
            )["items"]
        except (requests.RequestException, KeyError, OSError) as exc:
            logger.warning("could not read the namespace quota: %s", exc)
            return PoolCapacity(None, "the namespace quota could not be read")

        cpus, mem_gb = supervisor_cost()
        room: list[int] = []
        detail: dict[str, Any] = {"cpus_each": cpus, "mem_gb_each": mem_gb}
        for quota in quotas:
            status = quota.get("status") or {}
            hard, used = status.get("hard") or {}, status.get("used") or {}
            for key, per_one in (("requests.cpu", cpus), ("requests.memory", mem_gb)):
                if key not in hard:
                    continue
                free = _k8s_quantity(hard[key]) - _k8s_quantity(used.get(key, "0"))
                room.append(int(free // per_one))
                detail[f"free_{key}"] = round(free, 2)

        if not room:
            return PoolCapacity(None, "the namespace declares no cpu or memory quota")

        running = self.current() or 0
        # The quota counts what is already running, so the free room is the
        # room to *add*; the pool size is that plus what is there.
        return PoolCapacity(max(running, running + min(room)), "namespace quota", detail)


def _k8s_quantity(value: str) -> float:
    """A Kubernetes quantity as a plain number: cores for cpu, GiB for memory.

    Only the suffixes a quota actually uses. An unparseable value is zero
    room rather than an exception, because a quota we cannot read should
    make the pool smaller, never larger.
    """

    text = str(value).strip()
    units = {
        "m": 1 / 1000, "Ki": 1 / 1024 ** 2, "Mi": 1 / 1024, "Gi": 1.0,
        "Ti": 1024.0, "K": 1e3 / 1024 ** 3, "M": 1e6 / 1024 ** 3, "G": 1e9 / 1024 ** 3,
    }
    for suffix, factor in sorted(units.items(), key=lambda kv: -len(kv[0])):
        if text.endswith(suffix):
            try:
                return float(text[: -len(suffix)]) * factor
            except ValueError:
                return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


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

    def capacity(self) -> PoolCapacity:
        """How many supervisors the partition has free CPUs for right now.

        Idle CPUs rather than total: a pool sized to a busy partition is a
        pool that mostly sits PENDING, which looks to a person like the fleet
        is broken rather than like the machine is full. Supervisors already
        running are counted back in, since their CPUs are no longer idle but
        are still ours.
        """

        if not self.in_slurm:
            return PoolCapacity(None, "no Slurm client here")

        out = self._run(
            "sinfo", "--noheader", "--partition", self.partition or "", "--format", "%C"
        ) if self.partition else None
        if not out:
            return PoolCapacity(
                None,
                "the partition's free CPUs could not be read"
                if self.partition else "no partition set (SLURM_PARTITION)",
            )

        # sinfo %C is "allocated/idle/other/total", aggregated over the partition.
        fields = out.strip().splitlines()[0].strip().split("/")
        try:
            idle, total = int(fields[1]), int(fields[3])
        except (IndexError, ValueError):
            logger.warning("could not parse sinfo CPU counts %r", out.strip())
            return PoolCapacity(None, "the partition's CPU counts were unreadable")

        cpus, _ = supervisor_cost()
        running = self.current() or 0
        return PoolCapacity(
            max(running, running + int(idle // cpus)),
            f"idle CPUs in partition {self.partition}",
            {"idle_cpus": idle, "total_cpus": total, "cpus_each": cpus,
             "running": running},
        )


class RelayPoolScaler:
    """Asks a host-side relay to do the scaling, through two small files.

    Slurm's clients cannot run inside the container -- ``sbatch`` needs the
    host's munge socket and user database -- so the conductor states its
    intent instead of acting on it. ``pool.want`` holds the number it wants;
    ``pool.state`` is what the relay last managed to achieve. Writes are
    atomic renames, so a relay reading mid-write sees the old number rather
    than half of the new one.

    Reporting the relay's state rather than the wanted number keeps the
    conductor honest: if the queue is full and nothing starts, it sees the
    pool it actually has.
    """

    # A relay that has not written for this long is treated as gone, so a
    # dead loop surfaces as "unknown" instead of a stale count the conductor
    # would keep reasoning about.
    STALE_AFTER = 300.0

    def __init__(self, control_dir: Optional[str] = None) -> None:
        self.dir = Path(control_dir or SUPERVISOR_POOL_CONTROL or "")

    @property
    def available(self) -> bool:
        return bool(self.dir.name) and self.dir.is_dir()

    def _state(self) -> Optional[dict[str, Any]]:
        """The relay's last report, or None if there is not a fresh one."""

        if not self.available:
            return None
        try:
            state = json.loads((self.dir / "pool.state").read_text())
        except (OSError, ValueError) as exc:
            logger.warning("no state from the pool relay: %s", exc)
            return None

        at = state.get("at")
        if isinstance(at, (int, float)) and time.time() - at > self.STALE_AFTER:
            logger.warning("the pool relay has not reported for %.0fs", time.time() - at)
            return None
        return state

    def current(self) -> Optional[int]:
        state = self._state()
        if state is None:
            return None
        replicas = state.get("replicas")
        return replicas if isinstance(replicas, int) else None

    def capacity(self) -> PoolCapacity:
        """Whatever the relay measured, since sinfo does not work in here either."""

        state = self._state()
        if state is None:
            return PoolCapacity(None, "the pool relay has not reported")
        measured = state.get("max_supervisors")
        if not isinstance(measured, int):
            return PoolCapacity(None, "the pool relay reported no capacity")
        return PoolCapacity(
            measured,
            state.get("capacity_basis") or "measured by the pool relay",
            {k: state[k] for k in ("idle_cpus", "cpus_each", "partition")
             if k in state},
        )

    def scale(self, replicas: int) -> bool:
        if not self.available:
            logger.info("no pool relay here; would size the pool to %d", replicas)
            return False

        tmp = self.dir / f"pool.want.{os.getpid()}"
        try:
            tmp.write_text(f"{replicas}\n")
            tmp.replace(self.dir / "pool.want")
        except OSError as exc:
            logger.error("could not ask the relay for %d supervisors: %s", replicas, exc)
            tmp.unlink(missing_ok=True)
            return False

        logger.info("asked the pool relay for %d supervisors", replicas)
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

    def capacity(self) -> PoolCapacity:
        return PoolCapacity(None, "no scheduler here")


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

    # Before Slurm, because where both are present we are in a container that
    # has the client tools bound in but cannot actually use them.
    relay = RelayPoolScaler()
    if relay.available:
        logger.info("scaling the supervisor pool through the host relay")
        return relay

    slurm = SlurmPoolScaler()
    if slurm.available:
        logger.info("scaling the supervisor pool through Slurm")
        return slurm

    return NullPoolScaler()


def pool_capacity(scaler: Optional[PoolScaler] = None) -> PoolCapacity:
    """The ceiling for this compute, with the operator's cap applied.

    Use this rather than ``scaler.capacity()`` directly: SUPERVISOR_MAX is a
    decision about the deployment rather than about the scheduler, so no
    scaler knows about it and every caller would otherwise have to.
    """

    return _with_override((scaler or make_pool_scaler()).capacity())
