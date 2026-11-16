"""Headless supervisor: claim a task, run a campaign, post the result, repeat.

The interactive agent is a REPL reached by ``kubectl exec``; a fleet needs the
same Driver + Reviewer pair running unattended, N pods at a time, each taking
work from ``campaign-store`` and reporting back something a conductor can rank.
This module is that loop. It runs in the existing ``dd-agent`` image -- there is
no supervisor image, because a supervisor *is* the agent with a different
entrypoint.

Two things are deliberately not left to the language model:

* **The weight vector.** A task's whole purpose is to try specific objective
  weights, so the supervisor applies them itself -- it reads the base TTP,
  overrides the named weights and transforms, and persists the result as a
  variant TTP the Driver is told to run by id. Asking the Driver to "use weight
  1.4 on affinity" invites it to round, reinterpret or forget, and the campaign
  would then be comparing points it cannot name. The variant is kept rather
  than cleaned up: it is the record of what the task actually ran.

* **The objective profile.** What the task achieved is read out of the results
  CSV, not summarised by an agent.

The profile is taken from the **single best-scoring molecule**, not the best
value seen per objective. A per-objective maximum would mix molecules and
describe a compound that does not exist, and a result has to describe
something the run actually made.

Env:
    CAMPAIGN_STORE_URL            the ledger (default the in-cluster Service)
    SUPERVISOR_ID                 identity in the ledger (default: hostname)
    SUPERVISOR_CAMPAIGN_ID        restrict claims to one campaign (default: any)
    SUPERVISOR_POLL_SECONDS       wait between claims when the queue is empty
    SUPERVISOR_LEASE_SECONDS      lease to ask for (default: the store's own)
    SUPERVISOR_HEARTBEAT_SECONDS  how often to renew it while working

Plus the REINVENT4_AGENT_* variables the CLI already reads, which configure the
backend, model and budgets identically here. One of them changes the shape of a
task rather than just its settings:

    REINVENT4_AGENT_MULTI_AGENT=false runs the Driver alone. That halves the
    token spend, since the Reviewer produces no molecules -- but note it also
    replaces the monitoring loop, because ``monitor_campaign`` *is* the
    Reviewer cycle (assess the run, hand the Driver a verdict). Without a
    Reviewer the supervisor simply watches the job's state until it stops or
    the task deadline fires.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import logging
import os
import re
import signal
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

from reinvent4_agent.campaign_client import (
    STORE_URL,
    CampaignStoreClient,
    LeaseLost,
    StoreUnavailable,
    TaskOutcome,
    supervisor_id,
)
from reinvent4_agent.service_registry import apply_registry
from reinvent4_agent.task_inbox import Inbox

logger = logging.getLogger(__name__)

POLL_SECONDS = float(os.getenv("SUPERVISOR_POLL_SECONDS", "15"))
HEARTBEAT_SECONDS = float(os.getenv("SUPERVISOR_HEARTBEAT_SECONDS", "60"))
# Comfortably inside the store's 120s presence TTL, so one missed beat does
# not drop a working supervisor out of the pool.
PRESENCE_SECONDS = float(os.getenv("SUPERVISOR_PRESENCE_SECONDS", "45"))
#: How often a design task waiting for a run slot looks again.
RUN_ROOM_POLL_SECONDS = 30.0


class RunRefused(RuntimeError):
    """The runner had no slot for the run; another may free up."""

__all__ = [
    "CampaignStoreClient",
    "CoscientistTaskRunner",
    "LeaseLost",
    "StoreUnavailable",
    "Supervisor",
    "TaskOutcome",
    "apply_overrides",
    "authored_ttp_id",
    "build_brief",
    "build_goal",
    "design_report",
    "default_run_substitutes_for",
    "driver_denied_tools",
    "job_for_ttp",
    "main",
    "objective_profile",
    "prepare_ttp_variant",
    "supervisor_id",
    "variant_ttp_id",
]


# ---------------------------------------------------------------------------
# Weight overrides
# ---------------------------------------------------------------------------

def _named_components(ttp: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Every weighted, named entry in a TTP, by name.

    Weights live in two places -- ``pre_scoring.components`` and
    ``scoring.objectives`` -- and a sweep may well want to lean on a
    drug-likeness descriptor, so both are addressable by the same name.
    """

    found: dict[str, dict[str, Any]] = {}
    pre_scoring = ttp.get("pre_scoring") or {}
    for entry in (pre_scoring.get("components") or []):
        found[entry["name"]] = entry
    for entry in ((ttp.get("scoring") or {}).get("objectives") or []):
        found[entry["name"]] = entry
    return found


def apply_overrides(
    ttp: dict[str, Any],
    weight_overrides: Optional[dict[str, float]] = None,
    transform_overrides: Optional[dict[str, dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Return `ttp` with the named weights and transforms replaced.

    An override naming something the TTP does not have is an error, not a
    no-op. Silently ignoring it would have the task report a profile for a
    weight vector it never ran, and the conductor would then draw conclusions
    from a point that was never actually tried.
    """

    updated = copy.deepcopy(ttp)
    addressable = _named_components(updated)

    requested = set(weight_overrides or {}) | set(transform_overrides or {})
    unknown = sorted(requested - set(addressable))
    if unknown:
        raise ValueError(
            f"TTP '{ttp.get('id')}' has no objective named {unknown}; "
            f"it has {sorted(addressable)}"
        )

    for name, weight in (weight_overrides or {}).items():
        addressable[name]["weight"] = weight
    for name, transform in (transform_overrides or {}).items():
        addressable[name]["transform"] = transform

    return updated


def variant_ttp_id(base_ttp_id: str, task_id: str) -> str:
    """The TTP id a task runs under.

    Derived rather than looked up, which is what lets the conductor find a
    task's TTP -- and through it the task's job -- for a task whose supervisor
    has already gone.
    """

    return f"{base_ttp_id}-{task_id}"


def authored_ttp_id(task_id: str) -> str:
    """The TTP id a task writes its own profile under.

    Prescribed rather than left to the agent for the same reason the variant
    id is derived: it is how the task's job is found afterwards, by anyone,
    without trusting a sentence the model wrote. An agent that uses a
    different id has not done the task, and the task fails and is retried --
    which is the right outcome, because nobody can tell what it ran.
    """

    return f"ttp-{task_id}"


def prepare_ttp_variant(
    base_ttp_id: str,
    task_id: str,
    weight_overrides: Optional[dict[str, float]] = None,
    transform_overrides: Optional[dict[str, dict[str, Any]]] = None,
) -> str:
    """Persist the task's weight vector as its own TTP and return its id.

    Named after the task so a result can be traced back to exactly the TTP
    that produced it, and so N supervisors writing into the shared TTP store
    cannot collide.
    """

    from reinvent4_mcp.tools.ttp import create_ttp, get_ttp, update_ttp

    base = get_ttp(base_ttp_id)
    if "error" in base:
        raise ValueError(f"Base TTP '{base_ttp_id}': {base['error']}")

    variant = apply_overrides(base, weight_overrides, transform_overrides)
    variant["id"] = variant_ttp_id(base_ttp_id, task_id)
    variant["name"] = f"{base.get('name', base_ttp_id)} (task {task_id})"
    # create() assigns version 1; carrying the base's version across would
    # claim a lineage this variant does not have.
    variant.pop("version", None)

    created = create_ttp(variant)
    if "error" in created:
        # The variant is already there, which means this task is being
        # retried. Its content is a function of the task, so rewriting it
        # changes nothing -- but failing here would make a requeued task
        # impossible to run, and every retry would die before it started.
        rewritten = {k: v for k, v in variant.items() if k not in ("id", "version")}
        created = update_ttp(variant["id"], rewritten)
        if "error" in created:
            raise ValueError(f"Could not persist TTP variant for task {task_id}: "
                             f"{created['error']}")
        logger.info("task %s is a retry; reused TTP %s", task_id, variant["id"])

    logger.info(
        "task %s runs TTP %s with weights %s",
        task_id, created["id"], weight_overrides or "unchanged",
    )
    return created["id"]


# ---------------------------------------------------------------------------
# Objective profile
# ---------------------------------------------------------------------------

def job_for_ttp(ttp_id: str) -> Optional[str]:
    """The RL job running this task's TTP, observed from the job table.

    Deliberately *not* read out of the Driver's handoff. That handoff is free
    text the model may never produce -- a session that ends ``incomplete`` has
    none -- and the first live run failed a task with "the Driver finished
    without starting an optimization job" while its job was running happily.
    A job that exists must not be abandoned because the agent did not format
    its answer.

    The variant TTP id is unique to this task, so the job is identifiable
    without trusting anything the model said.
    """

    from reinvent4_mcp.jobs.manager import get_job_manager

    mine = [
        job for job in get_job_manager().list_jobs()
        if job.get("ttp_id") == ttp_id
    ]
    if not mine:
        return None

    # Newest, in case a retry of this task left an older one behind.
    return max(mine, key=lambda job: job.get("created_at") or "")["job_id"]


def objective_profile(
    candidates: list[dict[str, Any]], objectives: Optional[list[str]] = None
) -> dict[str, float]:
    """The best molecule's score on each objective.

    Deliberately one molecule's scores rather than the best value seen per
    objective: the latter describes a compound nobody made.
    """

    scored = [c for c in candidates if isinstance(c.get("total_score"), (int, float))]
    best = max(scored, key=lambda c: c["total_score"], default=None)
    if best is None:
        return {}

    profile: dict[str, float] = {}
    for name, component in (best.get("components") or {}).items():
        if objectives is not None and name not in objectives:
            continue
        value = component.get("transformed")
        if isinstance(value, (int, float)):
            profile[name] = float(value)
    return profile


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

TaskRunner = Callable[[dict[str, Any]], Awaitable[TaskOutcome]]


class Supervisor:
    """Claim, run, report, repeat -- until told to stop.

    The campaign run itself is injected, so the loop's behaviour around it
    (leases, deadlines, failures, shutdown) is testable without an LLM, an MCP
    server or a GPU.
    """

    def __init__(
        self,
        store: CampaignStoreClient,
        run_task: TaskRunner,
        *,
        campaign_id: Optional[str] = None,
        poll_seconds: float = POLL_SECONDS,
        heartbeat_seconds: float = HEARTBEAT_SECONDS,
        presence_seconds: float = PRESENCE_SECONDS,
        lease_seconds: Optional[float] = None,
        job_id_source: Optional[Callable[[], Optional[str]]] = None,
    ) -> None:
        self.store = store
        self.run_task = run_task
        #: Asked, on the beat, for the run the current task has started.
        #: Passed in rather than read off the runner so a runner that cannot
        #: answer is a decision at the call site and not a silent nothing.
        self.job_id_source = job_id_source
        self.campaign_id = campaign_id
        self.poll_seconds = poll_seconds
        self.heartbeat_seconds = heartbeat_seconds
        self.presence_seconds = presence_seconds
        self.lease_seconds = lease_seconds
        self._stopping = asyncio.Event()
        self.tasks_completed = 0
        self.tasks_failed = 0
        # What the lifetime presence beat restates between task events.
        self._presence: tuple[str, Optional[str], Optional[str]] = (
            "starting", None, "joining the pool",
        )

    def stop(self) -> None:
        """Finish the task in hand, then exit without claiming another."""

        if not self._stopping.is_set():
            logger.info("shutdown requested; no further tasks will be claimed")
        self._stopping.set()

    @property
    def stopping(self) -> bool:
        return self._stopping.is_set()

    async def _report(
        self, state: str, task_id: Optional[str] = None, detail: Optional[str] = None
    ) -> None:
        """Put this supervisor in the pool listing, or refresh where it is.

        Never fatal: the conductor not being able to see us is worse than it
        seeing us late, but neither is worth losing an hour of RL over.
        """

        # Remembered so the lifetime beat can restate it without guessing.
        self._presence = (state, task_id, detail)
        try:
            await asyncio.to_thread(
                self.store.report_presence,
                state, self.campaign_id, task_id,
                self.tasks_completed, self.tasks_failed, detail,
            )
        except (StoreUnavailable, LeaseLost) as exc:
            logger.debug("could not report presence: %s", exc)

    async def _presence_forever(self) -> None:
        """Keep this supervisor in the pool for as long as it is running.

        Separate from the task lease heartbeat because the two answer
        different questions: the lease says "this task is still mine",
        presence says "this supervisor exists". Renewing presence inside the
        lease heartbeat tied them together, so a supervisor whose task was
        taken -- by a cancel, or by its campaign being finalized -- kept
        working on the job it had already started while disappearing from the
        pool. Two did exactly that: alive, holding a GPU, invisible.
        """

        while not self.stopping:
            await self._sleep_unless_stopping(self.presence_seconds)
            if self.stopping:
                return
            state, task_id, detail = self._presence
            await self._report(state, task_id, detail)

    async def _heartbeat_forever(self, task_id: str) -> None:
        """Renew the lease until cancelled.

        A store that has gone away is logged and retried rather than ending
        the task: the campaign is hours long and the work in flight is worth
        more than the ledger entry. If the store stays down the lease lapses
        and the task is requeued, which is the right outcome anyway.

        Presence is not renewed here -- ``_presence_forever`` owns that, so
        losing a lease does not also remove the supervisor from the pool.

        The beat also carries the run this task started, the first time there
        is one. The ledger used to learn that only from the result, which is
        the one moment it is no longer interesting: for the hours the run was
        actually going, nothing connected the task to the job and there was
        no way to look inside a working supervisor.
        """

        reported: Optional[str] = None
        while True:
            await asyncio.sleep(self.heartbeat_seconds)
            if reported is None and self.job_id_source is not None:
                try:
                    reported = await asyncio.to_thread(self.job_id_source)
                except Exception as exc:
                    # Knowing the job id is a convenience for whoever is
                    # watching; failing to find it must not drop the lease.
                    logger.debug("could not tell which job %s started: %s", task_id, exc)
            try:
                await asyncio.to_thread(
                    self.store.heartbeat, task_id, self.lease_seconds, reported
                )
            except LeaseLost:
                logger.error("task %s was taken from us; abandoning it", task_id)
                raise
            except StoreUnavailable as exc:
                logger.warning("heartbeat for %s failed: %s", task_id, exc)

    async def run_once(self) -> bool:
        """Claim and run one task. False when there was nothing to do."""

        await self._report("idle", detail="waiting for a task")
        task = await asyncio.to_thread(
            self.store.claim, self.campaign_id, self.lease_seconds
        )
        if not task:
            return False

        task_id = task["task_id"]
        logger.info("claimed task %s of campaign %s", task_id, task.get("campaign_id"))
        await self._report("busy", task_id, detail="starting task")

        heartbeat = asyncio.create_task(self._heartbeat_forever(task_id))
        try:
            outcome = await self.run_task(task)
        except asyncio.CancelledError:
            raise
        except LeaseLost as exc:
            # Someone else owns it; saying anything about it would overwrite
            # their result.
            logger.error("task %s: %s", task_id, exc)
            return True
        except Exception as exc:
            logger.exception("task %s failed", task_id)
            self.tasks_failed += 1
            with contextlib.suppress(StoreUnavailable, LeaseLost):
                await asyncio.to_thread(self.store.fail, task_id, str(exc))
            return True
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError, LeaseLost):
                await heartbeat

        try:
            await asyncio.to_thread(self.store.post_result, task_id, outcome)
        except LeaseLost as exc:
            logger.error("task %s finished but the lease was lost: %s", task_id, exc)
            return True

        self.tasks_completed += 1
        logger.info("task %s done: %s", task_id, outcome.objective_profile)
        return True

    async def run_forever(self) -> None:
        """Work until stopped, backing off when the queue or the store is down."""

        logger.info(
            "supervisor %s working %s from %s",
            self.store.owner, self.campaign_id or "any campaign", self.store.base_url,
        )
        await self._report("starting", detail="joining the pool")
        presence = asyncio.create_task(self._presence_forever())

        while not self.stopping:
            try:
                did_work = await self.run_once()
            except StoreUnavailable as exc:
                # The conductor may still be starting, or the store may be
                # rolling. Neither is fatal to a pod that lives for hours.
                logger.warning("store unavailable: %s", exc)
                did_work = False

            if not did_work and not self.stopping:
                await self._sleep_unless_stopping(self.poll_seconds)

        presence.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await presence

        # Leave on purpose rather than timing out of the listing, so a pool
        # drained on command does not read as three crashed supervisors for
        # the next two minutes.
        try:
            await asyncio.to_thread(self.store.drop_presence)
        except (StoreUnavailable, LeaseLost) as exc:
            logger.debug("could not leave the pool cleanly: %s", exc)

        logger.info(
            "supervisor %s stopped after %d task(s), %d failed",
            self.store.owner, self.tasks_completed, self.tasks_failed,
        )

    async def _sleep_unless_stopping(self, seconds: float) -> None:
        """Wait, but wake immediately on shutdown so a drain is not delayed."""

        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._stopping.wait(), timeout=seconds)


# ---------------------------------------------------------------------------
# The real task runner: Driver + Reviewer against one task
# ---------------------------------------------------------------------------

def _deadline_seconds(task: dict[str, Any]) -> Optional[float]:
    """Seconds left on the task's own deadline, if it has one."""

    deadline = (task.get("spec") or {}).get("deadline")
    if not deadline:
        return None
    try:
        when = datetime.fromisoformat(deadline)
    except ValueError:
        logger.warning("task %s has an unparseable deadline %r",
                       task.get("task_id"), deadline)
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


def _overdue_seconds(task: dict[str, Any]) -> Optional[float]:
    """Seconds since the task's deadline passed, if it has passed.

    A task's deadline is absolute and set when the campaign enqueues it, so a
    task that waited behind the job backend's concurrency limit can be claimed
    after its own deadline. :func:`_deadline_seconds` floors at zero, which
    reads as "stop now" and is how two LUMI attempts on one task were started,
    stopped inside their first RL step with nothing to checkpoint, and
    reported as an unreadable results CSV -- the real reason, that the
    deadline had passed two hours before the claim, appearing nowhere.
    """

    raw = (task.get("spec") or {}).get("deadline")
    if not raw:
        return None
    try:
        when = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    overdue = (datetime.now(timezone.utc) - when).total_seconds()
    return overdue if overdue > 0 else None


# Tools a supervisor's Driver must not have. It runs the task it was given; it
# does not get to redefine it, abandon it, or steer the fleet.
#
# `stop_optimization` is here because of what happened the first time a
# supervisor ran for real: the Driver polled the job thirteen times in six
# minutes, saw no completed steps -- a Boltz-scored step takes 3-10 minutes, so
# there was nothing wrong -- decided the run was broken and stopped it. That
# killed the job inside its first step, when REINVENT has nothing to
# checkpoint, and produced no candidates. Stopping is the supervisor's job: it
# owns the deadline and does it with a checkpoint.
#
# The TTP writers are here because the whole point of the variant TTP is that
# the weight vector is fixed in advance. A Driver that can edit or replace a
# TTP can quietly run something other than the point the campaign is measuring.
#
# Every fleet tool belongs here, not only the obvious ones. This list held
# enqueue_task and not enqueue_tasks, so a Driver could queue work for the
# whole campaign by reaching for the plural, and could rewrite the plan or
# claim proposals besides. The subset check against FLEET_TOOLS is what keeps
# the two in step as tools are added.
DRIVER_DENIED_TOOLS = frozenset({
    "stop_optimization",
    "create_ttp",
    "update_ttp",
    "delete_ttp",
    "start_campaign",
    "enqueue_task",
    "enqueue_tasks",
    "set_campaign_plan",
    "take_proposals",
    "drop_proposals",
    "set_concurrency",
    "cancel_supervisor",
    "close_campaign",
    "finalize_campaign",
    "conduct_campaign",
    "stop_conducting",
})


# What a sub-agent on a task that is not a design run must not start. The
# supervisor's deadline, monitoring and harvest are built around one RL job:
# a research task that starts one has nobody watching it, nobody stopping it
# at the deadline and nobody reading its results, so it would hold a GPU
# until the runner was restarted. A task that turns out to need a run says so
# in its report, and the conductor dispatches a design task for it.
GENERAL_DENIED_TOOLS = frozenset({
    "run_optimization",
    "run_transfer_learning",
    "run_generation_campaign",
    "create_generation_campaign",
})


#: Said in place of each tool a sub-agent may not call, so a refusal names
#: the way round it.  Without this the model is told only that the tool is
#: unavailable, and it either retries the same call or abandons the thread
#: of work the tool was part of.
DENIAL_REASONS: dict[str, str] = {
    "run_optimization": "This task is not a design run, so it has nobody "
        "watching a job at the deadline. Say in your report that the work "
        "argues for a run and what it should optimise; the conductor will "
        "dispatch a design task for it.",
    "run_transfer_learning": "This task is not a design run. Put the case "
        "for it in your report and the conductor will dispatch one.",
    "run_generation_campaign": "This task is not a design run. Put the case "
        "for it in your report and the conductor will dispatch one.",
    "create_generation_campaign": "This task is not a design run. Put the "
        "case for it in your report and the conductor will dispatch one.",
    "stop_optimization": "The supervisor owns the deadline and stops the run "
        "with a checkpoint. A run with no finished steps yet is not a stuck "
        "run: a Boltz-scored step takes minutes.",
    "create_ttp": "The profile is the experiment and it was fixed before "
        "this task. Use the TTP you were given.",
    "update_ttp": "The profile is the experiment and it was fixed before "
        "this task. Report what you would change instead.",
    "delete_ttp": "Profiles outlive this task and other runs cite them.",
    "start_campaign": "Only the conducting agent divides the work. Say what "
        "you would have asked for in your report.",
    "enqueue_task": "Only the conducting agent divides the work. Say what "
        "you would have asked for in your report.",
    "enqueue_tasks": "Only the conducting agent divides the work. Say what "
        "you would have asked for in your report.",
    "set_campaign_plan": "Only the conducting agent divides the goal into "
        "workstreams. Say what you would have changed in your report.",
    "take_proposals": "Only the conducting agent decides which proposals "
        "become tasks. Putting yours in your report is how it is considered.",
    "drop_proposals": "Only the conducting agent sets a proposal aside.",
    "set_concurrency": "Only the conducting agent sizes the pool.",
    "cancel_supervisor": "Only the conducting agent cancels a sub-agent.",
    "close_campaign": "Only the conducting agent closes the campaign.",
    "finalize_campaign": "Only the conducting agent finalises the campaign.",
    "conduct_campaign": "Only the conducting agent holds the pool.",
    "stop_conducting": "Only the conducting agent stops the pool.",
}


def denial_reasons(denied: frozenset[str]) -> dict[str, str]:
    """What to say instead, for each tool withheld from this sub-agent."""

    return {name: DENIAL_REASONS.get(name, "") for name in denied}


def driver_denied_tools(authored: bool, kind: str = "design") -> frozenset[str]:
    """What this particular Driver must not touch.

    ``create_ttp`` is denied to a Driver running a variant of the campaign's
    base TTP, where the weights are the experiment and writing a new profile
    would quietly measure something else. It cannot be denied to a Driver
    asked to write its own: that is the task.

    ``update_ttp`` and ``delete_ttp`` stay denied either way. Writing one
    profile is the task; rewriting a profile that already has a run behind
    it, or removing someone else's, is not.

    A task that is not a design run may write a profile -- preparing the TTP
    for a later run is a reasonable thing to ask of one, and ``create_ttp``
    cannot overwrite -- but loses the job starters, for the reason beside
    ``GENERAL_DENIED_TOOLS``.
    """

    if kind != "design":
        return (DRIVER_DENIED_TOOLS - {"create_ttp"}) | GENERAL_DENIED_TOOLS
    if not authored:
        return DRIVER_DENIED_TOOLS
    return DRIVER_DENIED_TOOLS - {"create_ttp"}

# What one RL step costs when every molecule is folded, measured on demo-lux.
# Quoted to the Driver so its patience has a basis: without a number it has no
# way to tell a slow step from a broken one.
STEP_SECONDS_HINT = "1 to 4 minutes at exploration-tier parameters"


def default_run_substitutes_for(
    spec: dict[str, Any], authored: bool = False
) -> bool:
    """Whether starting a plain run would still be this task's experiment.

    The supervisor starts the run itself when its agent finishes without
    doing so. That call takes only the TTP and the step count, so it generates
    from the default de novo prior -- a faithful stand-in while every task
    does, and not once a task's ``intent`` may have called for another prior.
    A substituted run would report a result for an experiment nobody asked
    for, which is worse than the task failing and being retried.

    Never a stand-in for a task that was asked to write its own TTP: the
    substitute run would name a TTP that does not exist, because writing it
    was the part the agent skipped.
    """

    return not authored and not (spec or {}).get("intent")


def build_goal(
    task: dict[str, Any],
    ttp_id: str,
    campaign_goal: Optional[str] = None,
    authored: bool = False,
    findings: Optional[list[dict[str, Any]]] = None,
) -> str:
    """The instruction handed to the Driver.

    Two shapes, because a campaign can work either way round. When the
    campaign has a base TTP, the task's variant is already written and the
    agent is told to run it and not touch its weights. When it has none --
    the usual case -- the agent writes the profile itself from what the task
    is for, which is the part of the work it is actually better at: it can
    look the target up and decide what is worth scoring, where a weight
    vector can only re-weight what somebody already thought of.

    Names the prepared TTP rather than restating the weights, so there is one
    source of truth for the weight vector and the agent cannot paraphrase it
    into something else.

    The task's ``intent`` goes in verbatim. Everything else here is structured
    and therefore only says what the schema anticipated: holding a chemotype,
    for instance, is a decision about which prior to generate from, and the
    agent is better placed to make it -- it can look the reference compound up
    and choose a model -- than a seed field would be. Without a channel for
    that the agent ran de novo not because it could not do better but because
    nobody asked it to.

    ``findings`` matters most where the agent writes its own profile: what
    the campaign has learnt about the target is exactly what decides which
    objectives are worth scoring, and it would otherwise have to rediscover
    it from the literature on every task.
    """

    spec = task.get("spec") or {}
    if authored:
        lines = [
            f"Run a lead-optimisation task, under TTP id '{ttp_id}'.",
            "",
            "There is no therapeutic target profile for this task yet, so "
            "your first step is to write one:",
            "",
            f"  1. Call create_ttp with the id '{ttp_id}' exactly. That id "
            "is how your run is found afterwards, so a different one loses "
            "the work even though it ran. Choose the objectives and their "
            "weights from what this task is for -- look the target up "
            "rather than recalling it, and say what you chose to score and "
            "why before you start.",
            f"  2. Start the run against '{ttp_id}'. run_optimization for a "
            "de novo or Mol2Mol run; run_generation_campaign when the profile "
            "asks for a scaffold (libinvent) or linker (linkinvent) run, "
            "which run_optimization cannot build and refuses.",
            "",
            "Do not start a run before step 1. A run with no TTP behind it "
            "scores nothing this task asked for and cannot be found later, "
            "so the task fails exactly as if you had done nothing. "
            "create_generation_campaign is one way of doing step 1, not a "
            f"way round it: give it campaign_id='{ttp_id}' so the profile it "
            "writes is the one this task is found by.",
        ]
        if campaign_goal:
            lines += [
                "",
                "The campaign this task belongs to:",
                str(campaign_goal).strip(),
            ]
    else:
        lines = [
            f"Run a lead-optimisation campaign against TTP '{ttp_id}'.",
            "",
            "The TTP already carries the exact objective weights and "
            "transforms this task is meant to explore -- use it as it is and "
            "do not modify its weights.",
        ]
    if spec.get("intent"):
        lines += [
            "",
            "What this run is for:",
            str(spec["intent"]).strip(),
            "",
            "That is the experiment. If it calls for generating from "
            "something other than the default de novo prior, choose the "
            "model and any seed molecules yourself -- look the reference "
            "compounds up rather than recalling their structures -- and say "
            "which you chose and why before starting. Holding a scaffold "
            "fixed and varying its substituents is a libinvent run, and "
            "joining two fragments with a linker is a linkinvent run; both "
            "are started with run_generation_campaign, not run_optimization.",
        ]
    if spec.get("max_steps"):
        lines.append(
            f"Run {spec['max_steps']} RL steps: the step count is budgeted "
            "against the campaign deadline, so do not raise it."
        )
    if spec.get("fidelity_tier") == "validation":
        lines.append(
            "This is a validation-tier task: results must not come from the "
            "scoring cache, so do not re-use a previous run's numbers."
        )
    if findings:
        lines += _atlas_slice(findings)
    if task.get("campaign_id"):
        branch = (spec or {}).get("workstream")
        lines += [
            "",
            f"This campaign is {task['campaign_id']} and your task is "
            f"{task.get('task_id')}"
            + (f", on branch '{branch}'." if branch else ".")
            + " When the run settles something about the target or the "
            "series -- an objective that fights another, a liability the "
            "scores keep hitting -- record it with atlas_finding so the "
            "next task does not rediscover it. Check with atlas_search "
            "first if you are unsure whether it is already known.",
        ]

    lines += [
        "",
        "Start the optimization and monitor it, then report the best molecules "
        "with their per-objective scores.",
        "",
        FOLLOW_UP_ASK,
        "",
        f"Pacing: one RL step takes {STEP_SECONDS_HINT}, because every molecule "
        "in the batch is folded by Boltz. A run that reports no completed steps "
        "for several minutes is normal and is not evidence of a problem -- "
        "check again later rather than concluding it has stalled. Do not try to "
        "stop, restart or replace the run: this session has a wall-clock "
        "deadline that is enforced for you, and stopping early loses the work.",
    ]
    return "\n".join(lines)


#: How the brief asks for the report, and how the supervisor finds it. Named
#: in the brief rather than parsed out of prose: the whole answer is the
#: report, and the heading is only there so a person scrolling a long
#: transcript can see where the conclusion starts.
REPORT_HEADING = "## Report"

#: Where a report proposes the tasks its findings argue for. A section with
#: one line per task rather than JSON: the model writing it is mid-prose, and
#: a malformed object would cost every proposal where a bad line costs one.
NEXT_TASKS_HEADING = "## Next tasks"
MAX_FOLLOW_UPS = 3
FOLLOW_UP_KINDS = ("design", "research", "analysis", "retrosynthesis", "question")

FOLLOW_UP_ASK = (
    f"End the report with a '{NEXT_TASKS_HEADING}' section: up to "
    f"{MAX_FOLLOW_UPS} tasks your results argue for, one line each, as "
    "'- (kind) Title: what to do, and the result of yours that argues for it'. "
    f"kind is one of {', '.join(FOLLOW_UP_KINDS)}. Each one is handed to "
    "another agent who has not read your report, so say enough for them to "
    "start. Leave the section out if nothing follows from what you found."
)

_FOLLOW_UP_LINE = re.compile(
    r"^\s*(?:[-*]|\d+[.)])\s+(?:\((?P<kind>[a-z]+)\)\s*)?(?P<body>.+?)\s*$"
)


def parse_follow_ups(report: str) -> list[dict[str, Any]]:
    """The tasks a report's next-tasks section proposes, at most MAX_FOLLOW_UPS.

    Lenient: a line without a kind is research, one without a title is
    titled by its first words, and anything that is not a list item ends the
    section.
    """

    lines = (report or "").splitlines()
    heading = NEXT_TASKS_HEADING.lstrip("#").strip().lower()
    start = next(
        (i for i, line in enumerate(lines) if line.strip().lstrip("#").strip().lower() == heading),
        None,
    )
    if start is None:
        return []
    proposals: list[dict[str, Any]] = []
    for line in lines[start + 1:]:
        if not line.strip():
            continue
        match = _FOLLOW_UP_LINE.match(line)
        if match is None:
            break
        body = match.group("body").replace("**", "").strip()
        kind = (match.group("kind") or "research").lower()
        title, sep, intent = body.partition(": ")
        if not sep or len(title) > 80:
            title, intent = " ".join(body.split()[:6]), body
        if len(intent.strip()) < 10:
            continue
        proposals.append({
            "kind": kind if kind in FOLLOW_UP_KINDS else "research",
            "title": title.strip()[:120],
            "intent": intent.strip(),
        })
        if len(proposals) == MAX_FOLLOW_UPS:
            break
    return proposals


#: An answer shorter than this did not report anything -- it acknowledged the
#: task. Measured on the whole reply, because a one-line "done" is exactly
#: what the second ask is for.
MIN_REPORT_CHARS = 200

#: What the sub-agent is told when its first answer was not a report. One
#: extra turn, not a loop: an agent that has nothing to say after being asked
#: twice is reporting that, and the conductor can read it.
REPORT_NUDGE = (
    "That is not the report. Write it now, in one message, starting with the "
    f"'{REPORT_HEADING}' heading: what you did, what you found with the "
    "numbers behind it, what you could not settle, and what you would do "
    "next. Do not call any more tools -- say what you already know."
)

#: What the sub-agent is told when its deadline passed mid-task.
WRAP_UP_PROMPT = (
    "Your time for this task is up. Stop what you are doing and write the "
    f"report now, in one message, starting with '{REPORT_HEADING}': what you "
    "did, what you found, what you did not get to, and what you would do "
    "next. Do not call any more tools."
)

#: What the sub-agent is told about a message from the person watching that
#: came in after it had written its report.
LATE_MESSAGE_PROMPT = (
    "The message above is from the person watching this task, and it came in "
    "after your report. Answer it, then give the report again in full, "
    f"starting with '{REPORT_HEADING}', changed by anything the answer adds."
)

#: How long the wrap-up turn gets after the deadline. Enough for one answer
#: with no tool calls; a sub-agent that cannot manage that has already spent
#: its task's time.
WRAP_UP_SECONDS = 180.0

#: Tools that put a job in the runner's job table. Watched so a task reports
#: every run it started, not only the one its molecules came from -- a design
#: task that started a transfer-learning run first would otherwise leave it
#: unattributed in the UI.
JOB_STARTING_TOOLS = frozenset({
    "run_optimization",
    "run_transfer_learning",
    "run_generation_campaign",
})

_JOB_ID = re.compile(r'"job_id"\s*:\s*"([^"]+)"')


class _JobIdCollector:
    """Notes the runs an agent starts, by watching its own tool results.

    The job table is not enough on its own: it is shared by every task in the
    pool, so "the jobs that exist" cannot be narrowed to "the jobs this task
    started" without the TTP id, and a task that is not a design run has no
    TTP of its own.
    """

    def __init__(self) -> None:
        self.job_ids: list[str] = []

    def on_tool_result(self, name: str, result: str, is_error: bool) -> None:
        if is_error or name not in JOB_STARTING_TOOLS:
            return
        for job_id in _JOB_ID.findall(result or ""):
            if job_id not in self.job_ids:
                self.job_ids.append(job_id)

    def on_thinking(self, text: str) -> None: ...
    def on_tool_call(self, name: str, arguments: dict[str, Any]) -> None: ...
    def on_error(self, message: str) -> None: ...
    def on_monitor_alert(self, alert: dict[str, Any]) -> None: ...
    def on_monitor_response(self, text: str) -> None: ...


def _inbox_for(args: Any, task: dict[str, Any]) -> Optional[Inbox]:
    """Where the UI leaves messages for this task, when there is a shared log directory to leave them in."""

    log_dir = getattr(args, "log_dir", None)
    return Inbox(log_dir, task["task_id"]) if log_dir else None


async def _within(turn: Awaitable[str], seconds: Optional[float]) -> str:
    """Await one agent turn, bounded when there is a bound to apply."""

    if seconds is None:
        return await turn
    return await asyncio.wait_for(turn, timeout=max(1.0, seconds))


def _first_line(text: str, limit: int = 160) -> str:
    """The first line of prose in a report, for the one-line summary."""

    for line in (text or "").splitlines():
        line = line.strip().lstrip("#").strip()
        if line and line.lower() != REPORT_HEADING.lstrip("#").strip().lower():
            return line if len(line) <= limit else line[: limit - 1] + "…"
    return "reported"


_HANDOFF_BLOCK = re.compile(r"```json\s*\{.*?\}\s*```\s*$", re.DOTALL)


def design_report(
    account: Optional[str],
    outcome: TaskOutcome,
    ttp_id: str,
    timed_out: bool,
) -> str:
    """The report a design task hands back to the conductor.

    The Driver's own closing message, with what the ledger already knows put
    above it. The handoff block is cut: it is addressed to the Reviewer, and
    a conductor reading a wall of SMILES-as-JSON learns nothing from it that
    the candidates do not already say.
    """

    head = [REPORT_HEADING, "", f"Ran TTP `{ttp_id}`."]
    if outcome.objective_profile:
        scores = ", ".join(
            f"{name}={value:.3g}" for name, value in sorted(outcome.objective_profile.items())
        )
        head.append(f"Best molecule scored {scores}.")
    head.append(f"{len(outcome.candidates)} molecule(s) reported.")
    if timed_out:
        head.append("Stopped at the task deadline, with a checkpoint.")

    body = _HANDOFF_BLOCK.sub("", (account or "").strip()).strip()
    if body:
        head += ["", body]
    return "\n".join(head)


#: Findings from the task's own branch put in its brief. A ceiling because
#: the brief has to stay a brief: past this the agent reads the atlas with
#: atlas_branch, which is what the tool is for.
ATLAS_BRIEF_FINDINGS = 8


def _atlas_slice(findings: list[dict[str, Any]]) -> list[str]:
    """What this task's branch of the atlas already holds, for the brief.

    Claims only, with their confidence and evidence: the agent has
    ``atlas_branch`` for the detail, and a brief that opened with ten
    paragraphs of someone else's notes would be read as the task.
    """

    lines = [
        "",
        "What this campaign already knows about your branch. Do not redo "
        "this work -- check it, build on it, or dispute it:",
    ]
    for finding in findings:
        evidence = ", ".join(finding.get("evidence") or []) or "no evidence recorded"
        lines.append(
            f"- [{finding['finding_id']}, {finding['confidence']}] "
            f"{finding['claim']} ({evidence})"
        )
    return lines


def build_brief(
    task: dict[str, Any],
    campaign_goal: Optional[str] = None,
    findings: Optional[list[dict[str, Any]]] = None,
) -> str:
    """The brief handed to a sub-agent for a task that is not a design run.

    The conductor decides how much of this to fill in -- a task can be one
    sentence or a paragraph of context and a named deliverable -- so every
    field is optional and only the ones it wrote are passed on. What is not
    optional is the report: the conductor cannot see the sub-agent's tools,
    its transcript or its reasoning, so a task whose findings stay in the
    transcript was not done as far as the campaign is concerned.

    ``findings`` is what the atlas holds for this task's branch. It is
    handed over rather than left to be fetched because the sub-agent that
    does not know the campaign already answered its question will not think
    to ask: the duplicated work is the failure the atlas exists to stop.
    """

    spec = task.get("spec") or {}
    kind = str(spec.get("kind") or "research")
    lines = [f"You have been given a {kind} task by the agent conducting this campaign."]

    if spec.get("title"):
        lines += ["", f"Task: {str(spec['title']).strip()}"]
    if campaign_goal:
        lines += ["", "The campaign this task belongs to:", str(campaign_goal).strip()]
    if spec.get("context"):
        lines += ["", "Context you have been given:", str(spec["context"]).strip()]
    if spec.get("intent"):
        lines += ["", "What to do:", str(spec["intent"]).strip()]
    if spec.get("deliverable"):
        lines += ["", "What to come back with:", str(spec["deliverable"]).strip()]

    campaign_id = task.get("campaign_id")
    branch = spec.get("workstream")
    if findings:
        lines += _atlas_slice(findings)

    if campaign_id:
        where = f" under branch '{branch}'" if branch else ""
        lines += [
            "",
            f"This campaign is {campaign_id} and your task is {task.get('task_id')}"
            + (f", on branch '{branch}'." if branch else "."),
            "Record what you find with atlas_finding as you find it"
            + where
            + ", one claim per call with its evidence, rather than leaving it "
            "all to the report: the report is read once by the conductor, "
            "and the atlas is what the next agent on this campaign reads. "
            "Use atlas_search before starting a line of work somebody may "
            "already have done, atlas_note to confirm or dispute a finding "
            "that is already there, and atlas_finding with supersedes to "
            "correct one you can show is wrong.",
        ]

    lines += [
        "",
        "Do the whole job yourself with the tools you have, and look things "
        "up rather than recalling them -- a number you remembered is worth "
        "nothing to the chemist reading this. You are one of several agents "
        "working on this campaign in parallel; you cannot see the others and "
        "they cannot see you, so do not wait on anything but your own work.",
        "",
        "You cannot start a reinforcement-learning run on this task. If what "
        "you find argues for one, say so in the report with the profile it "
        "should be scored on, and the conductor will dispatch it.",
        "",
        f"Finish with one message starting with '{REPORT_HEADING}': what you "
        "did, what you found with the numbers and sources behind it, what you "
        "could not settle, and what you would do next. That message is the "
        "only thing the conductor sees -- everything else stays in your own "
        "transcript.",
        "",
        FOLLOW_UP_ASK,
    ]
    return "\n".join(lines)


class CoscientistTaskRunner:
    """Runs one task as a Driver + Reviewer session over MCP.

    Holds the MCP connection and the two agents for the life of one task, then
    tears them down: an RL campaign is long enough that a connection kept
    across tasks would mostly be idle, and a fresh pair starts each task with a
    clean transcript and budget.
    """

    #: The variant TTP of the task running right now, or None between tasks.
    #: The job is looked up through it rather than remembered, because the
    #: agent starts the run itself and nothing tells us when it did.
    _ttp_id: Optional[str] = None

    def current_job_id(self) -> Optional[str]:
        """The run this task has started, if it has started one yet."""

        return job_for_ttp(self._ttp_id) if self._ttp_id else None

    def __init__(self, args: Any, store: CampaignStoreClient) -> None:
        self.args = args
        # Needed for the base TTP, which belongs to the campaign rather than
        # to each of its tasks.
        self.store = store

    def _objectives(self, task: dict[str, Any]) -> Optional[list[str]]:
        """The campaign's objectives, or None if there are none to read.

        None keeps the old behaviour -- report whatever the CSV scored --
        rather than reporting nothing, which would fail the task.

        An empty list has to become None rather than pass through. A
        campaign whose tasks write their own TTPs declares no objectives,
        and narrowing a result to an empty list keeps no scores at all: the
        run finishes, the molecules are there, and nothing says what they
        scored. Seen on LUMI, on a run that had produced 25 molecules.
        """

        try:
            return self.store.campaign(task["campaign_id"]).get("objectives") or None
        except Exception:
            logger.warning(
                "could not read campaign %s objectives; reporting every scored "
                "column", task.get("campaign_id"), exc_info=True,
            )
            return None

    def _campaign_of(self, task: dict[str, Any]) -> dict[str, Any]:
        return self.store.campaign(task["campaign_id"])

    def _atlas_for(self, task: dict[str, Any]) -> list[dict[str, Any]]:
        """What the atlas holds for this task's branch, for its brief.

        A store that cannot answer must not cost the task: the brief is worse
        without the slice, and useless if the claim is thrown away over it.
        """

        branch = (task.get("spec") or {}).get("workstream")
        if not branch:
            return []
        try:
            return self.store.findings(
                task["campaign_id"], branch=branch, limit=ATLAS_BRIEF_FINDINGS
            ).get("findings") or []
        except Exception as exc:
            logger.warning("could not read the atlas for task %s: %s",
                           task.get("task_id"), exc)
            return []

    def _base_ttp_id(self, task: dict[str, Any]) -> Optional[str]:
        """The TTP this task varies, or None when it writes its own.

        No longer an error to have none. A campaign that names a base TTP is
        sweeping a profile somebody already wrote; one that does not is
        asking each task to decide how its own experiment should be scored,
        which is the usual way round.
        """

        return self._campaign_of(task).get("base_ttp_id") or None

    def _backend_for(self, system_prompt: str) -> Any:
        from reinvent4_agent.backends import create_backend

        args = self.args
        return create_backend(
            args.backend,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            gateway_key=args.gateway_key,
            api_version=args.api_version,
            verify_ssl=not args.no_verify_ssl,
            thinking_budget=args.thinking_budget,
            reasoning_effort=args.reasoning_effort,
            system_prompt=system_prompt,
        )

    async def __call__(self, task: dict[str, Any]) -> TaskOutcome:
        spec = task.get("spec") or {}
        overdue = _overdue_seconds(task)
        if overdue is not None:
            # Refused before the model load and the GPU: with no time left a
            # design run would be stopped inside its first RL step, which has
            # nothing to checkpoint, and a research task would be cut off
            # before it had read anything.
            raise RuntimeError(
                f"the task deadline passed {overdue / 60:.0f} minutes before "
                "it was claimed, so there is no time to run it in"
            )

        kind = spec.get("kind") or "design"
        if kind == "lead":
            return await self._run_lead(task)
        if kind != "design":
            return await self._run_general(task)
        return await self._run_design(task)

    async def _run_lead(self, task: dict[str, Any]) -> TaskOutcome:
        """Conduct a workstream of the campaign and report it up.

        In a thread because the loop sleeps between ticks and reads the
        ledger synchronously; the heartbeat keeps the lead task's lease on
        this event loop meanwhile.
        """

        from reinvent4_agent.leading import conduct
        from reinvent4_agent.proposing import LLMProposer

        self._ttp_id = None
        return await asyncio.to_thread(
            conduct, self.store, task, LLMProposer(self._backend_for),
        )

    async def _run_general(self, task: dict[str, Any]) -> TaskOutcome:
        """Do a task that is not a design run, and report back in prose.

        One agent, no Reviewer and no RL job. The Reviewer exists to vet a
        batch of molecules against a profile, and there is neither here; a
        second model reading a literature search would cost a model's worth
        of tokens to say it agreed.

        The report is the deliverable, so the deadline is spent making sure
        there is one: an agent that answered without reporting is asked once
        more, and an agent still working when its time runs out is stopped
        and asked to write up what it has.
        """

        from reinvent4_agent.agent import Agent
        from reinvent4_agent.mcp_client import MCPClient
        from reinvent4_agent.system_prompt import DRIVER_SYSTEM_PROMPT, SUBAGENT_BRIEFING

        args = self.args
        spec = task.get("spec") or {}
        kind = str(spec.get("kind") or "research")
        # No TTP of its own, so nothing for current_job_id to look up: the
        # runs this one starts are the ones it is watched starting.
        self._ttp_id = None
        campaign = await asyncio.to_thread(self._campaign_of, task)
        findings = await asyncio.to_thread(self._atlas_for, task)
        brief = build_brief(task, campaign_goal=campaign.get("goal"), findings=findings)
        budget = _deadline_seconds(task)
        collector = _JobIdCollector()

        async with MCPClient(
            server_command=args.server_command, server_args=args.server_args
        ) as mcp:
            denied = driver_denied_tools(authored=False, kind=kind)
            tools = {tool["name"] for tool in await mcp.list_tools()} - denied

            prompt = DRIVER_SYSTEM_PROMPT + SUBAGENT_BRIEFING
            agent = Agent(
                mcp, self._backend_for(prompt),
                max_tokens=args.max_tokens, log_dir=args.log_dir,
                name=f"subagent-{task['task_id']}", system_prompt=prompt,
                allowed_tools=tools, event_handler=collector,
            )
            agent.withheld = denial_reasons(denied)
            inbox = _inbox_for(args, task)
            agent.inbox = inbox.take if inbox else None

            started = time.monotonic()
            timed_out = False
            try:
                report = await _within(agent.chat(brief), budget)
            except asyncio.TimeoutError:
                # The turn is cancelled mid-flight, which leaves the
                # conversation valid -- Agent.chat answers every tool call it
                # started -- so there is still an agent to ask for a write-up.
                logger.warning("task %s ran out of time; asking for a report",
                               task["task_id"])
                timed_out = True
                report = await _within(agent.chat(WRAP_UP_PROMPT), WRAP_UP_SECONDS)

            if not timed_out and len(report.strip()) < MIN_REPORT_CHARS:
                remaining = None if budget is None else max(
                    WRAP_UP_SECONDS, budget - (time.monotonic() - started)
                )
                logger.info("task %s answered without reporting; asking again",
                            task["task_id"])
                report = await _within(agent.chat(REPORT_NUDGE), remaining)

            # Written after the last tool round had been read, so never seen;
            # a question left unanswered reads as being ignored.
            late = inbox.take() if inbox else []
            if late and not timed_out:
                agent.interject(late)
                remaining = None if budget is None else max(
                    WRAP_UP_SECONDS, budget - (time.monotonic() - started)
                )
                answer = await _within(agent.chat(LATE_MESSAGE_PROMPT), remaining)
                if REPORT_HEADING in answer:
                    report = answer

            logger.info("task %s reported after %d tokens", task["task_id"],
                        agent.total_tokens)

        report = report.strip()
        if not report:
            raise RuntimeError(
                f"the {kind} task produced no report, so nothing it found "
                "can reach the conductor"
            )

        summary = _first_line(report)
        if timed_out:
            summary += " (stopped at the task deadline)"
        return TaskOutcome(
            summary=summary,
            report=report,
            job_ids=collector.job_ids,
            job_id=collector.job_ids[0] if collector.job_ids else None,
            candidates=[],
            follow_ups=parse_follow_ups(report),
        )

    async def _run_design(self, task: dict[str, Any]) -> TaskOutcome:
        from reinvent4_agent.agent import Agent
        from reinvent4_agent.coordinator import REVIEWER_READ_ONLY_TOOLS, Coordinator
        from reinvent4_agent.mcp_client import MCPClient
        from reinvent4_agent.system_prompt import (
            DRIVER_SYSTEM_PROMPT,
            REVIEWER_SYSTEM_PROMPT,
        )

        args = self.args
        spec = task.get("spec") or {}
        # Cleared before anything else: a TTP left over from the task before
        # would answer current_job_id() with the previous task's run for as
        # long as this one takes to prepare its own.
        self._ttp_id = None
        # Both reach the store and the TTP tools over HTTP, and a call that
        # blocks here also stops _heartbeat_forever and _presence_forever.
        campaign = await asyncio.to_thread(self._campaign_of, task)
        base_ttp_id = campaign.get("base_ttp_id") or None
        if base_ttp_id:
            ttp_id = await asyncio.to_thread(
                prepare_ttp_variant,
                base_ttp_id,
                task["task_id"],
                spec.get("weight_overrides"),
                spec.get("transform_overrides"),
            )
        else:
            # Nothing to prepare: the agent writes this one, and the id is
            # reserved for it so the run stays findable either way.
            ttp_id = authored_ttp_id(task["task_id"])
        goal = build_goal(
            task,
            ttp_id,
            campaign_goal=campaign.get("goal"),
            authored=not base_ttp_id,
            findings=await asyncio.to_thread(self._atlas_for, task),
        )
        await self._wait_for_run_room(task)
        budget = _deadline_seconds(task)
        # From here the run is findable by whoever is watching the campaign,
        # not only once this task reports.
        self._ttp_id = ttp_id
        collector = _JobIdCollector()

        async with MCPClient(
            server_command=args.server_command, server_args=args.server_args
        ) as mcp:
            # Everything except the tools that would let it redefine or
            # abandon its own task. Computed from the live tool list rather
            # than hardcoded, so a tool added later is allowed by default and
            # only the denied names have to be maintained.
            denied = driver_denied_tools(authored=not base_ttp_id)
            driver_tools = {
                tool["name"] for tool in await mcp.list_tools()
            } - denied

            driver = Agent(
                mcp, self._backend_for(DRIVER_SYSTEM_PROMPT),
                max_tokens=args.max_tokens, log_dir=args.log_dir,
                # Named after the task so the UI can find this conversation
                # among every other supervisor's in the shared log directory.
                name=f"driver-{task['task_id']}", system_prompt=DRIVER_SYSTEM_PROMPT,
                allowed_tools=driver_tools, event_handler=collector,
            )
            driver.withheld = denial_reasons(denied)
            inbox = _inbox_for(args, task)
            driver.inbox = inbox.take if inbox else None

            coordinator = None
            if getattr(args, "multi_agent", True):
                reviewer = Agent(
                    mcp, self._backend_for(REVIEWER_SYSTEM_PROMPT),
                    max_tokens=args.max_tokens, log_dir=args.log_dir,
                    name=f"reviewer-{task['task_id']}", system_prompt=REVIEWER_SYSTEM_PROMPT,
                    allowed_tools=set(REVIEWER_READ_ONLY_TOOLS),
                )
                coordinator = Coordinator(
                    driver, reviewer,
                    max_turns=spec.get("max_turns") or args.max_turns,
                    token_budget=spec.get("token_budget") or args.token_budget,
                )

            started = time.monotonic()
            if coordinator is not None:
                result = await coordinator.run(goal)
                verdict, turns = result.status, result.turns
                account = result.final_message
            else:
                # One pass. There is no second agent to hand a verdict to, so
                # there is nothing to take turns over.
                account = await driver.chat(goal)
                verdict, turns = "single-agent", 1

            job_id = await asyncio.to_thread(job_for_ttp, ttp_id)
            if not job_id and not base_ttp_id:
                raise RuntimeError(
                    f"the agent was asked to write TTP '{ttp_id}' and start a "
                    f"run against it, and no such run exists (verdict: "
                    f"{verdict}). Either it wrote the TTP under another id, "
                    "in which case the run cannot be found and does not "
                    "count, or it never started one"
                )
            if not job_id and not default_run_substitutes_for(spec):
                raise RuntimeError(
                    f"the agent did not start a run for TTP '{ttp_id}' "
                    f"(verdict: {verdict}), and this task carries an intent "
                    "that may call for a non-default prior -- starting a "
                    "default run here would silently run a different "
                    "experiment"
                )
            if not job_id:
                # The agent finished without starting the run. With a Reviewer
                # the Coordinator took several turns, so there were more
                # chances; a single pass gets one. Nothing about the call is
                # the agent's decision anyway -- the TTP and the step count
                # both come from the task -- so start it here rather than ask
                # again and hope.
                logger.warning(
                    "the agent did not start a run for TTP '%s' (verdict: %s); "
                    "starting it directly", ttp_id, verdict,
                )
                job_id = await self._start_when_room(task, ttp_id, spec.get("max_steps"))

            remaining = None if budget is None else max(
                0.0, budget - (time.monotonic() - started)
            )
            timed_out = await self._monitor(coordinator, goal, job_id, remaining)
            tokens = driver.total_tokens if coordinator is None else result.total_tokens

        try:
            outcome = await self._harvest_or_release(
                task, job_id, verdict, turns, tokens, timed_out
            )
        finally:
            self._ttp_id = None

        outcome.job_ids = list(dict.fromkeys([job_id, *collector.job_ids]))
        outcome.report = design_report(account, outcome, ttp_id, timed_out)
        outcome.follow_ups = parse_follow_ups(outcome.report)
        return outcome

    async def _harvest_or_release(
        self,
        task: dict[str, Any],
        job_id: str,
        verdict: str,
        turns: int,
        tokens: int,
        timed_out: bool,
    ) -> TaskOutcome:
        """Read the result, and release the job if there is no result to read.

        Once the task is failing nothing waits on its job, but it still holds
        a GPU and one of the runner's concurrency slots -- which is how two
        abandoned jobs came to block every later task in a campaign with
        "Maximum concurrent jobs (2) already active".
        """

        try:
            return await asyncio.to_thread(
                self._harvest, task, job_id, verdict, turns, tokens, timed_out
            )
        except Exception:
            logger.warning(
                "task %s failed; stopping its job %s", task["task_id"], job_id
            )
            # The task's own failure is the useful one to report, so a stop
            # that also fails must not replace it.
            with contextlib.suppress(Exception):
                await asyncio.to_thread(self._stop_job, job_id)
            raise

    async def _monitor(
        self,
        coordinator: Optional[Any],
        goal: str,
        job_id: str,
        remaining: Optional[float],
    ) -> bool:
        """Watch the job to its end or the deadline. True if the deadline won.

        On the deadline the job is stopped rather than left running: it asks
        REINVENT4 to checkpoint, so the GPU is handed back and the work can be
        resumed instead of thrown away.
        """

        monitor = (
            coordinator.monitor_campaign(
                goal, job_id=job_id, interval=self.args.monitor_interval
            )
            if coordinator is not None
            else self._watch_job(job_id)
        )
        if remaining is None:
            await monitor
            return False

        try:
            await asyncio.wait_for(monitor, timeout=remaining)
        except asyncio.TimeoutError:
            logger.warning("task deadline reached; stopping job %s", job_id)
            await asyncio.to_thread(self._stop_job, job_id)
            return True
        return False

    async def _watch_job(self, job_id: str) -> None:
        """Wait for the job to stop running. The single-agent monitor.

        ``monitor_campaign`` is the *Reviewer* cycle -- it assesses the run and
        hands the Driver a verdict to act on -- so with no Reviewer there is
        nobody to form that verdict and nothing to act on it. This watches the
        state instead; the task deadline above is what bounds the wait.
        """

        from reinvent4_mcp.jobs.manager import ACTIVE_JOB_STATES
        from reinvent4_mcp.tools.results import get_optimization_status

        # A job owes a result while it is queued *or* running. Waiting only on
        # "running" exits on the first poll, because a job is queued for the
        # moment before it starts -- which harvested a 0-byte CSV and failed
        # the task while its job carried on folding molecules.
        unfinished = {state.value for state in ACTIVE_JOB_STATES}

        while True:
            status = await asyncio.to_thread(get_optimization_status, job_id)

            # Drive off `state`, which is the authoritative field. Testing for
            # an "error" *key* does not work: a job's to_dict() always carries
            # one, set to None when nothing is wrong, so that test is true for
            # every job and returned on the first poll -- harvesting a 0-byte
            # CSV while the job was happily folding molecules.
            state = status.get("state")
            if not state:
                logger.warning(
                    "job %s cannot be described (%s); not waiting on it",
                    job_id, status.get("error") or "no state reported",
                )
                return
            if state not in unfinished:
                logger.info("job %s finished as %s", job_id, state)
                return
            await asyncio.sleep(self.args.monitor_interval)

    async def _wait_for_run_room(self, task: dict[str, Any]) -> None:
        """Hold a design task until the run window has room for its run.

        Before the agents start, not after: a sub-agent that writes its TTP
        and is then refused a run has spent its turns for nothing, and the
        task it fails is requeued with an attempt gone -- so a window that
        closed for an hour would fail every task it held for good. Waiting
        costs a supervisor, which is idle anyway, and no GPU. An unreadable
        window does not hold anything up; the run is refused on its own if
        it has to be, and :meth:`_start_when_room` waits then.
        """

        from reinvent4_mcp.tools.campaign import run_room

        waited = False
        while True:
            room = await asyncio.to_thread(run_room)
            left = _deadline_seconds(task)
            if room is None or room["free"] > 0 or (left is not None and left <= RUN_ROOM_POLL_SECONDS):
                if waited:
                    logger.info("task %s has room for its run now", task["task_id"])
                return
            if not waited:
                logger.info(
                    "task %s waits for a run: %s of %s runs are active%s",
                    task["task_id"], room["active"], room["limit"],
                    f" ({room['reason']})" if room.get("reason") else "",
                )
                waited = True
            await asyncio.sleep(RUN_ROOM_POLL_SECONDS)

    async def _start_when_room(
        self, task: dict[str, Any], ttp_id: str, max_steps: Optional[int]
    ) -> str:
        """Start the task's run, waiting out refusals for want of a slot.

        Several supervisors can see the same last free slot; the ones that
        lose the race wait for the next rather than fail the task. The wait
        is here and not in the worker thread so cancelling the task ends it:
        a thread still retrying would start a run nobody is watching.
        """

        fidelity_tier = (task.get("spec") or {}).get("fidelity_tier") or "exploration"
        while True:
            try:
                return await asyncio.to_thread(
                    self._start_job, ttp_id, max_steps, fidelity_tier
                )
            except RunRefused:
                left = _deadline_seconds(task)
                if left is not None and left <= RUN_ROOM_POLL_SECONDS:
                    raise
                logger.info("no slot for task %s's run yet; trying again", task["task_id"])
                await asyncio.sleep(RUN_ROOM_POLL_SECONDS)

    @staticmethod
    def _start_job(
        ttp_id: str, max_steps: Optional[int], fidelity_tier: str = "exploration"
    ) -> str:
        """Start the task's run without the agent. Returns the job id."""

        from reinvent4_mcp.tools.optimization import run_optimization

        kwargs: dict[str, Any] = {"ttp_id": ttp_id, "fidelity_tier": fidelity_tier}
        if max_steps:
            kwargs["max_steps"] = max_steps

        result = run_optimization(**kwargs)
        if "error" in result and result["error"]:
            # Only a refusal for want of a slot names the runs holding them.
            refused = RunRefused if "active_jobs" in result else RuntimeError
            raise refused(
                f"could not start a run for TTP '{ttp_id}': {result['error']}"
            )
        job_id = result.get("job_id")
        if not job_id:
            raise RuntimeError(
                f"starting a run for TTP '{ttp_id}' returned no job id: {result}"
            )

        logger.info("started job %s for TTP %s directly", job_id, ttp_id)
        return job_id

    @staticmethod
    def _stop_job(job_id: str) -> None:
        from reinvent4_mcp.tools.optimization import stop_optimization

        result = stop_optimization(job_id)
        logger.info("stopped job %s: %s", job_id, result.get("message") or result)

    def _harvest(
        self,
        task: dict[str, Any],
        job_id: str,
        verdict: str,
        turns: int,
        tokens: int,
        timed_out: bool,
    ) -> TaskOutcome:
        """Read the achieved profile out of the job's own results."""

        from reinvent4_mcp.tools.results import get_top_candidates

        top = get_top_candidates(job_id, top_n=25)
        if "error" in top:
            raise RuntimeError(f"job {job_id} produced no readable results: "
                               f"{top['error']}")

        # Each molecule names its run, which is where its complex is found: a
        # lead's result pools molecules from the runs of all its sub-tasks.
        candidates = [{**c, "job_id": job_id} for c in top.get("candidates") or []]
        if not candidates:
            raise RuntimeError(f"job {job_id} finished without a single molecule")
        # Restricted to the campaign's own objectives. A results CSV carries a
        # column per scorer diagnostic too -- std, confidence_score,
        # affinity_probability_binary and friends -- and reporting those as
        # objectives put a dozen keys on every result where three belonged.
        profile = objective_profile(candidates, self._objectives(task))

        summary = (
            f"verdict={verdict}, turns={turns}, "
            f"tokens={tokens}, candidates={len(candidates)}"
        )
        if not profile:
            summary += ", no per-objective scores were readable"
        if timed_out:
            summary += ", stopped at the task deadline (checkpointed)"

        return TaskOutcome(
            objective_profile=profile,
            candidates=candidates,
            job_id=job_id,
            summary=summary,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[list[str]] = None) -> Any:
    """Reuse the CLI's parser so the agent is configured the same way here."""

    from reinvent4_agent.cli import parse_args

    return parse_args(argv or [])


async def _run(args: Any) -> None:
    store = CampaignStoreClient(
        os.getenv("CAMPAIGN_STORE_URL", STORE_URL), supervisor_id()
    )
    lease = os.getenv("SUPERVISOR_LEASE_SECONDS")
    runner = CoscientistTaskRunner(args, store)
    supervisor = Supervisor(
        store,
        runner,
        campaign_id=os.getenv("SUPERVISOR_CAMPAIGN_ID") or None,
        lease_seconds=float(lease) if lease else None,
        job_id_source=runner.current_job_id,
    )

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        # A pod being deleted should stop claiming and let the task in hand
        # finish or lapse, not die mid-campaign with the ledger none the wiser.
        loop.add_signal_handler(sig, supervisor.stop)

    await supervisor.run_forever()


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # Before anything reads a service URL. A no-op on Kubernetes, where the
    # ConfigMap has already set every variable this could fill in.
    apply_registry()

    args = _parse_args(argv)
    from reinvent4_agent.cli import _resolve_api_key

    args.api_key = _resolve_api_key(args)
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
