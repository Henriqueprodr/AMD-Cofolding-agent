"""Campaign tools: the conductor's view of a supervisor fleet.

These are what an interactive conductor calls to open a campaign, see what the
fleet is doing, and steer it. Everything reads through ``campaign-store``, so
the answers are the same ones the supervisors are acting on rather than a
second opinion held in the agent's context.

Every function returns a plain dict and turns a failure into
``{"error": ...}`` rather than raising, matching the rest of the tool surface:
a conductor that asks about a campaign that does not exist should be told so
and carry on, not have its turn aborted.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _store():
    from reinvent4_agent.campaign_client import CampaignStoreClient

    return CampaignStoreClient()


def _guard(what: str, call) -> Any:
    from reinvent4_agent.campaign_client import LeaseLost, StoreUnavailable

    try:
        return call()
    except (StoreUnavailable, LeaseLost) as exc:
        logger.error("%s: %s", what, exc)
        return {"error": f"{what}: {exc}"}


def start_campaign(
    goal: str,
    base_ttp_id: Optional[str] = None,
    objectives: Optional[list[str]] = None,
    concurrency: int = 1,
    hours: Optional[float] = None,
    grace_hours: float = 0.0,
) -> dict[str, Any]:
    """Open a campaign: a goal, a pool size, and a wall-clock budget.

    Opening a campaign does not dispatch anything or start any supervisors --
    the conductor loop does that, so a campaign can be set up and inspected
    before it begins spending GPU.

    Parameters
    ----------
    goal:
        What the campaign is for, in a sentence. Carried in the ledger so a
        conductor restarting mid-campaign knows what it was doing.
    base_ttp_id:
        The TTP every task starts from, each running a variant of it with its
        own objective weights. Usually omitted: without one, you say what
        each task is for and the agent running it writes the profile that
        measures it. Name a base TTP only when the campaign really is a
        sweep of one profile and every task must be scored the same way --
        it is the stricter mode, and it limits tasks to re-weighting
        objectives somebody has already thought of.
    objectives:
        The objective names every result is reported on. Fixed at creation,
        so results stay comparable mid-campaign. Defaults to the base TTP's
        own objectives, and to none at all when there is no base TTP --
        tasks then bring their own, and each result is read in its own terms.
    concurrency:
        How many supervisors to hold in flight.
    hours:
        Length of the dispatch window. Omit for no deadline.
    grace_hours:
        Drain window after the dispatch window closes, during which
        in-flight tasks finish but nothing new is enqueued.

    Returns
    -------
    dict
        The campaign, including its id and the two deadlines.
    """

    from reinvent4_agent.conductor import start_campaign as _start

    return _guard(
        "Could not open the campaign",
        lambda: _start(
            _store(), goal, base_ttp_id, objectives,
            concurrency=concurrency, hours=hours, grace_hours=grace_hours,
        ),
    )


def get_campaign_state(campaign_id: str) -> dict[str, Any]:
    """Everything about a campaign in one call.

    Task counts by state, the supervisors currently holding tasks with their
    weight vectors and lease remaining, how many results are in, and how long
    is left in the dispatch and drain windows.

    ``window.phase`` follows the clock rather than the stored state, so a
    campaign whose conductor died and missed its own deadline still reports
    that it should be draining.
    """

    return _guard(
        f"Could not read campaign {campaign_id}",
        lambda: _store().status(campaign_id),
    )


def list_campaigns() -> dict[str, Any]:
    """Every campaign in the ledger, newest first."""

    result = _guard("Could not list campaigns", lambda: _store().campaigns())
    if isinstance(result, dict):
        return result
    return {"campaigns": result, "n_campaigns": len(result)}


def enqueue_task(
    campaign_id: str,
    intent: str,
    kind: str = "design",
    title: Optional[str] = None,
    context: Optional[str] = None,
    deliverable: Optional[str] = None,
    weight_overrides: Optional[dict[str, float]] = None,
    transform_overrides: Optional[dict[str, dict[str, Any]]] = None,
    fidelity_tier: str = "exploration",
    max_steps: Optional[int] = None,
    deadline: Optional[str] = None,
    derived_from: Optional[list[str]] = None,
    workstream: Optional[str] = None,
    priority: int = 0,
    concurrency: Optional[int] = None,
) -> dict[str, Any]:
    """Give one task to one sub-agent.

    The sub-agent that claims it does the whole job and reports back in
    prose; the report is read with ``get_task_reports``. Tasks queued
    together run at the same time and cannot see each other, so they should
    answer different questions.

    Parameters
    ----------
    campaign_id:
        Campaign to add the task to. Must still be in its dispatch window.
    intent:
        What to do, in prose, handed to the sub-agent verbatim. This is the
        task. Say what the work is for and let the sub-agent decide how --
        it can look a target or a compound up and pick its own tools. For a
        design run this is also where chemistry that the weights cannot
        express belongs, such as which chemotype to hold, since that is a
        choice of generative prior.
    kind:
        What sort of work it is, which decides what the sub-agent can do:

        * ``design`` -- a reinforcement-learning run. The sub-agent writes
          or is given a TTP, runs it, and reports the molecules with their
          scores. The run can build de novo, from seed molecules (Mol2Mol),
          by varying the substituents on a fixed scaffold (LibInvent), or by
          joining two fragments with a linker (LinkInvent). Say which in
          ``intent`` when the task turns on it, since it decides the prior
          and the starter tool.
        * ``research`` -- reading, target intelligence, a chain of
          prediction and scoring tools.
        * ``analysis`` -- making sense of results that already exist.
        * ``retrosynthesis`` -- route planning and building-block checks.
        * ``question`` -- something answerable without starting anything.
        * ``lead`` -- a whole workstream. The sub-agent opens a campaign of
          its own under this one, keeps ``concurrency`` sub-agents of this
          pool busy on it, and reports one synthesis with follow-ups. For
          when there are more questions than one conductor can read.

        Everything but ``design`` is refused the job-starting tools, because
        nothing would be watching the run or reading its results. A sub-agent
        that finds a run is needed says so in its report.
    title:
        A few words for the person watching the campaign, so a task is
        recognisable without reading the whole brief.
    context:
        What the sub-agent needs to know but could not find out: what the
        series is, what has already been tried, what another task found. It
        cannot see the conversation or the other sub-agents.
    deliverable:
        What to come back with, when the shape of the answer matters -- "a
        ranked list with references", "the three routes with their
        building-block availability".
    weight_overrides:
        Design tasks on a campaign with a base TTP: objective name to
        weight. Names must match the base TTP's objectives -- an unknown one
        fails the task rather than running with the wrong weights.
    transform_overrides:
        Objective name to a transform config, for reshaping how a raw value
        maps to [0, 1].
    fidelity_tier:
        ``exploration`` reads and writes the scoring cache; ``validation``
        bypasses the read so a reported result is never a cached sample.
    max_steps:
        RL steps for a design task. Budget it against the step latency.
    deadline:
        ISO timestamp by which the task must have stopped. A design task
        turns this into a checkpointing stop; any other kind is asked to
        write up what it has.
    derived_from:
        Ids of this campaign's tasks whose reports led to this one. The
        user's view of the campaign draws them as the lines from one
        experiment to the next; without them it is a heap of runs. Empty for
        the first tasks, which had nothing to go on.
    workstream:
        Id of the plan's workstream this task works on.
    priority:
        Higher is claimed first. Leave at 0 unless the task must jump the queue.
    concurrency:
        For a ``lead``: how many of the pool's sub-agents its workstream may
        hold at once. Ignored by every other kind.
    """

    spec = task_spec(
        intent, kind=kind, title=title, context=context, deliverable=deliverable,
        weight_overrides=weight_overrides, transform_overrides=transform_overrides,
        fidelity_tier=fidelity_tier, max_steps=max_steps, deadline=deadline,
        derived_from=derived_from, workstream=workstream, priority=priority,
        concurrency=concurrency,
    )
    return _guard(
        f"Could not enqueue a task on campaign {campaign_id}",
        lambda: _store().enqueue_task(campaign_id, **spec),
    )


#: What a task in a batch may say; the rest of a model's dict is not a field.
TASK_FIELDS = (
    "kind", "title", "context", "deliverable", "weight_overrides",
    "transform_overrides", "fidelity_tier", "max_steps", "deadline",
    "derived_from", "workstream", "priority", "concurrency",
)


def task_spec(intent: str, **fields: Any) -> dict[str, Any]:
    """The ledger's spec for one task, without the fields left empty."""

    spec: dict[str, Any] = {
        "kind": fields.get("kind") or "design",
        "intent": intent,
        "fidelity_tier": fields.get("fidelity_tier") or "exploration",
    }
    for name in TASK_FIELDS:
        value = fields.get(name)
        if name in spec or value in (None, "", [], {}, 0):
            continue
        spec[name] = list(value) if name == "derived_from" else value
    return spec


def enqueue_tasks(campaign_id: str, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Queue several tasks in one call, all of them or none.

    Each task is a dict with the fields ``enqueue_task`` takes, ``intent``
    required. Refused whole when any one is malformed, so the queue never
    holds half of what was meant.
    """

    if not tasks:
        return {"error": "The batch is empty: give at least one task"}
    specs = []
    for number, task in enumerate(tasks, 1):
        if not isinstance(task, dict):
            return {"error": f"Task {number} of the batch is not an object; nothing was queued"}
        intent = str(task.get("intent") or "").strip()
        if not intent:
            return {"error": f"Task {number} of the batch has no intent; nothing was queued"}
        unknown = sorted(set(task) - {"intent", *TASK_FIELDS})
        if unknown:
            return {"error": f"Task {number} of the batch has fields no task has "
                             f"({', '.join(unknown)}); nothing was queued"}
        specs.append(task_spec(intent, **{k: v for k, v in task.items() if k != "intent"}))

    outlook = _run_outlook(campaign_id, specs)

    queued = _guard(
        f"Could not enqueue tasks on campaign {campaign_id}",
        lambda: _store().enqueue_tasks(campaign_id, specs),
    )
    if isinstance(queued, dict):
        return queued
    return {
        "campaign_id": campaign_id,
        "queued": [
            {"task_id": t["task_id"], "title": (t.get("spec") or {}).get("title")}
            for t in queued
        ],
        "n_queued": len(queued),
        **({"runs": outlook} if outlook else {}),
    }


#: Who a task the conductor is doing itself belongs to. A reserved name
#: rather than the host name a supervisor uses, so the UI can tell the
#: conductor's own work from a sub-agent's and label it "C".
CONDUCTOR_OWNER = "conductor"


def start_own_task(
    campaign_id: str,
    intent: str,
    kind: str = "design",
    title: Optional[str] = None,
    context: Optional[str] = None,
    deliverable: Optional[str] = None,
    derived_from: Optional[list[str]] = None,
    workstream: Optional[str] = None,
) -> dict[str, Any]:
    """Record an experiment you are about to run yourself, and take it.

    ``enqueue_task`` gives work away; this keeps it. Use it before starting a
    run or a line of work of your own, so that what you do is in the ledger
    beside what sub-agents do: one trajectory the user can read, one set of
    reports, one pool of molecules, whether or not there are sub-agents.

    Working alone this is how a campaign has anything in it at all. Pair it
    with ``finish_own_task`` when the work is done -- a task left open shows
    as still running.

    Parameters
    ----------
    campaign_id:
        Campaign to record it under.
    intent:
        What the experiment is, in prose. The same field ``enqueue_task``
        hands to a sub-agent, except that here you are the one doing it.
    kind:
        ``design`` for a run, or ``research``, ``analysis``,
        ``retrosynthesis``, ``question``. Nothing is withheld on the strength
        of it -- you already have the tools -- but it is what the user's view
        of the campaign groups and colours the work by.
    title:
        A few words naming it, which is what the trajectory labels it.
    context:
        What the experiment rests on, for whoever reads the campaign later.
    deliverable:
        What you mean to come back with.
    derived_from:
        Ids of the tasks whose results led to this one, yours or a
        sub-agent's. Without them the campaign is a heap of experiments
        rather than a line of reasoning.
    workstream:
        Id of the plan's workstream this works on.

    Returns
    -------
    dict
        ``{"task_id": ...}``, to pass to ``finish_own_task``.
    """

    spec = task_spec(
        intent, kind=kind, title=title, context=context,
        deliverable=deliverable, derived_from=derived_from, workstream=workstream,
    )

    def take() -> dict[str, Any]:
        store = _store()
        store.owner = CONDUCTOR_OWNER
        task = store.enqueue_task(campaign_id, **spec)
        task_id = task["task_id"]
        # No lease: the conductor is not a pool member, and a swept lease
        # would requeue work that is being done for a sub-agent to redo.
        store.claim_named(task_id)
        # Include campaign_id so the caller can pass it to finish_own_task and
        # any get_task_reports call without having to remember it separately.
        return {"task_id": task_id, "title": spec.get("title"), "kind": spec["kind"],
                "campaign_id": campaign_id}

    return _guard(f"Could not record your own task on campaign {campaign_id}", take)


def finish_own_task(
    task_id: str,
    report: str,
    summary: Optional[str] = None,
    candidates: Optional[list[dict[str, Any]]] = None,
    job_ids: Optional[list[str]] = None,
    objective_profile: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    """Close an experiment you ran yourself, with what it found.

    The counterpart of ``start_own_task``. What you post here is what the
    campaign's reports, its pooled molecules and its objective progress are
    computed from, so a result posted is a result the user can see in the
    Campaign tab without reading the conversation back.

    Parameters
    ----------
    task_id:
        From ``start_own_task``.
    report:
        What happened and what it means, in prose. The account the campaign
        keeps of this experiment.
    summary:
        One line, for the trajectory.
    candidates:
        The molecules it put forward, each ``{"smiles", "total_score",
        "components"}`` as a sub-agent reports them. These are what the
        campaign's Molecules view pools.
    job_ids:
        Every run the experiment started.
    objective_profile:
        Objective name to the value reached, for the campaign's progress.
    """

    def post() -> dict[str, Any]:
        from reinvent4_agent.campaign_client import TaskOutcome

        store = _store()
        store.owner = CONDUCTOR_OWNER
        ids = list(job_ids or [])
        return store.post_result(task_id, TaskOutcome(
            objective_profile=objective_profile or {},
            candidates=candidates or [],
            job_id=ids[0] if ids else None,
            summary=summary,
            report=report,
            job_ids=ids,
        ))

    return _guard(f"Could not post your result for task {task_id}", post)


def _run_outlook(campaign_id: str, specs: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """How many of a batch's design tasks can start a run now, or None.

    Said rather than enforced. A fan-out is judged by its grouped demand
    before any of it runs, but refusing the batch would be wrong here: a
    task waiting in the ledger holds neither a GPU nor a supervisor, and a
    ready queue behind the team is what the batch is for. What a design
    task beyond the room does is wait, so the answer says how many will --
    a comparison whose branches start an hour apart should be read as one.
    Only design tasks start a run, and the ones of this campaign already
    waiting for one are ahead of the batch.
    """

    demand = sum(1 for spec in specs if spec.get("kind") == "design")
    if not demand:
        return None
    room = run_room()
    if room is None:
        return None
    tasks = _guard(f"Could not read the tasks of campaign {campaign_id}",
                   lambda: _store().tasks(campaign_id))
    waiting = 0 if isinstance(tasks, dict) else sum(
        1 for t in tasks
        if (t.get("spec") or {}).get("kind", "design") == "design"
        and (t.get("state") == "queued" or (t.get("state") == "claimed" and not t.get("job_id")))
    )
    start_now = min(demand, max(0, room["free"] - waiting))
    outlook = {"design_tasks": demand, "can_start_now": start_now, "window": room}
    if start_now < demand:
        why = f" ({room['reason']})" if room.get("reason") else ""
        outlook["detail"] = (
            f"{demand - start_now} of the {demand} design tasks will wait for a run: "
            f"{room['active']} of {room['limit']} runs are active or waiting{why}, "
            f"and {waiting} design tasks of this campaign were already waiting. "
            "They start as runs finish; compare their reports once all are in."
        )
    return outlook


def run_room() -> Optional[dict[str, Any]]:
    """Runs that can start now, from the run window, or None if unknown.

    A runner that reports no window enforces only its fixed limit, so that
    limit is the room.
    """

    jobs = _runner_concurrency()
    window = jobs.get("run_window")
    if window:
        return window
    ceiling, active = jobs.get("max_concurrent_jobs"), jobs.get("active_jobs")
    if not isinstance(ceiling, int) or not isinstance(active, int):
        return None
    return {"limit": ceiling, "ceiling": ceiling, "active": active,
            "free": max(0, ceiling - active), "reason": None}


def get_task_reports(
    campaign_id: str, task_id: Optional[str] = None
) -> dict[str, Any]:
    """What the sub-agents reported back.

    Each report is the sub-agent's own account of its task: what it did, what
    it found, what it could not settle and what it would do next. This is the
    only thing that crosses from a sub-agent to you -- their tool calls and
    transcripts are their own -- so read these before deciding what to
    dispatch next.

    Molecules are not included: ``get_supervisor_result`` has the candidates
    and scores for one task, and a dozen reports with twenty-five molecules
    each would not fit in a turn.

    Parameters
    ----------
    campaign_id:
        Campaign whose reports to read.
    task_id:
        One task, when you only want that one. Omit for every task that has
        reported.
    """

    results = _guard(
        f"Could not read results for campaign {campaign_id}",
        lambda: _store().results(campaign_id),
    )
    if isinstance(results, dict):
        return results

    reports = [
        {
            "task_id": result["task_id"],
            "job_ids": result.get("job_ids") or [],
            "objective_profile": result.get("objective_profile") or {},
            "n_candidates": len(result.get("candidates") or []),
            "summary": result.get("summary"),
            "report": result.get("report"),
            "posted_at": result.get("posted_at"),
        }
        for result in results
        if task_id is None or result["task_id"] == task_id
    ]
    if task_id is not None and not reports:
        return {"error": f"Campaign {campaign_id} has no result for task {task_id}"}

    unreported = [r["task_id"] for r in reports if not r["report"]]
    answer: dict[str, Any] = {
        "campaign_id": campaign_id,
        "reports": reports,
        "n_reports": len(reports),
    }
    if unreported:
        # A result posted by an older supervisor, or a run whose agent said
        # nothing. Named rather than silently absent, so a conductor does not
        # read "no report" as "nothing found".
        answer["without_a_report"] = unreported
    return answer


def set_campaign_plan(campaign_id: str, workstreams: list[dict[str, Any]]) -> dict[str, Any]:
    """Split the goal into the workstreams a team would divide it into.

    Each is ``{"id", "name", "question"}``, optionally ``"share"`` (the rough
    fraction of the team it deserves) and ``"status"`` (open, settled or
    dropped). Replaces the previous plan, so send all of them when one
    changes. Tasks name theirs in ``workstream``.
    """

    answer = _guard(
        f"Could not set the plan of campaign {campaign_id}",
        lambda: _store().set_plan(campaign_id, list(workstreams)),
    )
    if "error" in answer:
        return answer
    return {"campaign_id": campaign_id, "plan": answer.get("plan")}


def get_campaign_coverage(campaign_id: str) -> dict[str, Any]:
    """How the work spreads over the plan, per workstream.

    What is queued, running and done in each, its latest finding and how
    many proposals wait in it. ``gaps`` are open workstreams nobody is
    working on now. The view to decide from when there are more reports than
    can be read.
    """

    return _guard(
        f"Could not read the coverage of campaign {campaign_id}",
        lambda: _store().coverage(campaign_id),
    )


def list_proposals(campaign_id: str) -> dict[str, Any]:
    """The next tasks sub-agents proposed at the end of their reports, still open.

    A sub-agent sees only its own task, so these are candidates: rank them
    against the campaign, take the ones worth a sub-agent with
    ``take_proposals`` and drop the rest with ``drop_proposals``.
    """

    proposals = _guard(
        f"Could not read the proposals of campaign {campaign_id}",
        lambda: _store().proposals(campaign_id),
    )
    if isinstance(proposals, dict):
        return proposals
    return {
        "campaign_id": campaign_id,
        "proposals": [
            {
                "proposal_id": p["proposal_id"],
                "from_task_id": p["from_task_id"],
                **{key: p["spec"].get(key) for key in ("kind", "title", "intent", "workstream")},
            }
            for p in proposals
        ],
        "n_open": len(proposals),
    }


def take_proposals(campaign_id: str, proposal_ids: list[str], priority: int = 0) -> dict[str, Any]:
    """Queue proposals as tasks, each derived from the task that proposed it."""

    tasks = _guard(
        f"Could not take proposals on campaign {campaign_id}",
        lambda: _store().take_proposals(campaign_id, list(proposal_ids), priority),
    )
    if isinstance(tasks, dict):
        return tasks
    return {
        "campaign_id": campaign_id,
        "queued": [
            {"task_id": t["task_id"], "title": (t.get("spec") or {}).get("title")}
            for t in tasks
        ],
    }


def drop_proposals(campaign_id: str, proposal_ids: list[str]) -> dict[str, Any]:
    """Set proposals aside, so they are not offered again."""

    return _guard(
        f"Could not drop proposals on campaign {campaign_id}",
        lambda: _store().drop_proposals(campaign_id, list(proposal_ids)),
    )


def list_supervisors(campaign_id: Optional[str] = None) -> dict[str, Any]:
    """The pool: how many supervisors exist, and how many could take work now.

    ``idle`` is the number that would answer a dispatch immediately. That is
    not the same as ``concurrency``, which is only the ceiling this campaign
    asked for, nor the Deployment's replica count, which counts pods that are
    still starting or already shutting down. A supervisor appears here because
    it reported in, so one that crashed drops out after its presence expires.

    Dead presences and expired leases are both swept before answering, so this
    is who is alive rather than who was last seen.

    Parameters
    ----------
    campaign_id:
        Narrow to the supervisors eligible for one campaign, and include that
        campaign's in-flight tasks. Supervisors not pinned to any campaign
        count towards every campaign's pool. Omit for the whole fleet.
    """

    pool = _guard("Could not read the supervisor pool",
                  lambda: _store().supervisors(campaign_id))
    if "error" in pool:
        return pool

    if campaign_id is None:
        return pool

    status = _guard(
        f"Could not read campaign {campaign_id}",
        lambda: _store().status(campaign_id),
    )
    if "error" in status:
        return status
    return {
        **pool,
        "in_flight": status["in_flight"],
        "n_in_flight": len(status["in_flight"]),
        "concurrency": status["campaign"]["concurrency"],
    }


def get_supervisor_result(campaign_id: str, task_id: str) -> dict[str, Any]:
    """What one finished task achieved: its objective profile and molecules.

    Scores are narrowed to the campaign's declared objectives. A scorer also
    emits per-molecule diagnostics, which are in the job's results CSV via
    ``read_results`` rather than here.
    """

    results = _guard(
        f"Could not read results for campaign {campaign_id}",
        lambda: _store().results(campaign_id),
    )
    if isinstance(results, dict):
        return results

    for result in results:
        if result["task_id"] == task_id:
            return result
    return {"error": f"Campaign {campaign_id} has no result for task {task_id}"}


#: Files worth naming in a job directory, in the order a person reads them.
#: Everything else there (``tb_logs_*``, ``_config.json``, checkpoint shards)
#: is machinery rather than something to open.
JOB_FILES_OF_INTEREST = (
    "results_*.csv",
    "reinvent.log",
    "config.toml",
    "job_metadata.json",
    "stage.chkpt",
    "seed.smi",
)


def _job_files(job_dir: Any) -> list[dict[str, Any]]:
    """The notable files in one job directory, with sizes, newest pattern first."""

    files = []
    seen: set[str] = set()
    for pattern in JOB_FILES_OF_INTEREST:
        for path in sorted(job_dir.glob(pattern)):
            if path.name in seen or not path.is_file():
                continue
            seen.add(path.name)
            files.append({"name": path.name, "bytes": path.stat().st_size})
    return files


def get_campaign_files(campaign_id: str, limit: int = 20) -> dict[str, Any]:
    """Where a campaign's output actually is on disk, task by task.

    The ledger records what each task achieved; this says which directory it
    wrote while achieving it, so a result can be opened rather than only read
    about. Paths are as the container sees them -- the same volume the agent,
    the runner and the scorers share.

    A task with no directory either never got as far as starting a job or ran
    before this volume, and says which.

    Tasks come back newest first and carry no scores, so position here says
    nothing about which run was any good -- ``get_supervisor_result`` says
    what one achieved, and ``get_top_candidates`` ranks the molecules within
    one job.

    Parameters
    ----------
    campaign_id:
        Campaign to locate.
    limit:
        How many tasks to report, newest first. ``n_tasks`` is the true total.
    """

    from pathlib import Path

    from reinvent4_mcp.config import get_config

    tasks = _guard(
        f"Could not read tasks for campaign {campaign_id}",
        lambda: _store().tasks(campaign_id),
    )
    if isinstance(tasks, dict):
        return tasks

    results = _guard(
        f"Could not read results for campaign {campaign_id}",
        lambda: _store().results(campaign_id),
    )
    job_of = (
        {}
        if isinstance(results, dict)
        else {r["task_id"]: r.get("job_id") for r in results}
    )

    jobs_root = Path(get_config().jobs.working_dir)
    newest_first = sorted(
        tasks, key=lambda t: t.get("created_at") or "", reverse=True
    )

    entries = []
    for task in newest_first[:limit]:
        job_id = job_of.get(task["task_id"])
        entry: dict[str, Any] = {
            "task_id": task["task_id"],
            "state": task["state"],
            "job_id": job_id,
            "created_at": task.get("created_at"),
            "finished_at": task.get("finished_at"),
        }
        if not job_id:
            entry["directory"] = None
            entry["note"] = (
                "no job id in the ledger: the task did not get as far as "
                f"starting one ({task['state']})"
            )
        else:
            job_dir = jobs_root / f"job_{job_id}"
            entry["directory"] = str(job_dir)
            if job_dir.is_dir():
                entry["files"] = _job_files(job_dir)
            else:
                entry["note"] = "the ledger has this job but the directory is gone"
        entries.append(entry)

    return {
        "campaign_id": campaign_id,
        "jobs_root": str(jobs_root),
        "shared_volume": str(
            Path(os.environ.get("IP_SHARED_DIR", "/shared"))
        ),
        "tasks": entries,
        "n_tasks": len(tasks),
        "n_listed": len(entries),
        "hint": (
            "Cofolded structures are on the shared volume, not in the job "
            "directory -- use list_shared_assets(prefix=...) for those. Row "
            "data is in results_*.csv, which read_results(job_id) parses."
        ),
    }


def set_concurrency(campaign_id: str, concurrency: int) -> dict[str, Any]:
    """Change how many supervisors the campaign holds in flight.

    Recorded in the ledger rather than only in the conductor, so the number
    survives a restart. The conductor loop scales the supervisor Deployment to
    match on its next pass; this call on its own does not move pods.
    """

    return _guard(
        f"Could not set concurrency on campaign {campaign_id}",
        lambda: _store().set_concurrency(campaign_id, concurrency),
    )


def cancel_supervisor(task_id: str) -> dict[str, Any]:
    """Take a task back from the supervisor running it.

    The holder finds out on its next heartbeat and abandons the work without
    posting a result. Its RL job is not stopped by this call -- use
    ``stop_optimization`` on the job id if the GPU is what you need back.
    """

    return _guard(
        f"Could not cancel task {task_id}",
        lambda: _store().cancel_task(task_id),
    )


def get_pool_capacity(campaign_id: Optional[str] = None) -> dict[str, Any]:
    """Whether there is room for more work, and which limit is the binding one.

    Two separate things have to be true to dispatch: a supervisor must be free
    to claim, and Boltz must have queue room to score what it starts. Either
    one alone is misleading -- an idle supervisor handed a task when the
    scorer is saturated just moves the wait, and a quiet scorer is no use with
    nothing alive to dispatch to.

    ``queue_pressure`` is queued batches per healthy replica. ``None`` means
    the gateway did not answer, which is not the same as idle.

    Parameters
    ----------
    campaign_id:
        Count only the supervisors eligible for this campaign. Omit for the
        whole fleet.
    """

    from reinvent4_agent.conductor import GATEWAY_STATS_URL, queue_pressure

    pool = _guard("Could not read the supervisor pool",
                  lambda: _store().supervisors(campaign_id))
    idle = None if "error" in pool else pool["idle"]

    pressure = queue_pressure()
    if not GATEWAY_STATS_URL:
        # No gateway in this deployment, so its queue is not a limit here.
        # Distinct from "it did not answer": one is a deployment without a
        # proxy, the other is a proxy that is down, and only the second is a
        # reason to hold dispatch back.
        scorer = {
            "queue_pressure": None,
            "verdict": "not applicable",
            "detail": "no boltz gateway is configured here, so its queue "
                      "depth is not a constraint",
        }
    else:
        scorer = {
            "queue_pressure": None if pressure is None else round(pressure, 2),
            "verdict": "unknown" if pressure is None
                       else ("room" if pressure <= 2.0 else "saturated"),
            "detail": "the boltz gateway did not answer; treat capacity as "
                      "unknown rather than free" if pressure is None
                      else f"{pressure:.1f} batches in hand per healthy Boltz "
                           "replica, queued and in flight together",
        }

    runs = run_room()

    if idle is None:
        verdict, limit = "unknown", "supervisor pool unreadable"
    elif idle == 0:
        verdict, limit = "no", "every supervisor is busy"
    elif scorer["verdict"] == "saturated":
        verdict, limit = "no", "boltz is saturated"
    elif scorer["verdict"] == "unknown":
        verdict, limit = "unknown", "boltz queue depth unreadable"
    elif runs is not None and runs["free"] == 0:
        # A supervisor is free but no run could start: research and
        # question tasks still can, which is why this is not a plain no.
        verdict, limit = "not design", "the run window is full" + (
            f": {runs['reason']}" if runs.get("reason") else ""
        )
    else:
        verdict, limit = "yes", None

    return {
        "can_dispatch": verdict,
        "binding_limit": limit,
        "supervisors": {
            "idle": idle,
            "busy": None if "error" in pool else pool["busy"],
            "alive": None if "error" in pool else pool["n_alive"],
        },
        "scorer": scorer,
        "runs": runs,
        "ceiling": max_supervisors(),
    }


def max_supervisors() -> dict[str, Any]:
    """The largest supervisor pool this deployment could run.

    A different question from ``get_pool_capacity``'s: that one is about
    right now, this is the ceiling somebody picks a number under. Two things
    cap it and the smaller wins -- what the scheduler would start, and what
    the job backend will accept, since supervisors past the runner's
    concurrency limit have their tasks refused and spend their time retrying.

    ``max_supervisors`` is ``None`` when neither can be read. That is not the
    same as unlimited, and a caller offering somebody a choice should say so
    rather than pick a number.
    """

    from reinvent4_agent.pool import pool_capacity

    scheduler = pool_capacity()
    jobs = _runner_concurrency()

    limits = {
        "scheduler": {
            "max_supervisors": scheduler.max_supervisors,
            "basis": scheduler.basis,
            **({"detail": scheduler.detail} if scheduler.detail else {}),
        },
        "jobs": jobs,
    }

    candidates = [
        (value, name) for name, value in (
            ("scheduler", scheduler.max_supervisors),
            ("jobs", jobs.get("max_concurrent_jobs")),
        ) if isinstance(value, int)
    ]
    if not candidates:
        return {
            "max_supervisors": None,
            "binding_limit": None,
            "detail": "neither the scheduler nor the job backend would say; "
                      "this is unknown, not unlimited",
            "limits": limits,
        }

    ceiling, binding = min(candidates)
    return {
        "max_supervisors": ceiling,
        "binding_limit": binding,
        "detail": (
            f"the {'scheduler' if binding == 'scheduler' else 'job backend'} "
            f"is the binding limit at {ceiling}"
        ),
        "limits": limits,
    }


def _runner_concurrency() -> dict[str, Any]:
    """How many RL jobs can run at once, from the runner or from config.

    Asked of the runner rather than read from config when there is one: the
    runner is the process that enforces it, and in a split deployment its
    config is not necessarily the one this process loaded.
    """

    from reinvent4_mcp.config import get_config

    config = get_config()
    if not config.runner.enabled:
        from reinvent4_mcp.jobs.manager import local_run_window

        window = local_run_window()
        return {
            "max_concurrent_jobs": config.jobs.max_concurrent_jobs,
            "active_jobs": window["active"] if window else 0,
            "run_window": window,
            "source": "local job manager",
        }

    import requests

    try:
        health = requests.get(
            f"{config.runner.base_url.rstrip('/')}/health", timeout=10
        ).json()
    except Exception as exc:  # noqa: BLE001 - any failure is "could not say"
        logger.warning("could not ask the runner its job limit: %s", exc)
        return {"max_concurrent_jobs": None,
                "source": f"reinvent-runner at {config.runner.base_url}",
                "error": str(exc)}

    value = health.get("max_concurrent_jobs")
    window = health.get("run_window")
    return {
        "max_concurrent_jobs": value if isinstance(value, int) else None,
        "active_jobs": health.get("active_jobs"),
        # Absent from a runner older than the window; callers treat that
        # as the fixed limit, which is what such a runner enforces.
        "run_window": window if isinstance(window, dict) else None,
        "source": f"reinvent-runner at {config.runner.base_url}",
    }


# ---------------------------------------------------------------------------
# Conducting, in this process
# ---------------------------------------------------------------------------

def conduct_campaign(
    campaign_id: str,
    task_budget_hours: Optional[float] = None,
    max_steps: Optional[int] = None,
    concurrency: Optional[int] = None,
    tick_seconds: Optional[float] = None,
    propose: bool = False,
) -> dict[str, Any]:
    """Keep the sub-agent pool at size for a campaign you are directing.

    Opening a campaign does not run it. This starts the background loop that
    holds the supervisor pool at the campaign's concurrency, closes the queue
    when the dispatch window ends, and finalizes at the hard stop. It returns
    at once, so the conversation carries on.

    You give out the work with ``enqueue_tasks`` and read it back with
    ``get_task_reports``, because you are the one holding the conversation
    that knows what each task is for. If a sub-agent is left without work
    for about a minute and a half, the loop fills its slot itself -- from
    the next tasks the sub-agents proposed, then from its own proposer --
    at a priority under yours, so your tasks still go first. Set
    ``propose`` to have it fill every slot on its own, which is for
    leaving a campaign running unattended.

    Idempotent: asking twice does not start a second loop.
    """

    from reinvent4_agent.conducting import REGISTRY

    if concurrency is not None:
        changed = set_concurrency(campaign_id, concurrency)
        if "error" in changed:
            return changed

    campaign = _guard(
        f"Could not read campaign {campaign_id}",
        lambda: _store().campaign(campaign_id),
    )
    if "error" in campaign:
        return campaign

    status = REGISTRY.start(
        campaign_id,
        task_budget_hours=task_budget_hours,
        max_steps=max_steps,
        tick_seconds=tick_seconds,
        propose=propose,
    )
    return status.as_dict()


def stop_conducting(campaign_id: str) -> dict[str, Any]:
    """Stop dispatching to a campaign, leaving the campaign itself open.

    In-flight tasks keep running and their results still land in the ledger;
    what stops is the resizing of the pool and any proposing of new tasks.
    Use ``close_campaign`` to end the campaign itself.
    """

    from reinvent4_agent.conducting import REGISTRY

    return REGISTRY.stop(campaign_id).as_dict()


def conducting_status(campaign_id: Optional[str] = None) -> dict[str, Any]:
    """What this process is conducting, and what its last pass did."""

    from reinvent4_agent.conducting import REGISTRY

    conducting = [status.as_dict() for status in REGISTRY.status(campaign_id)]
    return {
        "conducting": conducting,
        "n_running": sum(1 for status in conducting if status["running"]),
    }


def close_campaign(campaign_id: str) -> dict[str, Any]:
    """Stop dispatching, let in-flight tasks finish.

    The drain window, brought forward by hand. Queued tasks stay claimable, so
    work already decided on still runs.
    """

    return _guard(
        f"Could not close campaign {campaign_id}",
        lambda: _store().close_campaign(campaign_id),
    )


def finalize_campaign(campaign_id: str) -> dict[str, Any]:
    """Close the books: abandon anything outstanding and stop the campaign.

    Tasks still queued or claimed are marked abandoned rather than left to
    look like work someone will do.
    """

    return _guard(
        f"Could not finalize campaign {campaign_id}",
        lambda: _store().finalize_campaign(campaign_id),
    )


# ---------------------------------------------------------------------------
# The atlas: what the campaign knows
# ---------------------------------------------------------------------------
#
# Reports are addressed to the conductor and read once, by one agent, in one
# turn. The atlas is addressed to whoever works the campaign next: the
# sub-agent dispatched tomorrow that would otherwise repeat the search done
# today, and the conductor whose context has been compacted past the report
# that said not to. Both sides read and write it, which is the point -- a
# memory only the conductor can write is a memory of what the conductor
# noticed.


def atlas_overview(campaign_id: str) -> dict[str, Any]:
    """The map of what this campaign knows, branch by branch.

    Per branch: how many findings stand, how many are disputed, the latest
    claim and the tags to search by. Branches from the plan appear even when
    empty, and an empty branch is the most useful entry here -- it is the
    part of the goal nobody has settled anything about.

    Read this before dispatching. It is a page, where the reports behind it
    are a chapter, so there is no turn in which you cannot afford it.
    """

    return _guard(
        f"Could not read the atlas of campaign {campaign_id}",
        lambda: _store().atlas(campaign_id),
    )


def atlas_branch(
    campaign_id: str, branch: str, limit: int = 50
) -> dict[str, Any]:
    """Everything one branch of the atlas holds, newest first.

    Each finding comes with the notes hung off it, so a claim another agent
    disputed arrives disputed rather than looking settled. Superseded
    findings are left out: what you get is what the campaign currently
    believes.

    Parameters
    ----------
    campaign_id:
        Campaign to read.
    branch:
        Workstream id, as ``atlas_overview`` lists it.
    limit:
        Most findings to return, newest first.
    """

    return _guard(
        f"Could not read branch {branch} of campaign {campaign_id}",
        lambda: _store().findings(campaign_id, branch=branch, limit=limit),
    )


def atlas_search(
    campaign_id: str,
    query: str = "",
    tag: Optional[str] = None,
    limit: int = 25,
) -> dict[str, Any]:
    """Findings matching words or a tag, across every branch.

    A finding matches when every word of ``query`` appears somewhere in its
    claim, detail or tags, so fewer words match more. Use this before
    starting work that someone may already have done: a search that returns
    nothing costs a second and is worth a sub-agent's hour.

    Parameters
    ----------
    campaign_id:
        Campaign to search.
    query:
        Words that must all appear. Empty returns the latest findings.
    tag:
        Exact tag to require, as ``atlas_overview`` lists per branch.
    limit:
        Most findings to return, newest first.
    """

    return _guard(
        f"Could not search the atlas of campaign {campaign_id}",
        lambda: _store().findings(campaign_id, q=query, tag=tag, limit=limit),
    )


def atlas_finding(
    campaign_id: str,
    branch: str,
    claim: str,
    detail: Optional[str] = None,
    evidence: Optional[list[str]] = None,
    confidence: str = "medium",
    tags: Optional[list[str]] = None,
    task_id: Optional[str] = None,
    supersedes: Optional[str] = None,
) -> dict[str, Any]:
    """Record something the campaign now knows.

    Record what is true, not what you did: "TYK2 JH2 binders need a hinge
    donor at Asn671" is a finding, "ran a ChEMBL search" is not. One claim
    per call, with the evidence that would let the next agent check it.

    Nothing is ever deleted. A finding that turns out wrong is corrected by
    recording the right one with ``supersedes`` set to the old id; the old
    one stays readable and stops counting as current.

    Parameters
    ----------
    campaign_id:
        Campaign this belongs to.
    branch:
        Workstream id to file it under. Findings are read a branch at a
        time, so one filed in the wrong branch is one nobody will see.
    claim:
        The finding in one sentence, stated so it could be disagreed with.
    detail:
        The numbers, conditions and caveats behind it.
    evidence:
        Where it came from: DOIs, ChEMBL ids, job ids, PDB codes.
    confidence:
        ``high``, ``medium`` or ``low``. Say low rather than say nothing.
    tags:
        Handles to find it by later: a gene, a scaffold, an assay.
    task_id:
        The task you are working, when you have one.
    supersedes:
        Id of the finding this corrects.
    """

    from reinvent4_agent.campaign_client import supervisor_id

    return _guard(
        f"Could not record a finding for campaign {campaign_id}",
        lambda: _store().record_finding(
            campaign_id, author=supervisor_id(), branch=branch, claim=claim,
            detail=detail, evidence=evidence or [], confidence=confidence,
            tags=tags or [], task_id=task_id, supersedes=supersedes,
        ),
    )


def atlas_note(
    campaign_id: str,
    finding_id: str,
    body: str,
    stance: str = "extends",
    task_id: Optional[str] = None,
) -> dict[str, Any]:
    """Hang a second opinion off a finding somebody else recorded.

    Use this rather than recording a near-duplicate: an independent
    confirmation is worth more attached to the claim it confirms than filed
    beside it, and a dispute is worth most of all -- it is how the next
    agent learns the number it was about to build on is contested.

    Parameters
    ----------
    campaign_id:
        Campaign the finding belongs to.
    finding_id:
        Finding to annotate.
    body:
        What you have to say, with your own evidence.
    stance:
        ``supports``, ``disputes`` or ``extends``.
    task_id:
        The task you are working, when you have one.
    """

    from reinvent4_agent.campaign_client import supervisor_id

    return _guard(
        f"Could not note finding {finding_id}",
        lambda: _store().note_finding(
            campaign_id, finding_id, author=supervisor_id(), body=body,
            stance=stance, task_id=task_id,
        ),
    )
