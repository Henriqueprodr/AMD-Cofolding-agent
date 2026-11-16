# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A sub-agent that leads a workstream instead of doing one task of it.

One conductor reading every report stops scaling somewhere past a dozen
sub-agents: the reports arrive faster than one model can read them, and the
slots it has not got round to sit empty. A team of a thousand chemists is not
run by one person reading a thousand notebooks; it is run through leads, each
owning a question and the people on it, who report up what their group
found.

A ``lead`` task is that. The sub-agent that claims it opens a campaign of its
own under the parent's, keeps it supplied with tasks from the proposer the
way the conductor loop does, and when the work drains reports one synthesis
up. Its tasks are claimed from the parent's pool -- a campaign's claim takes
from its leads' campaigns too -- so a lead adds no supervisors, only a level
of planning; and the parent's conductor counts its tasks against the
parent's slots so the two do not overfill the pool between them.

Kept apart from :mod:`reinvent4_agent.supervisor` so it can be tested with a
fake ledger and a fake proposer, without a model or an MCP server.
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from reinvent4_agent.campaign_client import CampaignStoreClient, StoreUnavailable, TaskOutcome
from reinvent4_agent.conductor import CampaignConductor
from reinvent4_agent.proposing import Proposer

logger = logging.getLogger(__name__)

#: Sub-agents a lead's workstream holds when the task does not say.
DEFAULT_TEAM = 2
#: Dispatch window of a lead with no deadline: long enough for a few rounds.
DEFAULT_LEAD_SECONDS = 2 * 3600.0
#: Tasks a lead dispatches before it stops and writes up, per sub-agent.
TASKS_PER_MEMBER = 3
#: How often the lead's loop reconciles its campaign.
LEAD_TICK_SECONDS = 30.0
#: Open proposals of the workstream handed up to the parent as follow-ups.
MAX_HANDED_UP = 3

FINDING_CHARS = 400


def _window_seconds(spec: dict[str, Any], now: float) -> float:
    deadline = spec.get("deadline")
    if deadline:
        try:
            when = datetime.fromisoformat(deadline)
        except ValueError:
            return DEFAULT_LEAD_SECONDS
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        # Dispatching stops early enough for the last round to come back
        # before the lead's own deadline.
        return max(60.0, (when.timestamp() - now) * 0.75)
    return DEFAULT_LEAD_SECONDS


def lead_campaign(store: CampaignStoreClient, task: dict[str, Any], now: float) -> dict[str, Any]:
    """The lead's campaign, opened now or found from an earlier claim.

    Found rather than reopened so a lead whose lease lapsed and was claimed
    again carries on with its workstream instead of starting a second one.
    """

    task_id = task["task_id"]
    for campaign in store.campaigns():
        if campaign.get("parent_task_id") == task_id and campaign.get("state") != "finalized":
            return campaign

    spec = task.get("spec") or {}
    parent = store.campaign(task["campaign_id"])
    window = _window_seconds(spec, now)
    goal = str(spec.get("intent") or spec.get("title") or "")
    if parent.get("goal"):
        goal = f"{goal}\n\nThis is one workstream of the campaign: {parent['goal']}"
    return store.create_campaign(
        goal=goal,
        base_ttp_id=parent.get("base_ttp_id"),
        objectives=parent.get("objectives") or [],
        concurrency=int(spec.get("concurrency") or DEFAULT_TEAM),
        dispatch_seconds=window,
        grace_seconds=window / 3,
        parent_campaign_id=task["campaign_id"],
        parent_task_id=task_id,
    )


def conduct(
    store: CampaignStoreClient,
    task: dict[str, Any],
    proposer: Proposer,
    *,
    tick_seconds: float = LEAD_TICK_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    should_stop: Callable[[], bool] = lambda: False,
    clock: Callable[[], float] = time.time,
) -> TaskOutcome:
    """Run the lead's workstream to the end and report it up."""

    campaign = lead_campaign(store, task, clock())
    sub_id = campaign["campaign_id"]
    team = int(campaign.get("concurrency") or DEFAULT_TEAM)
    budget = team * TASKS_PER_MEMBER
    logger.info("lead %s conducting campaign %s with %d sub-agent(s)", task["task_id"], sub_id, team)

    # No scaler: the parent's pool claims the workstream's tasks. No floor:
    # this loop proposes every tick, so a slot never waits on anyone.
    conductor = CampaignConductor(
        store, sub_id, scaler=None, proposer=proposer, propose=True,
        tick_seconds=tick_seconds, floor_seconds=None,
    )
    closed = False
    while True:
        try:
            report = conductor.tick()
        except StoreUnavailable as exc:
            logger.warning("lead %s: ledger unavailable: %s", task["task_id"], exc)
        else:
            # Finalized from above -- the parent stopped -- reads as a tick
            # with nothing to do, not as a finalize this loop did.
            if report.finalized or store.campaign(sub_id).get("state") == "finalized":
                break
            if not closed and (should_stop() or len(store.tasks(sub_id)) >= budget):
                store.close_campaign(sub_id)
                closed = True
        sleep(tick_seconds)

    return synthesis(store, task, sub_id)


def _best_profile(results: list[dict[str, Any]]) -> dict[str, float]:
    best: dict[str, float] = {}
    for result in results:
        for name, value in (result.get("objective_profile") or {}).items():
            if isinstance(value, (int, float)) and not math.isnan(value):
                best[name] = max(best.get(name, value), value)
    return best


def synthesis(store: CampaignStoreClient, task: dict[str, Any], sub_id: str) -> TaskOutcome:
    """What the workstream found, as one report for the parent's conductor.

    Assembled from the sub-agents' own summaries rather than rewritten by a
    model: the conductor reads a lead's report the way it reads any other,
    and a summary of summaries would be one more place for a finding to be
    lost.
    """

    spec = task.get("spec") or {}
    tasks = store.tasks(sub_id)
    results = {r["task_id"]: r for r in store.results(sub_id)}
    done = [t for t in tasks if t["task_id"] in results]
    failed = [t for t in tasks if t.get("state") == "failed"]

    title = spec.get("title") or "the workstream"
    lines = [
        "## Report",
        "",
        f"Led {title} as campaign {sub_id}: {len(done)} task(s) reported, {len(failed)} failed.",
        "",
        "## Findings",
        "",
    ]
    for t in done:
        name = (t.get("spec") or {}).get("title") or (t.get("spec") or {}).get("intent", "")[:60]
        summary = str(results[t["task_id"]].get("summary") or "no summary").strip()
        if len(summary) > FINDING_CHARS:
            summary = summary[: FINDING_CHARS - 1] + "…"
        lines.append(f"- {t['task_id']} ({name}): {summary}")
    if not done:
        lines.append("- Nothing came back.")

    candidates = sorted(
        (c for r in results.values() for c in (r.get("candidates") or [])),
        key=lambda c: c.get("score") if isinstance(c.get("score"), (int, float)) else float("-inf"),
        reverse=True,
    )[:20]
    job_ids = [j for r in results.values() for j in (r.get("job_ids") or [])]

    follow_ups: list[dict[str, Any]] = []
    try:
        open_ones = store.proposals(sub_id)
    except StoreUnavailable:
        open_ones = []
    for proposal in open_ones[:MAX_HANDED_UP]:
        follow_ups.append({
            key: proposal["spec"][key]
            for key in ("kind", "title", "intent", "rationale")
            if proposal["spec"].get(key)
        })
    if open_ones:
        # Handed up, so not left open where nobody conducts any more.
        try:
            store.drop_proposals(sub_id, [p["proposal_id"] for p in open_ones])
        except StoreUnavailable as exc:
            logger.warning("could not drop the proposals of %s: %s", sub_id, exc)

    return TaskOutcome(
        objective_profile=_best_profile(list(results.values())),
        candidates=candidates,
        summary=f"Led {title}: {len(done)} task(s) reported, {len(failed)} failed.",
        report="\n".join(lines),
        job_ids=job_ids,
        follow_ups=follow_ups,
    )
