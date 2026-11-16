"""Campaign store -- the control-plane ledger for a conductor-led agent fleet.

One conductor hands out work; N supervisors claim it, run a full RL campaign
each, and post back what they achieved. This service is the only thing that
knows the truth about that: which tasks are queued, who is holding which task
and until when, and what came back. Everything else is a client, so a
conductor that restarts rebuilds its loop by reading the ledger rather than
from memory.

A note on vocabulary, because the codebase already uses the word differently:
here a **campaign** is the whole sweep (one goal, N supervisors, a wall-clock
budget) and a **task** is one supervisor run -- which is what
``reinvent4_agent/coordinator.py`` calls a campaign. The plan locks these
names; the collision is only in the English.

Design notes:

* **SQLite, single writer.** The Deployment is pinned to one replica with
  ``Recreate``, so there is exactly one process writing. SQLite then buys
  durability and atomic claims for none of the operational cost of a database
  server, and the file lives on ``/shared`` beside the job working directories
  it refers to. WAL mode so the conductor's polling of ``/status`` never blocks
  a supervisor posting a result.

* **Claims are atomic.** ``POST /tasks/claim`` selects and marks in one
  ``BEGIN IMMEDIATE`` transaction. Starlette runs these sync endpoints in a
  threadpool, so two supervisors really can arrive together; without the
  write-lock held across the select both would be handed the same task and
  would run the same weight vector on separate GPUs.

* **Leases, not assignments.** A supervisor holds a task for
  ``lease_seconds`` and has to heartbeat to keep it. A pod that is OOM-killed
  mid-campaign therefore releases its task without anyone having to notice the
  crash. Expiry is swept inline on claim and status rather than by a background
  task, so the behaviour is a function of the requests made and a test does not
  have to wait on a timer.

Env:
    CAMPAIGN_STORE_DB       SQLite path (default /shared/campaign-store/campaigns.db)
    CAMPAIGN_STORE_PORT     listen port (default 8087)
    TASK_MAX_ATTEMPTS       claims a task gets before it is abandoned (default 2)
    DEFAULT_LEASE_SECONDS   lease granted when a claim does not ask (default 900)
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, Literal, Optional

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("campaign_store")

DB_PATH = Path(os.getenv("CAMPAIGN_STORE_DB", "/shared/campaign-store/campaigns.db"))
TASK_MAX_ATTEMPTS = int(os.getenv("TASK_MAX_ATTEMPTS", "2"))
DEFAULT_LEASE_SECONDS = float(os.getenv("DEFAULT_LEASE_SECONDS", "900"))

# How long a supervisor's presence stands without being renewed. Generous
# against the supervisor's poll interval, so a slow loop is not read as a dead
# pod; the cost of being wrong is a conductor dispatching to nobody.
DEFAULT_PRESENCE_TTL_SECONDS = float(os.getenv("PRESENCE_TTL_SECONDS", "120"))

# Campaign states: `dispatching` accepts new tasks, `draining` is the plan's
# post-T window where the queue is closed but claimed tasks run on.
TaskState = Literal["queued", "claimed", "done", "failed", "abandoned"]

# What a task asks for. `design` is the original one -- an RL run against a
# therapeutic target profile -- and the supervisor still treats it specially,
# because the profile and the weight vector are the experiment. The rest are
# ordinary work a drug-design agent does with its tools, and the supervisor
# runs them by handing the brief to an agent and reading back what it found.
#
# A closed list rather than free text: the kind decides which path the
# supervisor takes, and an unrecognised one would be run as the wrong thing.
TaskKind = Literal[
    "design", "research", "analysis", "retrosynthesis", "question", "lead",
]

# A follow-up a sub-agent proposed: `open` until the conductor or the floor
# takes it as a task, or drops it.
ProposalState = Literal["open", "taken", "dropped"]

# How sure the agent recording a finding is of it. The conductor reads these
# to know what it may build on and what still has to be settled, so a guess
# has to be sayable as a guess rather than withheld.
Confidence = Literal["high", "medium", "low"]

# What a note does to the finding it hangs off. Agreement is worth recording
# too: two sub-agents reaching the same number independently is a stronger
# fact than one, and nothing else in the ledger can say so.
Stance = Literal["supports", "disputes", "extends"]

#: Priority of a task the conductor's floor filled a free slot with. Below
#: the default 0, so whatever the conductor dispatches itself is claimed first.
FLOOR_PRIORITY = -1

# A task not yet finished one way or another, so it still owes the campaign an
# answer. Spelled as SQL because it appears in three IN clauses.
OPEN_TASK_STATES = "('queued', 'claimed')"


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS campaigns (
    campaign_id       TEXT PRIMARY KEY,
    goal              TEXT NOT NULL,
    base_ttp_id       TEXT NOT NULL,
    objectives        TEXT NOT NULL,
    concurrency       INTEGER NOT NULL,
    state             TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    dispatch_until    REAL,
    hard_stop_at      REAL,
    finalized_at      TEXT,
    -- The conversation whose conductor opened this. Nullable: a campaign
    -- started from the command line belongs to nobody, and campaigns that
    -- predate the column belong to nobody either. The store never reads it;
    -- it is here so a chat window can ask "which of these are mine" without
    -- the answer living in a browser that will be closed.
    session_id        TEXT,
    -- When a conductor last said it holds the campaign. Null for one that
    -- never had a conductor; see CONDUCTED_WITHIN_SECONDS.
    conducted_at      REAL
);

CREATE TABLE IF NOT EXISTS tasks (
    task_id           TEXT PRIMARY KEY,
    campaign_id       TEXT NOT NULL REFERENCES campaigns(campaign_id),
    spec              TEXT NOT NULL,
    state             TEXT NOT NULL,
    attempts          INTEGER NOT NULL DEFAULT 0,
    created_at        TEXT NOT NULL,
    lease_owner       TEXT,
    lease_expires_at  REAL,
    claimed_at        TEXT,
    finished_at       TEXT,
    error             TEXT,
    -- The run this task started, reported as soon as it starts rather than
    -- with the result. results.job_id has always had it afterwards, which is
    -- no use to somebody asking what a task is doing *now* -- the whole of a
    -- long run was the part nobody could look inside.
    job_id            TEXT
);

CREATE INDEX IF NOT EXISTS tasks_queue
    ON tasks (campaign_id, state, created_at);

CREATE TABLE IF NOT EXISTS results (
    task_id           TEXT PRIMARY KEY REFERENCES tasks(task_id),
    campaign_id       TEXT NOT NULL,
    job_id            TEXT,
    objective_profile TEXT NOT NULL,
    candidates        TEXT NOT NULL,
    summary           TEXT,
    posted_at         TEXT NOT NULL,
    report            TEXT,
    job_ids           TEXT
);

CREATE TABLE IF NOT EXISTS events (
    event_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id       TEXT,
    task_id           TEXT,
    kind              TEXT NOT NULL,
    payload           TEXT NOT NULL,
    created_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS events_feed ON events (campaign_id, event_id);

-- Which supervisors exist, as opposed to which hold a task. A supervisor
-- otherwise touches the ledger only when it claims, so an idle one -- exactly
-- the one a conductor wants to know about before dispatching -- was invisible.
CREATE TABLE IF NOT EXISTS supervisors (
    owner             TEXT PRIMARY KEY,
    campaign_id       TEXT,
    state             TEXT NOT NULL,
    task_id           TEXT,
    tasks_completed   INTEGER NOT NULL DEFAULT 0,
    tasks_failed      INTEGER NOT NULL DEFAULT 0,
    detail            TEXT,
    first_seen_at     TEXT NOT NULL,
    last_seen_at      TEXT NOT NULL,
    expires_at        REAL NOT NULL
);

-- Next tasks a sub-agent proposed at the end of its report. Kept apart from
-- the queue: a sub-agent sees one task, so what it proposes is a candidate
-- for the conductor to rank against the whole campaign, not an order.
CREATE TABLE IF NOT EXISTS proposals (
    proposal_id       TEXT PRIMARY KEY,
    campaign_id       TEXT NOT NULL REFERENCES campaigns(campaign_id),
    from_task_id      TEXT,
    spec              TEXT NOT NULL,
    state             TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    task_id           TEXT
);

CREATE INDEX IF NOT EXISTS proposals_open ON proposals (campaign_id, state, created_at);

-- What the campaign has learnt, as opposed to what it has run. A report is
-- addressed to the conductor and read once; this is addressed to whoever
-- works the campaign next, including the sub-agent that will redo the search
-- somebody already did. Append-only: a finding that turns out wrong is
-- superseded by the one that corrects it, because knowing the campaign
-- believed it for two hours is part of the record.
CREATE TABLE IF NOT EXISTS findings (
    finding_id        TEXT PRIMARY KEY,
    campaign_id       TEXT NOT NULL REFERENCES campaigns(campaign_id),
    -- Workstream id from the plan, so the atlas and the coverage view cut the
    -- campaign the same way. Free text: a finding may land before the plan.
    branch            TEXT NOT NULL,
    claim             TEXT NOT NULL,
    detail            TEXT,
    evidence          TEXT NOT NULL,
    confidence        TEXT NOT NULL,
    tags              TEXT NOT NULL,
    author            TEXT NOT NULL,
    task_id           TEXT,
    created_at        TEXT NOT NULL,
    supersedes        TEXT,
    superseded_by     TEXT
);

CREATE INDEX IF NOT EXISTS findings_branch
    ON findings (campaign_id, branch, created_at);

CREATE TABLE IF NOT EXISTS finding_notes (
    note_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    finding_id        TEXT NOT NULL REFERENCES findings(finding_id),
    campaign_id       TEXT NOT NULL,
    body              TEXT NOT NULL,
    stance            TEXT NOT NULL,
    author            TEXT NOT NULL,
    task_id           TEXT,
    created_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS finding_notes_feed ON finding_notes (finding_id, note_id);
"""


def _branch_key(name: str) -> str:
    """One spelling per branch of the atlas.

    Branches are free text because a finding can land before there is a plan,
    and free text is how a memory fragments: 'ADMET', 'admet' and 'admet '
    are three branches to anyone reading and one to everyone writing. Case
    and spacing carry no meaning here, so they are not kept -- on the way in
    and on the way out, or a workstream whose plan spells it differently
    would never meet its own findings.
    """

    return " ".join(name.split()).lower()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


def _age_seconds(stamp: Optional[str]) -> Optional[float]:
    """Seconds since an ISO timestamp this file wrote, or None if unreadable."""

    if not stamp:
        return None
    try:
        then = datetime.fromisoformat(stamp)
    except ValueError:
        return None
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return round((datetime.now(timezone.utc) - then).total_seconds(), 1)


def _connect() -> sqlite3.Connection:
    """A fresh connection; sqlite3 connections are not thread-safe to share.

    ``busy_timeout`` matters because claims take the write lock: a supervisor
    arriving mid-claim should wait its turn rather than get "database is
    locked".
    """

    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


#: Columns added after the first release, as (table, column, definition).
#: ``CREATE TABLE IF NOT EXISTS`` does nothing to a table that already
#: exists, so a ledger with campaigns in it would never see a new column and
#: every query naming it would fail -- on a store whose whole job is to
#: survive the things around it restarting.
_ADDED_COLUMNS = [
    ("campaigns", "session_id", "TEXT"),
    ("tasks", "job_id", "TEXT"),
    ("results", "report", "TEXT"),
    ("results", "job_ids", "TEXT"),
    ("campaigns", "conducted_at", "REAL"),
    # The workstreams the conductor split the goal into, as JSON.
    ("campaigns", "plan", "TEXT"),
    # A lead's campaign: the campaign and the task whose sub-agent conducts it.
    ("campaigns", "parent_campaign_id", "TEXT"),
    ("campaigns", "parent_task_id", "TEXT"),
    # Kept after the task finishes, unlike lease_owner: who ran it is what
    # says how long each sub-agent waited between two tasks.
    ("tasks", "claimed_by", "TEXT"),
]

#: How long a campaign's tasks stay claimable after its conductor last said
#: it was holding the campaign. Several of its ticks, so one slow pass does
#: not cost the campaign its supervisors.
CONDUCTED_WITHIN_SECONDS = float(os.getenv("CAMPAIGN_CONDUCTED_WITHIN_SECONDS", "300"))

#: Longest report kept. The writer is a model, and a runaway one should cost
#: the ledger a truncated report rather than the conductor its context.
MAX_REPORT_CHARS = int(os.getenv("MAX_REPORT_CHARS", "20000"))


def _add_missing_columns(conn: sqlite3.Connection) -> None:
    for table, column, definition in _ADDED_COLUMNS:
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
            logger.info("added %s.%s to an existing ledger", table, column)


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(SCHEMA)
        _add_missing_columns(conn)
    logger.info("campaign store ready at %s", DB_PATH)


@contextmanager
def _write() -> Iterator[sqlite3.Connection]:
    """A transaction that takes the write lock up front.

    Python's sqlite3 defers BEGIN until the first write, which would leave a
    select-then-update sequence racing another connection. ``BEGIN IMMEDIATE``
    is what makes the claim below safe.
    """

    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def _read() -> Iterator[sqlite3.Connection]:
    conn = _connect()
    try:
        yield conn
    finally:
        conn.close()


def _emit(
    conn: sqlite3.Connection,
    kind: str,
    campaign_id: Optional[str],
    task_id: Optional[str] = None,
    **payload: Any,
) -> None:
    conn.execute(
        "INSERT INTO events (campaign_id, task_id, kind, payload, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (campaign_id, task_id, kind, json.dumps(payload), _now_iso()),
    )


# ---------------------------------------------------------------------------
# Leases
# ---------------------------------------------------------------------------

def _sweep_expired_leases(conn: sqlite3.Connection, campaign_id: Optional[str] = None) -> int:
    """Requeue or abandon tasks whose holder stopped heartbeating.

    Called on the paths that care (claim, status) instead of from a timer, so
    an expired lease is always observed before it could matter.
    """

    sql = (
        "SELECT task_id, campaign_id, attempts, lease_owner FROM tasks "
        "WHERE state = 'claimed' AND lease_expires_at IS NOT NULL "
        "AND lease_expires_at < ?"
    )
    args: list[Any] = [time.time()]
    if campaign_id:
        sql += " AND campaign_id = ?"
        args.append(campaign_id)

    expired = conn.execute(sql, args).fetchall()

    for row in expired:
        attempts = row["attempts"]
        if attempts < TASK_MAX_ATTEMPTS:
            conn.execute(
                "UPDATE tasks SET state = 'queued', lease_owner = NULL, "
                "lease_expires_at = NULL, claimed_at = NULL WHERE task_id = ?",
                (row["task_id"],),
            )
            kind = "task_requeued"
        else:
            conn.execute(
                "UPDATE tasks SET state = 'abandoned', lease_owner = NULL, "
                "lease_expires_at = NULL, finished_at = ?, error = ? WHERE task_id = ?",
                (_now_iso(), f"lease expired {attempts} times", row["task_id"]),
            )
            kind = "task_abandoned"

        logger.warning(
            "%s: lease held by %s expired (attempt %d)",
            row["task_id"], row["lease_owner"], attempts,
        )
        _emit(
            conn, kind, row["campaign_id"], row["task_id"],
            previous_owner=row["lease_owner"], attempts=attempts,
        )

    return len(expired)


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------

class CampaignCreate(BaseModel):
    model_config = {"extra": "forbid"}

    goal: str
    base_ttp_id: Optional[str] = Field(
        default=None,
        description="The TTP every task varies. Omit it and each task builds "
                    "its own instead, which is the usual way round: the "
                    "conductor says what an experiment is for and the agent "
                    "running it decides how to measure that. Give one when "
                    "the campaign is a sweep of a profile you have already "
                    "written and every task must score the same way.",
    )
    objectives: list[str] = Field(
        default_factory=list,
        description="Objective names every result is reported on. Fixed at "
                    "creation so results stay comparable mid-campaign. Empty "
                    "when tasks bring their own TTPs and therefore their own "
                    "objectives, and a result is read in its own terms",
    )
    concurrency: int = Field(default=1, ge=1, description="Supervisors to hold in flight")
    dispatch_seconds: Optional[float] = Field(
        default=None, gt=0,
        description="Length of the dispatch window T. Omit for no deadline",
    )
    grace_seconds: float = Field(
        default=0.0, ge=0,
        description="Drain window after T, during which claimed tasks finish",
    )
    parent_campaign_id: Optional[str] = Field(
        default=None,
        description="For a lead's campaign: the campaign it is a workstream of. "
                    "Its tasks are drawn from the parent's pool, so the "
                    "parent's counts include them.",
    )
    parent_task_id: Optional[str] = Field(
        default=None, description="The parent campaign's task whose sub-agent leads this one",
    )


class Workstream(BaseModel):
    model_config = {"extra": "forbid"}

    id: str = Field(min_length=1, max_length=40, description="Short handle tasks name it by")
    name: str = Field(min_length=1, max_length=120)
    question: Optional[str] = Field(
        default=None, description="What this workstream has to settle for the goal",
    )
    share: Optional[float] = Field(
        default=None, gt=0, le=1,
        description="Rough share of the team it should have; unset means even",
    )
    status: Literal["open", "settled", "dropped"] = "open"


class PlanUpdate(BaseModel):
    model_config = {"extra": "forbid"}

    workstreams: list[Workstream] = Field(min_length=1, max_length=40)


class FindingPost(BaseModel):
    """Something the campaign now knows, worth the next agent's while."""

    model_config = {"extra": "forbid"}

    branch: str = Field(
        min_length=1, max_length=40,
        description="Workstream id this belongs under, the same handle tasks "
                    "name. Findings are read a branch at a time, so one "
                    "filed in the wrong place is one nobody will see.",
    )

    @field_validator("branch")
    @classmethod
    def _fold(cls, value: str) -> str:
        return _branch_key(value)
    claim: str = Field(
        min_length=1, max_length=400,
        description="The finding itself in one sentence, stated so it can be "
                    "disagreed with: what is true, not what was done.",
    )
    detail: Optional[str] = Field(
        default=None, max_length=4000,
        description="The numbers, conditions and caveats behind the claim",
    )
    evidence: list[str] = Field(
        default_factory=list, max_length=20,
        description="Where it came from: a DOI, a ChEMBL id, a job id, a PDB "
                    "code. A claim with no evidence is an opinion, and the "
                    "agent reading it has no way to check yours.",
    )
    confidence: Confidence = "medium"
    tags: list[str] = Field(
        default_factory=list, max_length=12,
        description="Handles to find it by later: a gene, a scaffold, an "
                    "assay, a liability.",
    )
    author: str = Field(min_length=1, description="Supervisor or conductor recording it")
    task_id: Optional[str] = None
    supersedes: Optional[str] = Field(
        default=None,
        description="Finding this one corrects. The old one stays readable "
                    "and stops counting as current.",
    )


class NotePost(BaseModel):
    """A second opinion on a finding somebody else recorded."""

    model_config = {"extra": "forbid"}

    body: str = Field(min_length=1, max_length=2000)
    stance: Stance = "extends"
    author: str = Field(min_length=1)
    task_id: Optional[str] = None


class FollowUp(BaseModel):
    """A next task a sub-agent would run, as it proposed it."""

    model_config = {"extra": "ignore"}

    intent: str = Field(min_length=1)
    title: Optional[str] = Field(default=None, max_length=120)
    kind: TaskKind = "research"
    rationale: Optional[str] = None
    workstream: Optional[str] = None


class TaskCreate(BaseModel):
    model_config = {"extra": "forbid"}

    kind: TaskKind = Field(
        default="design",
        description="What sort of work this is. `design` runs RL against a "
                    "profile and is what every task was before there were "
                    "kinds; the others hand the brief to an agent with the "
                    "full tool set and post back what it reports.",
    )
    title: Optional[str] = Field(
        default=None, max_length=120,
        description="A few words naming the task, for the person watching "
                    "the team: what a bubble on the timeline is labelled.",
    )
    context: Optional[str] = Field(
        default=None,
        description="What the sub-agent needs to know and would otherwise "
                    "have to rediscover: the target, what earlier tasks "
                    "found, constraints the user stated. A sub-agent sees "
                    "nothing of the conversation that dispatched it.",
    )
    deliverable: Optional[str] = Field(
        default=None,
        description="What the conductor wants back, so the report answers "
                    "the question that was asked rather than the one the "
                    "sub-agent found most interesting.",
    )
    weight_overrides: dict[str, float] = Field(default_factory=dict)
    transform_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)
    fidelity_tier: Literal["exploration", "validation"] = "exploration"
    max_steps: Optional[int] = Field(default=None, gt=0)
    deadline: Optional[str] = None
    token_budget: Optional[int] = Field(default=None, gt=0)
    max_turns: Optional[int] = Field(default=None, gt=0)
    intent: Optional[str] = Field(
        default=None,
        description="What this run is for, in prose, handed to the agent "
                    "verbatim. The channel for anything the structured fields "
                    "cannot say -- 'hold the JQ1 core, pick a prior suited to "
                    "that rather than de novo' -- so the agent does the "
                    "chemistry instead of being handed a seed.",
    )
    derived_from: list[str] = Field(
        default_factory=list,
        description="Task ids whose results led the conductor to this one. "
                    "The campaign's lineage: without it a finished campaign "
                    "is a bag of runs and nobody can see which result "
                    "prompted which experiment. Empty for the first tasks, "
                    "which had nothing to go on.",
    )
    rationale: Optional[str] = Field(
        default=None,
        description="One line on why this task and not another, for the "
                    "human watching. Not shown to the agent, which is told "
                    "what to do by `intent` and should not be second-guessing "
                    "why it was asked.",
    )
    workstream: Optional[str] = Field(
        default=None, max_length=40,
        description="Id of the plan's workstream this task belongs to",
    )
    priority: int = Field(
        default=0, ge=-10, le=10,
        description="Higher is claimed first; equal priorities go oldest first",
    )
    concurrency: Optional[int] = Field(
        default=None, ge=1, le=64,
        description="For a lead: how many sub-agents its workstream may hold",
    )


class TaskBatch(BaseModel):
    model_config = {"extra": "forbid"}

    tasks: list[TaskCreate] = Field(min_length=1, max_length=200)


class ProposalDecision(BaseModel):
    model_config = {"extra": "forbid"}

    proposal_ids: list[str] = Field(min_length=1, max_length=200)
    priority: int = Field(default=0, ge=-10, le=10)


class ConcurrencyUpdate(BaseModel):
    model_config = {"extra": "forbid"}

    concurrency: int = Field(ge=0, description="Supervisors to hold in flight")


class ClaimRequest(BaseModel):
    model_config = {"extra": "forbid"}

    owner: str = Field(description="Supervisor identity, normally the pod name")
    campaign_id: Optional[str] = None
    lease_seconds: float = Field(default=DEFAULT_LEASE_SECONDS, gt=0)


class SessionAttach(BaseModel):
    model_config = {"extra": "forbid"}

    session_id: str = Field(min_length=1)


class HeartbeatRequest(BaseModel):
    model_config = {"extra": "forbid"}

    owner: str
    lease_seconds: float = Field(default=DEFAULT_LEASE_SECONDS, gt=0)
    job_id: Optional[str] = Field(
        default=None,
        description="The run this task has started, once it has one. Carried "
                    "on the beat the supervisor already sends rather than on "
                    "a call of its own, so a task that is running is "
                    "discoverable the whole time it runs and not only once "
                    "it reports.",
    )


class ResultPost(BaseModel):
    model_config = {"extra": "forbid"}

    owner: str
    objective_profile: dict[str, float] = Field(
        default_factory=dict,
        description="Achieved value per objective. Empty for work that "
                    "scores nothing, such as a literature search.",
    )
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    job_id: Optional[str] = None
    summary: Optional[str] = None
    report: Optional[str] = Field(
        default=None,
        description="The sub-agent's own account of the task, for the "
                    "conductor that dispatched it: what it did, what it "
                    "found, and what it would do next.",
    )
    job_ids: list[str] = Field(
        default_factory=list,
        description="Every run the task started, `job_id` being the one its "
                    "molecules came from. A research task may start none.",
    )
    follow_ups: list[FollowUp] = Field(
        default_factory=list, max_length=5,
        description="Next tasks the sub-agent would run, kept as proposals",
    )


class FailurePost(BaseModel):
    model_config = {"extra": "forbid"}

    owner: str
    error: str
    requeue: bool = Field(
        default=True,
        description="Whether the task may be retried by another supervisor",
    )


class PresencePost(BaseModel):
    model_config = {"extra": "forbid"}

    state: Literal["idle", "busy", "starting", "stopping"] = Field(
        description="idle means it would take a task right now",
    )
    campaign_id: Optional[str] = Field(
        default=None,
        description="Campaign it is pinned to, or null if it claims from any",
    )
    task_id: Optional[str] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    detail: Optional[str] = Field(
        default=None, description="What it is doing, for a human reading the pool",
    )
    ttl_seconds: float = Field(default=DEFAULT_PRESENCE_TTL_SECONDS, gt=0)


# ---------------------------------------------------------------------------
# Row mapping
# ---------------------------------------------------------------------------

def _campaign_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "campaign_id": row["campaign_id"],
        "goal": row["goal"],
        # None, not "": a campaign whose tasks build their own TTPs has no
        # base TTP, and callers branch on that.
        "base_ttp_id": row["base_ttp_id"] or None,
        "objectives": json.loads(row["objectives"]),
        "concurrency": row["concurrency"],
        "state": row["state"],
        "created_at": row["created_at"],
        "dispatch_until": row["dispatch_until"],
        "hard_stop_at": row["hard_stop_at"],
        "finalized_at": row["finalized_at"],
        "session_id": row["session_id"],
        "conducted_at": row["conducted_at"],
        "plan": json.loads(row["plan"]) if row["plan"] else None,
        "parent_campaign_id": row["parent_campaign_id"],
        "parent_task_id": row["parent_task_id"],
    }


def _supervisor_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "owner": row["owner"],
        "state": row["state"],
        "campaign_id": row["campaign_id"],
        "task_id": row["task_id"],
        "tasks_completed": row["tasks_completed"],
        "tasks_failed": row["tasks_failed"],
        "detail": row["detail"],
        "first_seen_at": row["first_seen_at"],
        "last_seen_at": row["last_seen_at"],
        # Age rather than the deadline: "reported 4s ago" is what says whether
        # to believe the state beside it.
        "last_seen_seconds_ago": _age_seconds(row["last_seen_at"]),
    }


def _task_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "task_id": row["task_id"],
        "campaign_id": row["campaign_id"],
        "spec": json.loads(row["spec"]),
        "state": row["state"],
        "attempts": row["attempts"],
        "created_at": row["created_at"],
        "lease_owner": row["lease_owner"],
        "lease_expires_at": row["lease_expires_at"],
        "claimed_at": row["claimed_at"],
        "finished_at": row["finished_at"],
        "error": row["error"],
        "job_id": row["job_id"],
        "claimed_by": row["claimed_by"],
    }


def _proposal_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "proposal_id": row["proposal_id"],
        "campaign_id": row["campaign_id"],
        "from_task_id": row["from_task_id"],
        "spec": json.loads(row["spec"]),
        "state": row["state"],
        "created_at": row["created_at"],
        "task_id": row["task_id"],
    }


def _result_row(row: sqlite3.Row) -> dict[str, Any]:
    job_ids = json.loads(row["job_ids"]) if row["job_ids"] else []
    # Results posted before job_ids existed name their one run in job_id.
    if not job_ids and row["job_id"]:
        job_ids = [row["job_id"]]
    return {
        "task_id": row["task_id"],
        "campaign_id": row["campaign_id"],
        "job_id": row["job_id"],
        "job_ids": job_ids,
        "objective_profile": json.loads(row["objective_profile"]),
        "candidates": json.loads(row["candidates"]),
        "summary": row["summary"],
        "report": row["report"],
        "posted_at": row["posted_at"],
    }


def _get_campaign(conn: sqlite3.Connection, campaign_id: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM campaigns WHERE campaign_id = ?", (campaign_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"No campaign {campaign_id}")
    return row


def _owned_task(conn: sqlite3.Connection, task_id: str, owner: str) -> sqlite3.Row:
    """The task, if this owner still holds its lease.

    A supervisor whose lease expired and was handed to someone else must not be
    able to post a result over the top of the new holder's work, so the owner
    check is a precondition rather than advisory.
    """

    row = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"No task {task_id}")
    if row["state"] != "claimed":
        raise HTTPException(
            status_code=409,
            detail=f"Task {task_id} is {row['state']}, not claimed",
        )
    if row["lease_owner"] != owner:
        raise HTTPException(
            status_code=409,
            detail=f"Task {task_id} is held by {row['lease_owner']}, not {owner}",
        )
    return row


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    init_db()
    yield


app = FastAPI(
    title="campaign-store",
    description="Control-plane ledger for the conductor/supervisor fleet",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    with _read() as conn:
        campaigns = conn.execute(
            "SELECT COUNT(*) AS n FROM campaigns WHERE state = 'dispatching'"
        ).fetchone()["n"]
        open_tasks = conn.execute(
            f"SELECT COUNT(*) AS n FROM tasks WHERE state IN {OPEN_TASK_STATES}"
        ).fetchone()["n"]

    return {
        "status": "healthy",
        "db": str(DB_PATH),
        "active_campaigns": campaigns,
        "open_tasks": open_tasks,
    }


# -- campaigns --------------------------------------------------------------

@app.post("/campaigns", status_code=201)
def create_campaign(body: CampaignCreate) -> dict[str, Any]:
    campaign_id = _new_id()
    now = time.time()
    dispatch_until = now + body.dispatch_seconds if body.dispatch_seconds else None
    hard_stop_at = dispatch_until + body.grace_seconds if dispatch_until else None

    with _write() as conn:
        if body.parent_campaign_id:
            _get_campaign(conn, body.parent_campaign_id)
        conn.execute(
            "INSERT INTO campaigns (campaign_id, goal, base_ttp_id, objectives, "
            "concurrency, state, created_at, dispatch_until, hard_stop_at, "
            "parent_campaign_id, parent_task_id) "
            "VALUES (?, ?, ?, ?, ?, 'dispatching', ?, ?, ?, ?, ?)",
            (
                # Empty string rather than NULL for the absent TTP: the
                # column was declared NOT NULL and SQLite cannot relax that
                # without rebuilding the table, which is not worth doing to a
                # live ledger for a distinction _campaign_row restores on the
                # way out.
                campaign_id, body.goal, body.base_ttp_id or "",
                json.dumps(body.objectives), body.concurrency,
                _now_iso(), dispatch_until, hard_stop_at,
                body.parent_campaign_id, body.parent_task_id,
            ),
        )
        _emit(conn, "campaign_created", campaign_id, goal=body.goal,
              concurrency=body.concurrency)
        if body.parent_campaign_id:
            _emit(conn, "lead_campaign_opened", body.parent_campaign_id,
                  body.parent_task_id, campaign_id=campaign_id, goal=body.goal,
              concurrency=body.concurrency)
        row = _get_campaign(conn, campaign_id)

    logger.info("campaign %s created: %s", campaign_id, body.goal)
    return _campaign_row(row)


@app.post("/campaigns/{campaign_id}/session")
def attach_session(campaign_id: str, body: SessionAttach) -> dict[str, Any]:
    """Record which conversation this campaign belongs to.

    Separate from creating it because the two are known in different places:
    the campaign is opened by a tool running in a process that serves every
    conversation at once and so cannot know which one asked, while the chat
    gateway knows exactly. Idempotent, and it will not move a campaign from
    one conversation to another -- that is a mistake, not an edit.
    """

    with _write() as conn:
        row = _get_campaign(conn, campaign_id)
        if row["session_id"] and row["session_id"] != body.session_id:
            raise HTTPException(
                status_code=409,
                detail=f"Campaign {campaign_id} already belongs to "
                       f"conversation {row['session_id']}",
            )
        if not row["session_id"]:
            conn.execute(
                "UPDATE campaigns SET session_id = ? WHERE campaign_id = ?",
                (body.session_id, campaign_id),
            )
        row = _get_campaign(conn, campaign_id)
    return _campaign_row(row)


@app.get("/campaigns")
def list_campaigns(session_id: Optional[str] = None) -> list[dict[str, Any]]:
    """Every campaign newest first, each with enough to triage it.

    Task counts, how many results landed, and when the campaign was last
    touched -- so "what ran most recently and did it produce anything" is one
    call rather than one per campaign. Aggregated in two grouped queries
    rather than a query per campaign.

    With ``session_id``, only that conversation's campaigns. Campaigns
    belonging to nobody are left out of a filtered list rather than shown to
    everyone: the question being asked is "which are mine".
    """

    with _read() as conn:
        if session_id:
            rows = conn.execute(
                "SELECT * FROM campaigns WHERE session_id = ? "
                "ORDER BY created_at DESC", (session_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM campaigns ORDER BY created_at DESC"
            ).fetchall()

        counts: dict[str, dict[str, int]] = {}
        latest: dict[str, str] = {}
        for row in conn.execute(
            "SELECT campaign_id, state, COUNT(*) AS n, "
            "MAX(COALESCE(finished_at, claimed_at, created_at)) AS last_at "
            "FROM tasks GROUP BY campaign_id, state"
        ):
            cid = row["campaign_id"]
            counts.setdefault(cid, {})[row["state"]] = row["n"]
            if row["last_at"] and row["last_at"] > latest.get(cid, ""):
                latest[cid] = row["last_at"]

        results: dict[str, int] = {}
        for row in conn.execute(
            "SELECT campaign_id, COUNT(*) AS n, MAX(posted_at) AS last_at "
            "FROM results GROUP BY campaign_id"
        ):
            cid = row["campaign_id"]
            results[cid] = row["n"]
            if row["last_at"] and row["last_at"] > latest.get(cid, ""):
                latest[cid] = row["last_at"]

    overview = []
    for row in rows:
        cid = row["campaign_id"]
        campaign = _campaign_row(row)
        last_at = latest.get(cid) or campaign["created_at"]
        campaign.update({
            "task_counts": counts.get(cid, {}),
            "n_tasks": sum(counts.get(cid, {}).values()),
            "n_results": results.get(cid, 0),
            "last_activity_at": last_at,
            "last_activity_seconds_ago": _age_seconds(last_at),
            "phase": _window_phase(row, time.time()),
        })
        overview.append(campaign)
    return overview


@app.get("/campaigns/{campaign_id}")
def get_campaign(campaign_id: str) -> dict[str, Any]:
    with _read() as conn:
        return _campaign_row(_get_campaign(conn, campaign_id))


@app.post("/campaigns/{campaign_id}/close")
def close_campaign(campaign_id: str) -> dict[str, Any]:
    """Enter the drain window: no new tasks, claimed ones run on."""

    with _write() as conn:
        _get_campaign(conn, campaign_id)
        conn.execute(
            "UPDATE campaigns SET state = 'draining' WHERE campaign_id = ? "
            "AND state = 'dispatching'",
            (campaign_id,),
        )
        _emit(conn, "campaign_closed", campaign_id)
        return _campaign_row(_get_campaign(conn, campaign_id))


@app.post("/campaigns/{campaign_id}/conducted")
def campaign_conducted(campaign_id: str) -> dict[str, Any]:
    """A conductor saying it holds this campaign now; its loop says so every pass.

    Not inferred from reads of the status: the chat window reads that too,
    and a browser left open must not keep a campaign nobody conducts alive.
    """

    with _write() as conn:
        _get_campaign(conn, campaign_id)
        conn.execute(
            "UPDATE campaigns SET conducted_at = ? WHERE campaign_id = ?",
            (time.time(), campaign_id),
        )
        return _campaign_row(_get_campaign(conn, campaign_id))


@app.post("/campaigns/{campaign_id}/concurrency")
def set_concurrency(campaign_id: str, body: ConcurrencyUpdate) -> dict[str, Any]:
    """Change how many supervisors the campaign wants in flight.

    Held here rather than only in the conductor's memory, so the number
    survives a conductor restart and the pool it scales to is always
    reconcilable against the ledger.
    """

    with _write() as conn:
        _get_campaign(conn, campaign_id)
        conn.execute(
            "UPDATE campaigns SET concurrency = ? WHERE campaign_id = ?",
            (body.concurrency, campaign_id),
        )
        _emit(conn, "concurrency_changed", campaign_id, concurrency=body.concurrency)
        return _campaign_row(_get_campaign(conn, campaign_id))


@app.post("/tasks/{task_id}/cancel")
def cancel_task(task_id: str) -> dict[str, Any]:
    """Take a task back, whoever holds it.

    Deliberately not owner-checked: this is the conductor overriding a
    supervisor, not a supervisor reporting on itself. The holder finds out on
    its next heartbeat, which already treats a lost lease as "stop work and do
    not post a result".
    """

    with _write() as conn:
        row = conn.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"No task {task_id}")
        if row["state"] in ("done", "failed", "abandoned"):
            raise HTTPException(
                status_code=409,
                detail=f"Task {task_id} already finished as {row['state']}",
            )

        conn.execute(
            "UPDATE tasks SET state = 'abandoned', finished_at = ?, "
            "lease_owner = NULL, lease_expires_at = NULL, error = ? "
            "WHERE task_id = ?",
            (_now_iso(), "cancelled by the conductor", task_id),
        )
        _emit(conn, "task_cancelled", row["campaign_id"], task_id,
              previous_owner=row["lease_owner"])
        cancelled = conn.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
        ).fetchone()

    logger.warning("task %s cancelled (was held by %s)", task_id, row["lease_owner"])
    return _task_row(cancelled)


@app.post("/campaigns/{campaign_id}/finalize")
def finalize_campaign(campaign_id: str) -> dict[str, Any]:
    """Close the ledger. Still-queued tasks are abandoned, not left dangling."""

    with _write() as conn:
        _get_campaign(conn, campaign_id)
        # Its leads' campaigns end with it: their sub-agents were its team.
        for member in _team(conn, campaign_id):
            conn.execute(
                "UPDATE tasks SET state = 'abandoned', finished_at = ?, "
                "error = 'campaign finalized', lease_owner = NULL, "
                "lease_expires_at = NULL "
                f"WHERE campaign_id = ? AND state IN {OPEN_TASK_STATES}",
                (_now_iso(), member),
            )
            already = conn.execute(
                "SELECT state FROM campaigns WHERE campaign_id = ?", (member,)
            ).fetchone()["state"] == "finalized"
            # A lead finalized on its own already emitted its event; finalizing
            # the campaign above it must not announce the same ending twice.
            if not already:
                conn.execute(
                    "UPDATE campaigns SET state = 'finalized', finalized_at = ? "
                    "WHERE campaign_id = ?",
                    (_now_iso(), member),
                )
                _emit(conn, "campaign_finalized", member)
        return _campaign_row(_get_campaign(conn, campaign_id))


# -- tasks ------------------------------------------------------------------

def _open_for_dispatch(conn: sqlite3.Connection, campaign_id: str) -> sqlite3.Row:
    campaign = _get_campaign(conn, campaign_id)
    if campaign["state"] != "dispatching":
        raise HTTPException(
            status_code=409,
            detail=f"Campaign {campaign_id} is {campaign['state']}; "
                   "the queue is closed to new tasks",
        )
    return campaign


def _insert_task(
    conn: sqlite3.Connection, campaign_id: str, body: TaskCreate, known: set[str],
) -> sqlite3.Row:
    """Queue one task of a campaign already checked to be dispatching.

    *known* is the campaign's task ids, grown by each insert so a batch may
    derive a task from one queued earlier in the same batch.
    """

    # A lineage edge pointing at a task that does not exist, or at one in
    # another campaign, is worse than no edge: it is drawn, believed, and
    # wrong. Refuse it here, where the ids can still be checked.
    unknown = [ref for ref in body.derived_from if ref not in known]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"derived_from names {', '.join(unknown)}, which "
                   f"{'is' if len(unknown) == 1 else 'are'} not a task "
                   f"of campaign {campaign_id}",
        )

    task_id = _new_id()
    conn.execute(
        "INSERT INTO tasks (task_id, campaign_id, spec, state, created_at) "
        "VALUES (?, ?, ?, 'queued', ?)",
        (task_id, campaign_id, body.model_dump_json(), _now_iso()),
    )
    known.add(task_id)
    _emit(conn, "task_enqueued", campaign_id, task_id,
          task_kind=body.kind, title=body.title,
          weight_overrides=body.weight_overrides,
          derived_from=body.derived_from,
          fidelity_tier=body.fidelity_tier,
          workstream=body.workstream, priority=body.priority)
    return conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()


def _task_ids(conn: sqlite3.Connection, campaign_id: str) -> set[str]:
    return {
        row["task_id"] for row in conn.execute(
            "SELECT task_id FROM tasks WHERE campaign_id = ?", (campaign_id,)
        )
    }


@app.post("/campaigns/{campaign_id}/tasks", status_code=201)
def enqueue_task(campaign_id: str, body: TaskCreate) -> dict[str, Any]:
    with _write() as conn:
        _open_for_dispatch(conn, campaign_id)
        row = _insert_task(conn, campaign_id, body, _task_ids(conn, campaign_id))
    return _task_row(row)


@app.post("/campaigns/{campaign_id}/tasks/batch", status_code=201)
def enqueue_tasks(campaign_id: str, body: TaskBatch) -> list[dict[str, Any]]:
    """Queue several tasks at once, all or none.

    One transaction: a batch refused halfway would leave the conductor
    guessing which of its tasks exist.
    """

    with _write() as conn:
        _open_for_dispatch(conn, campaign_id)
        known = _task_ids(conn, campaign_id)
        rows = [_insert_task(conn, campaign_id, task, known) for task in body.tasks]
    return [_task_row(row) for row in rows]


# -- the plan ------------------------------------------------------------------

@app.put("/campaigns/{campaign_id}/plan")
def set_plan(campaign_id: str, body: PlanUpdate) -> dict[str, Any]:
    """The workstreams the goal is split into. Replaces the previous plan."""

    ids = [stream.id for stream in body.workstreams]
    if len(set(ids)) != len(ids):
        raise HTTPException(status_code=400, detail="workstream ids must be unique")
    plan = {"workstreams": [stream.model_dump() for stream in body.workstreams]}
    with _write() as conn:
        _get_campaign(conn, campaign_id)
        conn.execute(
            "UPDATE campaigns SET plan = ? WHERE campaign_id = ?",
            (json.dumps(plan), campaign_id),
        )
        _emit(conn, "plan_set", campaign_id, workstreams=ids)
        return _campaign_row(_get_campaign(conn, campaign_id))


def _team(conn: sqlite3.Connection, campaign_id: str) -> list[str]:
    """The campaign and every lead's campaign below it."""

    return [
        row["campaign_id"] for row in conn.execute(
            "WITH RECURSIVE team(campaign_id) AS ("
            " SELECT ? UNION"
            " SELECT c.campaign_id FROM campaigns c JOIN team t"
            " ON c.parent_campaign_id = t.campaign_id)"
            " SELECT campaign_id FROM team",
            (campaign_id,),
        )
    ]


@app.get("/campaigns/{campaign_id}/coverage")
def coverage(campaign_id: str) -> dict[str, Any]:
    """How the work spreads over the plan: per workstream, what is queued,
    running and done, and the latest finding.

    The view a conductor of many sub-agents decides from instead of rereading
    every report: a workstream nobody is working is a gap, one holding most
    of the team is a bet.
    """

    with _read() as conn:
        campaign = _campaign_row(_get_campaign(conn, campaign_id))
        rows = conn.execute(
            "SELECT t.task_id, t.state, t.spec, r.summary, r.posted_at "
            "FROM tasks t LEFT JOIN results r USING (task_id) "
            "WHERE t.campaign_id = ? ORDER BY t.created_at",
            (campaign_id,),
        ).fetchall()
        open_proposals = conn.execute(
            "SELECT spec FROM proposals WHERE campaign_id = ? AND state = 'open'",
            (campaign_id,),
        ).fetchall()

    streams = (campaign["plan"] or {}).get("workstreams") or []
    by_id: dict[str, dict[str, Any]] = {
        stream["id"]: {**stream, "counts": {}, "latest": None, "proposals": 0}
        for stream in streams
    }
    unplanned = {"id": None, "name": "Not in the plan", "counts": {}, "latest": None, "proposals": 0}
    for row in rows:
        spec = json.loads(row["spec"])
        entry = by_id.get(spec.get("workstream") or "", unplanned)
        entry["counts"][row["state"]] = entry["counts"].get(row["state"], 0) + 1
        if row["summary"] and (entry["latest"] is None or row["posted_at"] > entry["latest"]["posted_at"]):
            entry["latest"] = {"task_id": row["task_id"], "summary": row["summary"], "posted_at": row["posted_at"]}
    for row in open_proposals:
        spec = json.loads(row["spec"])
        by_id.get(spec.get("workstream") or "", unplanned)["proposals"] += 1

    workstreams = list(by_id.values())
    if unplanned["counts"] or unplanned["proposals"]:
        workstreams.append(unplanned)
    for entry in workstreams:
        counts = entry["counts"]
        entry["active"] = counts.get("queued", 0) + counts.get("claimed", 0)
    gaps = [
        entry["id"] for entry in workstreams
        if entry["id"] and entry.get("status", "open") == "open" and not entry["active"]
    ]
    return {
        "campaign_id": campaign_id,
        "goal": campaign["goal"],
        "has_plan": bool(streams),
        "workstreams": workstreams,
        "gaps": gaps,
        "open_proposals": len(open_proposals),
    }


# -- proposals -----------------------------------------------------------------

@app.get("/campaigns/{campaign_id}/proposals")
def list_proposals(campaign_id: str, state: Optional[ProposalState] = "open") -> list[dict[str, Any]]:
    with _read() as conn:
        _get_campaign(conn, campaign_id)
        sql = "SELECT * FROM proposals WHERE campaign_id = ?"
        args: list[Any] = [campaign_id]
        if state:
            sql += " AND state = ?"
            args.append(state)
        rows = conn.execute(sql + " ORDER BY created_at", args).fetchall()
    return [_proposal_row(row) for row in rows]


def _open_proposals(
    conn: sqlite3.Connection, campaign_id: str, ids: list[str],
) -> list[sqlite3.Row]:
    rows = [
        conn.execute(
            "SELECT * FROM proposals WHERE proposal_id = ? AND campaign_id = ?",
            (proposal_id, campaign_id),
        ).fetchone()
        for proposal_id in ids
    ]
    missing = [pid for pid, row in zip(ids, rows) if row is None]
    if missing:
        raise HTTPException(status_code=404, detail=f"No proposal {', '.join(missing)} in campaign {campaign_id}")
    closed = [row["proposal_id"] for row in rows if row["state"] != "open"]
    if closed:
        raise HTTPException(status_code=409, detail=f"Proposal {', '.join(closed)} is no longer open")
    return rows


@app.post("/campaigns/{campaign_id}/proposals/take", status_code=201)
def take_proposals(campaign_id: str, body: ProposalDecision) -> list[dict[str, Any]]:
    """Queue proposals as tasks, each derived from the task that proposed it."""

    with _write() as conn:
        _open_for_dispatch(conn, campaign_id)
        known = _task_ids(conn, campaign_id)
        rows = []
        for proposal in _open_proposals(conn, campaign_id, body.proposal_ids):
            spec = json.loads(proposal["spec"])
            source = proposal["from_task_id"]
            task = TaskCreate(
                kind=spec.get("kind") or "research",
                title=spec.get("title"),
                intent=spec["intent"],
                rationale=spec.get("rationale"),
                workstream=spec.get("workstream"),
                derived_from=[source] if source in known else [],
                priority=body.priority,
            )
            row = _insert_task(conn, campaign_id, task, known)
            conn.execute(
                "UPDATE proposals SET state = 'taken', task_id = ? WHERE proposal_id = ?",
                (row["task_id"], proposal["proposal_id"]),
            )
            rows.append(row)
    return [_task_row(row) for row in rows]


@app.post("/campaigns/{campaign_id}/proposals/drop")
def drop_proposals(campaign_id: str, body: ProposalDecision) -> dict[str, Any]:
    with _write() as conn:
        _get_campaign(conn, campaign_id)
        for proposal in _open_proposals(conn, campaign_id, body.proposal_ids):
            conn.execute(
                "UPDATE proposals SET state = 'dropped' WHERE proposal_id = ?",
                (proposal["proposal_id"],),
            )
    return {"dropped": body.proposal_ids}


# -- the atlas: what the campaign knows ----------------------------------------

def _finding_row(row: sqlite3.Row, notes: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
    return {
        "finding_id": row["finding_id"],
        "campaign_id": row["campaign_id"],
        "branch": row["branch"],
        "claim": row["claim"],
        "detail": row["detail"],
        "evidence": json.loads(row["evidence"]),
        "confidence": row["confidence"],
        "tags": json.loads(row["tags"]),
        "author": row["author"],
        "task_id": row["task_id"],
        "created_at": row["created_at"],
        "supersedes": row["supersedes"],
        "superseded_by": row["superseded_by"],
        "notes": notes if notes is not None else [],
    }


def _note_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "note_id": row["note_id"],
        "finding_id": row["finding_id"],
        "body": row["body"],
        "stance": row["stance"],
        "author": row["author"],
        "task_id": row["task_id"],
        "created_at": row["created_at"],
    }


def _notes_for(
    conn: sqlite3.Connection, finding_ids: list[str],
) -> dict[str, list[dict[str, Any]]]:
    """Every note on these findings, in one query rather than one per finding."""

    if not finding_ids:
        return {}
    marks = ",".join("?" * len(finding_ids))
    rows = conn.execute(
        f"SELECT * FROM finding_notes WHERE finding_id IN ({marks}) ORDER BY note_id",
        finding_ids,
    ).fetchall()
    notes: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        notes.setdefault(row["finding_id"], []).append(_note_row(row))
    return notes


@app.post("/campaigns/{campaign_id}/findings", status_code=201)
def record_finding(campaign_id: str, body: FindingPost) -> dict[str, Any]:
    """Record something the campaign now knows.

    Superseding is the only way to take a finding back, and it is checked
    here: the superseded finding has to belong to this campaign, so one
    campaign cannot rewrite another's record.
    """

    with _write() as conn:
        _get_campaign(conn, campaign_id)
        if body.supersedes:
            old = conn.execute(
                "SELECT finding_id FROM findings WHERE finding_id = ? AND campaign_id = ?",
                (body.supersedes, campaign_id),
            ).fetchone()
            if old is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"No finding {body.supersedes} in campaign {campaign_id}",
                )
        finding_id = _new_id()
        created_at = _now_iso()
        conn.execute(
            "INSERT INTO findings (finding_id, campaign_id, branch, claim, detail, "
            "evidence, confidence, tags, author, task_id, created_at, supersedes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                finding_id, campaign_id, body.branch, body.claim, body.detail,
                json.dumps(body.evidence), body.confidence, json.dumps(body.tags),
                body.author, body.task_id, created_at, body.supersedes,
            ),
        )
        if body.supersedes:
            conn.execute(
                "UPDATE findings SET superseded_by = ? WHERE finding_id = ?",
                (finding_id, body.supersedes),
            )
        _emit(
            conn, "finding_recorded", campaign_id, task_id=body.task_id,
            finding_id=finding_id, branch=body.branch, claim=body.claim,
        )
        row = conn.execute(
            "SELECT * FROM findings WHERE finding_id = ?", (finding_id,)
        ).fetchone()
    return _finding_row(row)


@app.post("/campaigns/{campaign_id}/findings/{finding_id}/notes", status_code=201)
def note_finding(campaign_id: str, finding_id: str, body: NotePost) -> dict[str, Any]:
    """Hang a second opinion off a finding, without overwriting it."""

    with _write() as conn:
        _get_campaign(conn, campaign_id)
        found = conn.execute(
            "SELECT finding_id FROM findings WHERE finding_id = ? AND campaign_id = ?",
            (finding_id, campaign_id),
        ).fetchone()
        if found is None:
            raise HTTPException(
                status_code=404,
                detail=f"No finding {finding_id} in campaign {campaign_id}",
            )
        cursor = conn.execute(
            "INSERT INTO finding_notes (finding_id, campaign_id, body, stance, "
            "author, task_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                finding_id, campaign_id, body.body, body.stance,
                body.author, body.task_id, _now_iso(),
            ),
        )
        _emit(
            conn, "finding_noted", campaign_id, task_id=body.task_id,
            finding_id=finding_id, stance=body.stance,
        )
        row = conn.execute(
            "SELECT * FROM finding_notes WHERE note_id = ?", (cursor.lastrowid,)
        ).fetchone()
    return _note_row(row)


@app.get("/campaigns/{campaign_id}/findings")
def list_findings(
    campaign_id: str,
    branch: Optional[str] = None,
    q: Optional[str] = None,
    tag: Optional[str] = None,
    limit: int = 50,
    include_superseded: bool = False,
) -> dict[str, Any]:
    """Findings, narrowed by branch, by words or by tag.

    ``q`` matches a finding that contains every word somewhere in its claim,
    detail or tags. LIKE rather than a full-text index: a campaign records
    hundreds of findings, not millions, and an index is a schema to migrate
    and keep in step for a scan that takes a millisecond.
    """

    limit = max(1, min(limit, 200))
    sql = "SELECT * FROM findings WHERE campaign_id = ?"
    args: list[Any] = [campaign_id]
    if branch:
        sql += " AND branch = ?"
        args.append(_branch_key(branch))
    if not include_superseded:
        sql += " AND superseded_by IS NULL"
    if tag:
        sql += " AND lower(tags) LIKE ?"
        args.append(f'%"{tag.lower()}"%')
    for word in (q or "").split():
        sql += " AND lower(claim || ' ' || COALESCE(detail, '') || ' ' || tags) LIKE ?"
        args.append(f"%{word.lower()}%")

    with _read() as conn:
        _get_campaign(conn, campaign_id)
        rows = conn.execute(
            sql + " ORDER BY created_at DESC LIMIT ?", args + [limit]
        ).fetchall()
        notes = _notes_for(conn, [row["finding_id"] for row in rows])
    return {
        "campaign_id": campaign_id,
        "branch": branch,
        "findings": [
            _finding_row(row, notes.get(row["finding_id"], [])) for row in rows
        ],
    }


@app.get("/campaigns/{campaign_id}/atlas")
def atlas(campaign_id: str) -> dict[str, Any]:
    """The map: which branches hold what, without the findings themselves.

    Branches come from the plan, so one the plan names but nobody has filed
    anything under still shows -- an empty branch is the most useful thing
    here, being the part of the goal the campaign knows nothing about.
    """

    with _read() as conn:
        campaign = _campaign_row(_get_campaign(conn, campaign_id))
        rows = conn.execute(
            "SELECT * FROM findings WHERE campaign_id = ? AND superseded_by IS NULL "
            "ORDER BY created_at",
            (campaign_id,),
        ).fetchall()
        disputed = {
            row["finding_id"]
            for row in conn.execute(
                "SELECT DISTINCT finding_id FROM finding_notes "
                "WHERE campaign_id = ? AND stance = 'disputes'",
                (campaign_id,),
            ).fetchall()
        }

    streams = (campaign["plan"] or {}).get("workstreams") or []
    by_branch: dict[str, dict[str, Any]] = {
        _branch_key(stream["id"]): {
            "branch": _branch_key(stream["id"]), "name": stream["name"],
            "in_plan": True,
            "findings": 0, "disputed": 0, "latest": None, "tags": [],
        }
        for stream in streams
    }
    seen: dict[str, set[str]] = {}
    for row in rows:
        entry = by_branch.setdefault(row["branch"], {
            "branch": row["branch"], "name": row["branch"], "in_plan": False,
            "findings": 0, "disputed": 0, "latest": None, "tags": [],
        })
        entry["findings"] += 1
        if row["finding_id"] in disputed:
            entry["disputed"] += 1
        # Rows arrive oldest first, so the last to overwrite this is the newest.
        entry["latest"] = {
            "finding_id": row["finding_id"],
            "claim": row["claim"],
            "confidence": row["confidence"],
            "created_at": row["created_at"],
        }
        seen.setdefault(row["branch"], set()).update(json.loads(row["tags"]))
    for branch, tags in seen.items():
        by_branch[branch]["tags"] = sorted(tags)[:12]

    branches = list(by_branch.values())
    return {
        "campaign_id": campaign_id,
        "goal": campaign["goal"],
        "findings": len(rows),
        "branches": branches,
        "empty_branches": [b["branch"] for b in branches if not b["findings"]],
    }


# -- how well the team is used -------------------------------------------------

def _epoch(stamp: Optional[str]) -> Optional[float]:
    if not stamp:
        return None
    try:
        then = datetime.fromisoformat(stamp)
    except ValueError:
        return None
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return then.timestamp()


@app.get("/campaigns/{campaign_id}/metrics")
def metrics(campaign_id: str) -> dict[str, Any]:
    """Whether the team is kept busy, measured rather than assumed.

    ``utilisation`` is the share of the slots' time spent holding a task,
    from the first claim until now (or until the campaign finalized), over
    the campaign and its leads' campaigns. ``handoff_seconds`` is how long a
    sub-agent waited between finishing one task and claiming its next: the
    ping-pong, timed.
    """

    with _write() as conn:
        campaign = _get_campaign(conn, campaign_id)
        _sweep_expired_leases(conn)
        team = _team(conn, campaign_id)
        marks = ",".join("?" * len(team))
        rows = conn.execute(
            f"SELECT task_id, state, claimed_at, finished_at, claimed_by FROM tasks "
            f"WHERE campaign_id IN ({marks})",
            team,
        ).fetchall()

    end = _epoch(campaign["finalized_at"]) or time.time()
    claims = [(row, _epoch(row["claimed_at"])) for row in rows]
    started = [at for _, at in claims if at is not None]
    first = min(started) if started else None
    busy = 0.0
    for row, at in claims:
        if at is None:
            continue
        stop = _epoch(row["finished_at"]) or (end if row["state"] == "claimed" else None)
        if stop is not None:
            busy += max(0.0, min(stop, end) - at)
    slots = max(int(campaign["concurrency"] or 1), 1)
    span = (end - first) if first is not None else 0.0
    utilisation = round(min(1.0, busy / (slots * span)), 3) if span > 0 else None

    by_owner: dict[str, list[tuple[float, Optional[float]]]] = {}
    for row, at in claims:
        if at is not None and row["claimed_by"]:
            by_owner.setdefault(row["claimed_by"], []).append((at, _epoch(row["finished_at"])))
    handoffs: list[float] = []
    for spans in by_owner.values():
        spans.sort()
        for (_, done), (nxt, _) in zip(spans, spans[1:]):
            if done is not None:
                handoffs.append(max(0.0, nxt - done))
    handoffs.sort()
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["state"]] = counts.get(row["state"], 0) + 1
    running = counts.get("claimed", 0)
    return {
        "campaign_id": campaign_id,
        "team": team,
        "concurrency": slots,
        "running": running,
        "queued": counts.get("queued", 0),
        "free_slots": max(0, slots - running),
        "backlog_target": backlog_target(slots),
        "utilisation": utilisation,
        "busy_seconds": round(busy, 1),
        "handoffs": len(handoffs),
        "handoff_seconds": {
            "median": round(handoffs[len(handoffs) // 2], 1) if handoffs else None,
            "max": round(handoffs[-1], 1) if handoffs else None,
        },
    }


def backlog_target(concurrency: int) -> int:
    """Tasks to keep queued beyond the running ones, so a sub-agent that
    finishes claims its next at once instead of waiting on the conductor.
    Half the team, at least one."""

    return max(1, math.ceil(concurrency / 2))


@app.get("/campaigns/{campaign_id}/tasks")
def list_tasks(
    campaign_id: str, state: Optional[TaskState] = None
) -> list[dict[str, Any]]:
    with _read() as conn:
        _get_campaign(conn, campaign_id)
        sql = "SELECT * FROM tasks WHERE campaign_id = ?"
        args: list[Any] = [campaign_id]
        if state:
            sql += " AND state = ?"
            args.append(state)
        rows = conn.execute(sql + " ORDER BY created_at", args).fetchall()
    return [_task_row(r) for r in rows]


@app.post("/tasks/claim")
def claim_task(body: ClaimRequest) -> Any:
    """Take the oldest queued task and lease it.

    Answers 204 when there is nothing to claim, which is the normal case for an
    idle supervisor and not worth an error.
    """

    with _write() as conn:
        _sweep_expired_leases(conn, body.campaign_id)

        sql = (
            "SELECT t.* FROM tasks t JOIN campaigns c USING (campaign_id) "
            "WHERE t.state = 'queued' AND c.state IN ('dispatching', 'draining')"
        )
        args: list[Any] = []
        if body.campaign_id:
            # Its leads' campaigns draw on the same pool.
            team = _team(conn, body.campaign_id)
            sql += f" AND t.campaign_id IN ({','.join('?' * len(team))})"
            args.extend(team)
        else:
            # A campaign whose conductor has gone quiet -- its conversation's
            # session ended, its gateway restarted -- keeps its queue, but a
            # supervisor working any campaign must not take from it: the
            # oldest task first would otherwise hand the next session's pool
            # to work nobody is reading. One that never had a conductor, a
            # campaign fed from the command line, is claimable as it was.
            sql += " AND (c.conducted_at IS NULL OR c.conducted_at >= ?)"
            args.append(time.time() - CONDUCTED_WITHIN_SECONDS)

        row = conn.execute(
            sql + " ORDER BY COALESCE(json_extract(t.spec, '$.priority'), 0) DESC, "
            "t.created_at LIMIT 1",
            args,
        ).fetchone()
        if row is None:
            return Response(status_code=204)

        expires = time.time() + body.lease_seconds
        conn.execute(
            "UPDATE tasks SET state = 'claimed', lease_owner = ?, claimed_by = ?, "
            "lease_expires_at = ?, claimed_at = ?, attempts = attempts + 1 "
            "WHERE task_id = ?",
            (body.owner, body.owner, expires, _now_iso(), row["task_id"]),
        )
        _emit(conn, "task_claimed", row["campaign_id"], row["task_id"],
              owner=body.owner, lease_seconds=body.lease_seconds)

        claimed = conn.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (row["task_id"],)
        ).fetchone()

    logger.info("task %s claimed by %s", claimed["task_id"], body.owner)
    return _task_row(claimed)


@app.post("/tasks/{task_id}/heartbeat")
def heartbeat(task_id: str, body: HeartbeatRequest) -> dict[str, Any]:
    with _write() as conn:
        existing = _owned_task(conn, task_id, body.owner)
        expires = time.time() + body.lease_seconds
        conn.execute(
            "UPDATE tasks SET lease_expires_at = ? WHERE task_id = ?",
            (expires, task_id),
        )
        # Written once and then left alone. A retried task runs a second job,
        # and overwriting would silently repoint anyone watching the first
        # at a run that is not the one they opened.
        if body.job_id and not existing["job_id"]:
            conn.execute(
                "UPDATE tasks SET job_id = ? WHERE task_id = ?",
                (body.job_id, task_id),
            )
            _emit(conn, "task_started_job", existing["campaign_id"], task_id,
                  job_id=body.job_id, owner=body.owner)
        row = conn.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
        ).fetchone()
    return _task_row(row)


# -- the pool itself --------------------------------------------------------

def _sweep_dead_supervisors(conn: sqlite3.Connection) -> int:
    """Forget supervisors that stopped reporting in.

    Swept on read for the same reason leases are: a stale row read as a live
    supervisor is what makes a conductor dispatch into a pool that cannot
    answer.
    """

    cursor = conn.execute(
        "DELETE FROM supervisors WHERE expires_at < ?", (time.time(),)
    )
    return cursor.rowcount or 0


@app.post("/supervisors/{owner}/presence")
def post_presence(owner: str, body: PresencePost) -> dict[str, Any]:
    """A supervisor saying it is alive, and whether it is free.

    Sent on every pass of the claim loop, including the passes that find the
    queue empty -- those are precisely the ones that tell a conductor it has
    somewhere to dispatch to.
    """

    now = _now_iso()
    with _write() as conn:
        conn.execute(
            "INSERT INTO supervisors (owner, campaign_id, state, task_id, "
            "tasks_completed, tasks_failed, detail, first_seen_at, "
            "last_seen_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(owner) DO UPDATE SET "
            "campaign_id = excluded.campaign_id, state = excluded.state, "
            "task_id = excluded.task_id, "
            "tasks_completed = excluded.tasks_completed, "
            "tasks_failed = excluded.tasks_failed, "
            "detail = excluded.detail, last_seen_at = excluded.last_seen_at, "
            "expires_at = excluded.expires_at",
            (
                owner, body.campaign_id, body.state, body.task_id,
                body.tasks_completed, body.tasks_failed, body.detail,
                now, now, time.time() + body.ttl_seconds,
            ),
        )
        row = conn.execute(
            "SELECT * FROM supervisors WHERE owner = ?", (owner,)
        ).fetchone()
    return _supervisor_row(row)


@app.delete("/supervisors/{owner}")
def drop_presence(owner: str) -> dict[str, Any]:
    """A supervisor leaving on purpose, rather than being timed out."""

    with _write() as conn:
        conn.execute("DELETE FROM supervisors WHERE owner = ?", (owner,))
    return {"owner": owner, "state": "gone"}


@app.get("/supervisors")
def list_supervisors(campaign_id: Optional[str] = None) -> dict[str, Any]:
    """The pool: who is alive, and which of them can take work now.

    ``idle`` is the number that would answer a dispatch immediately. It is not
    ``concurrency`` and not the Deployment's replica count: a pod that is
    starting, wedged, or shutting down is not somewhere to send a task.
    """

    with _write() as conn:
        _sweep_dead_supervisors(conn)
        sql = "SELECT * FROM supervisors"
        args: list[Any] = []
        if campaign_id:
            # A supervisor with no campaign claims from any of them, so it
            # counts towards this campaign's pool too.
            sql += " WHERE campaign_id = ? OR campaign_id IS NULL"
            args.append(campaign_id)
        rows = conn.execute(sql + " ORDER BY owner", args).fetchall()

    supervisors = [_supervisor_row(r) for r in rows]
    by_state: dict[str, int] = {}
    for supervisor in supervisors:
        by_state[supervisor["state"]] = by_state.get(supervisor["state"], 0) + 1
    return {
        "supervisors": supervisors,
        "n_alive": len(supervisors),
        "idle": by_state.get("idle", 0),
        "busy": by_state.get("busy", 0),
        "by_state": by_state,
    }


@app.post("/tasks/{task_id}/result")
def post_result(task_id: str, body: ResultPost) -> dict[str, Any]:
    report = body.report
    if report and len(report) > MAX_REPORT_CHARS:
        report = report[:MAX_REPORT_CHARS] + "\n\n[report truncated by the ledger]"
    job_ids = list(dict.fromkeys(
        [*body.job_ids, *([body.job_id] if body.job_id else [])]
    ))

    with _write() as conn:
        task = _owned_task(conn, task_id, body.owner)
        spec = json.loads(task["spec"])
        conn.execute(
            "INSERT OR REPLACE INTO results (task_id, campaign_id, job_id, "
            "objective_profile, candidates, summary, posted_at, report, job_ids) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                task_id, task["campaign_id"], body.job_id,
                json.dumps(body.objective_profile), json.dumps(body.candidates),
                body.summary, _now_iso(), report, json.dumps(job_ids),
            ),
        )
        conn.execute(
            "UPDATE tasks SET state = 'done', finished_at = ?, lease_owner = NULL, "
            "lease_expires_at = NULL WHERE task_id = ?",
            (_now_iso(), task_id),
        )
        for follow_up in body.follow_ups:
            proposal = follow_up.model_dump()
            # Unless it says otherwise, a next step continues its task's workstream.
            proposal["workstream"] = proposal.get("workstream") or spec.get("workstream")
            conn.execute(
                "INSERT INTO proposals (proposal_id, campaign_id, from_task_id, spec, "
                "state, created_at) VALUES (?, ?, ?, ?, 'open', ?)",
                (_new_id(), task["campaign_id"], task_id, json.dumps(proposal), _now_iso()),
        )
        _emit(conn, "task_done", task["campaign_id"], task_id,
              owner=body.owner, objective_profile=body.objective_profile,
              n_candidates=len(body.candidates),
              task_kind=spec.get("kind") or "design", title=spec.get("title"),
              summary=body.summary, job_ids=job_ids,
              workstream=spec.get("workstream"),
              n_follow_ups=len(body.follow_ups))
        row = conn.execute(
            "SELECT * FROM results WHERE task_id = ?", (task_id,)
        ).fetchone()

    logger.info("task %s done: %s", task_id, body.objective_profile)
    return _result_row(row)


@app.post("/tasks/{task_id}/fail")
def fail_task(task_id: str, body: FailurePost) -> dict[str, Any]:
    """Report a task as failed, requeuing it unless told not to."""

    with _write() as conn:
        task = _owned_task(conn, task_id, body.owner)
        requeue = body.requeue and task["attempts"] < TASK_MAX_ATTEMPTS

        if requeue:
            conn.execute(
                "UPDATE tasks SET state = 'queued', lease_owner = NULL, "
                "lease_expires_at = NULL, claimed_at = NULL, error = ? "
                "WHERE task_id = ?",
                (body.error, task_id),
            )
        else:
            conn.execute(
                "UPDATE tasks SET state = 'failed', finished_at = ?, "
                "lease_owner = NULL, lease_expires_at = NULL, error = ? "
                "WHERE task_id = ?",
                (_now_iso(), body.error, task_id),
            )

        _emit(conn, "task_requeued" if requeue else "task_failed",
              task["campaign_id"], task_id, owner=body.owner, error=body.error)
        row = conn.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
        ).fetchone()

    logger.warning("task %s failed (%s): %s", task_id,
                   "requeued" if requeue else "terminal", body.error)
    return _task_row(row)


# -- results, events --------------------------------------------------------

# Scorer columns that say whether an objective's value means anything, as
# opposed to re-expressing it. Kept beside the objectives they qualify: an
# affinity from a failed fold is not a good score, and a reader who cannot see
# that is worse off than one who sees no affinity at all.
VALIDITY_METRICS = ("skip_reason", "confidence_score", "ligand_iptm")


def _objective_scores(
    components: dict[str, Any], objectives: list[str]
) -> dict[str, Any]:
    """An objective's own score plus whatever qualifies it."""

    scores = {name: components[name] for name in objectives if name in components}
    for objective in objectives:
        for metric in VALIDITY_METRICS:
            key = f"{metric} ({objective})"
            value = components.get(key)
            if value is None:
                continue
            # A null ``transformed`` is the scorer saying "nothing to report",
            # which for skip_reason is the good case and not worth the tokens.
            if isinstance(value, dict) and value.get("transformed") is None:
                continue
            scores[key] = value
    return scores


def _scored_candidate(
    candidate: dict[str, Any], objectives: list[str]
) -> dict[str, Any]:
    """One molecule with the scores the campaign is actually steering on.

    A harvested candidate carries every column the scorer emitted, which for
    Boltz is ten per-molecule columns beside the one objective they explain.
    That is 906 characters per molecule, so a 25-molecule result is ~22k --
    two of them overflow the conductor's 32768-token context on their own.

    What is dropped is the redundant half: six ``affinity_pred_value*`` and
    ``affinity_probability_binary*`` re-expressions of the same prediction.
    What is kept is the objectives and the validity signals above, because a
    conductor shown an affinity with no confidence beside it will narrate a
    quality gate it cannot see -- one did, inventing PoseBusters numbers for
    a campaign that never ran PoseBusters.
    """

    components = candidate.get("components") or {}
    scored = {
        "smiles": candidate.get("smiles"),
        "total_score": candidate.get("total_score"),
        "components": _objective_scores(components, objectives),
    }
    # The run that scored it: eight characters, and how its complex is found.
    if candidate.get("job_id"):
        scored["job_id"] = candidate["job_id"]
    return scored


@app.get("/campaigns/{campaign_id}/results")
def list_results(campaign_id: str, full: bool = False) -> list[dict[str, Any]]:
    """Finished results, narrowed to the campaign's objectives.

    ``full=true`` returns the stored record untouched, including every scorer
    diagnostic. The ledger keeps all of it; this endpoint defaults to the
    narrow view because its caller is an LLM with a bounded context.

    A campaign whose tasks bring their own TTPs declares no objectives, so
    there is no campaign-wide list to narrow to. Each result is then narrowed
    to the objectives it reported for itself, which is the same idea applied
    one task at a time -- and not to nothing, which would hand the conductor
    molecules with no scores and let it narrate numbers it cannot see.
    """

    with _read() as conn:
        campaign = _get_campaign(conn, campaign_id)
        declared = json.loads(campaign["objectives"])
        rows = conn.execute(
            "SELECT * FROM results WHERE campaign_id = ? ORDER BY posted_at",
            (campaign_id,),
        ).fetchall()

    results = [_result_row(r) for r in rows]
    if full:
        return results

    narrowed = []
    for result in results:
        objectives = declared or list(result.get("objective_profile") or {})
        candidates = result.get("candidates") or []
        narrowed.append({
            **result,
            "objective_profile": _objective_scores(
                result["objective_profile"], objectives
            ),
            "candidates": [
                _scored_candidate(c, objectives) for c in candidates
            ],
            "detail_hint": (
                "These are the campaign's objectives and the signals that "
                "qualify them. The scorer's remaining per-molecule columns "
                "are in the job's results CSV via read_results(job_id)."
            ),
        })
    return narrowed


@app.get("/campaigns/{campaign_id}/events")
def list_events(campaign_id: str, since: int = 0, limit: int = 200) -> dict[str, Any]:
    """The conductor's completion feed: poll with the last event_id it saw."""

    with _read() as conn:
        _get_campaign(conn, campaign_id)
        rows = conn.execute(
            "SELECT * FROM events WHERE campaign_id = ? AND event_id > ? "
            "ORDER BY event_id LIMIT ?",
            (campaign_id, since, limit),
        ).fetchall()

    events = [
        {
            "event_id": r["event_id"],
            "task_id": r["task_id"],
            "kind": r["kind"],
            "payload": json.loads(r["payload"]),
            "created_at": r["created_at"],
        }
        for r in rows
    ]
    return {
        "events": events,
        "last_event_id": events[-1]["event_id"] if events else since,
    }


@app.get("/campaigns/{campaign_id}/status")
def campaign_status(campaign_id: str) -> dict[str, Any]:
    """The aggregate the conductor REPL prints as /status.

    Sweeps expired leases first, so in-flight counts reflect supervisors that
    are actually still alive rather than ones that died holding a task.
    """

    with _write() as conn:
        campaign = _get_campaign(conn, campaign_id)
        _sweep_expired_leases(conn, campaign_id)

        counts = {
            row["state"]: row["n"]
            for row in conn.execute(
                "SELECT state, COUNT(*) AS n FROM tasks WHERE campaign_id = ? "
                "GROUP BY state",
                (campaign_id,),
            )
        }
        in_flight = [
            {
                "task_id": r["task_id"],
                "owner": r["lease_owner"],
                "claimed_at": r["claimed_at"],
                "lease_expires_in": round(r["lease_expires_at"] - time.time(), 1),
                "spec": json.loads(r["spec"]),
            }
            for r in conn.execute(
                "SELECT * FROM tasks WHERE campaign_id = ? AND state = 'claimed' "
                "ORDER BY claimed_at",
                (campaign_id,),
            )
        ]
        n_results = conn.execute(
            "SELECT COUNT(*) FROM results WHERE campaign_id = ?", (campaign_id,)
        ).fetchone()[0]
        team = _team(conn, campaign_id)
        team_counts = dict(counts)
        if len(team) > 1:
            marks = ",".join("?" * (len(team) - 1))
            for row in conn.execute(
                f"SELECT state, COUNT(*) AS n FROM tasks WHERE campaign_id IN ({marks}) "
                "GROUP BY state",
                team[1:],
            ):
                team_counts[row["state"]] = team_counts.get(row["state"], 0) + row["n"]
        open_proposals = conn.execute(
            "SELECT COUNT(*) FROM proposals WHERE campaign_id = ? AND state = 'open'",
            (campaign_id,),
        ).fetchone()[0]

    now = time.time()

    return {
        "campaign": _campaign_row(campaign),
        "task_counts": counts,
        # With the leads' campaigns: what the shared pool is actually holding.
        "team_counts": team_counts,
        "team": team,
        "backlog_target": backlog_target(int(campaign["concurrency"] or 1)),
        "open_proposals": open_proposals,
        "in_flight": in_flight,
        "n_results": n_results,
        "window": {
            "dispatch_remaining": (
                round(campaign["dispatch_until"] - now, 1)
                if campaign["dispatch_until"] else None
            ),
            "hard_stop_remaining": (
                round(campaign["hard_stop_at"] - now, 1)
                if campaign["hard_stop_at"] else None
            ),
            "phase": _window_phase(campaign, now),
        },
    }


def _window_phase(campaign: sqlite3.Row, now: float) -> str:
    """Which deadline window the campaign is in, by the clock not the state.

    Reported separately from ``campaign.state`` so a conductor that died and
    missed its own deadline can see that it should be draining even though
    nothing has moved the state yet.
    """

    if campaign["state"] in ("finalized", "cancelled"):
        return campaign["state"]
    if campaign["dispatch_until"] is None:
        return "dispatching"
    if now < campaign["dispatch_until"]:
        return "dispatching"
    if campaign["hard_stop_at"] is not None and now < campaign["hard_stop_at"]:
        return "draining"
    return "past_hard_stop"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("CAMPAIGN_STORE_PORT", "8087")),
    )
