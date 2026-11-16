// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * The history of a conversation's work, as one line per step.
 *
 * Read from the ledger's own events rather than from any agent's account of
 * itself: who took which task, what run it started, what came back. The
 * conductor's log is folded in beside them, because what it decided and what
 * the ledger recorded are one story told from two sides.
 *
 * Drawn like a branch graph: one lane per agent, a node on the lane that
 * acted. Pure, so the shape of the history can be tested without a DOM.
 */

import type { CampaignDetail, CampaignFinding, FleetEvent } from "../types/api";
import { byAgent } from "./chemicalSpace";
import { CONDUCTOR, UNCLAIMED, taskAgent } from "./fleet";

/** What sort of step it is, which decides how it reads and what it opens. */
export type StepKind = "campaign" | "handout" | "claim" | "run" | "report" | "failure" | "note" | "close";

/** One thing that happened, in the terms the history reads out. */
export interface Step {
  /** Stable across polls, so React keeps the row it has. */
  id: string;
  at: number;
  /** C, or S1, S2… for the sub-agent that did it; S while nobody has claimed it. */
  agent: string;
  kind: StepKind;
  /** One line: what happened. */
  text: string;
  /** The rest of it, when there is more: the goal, the error, what was asked for. */
  detail: string | null;
  campaignId: string | null;
  taskId: string | null;
  jobId: string | null;
  /** On a handout: the tasks whose results the conductor wrote it from. */
  parents?: readonly string[];
}

export interface History {
  /** Newest first, which is the order it is read in. */
  steps: Step[];
  /** The agents with a lane in the graph, C first, then S1, S2… */
  lanes: string[];
}

/** Rows that say the same thing as the task they are about; the history reads better without them. */
const SKIP = new Set(["task_requeued"]);

/**
 * The history of every campaign of a conversation, newest first.
 *
 * *findings* are the conductor's log, which is kept against the conversation
 * and not against any campaign, so they are placed by time alone.
 */
export function history(
  campaigns: readonly (Pick<CampaignDetail, "campaign" | "tasks"> & { events?: readonly FleetEvent[] })[],
  labels: ReadonlyMap<string, string>,
  findings: readonly CampaignFinding[] = [],
): History {
  const steps: Step[] = [];
  const lanes = new Set<string>([CONDUCTOR]);

  for (const detail of campaigns) {
    const campaignId = detail.campaign.campaign_id;
    const tasks = new Map(detail.tasks.map((task) => [task.task_id, task]));
    const titles = new Map(detail.tasks.map((task) => [task.task_id, title(task.spec?.title, task.spec?.intent, task.task_id)]));

    for (const event of detail.events ?? []) {
      if (SKIP.has(event.kind)) continue;
      const step = asStep(event, campaignId, titles, tasks, labels);
      if (!step) continue;
      lanes.add(step.agent);
      steps.push(step);
    }
  }

  for (const [index, finding] of findings.entries()) {
    const at = Date.parse(finding.ts || "");
    if (!finding.text) continue;
    steps.push({
      id: `note:${finding.ts}:${index}`,
      at: Number.isNaN(at) ? 0 : at,
      agent: CONDUCTOR,
      kind: "note",
      text: finding.text,
      detail: null,
      campaignId: null,
      taskId: null,
      jobId: null,
    });
  }

  // Newest first, and within one moment the later event id last: a task
  // claimed and started in the same second must not read the other way round.
  steps.sort((a, b) => b.at - a.at || b.id.localeCompare(a.id));
  return { steps, lanes: [...lanes].sort(byAgent) };
}

function asStep(
  event: FleetEvent,
  campaignId: string,
  titles: ReadonlyMap<string, string>,
  tasks: ReadonlyMap<string, CampaignDetail["tasks"][number]>,
  labels: ReadonlyMap<string, string>,
): Step | null {
  const at = Date.parse(event.created_at || "");
  const taskId = event.task_id;
  const task = taskId ? tasks.get(taskId) : undefined;
  const of = taskId ? (titles.get(taskId) ?? taskId) : "";
  // Who the ledger says did it, falling back to the task's runner: a run
  // started before the claim was recorded would otherwise have no lane.
  const owner = text(event.payload.owner);
  const agent = owner ? (labels.get(owner) ?? UNCLAIMED) : task ? taskAgent(task, labels) : CONDUCTOR;
  const base = {
    id: `${campaignId}:${event.event_id}`,
    at: Number.isNaN(at) ? 0 : at,
    campaignId,
    taskId,
    jobId: text(event.payload.job_id),
  };

  switch (event.kind) {
    case "campaign_created":
      return { ...base, agent: CONDUCTOR, kind: "campaign", text: "Opened a campaign", detail: text(event.payload.goal) };
    case "task_enqueued":
      return {
        ...base,
        agent: CONDUCTOR,
        kind: "handout",
        text: `Gave out ${of}`,
        detail: task?.spec?.intent?.trim() || null,
        parents: (task?.spec?.derived_from ?? []).filter((parent) => parent !== taskId),
      };
    case "concurrency_changed":
      return { ...base, agent: CONDUCTOR, kind: "campaign", text: `Asked for ${count(event.payload.concurrency)} sub-agents at once`, detail: null };
    case "task_claimed":
      return { ...base, agent, kind: "claim", text: `Took ${of}`, detail: owner };
    case "task_started_job":
      return { ...base, agent, kind: "run", text: `Started a run for ${of}`, detail: base.jobId };
    case "task_done":
      return { ...base, agent, kind: "report", text: `Reported on ${of}`, detail: molecules(event.payload.n_candidates) };
    case "task_failed":
      return { ...base, agent, kind: "failure", text: `Failed ${of}`, detail: text(event.payload.error) };
    case "task_cancelled":
      return { ...base, agent: CONDUCTOR, kind: "failure", text: `Called off ${of}`, detail: null };
    case "campaign_closed":
      return { ...base, agent: CONDUCTOR, kind: "close", text: "Closed the queue to new tasks", detail: null };
    case "campaign_finalized":
      return { ...base, agent: CONDUCTOR, kind: "close", text: "Closed the campaign", detail: null };
    default:
      return null;
  }
}

/** What a task is called in one line of history. */
function title(...candidates: (string | undefined)[]): string {
  const found = candidates.map((candidate) => (candidate ?? "").trim()).find(Boolean) ?? "";
  return found.length > 44 ? `${found.slice(0, 43).trimEnd()}…` : found;
}

function text(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function count(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function molecules(value: unknown): string | null {
  const n = count(value);
  return n > 0 ? `${n} molecule${n === 1 ? "" : "s"}` : null;
}

/** Everything one agent did, newest first. */
export function stepsOf(steps: readonly Step[], agent: string): Step[] {
  return steps.filter((step) => step.agent === agent);
}

/**
 * One task's line in the graph, in row numbers of *steps* (0 is the newest).
 *
 * It forks off the conductor's lane at the handout, runs on the sub-agent's
 * lane from the claim to the report, and merges back only where the
 * conductor wrote a task from what it found: a result nothing built on is a
 * branch that ends, which is the point of drawing it this way.
 */
export interface Branch {
  agent: string;
  taskId: string;
  /** The handout, on the conductor's lane; null when it is not in the history. */
  from: number | null;
  /** The oldest of the task's steps on the sub-agent's lane. */
  start: number;
  /** Its report or failure; null while it is still being worked on, drawn up to the top. */
  end: number | null;
  /** Handouts written from its results. */
  into: number[];
}

const ENDS: ReadonlySet<StepKind> = new Set(["report", "failure"]);

/** The branches of a history, newest first as `history` returns it. */
export function branches(steps: readonly Step[]): Branch[] {
  const handout = new Map<string, number>();
  const builtOn = new Map<string, number[]>();
  const rows = new Map<string, { agent: string; rows: number[]; end: number | null }>();
  steps.forEach((step, row) => {
    if (!step.taskId) return;
    if (step.agent === CONDUCTOR) {
      if (step.kind !== "handout") return;
      handout.set(step.taskId, row);
      for (const parent of step.parents ?? []) builtOn.set(parent, [...(builtOn.get(parent) ?? []), row]);
      return;
    }
    if (step.agent === UNCLAIMED) return;
    const seen = rows.get(step.taskId) ?? { agent: step.agent, rows: [], end: null };
    seen.rows.push(row);
    // Newest first, so the first end met is the last one; a task claimed
    // again after failing ends where its final attempt did.
    if (ENDS.has(step.kind) && seen.end === null) seen.end = row;
    rows.set(step.taskId, seen);
  });

  return [...rows].map(([taskId, { agent, rows: on, end }]) => {
    const start = Math.max(...on);
    const from = handout.get(taskId);
    return {
      agent,
      taskId,
      from: from !== undefined && from > start ? from : null,
      start,
      end,
      into: end === null ? [] : (builtOn.get(taskId) ?? []).filter((row) => row < end),
    };
  });
}

/** Where each line and node of the graph goes, across: the conductor's column is 0. */
export interface Columns {
  /** By task id. */
  branch: Map<string, number>;
  /** By row of the history. */
  step: number[];
  /** Columns in all, the conductor's included. */
  count: number;
}

/**
 * Columns for the branches, the way git lays out its graph rather than one
 * per agent: a branch takes the leftmost column that is free over every row
 * it spans, from its handout up to its last merge (to the top while it is
 * still open). Two lines then never run along the same stretch of column, and
 * a fork or a merge only reaches sideways across the one row it is drawn in.
 *
 * Who did a step is said on its row, so the column need not say it too.
 */
export function columns(steps: readonly Step[], forks: readonly Branch[]): Columns {
  const spans = forks
    .map((fork) => ({ taskId: fork.taskId, bottom: fork.from ?? fork.start, top: fork.end === null ? 0 : Math.min(fork.end, ...fork.into) }))
    // Oldest first, so each column fills upwards.
    .sort((a, b) => b.bottom - a.bottom || a.top - b.top || a.taskId.localeCompare(b.taskId));
  /** Per column past the conductor's, the newest row its last branch reaches. */
  const reached: number[] = [];
  const branch = new Map<string, number>();
  for (const span of spans) {
    // Rows can be shared only where both are the conductor's, one merging in
    // and one forking out, and neither runs along the column there.
    let column = reached.findIndex((top) => top >= span.bottom);
    if (column === -1) column = reached.length;
    reached[column] = span.top;
    branch.set(span.taskId, column + 1);
  }
  // A sub-agent's step with no branch, say on a task nobody claimed, gets a
  // column of its own so it is not read as the conductor's.
  const loose = reached.length + 1;
  const step = steps.map((one) => (one.agent === CONDUCTOR ? 0 : ((one.taskId ? branch.get(one.taskId) : undefined) ?? loose)));
  return { branch, step, count: step.includes(loose) ? loose + 1 : loose };
}
