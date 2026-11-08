// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * What the Campaign tab shows, worked out from the ledger's rows.
 *
 * All of it is pure: the panels fetch and draw, the arithmetic is here where
 * it can be tested without a DOM. "Campaign" throughout this file means a
 * piece of work the conductor split into tasks, not the log one agent keeps
 * of its own goal (lib/campaign.ts).
 */

import type { CampaignDetail, FleetCampaign, FleetCandidate, FleetResult, FleetSupervisor, FleetTask } from "../types/api";

/** States a task rests in, as the ledger writes them. */
const DONE = new Set(["done", "completed", "succeeded"]);
const FAILED = new Set(["failed", "cancelled", "canceled", "abandoned"]);
const RUNNING = new Set(["claimed", "running", "leased"]);

export type TaskPhase = "pending" | "running" | "done" | "failed";

export function taskPhase(task: Pick<FleetTask, "state">): TaskPhase {
  const state = (task.state || "").toLowerCase();
  if (DONE.has(state)) return "done";
  if (FAILED.has(state)) return "failed";
  if (RUNNING.has(state)) return "running";
  return "pending";
}

/** Whether a campaign can still change, so the panel keeps polling. */
export function campaignActive(campaign: Pick<FleetCampaign, "state" | "finalized_at">): boolean {
  const state = (campaign.state || "").toLowerCase();
  return !campaign.finalized_at && state !== "closed" && state !== "finalized" && state !== "cancelled";
}

/** How far along a campaign is, in the terms the overview reads out. */
export interface CampaignProgress {
  tasks: number;
  done: number;
  running: number;
  failed: number;
  pending: number;
  /** Molecules put forward across every finished task. */
  molecules: number;
  /** Sub-agents that reported recently enough to be believed. */
  agentsLive: number;
  /** What the conductor wants working at once. */
  concurrency: number;
  /** done + failed over tasks; null before there is a task to divide by. */
  fraction: number | null;
}

/** A supervisor seen longer ago than this is not counted as alive. */
const STALE_AFTER_S = 120;

export function campaignProgress(detail: Pick<CampaignDetail, "campaign" | "tasks" | "results" | "supervisors">): CampaignProgress {
  const phases = detail.tasks.map(taskPhase);
  const count = (phase: TaskPhase) => phases.filter((p) => p === phase).length;
  const done = count("done");
  const failed = count("failed");
  const settled = done + failed;
  return {
    tasks: detail.tasks.length,
    done,
    failed,
    running: count("running"),
    pending: count("pending"),
    molecules: detail.results.reduce((total, result) => total + (result.candidates?.length ?? 0), 0),
    agentsLive: detail.supervisors.filter(isLive).length,
    concurrency: detail.campaign.concurrency,
    fraction: detail.tasks.length ? settled / detail.tasks.length : null,
  };
}

/** Whether a supervisor's last report is recent enough to believe its state. */
export function isLive(supervisor: Pick<FleetSupervisor, "last_seen_seconds_ago" | "state">): boolean {
  const age = supervisor.last_seen_seconds_ago;
  if (age !== null && age > STALE_AFTER_S) return false;
  return (supervisor.state || "").toLowerCase() !== "gone";
}

// ---------------------------------------------------------------------------
// the trajectory graph
// ---------------------------------------------------------------------------

/** A task placed in the graph. *x* is a fraction of the span, 0 at the left. */
export interface GraphNode {
  task: FleetTask;
  phase: TaskPhase;
  /** When it started: claimed if it has been, else proposed. */
  at: number;
  /** 0 at the first task, 1 at the newest; the horizontal axis is time. */
  x: number;
  /** Which row it sits in. Packing only, so nothing overlaps; rows mean nothing. */
  row: number;
  /** What the conductor asked for, which is what the node is labelled with. */
  label: string;
}

export interface GraphEdge {
  from: string;
  to: string;
}

export interface Graph {
  nodes: GraphNode[];
  edges: GraphEdge[];
  rows: number;
  /** Start and end of the time axis, in epoch ms. */
  span: [number, number];
}

function timeOf(task: FleetTask): number {
  const at = Date.parse(task.claimed_at || task.created_at || "");
  return Number.isNaN(at) ? 0 : at;
}

/**
 * The campaign as a graph read left to right over time.
 *
 * Deliberately not one row per sub-agent. That would say the agents are
 * independent and their tasks unrelated, which is false: the conductor
 * writes a task from what two earlier ones showed, or one, or none. Rows
 * here only keep nodes from landing on top of each other, and the lineage
 * the conductor recorded is in the edges.
 *
 * *now* places tasks that have not finished; pass it so the layout is a
 * function of its arguments and the tests do not chase the clock.
 *
 * *gap* is how much room a node needs to its right before another may share
 * its row, as a fraction of the width. The caller knows it, because it is
 * how wide a label is in the space the panel has; at the default, nodes only
 * avoid overlapping each other and their labels can still collide.
 */
export function campaignGraph(tasks: readonly FleetTask[], now: number, gap = 0.085): Graph {
  if (!tasks.length) return { nodes: [], edges: [], rows: 0, span: [now, now] };

  const ordered = [...tasks].sort((a, b) => timeOf(a) - timeOf(b) || a.task_id.localeCompare(b.task_id));
  const times = ordered.map(timeOf).filter((at) => at > 0);
  const first = times.length ? Math.min(...times) : now;
  // A campaign one task old has no span to divide by; give it a nominal one
  // so the single node lands at the left rather than dividing by zero.
  const last = Math.max(now, times.length ? Math.max(...times) : now, first + 60_000);

  const known = new Set(ordered.map((task) => task.task_id));
  const nodes: GraphNode[] = [];
  /** Right edge, in x, of the last node placed in each row. */
  const occupied: number[] = [];

  for (const task of ordered) {
    const at = timeOf(task) || first;
    const x = (at - first) / (last - first);
    let row = occupied.findIndex((right) => x >= right);
    if (row === -1) row = occupied.length;
    occupied[row] = x + gap;
    nodes.push({
      task,
      phase: taskPhase(task),
      at,
      x,
      row,
      label: (task.spec?.intent || task.spec?.rationale || task.task_id).trim(),
    });
  }

  const edges: GraphEdge[] = [];
  for (const task of ordered) {
    for (const parent of task.spec?.derived_from ?? []) {
      // A parent outside this campaign would draw an edge from nothing.
      if (known.has(parent) && parent !== task.task_id) edges.push({ from: parent, to: task.task_id });
    }
  }

  return { nodes, edges, rows: Math.max(1, occupied.length), span: [first, last] };
}

// ---------------------------------------------------------------------------
// results, three ways
// ---------------------------------------------------------------------------

/** A molecule with the task that found it, for the pooled list. */
export interface PooledCandidate {
  smiles: string;
  score: number | null;
  taskId: string;
  /** The sub-agent that ran the task, when the ledger still knows it. */
  owner: string | null;
  candidate: FleetCandidate;
}

function scoreOf(candidate: FleetCandidate): number | null {
  const value = candidate.score ?? (candidate as { total_score?: unknown }).total_score;
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/**
 * Every molecule the campaign produced, best first, deduplicated by SMILES.
 *
 * Pooled rather than grouped because this is what the chemist came for: the
 * output of the campaign, not of any one agent. Which agent found a molecule
 * rides along for the ones who want it.
 */
export function pooledCandidates(
  results: readonly FleetResult[],
  owners: ReadonlyMap<string, string> = new Map(),
  limit = 200,
): PooledCandidate[] {
  const best = new Map<string, PooledCandidate>();
  for (const result of results) {
    for (const candidate of result.candidates ?? []) {
      const smiles = typeof candidate?.smiles === "string" ? candidate.smiles.trim() : "";
      if (!smiles) continue;
      const entry: PooledCandidate = {
        smiles,
        score: scoreOf(candidate),
        taskId: result.task_id,
        owner: owners.get(result.task_id) ?? null,
        candidate,
      };
      const seen = best.get(smiles);
      // The same molecule found twice keeps the better score, and with it the
      // task that scored it that way.
      if (!seen || (entry.score ?? -Infinity) > (seen.score ?? -Infinity)) best.set(smiles, entry);
    }
  }
  return [...best.values()]
    .sort((a, b) => (b.score ?? -Infinity) - (a.score ?? -Infinity) || a.smiles.localeCompare(b.smiles))
    .slice(0, limit);
}

/** What one sub-agent did across the campaign. */
export interface AgentSummary {
  owner: string;
  live: boolean;
  state: string;
  /** Tasks it finished, failed, and holds now. */
  completed: number;
  failed: number;
  current: string | null;
  /** Molecules from the tasks it ran. */
  molecules: number;
  /** The best score it reached, across its tasks. */
  best: number | null;
  detail: string | null;
}

/**
 * One row per sub-agent: what each has been doing.
 *
 * Built from the supervisor rows and not from the tasks, so an agent that
 * has claimed its first task and reported nothing yet still appears. An
 * agent that has gone is kept when it finished work, because its results are
 * still in the pool above and the row is what explains where they came from.
 */
export function agentSummaries(detail: Pick<CampaignDetail, "tasks" | "results" | "supervisors">): AgentSummary[] {
  const taskOwner = new Map<string, string>();
  for (const task of detail.tasks) if (task.lease_owner) taskOwner.set(task.task_id, task.lease_owner);

  const molecules = new Map<string, number>();
  const best = new Map<string, number>();
  for (const result of detail.results) {
    const owner = taskOwner.get(result.task_id);
    if (!owner) continue;
    molecules.set(owner, (molecules.get(owner) ?? 0) + (result.candidates?.length ?? 0));
    for (const candidate of result.candidates ?? []) {
      const score = scoreOf(candidate);
      if (score !== null && score > (best.get(owner) ?? -Infinity)) best.set(owner, score);
    }
  }

  return detail.supervisors
    .map((supervisor) => ({
      owner: supervisor.owner,
      live: isLive(supervisor),
      state: supervisor.state,
      completed: supervisor.tasks_completed,
      failed: supervisor.tasks_failed,
      current: supervisor.task_id,
      molecules: molecules.get(supervisor.owner) ?? 0,
      best: best.get(supervisor.owner) ?? null,
      detail: supervisor.detail,
    }))
    .filter((agent) => agent.live || agent.completed > 0 || agent.failed > 0)
    .sort((a, b) => Number(b.live) - Number(a.live) || b.completed - a.completed || a.owner.localeCompare(b.owner));
}

/** How the campaign is doing against one objective. */
export interface ObjectiveProgress {
  name: string;
  /** Best task mean for this objective, and the task that reached it. */
  best: number;
  bestTask: string;
  /** Mean across the tasks that reported it. */
  mean: number;
  /** Tasks that scored it; fewer than the campaign's tasks when they vary. */
  tasks: number;
  /** Best so far minus best before the newest task that moved it; null when nothing has. */
  delta: number | null;
}

/**
 * Progress per objective, from the task profiles.
 *
 * The objectives come from the results and not from the campaign, because a
 * campaign whose tasks write their own TTPs declares none: what is being
 * optimised is only knowable from what came back. *declared* seeds the list
 * so a campaign that does declare objectives shows them before any result.
 */
export function objectiveProgress(results: readonly FleetResult[], declared: readonly string[] = []): ObjectiveProgress[] {
  const byName = new Map<string, { task: string; value: number }[]>();
  for (const name of declared) byName.set(name, []);
  // Oldest first, so "best before the newest improvement" means what it says.
  const ordered = [...results].sort((a, b) => (Date.parse(a.posted_at) || 0) - (Date.parse(b.posted_at) || 0));
  for (const result of ordered) {
    for (const [name, value] of Object.entries(result.objective_profile ?? {})) {
      if (typeof value !== "number" || !Number.isFinite(value)) continue;
      const list = byName.get(name) ?? [];
      list.push({ task: result.task_id, value });
      byName.set(name, list);
    }
  }

  const out: ObjectiveProgress[] = [];
  for (const [name, values] of byName) {
    if (!values.length) continue;
    let best = -Infinity;
    let bestTask = "";
    let previous: number | null = null;
    for (const { task, value } of values) {
      if (value > best) {
        previous = best === -Infinity ? null : best;
        best = value;
        bestTask = task;
      }
    }
    out.push({
      name,
      best,
      bestTask,
      mean: values.reduce((total, v) => total + v.value, 0) / values.length,
      tasks: values.length,
      delta: previous === null ? null : best - previous,
    });
  }
  return out.sort((a, b) => b.tasks - a.tasks || a.name.localeCompare(b.name));
}

/** The sub-agent that ran each task, for the pooled list and the drill-down. */
export function taskOwners(tasks: readonly FleetTask[]): Map<string, string> {
  const owners = new Map<string, string>();
  for (const task of tasks) if (task.lease_owner) owners.set(task.task_id, task.lease_owner);
  return owners;
}

/**
 * The TTP id a task's run was scored against.
 *
 * Mirrors the supervisor: a campaign with a base TTP gives each task a
 * variant of it, and one without has the task write its own under a fixed
 * id. Kept in step with reinvent4_agent/supervisor.py; the drill-down shows
 * the full profile and has to ask for the right one.
 */
export function ttpIdOf(campaign: Pick<FleetCampaign, "base_ttp_id">, taskId: string): string {
  return campaign.base_ttp_id ? `${campaign.base_ttp_id}-${taskId}` : `ttp-${taskId}`;
}
