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

/**
 * What a campaign is actually doing, as against what the ledger last wrote.
 *
 * ``closed`` is the ledger's word. The other three are read off the work,
 * because a conductor that dies with its allocation never closes the books:
 * a campaign whose every task had failed hours earlier still said
 * ``dispatching``, and the panel repeated it back as "ongoing".
 */
export type CampaignStanding = "running" | "stalled" | "stopped" | "closed";

export function campaignStanding(
  detail: Pick<CampaignDetail, "campaign" | "tasks" | "results" | "supervisors">,
): CampaignStanding {
  if (!campaignActive(detail.campaign)) return "closed";
  const { running, pending, agentsLive } = campaignProgress(detail);
  if (running || agentsLive) return "running";
  // Nothing is working. Whether that is temporary turns on there being
  // anything left to pick up: with tasks still queued a supervisor could
  // arrive, without them nobody is coming.
  return pending ? "stalled" : "stopped";
}

/** How the standing reads in the panel, as a phrase that follows the goal. */
export const STANDING_LABEL: Record<CampaignStanding, string> = {
  running: "ongoing",
  stalled: "waiting for a sub-agent",
  stopped: "stopped, with nothing left to run",
  closed: "closed",
};

/** How busy a campaign keeps its team, as the panel reads it out. */
export interface TeamLoad {
  campaignId: string;
  goal: string;
  busy: string;
  queue: string;
  /** Below the backlog target: a sub-agent that finishes now waits on the conductor. */
  short: boolean;
  utilisation: string | null;
  handoff: string | null;
}

const seconds = (s: number) => (s < 90 ? `${Math.round(s)} s` : `${Math.round(s / 60)} min`);

/** One line per open campaign the ledger measured, so idle sub-agents show. */
export function teamLoad(details: readonly Pick<CampaignDetail, "campaign" | "metrics">[]): TeamLoad[] {
  return details.flatMap((detail) => {
    const m = detail.metrics;
    if (!m || !campaignActive(detail.campaign)) return [];
    const median = m.handoff_seconds.median;
    return [{
      campaignId: detail.campaign.campaign_id,
      goal: detail.campaign.goal,
      busy: `${m.running} of ${m.concurrency} sub-agents working`,
      queue: `${m.queued} queued, ${m.backlog_target} to keep ready`,
      short: m.queued < m.backlog_target,
      utilisation: m.utilisation === null ? null : `${Math.round(m.utilisation * 100)}% busy since the first task`,
      handoff: median === null ? null : `${seconds(median)} median wait between tasks`,
    }];
  });
}

/** Whether a supervisor's last report is recent enough to believe its state. */
export function isLive(supervisor: Pick<FleetSupervisor, "last_seen_seconds_ago" | "state">): boolean {
  const age = supervisor.last_seen_seconds_ago;
  if (age !== null && age > STALE_AFTER_S) return false;
  return (supervisor.state || "").toLowerCase() !== "gone";
}

// ---------------------------------------------------------------------------
// who did what: C for the conductor, S1, S2… for the sub-agents
// ---------------------------------------------------------------------------

/** The label of the agent you are talking to, on everything it ran itself. */
export const CONDUCTOR = "C";
/** On a task nobody has claimed yet: a sub-agent's, but not yet anyone's. */
export const UNCLAIMED = "S";

/** The sub-agent that ran a task, or holds it now. */
export function ranBy(task: Pick<FleetTask, "lease_owner" | "ran_by">): string | null {
  return task.ran_by || task.lease_owner || null;
}

/**
 * "S1", "S2"… for each sub-agent of the conversation, in the order they
 * first turned up.
 *
 * Over every campaign at once, so a sub-agent is the same S-number in the
 * plot of all the work and in the list of one campaign's agents.
 */
export function agentLabels(campaigns: readonly (Pick<CampaignDetail, "tasks"> & Partial<Pick<CampaignDetail, "supervisors">>)[]): Map<string, string> {
  const first = new Map<string, number>();
  const seen = (owner: string, iso: string | null | undefined) => {
    const parsed = Date.parse(iso || "");
    const at = Number.isNaN(parsed) ? Infinity : parsed;
    const prior = first.get(owner);
    if (prior === undefined || at < prior) first.set(owner, at);
  };
  for (const detail of campaigns) {
    for (const task of detail.tasks) {
      const owner = ranBy(task);
      if (owner) seen(owner, task.claimed_at || task.created_at);
    }
    for (const supervisor of detail.supervisors ?? []) seen(supervisor.owner, supervisor.first_seen_at);
  }
  const order = [...first].sort((a, b) => a[1] - b[1] || a[0].localeCompare(b[0]));
  return new Map(order.map(([owner], i) => [owner, `S${i + 1}`]));
}

/** The label a task's work goes under. */
export function taskAgent(task: Pick<FleetTask, "lease_owner" | "ran_by">, labels: ReadonlyMap<string, string>): string {
  const owner = ranBy(task);
  return (owner && labels.get(owner)) || UNCLAIMED;
}

/** Whether the conversation has a sub-agent under that label, so a stale link does not open an empty page. */
export function knownAgent(agent: string | null, labels: ReadonlyMap<string, string>): boolean {
  return agent !== null && [...labels.values()].includes(agent);
}

/** What a sub-agent's page shows. */
export interface AgentPage {
  held: { task: FleetTask; detail: CampaignDetail }[];
  /** A task picked from the trajectory, rather than what it holds now. */
  stored: boolean;
  /** The last task it finished, for a page with nothing in hand. */
  last: { task: FleetTask; detail: CampaignDetail } | null;
}

/**
 * The task a sub-agent's page is about: the one picked from the trajectory,
 * else the one it holds now.
 *
 * Never everything it ever did: each task starts it on a fresh
 * conversation, and once it has reported, that conversation is stored
 * under the task's bubble rather than carried into the next.
 */
export function agentPage(details: readonly CampaignDetail[], labels: ReadonlyMap<string, string>, agent: string, picked: string | null): AgentPage {
  const mine = details.flatMap((detail) => detail.tasks.filter((task) => taskAgent(task, labels) === agent).map((task) => ({ task, detail })));
  const chosen = picked ? mine.find(({ task }) => task.task_id === picked) : undefined;
  if (chosen) return { held: [chosen], stored: true, last: null };
  const at = ({ task }: { task: FleetTask }) => Date.parse(task.finished_at || task.claimed_at || task.created_at) || 0;
  const over = mine.filter(({ task }) => taskPhase(task) === "done" || taskPhase(task) === "failed").sort((a, b) => at(b) - at(a));
  return { held: mine.filter(({ task }) => taskPhase(task) === "running"), stored: false, last: over[0] ?? null };
}

/** The runs those tasks started, by id. */
export function taskJobs(details: readonly Pick<CampaignDetail, "tasks" | "results">[], taskIds: ReadonlySet<string>): Set<string> {
  const out = new Set<string>();
  for (const detail of details) {
    for (const task of detail.tasks) if (task.job_id && taskIds.has(task.task_id)) out.add(task.job_id);
    for (const result of detail.results) if (taskIds.has(result.task_id)) for (const id of result.job_ids ?? []) out.add(id);
  }
  return out;
}

/** A sub-agent with a task in hand. */
export interface WorkingAgent {
  agent: string;
  title: string;
  /** When it took the task, in ms; null if the ledger did not say. */
  since: number | null;
  taskId: string;
  jobId: string | null;
}

/** The first *count* words of *text*, for a card too small for a sentence. */
export function fewWords(text: string, count = 5): string {
  const words = text.trim().split(/\s+/).filter(Boolean);
  return words.length > count ? `${words.slice(0, count).join(" ")}…` : words.join(" ");
}

/** The sub-agents working right now, S1 first: one per task they hold. */
export function workingAgents(
  campaigns: readonly Pick<CampaignDetail, "campaign" | "tasks">[],
  labels: ReadonlyMap<string, string>,
): WorkingAgent[] {
  const out: WorkingAgent[] = [];
  for (const detail of campaigns) {
    // A closed campaign is still draining what its sub-agents hold; only
    // finalizing ends their work.
    if (detail.campaign.finalized_at) continue;
    for (const task of detail.tasks) {
      if (taskPhase(task) !== "running") continue;
      const owner = ranBy(task);
      const agent = owner ? labels.get(owner) : undefined;
      if (!agent) continue;
      const since = Date.parse(task.claimed_at || "");
      out.push({
        agent,
        title: (task.spec?.title || task.spec?.intent || task.task_id).trim(),
        since: Number.isNaN(since) ? null : since,
        taskId: task.task_id,
        jobId: task.job_id,
      });
    }
  }
  return out.sort((a, b) => Number(a.agent.slice(1)) - Number(b.agent.slice(1)));
}

/** Each run a campaign started, with the label of the sub-agent that started it and the task's title. */
export function campaignRuns(
  campaigns: readonly Pick<CampaignDetail, "tasks" | "results">[],
  labels: ReadonlyMap<string, string>,
): Map<string, { agent: string; title: string }> {
  const out = new Map<string, { agent: string; title: string }>();
  for (const detail of campaigns) {
    const tasks = new Map(detail.tasks.map((task) => [task.task_id, task]));
    const about = (task: FleetTask | undefined, fallback: string) => ({
      agent: task ? taskAgent(task, labels) : UNCLAIMED,
      title: (task?.spec?.title || task?.spec?.intent || fallback).trim(),
    });
    for (const task of detail.tasks) if (task.job_id) out.set(task.job_id, about(task, task.task_id));
    // A task that started more than one run reports them all; only the one
    // it was harvested from is on the task row.
    for (const result of detail.results) {
      for (const id of result.job_ids ?? []) if (!out.has(id)) out.set(id, about(tasks.get(result.task_id), result.task_id));
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// the bubble field: every piece of work the conversation set going
// ---------------------------------------------------------------------------

/** Who started a piece of work. */
export type Launcher = "conductor" | "subagent";

/** A run the conversation started itself, as the bubble field reads one. */
export interface OwnRun {
  id: string;
  label: string;
  /** The job manager's word: completed, running, failed, queued… */
  state: string;
  /** When it started, in epoch ms. */
  at: number;
}

/** One piece of work. *x* is a fraction of the span, 0 at the left. */
export interface Bubble {
  /** The task id, the run id for a run with no task behind it, or campaign:<id> for the campaign itself. */
  id: string;
  launcher: Launcher;
  /** Who did it: C, or S1, S2… for the sub-agent that ran it; S while nobody has claimed it. */
  agent: string;
  phase: TaskPhase;
  /**
   * design, research, retrosynthesis…; "run" for a run the conductor started
   * itself, "campaign" for the conductor setting a campaign going.
   */
  kind: string;
  /** What was asked for, which is what the bubble is labelled with. */
  label: string;
  /** When it started: claimed if it has been, else written. */
  at: number;
  /** The REINVENT run behind it, when there is one. Clicking a bubble opens this. */
  jobId: string | null;
  /** Where to find it in the ledger; null for the conversation's own runs. */
  campaignId: string | null;
  taskId: string | null;
  x: number;
  /** Which row it sits in. Packing only, so nothing overlaps; rows mean nothing. */
  row: number;
}

export interface BubbleEdge {
  from: string;
  to: string;
}

export interface BubbleField {
  bubbles: Bubble[];
  edges: BubbleEdge[];
  rows: number;
  /** Start and end of the time axis, in epoch ms. */
  span: [number, number];
}

function timeOf(task: FleetTask): number {
  const at = Date.parse(task.claimed_at || task.created_at || "");
  return Number.isNaN(at) ? 0 : at;
}

/** The id of the bubble a campaign starts from. */
export const campaignBubbleId = (campaignId: string) => `campaign:${campaignId}`;

/** Going while it is open and has work still to finish, or none written yet. */
function campaignPhase(detail: Pick<CampaignDetail, "campaign" | "tasks">): TaskPhase {
  if (!campaignActive(detail.campaign)) return "done";
  const phases = detail.tasks.map(taskPhase);
  return !phases.length || phases.some((phase) => phase === "running" || phase === "pending") ? "running" : "done";
}

/** A job manager state as one of the four phases the field draws. */
function runPhase(state: string): TaskPhase {
  const word = (state || "").toLowerCase();
  // A stopped run finished early, usually at a checkpoint the chemist asked
  // for. Drawn as finished, because its molecules are real.
  if (word === "completed" || word === "stopped") return "done";
  if (FAILED.has(word)) return "failed";
  if (word === "running") return "running";
  return "pending";
}

/**
 * Everything the conversation set going, read left to right in the order it started.
 *
 * One bubble per piece of work, whoever started it: the runs the conductor
 * launched in the conversation itself and the tasks it handed to sub-agents,
 * which is the only place the two are seen together. Each campaign starts
 * from a bubble of the conductor's own, and lines are who asked for what on
 * the strength of what: a task written from what two earlier ones showed
 * has two coming into it, and one that builds on nothing hangs off the
 * conductor that wrote it.
 *
 * Deliberately not one row per sub-agent. That would say the agents are
 * independent and their work unrelated, which is false; rows here only keep
 * bubbles from landing on top of each other.
 *
 * *now* places work that has not finished; pass it so the layout is a
 * function of its arguments and the tests do not chase the clock.
 *
 * *gap* is how much room a bubble needs to its right before another may share
 * its row, as a fraction of the width. The caller knows it, because it is
 * how wide a label is in the space the panel has.
 */
export function jobBubbles(
  campaigns: readonly Pick<CampaignDetail, "campaign" | "tasks" | "results">[],
  runs: readonly OwnRun[],
  now: number,
  gap = 0.085,
): BubbleField {
  const made: Bubble[] = [];
  const labels = agentLabels(campaigns);

  for (const detail of campaigns) {
    const { campaign } = detail;
    const started = Date.parse(campaign.created_at || "");
    made.push({
      id: campaignBubbleId(campaign.campaign_id),
      launcher: "conductor",
      agent: CONDUCTOR,
      phase: campaignPhase(detail),
      kind: "campaign",
      label: (campaign.goal || "").trim() || "Campaign",
      at: Number.isNaN(started) ? 0 : started,
      jobId: null,
      campaignId: campaign.campaign_id,
      taskId: null,
      x: 0,
      row: 0,
    });
    for (const task of detail.tasks) {
      made.push({
        id: task.task_id,
        launcher: "subagent",
        agent: taskAgent(task, labels),
        phase: taskPhase(task),
        kind: task.spec?.kind || "design",
        label: (task.spec?.title || task.spec?.intent || task.spec?.rationale || task.task_id).trim(),
        at: timeOf(task),
        jobId: task.job_id,
        campaignId: campaign.campaign_id,
        taskId: task.task_id,
        x: 0,
        row: 0,
      });
    }
  }

  // A run a sub-agent started is already here as its task; the same work
  // twice would read as two experiments.
  const claimed = new Set(made.map((bubble) => bubble.jobId).filter(Boolean));
  for (const run of runs) {
    if (claimed.has(run.id)) continue;
    made.push({
      id: run.id,
      launcher: "conductor",
      agent: CONDUCTOR,
      phase: runPhase(run.state),
      kind: "run",
      label: run.label.trim() || run.id,
      at: run.at,
      jobId: run.id,
      campaignId: null,
      taskId: null,
      x: 0,
      row: 0,
    });
  }

  if (!made.length) return { bubbles: [], edges: [], rows: 0, span: [now, now] };

  // A campaign ahead of tasks written the same second, so the line from it
  // runs forward like every other.
  const lead = (bubble: Bubble) => (bubble.kind === "campaign" ? 0 : 1);
  const ordered = made.sort((a, b) => a.at - b.at || lead(a) - lead(b) || a.id.localeCompare(b.id));
  const times = ordered.map((bubble) => bubble.at).filter((at) => at > 0);
  const first = times.length ? Math.min(...times) : now;
  // A conversation one run old has no span to divide by; give it a nominal
  // one so the single bubble lands at the left rather than dividing by zero.
  const last = Math.max(now, times.length ? Math.max(...times) : now, first + 60_000);

  /** Right edge, in x, of the last bubble placed in each row. */
  const occupied: number[] = [];
  for (const [index, bubble] of ordered.entries()) {
    bubble.at = bubble.at || first;
    // In sequence and evenly spaced: an hour-long run would otherwise push
    // everything set going in the minute before it into one heap.
    bubble.x = ordered.length > 1 ? index / (ordered.length - 1) : 0;
    let row = occupied.findIndex((right) => bubble.x >= right);
    if (row === -1) row = occupied.length;
    occupied[row] = bubble.x + gap;
    bubble.row = row;
  }

  const known = new Set(ordered.map((bubble) => bubble.id));
  const edges: BubbleEdge[] = [];
  for (const detail of campaigns) {
    for (const task of detail.tasks) {
      // A parent outside what is drawn would be an edge from nothing.
      const parents = (task.spec?.derived_from ?? []).filter((parent) => known.has(parent) && parent !== task.task_id);
      for (const parent of parents.length ? parents : [campaignBubbleId(detail.campaign.campaign_id)]) {
        edges.push({ from: parent, to: task.task_id });
      }
    }
  }

  return { bubbles: ordered, edges, rows: Math.max(1, occupied.length), span: [first, last] };
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
  /** The run that scored it, where its complex is; the result's own for one posted before molecules named theirs. */
  jobId: string | null;
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
        jobId: (typeof candidate.job_id === "string" && candidate.job_id) || result.job_id || null,
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
  const taskOwner = taskOwners(detail.tasks);

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

/** What one sub-agent sent back, as the campaign's list of reports shows it. */
export interface TaskReport {
  taskId: string;
  title: string;
  kind: string;
  /** The opening of the report, or the summary when the sub-agent wrote none. */
  excerpt: string;
  /** Whether there is more to read than the excerpt. */
  full: boolean;
  molecules: number;
  owner: string | null;
  postedAt: string;
}

/**
 * The reports, newest first.
 *
 * The campaign's output when its tasks were not design runs: research, a
 * retrosynthesis or a question produces no molecules, and listing only what
 * reached the pool would show an empty campaign that in fact answered
 * everything it was asked.
 */
export function taskReports(detail: Pick<CampaignDetail, "tasks" | "results">): TaskReport[] {
  const specs = new Map(detail.tasks.map((task) => [task.task_id, task]));
  return detail.results
    .map((result) => {
      const task = specs.get(result.task_id);
      const report = (result.report ?? "").trim();
      const excerpt = opening(report) || (result.summary ?? "").trim();
      return {
        taskId: result.task_id,
        title: (task?.spec?.title || task?.spec?.intent || result.task_id).trim(),
        kind: task?.spec?.kind || "design",
        excerpt,
        full: report.length > excerpt.length,
        molecules: result.candidates?.length ?? 0,
        owner: task ? ranBy(task) : null,
        postedAt: result.posted_at,
      };
    })
    .sort((a, b) => (Date.parse(b.postedAt) || 0) - (Date.parse(a.postedAt) || 0));
}

/**
 * A report as the drill-down shows it, without the heading the supervisor
 * puts at the top: the card it is shown in is already titled "Report", and
 * the two together read as a heading repeated.
 */
export function reportBody(report: string): string {
  return report.replace(/^\s*#{1,6}\s*report\s*\n+/i, "").trim();
}

/** The first paragraph of a report, with its Markdown heading skipped. */
function opening(report: string): string {
  for (const block of report.split(/\n\s*\n/)) {
    const text = block
      .split("\n")
      .filter((line) => !/^\s*#{1,6}\s/.test(line))
      .join(" ")
      .trim();
    if (text) return text;
  }
  return "";
}

/** The sub-agent that ran each task, for the pooled list and the drill-down. */
export function taskOwners(tasks: readonly FleetTask[]): Map<string, string> {
  const owners = new Map<string, string>();
  for (const task of tasks) {
    const owner = ranBy(task);
    if (owner) owners.set(task.task_id, owner);
  }
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
