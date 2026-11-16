// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { CONDUCTOR, UNCLAIMED } from "./fleet";
import type { JobSummary } from "./jobs";

/** Where a run is in its life: on the GPUs, waiting for them, or done. */
export type RunGroup = "running" | "waiting" | "finished";

export const RUN_GROUPS: readonly RunGroup[] = ["running", "waiting", "finished"];

/** Who started a run that no conversation on screen accounts for. */
export const ELSEWHERE = "Other chat";

export interface JobRow {
  job: JobSummary;
  group: RunGroup;
  /** CONDUCTOR, a sub-agent's S-label, UNCLAIMED, or ELSEWHERE. */
  by: string;
  /** The campaign task it ran for, if a sub-agent's. */
  task: string | null;
  /** From creation until Slurm (or the runner) started it; so far, while it waits. */
  waitedMs: number | null;
  /** From its start until it ended; so far, while it runs. */
  ranMs: number | null;
}

const at = (iso: string | null) => {
  const ms = Date.parse(iso ?? "");
  return Number.isNaN(ms) ? null : ms;
};

const span = (from: number | null, to: number | null) => (from !== null && to !== null && to >= from ? to - from : null);

export function runGroup(job: Pick<JobSummary, "state">): RunGroup {
  return job.state === "running" ? "running" : job.state === "queued" ? "waiting" : "finished";
}

/**
 * One row per run, in *jobs*' order (newest first). A run a campaign task
 * names is that task's sub-agent's, else one of *own* is the conductor's;
 * the rest were started in another conversation.
 */
export function jobRows(
  jobs: readonly JobSummary[],
  own: ReadonlySet<string>,
  delegated: ReadonlyMap<string, { agent: string; title: string }>,
  now: number,
): JobRow[] {
  return jobs.map((job) => {
    const task = delegated.get(job.id);
    const group = runGroup(job);
    const created = at(job.createdAt);
    const started = at(job.startedAt);
    return {
      job,
      group,
      by: task?.agent ?? (own.has(job.id) ? CONDUCTOR : ELSEWHERE),
      task: task?.title || null,
      waitedMs: span(created, started ?? (group === "waiting" ? now : null)),
      ranMs: span(started, at(job.finishedAt) ?? (group === "running" ? now : null)),
    };
  });
}

/** Who started the runs in *rows*, for the filter: the conductor, sub-agents in order, the rest last. */
export function launchers(rows: readonly Pick<JobRow, "by">[]): string[] {
  const rank = (label: string) => (label === CONDUCTOR ? 0 : /^S\d+$/.test(label) ? Number(label.slice(1)) : label === UNCLAIMED ? 1e6 : 1e7);
  return [...new Set(rows.map((row) => row.by))].sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
}
