// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type {
  AtlasBranch,
  CampaignAtlas,
  CampaignDetail,
  Finding,
  FleetEvent,
} from "../types/api";

/** Ledger events that change what the campaign knows. */
const LEARNING = new Set(["finding_recorded", "finding_noted"]);

/**
 * How much the campaign has learnt so far: the atlas is read again when this
 * moves. The campaign and its history are polled already, so the atlas costs
 * a read when something was recorded and none when nothing was.
 *
 * The findings and notes in the history, and the tasks that have reported:
 * the history the panel holds is its first page, which a busy campaign
 * outgrows, while a sub-agent records what it settled before it reports.
 */
export function learnt(
  events: readonly FleetEvent[] | undefined,
  detail: {
    results: readonly unknown[];
    tasks: readonly Pick<CampaignDetail["tasks"][number], "finished_at">[];
  },
): number {
  let count = detail.results.length;
  for (const task of detail.tasks) if (task.finished_at) count += 1;
  for (const event of events ?? []) if (LEARNING.has(event.kind)) count += 1;
  return count;
}

export interface BranchView extends AtlasBranch {
  /** Its current findings, newest first. */
  items: Finding[];
}

/**
 * The branches as the view lists them: those with something disputed first,
 * since they are what somebody has to settle, then the rest that hold
 * findings, newest news first, and last the ones nothing is known about.
 * Within a group the plan's order is kept, so a branch does not jump about
 * between two reads.
 */
export function branchViews(
  atlas: Pick<CampaignAtlas, "branches" | "items">,
): BranchView[] {
  const byBranch = new Map<string, Finding[]>();
  for (const finding of atlas.items)
    byBranch.set(finding.branch, [
      ...(byBranch.get(finding.branch) ?? []),
      finding,
    ]);
  const views = atlas.branches.map((branch) => ({
    ...branch,
    items: byBranch.get(branch.branch) ?? [],
  }));
  const rank = (view: BranchView) =>
    view.disputed > 0 ? 0 : view.findings > 0 ? 1 : 2;
  return views
    .map((view, i) => ({ view, i }))
    .sort((a, b) => rank(a.view) - rank(b.view) || a.i - b.i)
    .map(({ view }) => view);
}

/** How a finding's notes lean: supports, disputes and extends counted. */
export function stances(finding: Pick<Finding, "notes">): {
  supports: number;
  disputes: number;
  extends: number;
} {
  const counts = { supports: 0, disputes: 0, extends: 0 };
  for (const note of finding.notes) if (note.stance) counts[note.stance] += 1;
  return counts;
}

/** A job id as the gateway serves jobs, bare or written "job_…" / "job …". */
const JOB = /^(?:job[_\s:]*)?([0-9a-f]{6,32})$/i;

/** The run a piece of evidence names, so it can open; null for a DOI, a ChEMBL id, a PDB code. */
export function evidenceRun(
  item: string,
  known: ReadonlySet<string>,
): string | null {
  const id = JOB.exec(item.trim())?.[1]?.toLowerCase();
  // A run of hex digits could be anything; only a run the gateway has is a run.
  return id && known.has(id) ? id : null;
}
