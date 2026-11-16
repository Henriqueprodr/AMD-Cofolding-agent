// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useMemo } from "react";
import { useCampaignJobs } from "../../hooks/useCampaignJobs";
import { useJobList } from "../../hooks/useJobList";
import { useJobOwners, useRunsKnown, useSessionRuns } from "../../hooks/useSessions";
import { conversationJobs, uniqueAlerts, type JobAlert, type JobSummary } from "../../lib/jobs";
import { scopedJob } from "../../lib/jobScope";
import { useFocusedRun } from "../../lib/workbenchBus";
import { usePanelShown } from "../../hooks/panelShownContext";

/** Shortest of the periods the panels used to poll at on their own. */
const REFRESH_MS = 10_000;

export interface JobScope {
  /** Every optimization job, null until the list and who launched what are known. */
  runs: JobSummary[] | null;
  /** The jobs this conversation launched; empty when the gateway does not say. */
  own: JobSummary[];
  /** The jobs it only referenced. */
  seen: JobSummary[];
  /** The job Summary, Results and Analysis are about. */
  job: JobSummary | null;
  /** Distinct monitor alerts per job id; jobs without any are absent. */
  alerts: ReadonlyMap<string, number>;
  error: string | null;
}

/** The jobs and the one the Jobs tab is about (scopedJob), read again when runs change and while the panel is on screen. */
export function useJobScope(sessionId: string | null): JobScope {
  const shown = usePanelShown();
  const { jobs, alerts: alertList, error } = useJobList(shown ? REFRESH_MS : null);
  const focused = useFocusedRun();
  const sessionRuns = useSessionRuns(sessionId);
  const owners = useJobOwners();
  const known = useRunsKnown();
  const campaignJobs = useCampaignJobs(sessionId);

  const scope = useMemo(() => {
    if (!jobs || known === null) return null;
    // Without per-conversation runs from the gateway, every job counts as the conversation's, as in Summary, Results and Analysis.
    const mine = known ? conversationJobs(jobs, sessionRuns, focused, owners, sessionId, campaignJobs) : { own: [], seen: [] };
    const optimization = (j: JobSummary) => j.type === "optimization";
    return {
      runs: jobs.filter(optimization),
      own: mine.own.filter(optimization),
      seen: mine.seen.filter(optimization),
      job: scopedJob(jobs, known ? mine.own : jobs, focused),
    };
  }, [jobs, known, sessionRuns, focused, owners, sessionId, campaignJobs]);

  const alerts = useMemo(() => alertCounts(alertList), [alertList]);
  return { runs: scope?.runs ?? null, own: scope?.own ?? [], seen: scope?.seen ?? [], job: scope?.job ?? null, alerts, error };
}

/** Distinct alerts per job, as the job's own alert list counts them. */
export function alertCounts(alerts: readonly JobAlert[]): Map<string, number> {
  const byJob = new Map<string, JobAlert[]>();
  for (const alert of alerts) byJob.set(alert.jobId, [...(byJob.get(alert.jobId) ?? []), alert]);
  return new Map([...byJob].map(([id, list]) => [id, uniqueAlerts(list).length]));
}

export type JobGroup = "own" | "seen" | "other";

/** The picker's list: this conversation's jobs, then those it referenced, then the rest, each in the gateway's order. */
export function jobOptions<T extends { id: string }>(runs: readonly T[], own: readonly T[], seen: readonly T[] = []): { job: T; group: JobGroup }[] {
  const mine = new Set(own.map((j) => j.id));
  const referenced = new Set(seen.map((j) => j.id).filter((id) => !mine.has(id)));
  const groupOf = (j: T): JobGroup => (mine.has(j.id) ? "own" : referenced.has(j.id) ? "seen" : "other");
  const order: JobGroup[] = ["own", "seen", "other"];
  return order.flatMap((group) => runs.filter((j) => groupOf(j) === group).map((job) => ({ job, group })));
}
