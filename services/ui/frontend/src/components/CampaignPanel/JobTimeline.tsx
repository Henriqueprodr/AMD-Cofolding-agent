// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useMemo } from "react";
import { useCampaignDetails } from "../../hooks/useCampaignJobs";
import { jobTitle, type JobSummary } from "../../lib/jobs";
import { CONDUCTOR, UNCLAIMED, type Bubble, type OwnRun } from "../../lib/fleet";
import { focusAgent, focusRun, openStoredTask, openTask, useFocusedRun, useFocusedTask } from "../../lib/workbenchBus";
import { useJobScope } from "../JobPicker/useJobScope";
import { JobBubbles } from "./JobBubbles";
import { useNow } from "./shared";

interface Props {
  sessionId: string | null;
  /** Show the run a bubble names, which is a view of this tab. */
  onOpenRun: () => void;
}

/**
 * The Campaign tab's opening plot: everything this conversation set going.
 *
 * Reads both halves of the work — the ledger for what the sub-agents were
 * given, the job manager for the runs the agent started here — because the
 * point of the plot is seeing them together.
 */
export function JobTimeline({ sessionId, onOpenRun }: Props) {
  const now = useNow();
  const campaigns = useCampaignDetails(sessionId);
  const { own } = useJobScope(sessionId);
  const focusedRun = useFocusedRun();
  const focusedTask = useFocusedTask();

  const runs = useMemo((): OwnRun[] => own.map(asRun), [own]);

  return (
    <div className="border-subtle shrink-0 overflow-x-auto border-b px-2 py-1">
      <JobBubbles
        campaigns={campaigns}
        runs={runs}
        now={now}
        selected={{ jobId: focusedRun, campaignId: focusedTask?.campaignId ?? null, taskId: focusedTask?.taskId ?? null }}
        onSelect={(bubble) => open(bubble, onOpenRun)}
      />
    </div>
  );
}

/**
 * A sub-agent's task as it saw it; else the run behind a bubble, under
 * whoever started it; else the task itself, else the campaign.
 */
function open(bubble: Bubble, onOpenRun: () => void): void {
  if (bubble.campaignId && bubble.taskId && bubble.agent !== CONDUCTOR && bubble.agent !== UNCLAIMED) {
    openStoredTask(bubble.agent, { campaignId: bubble.campaignId, taskId: bubble.taskId });
  } else if (bubble.jobId) {
    focusAgent(bubble.agent === CONDUCTOR || bubble.agent === UNCLAIMED ? null : bubble.agent);
    focusRun(bubble.jobId);
    onOpenRun();
  } else if (bubble.campaignId) {
    // Research, analysis and the rest start nothing; what there is to read
    // is the brief and the report.
    openTask({ campaignId: bubble.campaignId, taskId: bubble.taskId });
  }
}

function asRun(job: JobSummary): OwnRun {
  const at = Date.parse(job.startedAt || job.createdAt || "");
  return {
    id: job.id,
    label: jobTitle(job),
    state: job.state,
    at: Number.isNaN(at) ? 0 : at,
  };
}