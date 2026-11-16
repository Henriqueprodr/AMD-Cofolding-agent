// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useMemo } from "react";
import type { Theme } from "../../hooks/useTheme";
import { useCampaigns } from "../../hooks/useCampaignJobs";
import { agentLabels, campaignRuns, objectiveProgress, taskJobs } from "../../lib/fleet";
import { focusTask } from "../../lib/workbenchBus";
import { useJobScope } from "../JobPicker/useJobScope";
import { short } from "../RunAnalysis/plotUtils";
import { Card } from "../RunAnalysis/shared";
import { ConversationMap } from "./ConversationMap";
import { useAgentPage } from "./shared";
import { useConversationSpace } from "./useConversationSpace";

interface Props {
  sessionId: string | null;
  theme: Theme;
  onOpenMolecule?: (smiles: string) => void;
  /** Show the overview, where a task opened from here is read. */
  onShowTask: () => void;
}

/**
 * The overview's Analysis: every molecule of the conversation on one map,
 * whoever made it, and how the objectives have moved across the tasks; on a
 * sub-agent's page, only the task it is on, or the one picked in the trajectory.
 */
export function CampaignAnalysis({ sessionId, theme, onOpenMolecule, onShowTask }: Props) {
  const { details } = useCampaigns(sessionId);
  const labels = useMemo(() => agentLabels(details), [details]);
  const { agent, page } = useAgentPage(details, labels);
  const agents = useMemo(() => new Map([...campaignRuns(details, labels)].map(([jobId, run]) => [jobId, run.agent])), [details, labels]);
  const mine = useMemo(() => (page ? new Set(page.held.map(({ task }) => task.task_id)) : null), [page]);
  const { own } = useJobScope(sessionId);
  const runs = useMemo(() => {
    if (!mine) return own;
    const jobs = taskJobs(details, mine);
    return own.filter((run) => jobs.has(run.id));
  }, [own, details, mine]);
  const { space, loading } = useConversationSpace(runs, agents);

  const campaignOf = useMemo(() => new Map(details.flatMap((detail) => detail.tasks.map((task) => [task.task_id, detail.campaign.campaign_id] as const))), [details]);
  const objectives = useMemo(
    () =>
      objectiveProgress(
        details.flatMap((detail) => detail.results).filter((result) => !mine || mine.has(result.task_id)),
        [...new Set(details.flatMap((detail) => detail.campaign.objectives ?? []))],
      ),
    [details, mine],
  );
  const scope = page && agent ? `${agent}:${[...(mine ?? [])].sort().join(",")}` : null;
  const openTask = (taskId: string) => {
    const campaignId = campaignOf.get(taskId);
    if (!campaignId) return;
    focusTask({ campaignId, taskId });
    onShowTask();
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2.5 overflow-y-auto p-3">
      <Card
        title="Chemical space"
        actions={
          <span className="text-faint text-[11px]">{page ? (page.stored ? `this task of ${agent}` : `what ${agent} is on now`) : "every run of this conversation"}</span>
        }
      >
        <ConversationMap sessionId={sessionId} scope={scope} space={space} loading={loading} theme={theme} onOpenMolecule={onOpenMolecule} />
      </Card>
      {objectives.length > 0 && (
        <Card title="Objectives">
          <ul className="flex flex-col gap-1.5">
            {objectives.map((objective) => (
              <li key={objective.name}>
                <div className="flex items-baseline gap-2 text-[11.5px]">
                  <span className="min-w-0 flex-1 truncate">{objective.name}</span>
                  {objective.delta !== null && <span className="text-faint shrink-0 tabular-nums">+{short(objective.delta)}</span>}
                  <button type="button" onClick={() => openTask(objective.bestTask)} className="shrink-0 cursor-pointer tabular-nums underline-offset-2 hover:underline">
                    {short(objective.best)}
                  </button>
                </div>
                {/* Mean behind best, so a single lucky task is visibly not the
                    campaign; both on the same 0–1 scale the profiles use. */}
                <div className="bg-inset relative mt-1 h-1 overflow-hidden rounded-full">
                  <span className="absolute inset-y-0 left-0 rounded-full" style={{ width: `${bar(objective.mean)}%`, background: "var(--text-muted)", opacity: 0.35 }} />
                  <span className="absolute inset-y-0 left-0 w-px" style={{ left: `${bar(objective.best)}%`, background: "var(--text-body)" }} />
                </div>
                <p className="text-faint mt-0.5 text-[10px]">
                  mean {short(objective.mean)} over {objective.tasks} {objective.tasks === 1 ? "task" : "tasks"}
                </p>
              </li>
            ))}
          </ul>
        </Card>
      )}
    </div>
  );
}

/** A profile score as a percentage of the bar, clamped: profiles are 0–1 but nothing enforces it. */
const bar = (value: number) => Math.max(0, Math.min(100, value * 100));
