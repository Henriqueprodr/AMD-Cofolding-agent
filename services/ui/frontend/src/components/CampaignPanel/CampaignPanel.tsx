// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useMemo } from "react";
import type { Theme } from "../../hooks/useTheme";
import type { Campaign } from "../../types/api";
import { useCampaigns } from "../../hooks/useCampaignJobs";
import { useJobList } from "../../hooks/useJobList";
import { useStuckTasks } from "../../hooks/useTaskConversations";
import { history } from "../../lib/activity";
import { campaignView } from "../../lib/campaign";
import { agentLabels, knownAgent, taskPhase, taskReports } from "../../lib/fleet";
import { focusTask, openRun, useFocusedAgent, useFocusedTask } from "../../lib/workbenchBus";
import { Card, Placeholder } from "../RunAnalysis/shared";
import { AgentConversation } from "../SubAgents/AgentConversation";
import { CampaignMemory } from "./CampaignMemory";
import { IterationGraph } from "./IterationGraph";
import { useNow } from "./shared";
import { TaskDetail } from "./TaskDetail";

interface Props {
  /** The conversation on screen. */
  sessionId: string | null;
  /** The log the conductor keeps of the conversation with campaign_log. */
  log?: Campaign | null;
  theme: Theme;
  onOpenMolecule?: (smiles: string) => void;
  onShowProperties: () => void;
  /** Show a task's run, which is another view of this tab. */
  onOpenRun: () => void;
}

/**
 * The overview, which is the conductor's page and the whole campaign's; or
 * a sub-agent's conversation when one is picked under the trajectory.
 *
 * The overview is the iteration graph, what was handed out, who took it
 * and what came back across every campaign of the conversation, and under
 * it every report, newest first. The molecules and the analysis are views
 * of their own.
 */
export function CampaignPanel({ sessionId, log = null, theme, onOpenMolecule, onShowProperties, onOpenRun }: Props) {
  const now = useNow();
  const { details, events, error, loading } = useCampaigns(sessionId);
  const labels = useMemo(() => agentLabels(details), [details]);
  const focused = useFocusedTask();
  const openAgent = useFocusedAgent();
  const memory = useMemo(() => campaignView(log), [log]);

  const steps = useMemo(
    () => history(details.map((detail) => ({ ...detail, events: events.get(detail.campaign.campaign_id) })), labels, memory?.findings ?? []),
    [details, events, labels, memory],
  );
  const written = useMemo(
    () =>
      details
        .flatMap((detail) => taskReports(detail).map((report) => ({ ...report, campaignId: detail.campaign.campaign_id })))
        .sort((a, b) => (Date.parse(b.postedAt) || 0) - (Date.parse(a.postedAt) || 0)),
    [details],
  );
  const reports = useMemo(() => new Map(written.map((report) => [report.taskId, report.excerpt] as const)), [written]);
  const running = useMemo(() => details.flatMap((detail) => detail.tasks.filter((task) => taskPhase(task) === "running").map((task) => task.task_id)), [details]);
  const stuck = useStuckTasks(running);
  const jobs = useJobList().jobs;
  const knownRuns = useMemo(() => new Set((jobs ?? []).map((job) => job.id)), [jobs]);
  const showRun = (jobId: string) => {
    openRun(jobId);
    onOpenRun();
  };

  if (openAgent && knownAgent(openAgent, labels)) {
    return (
      <AgentConversation
        sessionId={sessionId}
        agent={openAgent}
        taskId={focused?.taskId ?? null}
        theme={theme}
        onOpenMolecule={onOpenMolecule ?? NOTHING}
        onShowProperties={onShowProperties}
      />
    );
  }

  const detail = focused ? details.find((one) => one.campaign.campaign_id === focused.campaignId) : undefined;
  if (focused?.taskId && detail) {
    return (
      <div className="min-h-0 flex-1 overflow-y-auto">
        <TaskDetail
          detail={detail}
          taskId={focused.taskId}
          labels={labels}
          theme={theme}
          onBack={() => focusTask(null)}
          onOpenMolecule={onOpenMolecule}
          onOpenJob={showRun}
        />
      </div>
    );
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2.5 overflow-y-auto p-3">
      {/* Above the graph: what the campaign knows is read far more often
          than the shape of how it got there. */}
      <CampaignMemory
        details={details}
        events={events}
        memory={memory}
        reports={written}
        labels={labels}
        runs={knownRuns}
        now={now}
        onOpenRun={showRun}
        onOpenReport={(report) => focusTask({ campaignId: report.campaignId, taskId: report.taskId })}
      />
      <Card title="Iteration graph">
        {error && !details.length ? (
          <p className="text-faint py-6 text-center text-xs">{error}</p>
        ) : loading && !details.length ? (
          <Placeholder height={80}>Reading the campaigns…</Placeholder>
        ) : (
          <IterationGraph
            steps={steps.steps}
            reports={reports}
            stuck={stuck}
            now={now}
            onOpenTask={(step) => step.campaignId && step.taskId && focusTask({ campaignId: step.campaignId, taskId: step.taskId })}
            onOpenRun={showRun}
          />
        )}
      </Card>
    </div>
  );
}

const NOTHING = () => undefined;
