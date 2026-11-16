// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useMemo } from "react";
import type { Theme } from "../../hooks/useTheme";
import type { CampaignDetail, FleetTask, Turn } from "../../types/api";
import { useCampaigns } from "../../hooks/useCampaignJobs";
import { useTaskConversations } from "../../hooks/useTaskConversations";
import { agentLabels, agentPage, reportBody, taskPhase } from "../../lib/fleet";
import { stuckOn, subagentTurns } from "../../lib/subagentChat";
import { focusTask } from "../../lib/workbenchBus";
import { Transcript } from "../Transcript/Transcript";

interface Props {
  sessionId: string | null;
  /** S1, S2…: whose conversation. */
  agent: string;
  /** A task of its picked from the trajectory; null for the one it is on now. */
  taskId: string | null;
  theme: Theme;
  onOpenMolecule: (smiles: string) => void;
  onShowProperties: () => void;
}

const NOTHING = () => undefined;

/**
 * One sub-agent's conversation on one task: what it was told, what it did,
 * what it answered. The task it is on now, or the one *taskId* names, which
 * is how a finished task is read again. A message typed while it is open
 * goes to it, and waits below until it has read it.
 *
 * From its transcript where it wrote one; else from what crossed back
 * through the ledger, the brief and the report, which is the same
 * conversation with the steps in between missing.
 */
export function AgentConversation({ sessionId, agent, taskId, theme, onOpenMolecule, onShowProperties }: Props) {
  const { details } = useCampaigns(sessionId);
  const labels = useMemo(() => agentLabels(details), [details]);
  const page = useMemo(() => agentPage(details, labels, agent, taskId), [details, labels, agent, taskId]);
  const held = page.held;

  const ids = useMemo(() => held.map(({ task }) => task.task_id), [held]);
  const live = useMemo(() => new Set(held.filter(({ task }) => taskPhase(task) === "running").map(({ task }) => task.task_id)), [held]);
  const conversations = useTaskConversations(ids, live);

  const turns = held.flatMap(({ task, detail }): Turn[] => {
    const running = live.has(task.task_id);
    const read = conversations.get(task.task_id);
    if (read?.found) return subagentTurns(read.events, task.task_id, running);
    return fromLedger(task, detail, running);
  });

  const working = live.size > 0;
  const stuck = [...live].map((id) => conversations.get(id)).flatMap((read) => (read?.found ? [stuckOn(read.events)] : [])).find(Boolean);
  const queued = held.flatMap(({ task }) => (live.has(task.task_id) ? (conversations.get(task.task_id)?.queued ?? []) : []));
  const title = held.length === 1 ? titleOf(held[0].task) : null;
  const state = working ? "working" : page.stored ? (taskPhase(held[0].task) === "failed" ? "failed" : "reported") : "waiting for its next task";
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="border-subtle flex shrink-0 items-center gap-2 border-b px-3 py-1.5 text-xs">
        <span className="text-body shrink-0 font-semibold">{agent}</span>
        {title && <span className="text-muted min-w-0 truncate">{title}</span>}
        <span className="text-faint shrink-0">{state}</span>
        {page.stored && (
          <button type="button" onClick={() => focusTask(null)} className="text-muted hover:text-body ml-auto shrink-0 cursor-pointer text-[11px]">
            {agent} now ›
          </button>
        )}
        {stuck && (
          <span title={`Resending ${stuck}, which fails the same way each time`} className="text-muted italic">
            stuck on {stuck}
          </span>
        )}
      </div>
      {turns.length ? (
        <Transcript
          turns={turns}
          theme={theme}
          openStructurePath={null}
          onOpenMolecule={onOpenMolecule}
          onShowProperties={onShowProperties}
          onPickPrompt={NOTHING}
          hasStructure={false}
        />
      ) : page.last ? (
        <div className="text-faint flex flex-1 flex-col items-center gap-2 px-8 py-12 text-center text-xs">
          <p>{agent} has reported and is waiting for its next task.</p>
          <button
            type="button"
            onClick={() => page.last && focusTask({ campaignId: page.last.detail.campaign.campaign_id, taskId: page.last.task.task_id })}
            className="text-muted hover:text-body cursor-pointer underline-offset-2 hover:underline"
          >
            Read its last task: {titleOf(page.last.task)}
          </button>
        </div>
      ) : (
        <p className="text-faint flex-1 px-8 py-12 text-center text-xs">{agent} has not been given a task yet.</p>
      )}
      {queued.length > 0 && (
        <ul aria-label={`Waiting for ${agent} to read`} className="border-subtle flex shrink-0 flex-col gap-1 border-t px-3 py-1.5">
          {queued.map((message) => (
            <li key={message.ts} className="text-muted flex items-baseline gap-2 text-[11.5px]">
              <span className="min-w-0 flex-1 truncate">{message.text}</span>
              <span className="text-faint shrink-0 text-[10px] italic">waiting for {agent} to read it</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

const titleOf = (task: FleetTask) => (task.spec?.title || task.spec?.intent || task.task_id).trim();

/** What the ledger has of a conversation with no transcript: the brief, then the report when there is one. */
function fromLedger(task: FleetTask, detail: CampaignDetail, running: boolean): Turn[] {
  const asked = Date.parse(task.claimed_at || task.created_at) || 0;
  const result = detail.results.find((one) => one.task_id === task.task_id);
  const report = reportBody(result?.report || "") || (result?.summary || "").trim();
  const failed = taskPhase(task) === "failed";
  const brief = [task.spec.title, task.spec.intent].filter(Boolean).join("\n\n");
  return [
    { role: "user", id: `${task.task_id}:brief`, content: brief || "(no brief recorded)", at: asked },
    {
      role: "agent",
      id: `${task.task_id}:report`,
      at: asked,
      endedAt: Date.parse(result?.posted_at || task.finished_at || "") || undefined,
      done: !running,
      blocks: running
        ? []
        : failed
          ? [{ kind: "error", id: `${task.task_id}:error`, message: task.error || "The task failed." }]
          : [{ kind: "text", id: `${task.task_id}:text`, content: report || "It posted nothing back." }],
    },
  ];
}
