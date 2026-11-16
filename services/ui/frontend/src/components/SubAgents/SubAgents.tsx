// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { memo, useEffect, useMemo, useState } from "react";
import type { Artifacts } from "../../lib/artifacts";
import { agentLabels, fewWords, workingAgents } from "../../lib/fleet";
import { focusAgent, focusTask, openAgentChat, openWorkbench, useAgentChat } from "../../lib/workbenchBus";
import { useOverflowCount } from "../../hooks/useOverflowCount";
import { refreshCampaigns, useCampaigns } from "../../hooks/useCampaignJobs";
import { useStuckTasks } from "../../hooks/useTaskConversations";
/** Narrowest a card gets and still reads: its label and a few words of its task, plus the gap. */
const CARD_MIN = 126;

/** Calls after which the ledger has something new to say about this conversation's campaigns.
 *
 * start_own_task / finish_own_task are the single-agent equivalents of
 * enqueue_task / conduct_campaign: they add a task and post a result. Without
 * them here the campaign panel would not refresh after own-task activity and
 * the Overview would stay empty. atlas_finding / atlas_note / set_campaign_plan
 * also mutate visible campaign state, as does take_proposals.
 */
const CHANGES_CAMPAIGNS = new Set([
  "start_campaign",
  "enqueue_task",
  "conduct_campaign",
  "cancel_supervisor",
  "close_campaign",
  "finalize_campaign",
  // Single-agent own-task tools — equivalent campaign mutations to the above.
  "start_own_task",
  "finish_own_task",
  // Plan and atlas writes update what the Overview shows.
  "set_campaign_plan",
  "atlas_finding",
  "atlas_note",
  "take_proposals",
]);

interface Props {
  sessionId: string | null;
  steps: Artifacts["steps"];
}

/**
 * The sub-agents at work, in the conversation with the conductor: one card
 * each for the first three, with what it is on. A card
 * opens that sub-agent's conversation in the workbench and addresses the
 * message box to it; again, and both go back to the conductor.
 */
export const SubAgents = memo(function SubAgents({ sessionId, steps }: Props) {
  const { details } = useCampaigns(sessionId);
  const working = useMemo(() => workingAgents(details, agentLabels(details)), [details]);
  const chatting = useAgentChat();
  const [overflowOpen, setOverflowOpen] = useState(false);
  const [rowRef, visibleCount] = useOverflowCount(working.length, CARD_MIN, 38);
  const stuck = useStuckTasks(useMemo(() => working.slice(0, visibleCount).map((one) => one.taskId), [working, visibleCount]));

  const changes = steps.filter((step) => step.result && CHANGES_CAMPAIGNS.has(step.name)).length;
  useEffect(() => {
    if (sessionId && changes > 0) refreshCampaigns(sessionId);
  }, [sessionId, changes]);

  if (working.length === 0) return null;
  const more = working.length - visibleCount;
  const open = (agent: string) => {
    const closing = chatting === agent;
    openAgentChat(closing ? null : agent);
    focusAgent(closing ? null : agent);
    focusTask(null);
    openWorkbench("campaign");
  };

  return (
    <div className="mx-auto mb-1.5 w-full max-w-3xl px-3 sm:px-4" aria-label="Sub-agents at work">
      <div ref={rowRef} className="flex items-stretch gap-1.5">
        {working.slice(0, visibleCount).map((one) => (
          <button
            key={one.taskId}
            type="button"
            onClick={() => open(one.agent)}
            aria-pressed={chatting === one.agent}
            title={one.title}
            className={`bg-sunken hover:bg-inset flex min-w-0 flex-1 cursor-pointer flex-col gap-0.5 rounded-lg border px-2 py-1 text-left text-[11px] ${
              chatting === one.agent ? "border-[var(--text-muted)]" : "border-subtle"
            }`}
          >
            <span className="flex items-center gap-1.5">
              <span aria-hidden="true" className="bg-[var(--text-muted)] size-1.5 shrink-0 animate-pulse rounded-full" />
              <span className="text-body font-semibold">{one.agent}</span>
              {stuck.has(one.taskId) && (
                <span title={`Resending ${stuck.get(one.taskId)}, which fails the same way each time`} className="text-muted text-[10px] italic">
                  stuck
                </span>
              )}
            </span>
            <span className="text-muted truncate">{fewWords(one.title)}</span>
          </button>
        ))}
        {/* "…" overflow button — shown when agents don't fit; clicking lists them all. */}
        {more > 0 && (
          <div className="relative flex shrink-0">
            <button
              type="button"
              onClick={() => setOverflowOpen((v) => !v)}
              aria-haspopup="listbox"
              aria-expanded={overflowOpen}
              title={`${more} more sub-agent${more === 1 ? "" : "s"} at work`}
              className="bg-sunken border-subtle hover:bg-inset flex h-full w-8 cursor-pointer items-center justify-center rounded-lg border text-xs text-[var(--text-muted)]"
            >
              …
            </button>
            {overflowOpen && (
              <>
                <button
                  type="button"
                  aria-label="Close sub-agent list"
                  tabIndex={-1}
                  className="fixed inset-0 z-30 cursor-default"
                  onClick={() => setOverflowOpen(false)}
                />
                <ul
                  role="listbox"
                  aria-label="Hidden sub-agents"
                  className="bg-raised border-subtle shadow-card absolute bottom-full right-0 z-40 mb-1 max-h-56 w-44 overflow-y-auto rounded-lg border py-1"
                >
                  {working.slice(visibleCount).map((one) => (
                    <li key={one.taskId} role="option" aria-selected={chatting === one.agent}>
                      <button
                        type="button"
                        onClick={() => { setOverflowOpen(false); open(one.agent); }}
                        className={`hover:bg-sunken flex w-full cursor-pointer flex-col gap-0.5 px-3 py-1.5 text-left text-[11px] ${chatting === one.agent ? "text-body font-semibold" : "text-muted"}`}
                      >
                        <span className="text-body font-semibold">{one.agent}</span>
                        <span className="text-muted truncate">{fewWords(one.title)}</span>
                      </button>
                    </li>
                  ))}
                </ul>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
});
