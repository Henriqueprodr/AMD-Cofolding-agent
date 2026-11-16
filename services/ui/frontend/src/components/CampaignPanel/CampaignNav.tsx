// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useMemo, useState } from "react";
import { useCampaigns } from "../../hooks/useCampaignJobs";
import { byAgent } from "../../lib/chemicalSpace";
import { agentLabels, campaignRuns, knownAgent } from "../../lib/fleet";
import type { DockView } from "../../lib/dock";
import { agentChat, focusAgent, focusRun, focusTask, openAgentChat, useFocusedAgent, useFocusedRun } from "../../lib/workbenchBus";
import { useOverflowCount } from "../../hooks/useOverflowCount";
import { JobPickerView } from "../JobPicker/JobPicker";
import { useJobScope } from "../JobPicker/useJobScope";

interface Props {
  sessionId: string | null;
  /** The Campaign tab's view: "campaign" for the page itself, or one of the views beside it. */
  view: DockView;
  onView: (view: DockView) => void;
}

/** Views that are about one run, under the run picker. */
const RUN_VIEWS: readonly DockView[] = ["jobs", "results", "activity"];

/**
 * Under the trajectory: the conductor, then one button per sub-agent.
 *
 * The conductor's page is the overview of the whole campaign: what happened, every molecule, and
 * the analysis across all of them. A sub-agent's page opens on its
 * conversation, with the same molecules and analysis beside it, limited to
 * the task it is on now.
 */
export function CampaignNav({ sessionId, view, onView }: Props) {
  const { details } = useCampaigns(sessionId);
  const labels = useMemo(() => agentLabels(details), [details]);
  const agents = useMemo(() => [...new Set(labels.values())].sort(byAgent), [labels]);
  const focused = useFocusedAgent();
  const agent = knownAgent(focused, labels) ? focused : null;
  const [overflowOpen, setOverflowOpen] = useState(false);
  // A sub-agent pill ("S12") is about 40 px with its gap.
  const [pillsRef, visibleAgents] = useOverflowCount(agents.length, 40, 32);

  const { runs, own, job, alerts, error } = useJobScope(sessionId);
  const delegated = useMemo(() => campaignRuns(details, labels), [details, labels]);
  // A sub-agent's page picks among its own runs, opened from its bubbles in the trajectory.
  const mine = useMemo(() => (agent ? own.filter((run) => delegated.get(run.id)?.agent === agent) : own), [own, delegated, agent]);
  const picked = useFocusedRun();
  const runView = RUN_VIEWS.includes(view);
  // Jobs with no run picked is the list of them all, which is the conductor's page.
  const listing = view === "jobs" && picked === null;

  useEffect(() => {
    if (listing && agent) focusAgent(null);
    // Results and Run analysis are of one run: the one they show, picked, so Summary shows it too.
    else if (runView && !listing && picked === null && job) focusRun(job.id);
  }, [listing, agent, runView, picked, job]);

  const views: { id: DockView; label: string }[] = [
    { id: "campaign", label: agent ? "Conversation" : "Overview" },
    { id: "pooled", label: "Molecules" },
    { id: "space", label: "Analysis" },
  ];
  if (!agent) views.push({ id: "jobs", label: "Jobs" });

  const overview = () => {
    focusAgent(null);
    focusTask(null);
    openAgentChat(null);
    onView("campaign");
  };
  const pick = (id: DockView) => {
    if (id === "campaign" && !agent) overview();
    else if (id === "jobs") {
      focusRun(null);
      onView("jobs");
    } else onView(id);
  };

  const viewPills = (
    <div role="group" aria-label={agent ? `${agent}'s views` : "Views"} className="flex shrink-0 items-center gap-0.5">
      {views.map((one) => {
        const on = one.id === view || (one.id === "jobs" && runView);
        return (
          <button
            key={one.id}
            type="button"
            aria-pressed={on}
            onClick={() => pick(one.id)}
            className={`cursor-pointer rounded-md px-2.5 py-0.5 text-[11px] touch:min-h-11 ${on ? "bg-inset text-body" : "text-muted hover:text-body"}`}
          >
            {one.label}
          </button>
        );
      })}
    </div>
  );

  return (
    <>
      {agents.length > 0 && (
        <div role="group" aria-label="Whose work" className="border-subtle relative flex shrink-0 items-center gap-1 border-b px-2 py-1.5">
          <Pill on={agent === null} onClick={overview}>
            Conductor
          </Pill>
          {/* Measured row: the sub-agents that fit, then "…" right after the last of them. */}
          <div ref={pillsRef} className="flex min-w-0 flex-1 items-center gap-1">
            {agents.slice(0, visibleAgents).map((label) => (
              <Pill
                key={label}
                on={agent === label}
                onClick={() => {
                  focusAgent(label);
                  // The pill is the sub-agent now; its finished tasks are read from their bubbles.
                  focusTask(null);
                  // The message box follows whoever is open, once it is addressed to a sub-agent at all.
                  if (agentChat()) openAgentChat(label);
                  onView("campaign");
                }}
              >
                {label}
              </Pill>
            ))}
          {agents.length > visibleAgents && (
            <div className="relative shrink-0">
              <button
                type="button"
                onClick={() => setOverflowOpen((v) => !v)}
                aria-haspopup="listbox"
                aria-expanded={overflowOpen}
                className="text-muted hover:text-body hover:bg-inset cursor-pointer rounded-md px-1.5 py-1 text-xs touch:min-h-11"
                title={`${agents.length - visibleAgents} more sub-agents`}
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
                    aria-label="Sub-agents"
                    className="bg-raised border-subtle shadow-card absolute top-full left-0 z-40 mt-1 max-h-56 w-28 overflow-y-auto rounded-lg border py-1"
                  >
                    {agents.slice(visibleAgents).map((label) => (
                      <li key={label} role="option" aria-selected={agent === label}>
                        <button
                          type="button"
                          onClick={() => {
                            setOverflowOpen(false);
                            focusAgent(label);
                            focusTask(null);
                            if (agentChat()) openAgentChat(label);
                            onView("campaign");
                          }}
                          className={`hover:bg-sunken w-full cursor-pointer px-3 py-1.5 text-left text-xs ${agent === label ? "text-body font-semibold" : "text-muted"}`}
                        >
                          {label}
                        </button>
                      </li>
                    ))}
                  </ul>
                </>
              )}
            </div>
          )}
          </div>
          {/* The views share the row: the sub-agents that do not fit go under "…" instead of a row of their own. */}
          <span aria-hidden="true" className="mx-1 h-4 w-px shrink-0" style={{ background: "var(--border-strong)" }} />
          {viewPills}
        </div>
      )}
      {agents.length === 0 && <div className="border-subtle flex shrink-0 items-center border-b px-2 py-1">{viewPills}</div>}
      {runView && !listing && (
        // One row for the run: which one it is and its views. Back to every
        // run is the Jobs view above, so the row does not repeat it.
        <div role="group" aria-label="This run's views" className="border-subtle flex shrink-0 items-center gap-0.5 border-b px-2 py-1">
          {RUN_TABS.map((one) => (
            <button
              key={one.id}
              type="button"
              aria-pressed={one.id === view}
              onClick={() => onView(one.id)}
              className={`shrink-0 cursor-pointer rounded-md px-2.5 py-0.5 text-[11px] touch:min-h-11 ${one.id === view ? "bg-inset text-body" : "text-muted hover:text-body"}`}
            >
              {one.label}
            </button>
          ))}
          <JobPickerView runs={runs ?? mine} own={mine} job={job} alerts={alerts} error={error} onPick={focusRun} className="flex-1 pl-2" />
        </div>
      )}
    </>
  );
}

const RUN_TABS: { id: DockView; label: string }[] = [
  { id: "jobs", label: "Summary" },
  { id: "results", label: "Results" },
  { id: "activity", label: "Run analysis" },
];

function Pill({ on, onClick, children }: { on: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      aria-pressed={on}
      onClick={onClick}
      className={`shrink-0 cursor-pointer rounded-md px-2.5 py-1 text-xs font-medium touch:min-h-11 ${on ? "bg-inset text-body" : "text-muted hover:text-body"}`}
    >
      {children}
    </button>
  );
}
