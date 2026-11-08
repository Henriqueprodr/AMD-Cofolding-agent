// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useMemo, useState } from "react";
import type { Theme } from "../../hooks/useTheme";
import type { CampaignDetail, FleetCampaign } from "../../types/api";
import { ApiError, fetchCampaign, fetchCampaigns } from "../../lib/api";
import { relativeTime } from "../../lib/campaign";
import { campaignActive, campaignProgress } from "../../lib/fleet";
import { openRun, openWorkbench } from "../../lib/workbenchBus";
import { Placeholder } from "../RunAnalysis/shared";
import { CampaignResults } from "./CampaignResults";
import { useNow } from "./shared";
import { TaskDetail } from "./TaskDetail";
import { TaskGraph } from "./TaskGraph";

interface Props {
  /** The conversation on screen; its campaigns are listed first. */
  sessionId: string | null;
  theme: Theme;
  onOpenMolecule?: (smiles: string) => void;
}

/** While anything is still running. Nothing is polled once every campaign has settled. */
const POLL_MS = 8_000;

type View = "trajectory" | "results";

/**
 * The campaigns of this conversation: what the conductor set going, how far
 * it has got, and what has come out.
 *
 * The multi-agent machinery is deliberately not the subject here. The tab
 * leads with the goal and the results; the sub-agents are a thing you can
 * look into, through the graph and the drill-down, when you want to know how
 * the search was run.
 */
export function CampaignPanel({ sessionId, theme, onOpenMolecule }: Props) {
  const now = useNow();
  const [campaigns, setCampaigns] = useState<FleetCampaign[] | null>(null);
  /** The ones shown belong to no conversation, so the tab has to say so. */
  const [orphan, setOrphan] = useState(false);
  const [listError, setListError] = useState<string | null>(null);
  const [chosen, setChosen] = useState<string | null>(null);
  const [detail, setDetail] = useState<CampaignDetail | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [view, setView] = useState<View>("trajectory");
  const [task, setTask] = useState<string | null>(null);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const mine = sessionId ? await fetchCampaigns(sessionId) : [];
        if (cancelled) return;
        if (mine.length) {
          setCampaigns(mine);
          setOrphan(false);
        } else {
          // The ledger leaves campaigns belonging to no conversation out of a
          // filtered list, which is right for "which are mine" and wrong for
          // a tab that would then say nothing is running while something is.
          // One started from the command line is still worth showing, said as
          // what it is.
          const all = await fetchCampaigns();
          if (cancelled) return;
          setCampaigns(all.filter((campaign) => !campaign.session_id));
          setOrphan(true);
        }
        setListError(null);
      } catch (reason) {
        if (!cancelled) setListError(message(reason));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [sessionId, tick]);

  // The newest campaign, until the reader picks another; a campaign opened
  // while they are reading an older one must not pull the panel away.
  const current = useMemo(() => {
    if (!campaigns?.length) return null;
    return campaigns.find((c) => c.campaign_id === chosen) ?? campaigns[0];
  }, [campaigns, chosen]);
  const currentId = current?.campaign_id ?? null;

  useEffect(() => {
    if (!currentId) return;
    let cancelled = false;
    fetchCampaign(currentId)
      .then((loaded) => {
        if (cancelled) return;
        setDetail(loaded);
        setDetailError(null);
      })
      .catch((reason: unknown) => {
        if (!cancelled) setDetailError(message(reason));
      });
    return () => {
      cancelled = true;
    };
  }, [currentId, tick]);

  // Showing one campaign and then another must not show the first's tasks
  // under the second's heading for a tick.
  const shown = detail && detail.campaign.campaign_id === currentId ? detail : null;
  const live = (campaigns ?? []).some(campaignActive);

  useEffect(() => {
    if (!live) return;
    const timer = window.setInterval(() => setTick((t) => t + 1), POLL_MS);
    return () => window.clearInterval(timer);
  }, [live]);

  if (listError) {
    return <Notice>{listError}</Notice>;
  }
  if (campaigns === null) {
    return <Placeholder height={120}>Reading the campaigns…</Placeholder>;
  }
  if (!campaigns.length) {
    return (
      <Placeholder height={160}>
        No campaign yet. Allow more than one sub-agent beside the message box, and the agent runs experiments side by side when the work divides.
      </Placeholder>
    );
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <Header
        campaigns={campaigns}
        current={current}
        now={now}
        orphan={orphan}
        onChoose={(id) => {
          setChosen(id);
          setTask(null);
        }}
      />
      {shown === null ? (
        detailError ? (
          <Notice>{detailError}</Notice>
        ) : (
          <Placeholder height={120}>Reading the campaign…</Placeholder>
        )
      ) : task ? (
        <div className="min-h-0 flex-1 overflow-y-auto">
          <TaskDetail
            detail={shown}
            taskId={task}
            theme={theme}
            onBack={() => setTask(null)}
            onOpenMolecule={onOpenMolecule}
            onOpenJob={(jobId) => {
              openRun(jobId);
              openWorkbench("jobs");
            }}
          />
        </div>
      ) : (
        <>
          <Overview detail={shown} />
          <div role="group" aria-label="Campaign" className="border-subtle flex shrink-0 items-center gap-0.5 border-b px-2 py-1.5">
            {(
              [
                ["trajectory", "Trajectory"],
                ["results", "Results"],
              ] as const
            ).map(([id, label]) => (
              <button
                key={id}
                type="button"
                aria-pressed={view === id}
                onClick={() => setView(id)}
                className={`cursor-pointer rounded-md px-2.5 py-1 text-xs touch:min-h-11 ${view === id ? "bg-inset text-body" : "text-muted hover:text-body"}`}
              >
                {label}
              </button>
            ))}
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto">
            {view === "trajectory" ? (
              <div className="p-3">
                <TaskGraph tasks={shown.tasks} selected={null} onSelect={(id) => id && setTask(id)} now={now} />
                <p className="text-faint mt-2 text-[10.5px] leading-relaxed">
                  Each dot is one task, placed when it started; lines show which results the conductor wrote it from. Click one to see how it went.
                </p>
              </div>
            ) : (
              <CampaignResults detail={shown} theme={theme} onOpenMolecule={onOpenMolecule} onOpenTask={setTask} />
            )}
          </div>
        </>
      )}
    </div>
  );
}

function Header({
  campaigns,
  current,
  now,
  orphan,
  onChoose,
}: {
  campaigns: readonly FleetCampaign[];
  current: FleetCampaign | null;
  now: number;
  orphan: boolean;
  onChoose: (id: string) => void;
}) {
  if (!current) return null;
  return (
    <div className="border-subtle shrink-0 border-b px-3 py-2">
      <div className="flex items-start gap-2">
        <p className="min-w-0 flex-1 text-[12px] leading-snug">{current.goal}</p>
        {campaigns.length > 1 && (
          <select
            value={current.campaign_id}
            onChange={(event) => onChoose(event.target.value)}
            aria-label="Campaign"
            className="bg-sunken border-subtle text-muted max-w-36 shrink-0 cursor-pointer rounded-md border px-1.5 py-0.5 text-[11px] outline-none"
          >
            {campaigns.map((campaign) => (
              <option key={campaign.campaign_id} value={campaign.campaign_id}>
                {campaignActive(campaign) ? "Ongoing" : "Past"} · {relativeTime(campaign.created_at, now)}
              </option>
            ))}
          </select>
        )}
      </div>
      <p className="text-faint mt-0.5 text-[10.5px]">
        {campaignActive(current) ? "ongoing" : current.state} · started {relativeTime(current.created_at, now)}
        {current.base_ttp_id && ` · varying ${current.base_ttp_id}`}
        {orphan && " · not this conversation's"}
      </p>
    </div>
  );
}

/** How far along, in one line of numbers and one bar. */
function Overview({ detail }: { detail: CampaignDetail }) {
  const progress = campaignProgress(detail);
  return (
    <div className="border-subtle shrink-0 border-b px-3 py-2">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5 text-[11.5px]">
        <Stat value={progress.molecules} label="molecules" />
        <Stat value={progress.done} label="tasks done" />
        {progress.running > 0 && <Stat value={progress.running} label="running" />}
        {progress.failed > 0 && <Stat value={progress.failed} label="failed" />}
        <span className="text-faint ml-auto" title={`${progress.concurrency} wanted at once`}>
          {progress.agentsLive} of {progress.concurrency} sub-agents
        </span>
      </div>
      {progress.fraction !== null && (
        <div className="bg-inset mt-1.5 h-1 overflow-hidden rounded-full">
          <span className="block h-full rounded-full transition-[width]" style={{ width: `${progress.fraction * 100}%`, background: "var(--color-ok)" }} />
        </div>
      )}
    </div>
  );
}

function Stat({ value, label }: { value: number; label: string }) {
  return (
    <span>
      <span className="tabular-nums">{value}</span> <span className="text-faint">{label}</span>
    </span>
  );
}

function Notice({ children }: { children: React.ReactNode }) {
  return <p className="text-faint flex-1 px-8 py-12 text-center text-xs leading-relaxed">{children}</p>;
}

/** A 503 is the ledger being away, which is not the same as an empty fleet. */
function message(reason: unknown): string {
  if (reason instanceof ApiError && reason.status === 503) {
    return `The campaign ledger cannot be reached, so what is running is unknown. ${reason.message}`;
  }
  if (reason instanceof ApiError && reason.status === 404) {
    // The route itself is absent, so this gateway is older than campaigns.
    // Left as the bare "Not Found" it reads as "you have no campaigns",
    // which sends the reader looking for a campaign that was never possible.
    return "This gateway does not serve campaigns. It is running a build from before the fleet existed, so no campaign can be started or read until it is updated.";
  }
  return reason instanceof Error ? reason.message : String(reason);
}
