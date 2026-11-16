// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { CampaignDetail, FleetEvent } from "../../types/api";
import type { CampaignView } from "../../lib/campaign";
import { learnt } from "../../lib/atlas";
import { relativeTime } from "../../lib/campaign";
import { Card } from "../RunAnalysis/shared";
import { CampaignBranches } from "./WhatWeKnow";

/** A report, flattened across the conversation's campaigns. */
export interface MemoryReport {
  taskId: string;
  campaignId: string;
  title: string;
  excerpt: string;
  postedAt: string;
}

interface Props {
  /** Every campaign of the conversation, so one card covers them all. */
  details: readonly CampaignDetail[];
  events: ReadonlyMap<string, readonly FleetEvent[]>;
  /** The conductor's own log, which is the only memory a single agent keeps. */
  memory: CampaignView | null;
  reports: readonly MemoryReport[];
  labels: ReadonlyMap<string, string>;
  runs: ReadonlySet<string>;
  now: number;
  onOpenRun: (jobId: string) => void;
  onOpenReport: (report: MemoryReport) => void;
}

/**
 * Everything the conversation knows, in one place: the goal it is working
 * to, the log the conductor keeps, what each campaign has settled, and the
 * reports that came back.
 *
 * One card rather than four. "What we know", "Memory" and "Reports" were
 * three cards saying overlapping things, and a conversation with two
 * campaigns got two more "What we know" headings on top of that, which
 * reads as the panel duplicating itself. Scrolls internally so it can sit
 * above the iteration graph without pushing it off the screen.
 *
 * Single-agent and multi-agent fill the same card: a solo conductor has no
 * sub-agents but does keep a log and does post its own task reports, so
 * the memory and report sections are populated either way.
 */
export function CampaignMemory({
  details,
  events,
  memory,
  reports,
  labels,
  runs,
  now,
  onOpenRun,
  onOpenReport,
}: Props) {
  const goal = memory?.goal ?? details[0]?.campaign.goal ?? null;
  const findings = memory?.findings ?? [];
  const nothing =
    !goal && !memory?.next && findings.length === 0 && details.length === 0 && reports.length === 0;
  if (nothing) return null;

  return (
    <Card
      title="Memory"
      actions={
        reports.length > 0 ? (
          <span className="text-faint text-[11px]">
            {reports.length} {reports.length === 1 ? "report" : "reports"}
          </span>
        ) : null
      }
    >
      {/* Capped rather than unbounded: this sits above the iteration graph,
          and a campaign with many findings would otherwise push it away. */}
      <div className="flex max-h-[26rem] flex-col gap-2.5 overflow-y-auto pr-0.5">
        {goal && (
          <p className="text-[11.5px] leading-relaxed">
            <span className="text-faint mr-1.5 font-medium tracking-wide uppercase">Goal</span>
            {goal}
          </p>
        )}
        {memory?.next && (
          <p className="text-[11px] leading-relaxed">
            <span className="text-faint mr-1.5 font-medium tracking-wide uppercase">Next</span>
            {memory.next}
          </p>
        )}
        {findings.length > 0 && (
          <ul className="border-subtle space-y-1 border-t pt-2 text-[11px] leading-relaxed">
            {findings.map((finding, index) => (
              <li key={`${finding.ts}-${index}`} className="text-muted">
                • {finding.text}
              </li>
            ))}
          </ul>
        )}

        {details.map((one) => (
          <section key={one.campaign.campaign_id} className="border-subtle border-t pt-2">
            {/* Named only when there is more than one, so a single campaign
                does not get a heading repeating the goal just above. */}
            {details.length > 1 && (
              <h4 className="text-faint mb-1 text-[10px] font-semibold tracking-wider uppercase">
                {one.campaign.goal}
              </h4>
            )}
            <CampaignBranches
              campaignId={one.campaign.campaign_id}
              learnt={learnt(events.get(one.campaign.campaign_id), one)}
              labels={labels}
              runs={runs}
              now={now}
              onOpenRun={onOpenRun}
            />
          </section>
        ))}

        {reports.length > 0 && (
          <ul className="border-subtle flex flex-col gap-1.5 border-t pt-2">
            {reports.map((report) => (
              <li key={`${report.campaignId}-${report.taskId}`}>
                <button
                  type="button"
                  onClick={() => onOpenReport(report)}
                  className="border-subtle hover:bg-inset grid w-full cursor-pointer grid-cols-[auto_minmax(0,2fr)_minmax(0,3fr)] items-baseline gap-2 rounded-lg border px-2.5 py-2 text-left"
                >
                  <ReportIcon />
                  <span className="min-w-0 truncate text-[11.5px] font-medium">
                    {report.title}
                    <span className="text-faint ml-1.5 font-normal">
                      {relativeTime(report.postedAt, now)}
                    </span>
                  </span>
                  <span className="text-muted line-clamp-3 text-[11px] leading-snug">
                    {report.excerpt || "No summary written."}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </Card>
  );
}

/** Marks a row as a report rather than a finding, which read alike without it. */
function ReportIcon() {
  return (
    <svg
      viewBox="0 0 12 12"
      aria-hidden="true"
      className="text-faint size-3 shrink-0 translate-y-[1px]"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M3 1.5h4.2L9.5 3.8V10a.5.5 0 0 1-.5.5H3a.5.5 0 0 1-.5-.5V2a.5.5 0 0 1 .5-.5Z" />
      <path d="M7 1.6V4h2.4M4.3 6.2h3.4M4.3 8h2.3" />
    </svg>
  );
}
