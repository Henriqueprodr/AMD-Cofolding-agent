// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useMemo, useState } from "react";
import { useCampaignAtlas } from "../../hooks/useCampaignAtlas";
import {
  branchViews,
  evidenceRun,
  stances,
  type BranchView,
} from "../../lib/atlas";
import { relativeTime } from "../../lib/campaign";
import type { Finding, FindingConfidence, FindingNote } from "../../types/api";
import { Card, Placeholder } from "../RunAnalysis/shared";

interface Props {
  campaignId: string;
  /** The campaign's goal, shown when the conversation has more than one campaign. */
  goal?: string | null;
  /** Findings and notes in the campaign's history so far; the atlas is read again when it moves. */
  learnt: number;
  /** Supervisor id → "S3". */
  labels: ReadonlyMap<string, string>;
  /** Runs the gateway has, so a job id among the evidence opens. */
  runs: ReadonlySet<string>;
  now: number;
  onOpenRun: (jobId: string) => void;
}

const CONFIDENCE: Record<FindingConfidence, { colour: string; label: string }> =
  {
    high: { colour: "var(--color-ok)", label: "high confidence" },
    medium: { colour: "var(--color-warn)", label: "medium confidence" },
    low: { colour: "var(--text-faint)", label: "low confidence" },
  };

const STANCE: Record<NonNullable<FindingNote["stance"]>, string> = {
  supports: "var(--color-ok)",
  disputes: "var(--color-accent)",
  extends: "var(--color-warn)",
};

/**
 * What the campaign has settled, branch by branch, from the findings its
 * agents recorded: not what ran, which the graph above shows, but what was
 * learnt from it. A branch the plan named that holds nothing is listed too,
 * since it is the part of the goal nobody knows anything about yet; one
 * with a finding somebody disputed comes first, since it needs settling.
 */
export function WhatWeKnow({
  campaignId,
  goal,
  learnt,
  labels,
  runs,
  now,
  onOpenRun,
}: Props) {
  const { atlas } = useCampaignAtlas(campaignId, learnt);
  const views = useMemo(() => (atlas ? branchViews(atlas) : []), [atlas]);
  const disputed = views.reduce((sum, view) => sum + view.disputed, 0);

  // Before anything is planned or recorded there is nothing to say; the graph already says the campaign started.
  if (atlas && atlas.findings === 0 && atlas.branches.length === 0) return null;

  return (
    <Card
      title={goal ? `What we know · ${goal}` : "What we know"}
      actions={
        atlas ? (
          <span className="text-faint text-[11px]">
            {atlas.findings} {atlas.findings === 1 ? "finding" : "findings"}
            {disputed > 0 && (
              <span style={{ color: "var(--color-accent)" }}>
                {" "}
                · {disputed} disputed
              </span>
            )}
          </span>
        ) : null
      }
    >
      <CampaignBranches
        campaignId={campaignId}
        learnt={learnt}
        labels={labels}
        runs={runs}
        now={now}
        onOpenRun={onOpenRun}
      />
    </Card>
  );
}

/**
 * One campaign's branches, without a card of its own.
 *
 * Split out so several campaigns can share a single Memory card: a card
 * each meant a conversation with two campaigns showed two "What we know"
 * headings, which reads as the panel having duplicated itself rather than
 * as two campaigns.
 */
export function CampaignBranches({
  campaignId,
  learnt,
  labels,
  runs,
  now,
  onOpenRun,
}: Omit<Props, "goal">) {
  const { atlas, error, loading, retry } = useCampaignAtlas(campaignId, learnt);
  const views = useMemo(() => (atlas ? branchViews(atlas) : []), [atlas]);
  const [open, setOpen] = useState<ReadonlySet<string>>(() => new Set());
  const who = (author: string) => labels.get(author) ?? author;

  const toggle = (branch: string) =>
    setOpen((previous) => {
      const next = new Set(previous);
      if (!next.delete(branch)) next.add(branch);
      return next;
    });

  if (loading) {
    return <Placeholder height={48}>Reading what the campaign knows…</Placeholder>;
  }

  if (!atlas) {
    return (
      <p className="text-faint py-3 text-center text-[11px]">
        {/* Read, but nothing came back for this campaign. Saying so beats
            sitting on a spinner that will never resolve. */}
        {error ?? "Nothing is recorded for this campaign yet."}{" "}
        <button
          type="button"
          onClick={retry}
          className="text-muted hover:text-body cursor-pointer underline"
        >
          Try again
        </button>
      </p>
    );
  }

  return (
    <>
      {error && (
        <p className="text-faint pb-1.5 text-[11px]">Could not refresh: {error}</p>
      )}
      <ul className="flex flex-col">
        {views.map((view) => (
          <BranchRow
            key={view.branch}
            view={view}
            open={open.has(view.branch)}
            onToggle={() => toggle(view.branch)}
            who={who}
            runs={runs}
            now={now}
            onOpenRun={onOpenRun}
          />
        ))}
      </ul>
      {atlas.truncated && (
        <p className="text-faint pt-1.5 text-[11px]">
          Showing the newest {atlas.items.length} of {atlas.findings}.
        </p>
      )}
    </>
  );
}

function BranchRow({
  view,
  open,
  onToggle,
  who,
  runs,
  now,
  onOpenRun,
}: {
  view: BranchView;
  open: boolean;
  onToggle: () => void;
  who: (author: string) => string;
  runs: ReadonlySet<string>;
  now: number;
  onOpenRun: (jobId: string) => void;
}) {
  const empty = view.findings === 0;
  return (
    <li className="border-subtle border-b last:border-b-0">
      <button
        type="button"
        onClick={onToggle}
        disabled={empty}
        aria-expanded={empty ? undefined : open}
        /* Workstream names are full phrases ("JQ1-based Lead Optimization"),
           which 9rem truncated to the point of being unreadable. */
        className="grid w-full grid-cols-[14rem_minmax(0,1fr)_auto] items-baseline gap-3 py-1.5 text-left enabled:cursor-pointer enabled:hover:bg-inset rounded-md px-1"
      >
        <span
          className={`min-w-0 truncate text-[11.5px] ${empty ? "text-faint" : "font-medium"}`}
          title={view.name}
        >
          {view.name}
          {!view.in_plan && (
            <span className="text-faint font-normal"> · not in the plan</span>
          )}
        </span>
        <span
          className={`min-w-0 truncate text-[11px] ${empty ? "text-faint italic" : "text-muted"}`}
        >
          {empty ? "Nothing known yet" : open ? "" : view.latest?.claim}
        </span>
        <span className="text-faint shrink-0 text-[11px] tabular-nums">
          {view.disputed > 0 && (
            <span style={{ color: "var(--color-accent)" }}>
              {view.disputed} disputed ·{" "}
            </span>
          )}
          {empty ? "" : view.findings}
        </span>
      </button>
      {open && (
        <ul className="flex flex-col gap-2 pb-2 pl-1">
          {view.items.map((finding) => (
            <FindingItem
              key={finding.finding_id}
              finding={finding}
              who={who}
              runs={runs}
              now={now}
              onOpenRun={onOpenRun}
            />
          ))}
          {view.items.length < view.findings && (
            <li className="text-faint text-[11px]">
              {view.findings - view.items.length} older not shown.
            </li>
          )}
        </ul>
      )}
    </li>
  );
}

function FindingItem({
  finding,
  who,
  runs,
  now,
  onOpenRun,
}: {
  finding: Finding;
  who: (author: string) => string;
  runs: ReadonlySet<string>;
  now: number;
  onOpenRun: (jobId: string) => void;
}) {
  const confidence = CONFIDENCE[finding.confidence] ?? CONFIDENCE.medium;
  const lean = stances(finding);
  return (
    <li className="border-subtle rounded-lg border px-2.5 py-1.5">
      <p className="flex items-baseline gap-1.5 text-[11.5px] leading-snug">
        <span
          className="inline-block size-1.5 shrink-0 translate-y-[-1px] rounded-full"
          style={{ background: confidence.colour }}
          title={confidence.label}
          aria-label={confidence.label}
        />
        <span className="font-medium">{finding.claim}</span>
      </p>
      {finding.detail && (
        <p className="text-muted mt-0.5 text-[11px] leading-snug whitespace-pre-line">
          {finding.detail}
        </p>
      )}
      <p className="text-faint mt-1 flex flex-wrap items-center gap-x-1.5 gap-y-0.5 text-[10.5px]">
        <span>{who(finding.author)}</span>
        <span>· {relativeTime(finding.created_at, now)}</span>
        {finding.supersedes && <span>· corrects an earlier finding</span>}
        {lean.supports > 0 && (
          <span style={{ color: STANCE.supports }}>
            · {lean.supports} support
          </span>
        )}
        {lean.disputes > 0 && (
          <span style={{ color: STANCE.disputes }}>
            · {lean.disputes} dispute
          </span>
        )}
        {finding.tags.map((tag) => (
          <span key={tag} className="bg-inset rounded px-1">
            {tag}
          </span>
        ))}
      </p>
      {finding.evidence.length > 0 && (
        <p className="mt-1 flex flex-wrap items-center gap-1 text-[10.5px]">
          <span className="text-faint">Evidence:</span>
          {finding.evidence.map((item) => {
            const run = evidenceRun(item, runs);
            return run ? (
              <button
                key={item}
                type="button"
                onClick={() => onOpenRun(run)}
                className="text-muted hover:text-body cursor-pointer rounded px-1 underline"
                title="Open the run"
              >
                {item}
              </button>
            ) : (
              <span key={item} className="text-muted px-1">
                {item}
              </span>
            );
          })}
        </p>
      )}
      {finding.notes.length > 0 && (
        <ul className="border-subtle mt-1.5 flex flex-col gap-1 border-l-2 pl-2">
          {finding.notes.map((note) => (
            <li key={note.note_id} className="text-[11px] leading-snug">
              {note.stance && (
                <span
                  className="font-medium"
                  style={{ color: STANCE[note.stance] }}
                >
                  {note.stance}{" "}
                </span>
              )}
              <span className="text-muted">{note.body}</span>
              <span className="text-faint">
                {" "}
                — {who(note.author)}, {relativeTime(note.created_at, now)}
              </span>
            </li>
          ))}
        </ul>
      )}
    </li>
  );
}
