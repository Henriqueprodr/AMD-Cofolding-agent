// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useMemo, useState } from "react";
import type { ReactNode } from "react";
import { useCampaigns } from "../../hooks/useCampaignJobs";
import { agentLabels, campaignRuns, CONDUCTOR, UNCLAIMED } from "../../lib/fleet";
import { duration } from "../../lib/format";
import { ELSEWHERE, jobRows, launchers, RUN_GROUPS, type JobRow, type RunGroup } from "../../lib/jobList";
import { isActive, jobTitle, stateTone, toneColor } from "../../lib/jobs";
import { focusRun } from "../../lib/workbenchBus";
import { Card, Placeholder } from "../RunAnalysis/shared";
import { useJobScope } from "../JobPicker/useJobScope";
import { useNow } from "./shared";

interface Props {
  sessionId: string | null;
}

/** Rows shown per group before "Show more", so hundreds of runs still draw at once. */
const PAGE = 50;

const GROUP_TITLE: Record<RunGroup, string> = { running: "Running", waiting: "Waiting for GPUs", finished: "Finished" };

const byLabel = (by: string) => (by === CONDUCTOR ? "Conductor" : by === UNCLAIMED ? "Sub-agent" : by);

/**
 * The Conductor page's Jobs view: every REINVENT run of the conversation —
 * the conductor's and every sub-agent's — or of every conversation, running,
 * waiting and finished. Picking one opens it in this view.
 */
export function CampaignJobs({ sessionId }: Props) {
  const now = useNow();
  const { runs, own, alerts, error } = useJobScope(sessionId);
  const { details } = useCampaigns(sessionId);
  const delegated = useMemo(() => campaignRuns(details, agentLabels(details)), [details]);
  const [state, setState] = useState<RunGroup | null>(null);
  const [by, setBy] = useState<string | null>(null);
  const [shown, setShown] = useState<Partial<Record<RunGroup, number>>>({});

  const rows = useMemo(() => {
    if (!runs) return null;
    const ownIds = new Set(own.map((run) => run.id));
    // Always scoped to this conversation only: runs the conductor or its sub-agents started.
    return jobRows(runs.filter((run) => ownIds.has(run.id)), ownIds, delegated, now);
  }, [runs, own, delegated, now]);
  const who = useMemo(() => launchers(rows ?? []), [rows]);
  const kept = (rows ?? []).filter((row) => (state === null || row.group === state) && (by === null || row.by === by));

  if (!rows) {
    return <Placeholder>{error ? `Could not load the jobs: ${error}` : "Reading the jobs…"}</Placeholder>;
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2.5 overflow-y-auto p-3 text-xs">
      <div className="flex flex-wrap items-center gap-1.5">
        <Segment label="State" value={state} onChange={setState} options={[[null, "All"], ...RUN_GROUPS.map((group) => [group, GROUP_TITLE[group]] as [RunGroup, string])]} />
        {who.length > 1 && (
          <select
            aria-label="Started by"
            value={by ?? ""}
            onChange={(event) => setBy(event.target.value || null)}
            className="bg-raised border-subtle text-body ml-auto cursor-pointer rounded-md border px-1.5 py-0.5 text-[11px]"
          >
            <option value="">Anyone</option>
            {who.map((label) => (
              <option key={label} value={label}>
                {byLabel(label)}
              </option>
            ))}
          </select>
        )}
      </div>
      {!kept.length && (
        <Placeholder height={80}>{rows.length ? "No run matches." : "No run in this conversation yet."}</Placeholder>
      )}
      {RUN_GROUPS.map((group) => {
        const list = kept.filter((row) => row.group === group);
        if (!list.length) return null;
        const limit = shown[group] ?? PAGE;
        return (
          <Card key={group} title={GROUP_TITLE[group]} actions={<span className="text-faint text-[11px]">{list.length}</span>}>
            <ul className="flex flex-col gap-1">
              {list.slice(0, limit).map((row) => (
                <li key={row.job.id}>
                  <Row row={row} alerts={alerts.get(row.job.id) ?? 0} />
                </li>
              ))}
            </ul>
            {list.length > limit && (
              <button
                type="button"
                onClick={() => setShown((was) => ({ ...was, [group]: limit + PAGE }))}
                className="text-muted hover:text-body mt-1.5 cursor-pointer text-[11px] underline decoration-dotted"
              >
                Show {Math.min(PAGE, list.length - limit)} more
              </button>
            )}
          </Card>
        );
      })}
    </div>
  );
}

function Row({ row, alerts }: { row: JobRow; alerts: number }) {
  const { job } = row;
  const facts = [
    row.by === ELSEWHERE ? ELSEWHERE : byLabel(row.by) + (row.task ? ` · ${row.task}` : ""),
    job.usesBoltz === null ? null : job.usesBoltz ? "Boltz" : "no Boltz",
    job.slurmJobId && `Slurm ${job.slurmJobId}`,
    job.maxSteps !== null && `${job.maxSteps} steps`,
    row.waitedMs !== null && `waited ${duration(row.waitedMs)}`,
  ].filter(Boolean);
  return (
    <button
      type="button"
      onClick={() => focusRun(job.id)}
      className="border-subtle hover:bg-inset flex w-full cursor-pointer flex-col gap-0.5 rounded-lg border px-2.5 py-1.5 text-left"
    >
      <span className="flex min-w-0 items-center gap-2">
        <span
          aria-hidden="true"
          className={`size-1.5 shrink-0 rounded-full ${isActive(job.state) ? "animate-pulse" : ""}`}
          style={{ background: toneColor(stateTone(job.state)) }}
        />
        <span className="min-w-0 truncate text-[11.5px] font-medium">{jobTitle(job)}</span>
        <span className="text-faint shrink-0 font-mono text-[10.5px]">{job.id}</span>
        <span className="text-muted shrink-0">{job.state}</span>
        {alerts > 0 && (
          <span className="text-warn shrink-0 text-[10.5px]">
            {alerts} alert{alerts === 1 ? "" : "s"}
          </span>
        )}
        {row.ranMs !== null && <span className="text-muted ml-auto shrink-0 tabular-nums">{duration(row.ranMs)}</span>}
      </span>
      <span className="text-faint truncate text-[11px]">{facts.join(" · ")}</span>
    </button>
  );
}

function Segment<T>({ label, value, onChange, options }: { label: string; value: T; onChange: (value: T) => void; options: [T, ReactNode][] }) {
  return (
    <div role="group" aria-label={label} className="bg-inset flex items-center gap-0.5 rounded-md p-0.5">
      {options.map(([option, text]) => (
        <button
          key={String(option)}
          type="button"
          aria-pressed={option === value}
          onClick={() => onChange(option)}
          className={`cursor-pointer rounded px-2 py-0.5 text-[11px] touch:min-h-11 ${option === value ? "bg-raised text-body" : "text-muted hover:text-body"}`}
        >
          {text}
        </button>
      ))}
    </div>
  );
}
