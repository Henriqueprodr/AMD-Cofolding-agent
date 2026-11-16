// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useMemo } from "react";
import { branches, columns, type Step, type StepKind } from "../../lib/activity";
import { relativeTime } from "../../lib/campaign";
import { CONDUCTOR } from "../../lib/fleet";
import { Placeholder } from "../RunAnalysis/shared";

interface Props {
  /** Newest first, as `history` returns them. */
  steps: readonly Step[];
  /** The opening of each task's report, by task id: what came back, shown under the step that sent it. */
  reports: ReadonlyMap<string, string>;
  /** Running tasks whose agent keeps resending a failing call, with that call; said on the task's newest step. */
  stuck: ReadonlyMap<string, string>;
  now: number;
  /** Open what a step was about: the sub-agent's task, or the run it started. */
  onOpenTask: (step: Step) => void;
  onOpenRun: (jobId: string) => void;
}

const LANE = 14;
const R = 3.5;
/** A row of one line; one with what came back under it is taller. */
const ROW_H = 24;
const TALL_H = 52;

/** A node drawn hollow, because the step is a decision and not a result. */
const HOLLOW: ReadonlySet<StepKind> = new Set(["note", "handout", "campaign", "claim"]);

const LINE = "var(--text-faint)";
const NODE = "var(--text-muted)";

/**
 * The iteration graph: what happened, newest first, drawn the way a
 * repository draws its branches.
 *
 * The conductor's lane on the left and the tasks' beside it, laid out by
 * `columns`. Each task forks off the conductor where it was handed out, runs
 * up its column until it reports, and merges back where the conductor wrote
 * a task from what it found; one nothing built on just ends. What came back
 * is read on the row of the report, so the graph and the results are one
 * thing. Taller histories scroll inside the graph.
 */
export function IterationGraph({ steps, reports, stuck, now, onOpenTask, onOpenRun }: Props) {
  const forks = useMemo(() => branches(steps), [steps]);
  const laid = useMemo(() => columns(steps, forks), [steps, forks]);
  const stuckAt = useMemo(() => {
    const at = new Map<string, string>();
    for (const [taskId, name] of stuck) {
      const newest = steps.find((step) => step.taskId === taskId);
      if (newest) at.set(newest.id, name);
    }
    return at;
  }, [steps, stuck]);
  const layout = useMemo(() => {
    const tops: number[] = [];
    let y = 0;
    for (const step of steps) {
      tops.push(y);
      y += second(step, reports) ? TALL_H : ROW_H;
    }
    return { tops, height: y };
  }, [steps, reports]);

  if (!steps.length) {
    return <Placeholder height={64}>Nothing yet. What the conductor sets going and what the sub-agents send back appears here, step by step.</Placeholder>;
  }

  const x = (column: number) => column * LANE + LANE / 2;
  const y = (row: number) => layout.tops[row] + ROW_H / 2;
  const width = laid.count * LANE;
  const conductorRows = steps.flatMap((step, row) => (step.agent === CONDUCTOR ? [row] : []));
  /** From one row's node to the next row up: leaves and arrives upright, crossing sideways in between. */
  const curve = (x1: number, y1: number, x2: number, y2: number) => {
    const middle = (y1 + y2) / 2;
    return `M${x1},${y1}C${x1},${middle} ${x2},${middle} ${x2},${y2}`;
  };

  return (
    <div className="max-h-[360px] overflow-y-auto overscroll-contain">
      <ol className="relative flex flex-col">
        <svg width={width} height={layout.height} aria-hidden="true" className="pointer-events-none absolute top-0 left-0">
          {conductorRows.length > 1 && (
            <line x1={x(0)} x2={x(0)} y1={y(conductorRows[0])} y2={y(conductorRows[conductorRows.length - 1])} stroke={LINE} strokeWidth={1.2} />
          )}
          {forks.map((fork) => {
            const lane = x(laid.branch.get(fork.taskId) ?? 0);
            // Up its column from the row after the handout to the row before
            // its newest merge, its report, or the top while it is open.
            const bottom = fork.from === null ? y(fork.start) : y(fork.from - 1);
            const merges = fork.end === null ? [] : fork.into;
            const top = fork.end === null ? 0 : merges.length ? y(Math.min(...merges) + 1) : y(fork.end);
            return (
              <g key={fork.taskId} fill="none" stroke={LINE} strokeWidth={1.2}>
                <line x1={lane} x2={lane} y1={bottom} y2={top} strokeDasharray={fork.end === null ? "3 2" : undefined} />
                {fork.from !== null && <path d={curve(x(0), y(fork.from), lane, y(fork.from - 1))} />}
                {merges.map((row) => (
                  <path key={row} d={curve(lane, y(row + 1), x(0), y(row))} />
                ))}
              </g>
            );
          })}
          {steps.map((step, row) => {
            const failed = step.kind === "failure";
            const cx = x(laid.step[row]);
            return HOLLOW.has(step.kind) || failed ? (
              <circle key={step.id} cx={cx} cy={y(row)} r={R} fill="var(--surface-raised)" stroke={failed ? "var(--color-accent)" : NODE} strokeWidth={1.3} />
            ) : (
              <circle key={step.id} cx={cx} cy={y(row)} r={R} fill={NODE} />
            );
          })}
        </svg>
        {steps.map((step) => (
          <Row key={step.id} step={step} below={second(step, reports)} stuck={stuckAt.get(step.id) ?? null} now={now} gutter={width} onOpenTask={onOpenTask} onOpenRun={onOpenRun} />
        ))}
      </ol>
    </div>
  );
}

/** What a step has under its line: what came back with a report, why with a failure. */
function second(step: Step, reports: ReadonlyMap<string, string>): string | null {
  if (step.kind === "report" && step.taskId) return reports.get(step.taskId) || null;
  if (step.kind === "failure") return step.detail;
  return null;
}

function Row({
  step,
  below,
  stuck,
  now,
  gutter,
  onOpenTask,
  onOpenRun,
}: {
  step: Step;
  below: string | null;
  stuck: string | null;
  now: number;
  gutter: number;
  onOpenTask: (step: Step) => void;
  onOpenRun: (jobId: string) => void;
}) {
  const jobId = step.jobId;
  const open = jobId ? () => onOpenRun(jobId) : step.taskId ? () => onOpenTask(step) : null;
  // A step the ledger gave no time for; better no time than 1970.
  const at = step.at > 0 ? new Date(step.at).toISOString() : null;

  const body = (
    <>
      <span className="flex items-center gap-2" style={{ height: ROW_H }}>
        <span className="text-faint w-[18px] shrink-0 text-[10px] font-semibold tabular-nums">{step.agent}</span>
        <span className="min-w-0 flex-1 truncate text-[11.5px]">{step.text}</span>
        {stuck && (
          <span title={`Resending ${stuck}, which fails the same way each time`} className="text-muted shrink-0 text-[10px] italic">
            stuck
          </span>
        )}
        {at && (
          <time dateTime={at} className="text-faint shrink-0 text-[10px] tabular-nums">
            {relativeTime(at, now)}
          </time>
        )}
      </span>
      {below && <span className="text-muted -mt-1 line-clamp-2 pl-[26px] text-[11px] leading-snug">{below}</span>}
    </>
  );

  return (
    <li style={{ height: below ? TALL_H : ROW_H, paddingLeft: gutter + 4 }} className="min-w-0">
      {open ? (
        <button type="button" onClick={open} title={step.detail ?? undefined} className="hover:bg-inset block w-full cursor-pointer rounded text-left">
          {body}
        </button>
      ) : (
        <div title={step.detail ?? undefined}>{body}</div>
      )}
    </li>
  );
}
