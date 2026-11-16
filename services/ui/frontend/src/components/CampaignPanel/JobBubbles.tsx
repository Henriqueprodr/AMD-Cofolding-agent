// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useMemo, useState } from "react";
import type { CampaignDetail } from "../../types/api";
import { campaignBubbleId, jobBubbles, type Bubble, type OwnRun, type TaskPhase } from "../../lib/fleet";
import { useWidth } from "../RunAnalysis/plotUtils";
import { Placeholder } from "../RunAnalysis/shared";

interface Props {
  /** Every campaign of this conversation, with its tasks and results. */
  campaigns: readonly Pick<CampaignDetail, "campaign" | "tasks" | "results">[];
  /** The runs the conversation started itself, outside any campaign. */
  runs: readonly OwnRun[];
  /** Placed where "now" is on the axis; passed in so the plot does not chase the clock. */
  now: number;
  /** What is open below, drawn with a ring around it; a campaign with no task is the campaign itself. */
  selected: { jobId: string | null; campaignId: string | null; taskId: string | null };
  onSelect: (bubble: Bubble) => void;
}

const ROW_H = 30;
/** The plot is always this many rows tall, so the tab does not jump as work is added. */
const ROWS = 3;
/** Wide enough on the left for a line bowed out past the first bubble. */
const PAD = { left: 26, right: 10, top: 8, bottom: 8 };
/** Room for "S12" inside. */
const R = 9;
/** Room for the label to the right of a bubble before the next one. */
const LABEL_W = 116;
/** Characters of a label that fit in LABEL_W at its font size. */
const LABEL_CHARS = 20;
/** A bubble, its label and a margin: what one row needs before the next bubble on it. */
const SLOT = 2 * R + 5 + LABEL_W + 14;
/**
 * Closest two neighbours get. Consecutive bubbles go on consecutive rows, so
 * a row's next bubble is ROWS steps on; past that the plot scrolls sideways.
 */
const MIN_STEP = SLOT / ROWS;
const HEIGHT = PAD.top + PAD.bottom + ROWS * ROW_H;
/** The workbench's, which the plot sits on. */
const BACKGROUND = "var(--surface-raised)";

/** Dashed while the work is not finished, solid once it is: the border is the only state drawn. */
const DASH: Record<TaskPhase, string | undefined> = {
  done: undefined,
  failed: undefined,
  running: "3 2",
  pending: "1.5 2.5",
};

/**
 * Everything the conversation set going, as bubbles in the order it started.
 *
 * The overview the Campaign tab opens with: the runs the agent you are
 * talking to started itself and the tasks it handed to sub-agents, side by
 * side, because they are one body of work and nowhere else shows both. The
 * letter says who did it, C or the sub-agent's number; the lines run from
 * the conductor and from the results it wrote each task on. Grey
 * throughout, except a failure. Clicking one opens it below.
 */
export function JobBubbles({ campaigns, runs, now, selected, onSelect }: Props) {
  const [box, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<string | null>(null);

  const count = useMemo(() => {
    const delegated = new Set(campaigns.flatMap((detail) => detail.tasks.map((task) => task.job_id)));
    return campaigns.reduce((sum, detail) => sum + 1 + detail.tasks.length, 0) + runs.filter((run) => !delegated.has(run.id)).length;
  }, [campaigns, runs]);
  // Leave room on the right for the last bubble's label to sit beside it.
  const plotW = Math.max(60, Math.max(260, width) - PAD.left - PAD.right - LABEL_W, (count - 1) * MIN_STEP);
  const w = plotW + PAD.left + PAD.right + LABEL_W;
  // Two bubbles share a row only if the first one's label has finished
  // before the second one starts, so how wide a label is decides how many
  // rows are needed. The layout cannot know that; the panel can. A hair under
  // a slot, so rounding never pushes a bubble onto a fourth row.
  const field = useMemo(() => jobBubbles(campaigns, runs, now, (SLOT - 0.5) / plotW), [campaigns, runs, now, plotW]);

  const open = useMemo(() => {
    const found = field.bubbles.find(
      (bubble) =>
        (selected.jobId !== null && bubble.jobId === selected.jobId) ||
        (selected.taskId !== null && bubble.taskId === selected.taskId) ||
        (selected.taskId === null && selected.campaignId !== null && bubble.id === campaignBubbleId(selected.campaignId)),
    );
    return found?.id ?? null;
  }, [field.bubbles, selected.jobId, selected.campaignId, selected.taskId]);

  /** The bubble under the pointer or shown below, with what it came from and what came of it. */
  const related = useMemo(() => {
    const focus = hover ?? open;
    if (!focus) return null;
    const kin = new Set<string>([focus]);
    for (const edge of field.edges) {
      if (edge.to === focus) kin.add(edge.from);
      if (edge.from === focus) kin.add(edge.to);
    }
    return kin;
  }, [field.edges, hover, open]);

  if (!field.bubbles.length) {
    return <Placeholder height={HEIGHT}>Nothing has been set going yet. Runs and the tasks of a campaign appear here as they start.</Placeholder>;
  }

  const height = HEIGHT;
  // Fewer rows than there is room for sit in the middle rather than at the top.
  const offset = (Math.max(0, ROWS - field.rows) * ROW_H) / 2;
  const xOf = (bubble: Bubble) => PAD.left + bubble.x * plotW;
  const yOf = (bubble: Bubble) => PAD.top + offset + bubble.row * ROW_H + ROW_H / 2;
  const place = new Map(field.bubbles.map((bubble) => [bubble.id, bubble]));

  return (
    <div ref={box} className="min-w-0">
      {width > 0 && (
        <svg width={w} height={height} className="block select-none" role="img" aria-label={`${field.bubbles.length} runs and tasks, in the order they started`}>
          {field.edges.map((edge) => {
            const a = place.get(edge.from);
            const b = place.get(edge.to);
            if (!a || !b) return null;
            // Labels run along the rows, so lines run between them: out of
            // the bubble's bottom (its top, to a row above), along the gap
            // next to its row, and into the other bubble from that side.
            const [x1, y1, x2, y2] = [xOf(a), yOf(a), xOf(b), yOf(b)];
            const side = y2 < y1 ? -1 : 1;
            const gap = y1 + (side * ROW_H) / 2;
            const into = y2 < gap ? 1 : -1;
            const turn = Math.min(6, Math.abs(x2 - x1) / 2) * Math.sign(x2 - x1);
            const d =
              `M${x1},${y1 + side * R}L${x1},${gap - side * 6}Q${x1},${gap} ${x1 + turn},${gap}` +
              `L${x2 - turn},${gap}Q${x2},${gap} ${x2},${gap - into * 6}L${x2},${y2 + into * R}`;
            const lit = related?.has(edge.from) && related.has(edge.to);
            return (
              <path
                key={`${edge.from}-${edge.to}`}
                d={d}
                fill="none"
                stroke={lit ? "var(--text-muted)" : "var(--text-faint)"}
                strokeWidth={lit ? 1.5 : 1}
                strokeLinecap="round"
                opacity={lit ? 0.9 : related ? 0.18 : 0.42}
              />
            );
          })}
          {field.bubbles.map((bubble) => {
            const [x, y, r] = [xOf(bubble), yOf(bubble), R];
            const ring = bubble.phase === "failed" ? "var(--color-accent)" : "var(--text-muted)";
            const dim = related && !related.has(bubble.id) ? 0.3 : 1;
            const shape = { fill: BACKGROUND, stroke: ring, strokeWidth: 1.3, strokeDasharray: DASH[bubble.phase] };
            return (
              <g
                key={bubble.id}
                opacity={dim}
                className="cursor-pointer transition-opacity"
                onPointerEnter={() => setHover(bubble.id)}
                onPointerLeave={() => setHover(null)}
                onClick={() => onSelect(bubble)}
              >
                {bubble.launcher === "conductor" ? (
                  // A square for the work the agent you are talking to did
                  // itself, a circle for a sub-agent's: at a glance, how much
                  // of the campaign was delegated.
                  <rect x={x - r} y={y - r} width={r * 2} height={r * 2} rx={2.5} {...shape} />
                ) : (
                  <circle cx={x} cy={y} r={r} {...shape} />
                )}
                <text
                  x={x}
                  y={y}
                  dy="0.34em"
                  textAnchor="middle"
                  fontSize={bubble.agent.length > 2 ? 7 : 8.5}
                  fontWeight={600}
                  fill="var(--text-muted)"
                  className="pointer-events-none"
                >
                  {bubble.agent}
                </text>
                {bubble.id === open && <circle cx={x} cy={y} r={r + 4} fill="none" stroke="var(--text-body)" strokeWidth={1.2} />}
                {/* Outlined in the background, so a line passing behind does not cut through the words. */}
                <text x={x + r + 5} y={y} dy="0.32em" fontSize={10.5} fill="var(--text-muted)" stroke={BACKGROUND} strokeWidth={3} paintOrder="stroke">
                  {clip(bubble.label, LABEL_CHARS)}
                </text>
              </g>
            );
          })}
        </svg>
      )}
    </div>
  );
}

/** *text* to at most *max* characters, on a word boundary where there is one. */
function clip(text: string, max: number): string {
  if (text.length <= max) return text;
  const cut = text.slice(0, max);
  const space = cut.lastIndexOf(" ");
  return `${space > max * 0.6 ? cut.slice(0, space) : cut}…`;
}
