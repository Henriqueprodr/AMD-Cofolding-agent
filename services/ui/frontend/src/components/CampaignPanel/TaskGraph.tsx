// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useMemo, useState } from "react";
import type { FleetTask } from "../../types/api";
import { campaignGraph, type GraphNode, type TaskPhase } from "../../lib/fleet";
import { useWidth } from "../RunAnalysis/plotUtils";
import { Placeholder } from "../RunAnalysis/shared";

interface Props {
  tasks: readonly FleetTask[];
  /** Highlighted, and drawn with its lineage picked out. */
  selected: string | null;
  onSelect: (taskId: string | null) => void;
  /** Placed where "now" is on the axis; the campaign's clock, not the browser's. */
  now: number;
}

const ROW_H = 46;
const PAD = { left: 10, right: 10, top: 14, bottom: 18 };
const DOT = 5.5;
/** Room for the label to the right of a dot before the next one. */
const LABEL_W = 132;

const PHASE_COLOUR: Record<TaskPhase, string> = {
  done: "var(--color-ok)",
  running: "var(--color-structure)",
  failed: "var(--color-accent)",
  pending: "var(--text-faint)",
};

const PHASE_WORD: Record<TaskPhase, string> = {
  done: "finished",
  running: "running",
  failed: "failed",
  pending: "waiting for an agent",
};

/**
 * The campaign read left to right over time.
 *
 * Each dot is one task, placed where it started; the lines between them are
 * the lineage the conductor recorded, so a task written from what two
 * earlier ones showed has two lines coming into it. Rows carry no meaning
 * and are not per sub-agent: the agents are not independent, and a row each
 * would draw them as though they were.
 */
export function TaskGraph({ tasks, selected, onSelect, now }: Props) {
  const [box, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<string | null>(null);

  const w = Math.max(260, width);
  // Leave room on the right for the last node's label to sit beside its dot.
  const plotW = Math.max(60, w - PAD.left - PAD.right - LABEL_W);
  // Two nodes share a row only if the first one's label has finished before
  // the second one starts, so how wide a label is decides how many rows the
  // graph needs. The layout cannot know that; the panel can.
  const graph = useMemo(() => campaignGraph(tasks, now, LABEL_W / plotW), [tasks, now, plotW]);

  /** The selected task, what it came from and what came of it. */
  const related = useMemo(() => {
    const focus = hover ?? selected;
    if (!focus) return null;
    const kin = new Set<string>([focus]);
    for (const edge of graph.edges) {
      if (edge.to === focus) kin.add(edge.from);
      if (edge.from === focus) kin.add(edge.to);
    }
    return kin;
  }, [graph.edges, hover, selected]);

  if (!graph.nodes.length) {
    return <Placeholder height={120}>No tasks yet. They appear here as the conductor writes them.</Placeholder>;
  }

  const height = PAD.top + PAD.bottom + graph.rows * ROW_H;
  const xOf = (node: GraphNode) => PAD.left + node.x * plotW;
  const yOf = (node: GraphNode) => PAD.top + node.row * ROW_H + ROW_H / 2;
  const position = new Map(graph.nodes.map((node) => [node.task.task_id, node]));

  const [from, to] = graph.span;
  const clock = (at: number) => new Date(at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  return (
    <div ref={box}>
      {width > 0 && (
        <svg width={w} height={height} className="block select-none" role="img" aria-label={`${graph.nodes.length} tasks over time`}>
          {graph.edges.map((edge) => {
            const a = position.get(edge.from);
            const b = position.get(edge.to);
            if (!a || !b) return null;
            const [x1, y1, x2, y2] = [xOf(a), yOf(a), xOf(b), yOf(b)];
            // A flat curve, so an edge between two rows reads as a link
            // rather than as another task's path.
            const mid = (x1 + x2) / 2;
            const lit = related?.has(edge.from) && related.has(edge.to);
            return (
              <path
                key={`${edge.from}-${edge.to}`}
                d={`M${x1},${y1}C${mid},${y1} ${mid},${y2} ${x2},${y2}`}
                fill="none"
                stroke={lit ? "var(--color-accent)" : "var(--border-strong)"}
                strokeWidth={lit ? 1.6 : 1}
                opacity={related && !lit ? 0.25 : 0.7}
              />
            );
          })}
          {graph.nodes.map((node) => {
            const id = node.task.task_id;
            const [x, y] = [xOf(node), yOf(node)];
            const colour = PHASE_COLOUR[node.phase];
            const dim = related && !related.has(id) ? 0.3 : 1;
            return (
              <g
                key={id}
                opacity={dim}
                className="cursor-pointer transition-opacity"
                onPointerEnter={() => setHover(id)}
                onPointerLeave={() => setHover(null)}
                onClick={() => onSelect(selected === id ? null : id)}
              >
                <title>{`${node.label}\n${PHASE_WORD[node.phase]} · started ${clock(node.at)}`}</title>
                {id === selected && <circle cx={x} cy={y} r={DOT + 3.5} fill="none" stroke="var(--color-accent)" strokeWidth={1.4} />}
                {node.phase === "running" && <circle cx={x} cy={y} r={DOT + 2} fill={colour} opacity={0.22} />}
                <circle cx={x} cy={y} r={DOT} fill={node.phase === "pending" ? "var(--surface-raised)" : colour} stroke={colour} strokeWidth={1.4} />
                <text x={x + DOT + 5} y={y} dy="0.32em" fontSize={10.5} fill="var(--text-muted)">
                  {clip(node.label, 24)}
                </text>
              </g>
            );
          })}
          <text x={PAD.left} y={height - 5} fontSize={9.5} fill="var(--text-faint)">
            {clock(from)}
          </text>
          <text x={PAD.left + plotW} y={height - 5} textAnchor="end" fontSize={9.5} fill="var(--text-faint)">
            {clock(to)}
          </text>
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
