// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useMemo, useRef, useState, type PointerEvent } from "react";
import type { Theme } from "../../hooks/useTheme";
import { agentColour, type ConversationSpace } from "../../lib/chemicalSpace";
import { colourScale, formatValue, MIN_MAP_POINTS, pointDensity, scaleCss, TOTAL } from "../../lib/runAnalysis";
import { MolDepiction } from "../MolDepiction/MolDepiction";
import { useWidth } from "../RunAnalysis/plotUtils";
import { Chip, Placeholder } from "../RunAnalysis/shared";
import { useUmap } from "../RunAnalysis/useUmap";

interface Props {
  /** The conversation, which the map is computed and kept for. */
  sessionId: string | null;
  /** Names the part of the conversation mapped, whose layout is kept apart; null for all of it. */
  scope?: string | null;
  space: ConversationSpace | null;
  /** Some runs' results are still being read. */
  loading: boolean;
  theme: Theme;
  onOpenMolecule?: (smiles: string) => void;
}

const PAD = 10;
const HOVER_PX = 8;
const NAN_COLOUR = "rgba(140,140,150,0.55)";

type ColourBy = "score" | "density" | "agent" | "run" | "made";

const COLOUR_BY: { id: ColourBy; label: string; ends?: [string, string] }[] = [
  { id: "score", label: "Score", ends: ["low", "high"] },
  { id: "density", label: "Density", ends: ["sparse", "dense"] },
  { id: "agent", label: "Agent" },
  { id: "run", label: "Run", ends: ["first run", "last run"] },
  { id: "made", label: "Time", ends: ["earlier", "later"] },
];

/**
 * Every molecule of the conversation on one map of their fingerprints,
 * coloured by score, by how crowded that part of the map is, by the agent
 * or the run that made it, or by when: where the search went, and where
 * the agents' work overlapped. Hover a point for its structure, click it to
 * open the molecule; the chips hide and show one agent's.
 */
export function ConversationMap({ sessionId, scope = null, space, loading, theme, onOpenMolecule }: Props) {
  const { layout, working } = useUmap(sessionId && space ? `conversation:${sessionId}${scope ? `:${scope}` : ""}` : null, space);
  const [measure, width] = useWidth<HTMLDivElement>();
  const canvas = useRef<HTMLCanvasElement>(null);
  const [hidden, setHidden] = useState<ReadonlySet<string>>(() => new Set());
  const [hover, setHover] = useState<number | null>(null);
  const [colourBy, setColourBy] = useState<ColourBy>("score");
  const height = Math.round(Math.min(360, Math.max(220, width * 0.7)));
  const dark = theme === "dark";

  // Points of the layout that are still molecules of the conversation.
  const points = useMemo(() => {
    if (!layout?.smiles.length || !space) return null;
    const at = new Map(space.smiles.map((s, i) => [s, i]));
    const index: number[] = [];
    const xs: number[] = [];
    const ys: number[] = [];
    layout.smiles.forEach((s, i) => {
      const m = at.get(s);
      if (m === undefined) return;
      index.push(m);
      xs.push(layout.coords[2 * i]);
      ys.push(layout.coords[2 * i + 1]);
    });
    return index.length ? { index, xs, ys } : null;
  }, [layout, space]);

  const screen = useMemo(() => {
    if (!points || !width) return null;
    const [x0, x1] = [Math.min(...points.xs), Math.max(...points.xs)];
    const [y0, y1] = [Math.min(...points.ys), Math.max(...points.ys)];
    const scale = Math.min((width - 2 * PAD) / (x1 - x0 || 1), (height - 2 * PAD) / (y1 - y0 || 1));
    const ox = (width - (x1 - x0) * scale) / 2;
    const oy = (height - (y1 - y0) * scale) / 2;
    return { px: points.xs.map((x) => ox + (x - x0) * scale), py: points.ys.map((y) => oy + (y1 - y) * scale) };
  }, [points, width, height]);

  const colouring = useMemo(() => {
    if (!points || !space) return null;
    if (colourBy === "agent") return { css: points.index.map((m) => agentColour(space.agent[m])), gradient: "" };
    const total = space.values.get(TOTAL);
    const raw =
      colourBy === "density"
        ? pointDensity(points.xs, points.ys)
        : Float64Array.from(points.index, (m) => (colourBy === "score" ? (total?.[m] ?? NaN) : colourBy === "run" ? space.run[m] : space.made[m]));
    const scale = colourScale(raw, "linear", { direction: "up" });
    const css = Array.from(raw, (value) => {
      const p = scale ? scale.position(value) : NaN;
      if (!scale || !Number.isFinite(p)) return NAN_COLOUR;
      const [r, g, b] = scale.paint(p, dark);
      return `rgb(${r},${g},${b})`;
    });
    return { css, gradient: scale ? scaleCss(scale, dark) : "" };
  }, [points, space, colourBy, dark]);

  useEffect(() => {
    const element = canvas.current;
    if (!element || !screen || !points || !space || !colouring) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    element.width = Math.round(width * dpr);
    element.height = Math.round(height * dpr);
    const ctx = element.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const radius = points.index.length > 1500 ? 2 : 2.6;
    ctx.globalAlpha = 0.8;
    points.index.forEach((m, i) => {
      if (hidden.has(space.agent[m])) return;
      ctx.fillStyle = colouring.css[i];
      ctx.beginPath();
      ctx.arc(screen.px[i], screen.py[i], radius, 0, Math.PI * 2);
      ctx.fill();
    });
    ctx.globalAlpha = 1;
    if (hover !== null) {
      ctx.strokeStyle = dark ? "#f4f4f5" : "#0a0a0b";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(screen.px[hover], screen.py[hover], radius + 3.5, 0, Math.PI * 2);
      ctx.stroke();
    }
  }, [screen, points, space, colouring, hidden, hover, width, height, dark]);

  if (!space) {
    return (
      <div ref={measure}>
        <Placeholder height={120}>
          {loading ? "Reading the runs' molecules…" : "No molecules yet. Every run's molecules are mapped here together."}
        </Placeholder>
      </div>
    );
  }

  const nearest = (event: PointerEvent<HTMLDivElement>) => {
    if (!screen || !points) return null;
    const rect = event.currentTarget.getBoundingClientRect();
    const [x, y] = [event.clientX - rect.left, event.clientY - rect.top];
    let best: number | null = null;
    let bestD = HOVER_PX * HOVER_PX;
    for (let i = 0; i < screen.px.length; i++) {
      if (hidden.has(space.agent[points.index[i]])) continue;
      const d = (screen.px[i] - x) ** 2 + (screen.py[i] - y) ** 2;
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    return best;
  };

  // A point hovered on the previous map may be past the end of this one.
  const shown = hover !== null && screen && points && hover < points.index.length ? { m: points.index[hover], x: screen.px[hover], y: screen.py[hover] } : null;
  const total = shown ? (space.values.get(TOTAL)?.[shown.m] ?? NaN) : NaN;
  const toggle = (agent: string) =>
    setHidden((previous) => {
      const next = new Set(previous);
      if (!next.delete(agent)) next.add(agent);
      return next;
    });

  const mode = COLOUR_BY.find((one) => one.id === colourBy)!;
  return (
    <div className="min-w-0">
      <div className="mb-1 flex items-center gap-2 text-[11px]">
        <label className="text-faint flex items-center gap-1.5">
          Colour by
          <select
            value={colourBy}
            onChange={(event) => setColourBy(event.target.value as ColourBy)}
            className="bg-sunken border-subtle text-body cursor-pointer rounded border px-1 py-0.5 text-[11px] outline-none"
          >
            {COLOUR_BY.map((one) => (
              <option key={one.id} value={one.id}>
                {one.label}
              </option>
            ))}
          </select>
        </label>
        {mode.ends && colouring?.gradient && (
          <span className="text-faint ml-auto flex items-center gap-1.5 text-[10px]">
            {mode.ends[0]}
            <span className="h-1.5 w-24 rounded-full" style={{ background: colouring.gradient }} />
            {mode.ends[1]}
          </span>
        )}
      </div>
      <div
        ref={measure}
        onPointerMove={(event) => setHover(nearest(event))}
        onPointerLeave={() => setHover(null)}
        onClick={() => shown && onOpenMolecule?.(space.smiles[shown.m])}
        className={`relative overflow-hidden rounded-lg select-none ${shown ? "cursor-pointer" : ""}`}
        style={{ height }}
        role="img"
        aria-label={`Map of the conversation's ${space.n} molecules by structure, coloured by ${mode.label.toLowerCase()}`}
      >
        {points && screen ? (
          <canvas ref={canvas} style={{ width, height }} className="block" />
        ) : (
          <Placeholder height={height}>
            {space.n < MIN_MAP_POINTS ? "Too few molecules for a map yet." : layout?.failed && !working ? layout.failed : "Mapping the molecules…"}
          </Placeholder>
        )}
        {(working || loading) && <span className="absolute top-1 right-1 size-1.5 animate-pulse rounded-full" style={{ background: "var(--color-structure)" }} />}
        {shown && (
          <div
            className="bg-raised border-subtle shadow-card pointer-events-none absolute z-10 rounded-lg border p-1"
            style={{ left: Math.min(Math.max(0, shown.x + 12), width - 148), top: shown.y > height - 126 ? shown.y - 126 : shown.y + 12 }}
          >
            <MolDepiction smiles={space.smiles[shown.m]} theme={theme} width={140} height={96} className="h-24 w-[140px]" />
            <div className="text-muted flex justify-between gap-2 px-1 text-[10px] tabular-nums">
              <span className="font-semibold">
                {space.agent[shown.m]} · run {space.run[shown.m]}
              </span>
              {Number.isFinite(total) && <span className="text-body font-mono">score {formatValue(total, 2)}</span>}
            </div>
          </div>
        )}
      </div>
      <div className="mt-1 flex flex-wrap items-center gap-x-1 gap-y-0.5">
        {space.counts.map(([agent, count]) => (
          <Chip
            key={agent}
            label={`${agent} · ${count}`}
            colour={colourBy === "agent" ? agentColour(agent) : "var(--text-muted)"}
            on={!hidden.has(agent)}
            onClick={() => toggle(agent)}
          />
        ))}
      </div>
    </div>
  );
}
