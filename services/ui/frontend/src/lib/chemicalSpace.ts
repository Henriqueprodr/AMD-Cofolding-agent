// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * Every molecule a conversation made, for the map on the Campaign tab: the
 * molecules of all its runs pooled, each once, with the agent that made it.
 */

import { CONDUCTOR } from "./fleet";
import { TOTAL, type Molecules } from "./runAnalysis";

/** What one run brings: its molecules, each once, with its best total. */
export interface RunMolecules {
  jobId: string;
  /** C for a run the conductor started, S1, S2… for a sub-agent's. */
  agent: string;
  smiles: readonly string[];
  /** Total score per molecule, NaN where it has none. */
  total: ArrayLike<number>;
  /** The step each molecule was made at, when the run said. */
  step?: ArrayLike<number>;
  lastStep: number;
}

export interface ConversationSpace extends Pick<Molecules, "n" | "smiles" | "values" | "lastStep"> {
  /** Who made each molecule first. */
  agent: string[];
  jobId: string[];
  /** Which run made it, 1 for the first run started. */
  run: number[];
  /** When it was made, as an order: its run's number plus how far through that run. */
  made: number[];
  /** Molecules per agent: C first, then S1, S2… */
  counts: [string, number][];
}

/** A run's molecules as its results table gives them. */
export function runMolecules(
  jobId: string,
  agent: string,
  molecules: Pick<Molecules, "n" | "smiles" | "values" | "lastStep"> & { step?: ArrayLike<number> },
): RunMolecules {
  return {
    jobId,
    agent,
    smiles: molecules.smiles,
    total: molecules.values.get(TOTAL) ?? new Float64Array(molecules.n).fill(NaN),
    step: molecules.step,
    lastStep: molecules.lastStep,
  };
}

/**
 * Pooled, once per SMILES, in the order given: pass the runs oldest first.
 *
 * A molecule two runs both made is the first one's, because which agent got
 * there first is what the map is for; the totals of runs scored against
 * different profiles are not comparable, so the better score could not say.
 * The best total is kept all the same, so a map of more molecules than it
 * can draw keeps the best of them.
 *
 * *lastStep* adds up the runs' steps, so the map is drawn again as any run
 * moves on and not only the one furthest along.
 */
export function mergeRuns(runs: readonly RunMolecules[]): ConversationSpace {
  const at = new Map<string, number>();
  const smiles: string[] = [];
  const totals: number[] = [];
  const agent: string[] = [];
  const jobId: string[] = [];
  const number: number[] = [];
  const made: number[] = [];
  let lastStep = 0;
  for (const [index, run] of runs.entries()) {
    lastStep += run.lastStep;
    run.smiles.forEach((s, i) => {
      const score = run.total[i];
      const seen = at.get(s);
      if (seen === undefined) {
        at.set(s, smiles.length);
        smiles.push(s);
        totals.push(score);
        agent.push(run.agent);
        jobId.push(run.jobId);
        number.push(index + 1);
        const step = run.step?.[i];
        made.push(index + 1 + (step !== undefined && step >= 0 ? step / (run.lastStep + 1) : 0));
      } else if (Number.isFinite(score) && !(totals[seen] >= score)) {
        totals[seen] = score;
      }
    });
  }
  const counts = new Map<string, number>();
  for (const label of agent) counts.set(label, (counts.get(label) ?? 0) + 1);
  return {
    n: smiles.length,
    smiles,
    values: new Map([[TOTAL, Float64Array.from(totals)]]),
    lastStep,
    agent,
    jobId,
    run: number,
    made,
    counts: [...counts].sort((a, b) => byAgent(a[0], b[0])),
  };
}

/** C, then S1, S2… by number, then anything else. */
export function byAgent(a: string, b: string): number {
  const rank = (label: string) => (label === CONDUCTOR ? 0 : /^S\d+$/.test(label) ? Number(label.slice(1)) : Number.MAX_SAFE_INTEGER);
  return rank(a) - rank(b) || a.localeCompare(b);
}

/** Told apart on light and dark alike, and by most colour-blind readers (Okabe and Ito). */
const PALETTE = ["#0072b2", "#e69f00", "#009e73", "#cc79a7", "#56b4e9", "#d55e00", "#b8a400", "#8c8c8c"];
const CONDUCTOR_COLOUR = "#7b5cc4";
const UNKNOWN_COLOUR = "#9a9aa0";

/** One colour per agent: a plain colour and not a theme variable, because the map is drawn on a canvas. */
export function agentColour(label: string): string {
  if (label === CONDUCTOR) return CONDUCTOR_COLOUR;
  const number = /^S(\d+)$/.exec(label);
  return number ? PALETTE[(Number(number[1]) - 1) % PALETTE.length] : UNKNOWN_COLOUR;
}
