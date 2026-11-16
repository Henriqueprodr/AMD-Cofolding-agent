// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import { agentColour, byAgent, mergeRuns, runMolecules, type RunMolecules } from "./chemicalSpace";
import { TOTAL } from "./runAnalysis";

const run = (jobId: string, agent: string, smiles: string[], total: number[], lastStep = 10): RunMolecules => ({
  jobId,
  agent,
  smiles,
  total: Float64Array.from(total),
  lastStep,
});

describe("mergeRuns", () => {
  it("pools every run's molecules, each once, with who made it", () => {
    const space = mergeRuns([run("job-1", "C", ["CCO", "CCN"], [0.2, 0.4]), run("job-2", "S1", ["c1ccccc1"], [0.6])]);

    expect(space.n).toBe(3);
    expect(space.smiles).toEqual(["CCO", "CCN", "c1ccccc1"]);
    expect(space.agent).toEqual(["C", "C", "S1"]);
    expect(space.jobId).toEqual(["job-1", "job-1", "job-2"]);
  });

  it("credits a molecule two runs made to the one that made it first, keeping the better score", () => {
    // Scored against different profiles, the two totals say nothing about
    // who did better; who got there first is what the map shows.
    const space = mergeRuns([run("job-1", "S1", ["CCO"], [0.2]), run("job-2", "S2", ["CCO"], [0.9])]);

    expect(space.n).toBe(1);
    expect(space.agent).toEqual(["S1"]);
    expect(Array.from(space.values.get(TOTAL)!)).toEqual([0.9]);
  });

  it("keeps a score where the first run had none", () => {
    const space = mergeRuns([run("job-1", "S1", ["CCO"], [NaN]), run("job-2", "S2", ["CCO"], [0.5])]);

    expect(Array.from(space.values.get(TOTAL)!)).toEqual([0.5]);
  });

  it("counts the molecules per agent, the conductor first and the sub-agents by number", () => {
    const space = mergeRuns([
      run("a", "S10", ["C1"], [1]),
      run("b", "S2", ["C2", "C3"], [1, 1]),
      run("c", "C", ["C4"], [1]),
      run("d", "S1", ["C5"], [1]),
    ]);

    expect(space.counts).toEqual([
      ["C", 1],
      ["S1", 1],
      ["S2", 2],
      ["S10", 1],
    ]);
  });

  it("adds up the runs' steps, so the map is redrawn as any of them moves on", () => {
    expect(mergeRuns([run("a", "C", ["C"], [1], 12), run("b", "S1", ["N"], [1], 30)]).lastStep).toBe(42);
  });

  it("numbers the runs in the order given and orders the molecules by when they were made", () => {
    const space = mergeRuns([
      { ...run("a", "C", ["C1", "C2"], [1, 1], 9), step: [0, 9] },
      { ...run("b", "S1", ["C3"], [1], 4), step: [2] },
    ]);

    expect(space.run).toEqual([1, 1, 2]);
    expect(space.made[0]).toBeLessThan(space.made[1]);
    expect(space.made[1]).toBeLessThan(space.made[2]);
  });

  it("is empty for no runs", () => {
    expect(mergeRuns([])).toMatchObject({ n: 0, smiles: [], counts: [] });
  });
});

describe("runMolecules", () => {
  it("reads the totals off the run's molecules, and NaN for a run that has none", () => {
    const scored = runMolecules("job-1", "S1", { n: 1, smiles: ["CCO"], values: new Map([[TOTAL, Float64Array.from([0.3])]]), lastStep: 4 });
    const unscored = runMolecules("job-2", "C", { n: 2, smiles: ["CCO", "CCN"], values: new Map(), lastStep: 1 });

    expect(Array.from(scored.total)).toEqual([0.3]);
    expect(Array.from(unscored.total).every(Number.isNaN)).toBe(true);
  });
});

describe("byAgent", () => {
  it("puts the conductor first, then the sub-agents by number, then the unclaimed", () => {
    expect(["S", "S11", "C", "S2"].sort(byAgent)).toEqual(["C", "S2", "S11", "S"]);
  });
});

describe("agentColour", () => {
  it("gives each agent its own colour and the conductor one of its own", () => {
    const colours = ["C", "S1", "S2", "S3"].map(agentColour);

    expect(new Set(colours).size).toBe(4);
    expect(agentColour("S1")).toBe(agentColour("S1"));
  });
});
