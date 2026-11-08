// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import type { FleetCampaign, FleetResult, FleetSupervisor, FleetTask } from "../types/api";
import {
  agentSummaries,
  campaignGraph,
  campaignProgress,
  objectiveProgress,
  pooledCandidates,
  taskPhase,
  ttpIdOf,
} from "./fleet";

const T0 = Date.parse("2026-11-08T10:00:00Z");
const minutes = (n: number) => new Date(T0 + n * 60_000).toISOString();

function task(id: string, over: Partial<FleetTask> = {}): FleetTask {
  return {
    task_id: id,
    campaign_id: "camp-1",
    spec: {},
    state: "done",
    attempts: 1,
    created_at: minutes(0),
    lease_owner: null,
    lease_expires_at: null,
    claimed_at: null,
    finished_at: null,
    error: null,
    job_id: null,
    ...over,
  };
}

function result(taskId: string, over: Partial<FleetResult> = {}): FleetResult {
  return {
    task_id: taskId,
    campaign_id: "camp-1",
    job_id: null,
    objective_profile: {},
    candidates: [],
    summary: null,
    posted_at: minutes(10),
    ...over,
  };
}

function supervisor(owner: string, over: Partial<FleetSupervisor> = {}): FleetSupervisor {
  return {
    owner,
    state: "idle",
    campaign_id: "camp-1",
    task_id: null,
    tasks_completed: 0,
    tasks_failed: 0,
    detail: null,
    first_seen_at: minutes(0),
    last_seen_at: minutes(1),
    last_seen_seconds_ago: 5,
    ...over,
  };
}

const campaign: FleetCampaign = {
  campaign_id: "camp-1",
  goal: "Optimise the JQ1 series",
  base_ttp_id: null,
  objectives: [],
  concurrency: 2,
  state: "open",
  created_at: minutes(0),
  dispatch_until: null,
  hard_stop_at: null,
  finalized_at: null,
  session_id: "sess-1",
};

describe("taskPhase", () => {
  it("reads the ledger's states", () => {
    expect(taskPhase({ state: "done" })).toBe("done");
    expect(taskPhase({ state: "claimed" })).toBe("running");
    expect(taskPhase({ state: "failed" })).toBe("failed");
    expect(taskPhase({ state: "queued" })).toBe("pending");
  });
});

describe("campaignProgress", () => {
  it("counts tasks, molecules and the agents still reporting", () => {
    const progress = campaignProgress({
      campaign,
      tasks: [task("a"), task("b", { state: "claimed" }), task("c", { state: "failed" }), task("d", { state: "queued" })],
      results: [result("a", { candidates: [{ smiles: "CCO" }, { smiles: "CCN" }] })],
      supervisors: [supervisor("s1"), supervisor("s2", { last_seen_seconds_ago: 600 })],
    });
    expect(progress).toMatchObject({ tasks: 4, done: 1, running: 1, failed: 1, pending: 1, molecules: 2, agentsLive: 1 });
    expect(progress.fraction).toBeCloseTo(0.5);
  });

  it("has no fraction before there is a task", () => {
    expect(campaignProgress({ campaign, tasks: [], results: [], supervisors: [] }).fraction).toBeNull();
  });
});

describe("campaignGraph", () => {
  it("places tasks left to right by time, not by agent", () => {
    const graph = campaignGraph(
      [
        task("a", { claimed_at: minutes(0), lease_owner: "s1" }),
        task("b", { claimed_at: minutes(60), lease_owner: "s2" }),
        task("c", { claimed_at: minutes(30), lease_owner: "s1" }),
      ],
      T0 + 60 * 60_000,
    );
    const x = (id: string) => graph.nodes.find((n) => n.task.task_id === id)!.x;
    expect(x("a")).toBeCloseTo(0);
    expect(x("c")).toBeCloseTo(0.5);
    expect(x("b")).toBeCloseTo(1);
    // Two tasks by the same agent are not forced into one row, and two by
    // different agents are not forced apart: rows are packing only.
    expect(graph.nodes.every((n) => n.row === 0)).toBe(true);
  });

  it("packs overlapping tasks into separate rows", () => {
    const graph = campaignGraph(
      [task("a", { claimed_at: minutes(0) }), task("b", { claimed_at: minutes(1) }), task("c", { claimed_at: minutes(60) })],
      T0 + 60 * 60_000,
    );
    const row = (id: string) => graph.nodes.find((n) => n.task.task_id === id)!.row;
    expect(row("a")).toBe(0);
    expect(row("b")).toBe(1);
    expect(row("c")).toBe(0);
    expect(graph.rows).toBe(2);
  });

  it("gives a row to each task when the labels are wide enough to collide", () => {
    const tasks = [0, 20, 40].map((at, i) => task(`t${i}`, { claimed_at: minutes(at) }));
    // Labels wider than the space between neighbouring tasks, so the middle
    // one has to drop to a row of its own.
    const graph = campaignGraph(tasks, T0 + 40 * 60_000, 0.6);
    expect(graph.rows).toBe(2);
    expect(graph.nodes.map((n) => n.row)).toEqual([0, 1, 0]);
  });

  it("draws the lineage the conductor recorded", () => {
    const graph = campaignGraph(
      [task("a"), task("b"), task("c", { spec: { intent: "merge", derived_from: ["a", "b"] } })],
      T0,
    );
    expect(graph.edges).toEqual([
      { from: "a", to: "c" },
      { from: "b", to: "c" },
    ]);
  });

  it("drops a parent that is not in this campaign, which would point at nothing", () => {
    const graph = campaignGraph([task("a", { spec: { derived_from: ["gone", "a"] } })], T0);
    expect(graph.edges).toEqual([]);
  });

  it("labels a node with the intent, falling back to the rationale", () => {
    const graph = campaignGraph(
      [task("a", { spec: { intent: "Hold the JQ1 scaffold" } }), task("b", { spec: { rationale: "try the other pocket" } })],
      T0,
    );
    expect(graph.nodes.map((n) => n.label)).toEqual(["Hold the JQ1 scaffold", "try the other pocket"]);
  });

  it("puts a lone task at the left rather than dividing by a zero span", () => {
    const graph = campaignGraph([task("a", { claimed_at: minutes(0) })], T0);
    expect(graph.nodes[0].x).toBe(0);
    expect(Number.isFinite(graph.nodes[0].x)).toBe(true);
  });

  it("is empty for a campaign with no tasks", () => {
    expect(campaignGraph([], T0)).toMatchObject({ nodes: [], edges: [], rows: 0 });
  });
});

describe("pooledCandidates", () => {
  it("pools every task's molecules, best first", () => {
    const pooled = pooledCandidates([
      result("a", { candidates: [{ smiles: "CCO", score: 0.4 }] }),
      result("b", { candidates: [{ smiles: "CCN", score: 0.9 }] }),
    ]);
    expect(pooled.map((c) => c.smiles)).toEqual(["CCN", "CCO"]);
    expect(pooled[0].taskId).toBe("b");
  });

  it("keeps the better score when two tasks find the same molecule", () => {
    const pooled = pooledCandidates([
      result("a", { candidates: [{ smiles: "CCO", score: 0.4 }] }),
      result("b", { candidates: [{ smiles: "CCO", score: 0.8 }] }),
    ]);
    expect(pooled).toHaveLength(1);
    expect(pooled[0]).toMatchObject({ score: 0.8, taskId: "b" });
  });

  it("names the sub-agent that found each molecule", () => {
    const pooled = pooledCandidates([result("a", { candidates: [{ smiles: "CCO" }] })], new Map([["a", "s1"]]));
    expect(pooled[0].owner).toBe("s1");
  });

  it("skips entries with no SMILES rather than listing a blank row", () => {
    expect(pooledCandidates([result("a", { candidates: [{ smiles: "  " }, { smiles: "CCO" }] })])).toHaveLength(1);
  });
});

describe("agentSummaries", () => {
  it("credits each agent with the molecules from the tasks it ran", () => {
    const [agent] = agentSummaries({
      tasks: [task("a", { lease_owner: "s1" })],
      results: [result("a", { candidates: [{ smiles: "CCO", score: 0.3 }, { smiles: "CCN", score: 0.7 }] })],
      supervisors: [supervisor("s1", { tasks_completed: 1 })],
    });
    expect(agent).toMatchObject({ owner: "s1", molecules: 2, best: 0.7, live: true });
  });

  it("keeps an agent that has gone but did work, since its results are still pooled", () => {
    const summaries = agentSummaries({
      tasks: [],
      results: [],
      supervisors: [supervisor("s1", { state: "gone", tasks_completed: 2 }), supervisor("s2", { state: "gone" })],
    });
    expect(summaries.map((a) => a.owner)).toEqual(["s1"]);
  });

  it("lists the live agents first", () => {
    const summaries = agentSummaries({
      tasks: [],
      results: [],
      supervisors: [supervisor("s1", { state: "gone", tasks_completed: 9 }), supervisor("s2", { tasks_completed: 1 })],
    });
    expect(summaries.map((a) => a.owner)).toEqual(["s2", "s1"]);
  });
});

describe("objectiveProgress", () => {
  it("finds the best and the mean per objective, and what moved the best", () => {
    const [qed] = objectiveProgress([
      result("a", { objective_profile: { QED: 0.4 }, posted_at: minutes(1) }),
      result("b", { objective_profile: { QED: 0.8 }, posted_at: minutes(2) }),
      result("c", { objective_profile: { QED: 0.6 }, posted_at: minutes(3) }),
    ]);
    expect(qed).toMatchObject({ name: "QED", best: 0.8, bestTask: "b", tasks: 3 });
    expect(qed.mean).toBeCloseTo(0.6);
    expect(qed.delta).toBeCloseTo(0.4);
  });

  it("has no delta when one task scored the objective", () => {
    expect(objectiveProgress([result("a", { objective_profile: { QED: 0.5 } })])[0].delta).toBeNull();
  });

  it("reads the objectives off the results when the campaign declares none", () => {
    const progress = objectiveProgress([result("a", { objective_profile: { QED: 0.5, MW: 0.2 } })], []);
    expect(progress.map((o) => o.name).sort()).toEqual(["MW", "QED"]);
  });

  it("does not invent a row for a declared objective nothing has scored", () => {
    expect(objectiveProgress([], ["QED"])).toEqual([]);
  });
});

describe("ttpIdOf", () => {
  it("derives the variant id when the campaign has a base TTP", () => {
    expect(ttpIdOf({ base_ttp_id: "brd4-v1" }, "abc")).toBe("brd4-v1-abc");
  });

  it("uses the authored id when each task writes its own", () => {
    expect(ttpIdOf({ base_ttp_id: null }, "abc")).toBe("ttp-abc");
  });
});
