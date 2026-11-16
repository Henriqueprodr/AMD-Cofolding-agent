// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import type { CampaignMetrics, FleetCampaign, FleetResult, FleetSupervisor, FleetTask } from "../types/api";
import {
  agentLabels,
  agentPage,
  agentSummaries,
  bestBySmiles,
  campaignProgress,
  campaignRuns,
  campaignStanding,
  CONDUCTOR,
  CONDUCTOR_OWNER,
  jobBubbles,
  knownAgent,
  objectiveProgress,
  pooledCandidates,
  reportBody,
  taskAgent,
  taskJobs,
  taskOwners,
  taskPhase,
  taskReports,
  teamLoad,
  ttpIdOf,
  UNCLAIMED,
  workingAgents,
  type OwnRun,
  type PooledCandidate,
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

describe("campaignStanding", () => {
  const standing = (tasks: FleetTask[], supervisors: FleetSupervisor[], c = campaign) =>
    campaignStanding({ campaign: c, tasks, results: [], supervisors });

  it("is running while a task is", () => {
    expect(standing([task("a", { state: "claimed" })], [])).toBe("running");
  });

  it("is running while a sub-agent is reporting, before it claims anything", () => {
    expect(standing([task("a", { state: "queued" })], [supervisor("s1")])).toBe("running");
  });

  it("is waiting when work is queued and nobody is there to take it", () => {
    expect(standing([task("a", { state: "queued" })], [supervisor("s1", { last_seen_seconds_ago: 600 })])).toBe("stalled");
  });

  it("is stopped when every task is settled and no agent is left", () => {
    // The BRD4 campaign: both tasks failed, the conductor died with the
    // allocation, and the ledger still said "dispatching" five hours on.
    const dispatching = { ...campaign, state: "dispatching" };
    const dead = [task("a", { state: "failed" }), task("b", { state: "failed" })];

    expect(standing(dead, [supervisor("s1", { last_seen_seconds_ago: 9000 })], dispatching)).toBe("stopped");
  });

  it("takes the ledger's word once the books are closed", () => {
    const closed = { ...campaign, finalized_at: minutes(1) };

    expect(standing([task("a", { state: "claimed" })], [supervisor("s1")], closed)).toBe("closed");
  });
});

describe("jobBubbles", () => {
  const field = (tasks: FleetTask[], runs: OwnRun[] = [], now = T0, gap?: number, results: FleetResult[] = []) =>
    jobBubbles([{ campaign, tasks, results }], runs, now, gap);

  const run = (id: string, over: Partial<OwnRun> = {}): OwnRun => ({
    id,
    label: `run ${id}`,
    state: "completed",
    at: T0,
    ...over,
  });

  const runsOnly = (runs: OwnRun[], now = T0, gap?: number) => jobBubbles([], runs, now, gap);
  const tasksOf = (placed: ReturnType<typeof field>) => placed.bubbles.filter((b) => b.taskId !== null);
  const start = "campaign:camp-1";

  it("places work left to right in the order it started, not by agent", () => {
    const placed = field(
      [
        task("a", { claimed_at: minutes(0), lease_owner: "s1" }),
        task("b", { claimed_at: minutes(60), lease_owner: "s2" }),
        task("c", { claimed_at: minutes(30), lease_owner: "s1" }),
      ],
      [],
      T0 + 60 * 60_000,
    );
    const order = [...placed.bubbles].sort((x, y) => x.x - y.x).map((b) => b.id);
    expect(order).toEqual([start, "a", "c", "b"]);
    // Rows are packing only: two tasks by the same agent are not forced into one.
    expect(new Set(placed.bubbles.map((b) => b.row))).toEqual(new Set([0]));
  });

  it("spaces work evenly, however long it was between", () => {
    const placed = runsOnly([run("a", { at: T0 }), run("b", { at: T0 + 60_000 }), run("c", { at: T0 + 60 * 60_000 })], T0 + 60 * 60_000);
    const x = (id: string) => placed.bubbles.find((b) => b.id === id)!.x;
    expect(x("a")).toBeCloseTo(0);
    expect(x("b")).toBeCloseTo(0.5);
    expect(x("c")).toBeCloseTo(1);
  });

  it("gives a row to each when the labels are wide enough to collide", () => {
    const runs = [0, 20, 40].map((at, i) => run(`r${i}`, { at: T0 + at * 60_000 }));
    // Labels wider than the space between neighbours, so the middle one has
    // to drop to a row of its own.
    const placed = runsOnly(runs, T0 + 40 * 60_000, 0.6);
    expect(placed.rows).toBe(2);
    expect(placed.bubbles.map((b) => b.row)).toEqual([0, 1, 0]);
  });

  it("shows the runs started here beside the tasks given away", () => {
    const placed = field([task("a", { claimed_at: minutes(10) })], [run("job-1", { at: T0 })], T0 + 60 * 60_000);

    expect(placed.bubbles.map((b) => [b.id, b.launcher])).toEqual([
      [start, "conductor"],
      ["job-1", "conductor"],
      ["a", "subagent"],
    ]);
  });

  it("starts each campaign from a bubble of the conductor's, ahead of the tasks it wrote", () => {
    const placed = field([task("a")]);

    expect(placed.bubbles[0]).toMatchObject({
      id: start,
      kind: "campaign",
      launcher: "conductor",
      agent: "C",
      label: "Optimise the JQ1 series",
      campaignId: "camp-1",
      taskId: null,
      jobId: null,
    });
  });

  it("draws a sub-agent's run once, as its task", () => {
    // The same run read twice — off the task and out of the job list —
    // would read as two experiments where there was one.
    const placed = field([task("a", { claimed_at: minutes(0), job_id: "job-1" })], [run("job-1")], T0);
    const drawn = placed.bubbles.filter((b) => b.jobId === "job-1");

    expect(drawn).toHaveLength(1);
    expect(drawn[0]).toMatchObject({ id: "a", launcher: "subagent" });
  });

  it("carries the run to open and the task to read", () => {
    const placed = field([task("a", { job_id: "job-1" }), task("b")], [run("job-2")], T0);
    const found = (id: string) => placed.bubbles.find((b) => b.id === id)!;

    expect(found("a")).toMatchObject({ jobId: "job-1", campaignId: "camp-1", taskId: "a" });
    expect(found("b")).toMatchObject({ jobId: null, campaignId: "camp-1", taskId: "b" });
    expect(found("job-2")).toMatchObject({ jobId: "job-2", campaignId: null, taskId: null });
  });

  it("letters each piece of work with who did it: C, or the sub-agent's number", () => {
    const placed = field(
      [
        task("a", { claimed_at: minutes(5), lease_owner: "sup-b" }),
        // Finished, so the lease is gone; who ran it comes from the claims.
        task("b", { claimed_at: minutes(1), lease_owner: null, ran_by: "sup-a" }),
        task("c", { state: "queued" }),
      ],
      [run("job-1")],
      T0 + 10 * 60_000,
    );
    const agent = (id: string) => placed.bubbles.find((b) => b.id === id)!.agent;

    expect(agent("job-1")).toBe("C");
    // Numbered in the order they turned up, not by their names.
    expect(agent("b")).toBe("S1");
    expect(agent("a")).toBe("S2");
    expect(agent("c")).toBe("S");
  });

  it("says what kind of work a task was, defaulting to a design run", () => {
    const placed = field([task("a", { spec: { kind: "retrosynthesis" } }), task("b")]);

    expect(tasksOf(placed).map((b) => b.kind)).toEqual(["retrosynthesis", "design"]);
  });

  it("reads a run's state as a phase, counting a stopped run as finished", () => {
    // It stopped at a checkpoint somebody asked for; its molecules are real.
    const placed = runsOnly([run("a", { state: "stopped" }), run("b", { state: "running" }), run("c", { state: "failed" })]);

    expect(placed.bubbles.map((b) => b.phase)).toEqual(["done", "running", "failed"]);
  });

  it("draws the lineage the conductor recorded, and the conductor behind the first tasks", () => {
    const placed = field([task("a"), task("b"), task("c", { spec: { intent: "merge", derived_from: ["a", "b"] } })]);

    expect(placed.edges).toEqual([
      { from: start, to: "a" },
      { from: start, to: "b" },
      { from: "a", to: "c" },
      { from: "b", to: "c" },
    ]);
  });

  it("hangs a task whose parents are not drawn off the conductor, rather than off nothing", () => {
    expect(field([task("a", { spec: { derived_from: ["gone", "a"] } })]).edges).toEqual([{ from: start, to: "a" }]);
  });

  it("shows a campaign as going until its work is done or it is closed", () => {
    const phase = (tasks: FleetTask[], c = campaign) => jobBubbles([{ campaign: c, tasks, results: [] }], [], T0).bubbles[0].phase;

    expect(phase([])).toBe("running");
    expect(phase([task("a", { state: "claimed" })])).toBe("running");
    expect(phase([task("a"), task("b", { state: "failed" })])).toBe("done");
    expect(phase([task("a", { state: "claimed" })], { ...campaign, finalized_at: minutes(1) })).toBe("done");
  });

  it("labels a bubble with the title, then the intent, then the rationale", () => {
    const placed = field([
      task("a", { spec: { title: "JQ1 scaffold", intent: "Hold the JQ1 scaffold" } }),
      task("b", { spec: { intent: "Hold the JQ1 scaffold" } }),
      task("c", { spec: { rationale: "try the other pocket" } }),
    ]);

    expect(tasksOf(placed).map((b) => b.label)).toEqual(["JQ1 scaffold", "Hold the JQ1 scaffold", "try the other pocket"]);
  });

  it("puts lone work at the left rather than dividing by a zero span", () => {
    const placed = runsOnly([run("a")]);
    expect(placed.bubbles[0].x).toBe(0);
    expect(Number.isFinite(placed.bubbles[0].x)).toBe(true);
  });

  it("is empty for a conversation that has set nothing going", () => {
    expect(jobBubbles([], [], T0)).toMatchObject({ bubbles: [], edges: [], rows: 0 });
  });
});

describe("taskReports", () => {
  const reports = (tasks: FleetTask[], results: FleetResult[]) => taskReports({ tasks, results });

  it("is what the sub-agent wrote, newest first", () => {
    const list = reports(
      [task("a", { spec: { title: "JAK3 selectivity" } }), task("b", { spec: { title: "Routes" } })],
      [
        result("a", { report: "Held the scaffold and scored 12.", posted_at: minutes(5) }),
        result("b", { report: "Three routes under six steps.", posted_at: minutes(9) }),
      ],
    );

    expect(list.map((r) => [r.title, r.excerpt])).toEqual([
      ["Routes", "Three routes under six steps."],
      ["JAK3 selectivity", "Held the scaffold and scored 12."],
    ]);
  });

  it("opens with the first paragraph, not the heading above it", () => {
    const [report] = reports([task("a")], [result("a", { report: "## Report\n\nThe pocket is shallow.\n\nMore below." })]);

    expect(report.excerpt).toBe("The pocket is shallow.");
    expect(report.full).toBe(true);
  });

  it("falls back to the summary for a result posted without a report", () => {
    // An older supervisor, or a run harvested before reports existed.
    const [report] = reports([task("a")], [result("a", { summary: "12 candidates, best 0.81" })]);

    expect(report.excerpt).toBe("12 candidates, best 0.81");
    expect(report.full).toBe(false);
  });

  it("carries the kind and the molecules, so a task that ran nothing still reads as work", () => {
    const list = reports(
      [task("a", { spec: { kind: "research" } }), task("b", { spec: { kind: "design" } })],
      [result("a"), result("b", { candidates: [{ smiles: "CCO" }] })],
    );
    const found = (id: string) => list.find((r) => r.taskId === id)!;

    expect(found("a")).toMatchObject({ kind: "research", molecules: 0 });
    expect(found("b")).toMatchObject({ kind: "design", molecules: 1 });
  });

  it("drops the heading the supervisor writes above every report", () => {
    // The card it is shown in is already titled "Report".
    expect(reportBody("## Report\n\nThe pocket is shallow.")).toBe("The pocket is shallow.");
    expect(reportBody("## What I found\n\nThe pocket is shallow.")).toBe("## What I found\n\nThe pocket is shallow.");
  });

  it("names a result whose task is no longer in the campaign", () => {
    const [report] = reports([], [result("gone", { report: "Done." })]);

    expect(report).toMatchObject({ title: "gone", kind: "design", owner: null });
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

  it("names the run each molecule came from, the result's own when the molecule does not say", () => {
    const pooled = pooledCandidates([
      result("lead", { job_id: null, candidates: [{ smiles: "CCO", score: 0.9, job_id: "r1" }] }),
      result("a", { job_id: "r2", candidates: [{ smiles: "CCN", score: 0.5 }] }),
      result("b", { job_id: null, candidates: [{ smiles: "CCC", score: 0.1 }] }),
    ]);
    expect(pooled.map((c) => c.jobId)).toEqual(["r1", "r2", null]);
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

  it("credits a finished task to the agent that ran it, after its lease is gone", () => {
    const [agent] = agentSummaries({
      tasks: [task("a", { lease_owner: null, ran_by: "s1" })],
      results: [result("a", { candidates: [{ smiles: "CCO", score: 0.5 }] })],
      supervisors: [supervisor("s1", { tasks_completed: 1 })],
    });
    expect(agent).toMatchObject({ owner: "s1", molecules: 1, best: 0.5 });
  });
});

describe("agentLabels", () => {
  it("numbers the sub-agents in the order they turned up, across every campaign", () => {
    const labels = agentLabels([
      { tasks: [task("a", { claimed_at: minutes(20), lease_owner: "zeta" })] },
      { tasks: [task("b", { claimed_at: minutes(5), ran_by: "alpha" }), task("c", { claimed_at: minutes(30), ran_by: "zeta" })] },
    ]);
    expect([...labels]).toEqual([
      ["alpha", "S1"],
      ["zeta", "S2"],
    ]);
  });

  it("counts a sub-agent from when it reported, before it claimed anything", () => {
    const labels = agentLabels([
      { tasks: [task("a", { claimed_at: minutes(5), lease_owner: "late" })], supervisors: [supervisor("early", { first_seen_at: minutes(1) })] },
    ]);
    expect(labels.get("early")).toBe("S1");
    expect(labels.get("late")).toBe("S2");
  });

  it("does not give the conductor an S-number of its own", () => {
    // Working alone it owns every task, and the trajectory would read as a
    // team of one stranger rather than as the agent being talked to.
    const labels = agentLabels([
      { tasks: [task("a", { claimed_at: minutes(5), ran_by: CONDUCTOR_OWNER }), task("b", { claimed_at: minutes(9), ran_by: "sub" })] },
    ]);
    expect([...labels]).toEqual([["sub", "S1"]]);
  });
});

describe("taskAgent", () => {
  it("labels a task the conductor ran itself C, beside the sub-agents", () => {
    expect(taskAgent(task("a", { ran_by: CONDUCTOR_OWNER }), new Map())).toBe(CONDUCTOR);
  });

  it("labels one nobody has claimed S", () => {
    expect(taskAgent(task("a", {}), new Map())).toBe(UNCLAIMED);
  });
});

describe("bestBySmiles", () => {
  const one = (smiles: string, score: number | null, taskId: string): PooledCandidate => ({
    smiles,
    score,
    taskId,
    owner: null,
    jobId: null,
    candidate: { smiles },
  });

  it("merges what the sub-agents posted with what the runs produced", () => {
    // Working alone the Molecules view was empty however many runs had
    // finished; with sub-agents it left out everything the conductor ran.
    const pooled = bestBySmiles([[one("CCO", 0.4, "posted")], [one("CCN", 0.9, "run")]]);
    expect(pooled.map((c) => c.smiles)).toEqual(["CCN", "CCO"]);
  });

  it("keeps one entry per molecule, with the better score", () => {
    const pooled = bestBySmiles([[one("CCO", 0.4, "posted")], [one("CCO", 0.8, "run")]]);
    expect(pooled).toHaveLength(1);
    expect(pooled[0]).toMatchObject({ score: 0.8, taskId: "run" });
  });

  it("caps the pool", () => {
    expect(bestBySmiles([[one("CCO", 0.4, "a"), one("CCN", 0.9, "b")]], 1).map((c) => c.smiles)).toEqual(["CCN"]);
  });
});

describe("workingAgents", () => {
  it("lists who holds a task until the campaign is finalized, S1 first", () => {
    const labels = new Map([
      ["alpha", "S1"],
      ["beta", "S2"],
      ["gamma", "S10"],
    ]);
    const working = workingAgents(
      [
        {
          campaign: { ...campaign, state: "closed" },
          tasks: [
            task("a", { state: "claimed", lease_owner: "gamma", spec: { title: "Grow the hinge binders" }, claimed_at: minutes(3) }),
            task("b", { state: "claimed", lease_owner: "alpha", spec: { intent: "Read the JAK3 SAR" }, claimed_at: minutes(1), job_id: "job-1" }),
            task("c", { state: "done", ran_by: "beta" }),
            task("d", { state: "queued" }),
          ],
        },
        { campaign: { ...campaign, campaign_id: "camp-2", state: "finalized", finalized_at: minutes(9) }, tasks: [task("e", { state: "claimed", lease_owner: "beta" })] },
      ],
      labels,
    );
    expect(working.map((one) => [one.agent, one.title])).toEqual([
      ["S1", "Read the JAK3 SAR"],
      ["S10", "Grow the hinge binders"],
    ]);
    expect(working[0]).toMatchObject({ since: T0 + 60_000, jobId: "job-1", taskId: "b" });
  });
});

describe("knownAgent", () => {
  it("only recognises a label some sub-agent goes under", () => {
    const labels = new Map([["alpha", "S1"]]);
    expect(knownAgent("S1", labels)).toBe(true);
    expect(knownAgent("S2", labels)).toBe(false);
    expect(knownAgent("alpha", labels)).toBe(false);
    expect(knownAgent(null, labels)).toBe(false);
  });
});

describe("agentPage", () => {
  const labels = new Map([
    ["alpha", "S1"],
    ["beta", "S2"],
  ]);
  const detail = {
    campaign,
    tasks: [
      task("old", { ran_by: "alpha", finished_at: minutes(5) }),
      task("older", { ran_by: "alpha", finished_at: minutes(2) }),
      task("now", { state: "claimed", lease_owner: "alpha", claimed_at: minutes(6) }),
      task("theirs", { ran_by: "beta" }),
    ],
    results: [],
    supervisors: [],
  };
  const ids = (page: ReturnType<typeof agentPage>) => page.held.map(({ task }) => task.task_id);

  it("is the task the sub-agent holds now, not everything it did", () => {
    const page = agentPage([detail], labels, "S1", null);
    expect(ids(page)).toEqual(["now"]);
    expect(page.stored).toBe(false);
  });

  it("is a finished task of its when one is picked, and only its own", () => {
    expect(ids(agentPage([detail], labels, "S1", "older"))).toEqual(["older"]);
    expect(agentPage([detail], labels, "S1", "older").stored).toBe(true);
    expect(ids(agentPage([detail], labels, "S1", "theirs"))).toEqual(["now"]);
  });

  it("is empty once it has reported, pointing at what it reported last", () => {
    const idle = { ...detail, tasks: detail.tasks.filter((one) => one.task_id !== "now") };
    const page = agentPage([idle], labels, "S1", null);
    expect(page.held).toEqual([]);
    expect(page.last?.task.task_id).toBe("old");
  });
});

describe("taskJobs", () => {
  it("collects the runs on the tasks and the ones only their results name", () => {
    const jobs = taskJobs(
      [{ tasks: [task("a", { job_id: "job-1" }), task("b", { job_id: "job-3" })], results: [result("a", { job_ids: ["job-1", "job-2"] })] }],
      new Set(["a"]),
    );
    expect([...jobs].sort()).toEqual(["job-1", "job-2"]);
  });
});

describe("campaignRuns", () => {
  it("puts each run down to its task and the sub-agent that ran it, including runs only the result names", () => {
    const runs = campaignRuns(
      [
        {
          tasks: [task("a", { job_id: "job-1", ran_by: "s1", spec: { title: "Scaffold hop" } }), task("b", { state: "queued", job_id: "job-3" })],
          results: [result("a", { job_ids: ["job-1", "job-2"] })],
        },
      ],
      new Map([["s1", "S1"]]),
    );
    expect(Object.fromEntries(runs)).toEqual({
      "job-1": { agent: "S1", title: "Scaffold hop" },
      "job-2": { agent: "S1", title: "Scaffold hop" },
      "job-3": { agent: "S", title: "b" },
    });
  });
});

describe("taskOwners", () => {
  it("names the sub-agent of a task that holds it and of one that finished", () => {
    const owners = taskOwners([task("a", { state: "claimed", lease_owner: "s1" }), task("b", { lease_owner: null, ran_by: "s2" }), task("c", { state: "queued" })]);
    expect(Object.fromEntries(owners)).toEqual({ a: "s1", b: "s2" });
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

describe("teamLoad", () => {
  const metrics = (over: Partial<CampaignMetrics> = {}): CampaignMetrics => ({
    concurrency: 4,
    running: 3,
    queued: 1,
    free_slots: 1,
    backlog_target: 2,
    utilisation: 0.784,
    busy_seconds: 3600,
    handoffs: 5,
    handoff_seconds: { median: 12.4, max: 300 },
    ...over,
  });

  it("reads out who is working, the queue against its target, and the ping-pong", () => {
    const [line] = teamLoad([{ campaign, metrics: metrics() }]);
    expect(line).toMatchObject({
      busy: "3 of 4 sub-agents working",
      queue: "1 queued, 2 to keep ready",
      short: true,
      utilisation: "78% busy since the first task",
      handoff: "12 s median wait between tasks",
    });
  });

  it("says nothing it has not measured", () => {
    const [line] = teamLoad([{ campaign, metrics: metrics({ utilisation: null, handoff_seconds: { median: null, max: null }, queued: 2 }) }]);
    expect(line).toMatchObject({ utilisation: null, handoff: null, short: false });
  });

  it("leaves out closed campaigns and ones the ledger could not measure", () => {
    expect(teamLoad([{ campaign: { ...campaign, state: "finalized" }, metrics: metrics() }, { campaign, metrics: null }, { campaign }])).toEqual([]);
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
