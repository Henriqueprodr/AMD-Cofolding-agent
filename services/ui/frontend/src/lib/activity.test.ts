// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import type { CampaignFinding, FleetCampaign, FleetEvent, FleetTask } from "../types/api";
import { branches, columns, history, stepsOf } from "./activity";

const T0 = Date.parse("2026-11-08T10:00:00Z");
const minutes = (n: number) => new Date(T0 + n * 60_000).toISOString();

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

function task(id: string, over: Partial<FleetTask> = {}): FleetTask {
  return {
    task_id: id,
    campaign_id: "camp-1",
    spec: { title: `Grow ${id}` },
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

function event(event_id: number, kind: string, over: Partial<FleetEvent> = {}): FleetEvent {
  return { event_id, kind, task_id: null, payload: {}, created_at: minutes(event_id), ...over };
}

describe("history", () => {
  it("reads the ledger as one line each, newest first", () => {
    const { steps } = history(
      [
        {
          campaign,
          tasks: [task("t1", { lease_owner: "sup-a" })],
          events: [
            event(1, "campaign_created", { payload: { goal: "Optimise the JQ1 series" } }),
            event(2, "task_enqueued", { task_id: "t1" }),
            event(3, "task_claimed", { task_id: "t1", payload: { owner: "sup-a" } }),
            event(4, "task_started_job", { task_id: "t1", payload: { owner: "sup-a", job_id: "job-9" } }),
            event(5, "task_done", { task_id: "t1", payload: { owner: "sup-a", n_candidates: 3 } }),
          ],
        },
      ],
      new Map([["sup-a", "S1"]]),
    );
    expect(steps.map((step) => [step.kind, step.text])).toEqual([
      ["report", "Reported on Grow t1"],
      ["run", "Started a run for Grow t1"],
      ["claim", "Took Grow t1"],
      ["handout", "Gave out Grow t1"],
      ["campaign", "Opened a campaign"],
    ]);
    expect(steps[0]).toMatchObject({ agent: "S1", detail: "3 molecules", taskId: "t1" });
    expect(steps[1].jobId).toBe("job-9");
  });

  it("gives each agent a lane, the conductor first", () => {
    const { lanes } = history(
      [
        {
          campaign,
          tasks: [task("t1", { ran_by: "sup-b" }), task("t2", { ran_by: "sup-a" })],
          events: [
            event(1, "task_claimed", { task_id: "t1", payload: { owner: "sup-b" } }),
            event(2, "task_claimed", { task_id: "t2", payload: { owner: "sup-a" } }),
          ],
        },
      ],
      new Map([
        ["sup-a", "S1"],
        ["sup-b", "S2"],
      ]),
    );
    expect(lanes).toEqual(["C", "S1", "S2"]);
  });

  it("puts what nobody has claimed on its own lane", () => {
    const { steps, lanes } = history(
      [{ campaign, tasks: [task("t1")], events: [event(1, "task_claimed", { task_id: "t1", payload: { owner: "sup-gone" } })] }],
      new Map(),
    );
    expect(steps[0].agent).toBe("S");
    expect(lanes).toEqual(["C", "S"]);
  });

  it("keeps a failure's reason and leaves out the requeue that follows it", () => {
    const { steps } = history(
      [
        {
          campaign,
          tasks: [task("t1", { state: "failed", ran_by: "sup-a", error: "ChEMBL said 500" })],
          events: [
            event(1, "task_failed", { task_id: "t1", payload: { owner: "sup-a", error: "ChEMBL said 500" } }),
            event(2, "task_requeued", { task_id: "t1" }),
          ],
        },
      ],
      new Map([["sup-a", "S1"]]),
    );
    expect(steps.map((step) => step.kind)).toEqual(["failure"]);
    expect(steps[0].detail).toBe("ChEMBL said 500");
  });

  it("folds the conductor's log in by time", () => {
    const { steps } = history(
      [{ campaign, tasks: [], events: [event(2, "campaign_created")] }],
      new Map(),
      [
        { ts: minutes(3), text: "Two series look worth growing" },
        { ts: minutes(1), text: "" },
      ] as CampaignFinding[],
    );
    expect(steps.map((step) => [step.kind, step.text])).toEqual([
      ["note", "Two series look worth growing"],
      ["campaign", "Opened a campaign"],
    ]);
    expect(steps[0].agent).toBe("C");
  });

  it("shortens a long task title so a line stays one line", () => {
    const long = "Grow the aminopyrimidine series towards selectivity over JAK1 and JAK2";
    const { steps } = history(
      [{ campaign, tasks: [task("t1", { spec: { title: long } })], events: [event(1, "task_enqueued", { task_id: "t1" })] }],
      new Map(),
    );
    expect(steps[0].text).toBe("Gave out Grow the aminopyrimidine series towards sel…");
  });

  it("ignores a kind it does not know", () => {
    expect(history([{ campaign, tasks: [], events: [event(1, "lease_renewed")] }], new Map()).steps).toEqual([]);
  });
});

describe("branches", () => {
  const labels = new Map([
    ["sup-a", "S1"],
    ["sup-b", "S2"],
  ]);

  it("forks at the handout, runs from the claim to the report, and merges where a task built on it", () => {
    const { steps } = history(
      [
        {
          campaign,
          tasks: [task("t1", { ran_by: "sup-a" }), task("t2", { spec: { title: "Grow t2", derived_from: ["t1"] }, state: "running", ran_by: "sup-b" })],
          events: [
            event(1, "task_enqueued", { task_id: "t1" }),
            event(2, "task_claimed", { task_id: "t1", payload: { owner: "sup-a" } }),
            event(3, "task_done", { task_id: "t1", payload: { owner: "sup-a" } }),
            event(4, "task_enqueued", { task_id: "t2" }),
            event(5, "task_claimed", { task_id: "t2", payload: { owner: "sup-b" } }),
          ],
        },
      ],
      labels,
    );
    // Rows, newest first: 0 claimed t2, 1 gave out t2, 2 reported t1, 3 took t1, 4 gave out t1.
    const byTask = new Map(branches(steps).map((branch) => [branch.taskId, branch]));
    expect(byTask.get("t1")).toEqual({ agent: "S1", taskId: "t1", from: 4, start: 3, end: 2, into: [1] });
    // Still being worked on: no end, so nothing can have been built on it yet.
    expect(byTask.get("t2")).toEqual({ agent: "S2", taskId: "t2", from: 1, start: 0, end: null, into: [] });
  });

  it("leaves a result nothing built on unmerged", () => {
    const { steps } = history(
      [
        {
          campaign,
          tasks: [task("t1", { ran_by: "sup-a" })],
          events: [
            event(1, "task_enqueued", { task_id: "t1" }),
            event(2, "task_claimed", { task_id: "t1", payload: { owner: "sup-a" } }),
            event(3, "task_done", { task_id: "t1", payload: { owner: "sup-a" } }),
          ],
        },
      ],
      labels,
    );
    expect(branches(steps)[0].into).toEqual([]);
  });
});

describe("columns", () => {
  const labels = new Map([
    ["sup-a", "S1"],
    ["sup-b", "S2"],
  ]);

  it("gives tasks worked at once columns of their own and reuses one once it is free", () => {
    const { steps } = history(
      [
        {
          campaign,
          tasks: [task("t1", { ran_by: "sup-a" }), task("t2", { ran_by: "sup-b" }), task("t3", { state: "running", ran_by: "sup-a" })],
          events: [
            event(1, "task_enqueued", { task_id: "t1" }),
            event(2, "task_enqueued", { task_id: "t2" }),
            event(3, "task_claimed", { task_id: "t1", payload: { owner: "sup-a" } }),
            event(4, "task_claimed", { task_id: "t2", payload: { owner: "sup-b" } }),
            event(5, "task_done", { task_id: "t1", payload: { owner: "sup-a" } }),
            event(6, "task_done", { task_id: "t2", payload: { owner: "sup-b" } }),
            event(7, "task_enqueued", { task_id: "t3" }),
            event(8, "task_claimed", { task_id: "t3", payload: { owner: "sup-a" } }),
          ],
        },
      ],
      labels,
    );
    const laid = columns(steps, branches(steps));
    expect(Object.fromEntries(laid.branch)).toEqual({ t1: 1, t2: 2, t3: 1 });
    // Rows, newest first: t3 claimed, t3 given out, t2 done, t1 done, t2 claimed, t1 claimed, t2 and t1 given out.
    expect(laid.step).toEqual([1, 0, 2, 1, 2, 1, 0, 0]);
    expect(laid.count).toBe(3);
  });

  it("lets a task forked where another merged take that one's column", () => {
    const { steps } = history(
      [
        {
          campaign,
          tasks: [task("t1", { ran_by: "sup-a" }), task("t2", { spec: { title: "Grow t2", derived_from: ["t1"] }, state: "running", ran_by: "sup-b" })],
          events: [
            event(1, "task_enqueued", { task_id: "t1" }),
            event(2, "task_claimed", { task_id: "t1", payload: { owner: "sup-a" } }),
            event(3, "task_done", { task_id: "t1", payload: { owner: "sup-a" } }),
            event(4, "task_enqueued", { task_id: "t2" }),
            event(5, "task_claimed", { task_id: "t2", payload: { owner: "sup-b" } }),
          ],
        },
      ],
      labels,
    );
    expect(Object.fromEntries(columns(steps, branches(steps)).branch)).toEqual({ t1: 1, t2: 1 });
  });
});

describe("stepsOf", () => {
  it("keeps only what one agent did", () => {
    const { steps } = history(
      [
        {
          campaign,
          tasks: [task("t1", { ran_by: "sup-a" })],
          events: [event(1, "campaign_created"), event(2, "task_claimed", { task_id: "t1", payload: { owner: "sup-a" } })],
        },
      ],
      new Map([["sup-a", "S1"]]),
    );
    expect(stepsOf(steps, "S1").map((step) => step.text)).toEqual(["Took Grow t1"]);
  });
});
