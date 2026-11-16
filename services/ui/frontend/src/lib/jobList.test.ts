// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import { CONDUCTOR, UNCLAIMED } from "./fleet";
import { ELSEWHERE, jobRows, launchers } from "./jobList";
import { parseJob, type JobSummary } from "./jobs";

const T0 = Date.UTC(2026, 10, 13, 12, 0, 0);
const iso = (minutes: number) => new Date(T0 + minutes * 60_000).toISOString();

function job(id: string, state: string, times: { created?: number; started?: number; finished?: number } = {}): JobSummary {
  return parseJob({
    job_id: id,
    state,
    created_at: times.created === undefined ? null : iso(times.created),
    started_at: times.started === undefined ? null : iso(times.started),
    finished_at: times.finished === undefined ? null : iso(times.finished),
  }) as JobSummary;
}

describe("jobRows", () => {
  const now = T0 + 60 * 60_000;

  it("names who started each run: a task's sub-agent, else the conductor for its own, else another chat", () => {
    const rows = jobRows(
      [job("a", "running"), job("b", "completed"), job("c", "completed")],
      new Set(["a", "b"]),
      new Map([["a", { agent: "S2", title: "Explore the morpholines" }]]),
      now,
    );
    expect(rows.map((row) => [row.by, row.task])).toEqual([
      ["S2", "Explore the morpholines"],
      [CONDUCTOR, null],
      [ELSEWHERE, null],
    ]);
  });

  it("groups by state and times the wait for the GPUs apart from the run", () => {
    const [running, waiting, done, failed] = jobRows(
      [
        job("r", "running", { created: 0, started: 10 }),
        job("q", "queued", { created: 30 }),
        job("d", "completed", { created: 0, started: 5, finished: 25 }),
        job("f", "failed", { created: 0 }),
      ],
      new Set(),
      new Map(),
      now,
    );
    expect([running.group, waiting.group, done.group, failed.group]).toEqual(["running", "waiting", "finished", "finished"]);
    expect([running.waitedMs, running.ranMs]).toEqual([10 * 60_000, 50 * 60_000]);
    expect([waiting.waitedMs, waiting.ranMs]).toEqual([30 * 60_000, null]);
    expect([done.waitedMs, done.ranMs]).toEqual([5 * 60_000, 20 * 60_000]);
    // Failed before it ever started: neither is known.
    expect([failed.waitedMs, failed.ranMs]).toEqual([null, null]);
  });
});

describe("launchers", () => {
  it("lists the conductor, then sub-agents by number, then the unclaimed and other chats", () => {
    const rows = ["S10", ELSEWHERE, "S2", CONDUCTOR, UNCLAIMED, "S2"].map((by) => ({ by }));
    expect(launchers(rows)).toEqual([CONDUCTOR, "S2", "S10", UNCLAIMED, ELSEWHERE]);
  });
});
