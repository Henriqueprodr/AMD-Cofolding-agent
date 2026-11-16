// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import { nextRead, shortest, type JobsRead } from "./useJobList";

const UNREAD: JobsRead = { jobs: null, alerts: [], error: null };
const job = (id: string) => ({
  job_id: id,
  job_type: "optimization",
  state: "running",
  created_at: "2026-11-13T10:00:00+00:00",
});
const alert = (id: string) => ({
  job_id: id,
  severity: "warning",
  message: "Scores are flat",
});

describe("nextRead", () => {
  it("takes the jobs and the alerts of a good answer", () => {
    const read = nextRead(UNREAD, [job("a1")], [alert("a1")]);
    expect(read.jobs?.map((j) => j.id)).toEqual(["a1"]);
    expect(read.alerts).toHaveLength(1);
    expect(read.error).toBeNull();
  });

  it("keeps the last good list when the gateway answers with an error", () => {
    // One bad poll is not the run having vanished; the error is shown beside it.
    const good = nextRead(UNREAD, [job("a1")], [alert("a1")]);
    const bad = nextRead(good, { error: "Runner unreachable" }, null);
    expect(bad.jobs?.map((j) => j.id)).toEqual(["a1"]);
    expect(bad.alerts).toHaveLength(1);
    expect(bad.error).toBe("Runner unreachable");
  });

  it("keeps the alerts when only they could not be read", () => {
    const good = nextRead(UNREAD, [job("a1")], [alert("a1")]);
    expect(nextRead(good, [job("a1"), job("b2")], null).alerts).toHaveLength(1);
  });
});

describe("shortest", () => {
  it("runs the one timer as often as the most impatient caller asked", () => {
    expect(shortest([60_000, 10_000, 15_000])).toBe(10_000);
  });

  it("is no timer when nobody is watching", () => {
    expect(shortest([])).toBe(0);
  });
});
