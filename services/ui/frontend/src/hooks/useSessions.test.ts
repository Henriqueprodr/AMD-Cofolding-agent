// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import type { SessionSummary } from "../types/api";
import { runsKnown } from "./useSessions";

const session = (id: string, runs?: SessionSummary["runs"]): SessionSummary =>
  ({ id, title: id, created: "", updated: "", turns: 0, busy: false, ...(runs ? { runs } : {}) });

describe("runsKnown", () => {
  it("waits while the first read is in flight", () => {
    expect(runsKnown(null, false)).toBeNull();
  });

  it("says the gateway does not say when a read came back without a list", () => {
    // Staying null held the job scope empty, and Summary, Results and Analysis
    // all render from it: an unreachable gateway showed three blank panels with
    // no error and no retry, rather than falling back to every run being the
    // conversation's.
    expect(runsKnown(null, true)).toBe(false);
  });

  it("says so when a conversation carries its runs", () => {
    expect(runsKnown([session("a", { running: 0, total: 0, jobs: [] })], true)).toBe(true);
  });

  it("says a gateway too old to carry them does not", () => {
    expect(runsKnown([session("a")], true)).toBe(false);
  });

  it("counts an empty list as knowing: there is nothing to attribute", () => {
    expect(runsKnown([], true)).toBe(true);
  });
});
