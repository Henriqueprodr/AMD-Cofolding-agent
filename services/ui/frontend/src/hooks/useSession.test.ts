// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import type { AutoEvent, SessionSummary, StatusEvent } from "../types/api";
import { applyEvent, EMPTY_TRANSCRIPT } from "../lib/transcript";
import { autoDeadline, autoMessage } from "./useSession";

const UNTIL = "2026-10-29T12:00:00+00:00";

const summary = (auto: string | null): SessionSummary => ({
  id: "s-20261029-080000-abcdef",
  title: "BRD4",
  created: UNTIL,
  updated: UNTIL,
  turns: 1,
  busy: false,
  auto_until: auto,
});

describe("the autonomy budget over the socket", () => {
  it("asks for whole minutes, and 0 to stop", () => {
    expect(autoMessage(30)).toEqual({ type: "auto", minutes: 30 });
    expect(autoMessage(0)).toEqual({ type: "auto", minutes: 0 });
    expect(autoMessage(90.4)).toEqual({ type: "auto", minutes: 90 });
    expect(autoMessage(-5)).toEqual({ type: "auto", minutes: 0 });
  });

  it("reads the deadline back from the event the gateway answers with", () => {
    expect(autoDeadline({ type: "auto", until: UNTIL } as AutoEvent)).toBe(UNTIL);
    expect(autoDeadline({ type: "auto", until: null } as AutoEvent)).toBeNull();
  });

  it("reads it from the session too, so a reconnect keeps the time left", () => {
    expect(autoDeadline(summary(UNTIL))).toBe(UNTIL);
    expect(autoDeadline(summary(null))).toBeNull();
    // Recorded before the budget existed.
    expect(autoDeadline(summary(undefined as unknown as null))).toBeNull();
  });
});

describe("automatic turns", () => {
  const busy = (auto?: boolean): StatusEvent => ({
    type: "status",
    state: "busy",
    seq: 1,
    ts: UNTIL,
    ...(auto === undefined ? {} : { auto }),
  });

  it("are marked, and turns the user asked for are not", () => {
    const [turn] = applyEvent(EMPTY_TRANSCRIPT, busy(true)).turns;
    expect(turn.role === "agent" && turn.auto).toBe(true);
    const [asked] = applyEvent(EMPTY_TRANSCRIPT, busy()).turns;
    expect(asked.role === "agent" && asked.auto).toBeUndefined();
  });
});

describe("how far a running tool has got", () => {
  const started = () =>
    applyEvent(
      applyEvent(EMPTY_TRANSCRIPT, { type: "status", state: "busy", seq: 1, ts: UNTIL }),
      { type: "tool_call", seq: 2, ts: UNTIL, id: "call-1", name: "rcsb_fetch_files", arguments: {} },
    );

  const toolBlock = (state: ReturnType<typeof started>) => {
    const turn = state.turns[0];
    if (turn.role !== "agent") throw new Error("expected an agent turn");
    const block = turn.blocks[0];
    if (block.kind !== "tool") throw new Error("expected a tool block");
    return block;
  };

  const progress = (seq: number, done: number) =>
    ({ type: "tool_progress", seq, ts: UNTIL, id: "call-1", name: "rcsb_fetch_files", done, total: 4, message: `at ${done}` }) as const;

  it("lands on the step it belongs to rather than adding a row", () => {
    const state = applyEvent(started(), progress(3, 1));

    const turn = state.turns[0];
    expect(turn.role === "agent" && turn.blocks).toHaveLength(1);
    expect(toolBlock(state).progress).toEqual({ done: 1, total: 4, message: "at 1" });
  });

  it("keeps only the latest report", () => {
    const state = applyEvent(applyEvent(started(), progress(3, 1)), progress(4, 3));

    expect(toolBlock(state).progress?.done).toBe(3);
  });

  it("is ignored when no such call is open", () => {
    const state = applyEvent(started(), { ...progress(3, 1), id: "call-9", name: "other_tool" });

    expect(toolBlock(state).progress).toBeUndefined();
  });
});
