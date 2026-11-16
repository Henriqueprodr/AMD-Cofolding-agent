// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import type { AgentTurn, SubagentEvent } from "../types/api";
import { OUTPUT_NOT_KEPT, stuckOn, subagentTurns } from "./subagentChat";

const at = (s: number) => new Date(Date.UTC(2026, 10, 11, 10, 0, s)).toISOString();

const research: SubagentEvent[] = [
  { ts: at(0), event: "user_message", content: "Read the JAK3 SAR" },
  { ts: at(1), event: "thinking", content: "Start with the target." },
  { ts: at(2), event: "tool_call", name: "fetch_target_info", arguments: { query: "JAK3" } },
  { ts: at(3), event: "tool_call", name: "search_chembl", arguments: { target: "CHEMBL2148" } },
  { ts: at(4), event: "tool_result", name: "search_chembl", elapsed_s: 1, is_error: true, error: "ChEMBL said 500" },
  { ts: at(5), event: "tool_result", name: "fetch_target_info", elapsed_s: 3.2, is_error: false },
  { ts: at(9), event: "assistant_message", content: "## Report\nCys909 sets JAK3 apart." },
];

describe("subagentTurns", () => {
  it("reads as the brief, then one turn of steps and the answer", () => {
    const turns = subagentTurns(research, "t1", false);
    expect(turns.map((turn) => turn.role)).toEqual(["user", "agent"]);
    const answer = turns[1] as AgentTurn;
    expect(answer.blocks.map((block) => block.kind)).toEqual(["thinking", "tool", "tool", "text"]);
    expect(answer).toMatchObject({ done: true, at: Date.parse(at(1)), endedAt: Date.parse(at(9)) });
  });

  it("pairs each result with its call by name, keeping time and failure", () => {
    const [, answer] = subagentTurns(research, "t1", false) as [unknown, AgentTurn];
    const [fetch, search] = answer.blocks.filter((block) => block.kind === "tool");
    expect(fetch.kind === "tool" && fetch.result).toMatchObject({ isError: false, elapsedMs: 3200, result: OUTPUT_NOT_KEPT });
    expect(search.kind === "tool" && search.result).toMatchObject({ isError: true, result: "ChEMBL said 500" });
  });

  it("leaves the last turn open while the task is running", () => {
    const turns = subagentTurns(research.slice(0, 4), "t1", true);
    const live = turns[1] as AgentTurn;
    expect(live.done).toBe(false);
    expect(live.blocks.filter((block) => block.kind === "tool" && !block.result)).toHaveLength(2);
  });

  it("closes what a finished task left hanging", () => {
    const [, turn] = subagentTurns(research.slice(0, 3), "t1", false) as [unknown, AgentTurn];
    expect(turn.done).toBe(true);
    const call = turn.blocks.find((block) => block.kind === "tool");
    expect(call?.kind === "tool" && call.result?.isError).toBe(true);
  });

  it("starts a new turn when it is asked again for its report", () => {
    const turns = subagentTurns(
      [...research, { ts: at(10), event: "user_message", content: "Write your report." }, { ts: at(12), event: "assistant_message", content: "Done." }],
      "t1",
      false,
    );
    expect(turns.map((turn) => turn.role)).toEqual(["user", "agent", "user", "agent"]);
  });

  it("shows what came back where the transcript kept it", () => {
    const [answer] = subagentTurns(
      [
        { ts: at(0), event: "tool_call", name: "list_ttps", arguments: {} },
        { ts: at(1), event: "tool_result", name: "list_ttps", excerpt: '{"ttps": [', content_length: 900 },
      ],
      "t1",
      false,
    ) as [AgentTurn];
    const [call] = answer.blocks;
    expect(call.kind === "tool" && call.result).toMatchObject({ result: '{"ttps": [', truncated: true });
  });
});

const INVALID = "Invalid TTP: objective 'warhead_presence' requires a 'component' field";
const resend = (from: number, times: number, error = INVALID): SubagentEvent[] =>
  Array.from({ length: times }, (_, n): SubagentEvent[] => [
    { ts: at(from + 3 * n), event: "thinking", content: "Try again." },
    { ts: at(from + 3 * n + 1), event: "tool_call", name: "create_ttp", arguments: { ttp: { id: "t" } } },
    { ts: at(from + 3 * n + 2), event: "tool_result", name: "create_ttp", is_error: true, error },
  ]).flat();

describe("a call sent again and again", () => {
  it("is one row with how many times it went out", () => {
    const [answer] = subagentTurns(resend(0, 5), "t1", true) as [AgentTurn];
    const tools = answer.blocks.filter((block) => block.kind === "tool");
    expect(tools).toHaveLength(1);
    expect(tools[0]).toMatchObject({ repeats: 5, result: { isError: true, result: INVALID } });
  });

  it("is kept apart when the answer changed", () => {
    const [answer] = subagentTurns([...resend(0, 2), ...resend(10, 1, "Invalid TTP: something else")], "t1", true) as [AgentTurn];
    expect(answer.blocks.filter((block) => block.kind === "tool").map((block) => block.kind === "tool" && block.repeats)).toEqual([2, undefined]);
  });

  it("ends the turn with the reason the agent stopped", () => {
    const turns = subagentTurns([...resend(0, 6), { ts: at(30), event: "stuck", content: "create_ttp was sent 6 times in a row" }], "t1", true);
    const answer = turns[0] as AgentTurn;
    expect(answer.done).toBe(true);
    expect(answer.blocks.at(-1)).toMatchObject({ kind: "error", message: "Stopped: create_ttp was sent 6 times in a row" });
  });

  it("says the agent is stuck once it has failed the same way three times", () => {
    expect(stuckOn(resend(0, 2))).toBeNull();
    expect(stuckOn(resend(0, 3))).toBe("create_ttp");
  });

  it("does not call a changed attempt, a success or a new answer stuck", () => {
    expect(stuckOn([...resend(0, 3), { ts: at(20), event: "tool_call", name: "create_ttp", arguments: { ttp: { id: "u" } } }])).toBeNull();
    expect(stuckOn([...resend(0, 3), { ts: at(20), event: "assistant_message", content: "Done." }])).toBeNull();
    expect(stuckOn([...resend(0, 3), { ts: at(20), event: "tool_call", name: "list_ttps", arguments: {} }, { ts: at(21), event: "tool_result", name: "list_ttps", excerpt: "[]" }])).toBeNull();
  });
});
