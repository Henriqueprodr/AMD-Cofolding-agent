// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * A sub-agent's transcript as turns of a conversation, so it reads like the
 * one with the conductor. Pure, so the mapping can be tested without a DOM.
 */

import type { AgentTurn, SubagentEvent, ToolResultEvent, Turn, TurnBlock } from "../types/api";

type ToolBlock = Extract<TurnBlock, { kind: "tool" }>;

/** Older transcripts keep how long a call took and whether it failed, not what it returned. */
export const OUTPUT_NOT_KEPT = "The sub-agent's transcript does not keep tool output.";

/** Identical failing calls in a row after which an agent is said to be stuck; the agent is warned at the same count. */
const STUCK_AFTER = 3;

/**
 * *running*: the task is still in the sub-agent's hands, so its last turn
 * is open and a call without a result is still going.
 */
export function subagentTurns(events: readonly SubagentEvent[], taskId: string, running: boolean): Turn[] {
  const turns: Turn[] = [];
  let agent: AgentTurn | null = null;
  // Results come back by name, in the order the calls went out.
  let pending: ToolBlock[] = [];

  const at = (event: SubagentEvent, fallback: number) => {
    const parsed = Date.parse(event.ts || "");
    return Number.isNaN(parsed) ? fallback : parsed;
  };
  const close = (end: number) => {
    if (!agent) return;
    agent.done = true;
    agent.endedAt = end;
    for (const block of pending) block.result = result(block, end, { error: "No result was recorded." });
    pending = [];
    agent = null;
  };
  const open = (when: number): AgentTurn => {
    agent ??= (() => {
      const turn: AgentTurn = { role: "agent", id: `${taskId}:a${turns.length}`, at: when, blocks: [], done: false };
      turns.push(turn);
      return turn;
    })();
    return agent;
  };

  let last = 0;
  events.forEach((event, index) => {
    const when = at(event, last);
    last = when;
    const id = `${taskId}:${index}`;
    switch (event.event) {
      case "user_message":
        close(when);
        turns.push({ role: "user", id, content: event.content ?? "", at: when });
        break;
      case "thinking":
        if (event.content?.trim()) open(when).blocks.push({ kind: "thinking", id, content: event.content });
        break;
      case "tool_call": {
        const block: ToolBlock = { kind: "tool", id, name: event.name ?? "tool", arguments: event.arguments ?? {}, at: when };
        open(when).blocks.push(block);
        pending.push(block);
        break;
      }
      case "tool_result": {
        const call = pending.find((block) => block.name === event.name);
        if (!call) break;
        pending = pending.filter((block) => block !== call);
        call.result = result(call, when, event);
        if (agent) foldRepeat(agent, call);
        break;
      }
      case "assistant_message":
        if (event.content?.trim()) open(when).blocks.push({ kind: "text", id, content: event.content });
        close(when);
        break;
      case "max_rounds":
        open(when).blocks.push({ kind: "error", id, message: "Stopped at the limit on tool rounds." });
        close(when);
        break;
      case "stuck":
        open(when).blocks.push({ kind: "error", id, message: `Stopped: ${event.content ?? "it kept sending the same failing call."}` });
        close(when);
        break;
    }
  });
  if (!running) close(last);
  return turns;
}

/**
 * The name of the call a running agent keeps resending, when its last
 * calls were all that one, with the same arguments, failing the same way.
 */
export function stuckOn(events: readonly SubagentEvent[]): string | null {
  const calls: SubagentEvent[] = [];
  const results: SubagentEvent[] = [];
  for (let index = events.length - 1; index >= 0 && (results.length < STUCK_AFTER || calls.length < STUCK_AFTER); index--) {
    const event = events[index];
    if (event.event === "tool_call") calls.push(event);
    else if (event.event === "tool_result") results.push(event);
    else if (event.event !== "thinking") return null;
  }
  if (results.length < STUCK_AFTER || calls.length < STUCK_AFTER) return null;
  const [last] = results;
  const said = (event: SubagentEvent) => event.error ?? event.excerpt;
  const sameAnswer = results.every((event) => (event.is_error || event.error) && event.name === last.name && said(event) === said(last));
  const sameCall = calls.every((event) => event.name === last.name && JSON.stringify(event.arguments) === JSON.stringify(calls[0].arguments));
  return sameAnswer && sameCall ? (last.name ?? null) : null;
}

/** Fold *call* into the call before it in *turn* when it repeats it word for word, answer included. */
function foldRepeat(turn: AgentTurn, call: ToolBlock) {
  const at = turn.blocks.indexOf(call);
  let before = at - 1;
  while (before >= 0 && turn.blocks[before].kind === "thinking") before--;
  const prev = turn.blocks[before];
  if (prev?.kind !== "tool" || !prev.result || !call.result) return;
  const same =
    prev.name === call.name &&
    prev.result.isError === call.result.isError &&
    prev.result.result === call.result.result &&
    JSON.stringify(prev.arguments) === JSON.stringify(call.arguments);
  if (!same) return;
  prev.repeats = (prev.repeats ?? 1) + 1;
  prev.result = call.result;
  turn.blocks.splice(before + 1, at - before);
}

function result(call: ToolBlock, when: number, event: Pick<SubagentEvent, "elapsed_s" | "is_error" | "error" | "excerpt" | "content_length">): ToolResultEvent {
  const failed = Boolean(event.is_error || event.error);
  const excerpt = event.excerpt || null;
  return {
    type: "tool_result",
    seq: 0,
    ts: new Date(when).toISOString(),
    id: call.id,
    name: call.name,
    result: event.error ?? excerpt ?? (failed ? "It failed; the transcript does not say how." : OUTPUT_NOT_KEPT),
    truncated: !event.error && excerpt !== null && (event.content_length ?? 0) > excerpt.length,
    isError: failed,
    panels: [],
    elapsedMs: typeof event.elapsed_s === "number" ? Math.round(event.elapsed_s * 1000) : null,
  };
}
