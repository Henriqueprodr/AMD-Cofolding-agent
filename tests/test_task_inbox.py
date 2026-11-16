# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Messages typed to a sub-agent at work: left by the gateway, read between its steps."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from reinvent4_agent.agent import Agent
from reinvent4_agent.backends import ChatResult, LLMBackend, ToolCall, ToolResult
from reinvent4_agent.task_inbox import Inbox, inbox_path, messages, post
from reinvent4_agent.web.subagent_transcript import read_conversation


class _TwoRounds(LLMBackend):
    """A tool call, then a reply; remembers the order things were added in."""

    def __init__(self) -> None:
        self.history: list[tuple[str, Any]] = []
        self.calls = 0

    def complete(self, tools: list[dict[str, Any]], max_tokens: int, *, force_tool_use: bool = False) -> ChatResult:
        self.calls += 1
        if self.calls == 1:
            return ChatResult(tool_calls=[ToolCall(id="call_1", name="read_results", arguments={})], needs_tool_response=True)
        return ChatResult(text="done", needs_tool_response=False)

    def add_user_message(self, content: str) -> None:
        self.history.append(("user", content))

    def add_tool_results(self, results: list[ToolResult]) -> None:
        self.history.append(("tool", [result.tool_call_id for result in results]))

    def repair_dangling_tool_use(self) -> None: ...
    def reset(self, system_prompt: str) -> None: ...
    def export_history(self) -> list[dict[str, Any]]: return []
    def import_history(self, messages: list[dict[str, Any]]) -> None: ...


class _MCP:
    async def list_tools(self) -> list[dict[str, Any]]:
        return [{"name": "read_results", "description": "", "input_schema": {}}]

    def resolve_tool_name(self, name: str) -> str | None:
        return name

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        return "ok"


def test_what_is_left_is_read_once_and_in_order(tmp_path):
    post(tmp_path, "t1", "first")
    inbox = Inbox(tmp_path, "t1")
    assert inbox.take() == ["first"]
    assert inbox.take() == []

    post(tmp_path, "t1", "second")
    post(tmp_path, "t1", "third")
    assert inbox.take() == ["second", "third"]


def test_a_half_written_line_is_skipped(tmp_path):
    post(tmp_path, "t1", "kept")
    with inbox_path(tmp_path, "t1").open("a", encoding="utf-8") as handle:
        handle.write('{"text": "cut of')
    assert [record["text"] for record in messages(tmp_path, "t1")] == ["kept"]


def test_a_message_goes_in_after_the_tool_results_and_is_marked(tmp_path):
    backend = _TwoRounds()
    agent = Agent(_MCP(), backend, log_dir=str(tmp_path), name="subagent-t1")
    post(tmp_path, "t1", "Prefer the Cys909 series")
    agent.inbox = Inbox(tmp_path, "t1").take

    assert asyncio.run(agent.chat("Read the SAR")) == "done"

    # Between the tool's answer and the next completion, never inside a call's pairing.
    assert backend.history == [("user", "Read the SAR"), ("tool", ["call_1"]), ("user", "Prefer the Cys909 series")]
    [transcript] = tmp_path.glob("session_*_subagent-t1.jsonl")
    users = [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines()]
    users = [event for event in users if event["event"] == "user_message"]
    assert [event.get("interjected", False) for event in users] == [False, True]


def test_the_conversation_says_which_messages_are_still_waiting(tmp_path):
    (tmp_path / "session_x_subagent-t1.jsonl").write_text(
        json.dumps({"event": "user_message", "content": "brief"}) + "\n"
        + json.dumps({"event": "user_message", "content": "one", "interjected": True}) + "\n",
        encoding="utf-8",
    )
    post(tmp_path, "t1", "one")
    post(tmp_path, "t1", "two")

    read = read_conversation(tmp_path, "t1")
    assert [record["text"] for record in read["queued"]] == ["two"]
    assert read["events"][1]["interjected"] is True


def test_a_message_before_any_transcript_is_queued(tmp_path):
    post(tmp_path, "t1", "early")
    assert [record["text"] for record in read_conversation(tmp_path, "t1")["queued"]] == ["early"]
