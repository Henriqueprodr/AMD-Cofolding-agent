# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A tool that answers with an error, and a model that keeps sending the same call anyway."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from reinvent4_agent.agent import REPEAT_LIMIT, REPEAT_WARN, Agent, RepeatedToolCall
from reinvent4_agent.backends import ChatResult, LLMBackend, ToolCall, ToolResult

_INVALID = json.dumps({"error": "Invalid TTP: External objective 'Affinity' requires a 'component' field"})


class _SameCall(LLMBackend):
    """Sends one call ``times`` times, then answers; keeps what it was told."""

    def __init__(self, times: int, arguments: Any = None) -> None:
        self.times = times
        self.arguments = arguments or (lambda n: {"ttp": {"id": "t"}})
        self.results: list[ToolResult] = []
        self.calls = 0

    def complete(self, tools: list[dict[str, Any]], max_tokens: int, *, force_tool_use: bool = False) -> ChatResult:
        self.calls += 1
        if self.calls <= self.times:
            call = ToolCall(id=f"call_{self.calls}", name="create_ttp", arguments=self.arguments(self.calls))
            return ChatResult(tool_calls=[call], needs_tool_response=True)
        return ChatResult(text="done", needs_tool_response=False)

    def add_user_message(self, content: str) -> None: ...

    def add_tool_results(self, results: list[ToolResult]) -> None:
        self.results.extend(results)

    def repair_dangling_tool_use(self) -> None: ...
    def reset(self, system_prompt: str) -> None: ...
    def export_history(self) -> list[dict[str, Any]]: return []
    def import_history(self, messages: list[dict[str, Any]]) -> None: ...


class _MCP:
    def __init__(self, answer: str) -> None:
        self.answer = answer

    async def list_tools(self) -> list[dict[str, Any]]:
        return [{"name": "create_ttp", "description": "", "input_schema": {}}]

    def resolve_tool_name(self, name: str) -> str | None:
        return name

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        return self.answer


def _events(tmp_path) -> list[dict[str, Any]]:
    [transcript] = tmp_path.glob("session_*_driver-t1.jsonl")
    return [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines()]


def test_an_error_in_the_answer_counts_as_a_failed_call(tmp_path):
    backend = _SameCall(1)
    asyncio.run(Agent(_MCP(_INVALID), backend, log_dir=str(tmp_path), name="driver-t1").chat("go"))

    [result] = [event for event in _events(tmp_path) if event["event"] == "tool_result"]
    assert result["is_error"] is True
    assert result["error"].startswith("Invalid TTP")
    assert result["excerpt"] == _INVALID
    assert backend.results[0].is_error is True


def test_an_answer_that_only_mentions_an_error_is_not_a_failure(tmp_path):
    status = json.dumps({"job_id": "j1", "status": "failed", "error": "REINVENT exited with code 1"})
    backend = _SameCall(1)
    asyncio.run(Agent(_MCP(status), backend, log_dir=str(tmp_path), name="driver-t1").chat("go"))

    [result] = [event for event in _events(tmp_path) if event["event"] == "tool_result"]
    assert result["is_error"] is False


def test_the_model_is_told_once_it_repeats_a_failing_call(tmp_path):
    backend = _SameCall(REPEAT_WARN)
    assert asyncio.run(Agent(_MCP(_INVALID), backend, log_dir=str(tmp_path), name="driver-t1").chat("go")) == "done"

    told = ["will not change the answer" in result.content for result in backend.results]
    assert told == [False] * (REPEAT_WARN - 1) + [True]


def test_the_turn_ends_when_the_same_call_keeps_failing(tmp_path):
    backend = _SameCall(REPEAT_LIMIT + 10)
    with pytest.raises(RepeatedToolCall, match=f"create_ttp was sent {REPEAT_LIMIT} times"):
        asyncio.run(Agent(_MCP(_INVALID), backend, log_dir=str(tmp_path), name="driver-t1").chat("go"))

    assert backend.calls == REPEAT_LIMIT
    assert _events(tmp_path)[-1]["event"] == "stuck"


def test_changing_the_call_starts_the_count_again(tmp_path):
    backend = _SameCall(REPEAT_LIMIT + 2, arguments=lambda n: {"ttp": {"id": "t", "try": n // 2}})
    assert asyncio.run(Agent(_MCP(_INVALID), backend, log_dir=str(tmp_path), name="driver-t1").chat("go")) == "done"


def test_polling_that_gets_the_same_answer_is_not_a_loop(tmp_path):
    backend = _SameCall(REPEAT_LIMIT + 2)
    queued = json.dumps({"job_id": "j1", "status": "queued"})
    assert asyncio.run(Agent(_MCP(queued), backend, log_dir=str(tmp_path), name="driver-t1").chat("go")) == "done"
    assert not any("will not change" in result.content for result in backend.results)
