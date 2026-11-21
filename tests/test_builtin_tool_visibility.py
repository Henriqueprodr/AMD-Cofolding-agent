# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests that harness-run tools (Claude's WebSearch/WebFetch) stay visible.

The user must be able to tell an answer retrieved from the open web from one
computed by the chemistry tools, so these never pass silently.

Covers:
- ChatResult.builtin_tool_uses default and field ordering
- _absorb_message: call/result pairing, ours-vs-theirs filtering, both
  tool-use block shapes, text/thinking accumulation
- _result_text flattening
- Agent.chat surfacing: on_builtin_tool, the legacy handler fallback, and
  the distinct JSONL event type
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from reinvent4_agent.backends import BuiltinToolUse, ChatResult, ToolCall
from reinvent4_agent.backends.claude_code_backend import (
    _absorb_message,
    _result_text,
)


# ---------------------------------------------------------------------------
# Stand-ins for the SDK's message/block classes
# ---------------------------------------------------------------------------
# _absorb_message dispatches on class NAME, so these need no SDK installed.


class TextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class ThinkingBlock:
    def __init__(self, thinking: str) -> None:
        self.thinking = thinking


class ToolUseBlock:
    def __init__(self, id: str, name: str, input: dict) -> None:
        self.id, self.name, self.input = id, name, input


class ServerToolUseBlock(ToolUseBlock):
    pass


class ToolResultBlock:
    def __init__(self, tool_use_id: str, content: Any, is_error: bool = False):
        self.tool_use_id, self.content, self.is_error = (
            tool_use_id,
            content,
            is_error,
        )


class AssistantMessage:
    def __init__(self, content: list) -> None:
        self.content = content


class UserMessage:
    def __init__(self, content: Any) -> None:
        self.content = content


class ResultMessage:
    def __init__(self, is_error: bool = False, result: str | None = None):
        self.is_error, self.result, self.subtype = is_error, result, "success"
        self.usage = {"input_tokens": 11, "output_tokens": 22}


@pytest.fixture
def buf():
    return {"text": [], "thinking": [], "builtin": []}


# ---------------------------------------------------------------------------
# ChatResult
# ---------------------------------------------------------------------------


class TestChatResultDefaults:
    """The field is additive: the other two backends must not need edits."""

    def test_defaults_to_empty_list(self):
        assert ChatResult().builtin_tool_uses == []

    def test_independent_lists_per_instance(self):
        a, b = ChatResult(), ChatResult()
        a.builtin_tool_uses.append(BuiltinToolUse(name="WebSearch"))
        assert b.builtin_tool_uses == []

    def test_existing_positional_construction_still_works(self):
        result = ChatResult("hello", [ToolCall("1", "t", {})])
        assert result.text == "hello"
        assert result.builtin_tool_uses == []


# ---------------------------------------------------------------------------
# Message folding
# ---------------------------------------------------------------------------


class TestAbsorbMessage:
    """Turning the SDK's message stream into buffers and records."""

    def test_text_and_thinking_accumulate(self, buf):
        _absorb_message(
            AssistantMessage([TextBlock("hi "), ThinkingBlock("hmm")]), buf, {}
        )
        _absorb_message(AssistantMessage([TextBlock("there")]), buf, {})

        assert "".join(buf["text"]) == "hi there"
        assert "".join(buf["thinking"]) == "hmm"

    def test_our_own_mcp_tools_are_not_recorded(self, buf):
        pending: dict = {}

        _absorb_message(
            AssistantMessage(
                [ToolUseBlock("t1", "mcp__dd__score_molecules", {"smiles": []})]
            ),
            buf,
            pending,
        )

        # dd-agent executes and displays these itself.
        assert buf["builtin"] == []
        assert pending == {}

    @pytest.mark.parametrize("block_cls", [ToolUseBlock, ServerToolUseBlock])
    def test_both_block_shapes_are_captured(self, buf, block_cls):
        pending: dict = {}

        _absorb_message(
            AssistantMessage(
                [block_cls("t1", "WebSearch", {"query": "BET inhibitors"})]
            ),
            buf,
            pending,
        )

        assert len(buf["builtin"]) == 1
        assert buf["builtin"][0].name == "WebSearch"
        assert buf["builtin"][0].arguments == {"query": "BET inhibitors"}
        assert pending["t1"] is buf["builtin"][0]

    def test_result_arrives_on_a_user_message_and_pairs_by_id(self, buf):
        pending: dict = {}
        _absorb_message(
            AssistantMessage([ToolUseBlock("t1", "WebFetch", {"url": "u"})]),
            buf,
            pending,
        )

        _absorb_message(
            UserMessage([ToolResultBlock("t1", [{"type": "text", "text": "page"}])]),
            buf,
            pending,
        )

        assert buf["builtin"][0].result == "page"
        assert buf["builtin"][0].is_error is False
        assert pending == {}  # consumed

    def test_results_for_our_tools_are_ignored(self, buf):
        pending: dict = {}

        _absorb_message(
            UserMessage([ToolResultBlock("other", "some mcp payload")]),
            buf,
            pending,
        )

        assert buf["builtin"] == []

    def test_string_user_content_does_not_crash(self, buf):
        assert _absorb_message(UserMessage("plain text"), buf, {}) is None

    def test_error_result_is_flagged(self, buf):
        pending: dict = {}
        _absorb_message(
            AssistantMessage([ToolUseBlock("t1", "WebFetch", {})]), buf, pending
        )

        _absorb_message(
            UserMessage([ToolResultBlock("t1", "403", is_error=True)]),
            buf,
            pending,
        )

        assert buf["builtin"][0].is_error is True

    def test_result_message_returns_usage(self, buf):
        usage = _absorb_message(ResultMessage(), buf, {})

        assert usage == {"input_tokens": 11, "output_tokens": 22}

    def test_result_message_error_is_surfaced_as_text(self, buf):
        _absorb_message(ResultMessage(is_error=True, result="limit"), buf, {})

        assert "claude-code: limit" in "".join(buf["text"])

    def test_unknown_blocks_are_ignored(self, buf):
        class MysteryBlock:
            pass

        _absorb_message(AssistantMessage([MysteryBlock()]), buf, {})

        assert buf["text"] == [] and buf["builtin"] == []


class TestResultText:
    """SDK tool-result payloads come in three shapes."""

    def test_none(self):
        assert _result_text(None) == ""

    def test_plain_string(self):
        assert _result_text("hello") == "hello"

    def test_text_blocks_are_joined(self):
        content = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        assert _result_text(content) == "a\nb"

    def test_non_text_list_falls_back_to_json(self):
        assert "url" in _result_text([{"type": "image", "url": "x"}])


# ---------------------------------------------------------------------------
# Agent surfacing
# ---------------------------------------------------------------------------


class _FakeBackend:
    """Returns one turn carrying a built-in tool use, then stops."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def complete(self, tools, max_tokens):  # noqa: ARG002
        return ChatResult(
            text="done",
            builtin_tool_uses=[
                BuiltinToolUse(
                    name="WebSearch",
                    arguments={"query": "BET clinical"},
                    result="three results",
                )
            ],
        )

    def add_user_message(self, content: str) -> None:
        self.messages.append(content)

    def add_tool_results(self, results) -> None: ...
    def repair_dangling_tool_use(self) -> None: ...
    def reset(self, system_prompt: str) -> None: ...


class _FakeMCP:
    async def list_tools(self):
        return []

    async def call_tool(self, name, arguments):  # pragma: no cover
        raise AssertionError("no MCP tools in this test")


class _RecordingHandler:
    def __init__(self) -> None:
        self.builtin: list[tuple] = []

    def on_thinking(self, text): ...
    def on_tool_call(self, name, arguments): ...
    def on_tool_result(self, name, result, is_error): ...
    def on_error(self, message): ...
    def on_monitor_alert(self, alert): ...
    def on_monitor_response(self, text): ...

    def on_builtin_tool(self, name, arguments, result, is_error):
        self.builtin.append((name, arguments, result, is_error))


class _LegacyHandler:
    """A handler written before on_builtin_tool existed."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.results: list[tuple] = []

    def on_thinking(self, text): ...
    def on_error(self, message): ...
    def on_monitor_alert(self, alert): ...
    def on_monitor_response(self, text): ...

    def on_tool_call(self, name, arguments):
        self.calls.append((name, arguments))

    def on_tool_result(self, name, result, is_error):
        self.results.append((name, result, is_error))


class TestAgentSurfacesBuiltinTools:
    """The agent must not let harness tools pass silently."""

    def test_handler_is_notified(self):
        from reinvent4_agent.agent import Agent

        handler = _RecordingHandler()
        agent = Agent(_FakeMCP(), _FakeBackend(), event_handler=handler)

        asyncio.run(agent.chat("anything new on BET?"))

        assert handler.builtin == [
            ("WebSearch", {"query": "BET clinical"}, "three results", False)
        ]

    def test_legacy_handler_falls_back_to_the_generic_pair(self):
        from reinvent4_agent.agent import Agent

        handler = _LegacyHandler()
        agent = Agent(_FakeMCP(), _FakeBackend(), event_handler=handler)

        asyncio.run(agent.chat("anything new?"))

        assert handler.calls == [("WebSearch", {"query": "BET clinical"})]
        assert handler.results == [("WebSearch", "three results", False)]

    def test_transcript_uses_a_distinct_event_type(self, tmp_path):
        from reinvent4_agent.agent import Agent

        agent = Agent(_FakeMCP(), _FakeBackend(), log_dir=str(tmp_path))

        asyncio.run(agent.chat("anything new?"))

        events = [
            json.loads(line)
            for path in tmp_path.glob("*.jsonl")
            for line in path.read_text().splitlines()
        ]
        kinds = [e["event"] for e in events]
        assert "builtin_tool_use" in kinds
        # Must NOT be filed as a normal tool_call, or post-hoc analysis
        # could no longer separate web text from dd-mcp evidence.
        assert "tool_call" not in kinds
        record = next(e for e in events if e["event"] == "builtin_tool_use")
        assert record["name"] == "WebSearch"
        assert record["content_length"] == len("three results")
