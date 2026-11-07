# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tools answered by the agent itself rather than over MCP.

The MCP server is one process behind one pipe for every conversation at
once, so a tool that has to know *which* conversation called it cannot live
there. These are the ones that do.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from reinvent4_agent.agent import Agent
from reinvent4_agent.backends import ChatResult, ToolCall


class FakeMCP:
    """An MCP connection offering one tool."""

    def __init__(self, tools: list[str] | None = None) -> None:
        self._tools = tools if tools is not None else ["list_jobs"]
        self.called: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> list[dict[str, Any]]:
        return [{"name": name, "description": name, "input_schema": {}} for name in self._tools]

    def resolve_tool_name(self, name: str) -> str | None:
        return name if name in self._tools else None

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.called.append((name, arguments))
        return f"{name} over mcp"


class FakeBackend:
    """Asks for the given tool calls on the first turn, then answers."""

    def __init__(self, calls: list[ToolCall]) -> None:
        self._calls = calls
        self.round = 0
        self.tool_results: list[Any] = []
        self.offered: list[list[dict[str, Any]]] = []

    def add_user_message(self, content: str) -> None:
        pass

    def add_tool_results(self, results: list[Any]) -> None:
        self.tool_results.extend(results)

    def complete(self, tools, max_tokens, **kwargs) -> ChatResult:
        self.offered.append(tools)
        self.round += 1
        if self.round == 1 and self._calls:
            return ChatResult(
                text=None, tool_calls=list(self._calls), needs_tool_response=True
            )
        return ChatResult(text="done", tool_calls=[])


def _agent(mcp: FakeMCP, backend: FakeBackend, **kwargs) -> Agent:
    return Agent(mcp, backend, **kwargs)


def _call(name: str, arguments: dict[str, Any] | None = None) -> ToolCall:
    return ToolCall(id=f"c-{name}", name=name, arguments=arguments or {})


DEFINITION = {"name": "notepad", "description": "write it down", "input_schema": {}}


class TestLocalTools:

    def test_a_local_tool_is_offered_to_the_model(self):
        """One it is never offered is one it will never call."""
        mcp, backend = FakeMCP(), FakeBackend([])
        agent = _agent(mcp, backend)
        agent.local_tools["notepad"] = (DEFINITION, lambda args: "noted")

        asyncio.run(agent.chat("hello"))

        assert "notepad" in {t["name"] for t in backend.offered[0]}

    def test_a_local_tool_is_answered_without_touching_mcp(self):
        mcp, backend = FakeMCP(), FakeBackend([_call("notepad", {"text": "x"})])
        agent = _agent(mcp, backend)
        seen: list[dict[str, Any]] = []

        def handler(arguments):
            seen.append(arguments)
            return "noted"

        agent.local_tools["notepad"] = (DEFINITION, handler)
        asyncio.run(agent.chat("hello"))

        assert seen == [{"text": "x"}]
        assert mcp.called == []
        assert backend.tool_results[0].content == "noted"

    def test_an_async_handler_is_awaited(self):
        """Otherwise the model is handed a coroutine's repr as its answer."""
        mcp, backend = FakeMCP(), FakeBackend([_call("notepad")])
        agent = _agent(mcp, backend)

        async def handler(arguments):
            return "noted later"

        agent.local_tools["notepad"] = (DEFINITION, handler)
        asyncio.run(agent.chat("hello"))

        assert backend.tool_results[0].content == "noted later"

    def test_a_handler_that_raises_is_a_failed_call_not_a_failed_turn(self):
        """Losing the conversation because the notepad broke is a poor trade."""
        mcp, backend = FakeMCP(), FakeBackend([_call("notepad")])
        agent = _agent(mcp, backend)

        def handler(arguments):
            raise RuntimeError("the disk is full")

        agent.local_tools["notepad"] = (DEFINITION, handler)
        answer = asyncio.run(agent.chat("hello"))

        assert answer == "done"
        assert backend.tool_results[0].is_error is True
        assert "the disk is full" in backend.tool_results[0].content

    def test_a_local_tool_replaces_the_mcp_tool_of_the_same_name(self):
        """Two tools with one name leaves the backend to pick, which is nobody's decision."""
        mcp, backend = FakeMCP(["list_jobs"]), FakeBackend([])
        agent = _agent(mcp, backend)
        agent.local_tools["list_jobs"] = (
            {"name": "list_jobs", "description": "mine", "input_schema": {}},
            lambda args: "local",
        )

        asyncio.run(agent.chat("hello"))

        offered = [t for t in backend.offered[0] if t["name"] == "list_jobs"]
        assert len(offered) == 1
        assert offered[0]["description"] == "mine"

    def test_the_allow_list_does_not_hide_a_local_tool(self):
        """It was installed by whoever owns the agent, which is the decision."""
        mcp, backend = FakeMCP(["list_jobs", "delete_everything"]), FakeBackend([])
        agent = _agent(mcp, backend, allowed_tools={"list_jobs"})
        agent.local_tools["notepad"] = (DEFINITION, lambda args: "noted")

        asyncio.run(agent.chat("hello"))

        names = {t["name"] for t in backend.offered[0]}
        assert names == {"list_jobs", "notepad"}

    def test_without_a_local_tool_nothing_changes(self):
        mcp, backend = FakeMCP(), FakeBackend([_call("list_jobs")])
        agent = _agent(mcp, backend)

        asyncio.run(agent.chat("hello"))

        assert mcp.called == [("list_jobs", {})]
