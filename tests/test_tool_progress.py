"""How far a running tool has got, from the MCP call to the browser.

Tools that call ``Context.report_progress`` send progress notifications
while they work. These cover the path that carries them: the agent asks
the client for them, the web sink turns them into an event keyed to the
call already on screen, and anything that cannot report them is left to
behave exactly as it did before.
"""

from __future__ import annotations

import pytest

from reinvent4_agent.agent import _reports_progress
from reinvent4_agent.mcp_client import MCPClient
from reinvent4_agent.web.events import WebSocketEventHandler


class TestAskingForProgress:
    def test_the_real_client_is_asked_for_progress(self):
        assert _reports_progress(MCPClient.__new__(MCPClient))

    def test_a_client_that_cannot_report_it_is_not_asked(self):
        class Older:
            async def call_tool(self, name, arguments):  # noqa: ANN001, ANN201
                return "{}"

        assert not _reports_progress(Older())

    def test_something_that_is_not_a_client_at_all_is_not_asked(self):
        assert not _reports_progress(object())


@pytest.fixture
def emitted():
    sent: list[dict] = []
    handler = WebSocketEventHandler(sink=sent.append)
    return handler, sent


class TestWhatTheBrowserIsTold:
    def test_progress_arrives_against_the_call_it_belongs_to(self, emitted):
        handler, sent = emitted
        handler.on_tool_call("rcsb_fetch_files", {"entry_ids": ["1IQK", "2OSL"]})
        call_id = sent[-1]["id"]

        handler.on_tool_progress("rcsb_fetch_files", 1.0, 2.0, "Fetching 2OSL")

        assert sent[-1]["type"] == "tool_progress"
        assert sent[-1]["id"] == call_id
        assert sent[-1]["done"] == 1.0
        assert sent[-1]["total"] == 2.0
        assert sent[-1]["message"] == "Fetching 2OSL"

    def test_a_tool_of_unknown_length_may_say_so(self, emitted):
        handler, sent = emitted
        handler.on_tool_call("cluster_molecules", {})

        handler.on_tool_progress("cluster_molecules", 3.0, None, None)

        assert sent[-1]["total"] is None
        assert sent[-1]["message"] is None

    def test_progress_for_a_call_that_already_answered_is_dropped(self, emitted):
        handler, sent = emitted
        handler.on_tool_call("rcsb_fetch_files", {})
        handler.on_tool_result("rcsb_fetch_files", "{}", is_error=False)
        before = len(sent)

        handler.on_tool_progress("rcsb_fetch_files", 1.0, 2.0, "late")

        # Nothing on screen is still waiting on it, so there is nowhere to
        # put it and a stray row would be worse than silence.
        assert len(sent) == before

    def test_a_tool_that_reports_nothing_sends_nothing(self, emitted):
        handler, sent = emitted
        handler.on_tool_call("list_jobs", {})
        handler.on_tool_result("list_jobs", "[]", is_error=False)

        assert [event["type"] for event in sent] == ["tool_call", "tool_result"]
