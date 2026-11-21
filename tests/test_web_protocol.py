# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the browser front end's wire protocol, hub and event handler.

No FastAPI needed: these cover the pure-Python layer between the agent and
the socket.

Covers:
- every protocol constructor carries type + ts
- SessionHub fan-out, replay buffer, unsubscribe
- WebEventHandler: thinking/tool/builtin messages, inline vs payload_ref,
  Stop raising TurnCancelled at the next tool call
- score-table extraction, including the gate and the row cap
"""

from __future__ import annotations

import asyncio
import json

import pytest

from reinvent4_agent.web import protocol
from reinvent4_agent.web.events import (
    INLINE_LIMIT,
    MAX_TABLE_ROWS,
    TurnCancelled,
    TurnContext,
    WebEventHandler,
    _score_table_payload,
    _summarize,
)
from reinvent4_agent.web.hub import SessionHub


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class TestProtocol:
    """Every message is self-describing."""

    def test_messages_carry_type_and_timestamp(self):
        messages = [
            protocol.status("hi"),
            protocol.busy(True),
            protocol.turn_start("t1", "driver", "user"),
            protocol.thinking("t1", "driver", "hmm"),
            protocol.tool_call("t1", "driver", 1, "score_molecules", {}),
            protocol.assistant_message("t1", "driver", "done"),
            protocol.session_reset(),
            protocol.pong(),
        ]
        for message in messages:
            assert message["type"]
            assert message["ts"]
            # Must survive the socket.
            json.dumps(message)

    def test_builtin_activity_is_its_own_type(self):
        message = protocol.builtin_activity(
            "t1", "driver", name="WebSearch", arguments={}, result="", is_error=False
        )

        # Not a tool_call: the distinction between web text and chemistry
        # evidence is the whole point of showing it.
        assert message["type"] == "builtin_activity"
        assert message["type"] != protocol.S_TOOL_CALL

    def test_session_message_has_no_credentials(self):
        message = protocol.session(
            session_id="x",
            backend="claude-code",
            model="sonnet",
            multi_agent=False,
            web_tools=True,
            tools=["score_molecules"],
            rdkit_available=True,
        )

        blob = json.dumps(message).lower()
        for secret in ("api_key", "token", "gateway", "sk-"):
            assert secret not in blob


# ---------------------------------------------------------------------------
# Hub
# ---------------------------------------------------------------------------


class TestSessionHub:
    """One session, many tabs."""

    def test_publish_reaches_every_subscriber(self):
        async def scenario():
            hub = SessionHub()
            a, b = hub.subscribe(), hub.subscribe()
            hub.publish(protocol.status("hello"))
            return await a.get(), await b.get()

        first, second = asyncio.run(scenario())
        assert first["text"] == second["text"] == "hello"

    def test_replay_returns_history_for_a_late_joiner(self):
        hub = SessionHub()
        hub.publish(protocol.status("one"))
        hub.publish(protocol.status("two"))

        assert [m["text"] for m in hub.replay()] == ["one", "two"]

    def test_replay_is_bounded(self):
        hub = SessionHub(replay_limit=3)
        for i in range(10):
            hub.publish(protocol.status(str(i)))

        replay = hub.replay()
        assert len(replay) == 3
        assert [m["text"] for m in replay] == ["7", "8", "9"]

    def test_unsubscribe_stops_delivery(self):
        async def scenario():
            hub = SessionHub()
            queue = hub.subscribe()
            hub.unsubscribe(queue)
            hub.publish(protocol.status("ignored"))
            return queue.qsize(), hub.subscriber_count

        assert asyncio.run(scenario()) == (0, 0)

    def test_clear_replay(self):
        hub = SessionHub()
        hub.publish(protocol.status("x"))
        hub.clear_replay()

        assert hub.replay() == []


# ---------------------------------------------------------------------------
# Event handler
# ---------------------------------------------------------------------------


def _handler(hub, ctx, **kwargs):
    return WebEventHandler(
        hub, asyncio.get_event_loop_policy().new_event_loop(),
        agent_label="driver", context=ctx, **kwargs
    )


class TestWebEventHandler:
    """Agent callbacks become socket messages."""

    def test_tool_call_and_result_share_a_call_id(self):
        hub, ctx = SessionHub(), TurnContext(turn_id="t1")
        handler = _handler(hub, ctx)

        handler.on_tool_call("score_molecules", {"smiles": ["CCO"]})
        handler.on_tool_result("score_molecules", '{"ok": true}', False)

        call, result = hub.replay()
        assert call["type"] == "tool_call"
        assert result["type"] == "tool_result"
        assert call["call_id"] == result["call_id"] == 1

    def test_large_results_are_referenced_not_inlined(self):
        hub, ctx = SessionHub(), TurnContext(turn_id="t1")
        stored = {}
        handler = _handler(
            hub, ctx, store_payload=lambda ref, body: stored.setdefault(ref, body) and ref
        )

        handler.on_tool_call("read_results", {})
        handler.on_tool_result("read_results", "x" * (INLINE_LIMIT + 10), False)

        result = hub.replay()[-1]
        assert result["inline"] is None
        assert result["payload_ref"]
        assert result["bytes"] == INLINE_LIMIT + 10

    def test_small_results_are_inlined(self):
        hub, ctx = SessionHub(), TurnContext(turn_id="t1")
        handler = _handler(hub, ctx)

        handler.on_tool_call("get_ttp", {})
        handler.on_tool_result("get_ttp", "small", False)

        assert hub.replay()[-1]["inline"] == "small"

    def test_builtin_tool_emits_web_activity(self):
        hub, ctx = SessionHub(), TurnContext(turn_id="t1")
        handler = _handler(hub, ctx)

        handler.on_builtin_tool("WebSearch", {"query": "BET"}, "results", False)

        message = hub.replay()[-1]
        assert message["type"] == "builtin_activity"
        assert message["name"] == "WebSearch"

    def test_stop_raises_at_the_next_tool_call(self):
        hub, ctx = SessionHub(), TurnContext(turn_id="t1")
        handler = _handler(hub, ctx)
        ctx.cancel = True

        with pytest.raises(TurnCancelled):
            handler.on_tool_call("run_optimization", {})

    def test_paths_in_results_are_noted_for_the_viewer(self):
        hub, ctx = SessionHub(), TurnContext(turn_id="t1")
        seen: list[str] = []
        handler = _handler(hub, ctx, note_payload=seen.append)

        handler.on_tool_call("list_shared_assets", {})
        handler.on_tool_result("list_shared_assets", '{"structures": ["a.pdb"]}', False)

        assert seen and "a.pdb" in seen[0]


# ---------------------------------------------------------------------------
# Score tables
# ---------------------------------------------------------------------------


def _scored(n: int) -> str:
    return json.dumps(
        {
            "molecules": [
                {
                    "smiles": f"C{'C' * i}O",
                    "total_score": 0.5,
                    "components": {"QED": {"transformed": 0.8, "raw": 0.8}},
                }
                for i in range(n)
            ],
            "summary": {"mean_total_score": 0.5, "n_valid": n, "n_molecules": n},
        }
    )


class TestScoreTable:
    """Only score_molecules produces a table, and it is bounded."""

    def test_extracts_rows_and_components(self):
        payload = _score_table_payload("score_molecules", _scored(2))

        assert payload is not None
        assert len(payload["molecules"]) == 2
        assert payload["components"] == ["QED"]
        assert payload["truncated_at"] is None

    def test_other_tools_do_not_make_tables(self):
        assert _score_table_payload("read_results", _scored(2)) is None

    def test_non_json_is_ignored(self):
        assert _score_table_payload("score_molecules", "not json") is None

    def test_row_cap(self):
        payload = _score_table_payload(
            "score_molecules", _scored(MAX_TABLE_ROWS + 25)
        )

        assert len(payload["molecules"]) == MAX_TABLE_ROWS
        assert payload["truncated_at"] == MAX_TABLE_ROWS


class TestSummaries:
    """The collapsed one-liner on each tool card."""

    def test_scored_molecules(self):
        assert "2 molecule(s) scored" in _summarize(
            "score_molecules", _scored(2), False
        )

    def test_bioactivity(self):
        body = json.dumps({"activities": [{}, {}], "total_available": 250})
        assert "2 measured activities (of 250)" in _summarize(
            "fetch_bioactivity", body, False
        )

    def test_error_payload(self):
        body = json.dumps({"error": "ChEMBL unreachable"})
        assert "ChEMBL unreachable" in _summarize("fetch_bioactivity", body, False)

    def test_non_json_falls_back_to_text(self):
        assert _summarize("whatever", "plain text", False) == "plain text"
