# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the sub-agent allowance of a chat session.

The allowance is a ceiling the user sets, not a fleet they start: nothing
here launches anything, and the only thing that changes is what the agent is
told it may spend.  Driven through ``LiveSession`` directly, as the autonomy
tests are, and inside a running loop because a turn is a task.
"""

from __future__ import annotations

import asyncio

from test_web_autonomy import FakeAgent, idle, make_session, published


def drain(queue) -> list[dict]:
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events


def test_a_new_session_works_alone(tmp_path):
    live = make_session(tmp_path, FakeAgent())
    assert live.agents_allowed == 1
    assert live.summary()["agents_allowed"] == 1


def test_below_one_is_working_alone(tmp_path):
    """Zero sub-agents is not a thing: the agent itself is always there."""

    async def run() -> None:
        live = make_session(tmp_path, FakeAgent())
        assert live.set_agents(0) == 1
        assert live.set_agents(-4) == 1

    asyncio.run(run())


def test_setting_it_reaches_every_tab_watching(tmp_path):
    async def run() -> None:
        live = make_session(tmp_path, FakeAgent())
        queue = live.subscribe()
        live.set_agents(3)
        assert live.agents_allowed == 3
        assert drain(queue) == [{"type": "agents", "count": 3}]
        # Read back by a browser that reconnects, which is how a tab opened
        # later shows the same setting.
        assert live.summary()["agents_allowed"] == 3

    asyncio.run(run())


def test_the_setting_is_not_part_of_the_conversation(tmp_path):
    """Replayed on a reconnect it would announce a change nobody just made."""

    async def run() -> None:
        live = make_session(tmp_path, FakeAgent())
        live.set_agents(3)
        assert published(live, "agents") == []

    asyncio.run(run())


def test_working_alone_adds_nothing_to_the_turn(tmp_path):
    async def run() -> None:
        agent = FakeAgent()
        live = make_session(tmp_path, agent)
        assert live.start_turn("optimise this series")
        await idle(live)
        assert agent.asked == ["optimise this series"]

    asyncio.run(run())


def test_the_allowance_reaches_the_agent(tmp_path):
    async def run() -> None:
        agent = FakeAgent()
        live = make_session(tmp_path, agent)
        live.set_agents(3)
        assert live.start_turn("optimise this series")
        await idle(live)
        assert agent.asked[0].startswith("optimise this series")
        assert "up to 3 sub-agents" in agent.asked[0]

    asyncio.run(run())


def test_the_user_message_is_recorded_as_they_typed_it(tmp_path):
    """The allowance is context for the agent, not part of what was said."""

    async def run() -> None:
        live = make_session(tmp_path, FakeAgent())
        live.set_agents(2)
        assert live.start_turn("optimise this series")
        await idle(live)
        assert published(live, "user_message")[-1]["content"] == "optimise this series"

    asyncio.run(run())


def test_the_agent_still_decides_whether_to_spend_them(tmp_path):
    """Said plainly: a bare number reads as an instruction to run a campaign."""

    async def run() -> None:
        agent = FakeAgent()
        live = make_session(tmp_path, agent)
        live.set_agents(2)
        assert live.start_turn("what does this molecule look like?")
        await idle(live)
        assert "if it does not, say so and do it yourself" in agent.asked[0]

    asyncio.run(run())
