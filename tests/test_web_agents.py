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

from reinvent4_agent.web import sessions as web_sessions
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
        assert "Team size: 3 sub-agents at once" in agent.asked[0]

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


def test_two_or_more_make_the_long_work_the_sub_agents(tmp_path):
    """Setting the allowance is the user choosing how the work is done."""

    async def run() -> None:
        agent = FakeAgent()
        live = make_session(tmp_path, agent)
        live.set_agents(2)
        assert live.start_turn("what does this molecule look like?")
        await idle(live)
        assert "Team size: 2 sub-agents at once" in agent.asked[0]
        assert "enqueue_tasks call" in agent.asked[0]
        # run starters are withheld — the note says so
        assert "withheld" in agent.asked[0]

    asyncio.run(run())


def test_working_alone_withholds_the_campaign_tools(tmp_path):
    """At one there are no sub-agents, so a campaign would have nobody to run it."""

    live = make_session(tmp_path, FakeAgent())
    live.apply_allowance()
    assert set(live.agent.withheld) == web_sessions.FLEET_TOOLS
    assert "no sub-agents" in live.agent.withheld["start_campaign"]


def test_two_or_more_withhold_the_run_starters_instead(tmp_path):
    async def run() -> None:
        live = make_session(tmp_path, FakeAgent())
        live.set_agents(2)
        assert set(live.agent.withheld) == web_sessions.RUN_STARTERS
        assert "enqueue_task" in live.agent.withheld["run_optimization"]
        # Back to one, and the agent has its own runs again.
        live.set_agents(1)
        assert set(live.agent.withheld) == web_sessions.FLEET_TOOLS

    asyncio.run(run())


def test_the_allowance_names_the_tools_a_campaign_is_run_with(tmp_path):
    """A number alone leaves the agent to guess how sub-agents are reached."""

    async def run() -> None:
        agent = FakeAgent()
        live = make_session(tmp_path, agent)
        live.set_agents(4)
        assert live.start_turn("optimise this series")
        await idle(live)
        for tool in ("start_campaign", "conduct_campaign", "enqueue_task", "enqueue_tasks"):
            assert tool in agent.asked[0]

    asyncio.run(run())


# -- what the conducting agent is told it is doing --------------------------


def test_the_agent_the_user_talks_to_is_told_how_to_conduct():
    """The allowance says a campaign is allowed; this says how to run one."""
    from reinvent4_agent.system_prompt import CONDUCTOR_PLAYBOOK

    assert "Conducting a campaign" in CONDUCTOR_PLAYBOOK
    assert "cannot see this conversation" in CONDUCTOR_PLAYBOOK


def test_it_is_told_to_keep_the_sub_agents_working():
    """What the user asked auto mode for: nobody idle while the budget runs."""
    from reinvent4_agent.system_prompt import CONDUCTOR_PLAYBOOK

    assert "no sub-agent should be idle" in CONDUCTOR_PLAYBOOK


def test_it_is_told_a_sub_agent_cannot_come_back_with_a_question():
    """The other half of the sub-agent's briefing, said to the side that
    writes the task."""
    from reinvent4_agent.system_prompt import CONDUCTOR_PLAYBOOK

    assert "cannot ask you a question" in CONDUCTOR_PLAYBOOK


def test_the_conductor_half_is_added_to_the_chemistry_not_swapped_for_it():
    from reinvent4_agent.system_prompt import CONDUCTOR_PLAYBOOK, SYSTEM_PROMPT

    prompt = SYSTEM_PROMPT + CONDUCTOR_PLAYBOOK
    assert "Medicinal Chemistry Knowledge" in prompt
    assert "Conducting a campaign" in prompt
