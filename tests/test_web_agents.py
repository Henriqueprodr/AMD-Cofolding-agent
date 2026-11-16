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


def test_working_alone_adds_nothing_about_a_team(tmp_path):
    """What is said at one is how to keep the campaign, not how to dispatch.

    A line offering one sub-agent reads as a standing instruction to run
    campaigns, which is why nothing was said here at all; the campaign note
    took its place once a conversation working alone could keep one.
    """

    async def run() -> None:
        agent = FakeAgent()
        live = make_session(tmp_path, agent)
        assert live.start_turn("optimise this series")
        await idle(live)
        asked = agent.asked[0]
        assert asked.startswith("optimise this series")
        for phrase in ("Team size", "sub-agent at once", "enqueue_tasks"):
            assert phrase not in asked, phrase

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


def test_working_alone_withholds_only_what_needs_somebody_to_dispatch_to(tmp_path):
    """At one there are no sub-agents, so a queue would have nobody to take from it."""

    live = make_session(tmp_path, FakeAgent())
    live.apply_allowance()
    assert set(live.agent.withheld) == web_sessions.DISPATCH_TOOLS
    assert "nobody to dispatch" in live.agent.withheld["enqueue_task"]


def test_working_alone_keeps_the_campaign_itself(tmp_path):
    """The campaign is the memory, and it needs no sub-agent.

    Withholding it left a conversation working alone with nothing that
    outlived its context: no plan, no record of what it had tried, no
    trajectory, where a campaign of two agents showed a line of reasoning.
    """

    live = make_session(tmp_path, FakeAgent())
    live.apply_allowance()
    for tool in web_sessions.CAMPAIGN_TOOLS:
        assert tool not in live.agent.withheld, tool


def test_two_or_more_withhold_the_run_starters_instead(tmp_path):
    async def run() -> None:
        live = make_session(tmp_path, FakeAgent())
        live.set_agents(2)
        assert set(live.agent.withheld) == web_sessions.RUN_STARTERS
        assert "enqueue_task" in live.agent.withheld["run_optimization"]
        # Back to one, and the agent has its own runs again.
        live.set_agents(1)
        assert set(live.agent.withheld) == web_sessions.DISPATCH_TOOLS

    asyncio.run(run())


def test_working_alone_is_told_to_keep_the_campaign(tmp_path):
    """Having the tools is not enough: told nothing, the agent keeps no record."""

    async def run() -> None:
        agent = FakeAgent()
        live = make_session(tmp_path, agent)
        assert live.start_turn("optimise this series")
        await idle(live)
        for phrase in ("start_campaign", "start_own_task", "finish_own_task", "atlas_finding"):
            assert phrase in agent.asked[0], phrase
        # Not an offer of one sub-agent, which reads as an instruction to dispatch.
        assert "Team size" not in agent.asked[0]

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
