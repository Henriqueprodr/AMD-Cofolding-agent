# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for what wakes an agent working on its own, and what it is told.

The gateway follows the runs itself and wakes the agent only with news: a
run that ended, a new problem with one, the user handing over, or a long
quiet spell.  These drive ``LiveSession`` and ``SessionManager`` with fake
agents and fake run readers, so they need neither FastAPI nor an LLM.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from pathlib import Path
from typing import Any

import pytest

from conftest import asked_question
from reinvent4_agent.web import campaign
from reinvent4_agent.web import sessions as web_sessions
from reinvent4_agent.web.sessions import LiveSession, SessionManager, SessionStore

SESSION_ID = "s-20260101-000000-abcdef"


class FakeBackend:
    def __init__(self) -> None:
        self.reasoning_effort: str | None = None

    def export_history(self) -> list[dict]:
        return []


class FakeAgent:
    """Answers at once and records what it was asked, like the real one's surface."""

    def __init__(self, answer: str = "working on it") -> None:
        self.asked: list[str] = []
        self.answer = answer
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.context_tokens = 0
        self.local_tools: dict[str, Any] = {}
        self.resets = 0
        self.backend = FakeBackend()

    @property
    def prompts(self) -> list[str]:
        """What was asked, without the standing note on every turn."""
        return [asked_question(one) for one in self.asked]

    async def chat(self, content: str) -> str:
        self.asked.append(content)
        return self.answer

    def reset_history(self) -> None:
        self.resets += 1


def make_store(tmp_path: Path) -> SessionStore:
    return SessionStore(tmp_path / "sessions")


def make_session(tmp_path: Path, agent: FakeAgent, runs: dict[str, str] | None = None) -> LiveSession:
    store = make_store(tmp_path)
    meta = {"id": SESSION_ID, "title": "BRD4", "created": "", "updated": "", "turns": 1}
    store.create(meta)
    live = LiveSession(meta=meta, store=store, agent=agent)
    live.auto_delay = 0.0
    live.auto_idle_delay = 0.0
    if runs is not None:
        state = dict(runs)

        async def observe(session: LiveSession) -> None:
            session.note_runs(state)

        live.observe = observe
        live.runs_state = state  # type: ignore[attr-defined]
    return live


async def wait_until(check: Any, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not check():
        if time.monotonic() > deadline:
            raise AssertionError("The session never reached the expected state")
        await asyncio.sleep(0.005)


async def settle(live: LiveSession) -> None:
    await wait_until(lambda: not live.busy and live.auto_task is None)
    await asyncio.sleep(0.02)


async def stop(live: LiveSession) -> None:
    live.set_autonomy(0)
    if live.running is not None:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(asyncio.shield(live.running), timeout=5)


# -- what wakes the agent --------------------------------------------------


def test_handing_over_the_campaign_wakes_the_agent_even_while_a_run_goes(tmp_path):
    async def run() -> None:
        agent = FakeAgent()
        live = make_session(tmp_path, agent, runs={"abc12345": "running"})
        live.set_autonomy(60)
        await wait_until(lambda: len(agent.asked) >= 1)
        await stop(live)

        assert agent.asked[0].startswith("[AUTO]")
        assert "The user has just left the campaign with you." in agent.asked[0]

    asyncio.run(run())


def test_with_a_run_going_and_nothing_new_the_agent_is_left_alone(tmp_path):
    async def run() -> None:
        agent = FakeAgent()
        live = make_session(tmp_path, agent, runs={"abc12345": "running"})
        live.set_autonomy(60)
        await wait_until(lambda: len(agent.asked) >= 1)
        await settle(live)

        # The turn did not finish the campaign, but the agent is waiting on a
        # run: no "still running" turn is paid for.
        assert len(agent.asked) == 1
        assert live.auto_on
        live.wake_if_due()
        await settle(live)
        assert len(agent.asked) == 1
        await stop(live)

    asyncio.run(run())


def test_a_stop_is_checked_at_once_even_while_a_run_goes(tmp_path):
    async def run() -> None:
        agent = FakeAgent(answer="Nothing more to do. AUTONOMY DONE")
        live = make_session(tmp_path, agent, runs={"abc12345": "running"})
        live.set_autonomy(60)
        await wait_until(lambda: len(agent.asked) >= 2)
        await stop(live)

        assert "You were about to hand the campaign back." in agent.asked[1]

    asyncio.run(run())


def test_a_run_that_ends_wakes_the_agent_with_the_news(tmp_path):
    async def run() -> None:
        agent = FakeAgent()
        live = make_session(tmp_path, agent, runs={"abc12345": "running"})
        live.set_autonomy(60)
        await wait_until(lambda: len(agent.asked) >= 1)
        await settle(live)

        live.note_runs({"abc12345": "completed"})
        live.wake_if_due()
        await wait_until(lambda: len(agent.asked) >= 2)
        await stop(live)

        assert "Since you last looked:\n- Run abc12345 completed." in agent.asked[1]

    asyncio.run(run())


def test_a_long_quiet_spell_wakes_the_agent_to_use_the_time(tmp_path, monkeypatch):
    async def run() -> None:
        monkeypatch.setattr(web_sessions, "AUTO_BACKSTOP_S", 0.0)
        agent = FakeAgent()
        live = make_session(tmp_path, agent, runs={"abc12345": "running"})
        live.set_autonomy(60)
        await wait_until(lambda: len(agent.asked) >= 1)
        await settle(live)

        live.wake_if_due()
        await wait_until(lambda: len(agent.asked) >= 2)
        await stop(live)

        assert "Runs still going: abc12345." in agent.asked[1]
        assert "use the time" in agent.asked[1]

    asyncio.run(run())


def test_a_quiet_spell_is_counted_from_the_last_turn_not_from_the_last_wake(tmp_path, monkeypatch):
    async def run() -> None:
        monkeypatch.setattr(web_sessions, "AUTO_BACKSTOP_S", 3600.0)
        agent = FakeAgent()
        live = make_session(tmp_path, agent, runs={"abc12345": "running"})
        live.meta["turns"] = 0
        live.set_autonomy(60)
        assert live.start_turn("Launch the JAK3 campaign")
        await settle(live)

        # The user's own turn just ended: an hour has not gone by.
        live.wake_if_due()
        await settle(live)
        await stop(live)

        assert agent.prompts == ["Launch the JAK3 campaign"]

    asyncio.run(run())


def test_with_no_run_to_wait_for_an_unfinished_campaign_carries_on(tmp_path):
    async def run() -> None:
        agent = FakeAgent()
        live = make_session(tmp_path, agent, runs={"abc12345": "completed"})
        live.set_autonomy(60)
        await wait_until(lambda: len(agent.asked) >= 2)
        await stop(live)

        assert "Carry on from your last turn." in agent.asked[1]

    asyncio.run(run())


def test_runs_that_ended_before_the_handover_are_not_news(tmp_path):
    live = make_session(tmp_path, FakeAgent())
    live.note_runs({"old00001": "completed"})
    live.note_runs({"old00001": "completed"})
    assert live.pending == []


def test_a_user_turn_takes_over_from_a_wake(tmp_path):
    async def run() -> None:
        agent = FakeAgent()
        live = make_session(tmp_path, agent, runs={"abc12345": "running"})
        live.set_autonomy(60)
        await wait_until(lambda: len(agent.asked) >= 1)
        await settle(live)
        assert live.start_turn("Try the morpholine series instead")
        await settle(live)
        await stop(live)

        assert agent.prompts[1] == "Try the morpholine series instead"

    asyncio.run(run())


# -- problems with runs ----------------------------------------------------


def alert(job_id: str, kind: str, message: str = "Score plateau since step 180", severity: str = "medium") -> dict:
    return {"job_id": job_id, "type": kind, "message": message, "severity": severity}


def test_a_problem_with_a_running_run_is_news_once(tmp_path):
    live = make_session(tmp_path, FakeAgent())
    live.note_runs({"abc12345": "running"})

    live.note_alerts([alert("abc12345", "score_plateau")])
    live.note_alerts([alert("abc12345", "score_plateau")])

    assert live.pending == ["Run abc12345 (medium): Score plateau since step 180"]


def test_problems_with_other_conversations_runs_are_not_ours(tmp_path):
    live = make_session(tmp_path, FakeAgent())
    live.note_runs({"abc12345": "running"})

    live.note_alerts([alert("zzz99999", "component_failure", "ADMET NaN for 90%", "high")])

    assert live.pending == []


def test_a_run_ending_is_told_once_not_twice(tmp_path):
    live = make_session(tmp_path, FakeAgent())
    live.note_runs({"abc12345": "running"})

    live.note_alerts([alert("abc12345", "job_completed", "Job abc12345 completed", "low")])
    live.note_runs({"abc12345": "completed"})

    assert live.pending == ["Run abc12345 completed."]


def test_handing_over_again_forgets_what_was_seen_before(tmp_path):
    async def run() -> None:
        live = make_session(tmp_path, FakeAgent())
        live.meta["turns"] = 0
        live.note_runs({"abc12345": "running"})
        live.note_alerts([alert("abc12345", "score_plateau")])

        live.set_autonomy(30)

        assert live.pending == []
        assert live.runs_seen == {}
        assert live.alerts_seen == set()
        live.set_autonomy(0)

    asyncio.run(run())


# -- the campaign log ------------------------------------------------------


def test_the_agent_gets_a_campaign_log_tool(tmp_path):
    agent = FakeAgent()
    make_session(tmp_path, agent)
    definition, _ = agent.local_tools[campaign.TOOL_NAME]
    assert definition["name"] == "campaign_log"
    assert set(definition["input_schema"]["properties"]) == {"goal", "finding", "next"}


def test_the_log_is_kept_published_and_survives_a_reload(tmp_path):
    agent = FakeAgent()
    live = make_session(tmp_path, agent)
    _, handler = agent.local_tools[campaign.TOOL_NAME]

    handler({"goal": "BRD4 BD1 binders with logS above -4"})
    handler({"finding": "Run abc12345: best 0.82, solubility flat after step 180", "next": "Add a morpholine seed set"})

    assert live.campaign["goal"] == "BRD4 BD1 binders with logS above -4"
    assert [f["text"] for f in live.campaign["findings"]] == ["Run abc12345: best 0.82, solubility flat after step 180"]
    events = [e for e in live.events if e["type"] == "campaign"]
    assert len(events) == 2
    assert events[-1]["campaign"]["next"] == "Add a morpholine seed set"

    reloaded = LiveSession(meta=live.meta, store=make_store(tmp_path), agent=FakeAgent())
    assert reloaded.campaign == live.campaign


# -- which campaigns this conversation owns ---------------------------------


def _claims(tmp_path, monkeypatch, name, result, is_error=False):
    """Run one tool result past a session and return what it claimed."""
    claimed: list[tuple[str, str]] = []

    class FakeLedger:
        def attach_session(self, campaign_id, session_id):
            claimed.append((campaign_id, session_id))
            return {}

    monkeypatch.setattr(
        "reinvent4_agent.campaign_client.CampaignStoreClient", FakeLedger
    )
    live = make_session(tmp_path, FakeAgent())
    live.note_tool_result(name, result, is_error)
    return claimed


def test_starting_a_campaign_records_it_against_the_conversation(tmp_path, monkeypatch):
    """The tool runs where it cannot know who called it; this end knows."""
    claimed = _claims(
        tmp_path, monkeypatch, "start_campaign", '{"campaign_id": "feed0001"}'
    )

    assert claimed == [("feed0001", SESSION_ID)]


def test_taking_over_conducting_claims_it_too(tmp_path, monkeypatch):
    """Whoever is conducting it is the conversation driving it."""
    claimed = _claims(
        tmp_path, monkeypatch, "conduct_campaign", '{"campaign_id": "feed0001"}'
    )

    assert claimed == [("feed0001", SESSION_ID)]


def test_a_tool_that_failed_claims_nothing(tmp_path, monkeypatch):
    claimed = _claims(
        tmp_path, monkeypatch, "start_campaign",
        '{"campaign_id": "feed0001"}', is_error=True,
    )

    assert claimed == []


def test_other_tools_are_left_alone(tmp_path, monkeypatch):
    """Reading a campaign is not owning it."""
    claimed = _claims(
        tmp_path, monkeypatch, "get_campaign_state", '{"campaign_id": "feed0001"}'
    )

    assert claimed == []


def test_an_answer_naming_no_campaign_claims_nothing(tmp_path, monkeypatch):
    claimed = _claims(
        tmp_path, monkeypatch, "start_campaign", '{"error": "no base TTP"}'
    )

    assert claimed == []


def test_a_ledger_that_is_away_does_not_break_the_turn(tmp_path, monkeypatch):
    """The campaign is running either way; only the claim is lost."""
    from reinvent4_agent.campaign_client import StoreUnavailable

    class Broken:
        def attach_session(self, campaign_id, session_id):
            raise StoreUnavailable("the ledger is unreachable")

    monkeypatch.setattr(
        "reinvent4_agent.campaign_client.CampaignStoreClient", Broken
    )
    live = make_session(tmp_path, FakeAgent())

    live.note_tool_result("start_campaign", '{"campaign_id": "feed0001"}', False)


def test_a_call_that_changes_nothing_is_refused(tmp_path):
    agent = FakeAgent()
    make_session(tmp_path, agent)
    _, handler = agent.local_tools[campaign.TOOL_NAME]
    with pytest.raises(ValueError):
        handler({})


def test_every_wake_shows_the_log(tmp_path):
    async def run() -> None:
        agent = FakeAgent()
        live = make_session(tmp_path, agent, runs={})
        _, handler = agent.local_tools[campaign.TOOL_NAME]
        handler({"goal": "Soluble JAK3 inhibitors", "next": "Cofold the top 5"})
        live.set_autonomy(60)
        await wait_until(lambda: len(agent.asked) >= 1)
        await stop(live)

        assert "Goal: Soluble JAK3 inhibitors" in agent.asked[0]
        assert "Next: Cofold the top 5" in agent.asked[0]

    asyncio.run(run())


def test_without_a_log_the_wake_asks_for_the_goal_first(tmp_path):
    async def run() -> None:
        agent = FakeAgent()
        live = make_session(tmp_path, agent, runs={})
        live.set_autonomy(60)
        await wait_until(lambda: len(agent.asked) >= 1)
        await stop(live)

        assert "set the goal with campaign_log" in agent.asked[0]

    asyncio.run(run())


# -- what the sub-agents send back -----------------------------------------


def done(event_id: int, task_id: str = "t-0001", **payload: Any) -> dict:
    body = {"task_kind": "design", "title": "JAK3 selectivity", "summary": "12 candidates, best 0.81"}
    body.update(payload)
    return {"event_id": event_id, "task_id": task_id, "kind": "task_done", "payload": body}


def news(events: list[dict], idle: int = 0, queued: int = 0, last: int | None = None, alive: int = 1) -> dict:
    return {
        "events": events,
        "last_event_id": last if last is not None else (events[-1]["event_id"] if events else 0),
        "idle": idle,
        "alive": alive,
        "queued": queued,
    }


def test_a_report_from_a_sub_agent_is_news(tmp_path):
    live = make_session(tmp_path, FakeAgent())
    live.campaigns_seen["feed0001"] = 0

    live.note_campaign("feed0001", news([done(7)]))

    assert live.pending == [
        "Sub-agent reported on JAK3 selectivity (task t-0001): 12 candidates, "
        "best 0.81. Read it with get_task_reports, and give the sub-agent it "
        "freed its next task."
    ]


def test_a_report_wakes_the_conductor_of_an_open_campaign_without_autonomy(tmp_path):
    """Without it a sub-agent that reported sits idle until the user next types."""
    live = make_session(tmp_path, FakeAgent())
    woken: list[str] = []
    live._wake = woken.append
    live.campaigns_seen["feed0001"] = 0
    live.campaigns_open.add("feed0001")
    assert not live.auto_on

    live.note_campaign("feed0001", news([done(7)]))
    live.wake_if_due()
    assert woken == ["events"]

    # Closed: its news is the user's to read when they come back.
    live.note_campaign("feed0001", {**news([done(8)]), "open": False})
    live.wake_if_due()
    assert woken == ["events"]


def test_a_report_cuts_short_the_pause_after_an_answer_that_saw_nothing_to_do(tmp_path):
    """The BRD4 campaign: S1 reported, and waited out the conductor's five minutes."""

    async def run() -> None:
        agent = FakeAgent(answer="Both tasks are in flight. AUTONOMY DONE")
        live = make_session(tmp_path, agent, runs={})
        live.auto_idle_delay = 300.0
        live.campaigns_seen["feed0001"] = 0
        live.campaigns_open.add("feed0001")
        live.set_autonomy(60)
        await wait_until(lambda: len(agent.asked) >= 2)
        await wait_until(lambda: not live.busy and live.auto_task is not None)

        live.note_campaign("feed0001", news([done(7)]))
        live.wake_if_due()
        await wait_until(lambda: len(agent.asked) >= 3, timeout=2)
        await stop(live)

        assert "Sub-agent reported on JAK3 selectivity" in agent.asked[2]

    asyncio.run(run())


def test_a_pause_with_nothing_new_is_left_to_run(tmp_path):
    async def run() -> None:
        agent = FakeAgent(answer="Both tasks are in flight. AUTONOMY DONE")
        live = make_session(tmp_path, agent, runs={})
        live.auto_idle_delay = 300.0
        live.set_autonomy(60)
        await wait_until(lambda: len(agent.asked) >= 2)
        await wait_until(lambda: not live.busy and live.auto_task is not None)

        live.wake_if_due()
        await asyncio.sleep(0.05)
        assert len(agent.asked) == 2
        assert live.auto_task is not None
        await stop(live)

    asyncio.run(run())


def test_a_report_that_proposed_next_tasks_says_so(tmp_path):
    live = make_session(tmp_path, FakeAgent())
    live.campaigns_seen["feed0001"] = 0

    live.note_campaign("feed0001", news([done(7, n_follow_ups=2)]))

    assert "It proposed 2 next task(s)" in live.pending[0]
    assert "take_proposals" in live.pending[0]


def test_a_ready_queue_running_low_is_told_once(tmp_path):
    """Kept just under target, it would otherwise wake the conductor every look."""
    live = make_session(tmp_path, FakeAgent())
    live.campaigns_seen["feed0001"] = 0

    live.note_campaign("feed0001", {**news([], queued=1), "backlog": 2})
    live.note_campaign("feed0001", {**news([], queued=1), "backlog": 2})

    assert len(live.pending) == 1
    assert "Only 1 task(s) are queued" in live.pending[0]
    assert "enqueue_tasks" in live.pending[0]


def test_a_refilled_queue_is_told_again_when_it_next_runs_low(tmp_path):
    live = make_session(tmp_path, FakeAgent())
    live.campaigns_seen["feed0001"] = 0

    live.note_campaign("feed0001", {**news([], queued=0, idle=0), "backlog": 2})
    live.note_campaign("feed0001", {**news([], queued=3), "backlog": 2})
    live.note_campaign("feed0001", {**news([], queued=1), "backlog": 2})

    assert len(live.pending) == 2


def test_a_closed_campaigns_queue_is_nobodys_concern(tmp_path):
    live = make_session(tmp_path, FakeAgent())
    live.campaigns_seen["feed0001"] = 0

    live.note_campaign("feed0001", {**news([], queued=0), "backlog": 2, "open": False})

    assert live.pending == []


def test_a_report_already_read_is_not_told_twice(tmp_path):
    live = make_session(tmp_path, FakeAgent())
    live.campaigns_seen["feed0001"] = 0

    live.note_campaign("feed0001", news([done(7)]))
    live.pending.clear()
    live.note_campaign("feed0001", news([], last=7))

    assert live.pending == []
    assert live.campaigns_seen["feed0001"] == 7


def test_a_task_that_failed_is_news_too(tmp_path):
    """A sub-agent that came back empty still freed its slot."""
    live = make_session(tmp_path, FakeAgent())
    live.campaigns_seen["feed0001"] = 0

    live.note_campaign("feed0001", news([{
        "event_id": 3, "task_id": "t-0002", "kind": "task_failed",
        "payload": {"title": "BRD4 retrosynthesis", "error": "no route under 6 steps"},
    }]))

    assert live.pending == [
        "Task t-0002 (BRD4 retrosynthesis) did not finish: no route under 6 steps"
    ]


def test_idle_sub_agents_with_nothing_queued_are_news(tmp_path):
    """The state a campaign stalls in: budget running, nobody working."""
    live = make_session(tmp_path, FakeAgent())
    live.campaigns_seen["feed0001"] = 0

    live.note_campaign("feed0001", news([], idle=3))

    assert live.pending == [
        "3 sub-agent(s) are idle on campaign feed0001 with nothing queued. "
        "Give them work or close the campaign."
    ]


def test_a_campaign_that_stays_stalled_is_told_once(tmp_path):
    """It woke the conductor every half minute to say the same thing, and a
    conductor waiting on the sub-agent that is working cannot act on it."""
    live = make_session(tmp_path, FakeAgent())
    live.campaigns_seen["feed0001"] = 0

    for _ in range(4):
        live.note_campaign("feed0001", news([], idle=1))

    assert len(live.pending) == 1


def test_a_campaign_that_stalls_again_is_told_again(tmp_path):
    """Told once while it lasts, not once for the whole campaign."""
    live = make_session(tmp_path, FakeAgent())
    live.campaigns_seen["feed0001"] = 0

    live.note_campaign("feed0001", news([], idle=1))
    live.note_campaign("feed0001", news([], idle=1, queued=2, alive=1))
    live.pending.clear()
    live.note_campaign("feed0001", news([], idle=1))

    assert len(live.pending) == 1


def test_sub_agents_with_work_waiting_for_them_are_not_news(tmp_path):
    live = make_session(tmp_path, FakeAgent())
    live.campaigns_seen["feed0001"] = 0

    live.note_campaign("feed0001", news([], idle=2, queued=4, alive=2))

    assert live.pending == []


def test_queued_work_with_no_sub_agent_to_do_it_is_news(tmp_path):
    """The pool rests at zero; a scale-up that did not happen looks like this."""
    live = make_session(tmp_path, FakeAgent())
    live.campaigns_seen["feed0001"] = 0

    live.note_campaign("feed0001", news([], queued=4, alive=0))

    assert live.pending == [
        "4 task(s) are queued on campaign feed0001 and no sub-agent is running "
        "to take them. Check the campaign with get_campaign_status before "
        "queueing more."
    ]


def test_a_reader_that_does_not_count_the_pool_raises_no_alarm(tmp_path):
    live = make_session(tmp_path, FakeAgent())
    live.campaigns_seen["feed0001"] = 0

    live.note_campaign("feed0001", {"events": [], "last_event_id": 3, "queued": 4})

    assert live.pending == []


def test_a_report_is_told_rather_than_the_idleness_it_leaves_behind(tmp_path):
    """Both are true at once; the report is what there is to act on."""
    live = make_session(tmp_path, FakeAgent())
    live.campaigns_seen["feed0001"] = 0

    live.note_campaign("feed0001", news([done(7)], idle=1))

    assert len(live.pending) == 1
    assert live.pending[0].startswith("Sub-agent reported")


def test_a_campaign_with_sub_agents_working_is_looked_at_after_a_quiet_spell(tmp_path, monkeypatch):
    """No run of the conversation's own is going, so only this wakes it."""

    async def run() -> None:
        monkeypatch.setattr(web_sessions, "AUTO_BACKSTOP_S", 0.0)
        agent = FakeAgent()
        live = make_session(tmp_path, agent, runs={})
        live.campaigns_seen["feed0001"] = 4
        live.set_autonomy(60)
        await wait_until(lambda: len(agent.asked) >= 2)
        await stop(live)

    asyncio.run(run())


def test_the_watcher_wakes_a_session_whose_sub_agent_reported(tmp_path):
    async def run() -> None:
        agents: list[FakeAgent] = []
        feed = [done(9)]
        asked_for: list[dict[str, int]] = []

        async def run_states(ids: list[str]) -> dict[str, dict[str, str]]:
            return {sid: {} for sid in ids}

        async def job_alerts() -> list[dict[str, Any]]:
            return []

        async def campaign_news(since: dict[str, int]) -> dict[str, dict[str, Any]]:
            asked_for.append(dict(since))
            out = {"feed0001": news(list(feed))}
            feed.clear()
            return out

        manager = SessionManager(make_store(tmp_path), lambda handler: agents.append(FakeAgent()) or agents[-1])
        manager.watch_interval = 0.01
        live = manager.create()
        live.auto_delay = 0.0
        live.meta["turns"] = 1
        live.campaigns_seen["feed0001"] = 4
        # Autonomy on before the watcher starts: the first tick consumes the
        # feed, so a tick landing between the two would clear the news while
        # nothing was listening and the session would never be woken. Rare
        # when this test runs alone, reachable under a loaded full suite.
        live.set_autonomy(60)
        manager.watch(run_states, job_alerts, campaign_news)
        try:
            await wait_until(lambda: any("Sub-agent reported" in asked for asked in agents[0].asked))
            assert asked_for[0] == {"feed0001": 4}
        finally:
            live.set_autonomy(0)
            await manager.close()

    asyncio.run(run())


def test_the_watcher_reads_no_campaign_for_a_conversation_that_owns_none(tmp_path):
    async def run() -> None:
        async def run_states(ids: list[str]) -> dict[str, dict[str, str]]:
            return {sid: {} for sid in ids}

        async def job_alerts() -> list[dict[str, Any]]:
            return []

        async def campaign_news(since: dict[str, int]) -> dict[str, dict[str, Any]]:
            raise AssertionError("no campaign should be read")

        manager = SessionManager(make_store(tmp_path), lambda handler: FakeAgent())
        manager.watch_interval = 0.01
        live = manager.create()
        live.auto_delay = 0.0
        live.meta["turns"] = 1
        manager.watch(run_states, job_alerts, campaign_news)
        try:
            live.set_autonomy(60)
            await asyncio.sleep(0.05)
        finally:
            live.set_autonomy(0)
            await manager.close()

    asyncio.run(run())


# -- keeping turns small ---------------------------------------------------


def test_a_long_history_is_condensed_before_an_automatic_turn(tmp_path):
    async def run() -> None:
        agent = FakeAgent()
        live = make_session(tmp_path, agent, runs={"abc12345": "completed"})
        live.publish({"type": "user_message", "content": "Design soluble JAK3 inhibitors"})
        live.publish({"type": "assistant_message", "content": "Launched run abc12345."})
        agent.context_tokens = web_sessions.COMPACT_ABOVE_TOKENS + 1
        live.set_autonomy(60)
        await wait_until(lambda: len(agent.asked) >= 1)
        await stop(live)

        assert agent.resets == 1
        assert "condensed" in agent.asked[0]
        assert "- Design soluble JAK3 inhibitors" in agent.asked[0]
        assert "Launched run abc12345." in agent.asked[0]
        assert "abc12345 (completed)" in agent.asked[0]

    asyncio.run(run())


def test_a_short_history_is_kept(tmp_path):
    async def run() -> None:
        agent = FakeAgent(answer="Nothing to add. AUTONOMY DONE")
        live = make_session(tmp_path, agent, runs={})
        agent.context_tokens = 5_000
        live.set_autonomy(60)
        await wait_until(lambda: len(agent.asked) >= 1)
        await stop(live)

        assert agent.resets == 0
        assert "condensed" not in agent.asked[0]

    asyncio.run(run())


def test_a_long_history_is_condensed_before_the_users_own_turn(tmp_path):
    """The conversation the user drives fills the context like any other."""

    async def run() -> None:
        agent = FakeAgent()
        live = make_session(tmp_path, agent, runs={})
        live.publish({"type": "user_message", "content": "Design soluble JAK3 inhibitors"})
        live.publish({"type": "assistant_message", "content": "Here are five."})
        agent.context_tokens = web_sessions.COMPACT_ABOVE_TOKENS + 1

        live.start_turn("Now check their ADMET")
        await wait_until(lambda: len(agent.asked) >= 1)
        await settle(live)

        assert agent.resets == 1
        assert "condensed" in agent.asked[0]
        # The question itself still comes last, after the brief.
        assert agent.prompts[0].rstrip().endswith("Now check their ADMET")

    asyncio.run(run())


def test_how_much_history_is_kept_follows_the_window_the_endpoint_serves(tmp_path):
    """A 32k endpoint cannot carry what a 128k one can, whatever the model."""

    async def run() -> None:
        agent = FakeAgent()
        live = make_session(tmp_path, agent, runs={})
        agent.context_window = 32_768
        agent.context_tokens = 30_000

        live.start_turn("Carry on")
        await wait_until(lambda: len(agent.asked) >= 1)
        await settle(live)

        # Well under the 100k fallback, but over 70% of what is served.
        assert agent.resets == 1

    asyncio.run(run())


# -- the watcher -----------------------------------------------------------


def test_the_watcher_wakes_a_session_whose_run_ended(tmp_path):
    async def run() -> None:
        agents: list[FakeAgent] = []

        def factory(handler: Any) -> FakeAgent:
            agents.append(FakeAgent())
            return agents[-1]

        states = {"abc12345": "running"}
        looked: list[list[str]] = []

        async def run_states(ids: list[str]) -> dict[str, dict[str, str]]:
            looked.append(ids)
            return {sid: dict(states) for sid in ids}

        async def job_alerts() -> list[dict[str, Any]]:
            return []

        manager = SessionManager(make_store(tmp_path), factory)
        manager.watch_interval = 0.01
        live = manager.create()
        live.auto_delay = 0.0
        live.meta["turns"] = 1
        manager.watch(run_states, job_alerts)
        try:
            live.set_autonomy(60)
            await wait_until(lambda: len(agents[0].asked) >= 1)
            await settle(live)
            assert len(agents[0].asked) == 1

            states["abc12345"] = "completed"
            await wait_until(lambda: len(agents[0].asked) >= 2)
            assert "Run abc12345 completed." in agents[0].asked[1]
        finally:
            live.set_autonomy(0)
            await manager.close()

    asyncio.run(run())


def test_the_watcher_reads_nothing_while_nobody_works_on_their_own(tmp_path):
    async def run() -> None:
        looked: list[list[str]] = []

        async def run_states(ids: list[str]) -> dict[str, dict[str, str]]:
            looked.append(ids)
            return {}

        async def job_alerts() -> list[dict[str, Any]]:
            raise AssertionError("no alerts should be read")

        manager = SessionManager(make_store(tmp_path), lambda handler: FakeAgent())
        manager.watch_interval = 0.01
        manager.create()
        manager.watch(run_states, job_alerts)
        await asyncio.sleep(0.05)
        await manager.close()

        assert all(ids == [] for ids in looked)

    asyncio.run(run())
