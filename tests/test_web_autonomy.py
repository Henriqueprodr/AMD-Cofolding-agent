# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the autonomy budget of a chat session.

These drive ``LiveSession`` directly, with a fake agent and a store under
``tmp_path``, so they need neither FastAPI nor an LLM.  The event loop is
started per test with ``asyncio.run`` rather than a plugin.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from conftest import asked_question
from reinvent4_agent.web import sessions as web_sessions
from reinvent4_agent.web.sessions import LiveSession, SessionStore


class FakeBackend:
    def __init__(self) -> None:
        self.history: list[dict] = []
        self.reasoning_effort: str | None = None

    def export_history(self) -> list[dict]:
        return list(self.history)


class FakeAgent:
    """Answers at once, recording what it was asked.

    *answer* replaces the default reply, which is how a test makes the agent
    say it is done; *hang* makes a turn run until it is cancelled.
    """

    def __init__(self, answer: str = "working on it", hang: bool = False) -> None:
        self.asked: list[str] = []
        self.answer = answer
        self.hang = hang
        self.cancelled = False
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.backend = FakeBackend()
        # Where a session installs the tools that belong to it.
        self.local_tools = {}

    @property
    def prompts(self) -> list[str]:
        """What was asked, without the standing note on every turn."""
        return [asked_question(one) for one in self.asked]

    async def chat(self, content: str) -> str:
        self.asked.append(content)
        if self.hang:
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        return self.answer


def make_session(tmp_path: Path, agent: FakeAgent) -> LiveSession:
    store = SessionStore(tmp_path / "sessions")
    meta = {"id": "s-20260101-000000-abcdef", "title": "BRD4", "created": "", "updated": "", "turns": 0}
    store.create(meta)
    live = LiveSession(meta=meta, store=store, agent=agent)
    # Real continuations wait a minute or five; the tests must not.
    live.auto_delay = 0.0
    live.auto_idle_delay = 0.0
    return live


def published(live: LiveSession, kind: str) -> list[dict[str, Any]]:
    return [event for event in live.events if event.get("type") == kind]


def _drain(queue: Any) -> list[dict[str, Any]]:
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events


def busy_statuses(live: LiveSession) -> list[dict[str, Any]]:
    return [event for event in published(live, "status") if event["state"] == "busy"]


async def wait_until(check: Any, timeout: float = 5.0) -> None:
    """Run the loop until *check* holds.  Fails rather than hanging."""
    deadline = time.monotonic() + timeout
    while not check():
        if time.monotonic() > deadline:
            raise AssertionError("The session never reached the expected state")
        await asyncio.sleep(0.005)


async def quiesce(live: LiveSession) -> None:
    """Stop the autonomy loop and wait out the turn it is on."""
    live.set_autonomy(0)
    if live.running is not None:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(asyncio.shield(live.running), timeout=5)


async def idle(live: LiveSession) -> None:
    """Wait until nothing runs and nothing is queued."""
    await wait_until(lambda: not live.busy and live.auto_task is None)


async def settled(live: LiveSession, turns: int) -> None:
    """Wait for *turns* turns, then stop the loop and let the last one finish.

    Auto mode no longer winds down on its own, so a test with the budget on
    has to say when it has seen enough; waiting for the loop to go quiet
    would wait for the budget.
    """
    await wait_until(lambda: len(live.agent.asked) >= turns)
    await quiesce(live)


# -- the budget itself -----------------------------------------------------


def test_the_budget_is_set_extended_and_cleared(tmp_path):
    async def run() -> None:
        live = make_session(tmp_path, FakeAgent())
        seen: list[dict[str, Any]] = []
        queue = live.subscribe()

        assert live.auto_until is None
        assert live.summary()["auto_until"] is None

        first = live.set_autonomy(30)
        assert live.auto_on
        assert live.summary()["auto_until"] == first.isoformat(timespec="seconds")

        later = live.set_autonomy(120)
        assert later > first

        # Far beyond the cap: the deadline is the cap, not what was asked.
        capped = live.set_autonomy(10_000)
        assert (capped - later).total_seconds() < web_sessions.MAX_AUTONOMY_MINUTES * 60

        assert live.set_autonomy(0) is None
        assert not live.auto_on
        assert live.summary()["auto_until"] is None

        while not queue.empty():
            seen.append(queue.get_nowait())
        assert [event["type"] for event in seen] == ["auto"] * 4
        assert seen[-1]["until"] is None
        # The budget is live state, not part of the conversation.
        assert published(live, "auto") == []

    asyncio.run(run())


def test_a_budget_that_has_run_out_reports_nothing_left(tmp_path):
    async def run() -> None:
        live = make_session(tmp_path, FakeAgent())
        live.set_autonomy(60)
        live.auto_until = live.auto_until.replace(year=2000)
        assert not live.auto_on
        assert live.summary()["auto_until"] is None

    asyncio.run(run())


# -- continuation turns ----------------------------------------------------


def test_a_turn_continues_by_itself_while_the_budget_lasts(tmp_path):
    async def run() -> None:
        agent = FakeAgent()
        live = make_session(tmp_path, agent)
        live.set_autonomy(60)
        assert live.start_turn("Design ligands for BRD4")
        await wait_until(lambda: len(agent.asked) >= 2)
        await quiesce(live)

        assert agent.prompts[0] == "Design ligands for BRD4"
        assert agent.asked[1].startswith("[AUTO]")
        # Only the question is recorded as a user message; the instruction is not.
        assert len(published(live, "user_message")) == 1
        assert "auto" not in busy_statuses(live)[0]
        assert busy_statuses(live)[1]["auto"] is True

    asyncio.run(run())


def test_nothing_continues_once_the_budget_has_expired(tmp_path):
    async def run() -> None:
        agent = FakeAgent()
        live = make_session(tmp_path, agent)
        live.auto_delay = 0.05
        live.set_autonomy(60)
        assert live.start_turn("Design ligands for BRD4")
        await wait_until(lambda: live.auto_task is not None)
        # The budget runs out while the continuation waits out its pause.
        live.auto_until = live.auto_until.replace(year=2000)
        await idle(live)

        assert agent.prompts == ["Design ligands for BRD4"]

    asyncio.run(run())


def test_without_a_budget_a_turn_stands_alone(tmp_path):
    async def run() -> None:
        agent = FakeAgent()
        live = make_session(tmp_path, agent)
        assert live.start_turn("Design ligands for BRD4")
        await idle(live)
        await asyncio.sleep(0.05)
        assert agent.prompts == ["Design ligands for BRD4"]

    asyncio.run(run())


def test_turning_the_budget_on_between_turns_picks_the_campaign_up(tmp_path):
    async def run() -> None:
        agent = FakeAgent()
        live = make_session(tmp_path, agent)
        assert live.start_turn("Launch the campaign")
        await idle(live)
        assert agent.prompts == ["Launch the campaign"]

        live.set_autonomy(60)
        await wait_until(lambda: len(agent.asked) >= 2)
        await quiesce(live)
        assert agent.asked[1].startswith("[AUTO]")
        assert busy_statuses(live)[1]["auto"] is True

    asyncio.run(run())


def test_the_agent_saying_it_is_done_does_not_end_the_budget(tmp_path):
    """Auto mode is the user saying "carry on without me"; only they end it."""

    async def run() -> None:
        agent = FakeAgent(answer="Nothing left to try. AUTONOMY DONE")
        live = make_session(tmp_path, agent)
        live.set_autonomy(60)
        assert live.start_turn("Design ligands for BRD4")
        await settled(live, 3)

        # The first "done" is checked, and every one after it is told to look
        # again rather than taken as the end of the campaign.
        assert "You were about to hand the campaign back." in agent.asked[1]
        assert "look again rather than wait" in agent.asked[2]

    asyncio.run(run())


def test_the_looking_again_turn_waits_longer_than_a_working_one(tmp_path):
    """Once it has twice found nothing, what changes next is a run finishing."""

    async def run() -> None:
        waits: list[float] = []
        live = make_session(tmp_path, FakeAgent(answer="Nothing left. AUTONOMY DONE"))
        live.auto_idle_delay = web_sessions.AUTO_IDLE_DELAY_S
        live._schedule_continuation = lambda delay, reason="continue": waits.append(delay)
        live.set_autonomy(60)
        assert live.start_turn("Design ligands for BRD4")
        await wait_until(lambda: bool(waits))

        live.done_once = True
        await live._run_turn("again")

        assert waits == [0.0, web_sessions.AUTO_IDLE_DELAY_S]

    asyncio.run(run())


def test_work_found_at_the_check_keeps_the_loop_going(tmp_path):
    class Scripted(FakeAgent):
        def __init__(self, answers: list[str]) -> None:
            super().__init__()
            self.answers = answers

        async def chat(self, content: str) -> str:
            self.asked.append(content)
            return self.answers.pop(0) if self.answers else "Waiting on the run."

    async def run() -> None:
        agent = Scripted(
            [
                "Nothing left to try. AUTONOMY DONE",
                "Started a run with the fixed TTP.",
                "Still nothing left. AUTONOMY DONE",
                "Really nothing left. AUTONOMY DONE",
            ]
        )
        live = make_session(tmp_path, agent)
        live.set_autonomy(60)
        assert live.start_turn("Design ligands for BRD4")
        await settled(live, 4)

        checks = ["You were about to hand the campaign back." in asked for asked in agent.asked]
        # Work in between makes the next stop a first one again, so it is checked too.
        assert checks[:4] == [False, True, False, True]

    asyncio.run(run())


def test_the_stop_phrase_is_not_part_of_the_answer(tmp_path):
    async def run() -> None:
        agent = FakeAgent(answer="The campaign is finished.\n\n**AUTONOMY DONE**")
        live = make_session(tmp_path, agent)
        live.set_autonomy(60)
        assert live.start_turn("Design ligands for BRD4")
        await settled(live, 1)

        assert published(live, "assistant_message")[0]["content"] == "The campaign is finished."

    asyncio.run(run())


def test_nothing_but_the_budget_limits_how_many_turns_it_takes(tmp_path):
    """There used to be a cap of fifty, which stopped a campaign mid-flight."""

    async def run() -> None:
        agent = FakeAgent()
        live = make_session(tmp_path, agent)
        live.set_autonomy(120)
        assert live.start_turn("Design ligands for BRD4")
        await settled(live, 60)

        assert agent.prompts[0] == "Design ligands for BRD4"
        assert all(asked.startswith("[AUTO]") for asked in agent.asked[1:])

    asyncio.run(run())


def test_a_budget_that_runs_out_ends_it_and_says_so(tmp_path):
    async def run() -> None:
        agent = FakeAgent()
        live = make_session(tmp_path, agent)
        queue = live.subscribe()
        live.set_autonomy(60)
        assert live.start_turn("Design ligands for BRD4")
        await wait_until(lambda: len(agent.asked) >= 2)
        # The deadline passes mid-campaign, which is the budget running out
        # rather than anybody deciding the work is finished.
        live.auto_until = datetime.now(timezone.utc) - timedelta(seconds=1)
        await wait_until(lambda: not live.busy and live.auto_until is None)

        ended = [event for event in _drain(queue) if event["type"] == "auto"]
        assert ended[-1]["until"] is None

    asyncio.run(run())


# -- the hard safety rules -------------------------------------------------


def test_a_cancelled_turn_never_continues(tmp_path):
    async def run() -> None:
        agent = FakeAgent(hang=True)
        live = make_session(tmp_path, agent)
        live.set_autonomy(60)
        assert live.start_turn("Design ligands for BRD4")
        await wait_until(lambda: agent.prompts == ["Design ligands for BRD4"])
        assert live.cancel()
        await idle(live)
        await asyncio.sleep(0.05)

        assert agent.cancelled
        assert agent.prompts == ["Design ligands for BRD4"]
        assert published(live, "cancelled")
        # Stop means stop: the budget cannot start it up again a minute later.
        assert not live.auto_on

    asyncio.run(run())


def test_a_failed_turn_is_picked_up_again_rather_than_handed_back(tmp_path):
    """What fails a turn is usually a service, and services come back."""

    async def run() -> None:
        agent = FakeAgent()

        async def fail(content: str) -> str:
            agent.asked.append(content)
            raise RuntimeError("Boltz is unreachable")

        agent.chat = fail
        live = make_session(tmp_path, agent)
        waits: list[float] = []
        live._schedule_continuation = lambda delay, reason="continue": waits.append(delay)
        live.set_autonomy(60)
        assert live.start_turn("Design ligands for BRD4")
        await wait_until(lambda: bool(waits))
        await live._run_turn("again")

        assert published(live, "error")[0]["message"] == "Boltz is unreachable"
        assert live.auto_on
        # Each failure in a row waits longer, so a service that is down for
        # an hour is not retried two hundred times.
        assert waits == [web_sessions.AUTO_RETRY_DELAY_S, web_sessions.AUTO_RETRY_DELAY_S * 2]

    asyncio.run(run())


def test_the_retry_wait_stops_growing(tmp_path):
    async def run() -> None:
        live = make_session(tmp_path, FakeAgent())
        live.auto_failures = 20
        assert live._retry_delay() == web_sessions.AUTO_RETRY_MAX_S

    asyncio.run(run())


def test_a_user_message_takes_priority_over_a_pending_continuation(tmp_path):
    async def run() -> None:
        agent = FakeAgent()
        live = make_session(tmp_path, agent)
        live.auto_delay = 30.0
        live.set_autonomy(60)
        assert live.start_turn("Design ligands for BRD4")
        await wait_until(lambda: live.auto_task is not None)
        pending = live.auto_task

        assert live.start_turn("Use the other scaffold instead")
        assert live.auto_task is None
        assert live.auto_turns == 0
        await quiesce(live)
        await asyncio.sleep(0)

        assert pending.cancelled()
        assert agent.prompts == ["Design ligands for BRD4", "Use the other scaffold instead"]

    asyncio.run(run())


def test_two_turns_never_run_at_once(tmp_path):
    async def run() -> None:
        agent = FakeAgent(hang=True)
        live = make_session(tmp_path, agent)
        live.set_autonomy(60)
        assert live.start_turn("Design ligands for BRD4")
        await wait_until(lambda: live.busy and agent.asked != [])
        # Kept for the running turn rather than started beside it.
        assert live.start_turn("and another")
        # A continuation that wakes while a turn runs stands down.
        await live._continue_after(0.0)
        assert agent.prompts == ["Design ligands for BRD4"]

        agent.hang = False
        live.cancel()
        await idle(live)
        assert len(agent.asked) == 2

    asyncio.run(run())


def test_a_message_sent_mid_turn_is_read_after_the_current_step(tmp_path):
    async def run() -> None:
        agent = FakeAgent(hang=True)
        live = make_session(tmp_path, agent)
        assert live.start_turn("Design ligands for BRD4")
        await wait_until(lambda: live.busy and agent.asked != [])

        assert live.start_turn("Use the morpholine series")
        shown = published(live, "user_message")[-1]
        assert shown["content"] == "Use the morpholine series"
        assert shown["interjected"] is True
        # What Agent.chat calls between tool rounds.
        assert agent.inbox() == ["Use the morpholine series"]
        assert agent.inbox() == []

        agent.hang = False
        live.cancel()
        await idle(live)
        # Read mid-turn, so it does not start a turn of its own.
        assert agent.prompts == ["Design ligands for BRD4"]

    asyncio.run(run())


def test_a_message_the_turn_never_read_starts_the_next_one(tmp_path):
    async def run() -> None:
        agent = FakeAgent(hang=True)
        live = make_session(tmp_path, agent)
        live.auto_delay = 30.0
        live.set_autonomy(60)
        assert live.start_turn("Design ligands for BRD4")
        await wait_until(lambda: live.busy and agent.asked != [])
        assert live.start_turn("Use the morpholine series")
        assert live.start_turn("and check ADMET")

        agent.hang = False
        live.running.cancel()
        await wait_until(lambda: len(agent.asked) == 2 and not live.busy)
        assert "Use the morpholine series\n\nand check ADMET" in agent.asked[1]
        # The user led that turn, so the autonomy count starts over.
        assert live.auto_turns == 0
        await quiesce(live)

    asyncio.run(run())


def test_stopping_the_budget_drops_the_queued_continuation(tmp_path):
    async def run() -> None:
        agent = FakeAgent()
        live = make_session(tmp_path, agent)
        live.auto_delay = 30.0
        live.set_autonomy(60)
        assert live.start_turn("Design ligands for BRD4")
        await wait_until(lambda: live.auto_task is not None)

        live.set_autonomy(0)
        assert live.auto_task is None
        await asyncio.sleep(0.05)
        assert agent.prompts == ["Design ligands for BRD4"]

    asyncio.run(run())
