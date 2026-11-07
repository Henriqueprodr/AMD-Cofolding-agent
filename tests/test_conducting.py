# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Conducting a campaign from inside the agent process.

The loop itself is tested in test_conductor.py; what matters here is the
ownership around it -- one loop per campaign, a thread that reports rather
than dies, and a status a person can be told.
"""

from __future__ import annotations

import threading
import time

import pytest

from reinvent4_agent.campaign_client import StoreUnavailable
from reinvent4_agent.conducting import ConductingRegistry
from reinvent4_agent.conductor import TickReport


class _Conductor:
    """Stands in for CampaignConductor: records ticks, replays outcomes."""

    def __init__(self, campaign_id="c1", outcomes=None, tick_seconds=0.01):
        self.campaign_id = campaign_id
        self.tick_seconds = tick_seconds
        self.outcomes = list(outcomes or [])
        self.ticks = 0
        self.ticked = threading.Event()

    def tick(self):
        self.ticks += 1
        self.ticked.set()
        outcome = (
            self.outcomes.pop(0) if self.outcomes
            else TickReport(phase="dispatching", in_flight=1, queued=0)
        )
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _started(registry, conductor):
    """Start a loop around *conductor* and wait for its first pass."""
    from reinvent4_agent.conducting import _Conducting

    loop = _Conducting(conductor, settings={})
    registry._loops[conductor.campaign_id] = loop
    loop.start()
    assert conductor.ticked.wait(5), "the loop never ticked"
    return loop


@pytest.fixture
def registry():
    reg = ConductingRegistry()
    yield reg
    reg.stop_all("test over")
    for loop in list(reg._loops.values()):
        loop._thread.join(timeout=5)


class TestConducting:

    def test_the_loop_ticks_until_it_is_stopped(self, registry):
        conductor = _Conductor()

        loop = _started(registry, conductor)
        registry.stop("c1")
        loop._thread.join(timeout=5)

        assert conductor.ticks >= 1
        assert loop.status().running is False

    def test_the_last_pass_is_readable(self, registry):
        conductor = _Conductor(
            outcomes=[TickReport(phase="dispatching", in_flight=0, queued=2,
                                 dispatched=2)]
        )

        loop = _started(registry, conductor)
        registry.stop("c1")
        loop._thread.join(timeout=5)

        assert loop.status().last_tick["dispatched"] == 2
        assert loop.status().last_tick["queued"] == 2

    def test_a_finalized_campaign_ends_the_loop_on_its_own(self, registry):
        conductor = _Conductor(
            outcomes=[TickReport(phase="past_hard_stop", in_flight=0, queued=0,
                                 finalized=True)]
        )

        loop = _started(registry, conductor)
        loop._thread.join(timeout=5)

        assert loop.alive is False
        assert "hard stop" in loop.status().stopped_because

    def test_an_unreachable_ledger_does_not_end_the_campaign(self, registry):
        """Hours-long campaigns outlive a store that is briefly rolling."""
        conductor = _Conductor(outcomes=[StoreUnavailable("store is rolling")])

        loop = _started(registry, conductor)
        while conductor.ticks < 2 and loop.alive:
            time.sleep(0.01)
        status = loop.status()
        registry.stop("c1")
        loop._thread.join(timeout=5)

        assert "store is rolling" in status.last_error
        assert conductor.ticks >= 2
        # It recovered, and both halves of that are readable.
        assert status.errors == 1
        assert status.consecutive_errors == 0

    def test_an_unexpected_failure_is_recorded_not_raised(self, registry):
        conductor = _Conductor(outcomes=[RuntimeError("boom")])

        loop = _started(registry, conductor)
        while conductor.ticks < 2 and loop.alive:
            time.sleep(0.01)
        status = loop.status()
        registry.stop("c1")
        loop._thread.join(timeout=5)

        assert "RuntimeError: boom" in status.last_error

    def test_why_a_loop_ended_outlives_the_loop(self, registry):
        """Otherwise "it finished" and "nobody ever started it" look alike."""
        conductor = _Conductor(
            outcomes=[TickReport(phase="past_hard_stop", in_flight=0, queued=0,
                                 finalized=True)]
        )

        loop = _started(registry, conductor)
        loop._thread.join(timeout=5)

        reported = registry.status("c1")
        assert len(reported) == 1
        assert reported[0].running is False
        assert "hard stop" in reported[0].stopped_because

    def test_a_campaign_nobody_conducted_says_so(self, registry):
        assert registry.status("never-heard-of-it") == []

        stopped = registry.stop("never-heard-of-it")
        assert stopped.running is False
        assert "not conducting" in stopped.stopped_because


class TestOneLoopPerCampaign:
    """Two loops would each top the pool up to N and run the campaign at 2N."""

    def test_starting_twice_does_not_start_a_second_loop(self, registry, monkeypatch):
        made = []

        def _conductor(store, campaign_id, **kwargs):
            made.append(campaign_id)
            return _Conductor(campaign_id)

        monkeypatch.setattr("reinvent4_agent.conducting.CampaignConductor", _conductor)
        monkeypatch.setattr(
            "reinvent4_agent.conducting.CampaignStoreClient", lambda *a, **k: object()
        )
        monkeypatch.setattr(
            "reinvent4_agent.conducting.make_pool_scaler", lambda: None
        )

        first = registry.start("c1")
        second = registry.start("c1")

        assert made == ["c1"]
        assert first.campaign_id == second.campaign_id == "c1"
        assert len(registry._loops) == 1
