# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The AIMD admission windows: how much work is let in while a backend lags.

The rules are the whole design, so they are pinned here without a scheduler
or a scorer: start open, halve on congestion at most once a cooldown, never
close below the floor, and earn back one a round.
"""

from __future__ import annotations

from reinvent4_mcp.jobs.admission import AimdWindow, RunWindow


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _window(ceiling=16, clock=None) -> RunWindow:
    return RunWindow(ceiling, floor=2, target_wait=300.0, clock=clock or Clock())


class TestRunWindow:

    def test_it_starts_at_the_ceiling(self):
        """A cluster that starts runs promptly sees the fixed limit it had."""
        assert _window().limit == 16

    def test_a_run_waiting_past_the_target_halves_it(self):
        window = _window()

        window.observe(301.0)

        assert window.limit == 8
        assert "waited 5 min" in window.reason

    def test_a_short_wait_is_not_congestion(self):
        window = _window()

        window.observe(299.0)
        window.observe(None)

        assert window.limit == 16

    def test_one_congestion_halves_it_once(self):
        """Every admission check looks at the queue; one slow run is one cut."""
        clock = Clock()
        window = _window(clock=clock)

        window.observe(400.0)
        clock.now = 120.0
        window.observe(520.0)

        assert window.limit == 8

    def test_congestion_that_lasts_halves_it_again(self):
        clock = Clock()
        window = _window(clock=clock)

        window.observe(400.0)
        clock.now = 301.0
        window.observe(700.0)

        assert window.limit == 4

    def test_it_never_closes_below_the_floor(self):
        clock = Clock()
        window = _window(clock=clock)

        for step in range(10):
            clock.now = step * 301.0
            window.observe(10_000.0)

        assert window.limit == 2

    def test_prompt_starts_earn_back_about_one_run_a_window(self):
        window = _window()
        window.observe(400.0)

        for _ in range(7):
            window.started(30.0)
        assert window.limit == 8
        for _ in range(2):
            window.started(30.0)

        assert window.limit == 9

    def test_a_slow_start_earns_nothing(self):
        window = _window()
        window.observe(400.0)

        for _ in range(20):
            window.started(600.0)

        assert window.limit == 8

    def test_it_never_opens_past_the_ceiling(self):
        window = _window()

        for _ in range(100):
            window.started(1.0)

        assert window.limit == 16

    def test_the_view_says_why_it_is_closed(self):
        window = _window()
        window.observe(400.0)

        view = window.view(active=8, waiting=3, oldest_wait=400.0)

        assert (view["limit"], view["free"], view["waiting"]) == (8, 0, 3)
        assert "from 16 to 8" in view["reason"]

    def test_an_open_window_gives_no_reason(self):
        assert _window().view(active=1, waiting=0, oldest_wait=None)["reason"] is None


class TestAimdWindowCeiling:

    def test_an_open_window_follows_a_raised_ceiling_at_once(self):
        window = AimdWindow(4, floor=1, cooldown=60.0, clock=Clock())

        window.set_ceiling(8)

        assert window.limit == 8

    def test_a_closed_window_does_not_jump_open_with_it(self):
        window = AimdWindow(8, floor=1, cooldown=60.0, clock=Clock())
        window.cut("busy")

        window.set_ceiling(16)

        assert window.limit == 4

    def test_a_lowered_ceiling_closes_it(self):
        window = AimdWindow(8, floor=1, cooldown=60.0, clock=Clock())

        window.set_ceiling(3)

        assert window.limit == 3
