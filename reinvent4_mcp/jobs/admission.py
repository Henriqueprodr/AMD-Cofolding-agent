"""How much work to admit at once: AIMD windows instead of fixed thresholds.

A fixed limit is the wrong number most of the time. ``max_concurrent_jobs``
on a scheduler: each run is its own Slurm job, and how many of those the
partition will start depends on everybody else's work -- sixteen submitted
into a partition that starts two leaves fourteen sub-agents holding a task, a
supervisor and a deadline for a GPU that is not coming, and the account's
fair-share drops for every one of them. A fixed Boltz queue threshold: under
it every free slot is filled at once, over it nothing is, so the fleet
oscillates around the number rather than settling under it.

So admission follows what the bottleneck is actually doing, the way MARS
(arXiv 2604.26963) sizes an agent fleet to its serving engine: additive
increase while it keeps up, multiplicative decrease when it does not, with
hysteresis so one bad reading is not a collapse. A window starts at its
ceiling, so a backend that keeps up behaves exactly as the fixed limit did,
and never closes below a floor that keeps a campaign moving at all.

Not thread-safe; callers hold their own lock.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Optional

#: How long a run may wait for Slurm before the queue counts as congested.
#: Long enough to cover the minute or two a healthy partition takes to place
#: a small job; short enough that a stalled one is noticed within a task.
TARGET_WAIT_SECONDS = 300.0

#: A window never closes below this: a campaign that can start nothing
#: learns nothing, and two waiting is no harm to anyone.
FLOOR = 2


class AimdWindow:
    """An additive-increase, multiplicative-decrease admission window."""

    def __init__(
        self,
        ceiling: int,
        *,
        floor: int = FLOOR,
        cooldown: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ceiling = max(1, int(ceiling))
        self.floor = max(1, min(int(floor), self.ceiling))
        self.cooldown = float(cooldown)
        self._clock = clock
        self.window = float(self.ceiling)
        self._last_cut: Optional[float] = None
        #: Why the window last closed, for whoever is refused because of it.
        self.reason: Optional[str] = None

    @property
    def limit(self) -> int:
        """How much may be admitted now."""

        return max(self.floor, min(self.ceiling, math.floor(self.window)))

    def set_ceiling(self, ceiling: int) -> None:
        """Follow a ceiling somebody changed.

        An open window moves with it, so raising the ceiling on a backend
        that keeps up takes effect at once instead of one step at a time.
        """

        ceiling = max(1, int(ceiling))
        if self.window >= self.ceiling:
            self.window = float(ceiling)
        self.ceiling = ceiling
        self.floor = min(self.floor, ceiling)
        self.window = min(self.window, float(ceiling))

    def grow(self, step: Optional[float] = None) -> None:
        """The backend kept up: widen by *step* -- the additive half.

        By default 1/window, for a signal that arrives once per admitted
        item, so a full window of them adds one. A signal read once per
        control period passes 1 instead, which is the same one per round.
        """

        step = 1.0 / max(self.window, 1.0) if step is None else float(step)
        self.window = min(float(self.ceiling), self.window + step)

    def cut(self, reason: str) -> None:
        """The backend fell behind: halve -- the multiplicative half -- but
        not again within a cooldown. Without that every check during one
        congestion would halve it again, and a few minutes of a busy backend
        would close it to the floor."""

        now = self._clock()
        if self._last_cut is not None and now - self._last_cut < self.cooldown:
            return
        before = self.limit
        self.window = max(float(self.floor), self.window / 2.0)
        self._last_cut = now
        if self.limit < before:
            self.reason = f"{reason}, so the window went from {before} to {self.limit}"


class RunWindow(AimdWindow):
    """The run window, driven by how long runs wait for the scheduler.

    The queue wait -- from a run being handed to Slurm to Slurm starting it --
    is the one thing on a shared cluster that says whether another run would
    start or only wait. Runs that start without queueing (the local launcher)
    give no signal of contention, so on a workstation it stays at its ceiling.
    """

    def __init__(
        self,
        ceiling: int,
        *,
        floor: int = FLOOR,
        target_wait: float = TARGET_WAIT_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(ceiling, floor=floor, cooldown=target_wait, clock=clock)
        self.target_wait = float(target_wait)

    def started(self, waited: float) -> None:
        """A run left the queue after *waited* seconds. Only a prompt start
        grows the window; a slow one is counted by :meth:`observe`."""

        if waited <= self.target_wait:
            self.grow()

    def observe(self, oldest_wait: Optional[float]) -> None:
        """Look at the queue: *oldest_wait* is the longest any run has waited."""

        if oldest_wait is not None and oldest_wait > self.target_wait:
            self.cut(f"a run has waited {oldest_wait / 60:.0f} min for the scheduler to start it")

    def view(self, active: int, waiting: int, oldest_wait: Optional[float]) -> dict[str, Any]:
        """The window as callers report it."""

        return {
            "limit": self.limit,
            "ceiling": self.ceiling,
            "active": active,
            "waiting": waiting,
            "free": max(0, self.limit - active),
            "oldest_wait_seconds": None if oldest_wait is None else round(oldest_wait),
            "reason": self.reason if self.limit < self.ceiling else None,
        }
