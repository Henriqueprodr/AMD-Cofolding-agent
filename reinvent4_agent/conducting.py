"""Running the conductor loop inside the agent, instead of beside it.

:mod:`reinvent4_agent.conductor` is a blocking loop with a ``main()``, which
means conducting a campaign has been a thing a person starts in a second
terminal: a ``kubectl exec`` on the cluster, an ``srun`` on a scheduler. The
agent in the chat window could open a campaign and set its concurrency, but
nothing acted on either -- ``set_concurrency`` writes a number into the ledger
and the loop is what makes it true. So a campaign started by talking sat
there doing nothing, which is the one failure mode a person cannot diagnose
from the conversation.

This runs the same loop on a background thread in the agent's own process, so
the conductor is the model the human is talking to. Nothing about the loop
changes: ``CampaignConductor.tick()`` is unchanged, still idempotent, still
the only thing that decides anything. What is added is ownership -- one
thread per campaign, startable and stoppable by name, with the last pass
readable so the agent can say what it did.

A thread rather than a task on the agent's event loop: ``tick()`` is
synchronous and talks to the ledger over blocking HTTP, so on the loop it
would stall the conversation for as long as the store took to answer.

The registry is process-global for the same reason the cluster runs one
conductor pod: two loops on one campaign would each top the pool up to N and
the campaign would quietly run at 2N.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from reinvent4_agent.campaign_client import CampaignStoreClient, StoreUnavailable
from reinvent4_agent.conductor import CampaignConductor, ParetoProposer, TickReport
from reinvent4_agent.pool import PoolScaler, make_pool_scaler

logger = logging.getLogger(__name__)


@dataclass
class ConductingStatus:
    """What a conductor thread has done, as the agent reports it."""

    campaign_id: str
    running: bool
    ticks: int = 0
    started_at: Optional[float] = None
    last_tick_at: Optional[float] = None
    last_tick: Optional[dict[str, Any]] = None
    #: The most recent failure, kept after a later pass succeeds: a campaign
    #: that recovered still had something go wrong, and that is worth saying.
    last_error: Optional[str] = None
    errors: int = 0
    #: Failures since the last good pass. Zero with last_error set means it
    #: recovered; climbing means it is broken now.
    consecutive_errors: int = 0
    stopped_because: Optional[str] = None
    settings: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class _Conducting:
    """One campaign's loop, on its own thread."""

    def __init__(
        self,
        conductor: CampaignConductor,
        settings: dict[str, Any],
    ) -> None:
        self.conductor = conductor
        self.campaign_id = conductor.campaign_id
        self._stop = threading.Event()
        self._state = ConductingStatus(
            campaign_id=self.campaign_id, running=False, settings=settings
        )
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run, name=f"conductor-{self.campaign_id}", daemon=True
        )

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            self._state.running = True
            self._state.started_at = time.time()
        self._thread.start()

    def stop(self, reason: str) -> None:
        """Ask the thread to finish its wait and exit. Does not block."""
        with self._lock:
            self._state.stopped_because = reason
        self._stop.set()

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def status(self) -> ConductingStatus:
        with self._lock:
            return ConductingStatus(**asdict(self._state))

    # -- the loop ----------------------------------------------------------

    def _run(self) -> None:
        logger.info("conducting campaign %s", self.campaign_id)
        while not self._stop.is_set():
            try:
                report = self.conductor.tick()
            except StoreUnavailable as exc:
                # The ledger may be rolling. A campaign is hours long; losing
                # the store for a minute is not losing the campaign.
                self._record(error=f"ledger unavailable: {exc}")
            except Exception as exc:  # noqa: BLE001 - a thread must not die silently
                logger.exception("conducting %s failed", self.campaign_id)
                self._record(error=f"{type(exc).__name__}: {exc}")
            else:
                self._record(report=report)
                if report.finalized:
                    self.stop("the campaign reached its hard stop")
                    break

            self._stop.wait(self.conductor.tick_seconds)

        with self._lock:
            self._state.running = False
            self._state.stopped_because = (
                self._state.stopped_because or "stopped"
            )
        logger.info(
            "stopped conducting %s: %s",
            self.campaign_id, self._state.stopped_because,
        )

    def _record(
        self, report: Optional[TickReport] = None, error: Optional[str] = None
    ) -> None:
        with self._lock:
            self._state.ticks += 1
            self._state.last_tick_at = time.time()
            if error is not None:
                self._state.last_error = error
                self._state.errors += 1
                self._state.consecutive_errors += 1
            else:
                self._state.consecutive_errors = 0
                self._state.last_tick = asdict(report) if report else None


class ConductingRegistry:
    """The campaigns this process is conducting, at most one loop each."""

    def __init__(self) -> None:
        self._loops: dict[str, _Conducting] = {}
        # Why a loop ended outlives the loop: "the campaign reached its hard
        # stop" and "nobody ever conducted this" are the same silence
        # otherwise, and the agent has to tell a person which it was.
        self._finished: dict[str, ConductingStatus] = {}
        self._lock = threading.Lock()

    def start(
        self,
        campaign_id: str,
        *,
        store: Optional[CampaignStoreClient] = None,
        scaler: Optional[PoolScaler] = None,
        task_budget_hours: Optional[float] = None,
        max_steps: Optional[int] = None,
        tick_seconds: Optional[float] = None,
        objectives: Optional[list[str]] = None,
    ) -> ConductingStatus:
        """Begin conducting, or return the status of the loop already doing it.

        Starting twice is not an error and does not start a second loop: the
        agent may well ask again after a restart or a reconnect, and the
        answer it wants either way is "this campaign is being conducted".
        """

        with self._lock:
            self._reap()
            existing = self._loops.get(campaign_id)
            if existing is not None:
                return existing.status()

            settings: dict[str, Any] = {
                "task_budget_hours": task_budget_hours,
                "max_steps": max_steps,
                "tick_seconds": tick_seconds,
            }
            conductor = CampaignConductor(
                store or CampaignStoreClient(),
                campaign_id,
                scaler=scaler or make_pool_scaler(),
                proposer=ParetoProposer(objectives) if objectives else None,
                task_budget_seconds=(
                    task_budget_hours * 3600 if task_budget_hours else None
                ),
                max_steps=max_steps,
                **({"tick_seconds": tick_seconds} if tick_seconds else {}),
            )
            loop = _Conducting(conductor, settings)
            self._loops[campaign_id] = loop

        loop.start()
        return loop.status()

    def stop(self, campaign_id: str, reason: str = "asked to stop") -> ConductingStatus:
        """Stop the loop. The campaign itself is untouched -- it stays open
        and its in-flight tasks keep running; this only stops dispatching."""

        with self._lock:
            loop = self._loops.get(campaign_id)
            finished = self._finished.get(campaign_id)
        if loop is None:
            return finished or ConductingStatus(
                campaign_id=campaign_id,
                running=False,
                stopped_because="this process was not conducting that campaign",
            )
        loop.stop(reason)
        return loop.status()

    def status(self, campaign_id: Optional[str] = None) -> list[ConductingStatus]:
        with self._lock:
            self._reap()
            if campaign_id:
                loop = self._loops.get(campaign_id)
                if loop is not None:
                    return [loop.status()]
                done = self._finished.get(campaign_id)
                return [done] if done else []
            return (
                [loop.status() for loop in self._loops.values()]
                + list(self._finished.values())
            )

    def stop_all(self, reason: str = "shutting down") -> None:
        with self._lock:
            loops = list(self._loops.values())
        for loop in loops:
            loop.stop(reason)

    def _reap(self) -> None:
        """Retire finished loops, so a campaign can be conducted again."""
        for campaign_id, loop in list(self._loops.items()):
            if not loop.alive and loop.status().started_at is not None:
                self._finished[campaign_id] = loop.status()
                del self._loops[campaign_id]


#: One per process, for the same reason the cluster runs one conductor pod.
REGISTRY = ConductingRegistry()
