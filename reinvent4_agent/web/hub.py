# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fan-out of session events to every connected browser, plus replay.

One process serves one agent session; any number of tabs may watch it. New
and reconnecting tabs replay a bounded history so they do not open on an
empty transcript.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)

#: Messages kept for replay. A long session trims the oldest; the JSONL
#: transcript on disk remains the complete record.
REPLAY_LIMIT = 2000


class SessionHub:
    """Publish/subscribe for one agent session."""

    def __init__(self, replay_limit: int = REPLAY_LIMIT) -> None:
        self._subscribers: set[asyncio.Queue] = set()
        self._replay: deque[dict[str, Any]] = deque(maxlen=replay_limit)

    # -- subscription --------------------------------------------------

    def subscribe(self) -> asyncio.Queue:
        """Register a connection. Queues are UNBOUNDED on purpose.

        Events are published from synchronous agent callbacks, where a
        ``QueueFull`` would have nowhere sensible to go. Volume is bounded
        where it is actually generated instead: large tool results are
        stored out of band and referenced, never inlined.
        """
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    # -- publishing ----------------------------------------------------

    def publish(self, message: dict[str, Any]) -> None:
        """Record and fan out. Safe to call from the event loop thread."""
        self._replay.append(message)
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(message)
            except Exception:  # pragma: no cover - unbounded queue
                logger.debug("dropped a message for a dead subscriber")

    def replay(self) -> list[dict[str, Any]]:
        return list(self._replay)

    def clear_replay(self) -> None:
        """Drop the history, e.g. after /reset."""
        self._replay.clear()
