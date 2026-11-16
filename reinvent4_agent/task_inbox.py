# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Messages for a sub-agent at work, from the person watching it.

One JSONL file per task in the shared log directory: the gateway appends
what was typed, and the supervisor running the task reads what it has not
read yet. A file and not the ledger, because the gateway and every
supervisor already share that directory, and the ledger is the record of
the campaign's work, which a remark in passing is not.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Longer than anyone types into a chat box; past it a paste is not a message.
MAX_CHARS = 8_000


def inbox_path(log_dir: str | Path, task_id: str) -> Path:
    return Path(log_dir) / "inbox" / f"{task_id}.jsonl"


def post(log_dir: str | Path, task_id: str, text: str) -> dict[str, Any]:
    """Leave *text* for the task's sub-agent; returns what was written."""

    record = {"ts": datetime.now(timezone.utc).isoformat(), "text": text[:MAX_CHARS]}
    path = inbox_path(log_dir, task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return record


def messages(log_dir: str | Path, task_id: str) -> list[dict[str, Any]]:
    """Every message left for a task, oldest first."""

    path = inbox_path(log_dir, task_id)
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except ValueError:
                # The gateway may be mid-write on the last line.
                continue
            if isinstance(record, dict) and isinstance(record.get("text"), str):
                out.append(record)
    return out


class Inbox:
    """The messages for one task that its agent has not been given yet.

    Counted rather than marked read: the file is the gateway's to write, and
    the agent's transcript, which records each message as it is handed over,
    is what tells the UI how many are still waiting.
    """

    def __init__(self, log_dir: str | Path, task_id: str) -> None:
        self.log_dir = log_dir
        self.task_id = task_id
        self._taken = 0

    def take(self) -> list[str]:
        try:
            found = messages(self.log_dir, self.task_id)
        except OSError:
            logger.warning("could not read the inbox of task %s", self.task_id, exc_info=True)
            return []
        new = [record["text"] for record in found[self._taken:]]
        self._taken = len(found)
        return new
