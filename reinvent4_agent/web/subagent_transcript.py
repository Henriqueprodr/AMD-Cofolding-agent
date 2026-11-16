# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A sub-agent's conversation, read back from the transcript it wrote.

A sub-agent runs in a supervisor job of its own, so nothing of its
conversation reaches the gateway except through the ledger -- the brief, the
claim, the report. The agent does keep a JSONL transcript in the shared log
directory, though, named after the task it was given, and that is the only
place the steps in between exist. This reads it for a person looking.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from reinvent4_agent.task_inbox import messages

#: A task id as the ledger issues them; anything else could walk the path.
_TASK_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

#: The agents a supervisor names after their task. The Reviewer is left out:
#: it second-guesses the Driver's work, and its side is not the sub-agent's.
_ROLES = ("subagent", "driver")

#: What a record may carry into the browser.
_KEPT = ("ts", "event", "content", "name", "arguments", "elapsed_s", "is_error", "error", "excerpt", "content_length", "round", "interjected")

#: Per text field. A tool call's arguments can embed a whole TTP and a
#: thought can run for pages; this is for reading, not for replay.
_MAX_TEXT = 20_000

#: A runaway turn writes thousands of rounds; the newest are what matter.
_MAX_EVENTS = 2_000


def valid_task_id(task_id: str) -> bool:
    return bool(_TASK_ID.match(task_id))


def transcript_path(log_dir: Path, task_id: str) -> Path | None:
    """The newest transcript of a task, or None if it never wrote one.

    Newest because a task whose lease lapsed is claimed again, and the run
    that is going on is the one worth reading.
    """

    found = [path for role in _ROLES for path in log_dir.glob(f"session_*_{role}-{task_id}.jsonl")]
    return max(found, key=lambda path: path.stat().st_mtime) if found else None


def read_conversation(log_dir: Path, task_id: str) -> dict[str, Any]:
    """The events of a task's conversation, oldest first.

    *queued* are the messages left for it that the agent has not been given
    yet: the inbox past as many as the transcript shows handed over.
    """

    left = messages(log_dir, task_id)
    path = transcript_path(log_dir, task_id)
    if path is None:
        return {"task_id": task_id, "found": False, "events": [], "truncated": False, "queued": left}

    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except ValueError:
                # The supervisor may be mid-write on the last line.
                continue
            if isinstance(record, dict):
                events.append(_kept(record))

    truncated = len(events) > _MAX_EVENTS
    handed = sum(1 for event in events if event.get("event") == "user_message" and event.get("interjected"))
    return {
        "task_id": task_id,
        "found": True,
        "events": events[-_MAX_EVENTS:],
        "truncated": truncated,
        "queued": left[handed:],
    }


def _kept(record: dict[str, Any]) -> dict[str, Any]:
    out = {key: record[key] for key in _KEPT if key in record}
    if isinstance(out.get("ts"), str):
        out["ts"] = _with_zone(out["ts"])
    for key in ("content", "error"):
        if isinstance(out.get(key), str) and len(out[key]) > _MAX_TEXT:
            out[key] = out[key][:_MAX_TEXT] + "\n… cut for display."
    if "arguments" in out:
        text = json.dumps(out["arguments"], default=str)
        if len(text) > _MAX_TEXT:
            out["arguments"] = {"_cut": text[:_MAX_TEXT] + "… cut for display."}
    return out


def _with_zone(ts: str) -> str:
    """The agent writes local time with no zone; a browser elsewhere would read it hours off.

    The supervisors and this gateway share a site and so a clock zone, which
    is the one assumption this makes.
    """

    try:
        moment = datetime.fromisoformat(ts)
    except ValueError:
        return ts
    return (moment if moment.tzinfo else moment.astimezone()).isoformat()
