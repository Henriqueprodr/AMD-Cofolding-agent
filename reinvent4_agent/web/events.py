# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Agent events -> WebSocket messages.

``AgentEventHandler`` methods are synchronous and are called from inside the
agent's coroutines, so nothing here may ``await``. Publishing is a plain
``put_nowait`` onto unbounded queues, which is safe from any thread via
``call_soon_threadsafe``.

In practice every callback already fires on the event-loop thread (only
``backend.complete`` is pushed to a worker, and it talks to the agent through
queues, not through this handler), but the thread check costs three lines and
removes a whole class of future bug.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from reinvent4_agent.web import protocol

logger = logging.getLogger(__name__)

#: Tool results larger than this are stored out of band and referenced, so
#: one big CSV cannot stall every connected tab.
INLINE_LIMIT = 16 * 1024
#: Rows rendered in a score table; the rest stay in the full payload.
MAX_TABLE_ROWS = 200


class TurnCancelled(Exception):
    """Raised inside the agent loop when the user presses Stop.

    Raised from ``on_tool_call`` so it lands between tool calls, where
    ``Agent._handle_tool_calls``'s ``finally`` still synthesises results for
    every unanswered ``tool_use`` -- leaving the conversation valid.
    """


@dataclass
class TurnContext:
    """Mutable per-turn state shared between the session and its handlers."""

    turn_id: str = ""
    cancel: bool = False
    call_seq: int = field(default=0)

    def next_call_id(self) -> int:
        self.call_seq += 1
        return self.call_seq


class WebEventHandler:
    """Implements ``AgentEventHandler`` by publishing to a ``SessionHub``."""

    def __init__(
        self,
        hub,
        loop,
        *,
        agent_label: str = "driver",
        context: TurnContext,
        store_payload: Callable[[str, str], str | None] | None = None,
        note_payload: Callable[[str], None] | None = None,
    ) -> None:
        self._hub = hub
        self._loop = loop
        self._label = agent_label
        self._ctx = context
        self._store = store_payload
        self._note = note_payload

    # -- plumbing ------------------------------------------------------

    def _emit(self, message: dict[str, Any]) -> None:
        if (
            threading.current_thread() is threading.main_thread()
            or not self._loop.is_running()
        ):
            self._hub.publish(message)
        else:  # pragma: no cover - defensive
            self._loop.call_soon_threadsafe(self._hub.publish, message)

    @property
    def _turn(self) -> str:
        return self._ctx.turn_id

    # -- AgentEventHandler --------------------------------------------

    def on_thinking(self, text: str) -> None:
        self._emit(protocol.thinking(self._turn, self._label, text))

    def on_tool_call(self, name: str, arguments: dict[str, Any]) -> None:
        if self._ctx.cancel:
            raise TurnCancelled(f"stopped before {name}")
        self._emit(
            protocol.tool_call(
                self._turn,
                self._label,
                self._ctx.next_call_id(),
                name,
                arguments,
            )
        )

    def on_tool_result(self, name: str, result: str, is_error: bool) -> None:
        call_id = self._ctx.call_seq
        size = len(result or "")
        # Every result, inline or not, may mention a structure path the
        # viewer is later allowed to open.
        if self._note is not None and not is_error:
            self._note(result or "")

        inline: str | None = result if size <= INLINE_LIMIT else None
        payload_ref = None
        if inline is None and self._store is not None:
            payload_ref = self._store(f"{self._turn}:{call_id}", result)

        self._emit(
            protocol.tool_result(
                self._turn,
                self._label,
                call_id,
                name,
                is_error=is_error,
                summary=_summarize(name, result, is_error),
                inline=inline,
                payload_ref=payload_ref,
                bytes_=size,
            )
        )

        table = _score_table_payload(name, result)
        if table is not None:
            self._emit(
                protocol.score_table(
                    self._turn, self._label, call_id, **table
                )
            )

    def on_builtin_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        result: str,
        is_error: bool,
    ) -> None:
        self._emit(
            protocol.builtin_activity(
                self._turn,
                self._label,
                name=name,
                arguments=arguments,
                result=(result or "")[:2000],
                is_error=is_error,
            )
        )

    def on_error(self, message: str) -> None:
        self._emit(protocol.error(message, self._turn))

    def on_monitor_alert(self, alert: dict[str, Any]) -> None:
        self._emit(protocol.monitor_alert(alert))

    def on_monitor_response(self, text: str) -> None:
        self._emit(protocol.monitor_response(self._label, text))


class WebCoordinatorEvents:
    """``CoordinatorEvents`` sink for the Driver/Reviewer campaign monitor."""

    def __init__(self, hub, loop) -> None:
        self._hub, self._loop = hub, loop

    def _emit(self, message: dict[str, Any]) -> None:
        if (
            threading.current_thread() is threading.main_thread()
            or not self._loop.is_running()
        ):
            self._hub.publish(message)
        else:  # pragma: no cover - defensive
            self._loop.call_soon_threadsafe(self._hub.publish, message)

    def on_turn_start(self, agent_name: str, turn: int) -> None:
        self._emit(protocol.coordinator_turn(agent_name, turn))

    def on_agent_message(self, agent_name: str, text: str) -> None:
        self._emit(protocol.assistant_message("", agent_name, text))

    def on_budget(
        self, turn: int, max_turns: int, total_tokens: int, budget: int | None
    ) -> None:
        self._emit(
            protocol.status(
                f"Review budget: turn {turn}/{max_turns}, "
                f"{total_tokens} tokens"
                + (f" of {budget}" if budget else "")
            )
        )

    def on_monitor_cycle_start(self, cycle: int, job_id: str | None) -> None:
        self._emit(
            protocol.status(
                f"Reviewer monitor cycle {cycle}"
                + (f" on job {job_id}" if job_id else "")
            )
        )

    def on_monitor_end(self, job_id: str | None, state: str | None) -> None:
        self._emit(
            protocol.status(
                f"Campaign monitor finished"
                + (f" (job {job_id}: {state})" if job_id else "")
            )
        )


# ----------------------------------------------------------------------
# Result shaping
# ----------------------------------------------------------------------


def _summarize(name: str, result: str, is_error: bool) -> str:
    """One line for the collapsed view, mirroring the terminal's brevity."""
    if is_error:
        return (result or "failed")[:200]
    try:
        data = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        text = (result or "")[:200]
        return text + ("..." if len(result or "") > 200 else "")

    if isinstance(data, dict):
        if "error" in data:
            return f"error: {str(data['error'])[:180]}"
        molecules = data.get("molecules")
        if isinstance(molecules, list):
            summary = data.get("summary") or {}
            mean = summary.get("mean_total_score")
            return (
                f"{len(molecules)} molecule(s) scored"
                + (f", mean={mean:.3f}" if isinstance(mean, (int, float)) else "")
            )
        activities = data.get("activities")
        if isinstance(activities, list):
            total = data.get("total_available", len(activities))
            return f"{len(activities)} measured activities (of {total})"
        if "smiles" in data:
            return f"SMILES: {str(data['smiles'])[:120]}"
        if data.get("format") in ("pdb", "cif", "sdf"):
            meta = data.get("meta") or {}
            return (
                f"{data['format']} structure"
                + (f", {meta['n_heavy_atoms']} heavy atoms"
                   if meta.get("n_heavy_atoms") else "")
            )
        if "stdout" in data:
            head = (data.get("stdout") or "").strip().splitlines()
            return head[0][:180] if head else f"exit {data.get('exit_code')}"
        keys = ", ".join(list(data)[:6])
        return f"{len(data)} field(s): {keys}"
    if isinstance(data, list):
        return f"{len(data)} item(s)"
    return str(data)[:200]


def _score_table_payload(name: str, result: str) -> dict[str, Any] | None:
    """Parse a ``score_molecules`` result into table-shaped data.

    Mirrors the terminal's ``_maybe_render_score_table`` gate so the two
    front ends show a table in exactly the same cases.
    """
    if name != "score_molecules":
        return None
    try:
        data = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None

    molecules = data.get("molecules")
    if not isinstance(molecules, list) or not molecules:
        return None

    components: list[str] = []
    for mol in molecules:
        for key in (mol.get("components") or {}):
            if key not in components:
                components.append(key)

    rows = []
    for idx, mol in enumerate(molecules[:MAX_TABLE_ROWS]):
        rows.append(
            {
                "idx": idx,
                "smiles": mol.get("smiles"),
                "total_score": mol.get("total_score"),
                "components": mol.get("components") or {},
            }
        )

    structures = data.get("structures")
    return {
        "molecules": rows,
        "summary": data.get("summary") or {},
        "components": components,
        # NOTE: this list is deduplicated upstream in scoring.py, so it is
        # NOT index-aligned with the rows -- it cannot be joined to them.
        "structures": structures if isinstance(structures, list) else None,
        "truncated_at": (
            MAX_TABLE_ROWS if len(molecules) > MAX_TABLE_ROWS else None
        ),
    }
