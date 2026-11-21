# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The WebSocket wire protocol, in one place.

Every server -> client message is a JSON object with ``type`` and ``ts``.
Turn-scoped messages also carry ``turn_id`` and ``agent`` so the browser can
attribute Driver and Reviewer output the way the terminal colours them.

One constructor per message type, so the set of types is greppable and the
frontend has exactly one table to switch on.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

# --- client -> server -------------------------------------------------
C_USER_MESSAGE = "user_message"
C_STOP = "stop"
C_PING = "ping"

# --- server -> client -------------------------------------------------
S_SESSION = "session"
S_REPLAY_START = "replay_start"
S_REPLAY_END = "replay_end"
S_STATUS = "status"
S_BUSY = "busy"
S_USER_ECHO = "user_echo"
S_TURN_START = "turn_start"
S_THINKING = "thinking"
S_TOOL_CALL = "tool_call"
S_TOOL_RESULT = "tool_result"
S_SCORE_TABLE = "score_table"
S_BUILTIN_ACTIVITY = "builtin_activity"
S_ASSISTANT_MESSAGE = "assistant_message"
S_TURN_END = "turn_end"
S_MONITOR_ALERT = "monitor_alert"
S_MONITOR_RESPONSE = "monitor_response"
S_COORDINATOR_TURN = "coordinator_turn"
S_SESSION_RESET = "session_reset"
S_ERROR = "error"
S_PONG = "pong"

#: Reserved. No backend streams tokens today -- what this UI streams is
#: events, not text deltas. Declared so a future streaming backend can be
#: added without a breaking wire change.
S_ASSISTANT_DELTA = "assistant_delta"


def _msg(type_: str, **fields: Any) -> dict[str, Any]:
    return {
        "type": type_,
        "ts": datetime.now(timezone.utc).isoformat(),
        **fields,
    }


# ----------------------------------------------------------------------
# Session lifecycle
# ----------------------------------------------------------------------


def session(
    *,
    session_id: str,
    backend: str,
    model: str,
    multi_agent: bool,
    web_tools: bool,
    tools: list[str],
    rdkit_available: bool,
) -> dict[str, Any]:
    """First message on every connection: what this session can do."""
    return _msg(
        S_SESSION,
        session_id=session_id,
        backend=backend,
        model=model,
        multi_agent=multi_agent,
        web_tools=web_tools,
        tools=tools,
        rdkit_available=rdkit_available,
    )


def replay_start(count: int) -> dict[str, Any]:
    return _msg(S_REPLAY_START, count=count)


def replay_end() -> dict[str, Any]:
    return _msg(S_REPLAY_END)


def session_reset() -> dict[str, Any]:
    return _msg(S_SESSION_RESET)


def status(text: str, level: str = "info") -> dict[str, Any]:
    return _msg(S_STATUS, level=level, text=text)


def busy(is_busy: bool, reason: str = "chat") -> dict[str, Any]:
    return _msg(S_BUSY, busy=is_busy, reason=reason)


def error(text: str, turn_id: str | None = None) -> dict[str, Any]:
    return _msg(S_ERROR, text=text, turn_id=turn_id)


def pong() -> dict[str, Any]:
    return _msg(S_PONG)


# ----------------------------------------------------------------------
# Turn events
# ----------------------------------------------------------------------


def user_echo(turn_id: str, text: str) -> dict[str, Any]:
    return _msg(S_USER_ECHO, turn_id=turn_id, text=text)


def turn_start(turn_id: str, agent: str, origin: str) -> dict[str, Any]:
    return _msg(S_TURN_START, turn_id=turn_id, agent=agent, origin=origin)


def turn_end(
    turn_id: str,
    agent: str,
    *,
    ok: bool,
    elapsed_ms: int,
    tokens: dict[str, int],
    model: str | None = None,
) -> dict[str, Any]:
    """End of a turn. ``model`` is what actually answered it.

    Carried here rather than only in the opening ``session`` message
    because with the claude-code backend the CLI picks the model, so it is
    not knowable until a reply has arrived.
    """
    return _msg(
        S_TURN_END,
        turn_id=turn_id,
        agent=agent,
        ok=ok,
        elapsed_ms=elapsed_ms,
        tokens=tokens,
        model=model,
    )


def thinking(turn_id: str, agent: str, text: str) -> dict[str, Any]:
    return _msg(S_THINKING, turn_id=turn_id, agent=agent, text=text)


def tool_call(
    turn_id: str, agent: str, call_id: int, name: str, arguments: dict
) -> dict[str, Any]:
    return _msg(
        S_TOOL_CALL,
        turn_id=turn_id,
        agent=agent,
        call_id=call_id,
        name=name,
        arguments=arguments,
    )


def tool_result(
    turn_id: str,
    agent: str,
    call_id: int,
    name: str,
    *,
    is_error: bool,
    summary: str,
    inline: str | None,
    payload_ref: str | None,
    bytes_: int,
) -> dict[str, Any]:
    return _msg(
        S_TOOL_RESULT,
        turn_id=turn_id,
        agent=agent,
        call_id=call_id,
        name=name,
        is_error=is_error,
        summary=summary,
        inline=inline,
        payload_ref=payload_ref,
        bytes=bytes_,
    )


def score_table(
    turn_id: str,
    agent: str,
    call_id: int,
    *,
    molecules: list[dict],
    summary: dict,
    components: list[str],
    structures: list[str] | None,
    truncated_at: int | None,
) -> dict[str, Any]:
    return _msg(
        S_SCORE_TABLE,
        turn_id=turn_id,
        agent=agent,
        call_id=call_id,
        molecules=molecules,
        summary=summary,
        components=components,
        structures=structures,
        truncated_at=truncated_at,
    )


def builtin_activity(
    turn_id: str,
    agent: str,
    *,
    name: str,
    arguments: dict,
    result: str,
    is_error: bool,
) -> dict[str, Any]:
    """A tool the LLM harness ran itself (Claude's WebSearch / WebFetch).

    Deliberately its own message type rather than a ``tool_call``: the point
    of showing it is that the user can tell a web-sourced claim from a
    chemistry-tool result.
    """
    return _msg(
        S_BUILTIN_ACTIVITY,
        turn_id=turn_id,
        agent=agent,
        name=name,
        arguments=arguments,
        result=result,
        is_error=is_error,
    )


def assistant_message(turn_id: str, agent: str, text: str) -> dict[str, Any]:
    return _msg(S_ASSISTANT_MESSAGE, turn_id=turn_id, agent=agent, text=text)


# ----------------------------------------------------------------------
# Background activity
# ----------------------------------------------------------------------


def monitor_alert(alert: dict) -> dict[str, Any]:
    return _msg(S_MONITOR_ALERT, alert=alert)


def monitor_response(agent: str, text: str) -> dict[str, Any]:
    return _msg(S_MONITOR_RESPONSE, agent=agent, text=text)


def coordinator_turn(agent: str, turn: int) -> dict[str, Any]:
    return _msg(S_COORDINATOR_TURN, agent=agent, turn=turn)
