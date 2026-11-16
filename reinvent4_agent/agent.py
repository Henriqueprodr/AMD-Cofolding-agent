# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Agent loop: user input -> LLM backend -> MCP tool calls -> display.

Implements the agentic loop where the LLM reasons about the user request,
decides which drug design tools to call, processes results, and continues
until a final text response is ready.  Backend-agnostic: works with both
Anthropic and OpenAI-compatible APIs via the LLMBackend abstraction.

Every interaction is logged at DEBUG level, and optionally written to a
structured JSONL transcript file for post-hoc analysis.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from reinvent4_agent.backends import ChatResult, LLMBackend, ToolResult
from reinvent4_agent.mcp_client import MCPClient

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    """A positive integer from the environment, or ``default``."""
    try:
        value = int(os.environ.get(name, ""))
    except ValueError:
        return default
    return value if value > 0 else default


#: Tool calls one turn may make before the agent gives up on it.  High
#: because a supervisor works a task for hours and a conductor runs a
#: campaign across a whole budget; what bounds a turn in practice is the
#: autonomy budget and the context, not this.
MAX_TOOL_ROUNDS = _env_int("REINVENT4_AGENT_MAX_TOOL_ROUNDS", 2000)

#: Characters of one tool result the model is given.  Results are the
#: largest thing in a conversation and the easiest to overrun a context
#: with -- a campaign's files, every result of a run -- and one that does
#: not fit takes the whole turn down with it.  The head and the tail are
#: kept, since a table's shape is at the top and its summary at the end.
MAX_TOOL_RESULT_CHARS = _env_int("REINVENT4_AGENT_MAX_TOOL_RESULT_CHARS", 60_000)

#: Characters of each tool result kept in the transcript, so whoever reads
#: it later can see what came back -- above all, why a call failed.
TOOL_EXCERPT_CHARS = 400

#: Identical calls in a row, answered with the same error, after which the
#: model is told that sending it again will not help, and after which the
#: turn is ended. A model that resends a failing call unchanged rarely
#: changes it later, and meanwhile it holds a task's lease and a share of
#: the LLM every other agent is waiting on.
REPEAT_WARN = _env_int("REINVENT4_AGENT_REPEAT_WARN", 3)
REPEAT_LIMIT = _env_int("REINVENT4_AGENT_REPEAT_LIMIT", 6)


class RepeatedToolCall(RuntimeError):
    """The model kept sending the same failing call."""


_ERROR_KEYS = frozenset({"error", "hint", "detail", "details"})


def _reported_error(text: str) -> str | None:
    """The error a tool put in its answer instead of failing the call.

    Most tools here catch their own exceptions and answer ``{"error": ...}``,
    so the call succeeds and only its content says it did not work. An
    answer that carries an error next to other fields -- a failed run's
    status -- is information about something else, not a failed call.
    """
    if not text.lstrip().startswith("{"):
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    if isinstance(data, dict) and data.get("error") and set(data) <= _ERROR_KEYS:
        return str(data["error"])
    return None


@runtime_checkable
class AgentEventHandler(Protocol):
    """Protocol for receiving agent lifecycle events.

    Implement to customise how the agent displays thinking, tool calls,
    tool results, and errors (e.g. Rich terminal, WebSocket, silent).
    """

    def on_thinking(self, text: str) -> None: ...
    def on_tool_call(self, name: str, arguments: dict[str, Any]) -> None: ...
    def on_tool_result(self, name: str, result: str, is_error: bool) -> None: ...
    #: How far a tool has got, for the ones that say. Optional: a handler
    #: without it is simply not told, and shows work of unknown length.
    def on_tool_progress(
        self, name: str, done: float, total: float | None, message: str | None
    ) -> None: ...
    def on_error(self, message: str) -> None: ...
    def on_monitor_alert(self, alert: dict[str, Any]) -> None: ...
    def on_monitor_response(self, text: str) -> None: ...


@functools.lru_cache(maxsize=None)
def _call_tool_takes_progress(call_tool: Any) -> bool:
    try:
        return "on_progress" in inspect.signature(call_tool).parameters
    except (TypeError, ValueError):
        return False


def _reports_progress(mcp: Any) -> bool:
    """Whether this MCP client can pass a tool's progress back to us.

    The agent is handed whatever speaks call_tool — the real client, a
    remote stand-in, a test double — and only the real one forwards MCP
    progress notifications. Asking first keeps the rest working unchanged.
    """
    return _call_tool_takes_progress(getattr(type(mcp), "call_tool", None))


class _NullEventHandler:
    """No-op event handler used when no handler is provided."""

    def on_thinking(self, text: str) -> None: pass
    def on_tool_call(self, name: str, arguments: dict[str, Any]) -> None: pass
    def on_tool_result(self, name: str, result: str, is_error: bool) -> None: pass
    def on_error(self, message: str) -> None: pass
    def on_monitor_alert(self, alert: dict[str, Any]) -> None: pass
    def on_monitor_response(self, text: str) -> None: pass


class Agent:
    """Conversational agent that connects an LLM to drug design tools via MCP.

    Maintains conversation history (inside the backend) across turns and
    handles the tool-use loop: the LLM proposes tool calls, the agent
    executes them against the MCP server, feeds results back, and repeats
    until the LLM produces a final text response.
    """

    def __init__(
        self,
        mcp_client: MCPClient,
        backend: LLMBackend,
        *,
        max_tokens: int = 4096,
        log_dir: str | None = None,
        event_handler: AgentEventHandler | None = None,
        name: str = "agent",
        system_prompt: str | None = None,
        allowed_tools: set[str] | None = None,
    ) -> None:
        self.mcp = mcp_client
        self.backend = backend
        self.max_tokens = max_tokens
        self.name = name
        self._system_prompt = system_prompt
        self._allowed_tools = allowed_tools
        #: Tools answered here rather than over MCP, as ``name -> (definition,
        #: handler)``. For things that belong to one agent and would be
        #: meaningless shared: the MCP server is one process behind one stdio
        #: pipe for every conversation at once, so a tool that needs to know
        #: *which* conversation called it cannot live there. Installed by
        #: whoever owns the agent, and therefore not subject to
        #: ``allowed_tools``, which filters the catalog the agent was handed.
        self.local_tools: dict[str, tuple[dict[str, Any], Any]] = {}
        #: Tools refused for now, as ``name -> what to do instead``. Unlike
        #: ``allowed_tools``, which is settled when the agent is built, this
        #: turns with the conversation: what the agent should do itself
        #: depends on whether it has sub-agents to do it instead.
        self.withheld: dict[str, str] = {}
        #: Messages that arrive while the agent works, as a callable returning
        #: what has come in since it was last called. Read after each round of
        #: tool results, the one point in a turn where the conversation can
        #: take a user message without breaking a tool call's pairing.
        self.inbox: Callable[[], list[str]] | None = None
        self._transcript_path = _init_transcript(log_dir, name)
        self._turn = 0
        self._events: AgentEventHandler = event_handler or _NullEventHandler()
        self._chat_lock = asyncio.Lock()
        self._monitor_task: asyncio.Task[None] | None = None
        self._monitor_running = False
        self._tool_in_flight = False  # True while a blocking tool call is executing
        self._seen_alerts: set[tuple[str, str]] = set()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        #: Input tokens of the last completion: the size of the conversation
        #: as the model last saw it.
        self.context_tokens = 0
        self._last_call: tuple[str, str, str] | None = None
        self._repeats = 0
        self._stuck: str | None = None

    @property
    def total_tokens(self) -> int:
        """Cumulative input + output tokens consumed by this agent."""
        return self.total_input_tokens + self.total_output_tokens

    async def _scoped_tools(self) -> list[dict[str, Any]]:
        """Return the MCP tools this agent is permitted to use.

        When ``allowed_tools`` is set (e.g. the read-only Reviewer), the
        shared client's tool list is filtered down to that subset so the
        agent physically cannot call mutating tools.
        """
        tools = await self.mcp.list_tools()
        if self._allowed_tools is not None:
            tools = [t for t in tools if t["name"] in self._allowed_tools]
        tools = [t for t in tools if t["name"] not in self.withheld]
        # Local tools last and by name, so installing one over an MCP tool
        # replaces it in the catalog rather than offering the model two tools
        # with the same name and letting the backend pick.
        local = {name: definition for name, (definition, _) in self.local_tools.items()}
        return [t for t in tools if t["name"] not in local] + list(local.values())

    def _tool_allowed(self, name: str) -> bool:
        """Whether this agent may execute the canonical tool ``name``."""
        if name in self.withheld:
            return False
        return self._allowed_tools is None or name in self._allowed_tools

    async def chat(self, user_message: str) -> str:
        """Process a user message and return the agent's text response.

        This may involve multiple rounds of tool calls before the LLM
        produces its final answer.
        """
        self._turn += 1
        logger.info("Turn %d — user: %s", self._turn, user_message[:200])
        self._write_event("user_message", {"content": user_message})

        self.backend.add_user_message(user_message)
        tools = await self._scoped_tools()
        self._last_call, self._repeats, self._stuck = None, 0, None

        for round_idx in range(MAX_TOOL_ROUNDS):
            t0 = time.monotonic()
            # Leave tool_choice on auto. Forcing a call on round 0 of every
            # user turn sends greetings and clarifying questions through an
            # arbitrary MCP tool, including mutating ones.
            result = self.backend.complete(tools, self.max_tokens)
            elapsed = time.monotonic() - t0

            self.total_input_tokens += result.input_tokens
            self.total_output_tokens += result.output_tokens
            # What the next request would carry, not what this turn cost:
            # the input of one completion is the whole conversation so far,
            # which is what tells a caller the context is filling up. Without
            # it the gateway's compaction never fired and turns kept growing
            # until the model refused one outright.
            self.context_tokens = result.input_tokens

            logger.info(
                "Turn %d round %d — LLM responded in %.2fs "
                "(text=%d chars, tool_calls=%d, thinking=%s, "
                "out_tokens=%d of which reasoning=%d)",
                self._turn,
                round_idx,
                elapsed,
                len(result.text or ""),
                len(result.tool_calls),
                "yes" if result.thinking else "no",
                result.output_tokens,
                result.reasoning_tokens,
            )

            if result.thinking:
                logger.debug("Thinking: %s", result.thinking[:500])
                self._write_event("thinking", {
                    "content": result.thinking,
                    "round": round_idx,
                })
                self._events.on_thinking(result.thinking)

            if not result.needs_tool_response:
                logger.info(
                    "Turn %d — final response: %s",
                    self._turn,
                    (result.text or "")[:200],
                )
                self._write_event("assistant_message", {
                    "content": result.text,
                    "round": round_idx,
                    "elapsed_s": round(elapsed, 3),
                    "total_input_tokens": self.total_input_tokens,
                    "total_output_tokens": self.total_output_tokens,
                })
                return result.text or ""

            tool_results: list[ToolResult] = []
            try:
                tool_results = await self._handle_tool_calls(result, round_idx)
            finally:
                # Even if a tool raised/cancelled, every tool_use must get a
                # matching tool_result, otherwise the conversation becomes
                # permanently invalid and the API rejects every later request.
                seen = {r.tool_call_id for r in tool_results}
                for tc in result.tool_calls:
                    if tc.id not in seen:
                        tool_results.append(ToolResult(
                            tool_call_id=tc.id,
                            content="Tool call aborted.",
                            is_error=True,
                        ))
                self.backend.add_tool_results(tool_results)
            if self._stuck is not None:
                logger.warning("Turn %d — stuck: %s", self._turn, self._stuck)
                self._write_event("stuck", {"content": self._stuck, "round": round_idx})
                raise RepeatedToolCall(self._stuck)
            if self.inbox is not None:
                self.interject(self.inbox())

        msg = (
            "I've reached the maximum number of tool-call rounds. "
            "Please try breaking your request into smaller steps."
        )
        logger.warning("Turn %d — max tool rounds reached", self._turn)
        self._write_event("max_rounds", {"turn": self._turn})
        return msg

    def interject(self, texts: list[str]) -> None:
        """Add messages to the conversation without starting a turn of their own.

        Marked in the transcript, so a reader can tell a message that came
        in mid-turn from the one the turn answers.
        """

        for text in texts:
            logger.info("Turn %d — interjected: %s", self._turn, text[:200])
            self._write_event("user_message", {"content": text, "interjected": True})
            self.backend.add_user_message(text)

    async def _call_local(
        self, tc: Any, handler: Any, round_idx: int, t0: float
    ) -> ToolResult:
        """Run one locally-installed tool, reporting it as any other tool.

        A local tool that raises is reported to the model as a failed tool
        call rather than being allowed to abort the turn: it is usually the
        conversation's own bookkeeping, and losing the answer because the
        notepad broke would be a poor trade.
        """

        try:
            answer = handler(tc.arguments)
            if inspect.isawaitable(answer):
                answer = await answer
            content, is_error = str(answer), False
        except Exception as exc:
            logger.exception("local tool %s failed", tc.name)
            content, is_error = f"{tc.name} failed: {exc}", True

        elapsed = time.monotonic() - t0
        self._write_event("tool_result", {
            "name": tc.name,
            "content_length": len(content),
            "excerpt": content[:TOOL_EXCERPT_CHARS],
            "elapsed_s": round(elapsed, 3),
            "is_error": is_error,
            "round": round_idx,
            "local": True,
        })
        self._events.on_tool_result(tc.name, content, is_error=is_error)
        return ToolResult(
            tool_call_id=tc.id,
            content=self._for_the_model(tc.name, content)
            + self._repeated(tc, content, is_error),
            is_error=is_error,
        )

    def _repeated(self, tc: Any, text: str, is_error: bool) -> str:
        """What to add to a result the model has already had, unchanged.

        Counts calls that match the one before in name, arguments and
        answer. Only failures count: polling a run returns the same answer
        many times while it is queued, and that is not a loop.
        """
        key = (tc.name, json.dumps(tc.arguments, sort_keys=True, default=str), text)
        if key == self._last_call:
            self._repeats += 1
        else:
            self._last_call, self._repeats = key, 1
        if not is_error or self._repeats < REPEAT_WARN:
            return ""
        if self._repeats >= REPEAT_LIMIT:
            self._stuck = (
                f"{tc.name} was sent {self._repeats} times in a row with the same "
                f"arguments and failed the same way each time: {text[:TOOL_EXCERPT_CHARS]}"
            )
        return (
            f"\n\n[You have now sent this exact {tc.name} call {self._repeats} "
            "times in a row and had this same error each time. Sending it "
            "again will not change the answer: change the arguments to fix "
            "what the error names, or take another approach.]"
        )

    async def _handle_tool_calls(
        self, result: ChatResult, round_idx: int
    ) -> list[ToolResult]:
        """Execute all tool calls and return results for the backend."""
        results: list[ToolResult] = []

        for tc in result.tool_calls:
            self._events.on_tool_call(tc.name, tc.arguments)
            logger.info(
                "Turn %d round %d — calling tool %s(%s)",
                self._turn,
                round_idx,
                tc.name,
                _summarize_dict(tc.arguments),
            )
            self._write_event("tool_call", {
                "name": tc.name,
                "arguments": tc.arguments,
                "round": round_idx,
            })

            t0 = time.monotonic()
            local = self.local_tools.get(tc.name)
            if local is not None:
                results.append(await self._call_local(tc, local[1], round_idx, t0))
                continue

            try:
                self._tool_in_flight = True
                resolved = self.mcp.resolve_tool_name(tc.name)
                # Resolve against the full catalog, then refuse anything
                # outside this agent's allow-list. An alias can name a
                # mutating tool the model was never offered.
                if resolved is not None and not self._tool_allowed(resolved):
                    denied = resolved
                    resolved = None
                else:
                    denied = None
                if resolved is None:
                    known = ", ".join(
                        t["name"] for t in await self._scoped_tools()
                    )
                    if denied is not None:
                        instead = self.withheld.get(denied, "")
                        result_text = (
                            f"Tool '{denied}' is not available to this "
                            f"agent. {instead} Available tools: {known}."
                        )
                    else:
                        result_text = (
                            f"Unknown tool '{tc.name}'. Available tools: "
                            f"{known}. Scoring components are listed by "
                            "list_scoring_components."
                        )
                    elapsed = time.monotonic() - t0
                    logger.warning(
                        "Turn %d round %d — unknown tool %s (%.2fs)",
                        self._turn,
                        round_idx,
                        tc.name,
                        elapsed,
                    )
                    self._write_event("tool_result", {
                        "name": tc.name,
                        "content_length": len(result_text),
                        "excerpt": result_text[:TOOL_EXCERPT_CHARS],
                        "elapsed_s": round(elapsed, 3),
                        "is_error": True,
                        "round": round_idx,
                    })
                    self._events.on_tool_result(
                        tc.name, result_text, is_error=True
                    )
                    results.append(ToolResult(
                        tool_call_id=tc.id,
                        content=result_text + self._repeated(tc, result_text, True),
                        is_error=True,
                    ))
                    continue
                if resolved != tc.name:
                    logger.info(
                        "Turn %d round %d — aliasing tool %s -> %s",
                        self._turn,
                        round_idx,
                        tc.name,
                        resolved,
                    )
                async def on_progress(
                    progress: float,
                    total: float | None,
                    message: str | None,
                    _name: str = tc.name,
                    _round: int = round_idx,
                ) -> None:
                    """Pass a tool's own progress on to whoever is watching.

                    Only tools that call ``Context.report_progress`` send
                    any; the rest never call this and are shown as work of
                    unknown length, as before.
                    """
                    self._write_event("tool_progress", {
                        "name": _name,
                        "done": progress,
                        "total": total,
                        "message": message,
                        "round": _round,
                    })
                    # Not every sink reports progress: the CLI and the
                    # supervisor render differently and do not need it.
                    sink = getattr(self._events, "on_tool_progress", None)
                    if sink is not None:
                        sink(_name, progress, total, message)

                if _reports_progress(self.mcp):
                    result_text = await self.mcp.call_tool(
                        resolved, tc.arguments, on_progress=on_progress
                    )
                else:
                    result_text = await self.mcp.call_tool(resolved, tc.arguments)
                elapsed = time.monotonic() - t0

                logger.info(
                    "Turn %d round %d — tool %s returned %d chars in %.2fs",
                    self._turn,
                    round_idx,
                    tc.name,
                    len(result_text),
                    elapsed,
                )
                reported = _reported_error(result_text)
                failed = reported is not None
                event = {
                    "name": tc.name,
                    "content_length": len(result_text),
                    "excerpt": result_text[:TOOL_EXCERPT_CHARS],
                    "elapsed_s": round(elapsed, 3),
                    "is_error": failed,
                    "round": round_idx,
                }
                if failed:
                    event["error"] = reported
                self._write_event("tool_result", event)

                self._events.on_tool_result(tc.name, result_text, is_error=failed)

                results.append(ToolResult(
                    tool_call_id=tc.id,
                    content=self._for_the_model(tc.name, result_text)
                    + self._repeated(tc, result_text, failed),
                    is_error=failed,
                ))
            except Exception as exc:
                elapsed = time.monotonic() - t0
                error_msg = f"Tool call failed: {exc}"
                logger.error(
                    "Turn %d round %d — tool %s failed after %.2fs: %s",
                    self._turn,
                    round_idx,
                    tc.name,
                    elapsed,
                    exc,
                    exc_info=True,
                )
                self._write_event("tool_result", {
                    "name": tc.name,
                    "error": str(exc),
                    "elapsed_s": round(elapsed, 3),
                    "is_error": True,
                    "round": round_idx,
                })
                self._events.on_tool_result(tc.name, error_msg, is_error=True)
                results.append(ToolResult(
                    tool_call_id=tc.id,
                    content=error_msg + self._repeated(tc, error_msg, True),
                    is_error=True,
                ))
            finally:
                self._tool_in_flight = False

        return results

    def _for_the_model(self, name: str, result: str) -> str:
        """A tool result as the conversation keeps it.

        The browser and the transcript are given the whole thing; only
        what the model re-reads on every later round is bounded.
        """
        if len(result) <= MAX_TOOL_RESULT_CHARS:
            return result
        head = MAX_TOOL_RESULT_CHARS * 2 // 3
        tail = MAX_TOOL_RESULT_CHARS - head
        cut = len(result) - MAX_TOOL_RESULT_CHARS
        logger.warning(
            "Turn %d — %s returned %d chars; keeping %d for the conversation",
            self._turn, name, len(result), MAX_TOOL_RESULT_CHARS,
        )
        return (
            result[:head]
            + f"\n\n[... {cut:,} characters cut from the middle of this "
            f"{name} result to keep the conversation inside the model's "
            "context. Ask again for the part you need: a narrower query, "
            "fewer molecules, or one record at a time.]\n\n"
            + result[-tail:]
        )

    @property
    def context_window(self) -> int | None:
        """Tokens the endpoint accepts in one request, when it says."""
        return self.backend.context_window

    def reset_history(self) -> None:
        """Clear conversation history to start a fresh session."""
        logger.info("Conversation history reset")
        self._write_event("reset", {})
        self._turn = 0
        if self._system_prompt is not None:
            prompt = self._system_prompt
        else:
            from reinvent4_agent.system_prompt import SYSTEM_PROMPT

            prompt = SYSTEM_PROMPT
        self.backend.reset(prompt)

    # ------------------------------------------------------------------
    # Background monitoring
    # ------------------------------------------------------------------

    async def start_monitoring(self, poll_interval: float = 120.0) -> None:
        """Start background job monitoring."""
        if self._monitor_task is not None:
            return
        self._monitor_running = True
        self._seen_alerts.clear()
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(poll_interval)
        )
        logger.info("Monitor started (poll every %.0fs)", poll_interval)

    async def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._monitor_running = False
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        logger.info("Monitor stopped")

    async def _monitor_loop(self, poll_interval: float) -> None:
        """Periodically call get_job_alerts and act on results."""
        while self._monitor_running:
            try:
                await asyncio.sleep(poll_interval)
                if not self._monitor_running:
                    break
                await self._check_alerts()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.error("Monitor loop error", exc_info=True)
                await asyncio.sleep(30)

    async def _check_alerts(self) -> None:
        """Fetch alerts and dispatch actions based on severity."""
        # Skip alert processing entirely when a long-running tool call is in
        # flight — firing self.chat() now would inject a user message into an
        # incomplete tool_use/tool_result sequence, which Anthropic rejects.
        if self._tool_in_flight:
            return
        async with self._chat_lock:
            alerts_json = await self.mcp.call_tool("get_job_alerts", {})

        try:
            alerts = json.loads(alerts_json)
        except (json.JSONDecodeError, TypeError):
            return

        if not alerts:
            return

        for alert in alerts:
            job_id = alert.get("job_id", "")
            alert_type = alert.get("type", "")
            key = (job_id, alert_type)
            if key in self._seen_alerts:
                continue
            self._seen_alerts.add(key)

            severity = alert.get("severity", "low")
            logger.info(
                "Monitor alert [%s] job=%s type=%s: %s",
                severity, job_id, alert_type,
                alert.get("message", "")[:200],
            )

            self._events.on_monitor_alert(alert)

            prompt = (
                f"[MONITOR] Alert ({severity}): "
                f"{alert.get('message', '')} "
                f"[job_id={job_id}, type={alert_type}]"
            )
            try:
                async with self._chat_lock:
                    response = await self.chat(prompt)
                self._events.on_monitor_response(response)
                self._log_monitor_action(alert, response)
            except Exception:
                logger.error(
                    "Monitor autonomous action failed",
                    exc_info=True,
                )

    def _log_monitor_action(
        self, alert: dict[str, Any], response: str | None,
    ) -> None:
        """Write a monitor action to the job's monitor_actions.jsonl."""
        self._write_event("monitor_alert", {
            "alert": alert,
            "response": response[:500] if response else None,
        })
        working_dir = (alert.get("details") or {}).get("working_dir")
        if not working_dir:
            return
        record = {
            "ts": datetime.now().isoformat(),
            "alert_type": alert.get("type"),
            "severity": alert.get("severity"),
            "message": alert.get("message"),
            "action": (
                "Autonomous LLM response" if response
                else "Notified user (medium severity)"
            ),
            "agent_response": response[:1000] if response else None,
        }
        try:
            p = Path(working_dir) / "monitor_actions.jsonl"
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError:
            logger.debug("Failed to write monitor action log", exc_info=True)

    # ------------------------------------------------------------------
    # Transcript
    # ------------------------------------------------------------------

    def _write_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Append a structured event to the JSONL transcript file."""
        if self._transcript_path is None:
            return
        record = {
            "ts": datetime.now().isoformat(),
            "turn": self._turn,
            "event": event_type,
            **data,
        }
        try:
            with open(self._transcript_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError:
            logger.debug("Failed to write transcript event", exc_info=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_transcript(log_dir: str | None, name: str = "agent") -> str | None:
    """Create the transcript JSONL file path if log_dir is set.

    ``name`` is included in the filename so that the Driver and Reviewer
    agents in a co-scientist run write to separate transcripts instead of
    interleaving into one file.
    """
    if not log_dir:
        return None
    d = Path(log_dir).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = str(d / f"session_{ts}_{name}.jsonl")
    logger.info("Transcript file: %s", path)
    return path


def _summarize_dict(d: dict[str, Any], max_len: int = 120) -> str:
    """Short repr of a dict for log messages."""
    s = json.dumps(d, default=str)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


