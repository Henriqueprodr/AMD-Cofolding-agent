# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Claude Code backend — drives a logged-in Claude CLI, no API key.

The ``anthropic`` and ``openai`` backends each need their own paid API
credential.  This one instead runs the Claude Code harness through the
Claude Agent SDK (``claude-agent-sdk``), which spawns the ``claude``
binary the user is already signed in to, so an ordinary Claude
subscription powers the agent.  Nothing reads or copies the login
itself: the CLI owns its credentials, exactly as it does when a human
runs it.  For a headless container, ``claude setup-token`` mints a
long-lived token that is passed in as ``CLAUDE_CODE_OAUTH_TOKEN``.

Bridging the two loops
----------------------
``dd-agent`` owns its agentic loop: it asks the backend to complete a
turn, executes any tool calls against the MCP server itself, feeds the
results back, and repeats.  The Agent SDK owns a loop too — it wants to
call tools itself.  To keep *this* project's loop (display, logging,
tool scoping, Driver/Reviewer handoff) intact, the SDK is given an
in-process MCP server whose tool handlers do not do any work: each one
parks the call on a queue, hands it to ``dd-agent`` as a normal
``ToolCall``, and waits for ``add_tool_results()`` to deliver the answer
before returning it to the model.

The SDK is async and ``LLMBackend.complete()`` is sync, so the session
lives in a worker thread with its own event loop, and the two sides
exchange :class:`queue.Queue` messages.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import uuid
from typing import Any

from reinvent4_agent.backends import (
    ChatResult,
    EndpointError,
    LLMBackend,
    ToolCall,
    ToolResult,
)

logger = logging.getLogger(__name__)

#: Name the in-process MCP server is mounted under.  Tool names the
#: model sees are ``mcp__<SERVER_NAME>__<tool>``.
SERVER_NAME = "dd"

#: Sentinel pushed to the worker to shut the session down.
_STOP = object()


class ClaudeCodeBackend(LLMBackend):
    """LLM backend that runs turns through the local Claude CLI.

    Parameters
    ----------
    model:
        Model alias or id the CLI understands (``sonnet``, ``opus``,
        ``claude-sonnet-4-5``, ...).  ``None`` → the CLI's own default.
    system_prompt:
        System prompt for the conversation.
    oauth_token:
        Value for ``CLAUDE_CODE_OAUTH_TOKEN`` (from ``claude
        setup-token``).  ``None`` → whatever the CLI is already logged
        in as, which is the normal case on a developer machine.
    cli_path:
        Explicit path to the ``claude`` binary, when it is not on PATH.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        system_prompt: str,
        oauth_token: str | None = None,
        cli_path: str | None = None,
    ) -> None:
        self.model = model
        self.base_url = None  # for CLI reporting parity with the others
        self._system_prompt = system_prompt
        self._cli_path = cli_path
        #: Only set where there is no interactive login to inherit; the
        #: CLI's own credentials are never read by this process.
        self._oauth_token = oauth_token

        # Queued user turns, drained by the next complete().
        self._pending: list[str] = []

        # Worker-thread plumbing, created on the first complete().
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._to_worker: queue.Queue[Any] = queue.Queue()
        self._to_caller: queue.Queue[tuple[str, Any]] = queue.Queue()
        #: call id -> asyncio.Future carrying the tool result back.
        self._waiting: dict[str, asyncio.Future] = {}
        self._tool_names: tuple[str, ...] = ()
        self._turn_active = False

    # ------------------------------------------------------------------
    # LLMBackend interface
    # ------------------------------------------------------------------

    def complete(
        self,
        tools: list[dict[str, Any]],
        max_tokens: int,
    ) -> ChatResult:
        names = tuple(t["name"] for t in tools)
        if self._thread is None or names != self._tool_names:
            # First turn, or the scoped tool set changed (Driver vs
            # Reviewer): the SDK fixes its tool list per session.
            self._restart(tools)

        if self._pending:
            prompt = "\n\n".join(self._pending)
            self._pending.clear()
            self._to_worker.put(("prompt", prompt))
            self._turn_active = True
        elif not self._turn_active:
            # Nothing to say and no turn in flight — the model has
            # already finished; report an empty turn rather than hang.
            return ChatResult(text=None)

        kind, payload = self._to_caller.get()

        if kind == "tool_call":
            call, text, thinking = payload
            return ChatResult(
                text=text or None,
                thinking=thinking or None,
                tool_calls=[call],
                needs_tool_response=True,
            )
        if kind == "end":
            text, thinking, usage = payload
            self._turn_active = False
            return ChatResult(
                text=text or None,
                thinking=thinking or None,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
            )
        if kind == "error":
            self._turn_active = False
            raise payload

        raise RuntimeError(f"unexpected worker message: {kind}")

    def add_user_message(self, content: str) -> None:
        self._pending.append(content)

    def add_tool_results(self, results: list[ToolResult]) -> None:
        if self._loop is None:
            return
        for res in results:
            self._loop.call_soon_threadsafe(self._deliver, res)

    def repair_dangling_tool_use(self) -> None:
        # The CLI session owns the transcript and never leaves a tool
        # call unanswered on our side: every parked call is resolved,
        # with an error payload if the turn is torn down.
        return

    def reset(self, system_prompt: str) -> None:
        self._system_prompt = system_prompt
        self._shutdown()
        self._pending.clear()
        self._turn_active = False

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _restart(self, tools: list[dict[str, Any]]) -> None:
        self._shutdown()
        self._tool_names = tuple(t["name"] for t in tools)
        ready: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._thread = threading.Thread(
            target=self._run_worker,
            args=(tools, ready),
            name="claude-code-backend",
            daemon=True,
        )
        self._thread.start()
        kind, payload = ready.get()
        if kind == "error":
            self._thread = None
            raise payload

    def _run_worker(
        self,
        tools: list[dict[str, Any]],
        ready: queue.Queue[tuple[str, Any]],
    ) -> None:
        try:
            asyncio.run(self._session(tools, ready))
        except BaseException as exc:  # noqa: BLE001 - reported to caller
            logger.exception("Claude Code session ended with an error")
            try:
                ready.put_nowait(("error", _wrap(exc)))
            except Exception:  # pragma: no cover - queue already served
                pass
            self._to_caller.put(("error", _wrap(exc)))

    async def _session(
        self,
        tools: list[dict[str, Any]],
        ready: queue.Queue[tuple[str, Any]],
    ) -> None:
        try:
            from claude_agent_sdk import (  # noqa: PLC0415 - optional dep
                AssistantMessage,
                ClaudeAgentOptions,
                ClaudeSDKClient,
                ResultMessage,
                TextBlock,
                ThinkingBlock,
                create_sdk_mcp_server,
            )
            from claude_agent_sdk import tool as sdk_tool  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - install problem
            raise EndpointError(
                "The claude-code backend needs the Claude Agent SDK.\n"
                "  Install it with:  pip install claude-agent-sdk\n"
                "  and make sure the `claude` CLI is on PATH "
                "(npm i -g @anthropic-ai/claude-code)."
            ) from exc

        self._loop = asyncio.get_running_loop()

        # Text/thinking seen so far in the current assistant message,
        # flushed to the caller with the next tool call or at turn end.
        buf = {"text": [], "thinking": []}

        def make_handler(tool_name: str):
            async def handler(args: dict[str, Any]) -> dict[str, Any]:
                call_id = f"cc_{uuid.uuid4().hex[:12]}"
                fut: asyncio.Future = self._loop.create_future()
                self._waiting[call_id] = fut
                text = "".join(buf["text"]).strip()
                thinking = "".join(buf["thinking"]).strip()
                buf["text"].clear()
                buf["thinking"].clear()
                self._to_caller.put((
                    "tool_call",
                    (
                        ToolCall(
                            id=call_id, name=tool_name, arguments=dict(args)
                        ),
                        text,
                        thinking,
                    ),
                ))
                result: ToolResult = await fut
                return {
                    "content": [{"type": "text", "text": result.content}],
                    "isError": bool(result.is_error),
                }

            return handler

        sdk_tools = [
            sdk_tool(
                spec["name"],
                (spec.get("description") or spec["name"])[:1000],
                spec.get("input_schema") or {"type": "object"},
            )(make_handler(spec["name"]))
            for spec in tools
        ]
        server = create_sdk_mcp_server(
            name=SERVER_NAME, version="0.1.0", tools=sdk_tools
        )
        allowed = [f"mcp__{SERVER_NAME}__{t['name']}" for t in tools]

        options = ClaudeAgentOptions(
            system_prompt=self._system_prompt,
            mcp_servers={SERVER_NAME: server},
            allowed_tools=allowed,
            # No Read/Write/Bash/WebSearch: this agent's only tools are
            # the scoped MCP ones above.
            tools=[],
            # Nothing to approve interactively in a chat session whose
            # entire tool surface is the list above.
            permission_mode="bypassPermissions",
            # Ignore ~/.claude and any project settings: the session is
            # configured here, not by whatever host it runs on.
            setting_sources=[],
            **({"model": self.model} if self.model else {}),
            **({"cli_path": self._cli_path} if self._cli_path else {}),
            **({"env": {"CLAUDE_CODE_OAUTH_TOKEN": self._oauth_token}}
               if self._oauth_token else {}),
        )

        async with ClaudeSDKClient(options=options) as client:
            ready.put(("ok", None))
            while True:
                item = await asyncio.get_running_loop().run_in_executor(
                    None, self._to_worker.get
                )
                if item is _STOP or item[0] == "stop":
                    break
                _, prompt = item
                buf["text"].clear()
                buf["thinking"].clear()
                usage: dict[str, int] = {}
                try:
                    await client.query(prompt)
                    async for msg in client.receive_response():
                        if isinstance(msg, AssistantMessage):
                            for block in msg.content:
                                if isinstance(block, TextBlock):
                                    buf["text"].append(block.text)
                                elif isinstance(block, ThinkingBlock):
                                    buf["thinking"].append(block.thinking)
                        elif isinstance(msg, ResultMessage):
                            usage = _usage_of(msg)
                            if msg.is_error:
                                detail = (
                                    getattr(msg, "result", None)
                                    or msg.subtype
                                )
                                buf["text"].append(
                                    f"\n[claude-code: {detail}]"
                                )
                except Exception as exc:  # noqa: BLE001 - surfaced below
                    self._fail_waiting(exc)
                    self._to_caller.put(("error", _wrap(exc)))
                    continue

                self._to_caller.put((
                    "end",
                    (
                        "".join(buf["text"]).strip(),
                        "".join(buf["thinking"]).strip(),
                        usage,
                    ),
                ))
                buf["text"].clear()
                buf["thinking"].clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _deliver(self, result: ToolResult) -> None:
        """Hand a tool result to the parked SDK handler (worker loop)."""
        fut = self._waiting.pop(result.tool_call_id, None)
        if fut is None:
            logger.warning(
                "tool result for unknown call %s", result.tool_call_id
            )
            return
        if not fut.done():
            fut.set_result(result)

    def _fail_waiting(self, exc: BaseException) -> None:
        for call_id, fut in list(self._waiting.items()):
            self._waiting.pop(call_id, None)
            if not fut.done():
                fut.set_exception(exc)

    def _shutdown(self) -> None:
        if self._thread is None:
            return
        self._to_worker.put(_STOP)
        self._thread.join(timeout=10)
        self._thread = None
        self._loop = None
        self._waiting.clear()
        self._tool_names = ()
        # Drop anything the dead session left behind.
        while True:
            try:
                self._to_caller.get_nowait()
            except queue.Empty:
                break


def _usage_of(msg: Any) -> dict[str, int]:
    raw = getattr(msg, "usage", None) or {}
    if not isinstance(raw, dict):
        raw = getattr(raw, "__dict__", {}) or {}
    return {
        "input_tokens": int(raw.get("input_tokens") or 0),
        "output_tokens": int(raw.get("output_tokens") or 0),
    }


def _wrap(exc: BaseException) -> BaseException:
    """Turn SDK startup failures into an actionable EndpointError."""
    name = type(exc).__name__
    if name in {"CLINotFoundError", "CLIConnectionError"}:
        return EndpointError(
            "Could not start the Claude CLI.\n"
            f"  Cause: {exc}\n"
            "  Fix:   install it (npm i -g @anthropic-ai/claude-code), "
            "log in with `claude`, and for a container pass a token from "
            "`claude setup-token` as CLAUDE_CODE_OAUTH_TOKEN."
        )
    if isinstance(exc, BaseException) and "credit" in str(exc).lower():
        return EndpointError(str(exc))
    return exc


__all__ = ["ClaudeCodeBackend", "SERVER_NAME"]
