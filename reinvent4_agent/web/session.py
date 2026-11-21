# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One agent session, driven from the browser.

Mirrors the construction block in ``cli.py`` -- MCP client, backend, Driver,
optional read-only Reviewer -- but imports nothing from ``display`` or any
other terminal code. The CLI and this module are two front ends over the same
agent.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import OrderedDict
from typing import Any

from reinvent4_agent.agent import Agent
from reinvent4_agent.backends import create_backend
from reinvent4_agent.coordinator import (
    REVIEWER_READ_ONLY_TOOLS,
    build_manual_review_prompt,
)
from reinvent4_agent.mcp_client import MCPClient
from reinvent4_agent.web import protocol
from reinvent4_agent.web.events import (
    TurnCancelled,
    TurnContext,
    WebEventHandler,
)
from reinvent4_agent.web.hub import SessionHub

logger = logging.getLogger(__name__)

#: Full tool-result payloads kept for the "full result" link.
RESULT_STORE_LIMIT = 200

HELP_TEXT = """\
Commands
  /help, /?        show this help
  /review [job]    summon the read-only Reviewer (multi-agent mode only)
  /reset, /clear   clear the conversation (both agents)
  /exit, /quit     close the tab; Ctrl-C in the launching terminal stops
                   the server

Anything else is sent to the agent. Click `3D` on a score-table row, or
enter a PDB ID in the viewer panel, to see a structure.
"""


class AgentSession:
    """Owns the agents and serialises turns for every connected tab."""

    def __init__(self, args, hub: SessionHub) -> None:
        self.args = args
        self.hub = hub
        self.session_id = uuid.uuid4().hex[:12]

        self.mcp: MCPClient | None = None
        #: Second, read-only MCP session for viewer lookups. The chat lock
        #: can be held for minutes by an LLM turn; a click-to-3D must not
        #: queue behind it.
        self.viewer_mcp: MCPClient | None = None

        self.driver: Agent | None = None
        self.reviewer: Agent | None = None
        self.tools: list[str] = []

        self._ctx = TurnContext()
        self._turn_guard = asyncio.Lock()
        self._results: OrderedDict[str, str] = OrderedDict()
        #: Shared-volume paths that actually appeared in a tool result, so
        #: the browser cannot enumerate the volume by guessing.
        self.seen_paths: set[str] = set()
        self._stack: list[Any] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        args = self.args
        loop = asyncio.get_running_loop()

        from reinvent4_agent.cli import _apply_web_addendum, _resolve_api_key
        from reinvent4_agent.system_prompt import (
            REVIEWER_SYSTEM_PROMPT,
            SYSTEM_PROMPT,
        )

        api_key = _resolve_api_key(args)
        driver_prompt, reviewer_prompt = _apply_web_addendum(
            args, SYSTEM_PROMPT, REVIEWER_SYSTEM_PROMPT
        )

        self.mcp = await self._enter(
            MCPClient(
                server_command=args.server_command,
                server_args=args.server_args,
            )
        )
        tools = await self.mcp.list_tools()
        self.tools = [t["name"] for t in tools]

        def make_backend(prompt: str):
            return create_backend(
                args.backend,
                model=args.model,
                base_url=args.base_url,
                api_key=api_key,
                gateway_key=args.gateway_key,
                api_version=args.api_version,
                verify_ssl=not args.no_verify_ssl,
                thinking_budget=args.thinking_budget,
                reasoning_effort=args.reasoning_effort,
                system_prompt=prompt,
            )

        self.driver = Agent(
            self.mcp,
            make_backend(driver_prompt),
            max_tokens=args.max_tokens,
            log_dir=args.log_dir,
            event_handler=WebEventHandler(
                self.hub,
                loop,
                agent_label="driver",
                context=self._ctx,
                store_payload=self._store_result,
                note_payload=self._remember_paths,
            ),
            name="driver",
            system_prompt=driver_prompt,
        )

        if args.multi_agent:
            self.reviewer = Agent(
                self.mcp,
                make_backend(reviewer_prompt),
                max_tokens=args.max_tokens,
                log_dir=args.log_dir,
                event_handler=WebEventHandler(
                    self.hub,
                    loop,
                    agent_label="reviewer",
                    context=self._ctx,
                    store_payload=self._store_result,
                    note_payload=self._remember_paths,
                ),
                name="reviewer",
                system_prompt=reviewer_prompt,
                allowed_tools=set(REVIEWER_READ_ONLY_TOOLS),
            )

        # Viewer sidecar: a second dd-mcp process, used only for the
        # stateless structure lookups the browser triggers.
        try:
            self.viewer_mcp = await self._enter(
                MCPClient(
                    server_command=args.server_command,
                    server_args=args.server_args,
                )
            )
            await self.viewer_mcp.list_tools()
        except Exception as exc:  # noqa: BLE001 - optional optimisation
            logger.warning("viewer MCP session unavailable: %s", exc)
            self.viewer_mcp = None

        if not args.no_monitor:
            await self.driver.start_monitoring(args.monitor_interval)

    async def _enter(self, ctx_manager):
        obj = await ctx_manager.__aenter__()
        self._stack.append(ctx_manager)
        return obj

    async def close(self) -> None:
        if self.driver is not None:
            try:
                await self.driver.stop_monitoring()
            except Exception:  # noqa: BLE001 - shutting down anyway
                logger.debug("monitor shutdown raised", exc_info=True)
        for ctx_manager in reversed(self._stack):
            try:
                await ctx_manager.__aexit__(None, None, None)
            except Exception:  # noqa: BLE001 - shutting down anyway
                logger.debug("MCP shutdown raised", exc_info=True)
        self._stack.clear()

    # ------------------------------------------------------------------
    # Session description
    # ------------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        return protocol.session(
            session_id=self.session_id,
            backend=self.args.backend,
            model=self.args.model,
            multi_agent=self.reviewer is not None,
            web_tools=(
                self.args.backend == "claude-code"
                and not getattr(self.args, "no_web_tools", False)
            ),
            tools=self.tools,
            rdkit_available="embed_molecule_3d" in self.tools,
        )

    # ------------------------------------------------------------------
    # Turns
    # ------------------------------------------------------------------

    def request_stop(self) -> None:
        """Cooperative stop: takes effect at the next tool call."""
        self._ctx.cancel = True

    async def handle_input(self, text: str) -> None:
        """Dispatch a line from the browser: command or chat."""
        stripped = (text or "").strip()
        if not stripped:
            return

        command = stripped.lower()
        if command in ("/help", "/?"):
            self.hub.publish(protocol.status(HELP_TEXT))
            return
        if command in ("/reset", "/clear"):
            self.driver.reset_history()
            if self.reviewer is not None:
                self.reviewer.reset_history()
            self.hub.clear_replay()
            self.hub.publish(protocol.session_reset())
            self.hub.publish(protocol.status("Conversation cleared."))
            return
        if command in ("/exit", "/quit"):
            self.hub.publish(
                protocol.status(
                    "Close the tab to leave. Ctrl-C in the launching "
                    "terminal stops the server."
                )
            )
            return
        if command.startswith("/review"):
            if self.reviewer is None:
                self.hub.publish(
                    protocol.status(
                        "The Reviewer is off. Restart with --multi-agent "
                        "to enable it.",
                        level="warn",
                    )
                )
                return
            parts = stripped.split(maxsplit=1)
            job_id = parts[1].strip() if len(parts) > 1 else None
            await self._run_turn(
                build_manual_review_prompt(job_id),
                agent=self.reviewer,
                origin="review",
                echo=stripped,
            )
            return

        await self._run_turn(
            stripped, agent=self.driver, origin="user", echo=stripped
        )

    async def _run_turn(
        self, prompt: str, *, agent: Agent, origin: str, echo: str
    ) -> None:
        if self._turn_guard.locked():
            self.hub.publish(
                protocol.status(
                    "A turn is already running; wait for it to finish.",
                    level="warn",
                )
            )
            return

        async with self._turn_guard:
            turn_id = uuid.uuid4().hex[:12]
            self._ctx.turn_id = turn_id
            self._ctx.cancel = False
            self._ctx.call_seq = 0

            self.hub.publish(protocol.user_echo(turn_id, echo))
            if agent._chat_lock.locked():
                self.hub.publish(
                    protocol.status(
                        "Waiting for the background job monitor to finish..."
                    )
                )
            self.hub.publish(protocol.busy(True, reason=origin))
            self.hub.publish(protocol.turn_start(turn_id, agent.name, origin))

            started = time.monotonic()
            ok = True
            try:
                # Same lock cli.py takes: the monitor and the Reviewer share
                # one MCP stdio session with the Driver.
                async with agent._chat_lock:
                    reply = await agent.chat(prompt)
                self.hub.publish(
                    protocol.assistant_message(turn_id, agent.name, reply)
                )
            except TurnCancelled:
                ok = False
                self.hub.publish(protocol.status("Turn stopped.", "warn"))
            except ConnectionError as exc:
                ok = False
                self.hub.publish(protocol.error(str(exc), turn_id))
            except Exception as exc:  # noqa: BLE001 - reported to the user
                ok = False
                logger.exception("turn failed")
                self.hub.publish(
                    protocol.error(f"{type(exc).__name__}: {exc}", turn_id)
                )
            finally:
                self.hub.publish(
                    protocol.turn_end(
                        turn_id,
                        agent.name,
                        ok=ok,
                        elapsed_ms=int((time.monotonic() - started) * 1000),
                        tokens={
                            "input": agent.total_input_tokens,
                            "output": agent.total_output_tokens,
                            "total": agent.total_tokens,
                        },
                    )
                )
                self.hub.publish(protocol.busy(False))
                self._ctx.cancel = False

    # ------------------------------------------------------------------
    # Result store
    # ------------------------------------------------------------------

    def _store_result(self, ref: str, payload: str) -> str:
        self._results[ref] = payload
        self._results.move_to_end(ref)
        while len(self._results) > RESULT_STORE_LIMIT:
            self._results.popitem(last=False)
        return ref

    def get_result(self, ref: str) -> str | None:
        return self._results.get(ref)

    def _remember_paths(self, payload: str) -> None:
        """Note /shared paths a tool mentioned, for the viewer allowlist."""
        for token in payload.replace('"', " ").replace(",", " ").split():
            if token.endswith((".pdb", ".cif", ".mmcif", ".ent", ".sdf")):
                self.seen_paths.add(token.lstrip("/"))
