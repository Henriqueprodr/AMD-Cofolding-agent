# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Every tool name written down somewhere must be a tool that exists.

Tool names are strings in a dozen places -- allow-lists, deny-lists, refusal
reasons and the system prompt -- and nothing linked them to the registry. A
stale name in a deny-list silently grants access, a stale name in an
allow-list silently removes a feature, and a misspelling in the prompt sends
the model after a tool that was never there. Each of those is invisible until
someone reads both sides at once.
"""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SERVER = _ROOT / "reinvent4_mcp" / "server.py"
_PROMPT = _ROOT / "reinvent4_agent" / "system_prompt.py"


def _server_tools() -> set[str]:
    """Names decorated with ``@mcp.tool()`` in server.py.

    Parsed rather than imported: the server needs an MCP version this
    checkout does not always have, and the decorator is the whole question.
    """

    tree = ast.parse(_SERVER.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            target = dec.func if isinstance(dec, ast.Call) else dec
            if isinstance(target, ast.Attribute) and target.attr == "tool":
                found.add(node.name)
    return found


def _extension_tools() -> set[str]:
    """Names the extension modules register, collected through a stand-in."""

    from reinvent4_mcp.registry import EXTENSION_MODULES

    found: set[str] = set()
    for dotted in EXTENSION_MODULES:
        module = importlib.import_module(dotted)
        mcp = MagicMock()

        def _tool():
            def _wrap(fn):
                found.add(fn.__name__)
                return fn
            return _wrap

        mcp.tool = _tool
        module.register_tools(mcp)
    return found


@pytest.fixture(scope="module")
def registered() -> frozenset[str]:
    names = _server_tools() | _extension_tools()
    # Both halves must contribute, or an empty set would pass every check.
    assert len(names) > 80, f"only {len(names)} tools found; collection is broken"
    return frozenset(names)


def _gating_lists() -> dict[str, frozenset[str]]:
    """Every allow-list and deny-list that decides what a caller may call."""

    from reinvent4_agent.coordinator import REVIEWER_READ_ONLY_TOOLS
    from reinvent4_agent.supervisor import (
        DENIAL_REASONS,
        DRIVER_DENIED_TOOLS,
        GENERAL_DENIED_TOOLS,
        JOB_STARTING_TOOLS,
    )
    from reinvent4_agent.web.server import DIRECT_TOOLS
    from reinvent4_agent.web.session_data import INVENTORY_TOOLS, LAUNCH_TOOLS
    from reinvent4_agent.web.sessions import FLEET_TOOLS, RUN_STARTERS

    return {
        "REVIEWER_READ_ONLY_TOOLS": frozenset(REVIEWER_READ_ONLY_TOOLS),
        "DRIVER_DENIED_TOOLS": frozenset(DRIVER_DENIED_TOOLS),
        "GENERAL_DENIED_TOOLS": frozenset(GENERAL_DENIED_TOOLS),
        "DENIAL_REASONS": frozenset(DENIAL_REASONS),
        "JOB_STARTING_TOOLS": frozenset(JOB_STARTING_TOOLS),
        "DIRECT_TOOLS": frozenset(DIRECT_TOOLS),
        "INVENTORY_TOOLS": frozenset(INVENTORY_TOOLS),
        "LAUNCH_TOOLS": frozenset(LAUNCH_TOOLS),
        "FLEET_TOOLS": frozenset(FLEET_TOOLS),
        "RUN_STARTERS": frozenset(RUN_STARTERS),
    }


@pytest.mark.parametrize("list_name", sorted(_gating_lists()))
def test_every_gated_name_is_a_tool_that_exists(list_name, registered):
    stale = _gating_lists()[list_name] - registered
    assert not stale, (
        f"{list_name} names {sorted(stale)}, which no module registers. "
        "In a deny-list that silently grants access; in an allow-list it "
        "silently removes a feature."
    )


#: Prefixes a tool name starts with in this codebase. A backticked word in
#: the prompt beginning with one of these is being named as a tool, so it has
#: to be one -- which is what catches a typo or a tool since renamed.
_TOOL_PREFIXES = (
    "get_", "list_", "fetch_", "search_", "run_", "predict_", "score_",
    "create_", "update_", "delete_", "start_", "stop_", "profile_",
    "validate_", "prepare_", "submit_", "enqueue_", "take_", "drop_",
    "conduct_", "close_", "finalize_", "atlas_", "rcsb_", "parse_",
    "read_", "resolve_", "extract_", "cancel_", "ingest_", "plan_",
    "check_", "characterize_",
)


def test_every_tool_the_prompt_names_is_registered(registered):
    """The prompt is how the model learns a tool exists.

    A name here that no module registers is a tool the model will try to
    call and be told does not exist, in the middle of the work it was
    explaining.
    """

    text = _PROMPT.read_text(encoding="utf-8")
    quoted = set(re.findall(r"`([a-z][a-z0-9_]+)`", text))
    # The prompt also names tools that do *not* exist, to steer the model off
    # a plausible guess: "listed by `list_scoring_components` (not
    # `list_components`)". Those are the one case where an unregistered name
    # is the point.
    disowned = set(re.findall(r"not `([a-z][a-z0-9_]+)`", text))
    named = {w for w in quoted - disowned if w.startswith(_TOOL_PREFIXES)}

    assert named, "no tool names found in the prompt; the pattern stopped matching"
    assert not named - registered, (
        f"the prompt names {sorted(named - registered)}, which no module registers"
    )
