# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Browser front end for the drug-discovery agent.

The same agent, MCP client and backends the terminal CLI uses, driven over a
WebSocket instead of a Rich console, plus a 3D structure viewer the terminal
cannot offer. ``dd-agent`` is untouched and keeps working.

Install the extra deps with ``pip install -e 'reinvent4_agent[web]'`` and run
``dd-web`` (or ``run-local.sh --web``).
"""

__all__ = ["create_app"]


def create_app(*args, **kwargs):  # pragma: no cover - thin re-export
    """Lazy re-export so importing this package needs no FastAPI."""
    from reinvent4_agent.web.app import create_app as _impl

    return _impl(*args, **kwargs)
