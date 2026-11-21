# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``dd-web`` entry point.

Takes the web server's own flags, then hands everything else to the ordinary
``dd-agent`` parser -- so every backend, model, MCP and monitor flag behaves
identically in the browser and in the terminal.
"""

from __future__ import annotations

import argparse
import logging
import sys
import webbrowser

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8765


def _parse_web_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        prog="dd-web",
        add_help=False,
        description=(
            "Browser front end for the drug-discovery agent. Accepts every "
            "dd-agent flag as well; run `dd-web --help` for both lists."
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1, loopback only)",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Port (default: 8765)"
    )
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help=(
            "Permit a non-loopback --host. There is NO authentication: "
            "anyone who can reach the port controls the agent."
        ),
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open a browser once the server is up",
    )
    parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        dest="show_help",
        help="Show this help, then the dd-agent options",
    )
    return parser.parse_known_args(argv)


def main() -> None:
    from reinvent4_agent.cli import (
        _apply_web_tools_env,
        _configure_logging,
        parse_args,
    )

    web_args, rest = _parse_web_args(sys.argv[1:])
    if web_args.show_help:
        _parse_web_args(["--nonexistent-to-print-usage"])  # pragma: no cover
        print(__doc__)
        print("\n--- dd-agent options ---\n")
        parse_args(["--help"])
        return

    args = parse_args(rest)
    _apply_web_tools_env(args)

    log_file = _configure_logging(args)
    if log_file:
        logger.info("Session log: %s", log_file)

    is_loopback = web_args.host in ("127.0.0.1", "localhost", "::1")
    if not is_loopback and not web_args.allow_remote:
        print(
            f"Refusing to bind {web_args.host} without --allow-remote: this "
            "server has no authentication.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    import uvicorn

    from reinvent4_agent.web.app import create_app

    origins = {
        f"http://{host}:{web_args.port}"
        for host in ("127.0.0.1", "localhost")
    }
    if not is_loopback:
        origins.add(f"http://{web_args.host}:{web_args.port}")

    app = create_app(args, allowed_origins=origins)

    url = f"http://{'127.0.0.1' if is_loopback else web_args.host}:{web_args.port}"
    print(f"\n  Drug Discovery Agent — {url}\n")
    if not is_loopback:
        print("  WARNING: reachable off this machine, with no auth.\n")
    if web_args.open:
        webbrowser.open(url)

    uvicorn.run(
        app,
        host=web_args.host,
        port=web_args.port,
        log_level="warning",
    )


if __name__ == "__main__":  # pragma: no cover
    main()
