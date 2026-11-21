# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run short Python snippets inside the drug-discovery container.

The agent can score, cofold and profile, but it cannot do the ten-line things
in between: convert nM to pIC50, aggregate a results CSV, compute a Murcko
scaffold, enumerate an R-group series, check that a SMILES parses. Without a
tool it does them in its head, and arithmetic done in an LLM's head is a guess
wearing a number's clothes.

This runs them for real, in the container that already has RDKit, NumPy,
pandas and REINVENT4.

Security model: THE CONTAINER IS THE BOUNDARY. There is no sandbox inside it.
A snippet can read and write everything the MCP server can -- /shared, the job
working directories, the TTP store -- and has outbound network. On the laptop
deployment that boundary is real (the agent runs on the host, this runs in the
container). Under docker-compose on a cluster, dd-agent and dd-mcp share one
container, so it grants nothing the agent did not already have. Operators who
want it off set ``code_exec.enabled: false`` or ``DD_MCP_CODE_EXEC=0``.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_RUN_ROOT_ENV = "DD_CODE_EXEC_DIR"
_DEFAULT_RUN_ROOT = "/shared/code_runs"
_DEFAULT_TIMEOUT = 60
_MAX_TIMEOUT = 300
_MAX_CODE_CHARS = 20_000
_MAX_OUTPUT_CHARS = 20_000
_MAX_LISTED_FILES = 50
_QUERY_ECHO_CHARS = 2_000


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _settings() -> tuple[bool, Path, int, int]:
    """Resolve (enabled, run_root, default_timeout, max_timeout).

    Config supplies the defaults; the environment can only tighten the
    switch, never loosen it, so an operator's ``DD_MCP_CODE_EXEC=0`` wins.
    """
    enabled, run_root, default_t, max_t = (
        True,
        _DEFAULT_RUN_ROOT,
        _DEFAULT_TIMEOUT,
        _MAX_TIMEOUT,
    )
    try:
        from reinvent4_mcp.config import get_config

        cfg = get_config().code_exec
        enabled, run_root = cfg.enabled, cfg.run_dir
        default_t, max_t = cfg.default_timeout_s, cfg.max_timeout_s
    except Exception:  # noqa: BLE001 - config is optional at this layer
        logger.debug("code_exec config unavailable; using built-in defaults")

    if os.environ.get("DD_MCP_CODE_EXEC", "1") == "0":
        enabled = False
    return enabled, Path(os.environ.get(_RUN_ROOT_ENV, run_root)), default_t, max_t


def _run_root() -> Path:
    return _settings()[1]


def _new_run_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _run_root() / f"{stamp}_{uuid.uuid4().hex[:6]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _resolve_work_dir(work_dir: str) -> Path:
    """Validate a caller-supplied run dir, confined to the run root."""
    root = _run_root().resolve()
    candidate = Path(work_dir)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(
            f"work_dir must be a directory previously returned by run_python "
            f"(under {root}); got {work_dir!r}"
        )
    if not resolved.is_dir():
        raise ValueError(f"work_dir does not exist: {work_dir!r}")
    return resolved


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit] + f"\n... [truncated at {limit} characters]", True


def _list_outputs(run_dir: Path) -> tuple[list[dict[str, Any]], bool]:
    """Files the snippet left behind, as /shared-relative paths."""
    try:
        shared = Path(os.environ.get("IP_SHARED_DIR", "/shared")).resolve()
    except Exception:  # noqa: BLE001
        shared = Path("/shared")

    files: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name == "snippet.py":
            continue
        try:
            rel = str(path.resolve().relative_to(shared))
        except ValueError:
            rel = str(path)
        files.append({"path": rel, "size_bytes": path.stat().st_size})
        if len(files) >= _MAX_LISTED_FILES:
            return files, True
    return files, False


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------


def run_python(
    code: str,
    timeout_s: int | None = None,
    work_dir: str | None = None,
) -> dict[str, Any]:
    """Execute a Python snippet in a fresh subprocess and return its output.

    Args:
        code: The snippet. ``print`` whatever should come back.
        timeout_s: Wall-clock limit; clamped to the configured ceiling.
        work_dir: A directory returned by an earlier call, to reuse its files.

    Returns:
        Dict with ``ok``, ``exit_code``, ``timed_out``, ``stdout``, ``stderr``,
        ``duration_s``, ``work_dir`` and ``files``. A snippet that raises is a
        normal result with ``ok: false`` -- not a tool error -- so the model
        can read the traceback and fix it. Only refusals to run at all
        (disabled, empty or oversized code, bad ``work_dir``) return
        ``{"error": ...}``.
    """
    enabled, _, default_timeout, max_timeout = _settings()
    if not enabled:
        return {
            "error": (
                "run_python is disabled on this server "
                "(code_exec.enabled=false or DD_MCP_CODE_EXEC=0). Use the "
                "dedicated tools instead, or ask the user to enable it."
            )
        }

    code = code or ""
    if not code.strip():
        return {"error": "Empty code. Provide a Python snippet to run."}
    if len(code) > _MAX_CODE_CHARS:
        return {
            "error": (
                f"Code is {len(code)} characters, over the {_MAX_CODE_CHARS} "
                "limit. Write intermediate results to a file in work_dir and "
                "continue in a second call."
            )
        }

    timeout = int(timeout_s or default_timeout)
    timeout = max(1, min(timeout, max_timeout))

    try:
        run_dir = _resolve_work_dir(work_dir) if work_dir else _new_run_dir()
    except ValueError as exc:
        return {"error": str(exc)}
    except OSError as exc:
        return {"error": f"Could not create a working directory: {exc}"}

    script = run_dir / "snippet.py"
    script.write_text(code)

    from reinvent4_mcp.utils import subprocess_env

    t0 = time.monotonic()
    timed_out = False
    # start_new_session so the whole process group can be killed: a plain
    # subprocess timeout kills only the direct child, leaving anything it
    # spawned (a multiprocessing pool, say) running in the container.
    proc = subprocess.Popen(
        [sys.executable, "-u", str(script)],
        cwd=str(run_dir),
        env=subprocess_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):  # pragma: no cover
            proc.kill()
        stdout, stderr = proc.communicate()
        stderr = (stderr or "") + f"\n[killed after {timeout}s]"

    duration = time.monotonic() - t0
    stdout, stdout_truncated = _truncate(stdout or "", _MAX_OUTPUT_CHARS)
    stderr, stderr_truncated = _truncate(stderr or "", _MAX_OUTPUT_CHARS)
    files, files_truncated = _list_outputs(run_dir)
    query, query_truncated = _truncate(code, _QUERY_ECHO_CHARS)

    return {
        "query": query,
        "query_truncated": query_truncated,
        "ok": (not timed_out) and proc.returncode == 0,
        "exit_code": proc.returncode,
        "timed_out": timed_out,
        "duration_s": round(duration, 3),
        "stdout": stdout,
        "stdout_truncated": stdout_truncated,
        "stderr": stderr,
        "stderr_truncated": stderr_truncated,
        "work_dir": str(run_dir),
        "work_dir_note": (
            "Files here persist and are visible to the other services (and on "
            "the host under data/shared/). Pass work_dir back to reuse this "
            "directory in a later call."
        ),
        "files": files,
        "files_truncated": files_truncated,
    }
