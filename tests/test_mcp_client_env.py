"""Environment forwarding contracts for the MCP subprocess."""

import ast
import inspect

from reinvent4_agent import pool
from reinvent4_agent.mcp_client import MCPClient
from reinvent4_agent.service_registry import REGISTRY_PATH_VAR, SERVICE_VARS


def test_internal_service_overrides_are_forwarded() -> None:
    required = {
        "BOLTZ_SERVICE_URL",
        "BOLTZ_SHARED_DIR",
        "SYNTHESIS_SERVICE_URL",
        "INTERACTION_PROFILER_URL",
        "SP_SERVICE_URL",
        "IP_SHARED_DIR",
        "ANNOTATION_SERVICE_URL",
    }

    assert required <= MCPClient._ENV_ALLOW_LIST


def test_every_service_the_registry_can_name_is_forwarded() -> None:
    """Discovery that cannot reach the tools is discovery that did not happen.

    The registry is how the Slurm launcher tells the tools which node each
    service landed on; a variable missing here leaves that tool reaching for
    the in-cluster Service name, which resolves nowhere outside Kubernetes.
    """
    missing = set(SERVICE_VARS.values()) - MCPClient._ENV_ALLOW_LIST

    assert not missing


def test_the_registry_itself_is_forwarded() -> None:
    """The subprocess applies the registry too, so it needs the path."""
    assert REGISTRY_PATH_VAR in MCPClient._ENV_ALLOW_LIST


def test_everything_the_pool_reads_is_forwarded() -> None:
    """``conduct_campaign`` sizes the pool from inside the subprocess.

    Read from ``pool`` rather than listed here so a scaler that starts
    consulting a new variable cannot quietly stop working: dropped, it finds
    neither relay nor scheduler, takes the null scaler, and holds a pool of
    zero -- a campaign that looks alive while no task is ever claimed.
    """
    read = {
        node.args[0].value
        for node in ast.walk(ast.parse(inspect.getsource(pool)))
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", getattr(node.func, "id", None))
        in {"getenv", "_env_number"}
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }

    assert read, "found no environment reads in pool.py; the scan is broken"
    assert read <= MCPClient._ENV_ALLOW_LIST
