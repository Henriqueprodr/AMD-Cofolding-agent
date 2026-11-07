"""Where the services are, when DNS cannot answer that.

On Kubernetes every service has a stable name, so the deployment sets
``SYNTHESIS_SERVICE_URL`` and friends once and the tools read them from the
environment. Under Slurm there are no stable names: each service lands on
whichever node the scheduler gave it, on a port chosen to avoid collisions,
and none of that is known until the job starts.

So the launcher writes what it allocated to a registry file, and this module
turns that file back into the same environment variables the tools already
read. Nothing downstream changes -- `structure_prep.py` still calls
``os.environ.get("SP_SERVICE_URL", ...)`` and neither knows nor cares that on
Slurm the answer arrived via a file.

**The environment always wins.** A value already set is never overwritten,
which is what makes this safe to call unconditionally: on Kubernetes every
variable is already set from the ConfigMap, so applying a registry is a no-op
even if a stale file is lying around.

Registry format -- a flat JSON object of service name to base URL:

    {
      "synthesis": "http://nid001234:8088",
      "structure_prep": "http://nid001234:8089"
    }

Env:
    DD_SERVICE_REGISTRY   path to the registry file
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import MutableMapping, Optional

logger = logging.getLogger(__name__)

REGISTRY_PATH_VAR = "DD_SERVICE_REGISTRY"

#: Service name in the registry -> the variable the tools read. Kept explicit
#: rather than derived, because these names are a contract with the tools and
#: a clever transformation would let a rename pass silently.
SERVICE_VARS: dict[str, str] = {
    "structure_prep": "SP_SERVICE_URL",
    "annotation": "ANNOTATION_SERVICE_URL",
    "synthesis": "SYNTHESIS_SERVICE_URL",
    "interaction_profiler": "INTERACTION_PROFILER_URL",
    "boltz": "BOLTZ_SERVICE_URL",
    "posebusters": "POSEBUSTERS_SERVICE_URL",
    "campaign_store": "CAMPAIGN_STORE_URL",
    "boltz_gateway_stats": "BOLTZ_GATEWAY_STATS_URL",
}


def load_registry(path: Optional[Path | str] = None) -> dict[str, str]:
    """Read the registry file, or return empty if there is not one to read.

    A missing file is the normal case on Kubernetes and is not worth a
    warning. A malformed one is worth a loud warning but not an exception:
    losing service discovery should degrade to the built-in defaults rather
    than stop the agent from starting.
    """

    raw = path or os.environ.get(REGISTRY_PATH_VAR)
    if not raw:
        return {}

    file = Path(raw)
    if not file.is_file():
        logger.warning("service registry %s does not exist", file)
        return {}

    try:
        loaded = json.loads(file.read_text())
    except (OSError, ValueError) as exc:
        logger.warning("could not read the service registry %s: %s", file, exc)
        return {}

    if not isinstance(loaded, dict):
        logger.warning("service registry %s is not an object", file)
        return {}

    return {str(k): str(v) for k, v in loaded.items()}


def apply_registry(
    path: Optional[Path | str] = None,
    environ: Optional[MutableMapping[str, str]] = None,
) -> list[str]:
    """Fill in service URL variables that are not already set.

    Returns the variables it set, so a caller can log what discovery actually
    contributed rather than guessing.
    """

    env = os.environ if environ is None else environ
    registry = load_registry(path)
    if not registry:
        return []

    applied: list[str] = []
    for service, url in registry.items():
        var = SERVICE_VARS.get(service)
        if var is None:
            logger.warning("service registry names %r, which maps to nothing", service)
            continue
        if env.get(var):
            continue
        env[var] = url
        applied.append(var)

    if applied:
        logger.info("service registry set %s", ", ".join(sorted(applied)))
    return applied


def write_registry(
    services: dict[str, str],
    path: Path | str,
) -> Path:
    """Write a registry file, for the Slurm launcher to call once per run.

    Written to a sibling temporary file and moved into place, so a reader
    starting at the wrong moment sees either the old registry or the new one
    and never half of either.
    """

    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    partial = file.with_suffix(file.suffix + ".part")
    partial.write_text(json.dumps(services, indent=2, sort_keys=True))
    partial.replace(file)
    return file
