"""Extension module registry for the DrugDiscovery MCP server.

Each string in EXTENSION_MODULES must resolve to a Python module that
exports ``register_tools(mcp: FastMCP) -> None``.  The server iterates
this list at startup and calls ``register_tools`` on each module so that
all extension tools are registered with the same ``mcp`` instance.

Ordering follows the block numbering in the project plan:
  01 — RCSB/PDB fetcher
  02 — ChEMBL fetcher
  03 — structure preparator
  06 — interaction profiler v2
  07 — AiZynthFinder synthesis planner
  08 — REINVENT generation contracts (LibInvent / LinkInvent)
  09 — Boltz structure conditioning
  10 — UniProt extended features + InterPro domain annotations
  11 — PubChem/ChEMBL cross-database structure check
  12 — scaffold derivation and R-group decomposition
  13 — matched molecular pairs and diversity clustering
  14 — biology beyond ChEMBL (expression, pathways, orthology, genetics)
  15 — patent / freedom-to-operate search (SureChEMBL, EPO OPS)
"""

import importlib
import logging
from typing import Any, Callable

log = logging.getLogger(__name__)

#: Every extension tool's implementation, by the name it is registered under.
#:
#: Extension tools are closures defined inside ``register_tools``, so unlike
#: the tools declared at module level in ``server.py`` they never appear in
#: that module's ``globals()``. Anything that needs to call a tool *by name*
#: -- ``run_ttp_pipeline`` does, since a pipeline step names the tool it
#: wants -- has no other way to reach them, and a pipeline that could drive
#: only the 58 server-module tools and none of the 50 extension ones would
#: not be able to chain the scaffold derivation it exists to chain.
TOOL_IMPLS: dict[str, Callable[..., Any]] = {}

EXTENSION_MODULES: list[str] = [
    "reinvent4_mcp.tools.rcsb",                 # 01 — RCSB/PDB fetcher
    "reinvent4_mcp.tools.chembl",               # 02 — ChEMBL fetcher
    "reinvent4_mcp.tools.structure_prep",        # 03 — structure preparator
    "reinvent4_mcp.tools.interaction_profiler",  # 06 — interaction profiler v2
    "reinvent4_mcp.tools.synthesis",             # 07 — AiZynthFinder synthesis planner
    "reinvent4_mcp.tools.generation",            # 08 — REINVENT generation contracts
    "reinvent4_mcp.tools.boltz_structure",       # 09 — Boltz structure conditioning
    "reinvent4_mcp.tools.uniprot_features",      # 10 — UniProt extended features
    "reinvent4_mcp.tools.interpro",              # 10 — InterPro domain annotations
    "reinvent4_mcp.tools.cross_check",           # 11 — PubChem/ChEMBL structure check
    "reinvent4_mcp.tools.scaffold",              # 12 — scaffold derivation, R-groups
    "reinvent4_mcp.tools.series_analysis",       # 13 — MMP, clustering
    "reinvent4_mcp.tools.biology",               # 14 — expression, pathways, orthology, genetics
    "reinvent4_mcp.tools.patents",               # 15 — patent/FTO search
]


class _RecordingMCP:
    """The MCP server, with every tool it registers also kept in TOOL_IMPLS.

    Registration still goes to the real server untouched; this only takes a
    reference on the way past, so a caller holding a tool *name* can reach
    the function behind it.
    """

    def __init__(self, mcp: object) -> None:
        self._mcp = mcp

    def tool(self, *args: Any, **kwargs: Any) -> Callable[..., Any]:
        decorate = self._mcp.tool(*args, **kwargs)  # type: ignore[attr-defined]

        def take(fn: Callable[..., Any]) -> Any:
            TOOL_IMPLS[fn.__name__] = fn
            return decorate(fn)

        return take

    def __getattr__(self, name: str) -> Any:
        return getattr(self._mcp, name)


def load_extensions(mcp: object) -> None:
    """Import each extension module and call its register_tools(mcp).

    A failure in one extension is logged and isolated;
    it does not prevent other extensions or existing tools from loading.
    """
    recorder = _RecordingMCP(mcp)
    for mod_path in EXTENSION_MODULES:
        try:
            mod = importlib.import_module(mod_path)
            mod.register_tools(recorder)  # type: ignore[attr-defined]
            log.info("Loaded extension: %s", mod_path)
        except Exception as exc:
            log.error(
                "Failed to load extension %s: %s", mod_path, exc, exc_info=True
            )
