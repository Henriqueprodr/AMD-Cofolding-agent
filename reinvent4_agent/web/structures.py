# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resolving a viewer request to 3D coordinates.

Three sources, deliberately split across two processes:

- ``smiles`` and ``shared`` go to the MCP server, because RDKit and the
  ``/shared`` mount live in that container, not on this host.
- ``pdb_id`` is fetched here: moving bytes needs no scientific stack, the
  cluster container may have no egress, and keeping outbound traffic in one
  process makes it trivial to audit (one hard-coded host, nothing else).

All three return ``{"format", "data", "meta"}`` -- a format tag and a text
blob, which is exactly what a molecular viewer takes.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: The only host this process ever fetches from.
_RCSB_HOST = "https://files.rcsb.org/download"
_PDB_ID_RE = re.compile(r"^[0-9A-Za-z]{4}$")
_FETCH_TIMEOUT = 10.0
_MAX_FETCH_BYTES = 8 * 1024 * 1024


class StructureError(Exception):
    """A viewer request that cannot be served, with a user-facing reason."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


# ----------------------------------------------------------------------
# MCP-backed sources
# ----------------------------------------------------------------------


async def _call_mcp(session, tool: str, arguments: dict) -> dict[str, Any]:
    """Call a structure tool, preferring the viewer's own MCP session.

    The chat lock can be held for minutes by an LLM turn; a click in the
    viewer must not queue behind it. Falls back to the shared session (under
    the lock) only when the sidecar could not be started.
    """
    client = session.viewer_mcp
    if client is not None:
        raw = await client.call_tool(tool, arguments)
    else:
        async with session.driver._chat_lock:
            raw = await session.mcp.call_tool(tool, arguments)

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise StructureError(f"{tool} returned unparseable output", 502) from exc

    if isinstance(data, dict) and data.get("error"):
        raise StructureError(str(data["error"]))
    if not isinstance(data, dict) or "data" not in data:
        raise StructureError(f"{tool} returned no structure", 502)
    return data


async def from_smiles(session, smiles: str) -> dict[str, Any]:
    if not (smiles or "").strip():
        raise StructureError("No SMILES given.")
    result = await _call_mcp(
        session, "embed_molecule_3d", {"smiles": smiles}
    )
    result.setdefault("meta", {})["source"] = "rdkit"
    return result


async def from_shared(session, path: str) -> dict[str, Any]:
    """Read a structure the session has actually seen referenced.

    The browser may only ask for paths that appeared in a tool result, so a
    page cannot enumerate the shared volume by guessing. The MCP tool
    confines the path again on its own side.
    """
    cleaned = (path or "").strip().lstrip("/")
    if not cleaned:
        raise StructureError("No path given.")
    if cleaned not in session.seen_paths:
        raise StructureError(
            "That path has not appeared in this session. Ask the agent to "
            "list the shared assets first.",
            403,
        )
    return await _call_mcp(
        session, "read_structure_file", {"path": cleaned}
    )


# ----------------------------------------------------------------------
# RCSB
# ----------------------------------------------------------------------


async def from_pdb_id(pdb_id: str, cache_dir: Path | None) -> dict[str, Any]:
    """Fetch an experimental structure from RCSB, with a disk cache."""
    pdb_id = (pdb_id or "").strip()
    if not _PDB_ID_RE.match(pdb_id):
        raise StructureError(
            f"{pdb_id!r} is not a 4-character PDB ID (e.g. 1ATP)."
        )
    pdb_id = pdb_id.upper()

    for suffix, fmt in ((".pdb", "pdb"), (".cif", "cif")):
        cached = _read_cache(cache_dir, pdb_id, suffix)
        if cached is not None:
            return _structure_payload(pdb_id, cached, fmt, cached_hit=True)

    import httpx

    last_status = None
    async with httpx.AsyncClient(
        timeout=_FETCH_TIMEOUT,
        follow_redirects=False,
        headers={"User-Agent": "dd-agent-viewer"},
    ) as client:
        # Large structures have no legacy PDB file; mmCIF is the fallback.
        for suffix, fmt in ((".pdb", "pdb"), (".cif", "cif")):
            url = f"{_RCSB_HOST}/{pdb_id}{suffix}"
            try:
                response = await client.get(url)
            except httpx.HTTPError as exc:
                raise StructureError(
                    f"Could not reach RCSB: {exc}", 502
                ) from exc
            last_status = response.status_code
            if response.status_code == 404:
                continue
            if response.status_code != 200:
                raise StructureError(
                    f"RCSB returned HTTP {response.status_code} for {pdb_id}.",
                    502,
                )
            text = response.text
            if len(text.encode()) > _MAX_FETCH_BYTES:
                raise StructureError(
                    f"{pdb_id}{suffix} is larger than the "
                    f"{_MAX_FETCH_BYTES}-byte limit.",
                    413,
                )
            _write_cache(cache_dir, pdb_id, suffix, text)
            return _structure_payload(pdb_id, text, fmt)

    raise StructureError(
        f"RCSB has no entry {pdb_id} (last status {last_status}).", 404
    )


def _read_cache(cache_dir: Path | None, pdb_id: str, suffix: str) -> str | None:
    if cache_dir is None:
        return None
    path = cache_dir / f"{pdb_id}{suffix}"
    try:
        return path.read_text() if path.is_file() else None
    except OSError:  # pragma: no cover - unreadable cache is not fatal
        return None


def _write_cache(
    cache_dir: Path | None, pdb_id: str, suffix: str, text: str
) -> None:
    if cache_dir is None:
        return
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"{pdb_id}{suffix}").write_text(text)
    except OSError:  # pragma: no cover - cache is best effort
        logger.debug("could not cache %s", pdb_id)


def _scan(text: str, fmt: str) -> dict[str, Any]:
    """Cheap content scan, so the viewer can pick a representation.

    Deliberately a local copy of the equivalent scan in
    ``reinvent4_mcp.tools.structures``: that package lives in the container
    and pulls in ``requests``/RDKit, which this host-side process has no
    business importing. Twenty lines of line-prefix matching is a cheaper
    price than coupling the two runtimes.
    """
    meta: dict[str, Any] = {
        "has_protein": False,
        "has_ligand": False,
        "ligand_resnames": [],
        "chains": [],
        "n_models": 0,
        "n_atoms": 0,
    }
    if fmt == "sdf":
        meta["has_ligand"] = True
        meta["n_models"] = text.count("$$$$") or 1
        return meta

    solvent = {"HOH", "WAT", "NA", "CL", "SO4", "ZN", "MG"}
    ligands: set[str] = set()
    chains: set[str] = set()
    for line in text.splitlines():
        if line.startswith("ATOM"):
            meta["has_protein"] = True
            meta["n_atoms"] += 1
            if len(line) > 21:
                chains.add(line[21])
        elif line.startswith("HETATM"):
            meta["n_atoms"] += 1
            if len(line) > 19:
                resname = line[17:20].strip()
                if resname not in solvent:
                    ligands.add(resname)
                    meta["has_ligand"] = True
            if len(line) > 21:
                chains.add(line[21])
        elif line.startswith("MODEL"):
            meta["n_models"] += 1

    meta["ligand_resnames"] = sorted(ligands)
    meta["chains"] = sorted(c for c in chains if c.strip())
    meta["n_models"] = meta["n_models"] or 1
    return meta


def _structure_payload(
    pdb_id: str, text: str, fmt: str, cached_hit: bool = False
) -> dict[str, Any]:
    meta = _scan(text, fmt)
    meta.update(
        {
            "source": "rcsb",
            "pdb_id": pdb_id,
            "bytes": len(text.encode()),
            "cached": cached_hit,
            "note": (
                "An EXPERIMENTAL structure from the PDB -- measured, not "
                "predicted and not computed."
            ),
        }
    )
    return {"format": fmt, "data": text, "meta": meta}


# ----------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------


async def resolve(
    session, request: dict[str, Any], cache_dir: Path | None
) -> dict[str, Any]:
    """Serve one viewer request. Raises ``StructureError`` on refusal."""
    source = (request.get("source") or "").strip()
    if source == "smiles":
        result = await from_smiles(session, request.get("smiles", ""))
    elif source == "shared":
        result = await from_shared(session, request.get("path", ""))
    elif source == "pdb_id":
        result = await from_pdb_id(request.get("pdb_id", ""), cache_dir)
    else:
        raise StructureError(
            "source must be one of: smiles, shared, pdb_id."
        )

    label = request.get("label")
    if label:
        result.setdefault("meta", {})["label"] = str(label)[:120]
    return result
