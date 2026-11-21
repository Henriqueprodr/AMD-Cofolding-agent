# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Structure access: 3D coordinates for a molecule, and reading shared files.

Two jobs, one module:

``embed_molecule_3d`` turns a SMILES into real 3D coordinates with RDKit.
Everything else in this stack gets its geometry from Boltz2, which needs a
GPU, so without this there is nothing to look at on a machine that has none --
and a generated molecule cannot be inspected until it has been cofolded.
These are computed conformers, NOT predicted poses and NOT experimental
structures; callers must say so.

``read_structure_file`` hands back a structure that already exists on the
shared volume (a Boltz cofold output, a PoseBusters pose, a file the user
dropped in). Implements the ``read_structure_file`` tool the README lists
under Phase 9.

Both return ``{"format": ..., "data": <text>, "meta": {...}}`` so a viewer can
hand ``data`` straight to a molecular renderer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from reinvent4_mcp.tools.external import (
    _STRUCTURE_EXTS,
    _normalize_shared_path,
    _shared_root,
)

logger = logging.getLogger(__name__)

#: 8 MiB. A whole-ribosome mmCIF blows past this; a protein-ligand complex
#: does not.
_MAX_BYTES = 8 * 1024 * 1024
_MAX_HEAVY_ATOMS = 150
_EMBED_SEED = 0xF00D
#: Recognised on top of the structure extensions the services already share.
_EXTRA_EXTS = {".sdf": "sdf", ".mol": "sdf"}
_FORMAT_BY_EXT = {
    ".pdb": "pdb",
    ".ent": "pdb",
    ".cif": "cif",
    ".mmcif": "cif",
    **_EXTRA_EXTS,
}


# ----------------------------------------------------------------------
# SMILES -> 3D
# ----------------------------------------------------------------------


def embed_molecule_3d(
    smiles: str,
    include_hydrogens: bool = False,
    seed: int = _EMBED_SEED,
) -> dict[str, Any]:
    """Generate a 3D conformer for a SMILES with RDKit (ETKDGv3 + MMFF).

    Args:
        smiles: The molecule. Must come from a tool or the user, never memory.
        include_hydrogens: Keep explicit hydrogens in the output.
        seed: Embedding seed, so the same SMILES gives the same coordinates.

    Returns:
        ``{"format": "sdf", "data": <molblock>, "meta": {...}}``, or
        ``{"error": ...}``. An SDF molblock rather than PDB: it carries bond
        orders, aromaticity and formal charges, which a PDB block loses --
        a viewer would then have to guess bonds from interatomic distances
        and would draw aromatic rings wrong.
    """
    smiles = (smiles or "").strip()
    if not smiles:
        return {"error": "Empty SMILES. Provide a molecule to embed."}

    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem, Descriptors
    except ImportError as exc:  # pragma: no cover - container always has it
        logger.warning("RDKit unavailable: %s", exc)
        return {
            "error": (
                "RDKit is not available in this environment, so 3D "
                "coordinates cannot be generated."
            ),
            "code": "rdkit_unavailable",
        }

    RDLogger.DisableLog("rdApp.*")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "error": (
                f"RDKit cannot parse {smiles!r}. If it came from memory or a "
                "web page, re-resolve it with fetch_compound; do not retry "
                "with a guess."
            )
        }

    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy > _MAX_HEAVY_ATOMS:
        return {
            "error": (
                f"{n_heavy} heavy atoms is over the {_MAX_HEAVY_ATOMS} limit "
                "for conformer generation."
            )
        }

    formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
    canonical = Chem.MolToSmiles(mol)
    mol_h = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.maxIterations = 200
    if AllChem.EmbedMolecule(mol_h, params) != 0:
        # Retry once with random coordinates: ETKDG's knowledge-based start
        # fails on some cages and macrocycles.
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol_h, params) != 0:
            return {
                "error": (
                    f"RDKit could not generate 3D coordinates for {smiles!r} "
                    "(embedding failed). The molecule may be over-constrained."
                )
            }

    forcefield = "MMFF94s"
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol_h):
            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=500)
        else:
            AllChem.UFFOptimizeMolecule(mol_h, maxIters=500)
            forcefield = "UFF"
    except Exception as exc:  # noqa: BLE001 - geometry is still usable
        logger.warning("Force-field optimisation failed for %s: %s", smiles, exc)
        forcefield = "none (embedding only)"

    out = mol_h if include_hydrogens else Chem.RemoveHs(mol_h)

    return {
        "format": "sdf",
        "data": Chem.MolToMolBlock(out),
        "meta": {
            "source": "rdkit",
            "method": f"ETKDGv3 + {forcefield}",
            "canonical_smiles": canonical,
            "formula": formula,
            "molecular_weight": round(Descriptors.MolWt(mol), 3),
            "n_atoms": out.GetNumAtoms(),
            "n_heavy_atoms": n_heavy,
            "includes_hydrogens": include_hydrogens,
            "seed": int(seed),
            "note": (
                "A COMPUTED conformer, not an experimental structure and not "
                "a predicted binding pose. Useful for inspecting a molecule's "
                "shape; says nothing about how it binds."
            ),
        },
    }


# ----------------------------------------------------------------------
# Shared-volume structure files
# ----------------------------------------------------------------------


def _resolve_shared(path: str) -> Path:
    """Resolve a /shared-relative path, refusing anything outside it."""
    root = _shared_root().resolve()
    normalized = _normalize_shared_path(path) or ""
    # _normalize_shared_path always re-roots at the literal "/shared", which
    # is the mount point inside the service containers. The root itself is
    # configurable (IP_SHARED_DIR), so re-anchor onto the real one.
    if normalized == "/shared":
        normalized = ""
    elif normalized.startswith("/shared/"):
        normalized = normalized[len("/shared/"):]

    candidate = Path(normalized)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(
            f"Path escapes the shared volume: {path!r}. Use a "
            "/shared-relative path from list_shared_assets."
        )
    return resolved


def _scan_structure(text: str, fmt: str) -> dict[str, Any]:
    """Cheap line scan so a viewer can pick a representation up front."""
    meta: dict[str, Any] = {
        "has_protein": False,
        "has_ligand": False,
        "ligand_resnames": [],
        "chains": [],
        "n_models": 0,
    }
    if fmt == "sdf":
        meta["has_ligand"] = True
        meta["n_models"] = text.count("$$$$") or 1
        return meta

    ligands: set[str] = set()
    chains: set[str] = set()
    for line in text.splitlines():
        if line.startswith("ATOM"):
            meta["has_protein"] = True
            if len(line) > 21:
                chains.add(line[21])
        elif line.startswith("HETATM"):
            if len(line) > 19:
                resname = line[17:20].strip()
                # Solvent and counter-ions are not "the ligand".
                if resname not in ("HOH", "WAT", "NA", "CL", "SO4", "ZN", "MG"):
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


def read_structure_file(
    path: str,
    max_bytes: int = _MAX_BYTES,
) -> dict[str, Any]:
    """Read a structure file from the shared volume.

    Args:
        path: ``/shared``-relative path, as ``list_shared_assets`` returns.
        max_bytes: Refuse anything larger (default 8 MiB).

    Returns:
        ``{"format": "pdb"|"cif"|"sdf", "data": <text>, "meta": {...}}`` or
        ``{"error": ...}``. Oversized files are refused outright rather than
        truncated: half a PDB is not a smaller structure, it is a broken one.
    """
    if not (path or "").strip():
        return {"error": "Empty path. Call list_shared_assets to find one."}

    try:
        resolved = _resolve_shared(path)
    except ValueError as exc:
        return {"error": str(exc)}

    if not resolved.is_file():
        return {
            "error": (
                f"No such file on the shared volume: {path!r}. Call "
                "list_shared_assets for the real paths -- never guess one."
            )
        }

    suffix = resolved.suffix.lower()
    fmt = _FORMAT_BY_EXT.get(suffix)
    if fmt is None:
        known = sorted(_STRUCTURE_EXTS | set(_EXTRA_EXTS))
        return {
            "error": (
                f"{suffix or 'that file'} is not a structure format this tool "
                f"reads (expected one of {', '.join(known)})."
            )
        }

    size = resolved.stat().st_size
    if size > max_bytes:
        return {
            "error": (
                f"{path} is {size} bytes, over the {max_bytes}-byte limit. "
                "Structures are never truncated -- a partial file renders as "
                "nonsense. Ask for a smaller variant, or raise max_bytes."
            ),
            "size_bytes": size,
        }

    try:
        text = resolved.read_text(errors="replace")
    except OSError as exc:
        return {"error": f"Could not read {path}: {exc}"}

    root = _shared_root().resolve()
    try:
        relative = str(resolved.relative_to(root))
    except ValueError:  # pragma: no cover - guarded by _resolve_shared
        relative = str(resolved)

    meta = _scan_structure(text, fmt)
    meta.update({"source": "shared", "path": relative, "bytes": size})
    return {"format": fmt, "data": text, "meta": meta}
