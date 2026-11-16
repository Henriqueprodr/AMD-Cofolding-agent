# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Is this structure already a known compound?

A generative run reports its output as new because nothing in the run has
ever seen it before, which is not the same claim. A scaffold hop can land on
a marketed drug, and a "novel" series can turn out to be a patent example;
both are cheap to find out and expensive to discover late.

The join is the InChIKey, not the name and not the SMILES: one structure has
many valid SMILES strings and a name matches neither reliably nor uniquely,
while the InChIKey is the same string in PubChem and in ChEMBL. The first
block of the key is the connectivity alone, so a stereoisomer or salt of a
known compound is recognised as related rather than missed outright.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# How many ChEMBL records to pull back for one structure. A handful: the
# filter is an exact key, so more than one means salts or parent/child
# records of the same compound.
_CHEMBL_LIMIT = 5


def _inchikey_from_smiles(smiles: str) -> tuple[Optional[str], Optional[str]]:
    """InChIKey for a SMILES, or a reason it could not be computed."""

    try:
        from rdkit import Chem  # noqa: PLC0415
        from rdkit import RDLogger  # noqa: PLC0415
    except ImportError:
        return None, (
            "RDKit is not available here, so the InChIKey cannot be computed. "
            "Pass inchikey= directly if you have it."
        )

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, f"RDKit cannot parse the SMILES {smiles!r}."
    key = Chem.MolToInchiKey(mol)
    if not key:
        return None, f"RDKit produced no InChIKey for {smiles!r}."
    return key, None


def _pubchem_by_inchikey(inchikey: str) -> dict[str, Any]:
    """What PubChem has for this exact key."""

    from reinvent4_mcp.tools.pubchem import fetch_compound_by_inchikey  # noqa: PLC0415

    found = fetch_compound_by_inchikey(inchikey)
    if "error" in found:
        return {"found": False, "note": found["error"]}
    return {
        "found": True,
        "cid": found.get("cid"),
        "title": found.get("title"),
        "iupac_name": found.get("iupac_name"),
        "url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{found.get('cid')}",
    }


def _chembl_by_inchikey(inchikey: str) -> dict[str, Any]:
    """What ChEMBL has for this exact key, including how far it got clinically."""

    from reinvent4_mcp.tools.chembl import (  # noqa: PLC0415
        ChEMBLError,
        _http_get,
        _normalize_molecule,
    )

    try:
        payload = _http_get(
            "/molecule",
            {"molecule_structures__standard_inchi_key": inchikey,
             "limit": _CHEMBL_LIMIT},
        )
    except ChEMBLError as exc:
        return {"found": False, "note": f"ChEMBL lookup failed: {exc}"}

    records = [_normalize_molecule(m) for m in payload.get("molecules") or []]
    if not records:
        return {"found": False}
    return {
        "found": True,
        "records": [
            {
                "molecule_chembl_id": r["molecule_chembl_id"],
                "pref_name": r["pref_name"],
                "max_phase": r["max_phase"],
                "first_approval": r["first_approval"],
                "molecule_type": r["molecule_type"],
            }
            for r in records
        ],
    }


def _check_known(
    smiles: Optional[str] = None,
    inchikey: Optional[str] = None,
) -> dict[str, Any]:
    """Look one structure up in PubChem and ChEMBL at once.

    Args:
        smiles: The structure to check. Converted to an InChIKey with RDKit.
        inchikey: Use this key directly instead of computing one.

    Returns:
        ``{inchikey, pubchem, chembl, verdict, known}``. ``known`` is True if
        either database has the exact structure. ``verdict`` says which, and
        names the clinical phase when ChEMBL has one, since a hit on an
        approved drug means something different from a hit on a screening
        compound. A database that could not be reached says so in its own
        block rather than being counted as "not found" -- an unreachable
        PubChem must not read as evidence of novelty.
    """

    if not smiles and not inchikey:
        return {"error": "Provide smiles= or inchikey=."}

    if not inchikey:
        inchikey, why = _inchikey_from_smiles(str(smiles).strip())
        if inchikey is None:
            return {"error": why}

    inchikey = inchikey.strip().upper()
    pubchem = _pubchem_by_inchikey(inchikey)
    chembl = _chembl_by_inchikey(inchikey)

    hits = []
    if pubchem.get("found"):
        hits.append(f"PubChem CID {pubchem.get('cid')}")
    if chembl.get("found"):
        first = chembl["records"][0]
        phase = first.get("max_phase")
        label = first.get("molecule_chembl_id")
        if first.get("pref_name"):
            label += f" ({first['pref_name']})"
        if phase is not None:
            label += f", max clinical phase {phase}"
        hits.append(f"ChEMBL {label}")

    unreachable = [
        name for name, block in (("PubChem", pubchem), ("ChEMBL", chembl))
        if not block.get("found") and block.get("note")
    ]

    if hits:
        verdict = "Known compound: " + "; ".join(hits) + "."
    elif unreachable:
        verdict = (
            "No match found, but " + " and ".join(unreachable) +
            " could not be queried, so this is not evidence of novelty."
        )
    else:
        verdict = "Not found in PubChem or ChEMBL by exact structure."

    return {
        "inchikey": inchikey,
        "query_smiles": smiles,
        "pubchem": pubchem,
        "chembl": chembl,
        "known": bool(hits),
        "verdict": verdict,
    }


def register_tools(mcp: Any) -> None:
    """Register the cross-database structure check."""

    @mcp.tool()
    def check_compound_is_known(
        smiles: Optional[str] = None,
        inchikey: Optional[str] = None,
    ) -> dict[str, Any]:
        """Check whether a structure is already a known compound.

        Looks the exact structure up in PubChem and ChEMBL by InChIKey, which
        is the identifier the two databases share. Use this before calling a
        generated molecule novel, and on any hit worth pursuing: a design that
        turns out to be an approved drug or a documented ChEMBL compound has
        published data to read instead of experiments to run.

        Args:
            smiles: Structure to check; its InChIKey is computed with RDKit.
            inchikey: Use this key directly instead of computing one.

        Returns:
            ``{inchikey, pubchem, chembl, known, verdict}``. A database that
            could not be reached says so rather than being read as a miss.
        """

        return _check_known(smiles=smiles, inchikey=inchikey)
