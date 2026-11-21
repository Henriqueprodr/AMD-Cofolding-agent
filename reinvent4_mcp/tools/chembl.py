# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ChEMBL bioactivity lookup: MEASURED potency, not predicted and not recalled.

Boltz2 affinity is a prediction on its own scale, and the agent's memory is not
a source at all - a recalled IC50 is indistinguishable from an invented one.
This tool is the only place a real IC50/Ki/Kd/EC50 enters a session.

It also gives the read-only Reviewer something to review against: without it,
a claim like "JQ1 is a potent BRD4 binder" can only be checked against the
Driver's own predictions.
"""

from __future__ import annotations

import logging
import re
import statistics
from collections.abc import Sequence
from typing import Any

import requests

logger = logging.getLogger(__name__)

_BASE = "https://www.ebi.ac.uk/chembl/api/data"
_TIMEOUT = 30

_DEFAULT_LIMIT = 20
_MAX_LIMIT = 50
_MAX_TARGET_CANDIDATES = 5
_DEFAULT_TYPES = ("IC50", "Ki", "Kd", "EC50")

_CHEMBL_ID_RE = re.compile(r"^CHEMBL\d+$", re.IGNORECASE)
# Same shape UniProt accessions take (see tools/uniprot.py).
_ACCESSION_RE = re.compile(
    r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$",
    re.IGNORECASE,
)

_MOLECULE_FIELDS = (
    "molecule_chembl_id,pref_name,max_phase,molecule_type,first_approval,"
    "molecule_structures"
)
_TARGET_FIELDS = (
    "target_chembl_id,pref_name,target_type,organism,target_components"
)
_ACTIVITY_FIELDS = (
    "molecule_chembl_id,molecule_pref_name,target_chembl_id,target_pref_name,"
    "target_organism,standard_type,standard_relation,standard_value,"
    "standard_units,pchembl_value,assay_chembl_id,assay_type,"
    "assay_description,document_chembl_id,bao_label"
)

# Ranked best-first: a measurement on a single protein is what a medicinal
# chemist means by "activity against this target"; complexes and PPIs are
# different experiments that happen to mention the same protein.
_TARGET_TYPE_RANK = {
    "SINGLE PROTEIN": 0,
    "PROTEIN COMPLEX": 1,
    "PROTEIN FAMILY": 2,
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _get(path: str, params: dict[str, Any]) -> dict[str, Any]:
    """One ChEMBL GET. Raises ``requests.RequestException`` on failure."""
    resp = requests.get(f"{_BASE}/{path}", params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json() or {}


def _num(value: Any) -> float | None:
    """ChEMBL returns every number as a string ('pchembl_value': '10.22')."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _looks_like_chembl_id(text: str) -> bool:
    return bool(_CHEMBL_ID_RE.match(text.strip()))


def _looks_like_accession(text: str) -> bool:
    return bool(_ACCESSION_RE.match(text.strip()))


def _clean_molecule(raw: dict[str, Any]) -> dict[str, Any]:
    """Flatten a ChEMBL molecule, dropping the molfile.

    ``only=`` does not prune nested objects, so ``molecule_structures``
    arrives with a ~4 KB molfile attached that would otherwise be dumped
    into the model's context on every call.
    """
    structures = raw.get("molecule_structures") or {}
    return {
        "chembl_id": raw.get("molecule_chembl_id"),
        "pref_name": raw.get("pref_name"),
        "canonical_smiles": structures.get("canonical_smiles"),
        "max_phase": _num(raw.get("max_phase")),
        "molecule_type": raw.get("molecule_type"),
        "first_approval": raw.get("first_approval"),
    }


def _clean_target(raw: dict[str, Any]) -> dict[str, Any]:
    components = raw.get("target_components") or []
    return {
        "chembl_id": raw.get("target_chembl_id"),
        "pref_name": raw.get("pref_name"),
        "target_type": raw.get("target_type"),
        "organism": raw.get("organism"),
        "uniprot_accessions": [
            c.get("accession") for c in components if c.get("accession")
        ],
    }


def _resolve_molecule(name: str) -> tuple[dict[str, Any] | None, str | None]:
    """Resolve a compound name or ChEMBL ID. Returns ``(molecule, note)``."""
    name = name.strip()

    if _looks_like_chembl_id(name):
        data = _get(f"molecule/{name.upper()}.json", {})
        if data.get("molecule_chembl_id"):
            return _clean_molecule(data), None
        return None, f"No ChEMBL molecule with id {name!r}."

    for field in (
        "pref_name__iexact",
        "molecule_synonyms__molecule_synonym__iexact",
    ):
        data = _get(
            "molecule.json",
            {field: name, "limit": 1, "only": _MOLECULE_FIELDS},
        )
        hits = data.get("molecules") or []
        if hits:
            return _clean_molecule(hits[0]), None

    # Last resort: ChEMBL's own fuzzy ranking, which can be wrong.
    data = _get(
        "molecule/search.json",
        {"q": name, "limit": 1, "only": _MOLECULE_FIELDS},
    )
    hits = data.get("molecules") or []
    if hits:
        mol = _clean_molecule(hits[0])
        return mol, (
            f"{name!r} was resolved by fuzzy ChEMBL text search to "
            f"{mol.get('chembl_id')} ({mol.get('pref_name')}); confirm the "
            "identity before relying on it."
        )
    return None, f"No ChEMBL molecule found for {name!r}."


def _rank_targets(
    targets: list[dict[str, Any]], organism: str | None
) -> list[dict[str, Any]]:
    """Best-first: single proteins, then the requested/human organism."""

    def key(t: dict[str, Any]) -> tuple[int, int, str]:
        type_rank = _TARGET_TYPE_RANK.get((t.get("target_type") or "").upper(), 9)
        want = (organism or "Homo sapiens").lower()
        organism_rank = 0 if (t.get("organism") or "").lower() == want else 1
        return (type_rank, organism_rank, t.get("chembl_id") or "")

    return sorted(targets, key=key)


def _resolve_target(
    query: str, organism: str | None
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str | None]:
    """Resolve a target. Returns ``(best, other_candidates, note)``."""
    query = query.strip()

    if _looks_like_chembl_id(query):
        data = _get(f"target/{query.upper()}.json", {})
        if data.get("target_chembl_id"):
            return _clean_target(data), [], None
        return None, [], f"No ChEMBL target with id {query!r}."

    note: str | None = None
    if _looks_like_accession(query):
        data = _get(
            "target.json",
            {
                "target_components__accession": query.upper(),
                "limit": 10,
                "only": _TARGET_FIELDS,
            },
        )
    else:
        data = _get(
            "target/search.json",
            {"q": query, "limit": 10, "only": _TARGET_FIELDS},
        )
        note = (
            "Free-text target search is ambiguous (it matches other organisms "
            "and protein complexes). Pass the UniProt accession from "
            "fetch_target_info as `target` for an exact match."
        )

    hits = [_clean_target(t) for t in (data.get("targets") or [])]
    if not hits:
        return None, [], f"No ChEMBL target found for {query!r}."

    ranked = _rank_targets(hits, organism)
    return ranked[0], ranked[1:_MAX_TARGET_CANDIDATES + 1], note


def _activities(
    filters: dict[str, str],
    limit: int,
    types: Sequence[str],
) -> tuple[list[dict[str, Any]], int]:
    """Top activities by pChEMBL. Returns ``(rows, total_available)``."""
    params: dict[str, Any] = {
        **filters,
        "standard_type__in": ",".join(types),
        "pchembl_value__isnull": "false",
        # Exact values only: drops censored rows like ">10000".
        "standard_relation": "=",
        "order_by": "-pchembl_value",
        "limit": limit,
        "only": _ACTIVITY_FIELDS,
    }
    data = _get("activity.json", params)
    rows = [
        {
            "molecule_chembl_id": a.get("molecule_chembl_id"),
            "molecule_pref_name": a.get("molecule_pref_name"),
            "target_chembl_id": a.get("target_chembl_id"),
            "target_pref_name": a.get("target_pref_name"),
            "target_organism": a.get("target_organism"),
            "standard_type": a.get("standard_type"),
            "standard_relation": a.get("standard_relation"),
            "standard_value": _num(a.get("standard_value")),
            "standard_units": a.get("standard_units"),
            "pchembl_value": _num(a.get("pchembl_value")),
            "assay_type": a.get("assay_type"),
            "assay_chembl_id": a.get("assay_chembl_id"),
            "assay_description": a.get("assay_description"),
            "document_chembl_id": a.get("document_chembl_id"),
            "bao_label": a.get("bao_label"),
        }
        for a in (data.get("activities") or [])
    ]
    total = int((data.get("page_meta") or {}).get("total_count") or len(rows))
    return rows, total


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pchembls = [r["pchembl_value"] for r in rows if r.get("pchembl_value")]
    type_counts: dict[str, int] = {}
    for row in rows:
        key = row.get("standard_type") or "?"
        type_counts[key] = type_counts.get(key, 0) + 1

    return {
        "best_pchembl": max(pchembls) if pchembls else None,
        "median_pchembl": (
            round(statistics.median(pchembls), 2) if pchembls else None
        ),
        "type_counts": type_counts,
        "n_distinct_targets": len(
            {r.get("target_chembl_id") for r in rows if r.get("target_chembl_id")}
        ),
        "n_distinct_compounds": len(
            {
                r.get("molecule_chembl_id")
                for r in rows
                if r.get("molecule_chembl_id")
            }
        ),
        "best_row": rows[0] if rows else None,
    }


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------


def fetch_bioactivity(
    compound: str | None = None,
    target: str | None = None,
    activity_types: list[str] | None = None,
    organism: str | None = None,
    limit: int = _DEFAULT_LIMIT,
) -> dict[str, Any]:
    """Look up measured bioactivity in ChEMBL.

    Args:
        compound: Compound name, synonym, or ``CHEMBLnnn`` id.
        target: UniProt accession (best), gene/protein name, or ``CHEMBLnnn``.
        activity_types: Defaults to IC50/Ki/Kd/EC50.
        organism: Preferred organism when a target name is ambiguous.
        limit: Rows to return, capped at 50 and ordered by pChEMBL.

    Returns:
        Dict with ``molecule``, ``target``, ``target_candidates``,
        ``activities``, ``n_activities``, ``total_available``, ``summary``
        and ``note``. On failure returns ``{"error": ...}`` and does NOT
        invent a value.
    """
    compound = (compound or "").strip()
    target = (target or "").strip()
    if not compound and not target:
        return {
            "error": (
                "Provide a compound (name or ChEMBL id), a target (UniProt "
                "accession, name, or ChEMBL id), or both."
            )
        }

    types = [t.strip() for t in (activity_types or _DEFAULT_TYPES) if t.strip()]
    if not types:
        types = list(_DEFAULT_TYPES)
    limit = max(1, min(int(limit or _DEFAULT_LIMIT), _MAX_LIMIT))

    notes: list[str] = []
    result: dict[str, Any] = {
        "query": {
            "compound": compound or None,
            "target": target or None,
            "activity_types": types,
            "limit": limit,
        },
        "molecule": None,
        "target": None,
    }

    try:
        filters: dict[str, str] = {}

        if compound:
            molecule, note = _resolve_molecule(compound)
            if note:
                notes.append(note)
            if molecule is None:
                result["note"] = " ".join(notes)
                result["activities"] = []
                result["n_activities"] = 0
                return result
            result["molecule"] = molecule
            filters["molecule_chembl_id"] = molecule["chembl_id"]

        if target:
            best, candidates, note = _resolve_target(target, organism)
            if note:
                notes.append(note)
            if best is None:
                result["note"] = " ".join(notes)
                result["activities"] = []
                result["n_activities"] = 0
                return result
            result["target"] = best
            if candidates:
                result["target_candidates"] = candidates
            filters["target_chembl_id"] = best["chembl_id"]

        rows, total = _activities(filters, limit, types)

    except requests.RequestException as exc:
        logger.warning("ChEMBL request failed: %s", exc)
        return {
            "error": (
                f"Could not reach ChEMBL ({exc}). Measured activity data was "
                "NOT retrieved; do not quote an IC50/Ki/Kd from memory or "
                "from a web page - say the lookup failed, or ask the user "
                "for the value."
            )
        }

    result["activities"] = rows
    result["n_activities"] = len(rows)
    result["total_available"] = total
    result["summary"] = _summarize(rows)

    if not rows:
        notes.append(
            "ChEMBL holds no pChEMBL-valued "
            f"{'/'.join(types)} measurement matching this query. That means "
            "no curated measurement exists - it does NOT mean the compound "
            "is inactive."
        )
    elif total > len(rows):
        notes.append(
            f"Showing the {len(rows)} highest-pChEMBL rows of {total}. These "
            "are the optimistic end of the data; quote the range and the "
            "number of assays rather than the single best value."
        )

    if notes:
        result["note"] = " ".join(notes)
    return result
