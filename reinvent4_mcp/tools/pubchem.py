# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PubChem compound lookup: resolve a molecule NAME to a real SMILES.

The agent must never recall a SMILES from memory (it hallucinates invalid or
wrong structures - e.g. bad JQ1 variants that RDKit cannot parse or 3D-embed).
This tool looks up a named compound on PubChem and returns its canonical
SMILES so downstream cofolding/scoring uses a real structure.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
_TIMEOUT = 20
# Property names in the current PubChem API: SMILES is the isomeric (stereo)
# form; ConnectivitySMILES is the flat/canonical form. InChIKey is what joins
# a hit here to one in ChEMBL -- a name matches neither reliably nor
# exclusively, and SMILES strings for one structure differ between databases.
_PROPS = (
    "SMILES,ConnectivitySMILES,MolecularFormula,MolecularWeight,"
    "IUPACName,Title,InChIKey"
)

# Namespaces this module looks a compound up by. PubChem takes the value in
# the URL path, which is fine for the first three; a SMILES carries "#", "/"
# and "+", so that one goes in a POST body instead of being quoted into a URL.
_NAMESPACES = frozenset({"name", "cid", "inchikey", "smiles"})

# How many sibling CIDs to name when a query matches more than one compound.
_MAX_OTHER_CIDS = 10

# PubChem throttles rather than failing: 503 PUGREST.ServerBusy and 429, both
# of which clear within seconds. One attempt turns that into a dead end -- a
# live run stalled on a 503 and asked the operator to paste a SMILES by hand,
# which is the one thing this module exists to prevent.
_RETRY_STATUS = frozenset({429, 503})
# Two retries: enough for a busy window, bounded so an interactive turn is not
# held for a minute by an endpoint that is actually down.
_RETRY_DELAYS = (1.0, 3.0)
# PubChem's own Retry-After is honoured, but not unboundedly.
_MAX_RETRY_WAIT = 5.0


def _retry_wait(response: Optional[requests.Response], fallback: float) -> float:
    """How long to wait, preferring what the server asked for."""

    header = response.headers.get("Retry-After") if response is not None else None
    try:
        asked = float(header) if header else fallback
    except (TypeError, ValueError):
        asked = fallback
    return min(max(asked, 0.0), _MAX_RETRY_WAIT)


def _request_properties(value: str, namespace: str = "name") -> requests.Response:
    """Fetch the property table for one identifier, riding out throttling.

    Raises the last transport error if every attempt failed, so the caller's
    handler still reports an unreachable PubChem honestly.
    """

    if namespace == "smiles":
        # POST, because a SMILES in a URL path is a quoting problem PubChem
        # itself tells callers to avoid.
        url = f"{_BASE}/compound/smiles/property/{_PROPS}/JSON"
        send = lambda: requests.post(  # noqa: E731
            url, data={"smiles": value}, timeout=_TIMEOUT
        )
    else:
        url = (
            f"{_BASE}/compound/{namespace}/{requests.utils.quote(str(value))}"
            f"/property/{_PROPS}/JSON"
        )
        send = lambda: requests.get(url, timeout=_TIMEOUT)  # noqa: E731

    last_error: Optional[requests.RequestException] = None
    last_response: Optional[requests.Response] = None

    for attempt, delay in enumerate((*_RETRY_DELAYS, None)):
        try:
            response = send()
        except requests.RequestException as exc:
            last_error, response = exc, None
        else:
            if response.status_code not in _RETRY_STATUS:
                return response
            last_error, last_response = None, response

        if delay is None:
            break
        logger.info(
            "PubChem is throttling %r (attempt %d); retrying in %.0fs",
            value, attempt + 1, _retry_wait(last_response, delay),
        )
        time.sleep(_retry_wait(last_response, delay))

    if last_response is not None:
        return last_response
    raise last_error  # type: ignore[misc]


def _lookup(value: str, namespace: str) -> dict[str, Any]:
    """One PubChem property lookup, whatever the identifier is."""

    value = str(value or "").strip()
    if not value:
        return {"error": f"Empty {namespace}. Provide a value to look up."}
    if namespace not in _NAMESPACES:
        return {
            "error": (
                f"Unknown PubChem namespace {namespace!r}; "
                f"expected one of {', '.join(sorted(_NAMESPACES))}."
            )
        }

    try:
        resp = _request_properties(value, namespace)
        if resp.status_code == 404:
            return {"error": f"No PubChem compound found for {namespace} {value!r}."}
        resp.raise_for_status()
        props = (resp.json().get("PropertyTable") or {}).get("Properties") or []
        if not props:
            return {"error": f"No properties returned by PubChem for {value!r}."}

        p = props[0]
        smiles = p.get("SMILES") or p.get("ConnectivitySMILES")
        if not smiles:
            return {
                "error": (
                    f"PubChem returned no SMILES for {value!r} (CID {p.get('CID')})."
                )
            }

        result = {
            "query": value,
            "namespace": namespace,
            "cid": p.get("CID"),
            "smiles": smiles,
            "connectivity_smiles": p.get("ConnectivitySMILES"),
            "molecular_formula": p.get("MolecularFormula"),
            "molecular_weight": p.get("MolecularWeight"),
            "iupac_name": p.get("IUPACName"),
            "title": p.get("Title"),
            "inchikey": p.get("InChIKey"),
        }

        # A name can match many compounds -- salts, stereoisomers, mixtures.
        # The first is still returned, because one structure has to be chosen
        # to cofold, but silently returning it as "the" compound is how a
        # racemate gets scored in place of the enantiomer that was asked for.
        if len(props) > 1:
            result["matches"] = len(props)
            result["other_cids"] = [
                q.get("CID") for q in props[1:_MAX_OTHER_CIDS + 1]
            ]
            result["ambiguous"] = (
                f"{len(props)} PubChem compounds match {value!r}; "
                f"CID {p.get('CID')} was taken. Check the others and look one "
                "up by cid= if this is not the intended structure."
            )
        return result
    except requests.RequestException as exc:
        logger.warning("PubChem request failed: %s", exc)
        return {
            "error": (
                f"Could not reach PubChem ({exc}). The SMILES was NOT retrieved; "
                "do not fabricate one - ask the user to paste the SMILES."
            )
        }


def fetch_compound(name: str) -> dict[str, Any]:
    """Resolve a compound name (or synonym) to a real SMILES via PubChem.

    Args:
        name: Common name, synonym, or identifier (e.g. "JQ1", "aspirin",
            "acetazolamide", "imatinib").

    Returns:
        Dict with ``cid``, ``smiles`` (isomeric, use this for cofolding/scoring),
        ``connectivity_smiles`` (flat), ``molecular_formula``, ``molecular_weight``,
        ``iupac_name``, ``title`` and ``inchikey``. When the name matches more
        than one compound it also carries ``matches``, ``other_cids`` and
        ``ambiguous``. On failure returns ``{"error": ...}`` and does NOT
        invent a structure.
    """

    return _lookup(name, "name")


def fetch_compound_by_cid(cid: int | str) -> dict[str, Any]:
    """Fetch one compound by PubChem CID — the unambiguous way to ask.

    This is what resolves the ambiguity :func:`fetch_compound` reports: a CID
    names exactly one structure, where a name may name a dozen.
    """

    return _lookup(cid, "cid")


def fetch_compound_by_smiles(smiles: str) -> dict[str, Any]:
    """Look a structure up by SMILES, to find out whether PubChem knows it.

    PubChem matches the structure, not the string, so a generated SMILES finds
    the registered compound it is the same molecule as.
    """

    return _lookup(smiles, "smiles")


def fetch_compound_by_inchikey(inchikey: str) -> dict[str, Any]:
    """Look a structure up by InChIKey — the identifier shared across databases."""

    return _lookup(inchikey, "inchikey")
