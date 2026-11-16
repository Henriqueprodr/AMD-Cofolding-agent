# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Patent search: freedom-to-operate evidence, not registry novelty.

The difference matters. A PubChem lookup answers "has this exact structure
been published as a compound?" A patent search answers "has this structure
been claimed as a composition of matter?" Those two questions have different
answers for a large fraction of drug candidates: a compound that is in
PubChem as a reference standard may or may not be in a granted claim; a
structure that is not in ChEMBL may appear in a pharmaceutical patent that
was filed but not yet published as a journal article.

Two operations:

``search_patents_by_structure``
    Exact-structure and similarity search in SureChEMBL, the public
    patent-chemistry database. Returns granted and pending claims.

``search_patents_by_target``
    Keyword + classification search via EPO Open Patent Services or
    SureChEMBL target text search. Returns patents mentioning the gene/
    protein in the claims.

Neither result is legal advice. A hit is a flag for a patent attorney to
evaluate, not evidence of infringement.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

_TIMEOUT = 15

# ---------------------------------------------------------------------------
# SureChEMBL — structure-based patent search
# ---------------------------------------------------------------------------

_SURECHEMBL_BASE = "https://www.surechembl.org/api"
#: One route for both exact and similarity search: a threshold below 1.0 is
#: what makes it the latter.
_SURECHEMBL_SEARCH = f"{_SURECHEMBL_BASE}/chemical/search"
# SureChEMBL migrated to EMBL-EBI hosting; the original domain now returns 404
# on the API path. Try the EBI URL as a fallback when the primary 404s.
_SURECHEMBL_SEARCH_EBI = "https://www.ebi.ac.uk/surechembl/api/chemical/search"

# Neither route above exists any more: surechembl.org answers them with a bare
# {"detail":"Not Found"} and the EBI host serves its single-page app for every
# path, so the JSON parse failed and the tool reported the service as down on
# every call. The rebuilt API splits the question in two — match the structure
# to compound ids, then ask which documents claim those ids.
_SURECHEMBL_STRUCTURE = f"{_SURECHEMBL_BASE}/search/structure"
_SURECHEMBL_DOCUMENTS = f"{_SURECHEMBL_BASE}/search/documents_for_structures"

#: A structure search runs against the whole compound index, so it is slower
#: than the metadata lookups the module's default timeout was chosen for.
_STRUCTURE_TIMEOUT = 90

# ---------------------------------------------------------------------------
# EPO Open Patent Services — text/classification search (no auth required
# for the free-tier /published-data/search endpoint)
# ---------------------------------------------------------------------------

_EPO_OPS = "https://ops.epo.org/3.2/rest-services/published-data/search"


def _by_structure(
    smiles: str,
    threshold: float = 1.0,
    limit: int = 20,
) -> dict[str, Any]:
    """SureChEMBL: patents containing this structure.

    At ``threshold=1.0``, exact-structure search. At ``threshold<1.0``,
    similarity search (Tanimoto). Both search the composition-of-matter
    claims across granted and pending patents.

    Args:
        smiles: The structure to search for.
        threshold: 1.0 = exact. 0.85 is a useful similarity cutoff for
            analogue freedom-to-operate.
        limit: Maximum patents to return.

    Returns:
        ``{patents, total, smiles, threshold}``. Each entry has
        ``surechembl_id``, ``patent_numbers``, ``title``, ``filing_date``,
        ``publication_date``, ``assignee``, ``status``.
    """

    # Step one: the structure to the compound ids that match it. A threshold
    # of 1.0 is an exact match, which the API calls a different search type
    # rather than a similarity search with the cutoff wound all the way up.
    request: dict[str, Any] = {
        "struct": smiles,
        "struct_search_type": "identical" if threshold >= 1.0 else "similarity",
        "page": 1,
        "items_per_page": limit,
    }
    if threshold < 1.0:
        request["threshold"] = threshold

    try:
        resp = requests.post(_SURECHEMBL_STRUCTURE, json=request, timeout=_STRUCTURE_TIMEOUT)
        resp.raise_for_status()
        matched = resp.json()
    except requests.RequestException as exc:
        return _surechembl_unavailable(smiles, threshold, str(exc))
    except ValueError:
        return _surechembl_unavailable(smiles, threshold, "non-JSON response")

    chemical_ids = [str(i) for i in (matched or {}).get("chemical_ids", [])]
    if not chemical_ids:
        # A clean miss, which is a real answer rather than an outage.
        return {
            "smiles": smiles,
            "threshold": threshold,
            "total": 0,
            "patents": [],
            "source": "SureChEMBL",
            "note": (
                "No SureChEMBL compound matches this structure, so nothing "
                "claims it under this search. That is not freedom to operate: "
                "a generic Markush claim can cover a structure the index has "
                "never seen as a discrete compound."
            ),
        }

    # Step two: the documents claiming those compounds. One id per request:
    # the endpoint reads several ids as "documents claiming all of these at
    # once", which for the salt and tautomer records an exact-match search
    # returns is nothing at all. What the caller wants is the union.
    patents: list[dict[str, Any]] = []
    seen: set[Any] = set()
    reached: str = ""
    for chemical_id in chemical_ids:
        if len(patents) >= limit:
            break
        try:
            resp = requests.get(
                _SURECHEMBL_DOCUMENTS,
                params={"chemicalIds": chemical_id, "page": 1, "itemsPerPage": limit},
                timeout=_STRUCTURE_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            reached = reached or str(exc)
            continue
        except ValueError:
            reached = reached or "non-JSON response"
            continue
        for hit in _parse_surechembl_hits(data):
            if hit["surechembl_id"] in seen:
                continue
            seen.add(hit["surechembl_id"])
            patents.append(hit)

    if not patents and reached:
        # Every compound's lookup failed, so this is an outage rather than a
        # structure that nothing claims.
        return _surechembl_unavailable(smiles, threshold, reached)
    return {
        "smiles": smiles,
        "threshold": threshold,
        "total": len(patents),
        "patents": patents[:limit],
        "source": "SureChEMBL",
        "note": (
            "These are composition-of-matter hits. A result here does not "
            "mean infringement; it means the structure or a close analogue "
            "has been claimed. Have a patent attorney evaluate the scope."
        ),
    }


def _surechembl_unavailable(smiles: str, threshold: float, reason: str) -> dict[str, Any]:
    return {
        "smiles": smiles,
        "threshold": threshold,
        "total": 0,
        "patents": [],
        "source": "SureChEMBL",
        "error": (
            f"SureChEMBL could not be reached ({reason}). "
            "Patent status is unknown, not clean — do not treat this as "
            "evidence of freedom to operate."
        ),
    }


def _parse_surechembl_hits(data: Any) -> list[dict[str, Any]]:
    """Normalise SureChEMBL API responses to a consistent shape."""
    hits: list[dict[str, Any]] = []
    if isinstance(data, dict):
        # "docs" is what the document search answers with; the others are the
        # shapes the older compound route used.
        rows = data.get("docs", data.get("results", data.get("hits", data.get("compounds", []))))
    elif isinstance(data, list):
        rows = data
    else:
        return hits

    for row in rows:
        if not isinstance(row, dict):
            continue
        if "ucid" in row:
            # A document row: the patent is named by its UCID, the applicants
            # by "pa", the title by "ttl_en" and the publication date by "pd"
            # as a plain YYYYMMDD integer.
            published = str(row.get("pd") or "")
            titles = row.get("ttl_en") or []
            applicants = row.get("pa") or []
            hits.append({
                "surechembl_id": row.get("ucid"),
                "patent_numbers": [row["ucid"]],
                "title": titles[0] if isinstance(titles, list) and titles else titles or None,
                "filing_date": None,
                "publication_date": (
                    f"{published[:4]}-{published[4:6]}-{published[6:8]}" if len(published) == 8 else None
                ),
                # Repeated spellings of the same company are common here.
                "assignee": sorted({a for a in applicants if a}) if isinstance(applicants, list) else applicants,
                "status": None,
                "similarity": row.get("score"),
            })
            continue
        # Patent numbers may be a list or a comma-separated string.
        pn = row.get("patent_numbers") or row.get("patents") or []
        if isinstance(pn, str):
            pn = [p.strip() for p in pn.split(",") if p.strip()]
        hits.append({
            "surechembl_id": row.get("surechembl_id") or row.get("id"),
            "patent_numbers": pn,
            "title": row.get("title") or row.get("patent_title"),
            "filing_date": row.get("filing_date"),
            "publication_date": row.get("publication_date"),
            "assignee": row.get("assignee") or row.get("applicant"),
            "status": row.get("status") or row.get("legal_status"),
            "similarity": row.get("similarity") or row.get("tanimoto"),
        })
    return hits


def _by_target(
    query: str,
    classification: Optional[str] = None,
    limit: int = 20,
) -> dict[str, Any]:
    """EPO OPS: patents mentioning a gene or target by name.

    Uses the EPO Open Patent Services text search over titles and abstracts.
    For a pharma freedom-to-operate, also check the claims with
    ``classification='A61K'`` (medicinal preparations) or
    ``'C07D'`` (heterocyclic compounds).

    Args:
        query: Gene name, protein name, or free text, e.g. "BRD4 bromodomain".
        classification: IPC/CPC class to restrict to, e.g. ``"A61K31"``
            (compositions containing organic compounds) or ``"C07D"``
            (heterocyclics).
        limit: Maximum patents to return (EPO allows up to 100).

    Returns:
        ``{patents, total, query}``. Each entry has ``patent_number``,
        ``title``, ``abstract``, ``filing_date``, ``applicant``.
    """

    # EPO OPS free-tier search — CQL query.
    q = f'txt="{query}"'
    if classification:
        q += f' AND cpc="{classification}"'

    try:
        resp = requests.get(
            _EPO_OPS,
            params={"q": q, "Range": f"1-{min(limit, 25)}"},
            headers={"Accept": "application/json"},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        return {
            "query": query,
            "total": 0,
            "patents": [],
            "source": "EPO OPS",
            "error": (
                f"EPO Open Patent Services could not be reached ({exc}). "
                "Target-based patent status is unknown."
            ),
        }
    except ValueError:
        return {
            "query": query,
            "total": 0,
            "patents": [],
            "source": "EPO OPS",
            "error": "EPO OPS returned non-JSON.",
        }

    patents = _parse_epo_ops(data)
    return {
        "query": query,
        "classification": classification,
        "total": len(patents),
        "patents": patents,
        "source": "EPO OPS",
        "note": (
            "Text search over titles and abstracts. The claims govern scope; "
            "a text match is a lead, not a determination of infringement."
        ),
    }


def _parse_epo_ops(data: Any) -> list[dict[str, Any]]:
    """Normalise EPO OPS publication-search JSON."""
    patents: list[dict[str, Any]] = []
    try:
        results = (
            data
            .get("ops:world-patent-data", {})
            .get("ops:biblio-search", {})
            .get("ops:search-result", {})
            .get("ops:publication-reference", [])
        )
        if isinstance(results, dict):
            results = [results]
        for ref in results:
            doc = ref.get("document-id", {})
            patents.append({
                "patent_number": doc.get("doc-number", {}).get("$"),
                "country": doc.get("country", {}).get("$"),
                "kind": doc.get("kind", {}).get("$"),
                "filing_date": doc.get("date", {}).get("$"),
            })
    except (AttributeError, TypeError, KeyError):
        pass
    return patents


def register_tools(mcp: Any) -> None:
    """Register the patent search tools."""

    @mcp.tool()
    def search_patents_by_structure(
        smiles: str,
        threshold: float = 1.0,
        limit: int = 20,
    ) -> dict[str, Any]:
        """SureChEMBL: patents claiming this structure.

        Use this before calling a generated molecule novel. An exact-structure
        hit (threshold=1.0) means the molecule has been claimed in a patent
        composition; a similarity hit (threshold ~0.85) flags close analogues
        that may read on the claims. Neither result is legal advice — a hit
        is a flag for a patent attorney.

        Args:
            smiles: The structure to search.
            threshold: 1.0 for exact, < 1.0 for similarity. 0.85 is a
                useful analogue cutoff.
            limit: Maximum patents to return.

        Returns:
            ``{patents, total}``. Check ``error`` first: an unreachable
            database is not evidence of freedom to operate.
        """

        return _by_structure(smiles=smiles, threshold=threshold, limit=limit)

    @mcp.tool()
    def search_patents_by_target(
        query: str,
        classification: Optional[str] = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """EPO OPS: patents mentioning a gene or target.

        Searches EPO titles and abstracts for a gene name or target name.
        Restrict to a patent class for relevance: ``'A61K'`` (medicinal
        preparations), ``'C07D'`` (heterocyclic compounds, most small-
        molecule pharmaceuticals), ``'A01N'`` (agrochemicals).

        Args:
            query: Gene/protein name or free text, e.g. ``'BRD4 bromodomain'``.
            classification: IPC/CPC class filter, e.g. ``'A61K31'``.
            limit: Maximum patents to return.

        Returns:
            ``{patents, total}``. A text match in the title/abstract is a
            lead, not evidence of infringement.
        """

        return _by_target(query=query, classification=classification, limit=limit)
