# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Biology beyond ChEMBL: expression, pathways, orthology, genetic validation.

ChEMBL answers "what has been made and what did it do in a biochemical assay."
These tools answer the target-validation questions that fall outside it:

- Is this target expressed where the disease is? (Human Protein Atlas)
- What pathways does it sit in? (Reactome, KEGG via UniProt)
- Does evidence from genetics support it? (OpenTargets)
- Is there an animal-model ortholog I can use? (UniProt orthology)

Each tool calls a public REST API without authentication. They are read-only
and do not start any job on the cluster.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)

_TIMEOUT = 10

# ---------------------------------------------------------------------------
# OpenTargets
# ---------------------------------------------------------------------------

_OT_API = "https://api.platform.opentargets.org/api/v4/graphql"

_OT_EVIDENCE_QUERY = """
query Evidence($target: String!, $n: Int!) {
  target(ensemblId: $target) {
    id
    approvedSymbol
    approvedName
    associatedDiseases(page: {index: 0, size: $n}) {
      count
      rows {
        disease { id name }
        score
        datatypeScores { id score }
      }
    }
  }
}
"""
# The `evidences()` field was nested under `target` in an older v4 schema
# draft; in the released platform API it is a top-level query. The
# `orderByScore: "score"` argument also changed its accepted type. Removing
# both makes the query valid across v4 versions while keeping the disease
# association data the tool actually uses.
#
# `associatedDiseases` takes no bare `size`: the released schema paginates it
# through a `Pagination` input object, and a `size` argument is rejected at
# parse time, so every call came back 400 rather than with partial data.


def _genetic_evidence(
    ensembl_id: str,
    n_diseases: int = 10,
) -> dict[str, Any]:
    """OpenTargets: genetic support for a target across diseases.

    Args:
        ensembl_id: Ensembl gene id (e.g. ``ENSG00000141510`` for TP53).
            Obtain from ``fetch_target_info`` → ``ensembl_id`` field.
        n_diseases: Top associated diseases to return.

    Returns:
        ``{target_symbol, target_name, top_diseases, genetic_evidence,
        total_diseases}``. Each entry in ``genetic_evidence`` carries a
        ``datasourceId`` (gwas_catalog, gene_burden, etc.) and score.
    """

    try:
        resp = requests.post(
            _OT_API,
            json={"query": _OT_EVIDENCE_QUERY, "variables": {"target": ensembl_id, "n": n_diseases}},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json().get("data", {}).get("target")
    except requests.RequestException as exc:
        return {"error": f"OpenTargets could not be reached: {exc}", "ensembl_id": ensembl_id}

    if not data:
        return {
            "error": f"No OpenTargets data for {ensembl_id!r}. Check the Ensembl id.",
            "ensembl_id": ensembl_id,
        }

    diseases = data.get("associatedDiseases", {})

    return {
        "ensembl_id": ensembl_id,
        "target_symbol": data.get("approvedSymbol"),
        "target_name": data.get("approvedName"),
        "total_diseases": diseases.get("count", 0),
        "top_diseases": [
            {
                "disease_id": r["disease"]["id"],
                "disease_name": r["disease"]["name"],
                "score": round(r.get("score", 0), 4),
                "genetic_score": next(
                    (d["score"] for d in r.get("datatypeScores", []) if d["id"] == "genetic_association"),
                    None,
                ),
            }
            for r in diseases.get("rows", [])
        ],
        # The evidences/geneticsPortal subquery was dropped because that field
        # moved to a top-level query in the released OpenTargets v4 API.
        # disease-level genetic_score (above) still surfaces the per-disease
        # genetic association score from datatypeScores.
        "genetic_evidence_rows": [],
        "total_genetic_evidence_rows": 0,
    }


# ---------------------------------------------------------------------------
# Human Protein Atlas — tissue expression
# ---------------------------------------------------------------------------

#: HPA keys its pages by its own gene id, so an accession is resolved to one
#: by following the redirect this search issues.
_HPA_SEARCH = "https://www.proteinatlas.org/search/uniprot:{acc}"
_HPA_TISSUE = "https://www.proteinatlas.org/{gene}/tissue.json"
_HPA_SINGLE_CELL = "https://www.proteinatlas.org/{gene}/single+cell+type.json"

#: The per-page ``/{gene}/tissue.json`` route above is gone from the current
#: HPA deployment — every form of it answers 404 with the site's HTML shell,
#: including the "{SYMBOL}-{ACCESSION}" id the redirect fallback builds, which
#: HPA has never used as a page id. What HPA does serve is a per-gene summary
#: at ``/{ENSG}.json`` and a column-selected query API, both keyed on the
#: Ensembl gene id. These are the routes the tool uses now.
_HPA_QUERY = "https://www.proteinatlas.org/api/search_download.php"
_HPA_GENE_JSON = "https://www.proteinatlas.org/{ensg}.json"

#: HPA's consensus tissue columns, as its own search UI names them. Requested
#: explicitly because the API has no "give me every tissue" group code.
_HPA_TISSUE_COLUMNS = (
    "adipose_tissue", "adrenal_gland", "amygdala", "appendix", "basal_ganglia",
    "blood_vessel", "bone_marrow", "breast", "cerebellum", "cerebral_cortex",
    "cervix", "choroid_plexus", "colon", "duodenum", "endometrium_1",
    "epididymis", "esophagus", "fallopian_tube", "gallbladder", "heart_muscle",
    "hippocampal_formation", "hypothalamus", "kidney", "liver", "lung",
    "lymph_node", "midbrain", "ovary", "pancreas", "parathyroid_gland",
    "pituitary_gland", "placenta", "prostate", "rectum", "retina",
    "salivary_gland", "seminal_vesicle", "skeletal_muscle", "skin_1",
    "small_intestine", "smooth_muscle", "spinal_cord", "spleen", "stomach_1",
    "testis", "thymus", "thyroid_gland", "tongue", "tonsil", "urinary_bladder",
    "vagina",
)

#: nTPM cut-offs HPA itself uses to turn a transcript count into a word, so
#: the rows keep reading the way the tool's callers expect.
def _hpa_level(ntpm: float) -> str:
    if ntpm >= 50:
        return "High"
    if ntpm >= 10:
        return "Medium"
    if ntpm >= 1:
        return "Low"
    return "Not detected"


def _hpa_row(record: dict[str, Any]) -> dict[str, Any] | None:
    """One ``search_download.php`` record, or None when the gene is unknown."""
    return record if record.get("Ensembl") else None


def _hpa_lookup(search: str, columns: str) -> dict[str, Any] | None:
    """One HPA query-API record for *search*, or None when nothing matched."""
    try:
        resp = requests.get(
            _HPA_QUERY,
            params={"search": search, "format": "json", "compress": "no", "columns": columns},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        records = resp.json()
    except (requests.RequestException, ValueError):
        return None
    if not isinstance(records, list) or not records:
        return None
    return _hpa_row(records[0])


def _expression(
    uniprot_accession: str,
    context: str = "tissue",
) -> dict[str, Any]:
    """Human Protein Atlas: where a target is expressed.

    Args:
        uniprot_accession: e.g. ``P00533`` for EGFR.
        context: ``tissue`` (bulk RNA, RNA consensus; standard for target
            selection), ``single_cell`` (single-cell RNA).

    Returns:
        ``{gene, protein_class, tissue_rows, context}``. ``tissue_rows``
        is a list of ``{tissue, cell_type, level, reliability}``.
    """

    if context not in ("tissue", "single_cell"):
        return {"error": f"Unknown context {context!r}; use 'tissue' or 'single_cell'."}

    # Resolve the Ensembl gene id, which is what every HPA route is keyed on.
    # Searching the accession is enough; the UniProt cross-reference is the
    # fallback for an accession HPA's own index does not carry.
    tissue_columns = ",".join(f"t_RNA_{name}" for name in _HPA_TISSUE_COLUMNS)
    found = _hpa_lookup(uniprot_accession, "g,eg,up,rnats,rnatd," + tissue_columns)
    ensg = found.get("Ensembl") if found else None

    if not ensg:
        entry = _uniprot_entry(uniprot_accession)
        if "_error" not in entry:
            ensg = next(
                (
                    x["id"].split(".")[0]
                    for x in entry.get("uniProtKBCrossReferences", [])
                    if x.get("database") == "Ensembl" and x.get("id", "").startswith("ENSG")
                ),
                None,
            )

    if not ensg:
        return {"error": f"No HPA entry found for {uniprot_accession!r}."}

    # The per-gene summary carries the specificity and distribution verdicts,
    # which answer "is it everywhere in healthy tissue" directly.
    summary: dict[str, Any] = {}
    try:
        resp = requests.get(_HPA_GENE_JSON.format(ensg=ensg), timeout=_TIMEOUT)
        resp.raise_for_status()
        loaded = resp.json()
        if isinstance(loaded, dict):
            summary = loaded
    except (requests.RequestException, ValueError):
        pass  # the per-tissue rows below stand on their own

    rows: list[dict[str, Any]] = []
    if context == "tissue":
        for name in _HPA_TISSUE_COLUMNS:
            raw = (found or {}).get(f"Tissue RNA - {name.replace('_', ' ')} [nTPM]")
            if raw in (None, ""):
                continue
            try:
                ntpm = float(raw)
            except (TypeError, ValueError):
                continue
            rows.append({
                "tissue": name.replace("_", " "),
                "cell_type": "bulk RNA (consensus)",
                "level": _hpa_level(ntpm),
                "ntpm": ntpm,
                "reliability": summary.get("Reliability (IH)"),
            })
        rows.sort(key=lambda row: row["ntpm"], reverse=True)

    prefix = "RNA single cell type" if context == "single_cell" else "RNA tissue"

    return {
        "uniprot_accession": uniprot_accession,
        "gene_id": ensg,
        "gene": summary.get("Gene") or (found or {}).get("Gene"),
        "context": context,
        "specificity": summary.get(f"{prefix} specificity") or (found or {}).get("RNA tissue specificity"),
        "distribution": summary.get(f"{prefix} distribution") or (found or {}).get("RNA tissue distribution"),
        "enriched_in": summary.get(f"{prefix} specific nTPM") or summary.get(f"{prefix} specific nCPM"),
        "tissue_rows": rows,
        "note": (
            "Levels are banded from consensus nTPM the way HPA bands them: "
            "High >=50, Medium >=10, Low >=1, Not detected below that. "
            "'specificity' and 'distribution' are HPA's own verdicts; "
            "'Low tissue specificity' with 'Detected in many' means the "
            "target is broadly expressed in healthy tissue."
            + (
                " Per-cell-type rows are only served for the 'tissue' context; "
                "for single_cell the specificity and distribution verdicts apply."
                if context == "single_cell"
                else ""
            )
        ),
    }


# ---------------------------------------------------------------------------
# UniProt — pathways (Reactome) and orthology
# ---------------------------------------------------------------------------

_UNIPROT_API = "https://rest.uniprot.org/uniprotkb/{acc}.json"


def _uniprot_entry(accession: str) -> dict[str, Any]:
    """The UniProt JSON entry for an accession."""
    try:
        resp = requests.get(_UNIPROT_API.format(acc=accession), timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        return {"_error": str(exc)}


def _pathways(
    uniprot_accession: str,
) -> dict[str, Any]:
    """Reactome pathways a target participates in, via UniProt cross-references.

    Args:
        uniprot_accession: e.g. ``P00533``.

    Returns:
        ``{gene, reactome_pathways, kegg_pathways, go_biological_process}``.
    """

    entry = _uniprot_entry(uniprot_accession)
    if "_error" in entry:
        return {"error": f"UniProt could not be reached: {entry['_error']}"}

    gene = ""
    genes = entry.get("genes", [])
    if genes:
        gene = genes[0].get("geneName", {}).get("value", "")

    reactome: list[dict[str, str]] = []
    kegg: list[dict[str, str]] = []
    go_bp: list[dict[str, str]] = []

    for xref in entry.get("uniProtKBCrossReferences", []):
        db = xref.get("database", "")
        props = {p.get("key"): p.get("value") for p in xref.get("properties", [])}
        if db == "Reactome":
            reactome.append({
                "id": xref.get("id", ""),
                "name": props.get("PathwayName", ""),
            })
        elif db == "KEGG":
            kegg.append({"id": xref.get("id", ""), "name": props.get("PathwayName", "")})

    for ann in entry.get("keywords", []):
        if ann.get("category") == "Biological process":
            go_bp.append({"name": ann.get("name", ""), "id": ann.get("id", "")})

    return {
        "uniprot_accession": uniprot_accession,
        "gene": gene,
        "reactome_pathways": reactome,
        "kegg_pathways": kegg,
        "go_biological_process": go_bp,
        "total_reactome": len(reactome),
        "total_kegg": len(kegg),
    }


#: OMA lists a human protein's orthologs across every genome it has; for a
#: well-studied target that is a few hundred rows, so it is read once here.
_OMA_ORTHOLOGS = "https://omabrowser.org/api/protein/{acc}/orthologs/"

#: Reading every genome OMA has takes longer than a metadata lookup.
_OMA_TIMEOUT = 30


def _oma_orthologs(accession: str) -> list[dict[str, Any]]:
    """OMA's ortholog rows for *accession*; empty when OMA cannot be read."""
    try:
        resp = requests.get(_OMA_ORTHOLOGS.format(acc=accession), timeout=_OMA_TIMEOUT)
        resp.raise_for_status()
        loaded = resp.json()
    except (requests.RequestException, ValueError):
        return []
    rows = loaded if isinstance(loaded, list) else loaded.get("objects", [])
    return [row for row in rows if isinstance(row, dict)]


def _orthology(
    uniprot_accession: str,
    species: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Orthologs of a human target in other species, from UniProt.

    Drug discovery uses rat and mouse most often; to reach a target the
    human protein is not expressed in, ortholog-based tool compounds are
    sometimes the only starting point.

    Args:
        uniprot_accession: e.g. ``P00533``.
        species: Organism names to filter by (case-insensitive substring
            match). None returns all orthologs.

    Returns:
        ``{gene, orthologs}`` where each ortholog has ``accession``,
        ``organism``, ``gene``, and ``identity_pct``.
    """

    entry = _uniprot_entry(uniprot_accession)
    if "_error" in entry:
        return {"error": f"UniProt could not be reached: {entry['_error']}"}

    gene = ""
    genes = entry.get("genes", [])
    if genes:
        gene = genes[0].get("geneName", {}).get("value", "")

    # UniProt's OMA cross-reference is only the OMA group fingerprint — it
    # carries no per-species rows, so reading orthologs out of it returned an
    # empty list for every target. OMA's own API is what lists them.
    orthologs: list[dict[str, Any]] = []
    for row in _oma_orthologs(uniprot_accession):
        organism = (row.get("species") or {}).get("species", "")
        if species and not any(s.lower() in organism.lower() for s in species):
            continue
        orthologs.append({
            "accession": row.get("canonicalid", ""),
            "oma_id": row.get("omaid", ""),
            "organism": organism,
            "taxon_id": (row.get("species") or {}).get("taxon_id"),
            "relation": row.get("rel_type"),
        })

    for xref in entry.get("uniProtKBCrossReferences", []):
        if xref.get("database") != "OMA":
            continue
        props = {p.get("key"): p.get("value") for p in xref.get("properties", [])}
        organism = props.get("Organism", "")
        if not organism or (species and not any(s.lower() in organism.lower() for s in species)):
            continue
        orthologs.append({
            "accession": xref.get("id", ""),
            "organism": organism,
            "gene": props.get("GeneName", ""),
            "identity_pct": props.get("Identity"),
        })

    # Also surface the taxonomic lineage for context.
    organism_name = entry.get("organism", {}).get("scientificName", "")

    return {
        "uniprot_accession": uniprot_accession,
        "gene": gene,
        "organism": organism_name,
        "orthologs": orthologs,
        "total": len(orthologs),
        "note": (
            "Orthologs via OMA (Orthologous MAtrix). For rat/mouse, "
            "filter with species=['Rattus norvegicus','Mus musculus']."
        ),
    }


def register_tools(mcp: Any) -> None:
    """Register the biology tools."""

    @mcp.tool()
    def fetch_target_genetic_evidence(
        ensembl_id: str,
        n_diseases: int = 10,
    ) -> dict[str, Any]:
        """OpenTargets: genetic support for a target across diseases.

        Use this to answer whether the target has human genetic evidence
        linking it to the disease of interest — GWAS hits, rare-variant
        burden, or pheWAS associations — which is the validation step that
        a biochemical assay cannot provide.

        Args:
            ensembl_id: Ensembl gene id (e.g. ``ENSG00000141510`` for TP53).
                Obtain from ``fetch_target_info`` via the ``ensembl_id`` field.
            n_diseases: Top associated diseases to return.

        Returns:
            ``{target_symbol, top_diseases, genetic_evidence_rows}``.
            Each disease has an overall association ``score`` and, where
            available, a ``genetic_score``.
        """

        return _genetic_evidence(ensembl_id=ensembl_id, n_diseases=n_diseases)

    @mcp.tool()
    def fetch_target_expression(
        uniprot_accession: str,
        context: str = "tissue",
    ) -> dict[str, Any]:
        """Human Protein Atlas: where a target is expressed.

        Use this to check whether the target is expressed in the disease
        tissue before committing to a campaign. A target that scores well
        in an assay but is absent from the organ of interest is a known
        reason for late-stage failures.

        Args:
            uniprot_accession: e.g. ``P00533`` for EGFR.
            context: ``tissue`` (bulk RNA, the standard for target selection)
                or ``single_cell``.

        Returns:
            ``{tissue_rows}`` with ``{tissue, cell_type, level, reliability}``
            per row. Level is 'High', 'Medium', 'Low', or 'Not detected'.
        """

        return _expression(uniprot_accession=uniprot_accession, context=context)

    @mcp.tool()
    def fetch_pathway_annotations(
        uniprot_accession: str,
    ) -> dict[str, Any]:
        """Reactome and KEGG pathways a target participates in.

        Answers pathway-context questions: what process does the target
        regulate, what else is in that pathway that could confound selectivity,
        and which diseases map to it.

        Args:
            uniprot_accession: e.g. ``P00533``.

        Returns:
            ``{reactome_pathways, kegg_pathways, go_biological_process}``.
        """

        return _pathways(uniprot_accession=uniprot_accession)

    @mcp.tool()
    def fetch_target_orthology(
        uniprot_accession: str,
        species: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Orthologs of a human target in other species.

        Use this to find the rat or mouse ortholog when the animal model
        is not obvious, or to check that a tool compound optimised on the
        human protein will transfer to the preclinical species.

        Args:
            uniprot_accession: e.g. ``P00533``.
            species: Filter by organism name, e.g.
                ``['Rattus norvegicus', 'Mus musculus']``.

        Returns:
            ``{orthologs}`` with ``{accession, organism, gene, identity_pct}``
            per entry.
        """

        return _orthology(uniprot_accession=uniprot_accession, species=species)
