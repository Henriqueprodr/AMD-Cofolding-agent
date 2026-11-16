"""Tests for the outside databases a target dossier is assembled from.

Every tool here is a thin wrapper over somebody else's HTTP API, so the way
they fail is not a wrong answer but a request the other side stopped
accepting: a GraphQL argument that was renamed, a page route that was
retired, a parameter that quietly changed meaning from "any of these" to
"all of these". None of that shows up as an exception in our code -- it
comes back as a 400, a 404, or an empty list, and the agent reports "the
service was unreachable" while the service is in fact perfectly well.

So these tests do not check that a dossier is correct. They pin the shape of
the request we send and the shape of the response we know how to read, which
is the part that silently rotted.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from reinvent4_mcp.tools import biology, patents


class _Response:
    """Enough of ``requests.Response`` for the wrappers under test."""

    def __init__(self, payload: Any, status: int = 200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> Any:
        return self._payload


class TestOpenTargetsAsksForAPageOfDiseases:
    """``associatedDiseases`` paginates through a Pagination input object.

    Passing a bare ``size`` is not a narrower result, it is a parse error:
    the whole query is rejected with a 400 before any data is looked up, so
    genetic evidence was unavailable for every target.
    """

    def test_the_query_does_not_pass_a_bare_size_argument(self):
        assert "associatedDiseases(size:" not in biology._OT_EVIDENCE_QUERY

    def test_the_query_paginates_through_the_page_object(self):
        assert "associatedDiseases(page: {index: 0, size: $n})" in biology._OT_EVIDENCE_QUERY

    def test_a_disease_row_carries_its_genetic_association_score(self):
        payload = {
            "data": {
                "target": {
                    "id": "ENSG00000146648",
                    "approvedSymbol": "EGFR",
                    "approvedName": "epidermal growth factor receptor",
                    "associatedDiseases": {
                        "count": 5860,
                        "rows": [{
                            "disease": {"id": "MONDO_0005233", "name": "non-small cell lung carcinoma"},
                            "score": 0.85,
                            "datatypeScores": [
                                {"id": "literature", "score": 0.99},
                                {"id": "genetic_association", "score": 0.69},
                            ],
                        }],
                    },
                }
            }
        }
        with patch.object(biology.requests, "post", return_value=_Response(payload)):
            result = biology._genetic_evidence("ENSG00000146648", n_diseases=1)

        assert result["target_symbol"] == "EGFR"
        assert result["top_diseases"][0]["genetic_score"] == 0.69


class TestProteinAtlasIsKeyedOnTheEnsemblGeneId:
    """HPA has no "{SYMBOL}-{ACCESSION}/tissue.json" route and never had one.

    The old code built that id from UniProt and asked for it, which answers
    404 with the site's HTML shell for every target alive. Everything HPA
    does serve is keyed on the Ensembl gene id instead.
    """

    #: One ``search_download.php`` record, trimmed to the columns read.
    RECORD = {
        "Gene": "EGFR",
        "Ensembl": "ENSG00000146648",
        "Uniprot": ["P00533"],
        "RNA tissue specificity": "Low tissue specificity",
        "RNA tissue distribution": "Detected in many",
        "Tissue RNA - liver [nTPM]": "32.2",
        "Tissue RNA - lung [nTPM]": "14.1",
        "Tissue RNA - placenta [nTPM]": "61.8",
        "Tissue RNA - bone marrow [nTPM]": "0.4",
    }

    SUMMARY = {
        "Gene": "EGFR",
        "RNA tissue specificity": "Low tissue specificity",
        "RNA tissue distribution": "Detected in many",
        "Reliability (IH)": "Enhanced",
    }

    def _expression(self, context: str = "tissue"):
        def get(url: str, **kwargs: Any) -> _Response:
            if url == biology._HPA_QUERY:
                return _Response([self.RECORD])
            return _Response(self.SUMMARY)

        with patch.object(biology.requests, "get", side_effect=get):
            return biology._expression("P00533", context=context)

    def test_the_gene_id_is_the_ensembl_id(self):
        assert self._expression()["gene_id"] == "ENSG00000146648"

    def test_tissue_columns_are_requested_by_name(self):
        """HPA has no group code for "every tissue", so each one is asked for."""
        captured: dict[str, Any] = {}

        def get(url: str, **kwargs: Any) -> _Response:
            if url == biology._HPA_QUERY:
                captured.update(kwargs.get("params") or {})
                return _Response([self.RECORD])
            return _Response(self.SUMMARY)

        with patch.object(biology.requests, "get", side_effect=get):
            biology._expression("P00533")

        assert "t_RNA_liver" in captured["columns"]
        assert "t_RNA_placenta" in captured["columns"]

    def test_rows_are_banded_the_way_hpa_bands_them(self):
        rows = {row["tissue"]: row["level"] for row in self._expression()["tissue_rows"]}
        assert rows["placenta"] == "High"       # >= 50
        assert rows["liver"] == "Medium"        # >= 10
        assert rows["bone marrow"] == "Not detected"  # < 1

    def test_rows_are_ordered_by_how_much_is_there(self):
        rows = self._expression()["tissue_rows"]
        assert [row["tissue"] for row in rows][:2] == ["placenta", "liver"]

    def test_the_broadly_expressed_verdict_is_surfaced(self):
        """The question this tool is asked is "is it all over healthy tissue",
        which HPA answers directly rather than leaving to be inferred."""
        result = self._expression()
        assert result["specificity"] == "Low tissue specificity"
        assert result["distribution"] == "Detected in many"


class TestOrthologsComeFromOma:
    """UniProt's OMA cross-reference is a group fingerprint, not a list.

    Reading orthologs out of it returned an empty list for every target,
    which reads as "this target has no mouse ortholog" rather than "we did
    not look it up".
    """

    ENTRY = {
        "genes": [{"geneName": {"value": "EGFR"}}],
        "organism": {"scientificName": "Homo sapiens"},
        # Exactly what UniProt carries: an id and a bare fingerprint.
        "uniProtKBCrossReferences": [
            {"database": "OMA", "id": "AAKQVCH", "properties": [{"key": "Fingerprint", "value": "-"}]}
        ],
    }

    OMA_ROWS = [
        {
            "omaid": "MOUSE12345",
            "canonicalid": "Q01279",
            "rel_type": "1:1",
            "species": {"species": "Mus musculus", "taxon_id": 10090},
        },
        {
            "omaid": "CAEEL6789",
            "canonicalid": "P34545",
            "rel_type": "m:1",
            "species": {"species": "Caenorhabditis elegans", "taxon_id": 6239},
        },
    ]

    def _orthology(self, species: list[str] | None = None):
        with patch.object(biology, "_uniprot_entry", return_value=self.ENTRY), \
             patch.object(biology, "_oma_orthologs", return_value=self.OMA_ROWS):
            return biology._orthology("P00533", species=species)

    def test_the_mouse_ortholog_is_found(self):
        result = self._orthology(species=["Mus musculus"])
        assert result["total"] == 1
        assert result["orthologs"][0]["accession"] == "Q01279"
        assert result["orthologs"][0]["relation"] == "1:1"

    def test_the_fingerprint_crossreference_is_not_mistaken_for_an_ortholog(self):
        """It names no organism, so it is not a row about a species."""
        accessions = [o["accession"] for o in self._orthology()["orthologs"]]
        assert "AAKQVCH" not in accessions

    def test_unfiltered_returns_every_species_oma_has(self):
        assert self._orthology()["total"] == len(self.OMA_ROWS)


class TestSureChemblStructureSearch:
    """The compound route this used to call is gone from both hosts.

    surechembl.org answers it with ``{"detail":"Not Found"}`` and the EBI
    host serves its single-page app, so the JSON parse failed and every
    search reported the database as unreachable. The rebuilt API matches a
    structure to compound ids first, then asks which documents claim them.
    """

    MATCHED = {"chemical_ids": [9423, 29444884], "total": 2}

    DOCS = {
        "numFound": 1289,
        "docs": [{
            "ucid": "CN-101365440-A",
            "pd": 20090211,
            "ttl_en": ["Methods of treating cancer"],
            "pa": ["MERCK & CO INC", "MERCK & CO INC"],
            "score": 1.0,
        }],
    }

    def _search(self, threshold: float = 1.0, get=None):
        def default_get(url: str, **kwargs: Any) -> _Response:
            return _Response(self.DOCS)

        with patch.object(patents.requests, "post", return_value=_Response(self.MATCHED)), \
             patch.object(patents.requests, "get", side_effect=get or default_get):
            return patents._by_structure("COc1ccccc1", threshold=threshold, limit=5)

    def test_an_exact_search_is_sent_as_an_identical_search(self):
        """Threshold 1.0 is a different search type, not a wound-up cutoff."""
        sent: dict[str, Any] = {}

        def post(url: str, **kwargs: Any) -> _Response:
            sent.update(kwargs.get("json") or {})
            return _Response(self.MATCHED)

        with patch.object(patents.requests, "post", side_effect=post), \
             patch.object(patents.requests, "get", return_value=_Response(self.DOCS)):
            patents._by_structure("COc1ccccc1", threshold=1.0, limit=5)

        assert sent["struct_search_type"] == "identical"
        assert "threshold" not in sent

    def test_a_similarity_search_carries_its_cutoff(self):
        sent: dict[str, Any] = {}

        def post(url: str, **kwargs: Any) -> _Response:
            sent.update(kwargs.get("json") or {})
            return _Response(self.MATCHED)

        with patch.object(patents.requests, "post", side_effect=post), \
             patch.object(patents.requests, "get", return_value=_Response(self.DOCS)):
            patents._by_structure("COc1ccccc1", threshold=0.85, limit=5)

        assert sent["struct_search_type"] == "similarity"
        assert sent["threshold"] == 0.85

    def test_compounds_are_looked_up_one_at_a_time(self):
        """Several ids in one request means "documents claiming all of them",
        which for the salt and tautomer records of one structure is nothing."""
        asked: list[Any] = []

        def get(url: str, **kwargs: Any) -> _Response:
            asked.append((kwargs.get("params") or {}).get("chemicalIds"))
            return _Response(self.DOCS)

        self._search(get=get)
        assert asked == ["9423", "29444884"]

    def test_a_document_is_read_out_of_the_solr_style_row(self):
        hit = self._search()["patents"][0]
        assert hit["surechembl_id"] == "CN-101365440-A"
        assert hit["publication_date"] == "2009-02-11"
        assert hit["title"] == "Methods of treating cancer"
        # The same company is spelled several times in one row.
        assert hit["assignee"] == ["MERCK & CO INC"]

    def test_the_same_patent_claiming_two_compounds_appears_once(self):
        result = self._search()
        assert result["total"] == 1

    def test_a_structure_nothing_claims_is_not_reported_as_an_outage(self):
        """"No patent matched" and "we could not ask" must not look alike."""
        with patch.object(patents.requests, "post", return_value=_Response({"chemical_ids": []})):
            result = patents._by_structure("COc1ccccc1", threshold=1.0, limit=5)

        assert result["total"] == 0
        assert "error" not in result
        assert "freedom to operate" in result["note"]

    def test_an_outage_is_still_reported_as_one(self):
        def get(url: str, **kwargs: Any) -> _Response:
            raise patents.requests.RequestException("connection refused")

        result = self._search(get=get)
        assert "could not be reached" in result["error"]
        assert result["patents"] == []
