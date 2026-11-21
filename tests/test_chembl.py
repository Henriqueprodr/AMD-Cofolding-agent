# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the ChEMBL measured-bioactivity lookup.

Covers:
- _resolve_molecule: ChEMBL id, exact name, synonym fallback, fuzzy fallback
- molfile stripping (only= does not prune nested objects)
- _resolve_target: accession path, target-type/organism ranking, candidates
- fetch_bioactivity: the three modes, string->float casting, limit clamping,
  summary statistics, the "no data != inactive" note, and the
  do-not-fabricate error path
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from reinvent4_mcp.tools import chembl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = status_code < 400
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


def _molecule(chembl_id: str = "CHEMBL1957266", **kwargs: Any) -> dict[str, Any]:
    mol = {
        "molecule_chembl_id": chembl_id,
        "pref_name": None,
        "max_phase": None,
        "molecule_type": "Small molecule",
        "first_approval": None,
        "molecule_structures": {
            "canonical_smiles": "Cc1sc2c(c1C)C(c1ccc(Cl)cc1)=N[C@@H](CC(=O)OC(C)(C)C)c1nnc(C)n1-2",
            # ChEMBL attaches this even when only= excludes it.
            "molfile": "\n     RDKit          2D\n" + "x" * 4000,
            "standard_inchi_key": "DNVXATUJJDPFDM-KRWDZBQOSA-N",
        },
    }
    mol.update(kwargs)
    return mol


def _target(
    chembl_id: str = "CHEMBL1163125",
    target_type: str = "SINGLE PROTEIN",
    organism: str = "Homo sapiens",
    pref_name: str = "Bromodomain-containing protein 4",
) -> dict[str, Any]:
    return {
        "target_chembl_id": chembl_id,
        "pref_name": pref_name,
        "target_type": target_type,
        "organism": organism,
        "target_components": [{"accession": "O60885"}],
    }


def _activity(pchembl: str = "8.41", **kwargs: Any) -> dict[str, Any]:
    row = {
        "molecule_chembl_id": "CHEMBL1957266",
        "molecule_pref_name": None,
        "target_chembl_id": "CHEMBL1163125",
        "target_pref_name": "Bromodomain-containing protein 4",
        "target_organism": "Homo sapiens",
        "standard_type": "IC50",
        "standard_relation": "=",
        # ChEMBL returns every number as a string.
        "standard_value": "3.9",
        "standard_units": "nM",
        "pchembl_value": pchembl,
        "assay_chembl_id": "CHEMBL3706306",
        "assay_type": "B",
        "assay_description": "Inhibition of BRD4",
        "document_chembl_id": "CHEMBL3705899",
        "bao_label": "single protein format",
    }
    row.update(kwargs)
    return row


def _page(key: str, items: list[dict], total: int | None = None) -> dict:
    return {
        key: items,
        "page_meta": {"total_count": total if total is not None else len(items)},
    }


# ---------------------------------------------------------------------------
# Molecule resolution
# ---------------------------------------------------------------------------


class TestResolveMolecule:
    """Name -> ChEMBL molecule, with the documented fallback cascade."""

    @patch("reinvent4_mcp.tools.chembl.requests.get")
    def test_chembl_id_resolves_directly(self, mock_get):
        mock_get.return_value = _mock_response(_molecule())

        mol, note = chembl._resolve_molecule("CHEMBL1957266")

        assert mol["chembl_id"] == "CHEMBL1957266"
        assert note is None
        assert mock_get.call_count == 1
        assert "molecule/CHEMBL1957266.json" in mock_get.call_args[0][0]

    @patch("reinvent4_mcp.tools.chembl.requests.get")
    def test_exact_name_hit_skips_later_lookups(self, mock_get):
        mock_get.return_value = _mock_response(_page("molecules", [_molecule()]))

        mol, note = chembl._resolve_molecule("JQ1")

        assert mol["chembl_id"] == "CHEMBL1957266"
        assert note is None
        assert mock_get.call_count == 1

    @patch("reinvent4_mcp.tools.chembl.requests.get")
    def test_synonym_fallback(self, mock_get):
        mock_get.side_effect = [
            _mock_response(_page("molecules", [])),          # pref_name__iexact
            _mock_response(_page("molecules", [_molecule()])),  # synonym
        ]

        mol, note = chembl._resolve_molecule("JQ1")

        assert mol["chembl_id"] == "CHEMBL1957266"
        assert note is None
        assert mock_get.call_count == 2

    @patch("reinvent4_mcp.tools.chembl.requests.get")
    def test_fuzzy_fallback_sets_a_warning_note(self, mock_get):
        mock_get.side_effect = [
            _mock_response(_page("molecules", [])),
            _mock_response(_page("molecules", [])),
            _mock_response(_page("molecules", [_molecule()])),
        ]

        mol, note = chembl._resolve_molecule("jay cue one")

        assert mol["chembl_id"] == "CHEMBL1957266"
        assert note is not None
        assert "fuzzy" in note.lower()

    @patch("reinvent4_mcp.tools.chembl.requests.get")
    def test_molfile_is_stripped(self, mock_get):
        mock_get.return_value = _mock_response(_page("molecules", [_molecule()]))

        mol, _ = chembl._resolve_molecule("JQ1")

        assert "molfile" not in str(mol)
        assert mol["canonical_smiles"].startswith("Cc1sc2")

    @patch("reinvent4_mcp.tools.chembl.requests.get")
    def test_unknown_name_returns_note_not_molecule(self, mock_get):
        mock_get.return_value = _mock_response(_page("molecules", []))

        mol, note = chembl._resolve_molecule("not-a-real-compound")

        assert mol is None
        assert "No ChEMBL molecule" in note


# ---------------------------------------------------------------------------
# Target resolution
# ---------------------------------------------------------------------------


class TestResolveTarget:
    """Accession beats free text; single proteins beat complexes."""

    @patch("reinvent4_mcp.tools.chembl.requests.get")
    def test_accession_prefers_single_protein(self, mock_get):
        mock_get.return_value = _mock_response(
            _page(
                "targets",
                [
                    _target("CHEMBL4106143", "PROTEIN COMPLEX", pref_name="BRD4/HDAC1"),
                    _target("CHEMBL1163125", "SINGLE PROTEIN"),
                ],
            )
        )

        best, candidates, note = chembl._resolve_target("O60885", None)

        assert best["chembl_id"] == "CHEMBL1163125"
        assert best["uniprot_accessions"] == ["O60885"]
        assert len(candidates) == 1
        assert note is None  # accessions are unambiguous

    @patch("reinvent4_mcp.tools.chembl.requests.get")
    def test_free_text_warns_and_prefers_human(self, mock_get):
        mock_get.return_value = _mock_response(
            _page(
                "targets",
                [
                    _target("CHEMBL3309085", organism="Danio rerio"),
                    _target("CHEMBL1163125", organism="Homo sapiens"),
                ],
            )
        )

        best, candidates, note = chembl._resolve_target("BRD4", None)

        assert best["organism"] == "Homo sapiens"
        assert note is not None and "ambiguous" in note
        assert candidates[0]["organism"] == "Danio rerio"

    @patch("reinvent4_mcp.tools.chembl.requests.get")
    def test_explicit_organism_overrides_human_default(self, mock_get):
        mock_get.return_value = _mock_response(
            _page(
                "targets",
                [
                    _target("CHEMBL1163125", organism="Homo sapiens"),
                    _target("CHEMBL3309085", organism="Danio rerio"),
                ],
            )
        )

        best, _, _ = chembl._resolve_target("BRD4", "Danio rerio")

        assert best["organism"] == "Danio rerio"


# ---------------------------------------------------------------------------
# fetch_bioactivity
# ---------------------------------------------------------------------------


class TestFetchBioactivity:
    """The public entry point: modes, casting, caps, summary, failure."""

    @patch("reinvent4_mcp.tools.chembl.requests.get")
    def test_no_input_errors_without_any_http_call(self, mock_get):
        result = chembl.fetch_bioactivity()

        assert "error" in result
        mock_get.assert_not_called()

    @patch("reinvent4_mcp.tools.chembl.requests.get")
    def test_pair_mode_shape_and_numeric_casting(self, mock_get):
        mock_get.side_effect = [
            _mock_response(_page("molecules", [_molecule()])),
            _mock_response(_page("targets", [_target()])),
            _mock_response(_page("activities", [_activity()], total=250)),
        ]

        result = chembl.fetch_bioactivity(compound="JQ1", target="O60885", limit=5)

        assert result["molecule"]["chembl_id"] == "CHEMBL1957266"
        assert result["target"]["chembl_id"] == "CHEMBL1163125"
        assert result["n_activities"] == 1
        assert result["total_available"] == 250
        row = result["activities"][0]
        # Strings in the API, floats in the result.
        assert row["standard_value"] == 3.9
        assert row["pchembl_value"] == 8.41
        assert row["assay_type"] == "B"
        # The user must be told these are the optimistic rows.
        assert "optimistic" in result["note"]

    @patch("reinvent4_mcp.tools.chembl.requests.get")
    def test_compound_only_mode_queries_without_target_filter(self, mock_get):
        mock_get.side_effect = [
            _mock_response(_page("molecules", [_molecule()])),
            _mock_response(_page("activities", [_activity()])),
        ]

        result = chembl.fetch_bioactivity(compound="JQ1")

        assert result["target"] is None
        params = mock_get.call_args_list[-1][1]["params"]
        assert "target_chembl_id" not in params
        assert params["molecule_chembl_id"] == "CHEMBL1957266"

    @patch("reinvent4_mcp.tools.chembl.requests.get")
    def test_activity_query_filters_censored_and_null_values(self, mock_get):
        mock_get.side_effect = [
            # A ChEMBL id resolves through GET target/<id>.json, which
            # returns the bare object rather than a paged list.
            _mock_response(_target()),
            _mock_response(_page("activities", [_activity()])),
        ]

        chembl.fetch_bioactivity(target="CHEMBL1163125")

        params = mock_get.call_args_list[-1][1]["params"]
        assert params["standard_relation"] == "="
        assert params["pchembl_value__isnull"] == "false"
        assert params["order_by"] == "-pchembl_value"
        assert params["standard_type__in"] == "IC50,Ki,Kd,EC50"

    @patch("reinvent4_mcp.tools.chembl.requests.get")
    def test_limit_is_clamped(self, mock_get):
        mock_get.side_effect = [
            _mock_response(_target()),
            _mock_response(_page("activities", [])),
        ]

        chembl.fetch_bioactivity(target="CHEMBL1163125", limit=5000)

        assert mock_get.call_args_list[-1][1]["params"]["limit"] == chembl._MAX_LIMIT

    @patch("reinvent4_mcp.tools.chembl.requests.get")
    def test_summary_statistics(self, mock_get):
        mock_get.side_effect = [
            _mock_response(_target()),
            _mock_response(
                _page(
                    "activities",
                    [
                        _activity("9.00"),
                        _activity("8.00", standard_type="Ki"),
                        _activity("7.00", target_chembl_id="CHEMBL2001"),
                    ],
                )
            ),
        ]

        summary = chembl.fetch_bioactivity(target="CHEMBL1163125")["summary"]

        assert summary["best_pchembl"] == 9.0
        assert summary["median_pchembl"] == 8.0
        assert summary["type_counts"] == {"IC50": 2, "Ki": 1}
        assert summary["n_distinct_targets"] == 2

    @patch("reinvent4_mcp.tools.chembl.requests.get")
    def test_empty_result_says_not_inactive(self, mock_get):
        mock_get.side_effect = [
            _mock_response(_page("molecules", [_molecule()])),
            _mock_response(_page("targets", [_target()])),
            _mock_response(_page("activities", [])),
        ]

        result = chembl.fetch_bioactivity(compound="JQ1", target="O60885")

        assert result["activities"] == []
        assert "error" not in result
        assert "does NOT mean" in result["note"]

    @patch("reinvent4_mcp.tools.chembl.requests.get")
    def test_network_failure_returns_do_not_fabricate_error(self, mock_get):
        import requests

        mock_get.side_effect = requests.ConnectionError("unreachable")

        result = chembl.fetch_bioactivity(compound="JQ1")

        assert "error" in result
        assert "do not quote" in result["error"]
        assert "activities" not in result
