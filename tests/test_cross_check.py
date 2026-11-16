"""Tests for reinvent4_mcp/tools/cross_check.py -- is this structure known?

A generator reports its output as new because nothing in the run has seen it
before, which is a much weaker claim than novelty. The cost of getting this
wrong is asymmetric: calling a known compound new sends a programme after
something that is already on the market, while the reverse merely spends one
lookup.

So the question these tests ask of each case is what the caller is entitled to
conclude -- in particular that a database which could not be reached is never
allowed to read as evidence of novelty.
"""

from __future__ import annotations

from typing import Any

import pytest

from reinvent4_mcp.tools import cross_check

_KEY = "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"


def _answers(monkeypatch, pubchem: dict[str, Any], chembl: dict[str, Any]) -> None:
    """Both databases answer as told, without a network."""

    monkeypatch.setattr(cross_check, "_pubchem_by_inchikey", lambda _k: pubchem)
    monkeypatch.setattr(cross_check, "_chembl_by_inchikey", lambda _k: chembl)


def _chembl_hit(**extra: Any) -> dict[str, Any]:
    record = {
        "molecule_chembl_id": "CHEMBL25",
        "pref_name": "ASPIRIN",
        "max_phase": 4,
        "first_approval": 1950,
        "molecule_type": "Small molecule",
        **extra,
    }
    return {"found": True, "records": [record]}


class TestAStructureBothDatabasesKnow:

    def test_it_is_called_known(self, monkeypatch):
        _answers(monkeypatch, {"found": True, "cid": 2244}, _chembl_hit())

        assert cross_check._check_known(inchikey=_KEY)["known"] is True

    def test_the_verdict_names_both_and_the_clinical_phase(self, monkeypatch):
        # Phase 4 is the difference between "someone has made this" and
        # "this is a marketed drug", and only one of those ends a project.
        _answers(monkeypatch, {"found": True, "cid": 2244}, _chembl_hit())

        verdict = cross_check._check_known(inchikey=_KEY)["verdict"]

        assert "2244" in verdict
        assert "CHEMBL25" in verdict
        assert "ASPIRIN" in verdict
        assert "phase 4" in verdict


class TestAStructureNeitherKnows:

    def test_it_is_not_called_known(self, monkeypatch):
        _answers(monkeypatch, {"found": False}, {"found": False})

        result = cross_check._check_known(inchikey=_KEY)

        assert result["known"] is False
        assert "Not found" in result["verdict"]

    def test_one_database_alone_is_enough_to_be_known(self, monkeypatch):
        _answers(monkeypatch, {"found": False}, _chembl_hit())

        assert cross_check._check_known(inchikey=_KEY)["known"] is True


class TestADatabaseThatCouldNotBeAsked:
    """A missing answer is not a negative one."""

    def test_an_unreachable_database_is_not_read_as_novelty(self, monkeypatch):
        _answers(
            monkeypatch,
            {"found": False, "note": "Could not reach PubChem (timeout)."},
            {"found": False},
        )

        verdict = cross_check._check_known(inchikey=_KEY)["verdict"]

        assert "not evidence of novelty" in verdict
        assert "PubChem" in verdict

    def test_a_real_hit_still_wins_over_an_unreachable_sibling(self, monkeypatch):
        _answers(
            monkeypatch,
            {"found": False, "note": "Could not reach PubChem (timeout)."},
            _chembl_hit(),
        )

        result = cross_check._check_known(inchikey=_KEY)

        assert result["known"] is True
        assert "Known compound" in result["verdict"]

    def test_a_clean_miss_is_not_dressed_up_as_an_outage(self, monkeypatch):
        _answers(monkeypatch, {"found": False}, {"found": False})

        assert "not evidence" not in cross_check._check_known(inchikey=_KEY)["verdict"]


class TestWhatItRefusesToGuess:

    def test_it_needs_something_to_look_up(self):
        assert "error" in cross_check._check_known()

    def test_an_unparseable_smiles_is_an_error_not_a_miss(self, monkeypatch):
        pytest.importorskip("rdkit")
        monkeypatch.setattr(
            cross_check, "_pubchem_by_inchikey",
            lambda _k: pytest.fail("should not have been asked"),
        )

        result = cross_check._check_known(smiles="not a molecule")

        assert "error" in result
        assert "known" not in result

    def test_a_key_given_directly_needs_no_rdkit(self, monkeypatch):
        # The tool stays usable where RDKit is absent, which is the case in
        # several of the service images.
        _answers(monkeypatch, {"found": False}, {"found": False})
        monkeypatch.setattr(
            cross_check, "_inchikey_from_smiles",
            lambda _s: pytest.fail("a given key must not be recomputed"),
        )

        assert cross_check._check_known(inchikey=_KEY.lower())["inchikey"] == _KEY


class TestItIsRegisteredAsATool:

    def test_register_tools_registers_the_check(self):
        names: list[str] = []

        class _Mcp:
            def tool(self):
                def decorate(fn):
                    names.append(fn.__name__)
                    return fn
                return decorate

        cross_check.register_tools(_Mcp())

        assert names == ["check_compound_is_known"]

    def test_the_module_is_loaded_by_the_registry(self):
        from reinvent4_mcp.registry import EXTENSION_MODULES

        assert "reinvent4_mcp.tools.cross_check" in EXTENSION_MODULES
