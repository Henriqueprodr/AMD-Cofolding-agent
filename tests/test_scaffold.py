"""Tests for reinvent4_mcp/tools/scaffold.py -- what does this series share?

A scaffold is the one input LibInvent cannot do without, and until now the
only way to get one was for the agent to read a list of SMILES and write the
core out by hand. That is the same act as recalling a SMILES from memory,
which the prompt forbids everywhere else, and it fails the same way: quietly,
with a string that parses into the wrong molecule.

So these tests are mostly about what the caller is entitled to believe. A
derived scaffold has to be usable by the generator that asked for it, which
means numbered attachment points at positions the series really varies; and
when the molecules share nothing worth calling a core, that has to be visible
rather than dressed up as a result at full coverage.
"""

from __future__ import annotations

import pytest

pytest.importorskip("rdkit", reason="deriving a scaffold is RDKit's work")

from reinvent4_mcp.tools import scaffold  # noqa: E402

#: A benzamide series substituted on both rings and nowhere else.
SERIES = [
    "CCOc1ccc(cc1)C(=O)Nc1ccccc1",
    "COc1ccc(cc1)C(=O)Nc1ccccc1",
    "Clc1ccc(cc1)C(=O)Nc1ccc(F)cc1",
    "Cc1ccc(cc1)C(=O)Nc1ccc(C)cc1",
    "c1ccc(cc1)C(=O)Nc1ccc(Cl)cc1",
]

#: The same anilide carrying a different ring each time, so no two molecules
#: share a Murcko scaffold and only the MCS can find the core.
RING_HOPPED = [
    "O=C(Nc1ccccc1)c1ccc2ccccc2c1",
    "O=C(Nc1ccccc1)c1ccc2[nH]ccc2c1",
    "O=C(Nc1ccccc1)c1ccoc1",
    "O=C(Nc1ccccc1)c1ccncc1",
]


class TestTheCoreOfASeries:

    def test_it_finds_the_shared_benzamide(self):
        result = scaffold.derive_scaffold(SERIES)

        assert result["bare_scaffold"] == "O=C(Nc1ccccc1)c1ccccc1"

    def test_every_molecule_is_accounted_for(self):
        result = scaffold.derive_scaffold(SERIES)

        assert result["coverage"] == 1.0
        assert result["n_matched"] == result["n_input"] == len(SERIES)

    def test_murcko_is_enough_for_a_real_series(self):
        # The fallback to MCS costs seconds and keeps shared side chains that
        # are not part of the core, so a series that Murcko handles should
        # never reach it.
        assert scaffold.derive_scaffold(SERIES)["method"] == "murcko"


class TestWhatLibInventNeeds:

    def test_the_scaffold_comes_back_with_numbered_points(self):
        # `[*]` without a number is refused by LibInventInput, so an
        # unnumbered scaffold would be useless at exactly the step it exists
        # for.
        result = scaffold.derive_scaffold(SERIES)

        assert "[*:1]" in result["scaffold"]
        assert "[*:2]" in result["scaffold"]

    def test_it_marks_the_two_positions_the_series_varies(self):
        result = scaffold.derive_scaffold(SERIES)

        assert result["attachment_points"] == 2

    def test_the_scaffold_is_a_molecule_rdkit_can_read(self):
        from rdkit import Chem

        result = scaffold.derive_scaffold(SERIES)

        assert Chem.MolFromSmiles(result["scaffold"]) is not None

    def test_the_bare_core_is_offered_too(self):
        # Profiling and substructure search want the core without dummies.
        result = scaffold.derive_scaffold(SERIES)

        assert "*" not in result["bare_scaffold"]

    def test_points_can_be_turned_off(self):
        result = scaffold.derive_scaffold(SERIES, attachment_points=False)

        assert "*" not in result["scaffold"]
        assert result["attachment_points"] == 0

    def test_an_unsubstituted_core_says_there_is_nothing_to_decorate(self):
        # Handing this to LibInvent produces the input back; better to know.
        result = scaffold.derive_scaffold(["c1ccccc1", "c1ccccc1"])

        assert result["attachment_points"] == 0
        assert "nothing for LibInvent to decorate" in result["note"]


class TestHowFarTheCoreGoes:

    def test_it_counts_the_distinct_groups_at_each_point(self):
        result = scaffold.derive_scaffold(SERIES)

        assert result["r_group_counts"] == {"R1": 3, "R2": 4}

    def test_a_split_set_falls_back_to_the_common_substructure(self):
        result = scaffold.derive_scaffold(RING_HOPPED)

        assert result["method"] == "mcs"
        assert "fell back" in result["note"]

    def test_the_fallback_recovers_a_real_core_across_the_ring_hop(self):
        # The varying part is the second ring (naphthalene/indole/furan/
        # pyridine), so it cannot appear in the shared core. The MCS should
        # recover the N-phenyl amide piece: 9 heavy atoms, all molecules match.
        from rdkit import Chem

        result = scaffold.derive_scaffold(RING_HOPPED)

        core = Chem.MolFromSmiles(result["bare_scaffold"])
        assert core is not None, f"bare_scaffold is not parseable: {result['bare_scaffold']!r}"
        assert core.GetNumHeavyAtoms() >= 6
        assert result["n_matched"] == len(RING_HOPPED)

    def test_the_fallback_can_be_refused_by_asking_for_murcko(self):
        result = scaffold.derive_scaffold(RING_HOPPED, method="murcko")

        assert result["method"] == "murcko"
        assert result["coverage"] < 1.0

    def test_a_demanding_threshold_forces_the_fallback(self):
        result = scaffold.derive_scaffold(SERIES, min_fraction=1.01)

        assert result["method"] == "mcs"

    def test_molecules_with_nothing_in_common_say_so(self):
        # MCS always finds *something* -- one carbon, at full coverage. Left
        # unremarked that reads as a perfect scaffold, and decorating it is
        # generating from scratch.
        result = scaffold.derive_scaffold(
            ["CCCCCCCO", "c1ccc2[nH]ccc2c1CCN", "O=C(O)c1ccccc1Cl", "N#Cc1ccncc1"]
        )

        assert "too little to be a series scaffold" in result["note"]


class TestWhenItCannotAnswer:

    def test_one_molecule_shares_a_core_with_nothing(self):
        result = scaffold.derive_scaffold(["c1ccccc1C(=O)N"])

        assert "at least 2" in result["error"]

    def test_unparsable_input_is_named_not_silently_dropped(self):
        result = scaffold.derive_scaffold(["definitely not smiles", "also(bad"])

        assert "definitely not smiles" in result["error"]

    def test_a_bad_molecule_among_good_ones_is_reported_beside_the_answer(self):
        result = scaffold.derive_scaffold([*SERIES, "also(bad"])

        assert result["unparsed"] == ["also(bad"]
        assert result["bare_scaffold"] == "O=C(Nc1ccccc1)c1ccccc1"

    def test_an_unknown_method_is_refused_rather_than_guessed(self):
        result = scaffold.derive_scaffold(SERIES, method="banana")

        assert "banana" in result["error"]

    def test_empty_strings_do_not_count_as_molecules(self):
        result = scaffold.derive_scaffold(["", "   ", "c1ccccc1C"])

        assert "at least 2" in result["error"]


class TestTheRGroupTable:

    def test_each_molecule_gets_a_row(self):
        result = scaffold.decompose_series(SERIES)

        assert len(result["rows"]) == len(SERIES)

    def test_a_row_carries_the_substituent_it_had(self):
        result = scaffold.decompose_series(SERIES)

        first = next(r for r in result["rows"] if r["smiles"] == SERIES[0])
        assert "CCO" in first["r_groups"].values()

    def test_the_summary_counts_what_a_position_has_explored(self):
        result = scaffold.decompose_series(SERIES)

        groups = {g["group"] for g in result["positions"]["R2"]["most_common"]}
        assert {"CCO", "CO", "Cl", "C"} <= groups

    def test_a_scaffold_from_the_other_tool_is_accepted_as_given(self):
        # The two tools are meant to chain, so the attachment points one
        # emits must not have to be stripped by hand before the other reads
        # them.
        derived = scaffold.derive_scaffold(SERIES)

        result = scaffold.decompose_series(SERIES, scaffold=derived["scaffold"])

        assert len(result["rows"]) == len(SERIES)
        assert not result["unmatched"]

    def test_a_molecule_off_the_series_is_listed_not_forced_in(self):
        result = scaffold.decompose_series([*SERIES, "CCCCCCCCCC"])

        assert result["unmatched"] == ["CCCCCCCCCC"]
        assert len(result["rows"]) == len(SERIES)

    def test_an_unreadable_scaffold_is_refused(self):
        result = scaffold.decompose_series(SERIES, scaffold="not a scaffold")

        assert "Cannot parse" in result["error"]


class TestTheToolsAreRegistered:

    def test_both_are_exposed_under_the_names_the_prompt_uses(self):
        names: list[str] = []

        class Recorder:
            def tool(self):
                def take(fn):
                    names.append(fn.__name__)
                    return fn

                return take

        scaffold.register_tools(Recorder())

        assert names == ["derive_scaffold_from_series", "decompose_series_r_groups"]
