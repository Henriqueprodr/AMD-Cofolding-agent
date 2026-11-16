"""Tests for reinvent4_mcp/tools/series_analysis.py -- reading a whole series.

Both tools answer a question the agent was previously answering by eye. A
matched pair is the claim "this one change caused this much of the activity
difference", and it is only worth making if the pair really does differ in
one place; a cluster set is the claim "these ten span the series", and it is
only worth making if the ten are actually spread out. Both claims are easy
to produce something plausible-looking for and hard to notice being wrong,
so the tests here check the property that makes the answer true rather than
the shape of the dictionary it arrives in.
"""

from __future__ import annotations

import pytest

pytest.importorskip("rdkit", reason="series analysis is RDKit's work")

from reinvent4_mcp.tools import series_analysis as sa  # noqa: E402

#: A benzamide series varying at one para position, so neighbouring members
#: are single-change pairs.
PARA_SERIES = [
    "O=C(Nc1ccccc1)c1ccc(C)cc1",
    "O=C(Nc1ccccc1)c1ccc(Cl)cc1",
    "O=C(Nc1ccccc1)c1ccc(F)cc1",
    "O=C(Nc1ccccc1)c1ccc(OC)cc1",
]

#: Three tight pairs from three unrelated chemotypes. Any clustering worth
#: the name separates the chemotypes and keeps each pair together.
THREE_CHEMOTYPES = [
    # Benzamides
    "O=C(Nc1ccccc1)c1ccc(C)cc1",
    "O=C(Nc1ccccc1)c1ccc(CC)cc1",
    # Sulfonamides on a long chain
    "CCCCCCCCS(=O)(=O)NCCCCCC",
    "CCCCCCCCCS(=O)(=O)NCCCCCC",
    # Purine-like heterocycles
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "Cn1cnc2c1c(=O)[nH]c(=O)n2C",
]


class TestWhatCountsAsAMatchedPair:

    def test_a_single_substituent_swap_is_a_pair(self):
        result = sa.compute_matched_molecular_pairs(PARA_SERIES)

        assert result["n_mmp_pairs"] > 0

    def test_the_transform_names_both_sides_of_the_change(self):
        result = sa.compute_matched_molecular_pairs(PARA_SERIES)

        for pair in result["pairs"]:
            assert ">>" in pair["transform"]
            before, after = pair["transform"].split(">>")
            assert before and after
            assert before != after

    def test_every_pair_is_two_of_the_molecules_given(self):
        result = sa.compute_matched_molecular_pairs(PARA_SERIES)

        for pair in result["pairs"]:
            assert pair["smiles_a"] in PARA_SERIES
            assert pair["smiles_b"] in PARA_SERIES
            assert pair["smiles_a"] != pair["smiles_b"]

    def test_the_pair_count_cannot_exceed_the_pairs_there_are(self):
        result = sa.compute_matched_molecular_pairs(PARA_SERIES)

        n = len(PARA_SERIES)
        assert result["n_total_pairs"] == n * (n - 1) // 2
        assert result["n_mmp_pairs"] <= result["n_total_pairs"]

    def test_unrelated_molecules_are_not_forced_into_pairs(self):
        result = sa.compute_matched_molecular_pairs(
            ["c1ccccc1", "CCCCCCCCCCCCCCCC(=O)O"]
        )

        assert result["n_mmp_pairs"] == 0
        assert result["pairs"] == []


class TestTheActivityDelta:

    ACTIVITIES = {
        "O=C(Nc1ccccc1)c1ccc(C)cc1": 6.0,
        "O=C(Nc1ccccc1)c1ccc(Cl)cc1": 8.5,
        "O=C(Nc1ccccc1)c1ccc(F)cc1": 6.2,
        "O=C(Nc1ccccc1)c1ccc(OC)cc1": 5.1,
    }

    def test_the_delta_is_b_minus_a(self):
        result = sa.compute_matched_molecular_pairs(
            PARA_SERIES, activities=self.ACTIVITIES
        )

        for pair in result["pairs"]:
            expected = self.ACTIVITIES[pair["smiles_b"]] - self.ACTIVITIES[pair["smiles_a"]]
            assert pair["delta_activity"] == pytest.approx(expected, abs=1e-3)

    def test_the_biggest_swing_is_reported_first(self):
        result = sa.compute_matched_molecular_pairs(
            PARA_SERIES, activities=self.ACTIVITIES
        )

        deltas = [abs(p["delta_activity"]) for p in result["pairs"]]
        assert deltas == sorted(deltas, reverse=True)

    def test_a_partly_annotated_series_still_sorts(self):
        """Activities usually cover some of a series, not all of it.

        The pairs that carry a delta are the ones worth reading first, and
        they only reach the top if a partly annotated set is sorted at all.
        """
        partial = {PARA_SERIES[0]: 5.0, PARA_SERIES[1]: 9.0}

        result = sa.compute_matched_molecular_pairs(PARA_SERIES, activities=partial)
        annotated = [p for p in result["pairs"] if "delta_activity" in p]

        if annotated:
            assert "delta_activity" in result["pairs"][0]

    def test_no_activities_means_no_delta_claimed(self):
        result = sa.compute_matched_molecular_pairs(PARA_SERIES)

        for pair in result["pairs"]:
            assert "delta_activity" not in pair


class TestWhenItCannotAnswer:

    def test_one_molecule_is_not_a_series(self):
        result = sa.compute_matched_molecular_pairs(["c1ccccc1"])

        assert "error" in result
        assert "at least 2" in result["error"]

    def test_unparsable_smiles_are_named_not_dropped_silently(self):
        result = sa.compute_matched_molecular_pairs(PARA_SERIES + ["not a molecule"])

        assert "not a molecule" in result["unparsed"]

    def test_an_oversized_series_is_refused_with_the_reason(self, monkeypatch):
        monkeypatch.setattr(sa, "_MAX_PAIRS", 2)

        result = sa.compute_matched_molecular_pairs(PARA_SERIES)

        assert "error" in result
        assert "sub-series" in result["error"]


class TestClusteringSeparatesChemotypes:

    def test_butina_does_not_merge_unrelated_chemotypes(self):
        result = sa.cluster_molecules(THREE_CHEMOTYPES, method="butina", cutoff=0.4)

        assert result["n_clusters"] >= 3

    def test_every_molecule_lands_in_exactly_one_cluster(self):
        result = sa.cluster_molecules(THREE_CHEMOTYPES, method="butina")

        placed = [m for c in result["clusters"] for m in c["members"]]
        assert sorted(placed) == sorted(THREE_CHEMOTYPES)

    def test_the_reported_size_matches_the_members(self):
        result = sa.cluster_molecules(THREE_CHEMOTYPES, method="butina")

        for cluster in result["clusters"]:
            assert cluster["size"] == len(cluster["members"])

    def test_one_representative_per_cluster(self):
        result = sa.cluster_molecules(THREE_CHEMOTYPES, method="butina")

        assert len(result["representatives"]) == result["n_clusters"]
        for rep, cluster in zip(result["representatives"], result["clusters"]):
            assert rep in cluster["members"]

    def test_representatives_can_be_turned_off(self):
        result = sa.cluster_molecules(THREE_CHEMOTYPES, pick_representative=False)

        assert result["representatives"] == []


class TestKMedoidsSpansTheSet:

    def test_it_returns_the_number_of_clusters_asked_for(self):
        result = sa.cluster_molecules(
            THREE_CHEMOTYPES, method="kmedoids", n_clusters=3
        )

        assert result["n_clusters"] == 3

    def test_the_seeds_spread_over_the_set_rather_than_clumping(self):
        """The point of asking for three clusters is three different answers.

        Seeding picks each medoid as the molecule furthest from those already
        chosen. Seeded the other way round -- nearest rather than furthest --
        all three medoids come from one chemotype, the remaining two
        chemotypes are split across whatever is left, and the caller who
        asked for a diverse subset gets near-duplicates. That failure leaves
        the cluster count intact, so the count is not what is checked here.
        """
        result = sa.cluster_molecules(
            THREE_CHEMOTYPES, method="kmedoids", n_clusters=3
        )

        from rdkit import Chem, DataStructs

        reps = result["representatives"]
        fps = [sa._fingerprint(Chem.MolFromSmiles(s)) for s in reps]
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                assert DataStructs.TanimotoSimilarity(fps[i], fps[j]) < 0.5

    def test_every_molecule_is_assigned(self):
        result = sa.cluster_molecules(
            THREE_CHEMOTYPES, method="kmedoids", n_clusters=3
        )

        placed = [m for c in result["clusters"] for m in c["members"]]
        assert sorted(placed) == sorted(THREE_CHEMOTYPES)

    def test_asking_for_more_clusters_than_molecules_is_not_an_error(self):
        result = sa.cluster_molecules(
            PARA_SERIES, method="kmedoids", n_clusters=99
        )

        assert "error" not in result
        assert result["n_clusters"] <= len(PARA_SERIES)


class TestClusteringRefusals:

    def test_an_unknown_method_names_the_ones_that_exist(self):
        result = sa.cluster_molecules(PARA_SERIES, method="dbscan")

        assert "error" in result
        assert "butina" in result["error"] and "kmedoids" in result["error"]

    def test_nothing_parsable_is_an_error_not_an_empty_answer(self):
        result = sa.cluster_molecules(["not a molecule", ""])

        assert "error" in result

    def test_unparsable_members_are_named(self):
        result = sa.cluster_molecules(PARA_SERIES + ["@@@"])

        assert "@@@" in result["unparsed"]


class TestTheToolsAreRegistered:

    def test_both_tools_reach_the_mcp_server(self):
        registered = {}

        class _FakeMCP:
            def tool(self, *a, **k):
                def take(fn):
                    registered[fn.__name__] = fn
                    return fn
                return take

        sa.register_tools(_FakeMCP())

        assert "find_matched_molecular_pairs" in registered
        assert "cluster_molecules_by_similarity" in registered
