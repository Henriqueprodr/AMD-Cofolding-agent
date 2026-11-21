# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for 3D conformer generation and shared-volume structure reading.

Covers:
- embed_molecule_3d: molblock output, metadata, unparseable SMILES, size cap
- read_structure_file: format sniffing, content scan, path confinement,
  size refusal, missing files
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from reinvent4_mcp.tools import structures

rdkit = pytest.importorskip("rdkit", reason="RDKit comes from REINVENT4")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_COMPLEX_PDB = """\
ATOM      1  N   MET A   1      27.340  24.430   2.614  1.00  9.67           N
ATOM      2  CA  MET A   1      26.266  25.413   2.842  1.00 10.38           C
ATOM      3  N   GLY B   2      25.000  24.000   2.000  1.00  9.00           N
HETATM  100  C1  LIG A 200      10.000  10.000  10.000  1.00 20.00           C
HETATM  101  O   HOH A 300       5.000   5.000   5.000  1.00 30.00           O
END
"""


@pytest.fixture
def shared(tmp_path):
    """Point the shared root at a tmp dir."""
    root = tmp_path / "shared"
    root.mkdir()
    with patch.object(structures, "_shared_root", return_value=root):
        yield root


# ---------------------------------------------------------------------------
# SMILES -> 3D
# ---------------------------------------------------------------------------


class TestEmbedMolecule3D:
    """RDKit conformer generation -- the only local source of geometry."""

    def test_aspirin_embeds_to_a_molblock(self):
        result = structures.embed_molecule_3d("CC(=O)Oc1ccccc1C(=O)O")

        assert result["format"] == "sdf"
        assert "V2000" in result["data"]  # a real molblock
        assert result["meta"]["formula"] == "C9H8O4"
        assert result["meta"]["n_heavy_atoms"] == 13
        assert "ETKDGv3" in result["meta"]["method"]

    def test_coordinates_are_three_dimensional(self):
        result = structures.embed_molecule_3d("CCO")

        # Molblock atom lines: x, y, z in the first three columns.
        atom_lines = result["data"].splitlines()[4:7]
        zs = [float(line.split()[2]) for line in atom_lines]
        assert any(abs(z) > 1e-6 for z in zs), "all z=0 means a flat structure"

    def test_result_declares_it_is_computed_not_measured(self):
        result = structures.embed_molecule_3d("CCO")

        note = result["meta"]["note"].lower()
        assert "computed" in note
        assert "not a predicted binding pose" in note

    def test_same_seed_is_reproducible(self):
        a = structures.embed_molecule_3d("CC(=O)Oc1ccccc1C(=O)O", seed=7)
        b = structures.embed_molecule_3d("CC(=O)Oc1ccccc1C(=O)O", seed=7)

        assert a["data"] == b["data"]

    def test_hydrogens_are_optional(self):
        without = structures.embed_molecule_3d("CCO")
        with_h = structures.embed_molecule_3d("CCO", include_hydrogens=True)

        assert with_h["meta"]["n_atoms"] > without["meta"]["n_atoms"]
        assert with_h["meta"]["includes_hydrogens"] is True

    def test_unparseable_smiles_errors_without_guessing(self):
        result = structures.embed_molecule_3d("not-a-smiles")

        assert "error" in result
        assert "fetch_compound" in result["error"]
        assert "data" not in result

    def test_empty_smiles_is_refused(self):
        assert "error" in structures.embed_molecule_3d("  ")

    def test_oversized_molecule_is_refused(self):
        huge = "C" * (structures._MAX_HEAVY_ATOMS + 10)

        result = structures.embed_molecule_3d(huge)

        assert "error" in result
        assert "heavy atoms" in result["error"]


# ---------------------------------------------------------------------------
# Shared-volume reads
# ---------------------------------------------------------------------------


class TestReadStructureFile:
    """Reading what a run produced, without leaving the shared volume."""

    def test_reads_a_pdb_and_scans_its_contents(self, shared):
        (shared / "complex.pdb").write_text(_COMPLEX_PDB)

        result = structures.read_structure_file("complex.pdb")

        assert result["format"] == "pdb"
        assert "HETATM" in result["data"]
        meta = result["meta"]
        assert meta["has_protein"] is True
        assert meta["has_ligand"] is True
        # Water is not "the ligand".
        assert meta["ligand_resnames"] == ["LIG"]
        assert meta["chains"] == ["A", "B"]

    def test_accepts_a_nested_predictions_path(self, shared):
        nested = shared / "predictions" / "out_x" / "mol_0"
        nested.mkdir(parents=True)
        (nested / "mol_0_model_0.pdb").write_text(_COMPLEX_PDB)

        result = structures.read_structure_file(
            "predictions/out_x/mol_0/mol_0_model_0.pdb"
        )

        assert result["format"] == "pdb"
        assert result["meta"]["path"].endswith("mol_0_model_0.pdb")

    def test_host_style_path_is_normalised(self, shared):
        (shared / "complex.pdb").write_text(_COMPLEX_PDB)

        result = structures.read_structure_file(
            "/home/someone/project/data/shared/complex.pdb"
        )

        assert result["format"] == "pdb"

    def test_cif_and_sdf_are_recognised(self, shared):
        (shared / "x.cif").write_text("data_x\n")
        (shared / "y.sdf").write_text("mol\n$$$$\n")

        assert structures.read_structure_file("x.cif")["format"] == "cif"
        sdf = structures.read_structure_file("y.sdf")
        assert sdf["format"] == "sdf"
        assert sdf["meta"]["has_ligand"] is True

    def test_traversal_is_rejected(self, shared):
        result = structures.read_structure_file("../../etc/passwd")

        assert "error" in result
        assert "escapes the shared volume" in result["error"]

    def test_absolute_outside_path_is_rejected(self, shared):
        result = structures.read_structure_file("/etc/passwd")

        assert "error" in result

    def test_missing_file_points_at_list_shared_assets(self, shared):
        result = structures.read_structure_file("nope.pdb")

        assert "error" in result
        assert "list_shared_assets" in result["error"]

    def test_unknown_extension_is_rejected(self, shared):
        (shared / "notes.txt").write_text("hello")

        result = structures.read_structure_file("notes.txt")

        assert "error" in result
        assert "not a structure format" in result["error"]

    def test_oversized_file_is_refused_not_truncated(self, shared):
        (shared / "big.pdb").write_text("ATOM\n" * 1000)

        result = structures.read_structure_file("big.pdb", max_bytes=100)

        assert "error" in result
        assert "never truncated" in result["error"]
        assert "data" not in result
        assert result["size_bytes"] > 100

    def test_empty_path_is_refused(self, shared):
        assert "error" in structures.read_structure_file("")
