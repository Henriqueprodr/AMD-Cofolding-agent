# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the ensemble mode hands back, and the tools that feed it.

The service computes fragment families and compatible pairs but the raw
response can exceed the context window, so the tool summarises it. For two
weeks the summary counted those two lists and dropped them, which left the
mode able to report only that some existed.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

from reinvent4_mcp.tools.interaction_profiler import _summarise_hotspot

_SERVER = Path(__file__).resolve().parents[1] / "reinvent4_mcp" / "server.py"


def _decorated_tools(path: Path) -> set[str]:
    """Names of the functions carrying an ``@mcp.tool()`` decorator."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            target = dec.func if isinstance(dec, ast.Call) else dec
            if isinstance(target, ast.Attribute) and target.attr == "tool":
                found.add(node.name)
    return found


def _response(n_families: int = 2, n_pairs: int = 2) -> dict:
    return {
        "hotspots": [
            {
                "target_residue": "HIS163",
                "occupancy": 0.82,
                "n_structures_observed": 41,
                "interaction_types_observed": ["hbond"],
            },
        ],
        "fragment_families": [
            {
                "family_id": f"F{i:03d}",
                "members": [f"/shared/ens/s{i}_{k}.pdb" for k in range(3)],
                "primary_residue": "HIS163",
                "primary_interaction": "hbond",
                "occupancy": 0.5 - i * 0.01,
                "subpocket_ids": ["S1"],
                "binding_mode_label": f"HBD-His163-S1-{i}",
            }
            for i in range(n_families)
        ],
        "compatible_pairs": [
            {
                "fragment_a": f"/shared/ens/a{i}.pdb",
                "fragment_b": f"/shared/ens/b{i}.pdb",
                "overlap_residues": [],
                "compatible": True,
                "linker_distance_A": 7.5,
                "notes": None,
            }
            for i in range(n_pairs)
        ],
        "warnings": ["Hotspot ensemble: 41/50 structures processed successfully"],
        "timing_seconds": 12.5,
    }


class TestTheFragmentAnalysisSurvivesTheSummary:
    """Which fragments bind alike, and which two could be joined, is the
    question the ensemble mode exists to answer."""

    def test_families_come_back_with_their_binding_mode(self):
        summary = json.loads(_summarise_hotspot(_response()))

        families = summary["top_fragment_families"]
        assert len(families) == 2
        assert families[0]["binding_mode_label"] == "HBD-His163-S1-0"
        assert families[0]["primary_residue"] == "HIS163"
        assert families[0]["n_members"] == 3

    def test_pairs_come_back_with_their_linker_distance(self):
        summary = json.loads(_summarise_hotspot(_response()))

        pairs = summary["top_compatible_pairs"]
        assert len(pairs) == 2
        assert pairs[0]["compatible"] is True
        assert pairs[0]["linker_distance_A"] == 7.5
        assert pairs[0]["n_overlap_residues"] == 0

    def test_the_true_count_survives_the_cap(self):
        """A head is returned, so the count is the only way to know there is
        more; reporting the length of the head instead would hide it."""

        summary = json.loads(_summarise_hotspot(_response(n_families=40, n_pairs=40)))

        assert summary["fragment_families_count"] == 40
        assert summary["compatible_pairs_count"] == 40
        assert len(summary["top_fragment_families"]) == 20
        assert len(summary["top_compatible_pairs"]) == 20

    def test_an_empty_ensemble_still_summarises(self):
        summary = json.loads(_summarise_hotspot({"hotspots": [], "warnings": []}))

        assert summary["fragment_families_count"] == 0
        assert summary["top_fragment_families"] == []
        assert summary["top_compatible_pairs"] == []

    def test_the_residue_occupancy_table_is_still_there(self):
        summary = json.loads(_summarise_hotspot(_response()))

        assert summary["n_structures_ok"] == 41
        assert summary["top_50_residues_by_occupancy"][0]["target_residue"] == "HIS163"


class TestTheToolsThatFeedIt:
    """Capability that is built but never registered reaches nobody."""

    def test_measured_activities_can_be_read_into_the_potency_shape(self):
        """``profile_complex_interaction`` takes a ``potency`` dict, and this
        is the only thing that builds one. It sat unregistered, so ligand
        efficiency came back empty unless a caller wrote the dict by hand.

        Read rather than imported: the server needs an MCP version this
        checkout does not always have, and the question is only whether the
        decorator is there.
        """

        assert "ingest_fragment_activities" in _decorated_tools(_SERVER)

    def test_the_component_dictionary_and_citations_are_reachable(self):
        """MetadataClient has fetched both for its own enrichment since the
        start; neither had a tool, so an agent could read an entry's ligand
        code but not what the code stands for."""

        from unittest.mock import MagicMock

        from reinvent4_mcp.tools import rcsb

        registered: dict = {}
        mcp = MagicMock()

        def _tool():
            def _wrap(fn):
                registered[fn.__name__] = fn
                return fn
            return _wrap

        mcp.tool = _tool
        rcsb.register_tools(mcp)

        assert "rcsb_get_chem_comps" in registered
        assert "rcsb_get_pubmed" in registered
