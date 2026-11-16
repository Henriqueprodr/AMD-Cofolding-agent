"""Every tool is registered, gated, and reachable by name.

The bug this file exists for: extension tools are closures defined inside
``register_tools``, so they never appear in ``server.py``'s ``globals()``.
``run_ttp_pipeline`` looks a tool up by the name its step gives, and looking
only in ``globals()`` found 58 of 108 — none of them the scaffold derivation a
pipeline most wants to chain. The failure was silent: the step returned "not a
tool of this MCP server" and the pipeline carried on.

So the question these tests ask is whether a name that appears anywhere -- in
a gating list, in the prompt, in a pipeline step -- actually resolves to
something callable.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


class _FakeMCP:
    """Accepts registration the way FastMCP does, recording nothing itself."""

    def __init__(self) -> None:
        self.names: list[str] = []

    def tool(self, *_args, **_kwargs):
        def take(fn):
            self.names.append(fn.__name__)
            return fn

        return take


@pytest.fixture(scope="module")
def extension_tools() -> dict[str, object]:
    """Every extension tool, by name, as the registry records them."""
    from reinvent4_mcp.registry import TOOL_IMPLS, load_extensions

    load_extensions(_FakeMCP())
    return dict(TOOL_IMPLS)


@pytest.fixture(scope="module")
def server_tools() -> set[str]:
    """Tools declared at module level in server.py.

    Parsed rather than imported: the local ``mcp`` package is 2.x, where
    ``mcp.server.fastmcp`` is gone, so ``reinvent4_mcp.server`` will not
    import here.
    """
    tree = ast.parse((ROOT / "reinvent4_mcp" / "server.py").read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            attr = dec.func.attr if isinstance(dec, ast.Call) else getattr(dec, "attr", "")
            if attr == "tool":
                found.add(node.name)
    return found


@pytest.fixture(scope="module")
def every_tool(extension_tools, server_tools) -> set[str]:
    """Every name a pipeline step or a gating list may legitimately use."""
    return set(extension_tools) | server_tools


class TestThePipelineCanReachEveryTool:

    def test_the_registry_records_the_extension_tools(self, extension_tools):
        # Without this the pipeline sees only server.py's half.
        assert len(extension_tools) >= 50

    def test_the_scaffold_derivation_is_reachable_by_name(self, extension_tools):
        # The specific tool a pipeline exists to chain, and the one that was
        # unreachable: it is defined inside scaffold.register_tools.
        assert "derive_scaffold_from_series" in extension_tools

    @pytest.mark.parametrize(
        "name",
        [
            "decompose_series_r_groups",
            "find_matched_molecular_pairs",
            "cluster_molecules_by_similarity",
            "fetch_target_genetic_evidence",
            "search_patents_by_structure",
            "fetch_all_activities",
            "plan_synthesis",
            "run_generation_campaign",
            "prepare_structure",
        ],
    )
    def test_a_pipeline_step_can_name_it(self, name, every_tool):
        assert name in every_tool

    def test_what_is_recorded_is_callable(self, extension_tools):
        fn = extension_tools["derive_scaffold_from_series"]

        result = fn(smiles=[
            "CCOc1ccc(cc1)C(=O)Nc1ccccc1",
            "COc1ccc(cc1)C(=O)Nc1ccccc1",
            "Clc1ccc(cc1)C(=O)Nc1ccc(F)cc1",
        ])

        assert "[*:1]" in result["scaffold"]

    def test_the_server_half_is_still_reachable(self, server_tools):
        # globals() remains the route to server.py's own tools; the registry
        # is an addition, not a replacement.
        assert "run_ttp_pipeline" in server_tools
        assert len(server_tools) >= 55


class TestTheGatingListsNameRealTools:

    def test_every_reviewer_tool_exists(self, every_tool):
        from reinvent4_agent.coordinator import REVIEWER_READ_ONLY_TOOLS

        missing = sorted(set(REVIEWER_READ_ONLY_TOOLS) - every_tool)
        assert not missing, f"named but not registered: {missing}"

    def test_every_direct_tool_exists(self, every_tool):
        from reinvent4_agent.web.server import DIRECT_TOOLS

        missing = sorted(set(DIRECT_TOOLS) - every_tool)
        assert not missing, f"the UI may call these but they do not exist: {missing}"

    @pytest.mark.parametrize(
        "name",
        [
            "derive_scaffold_from_series",
            "decompose_series_r_groups",
            "find_matched_molecular_pairs",
            "cluster_molecules_by_similarity",
            "fetch_target_genetic_evidence",
            "fetch_target_expression",
            "fetch_pathway_annotations",
            "fetch_target_orthology",
            "search_patents_by_structure",
            "search_patents_by_target",
        ],
    )
    def test_the_new_tools_reach_both_the_agent_and_the_ui(self, name):
        from reinvent4_agent.coordinator import REVIEWER_READ_ONLY_TOOLS
        from reinvent4_agent.web.server import DIRECT_TOOLS

        assert name in REVIEWER_READ_ONLY_TOOLS
        assert name in DIRECT_TOOLS

    def test_no_new_tool_is_withheld_from_a_driver(self):
        # These are reads; a Driver denied them would have to ask for work it
        # could do itself.
        from reinvent4_agent.supervisor import DRIVER_DENIED_TOOLS, GENERAL_DENIED_TOOLS

        new = {
            "derive_scaffold_from_series", "decompose_series_r_groups",
            "find_matched_molecular_pairs", "cluster_molecules_by_similarity",
            "fetch_target_genetic_evidence", "search_patents_by_structure",
        }
        assert not new & (DRIVER_DENIED_TOOLS | GENERAL_DENIED_TOOLS)

    def test_no_tool_keeps_the_placeholder_suffix(self, every_tool):
        # `_tool` was a way round a name collision with the implementation,
        # and it leaks an implementation detail into the agent's vocabulary.
        assert not [t for t in every_tool if t.endswith("_tool")]


class TestAiZynthFinderIsAScorerNotJustAPlanner:

    def test_it_is_a_known_external_service(self):
        from reinvent4_mcp.ttp.schema import KNOWN_EXTERNAL_SERVICES

        assert "synthesis_planner" in KNOWN_EXTERNAL_SERVICES

    def test_a_scoring_objective_on_it_resolves_its_url(self):
        # Without the config entry the objective fails validation with
        # "is not a service configured here", which is how a service that
        # has had a /score endpoint all along stayed unusable as a scorer.
        from reinvent4_mcp.ttp.schema import TTP

        ttp = TTP(
            id="synth", name="synth",
            scoring={"objectives": [{
                "name": "synthesisability",
                "type": "external",
                "service": "synthesis_planner",
                "service_params": {"score_fn": "solved_steps"},
            }]},
        )

        assert ttp.scoring.objectives[0].service_url

    def test_the_service_really_has_a_score_route(self):
        app = (ROOT / "services" / "synthesis-planner" / "app.py").read_text(encoding="utf-8")

        assert '@app.post("/score")' in app


@pytest.fixture(scope="module")
def ttp_ts() -> str:
    """The UI's TTP module, read as text."""
    return (ROOT / "services" / "ui" / "frontend" / "src" / "lib" / "ttp.ts").read_text(
        encoding="utf-8"
    )


class TestTheUiOffersWhatTheBackendHas:

    @pytest.mark.parametrize(
        "fragment",
        [
            '"synthesis_planner"',
            'synthesis_planner: "AiZynthFinder"',
            'case "synthesis_planner":',
            "Synthesisability (AiZynthFinder)",
        ],
    )
    def test_aizynthfinder_is_offered_as_a_scorer(self, fragment, ttp_ts):
        assert fragment in ttp_ts

    @pytest.mark.parametrize(
        "endpoint", ["CYP1A2", "CYP3A4", "hERG", "PPB", "microsomal_half_life"]
    )
    def test_the_drug_stack_endpoints_have_a_curve(self, endpoint, ttp_ts):
        # An endpoint with no transform still appears in the picker but scores
        # raw, which silently mixes scales in the aggregate.
        assert endpoint in ttp_ts

    @pytest.mark.parametrize("endpoint", ["BCF", "IGC50", "bee_LD50", "logKoc"])
    def test_the_agro_stack_endpoints_are_offered(self, endpoint, ttp_ts):
        assert endpoint in ttp_ts

    def test_the_two_stacks_are_separate_groups(self, ttp_ts):
        # Mixed into one list the agro endpoints read as drug scorers.
        assert "ADMET (${ms})" in ttp_ts

    def test_a_non_default_stack_is_named_in_the_objective(self, ttp_ts):
        # Without model_set in service_params the scorer silently answers
        # from the drug stack, where the agro endpoints do not exist.
        assert "sp.model_set = ms" in ttp_ts
