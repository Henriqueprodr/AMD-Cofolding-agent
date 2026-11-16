# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Drug Discovery MCP Server entry point.

Exposes REINVENT4 drug design capabilities via the Model Context Protocol
using stdio transport. Provides tools for scoring molecules, discovering
available components/models, and (in later phases) running RL optimization.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Annotated, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from reinvent4_mcp import __version__
from reinvent4_mcp.config import get_config
from reinvent4_mcp.utils import ensure_reinvent4_importable

logger = logging.getLogger("reinvent4_mcp")

mcp = FastMCP(
    "DrugDiscovery",
    instructions=(
        "MCP server for de novo drug design powered by REINVENT4. "
        "Score molecules, discover scoring components and generative models, "
        "and run multi-property optimization campaigns."
    ),
)


# ---------------------------------------------------------------------------
# Tool: list_scoring_components
# ---------------------------------------------------------------------------

@mcp.tool()
def list_scoring_components() -> str:
    """List all available native REINVENT4 scoring components.

    Returns component names, types (scorer/filter/penalty), and their
    parameter schemas. Use this to discover what components can be used
    in score_molecules or TTP definitions.

    Common components include: QED, MolecularWeight, HBondDonors,
    HBondAcceptors, SlogP, NumRotBond, TPSA, CustomAlerts, SAScore,
    TanimotoSimilarity, and many more.
    """

    from reinvent4_mcp.tools.discovery import list_scoring_components as _impl

    return json.dumps(_impl(), indent=2)


# ---------------------------------------------------------------------------
# Tool: list_available_models
# ---------------------------------------------------------------------------

@mcp.tool()
def list_available_models() -> str:
    """List all registered generative model priors and their availability.

    Returns model registry keys (e.g. '.reinvent', '.libinvent'),
    expected file paths, whether the prior file exists on disk, and
    model type descriptions. Use this to check which models are
    available before launching optimization or transfer learning.
    """

    from reinvent4_mcp.tools.discovery import list_available_models as _impl

    return json.dumps(_impl(), indent=2)


# ---------------------------------------------------------------------------
# Tool: list_transforms
# ---------------------------------------------------------------------------

@mcp.tool()
def list_transforms() -> str:
    """List all available score transform types and their parameters.

    Transforms map raw component scores to the 0-1 range for aggregation.
    Common transforms: sigmoid, reverse_sigmoid, double_sigmoid, step,
    right_step, left_step.
    """

    from reinvent4_mcp.tools.discovery import list_transforms as _impl

    return json.dumps(_impl(), indent=2)


# ---------------------------------------------------------------------------
# Tool: list_external_services
# ---------------------------------------------------------------------------

@mcp.tool()
def list_external_services() -> str:
    """Check health and capabilities of configured external services.

    For each enabled service (Boltz, PoseBusters, ADMET, RetroDMTA) sends a
    GET /health and reports the URL and status ('ok'/'unhealthy'/'unreachable').

    For the multi-endpoint predictors it also reports the selector the caller
    MUST supply so scoring does not silently fall back to a default:
      * admet     -> 'endpoints' (pick one via endpoint / predict_admet)
      * retrodmta -> 'targets'   (pick one via target / predict_retrodmta)

    Call this before authoring a TTP external objective or using
    predict_admet / predict_retrodmta so you use valid endpoint/target names.
    """

    from reinvent4_mcp.tools.external import list_external_services as _impl

    return json.dumps(_impl(), indent=2)


# ---------------------------------------------------------------------------
# Tool: predict_admet
# ---------------------------------------------------------------------------

@mcp.tool()
def predict_admet(
    smiles: Annotated[
        list[str],
        Field(description="SMILES strings to predict ADMET properties for"),
    ],
    endpoints: Annotated[
        list[str] | None,
        Field(
            default=None,
            description=(
                "Which ADMET endpoints to return (subset of the service's "
                "endpoint list, e.g. ['logp', 'caco2_permeability', "
                "'aqueous_solubility']). Omit to return all endpoints. Call "
                "list_external_services to see the valid endpoint names."
            ),
        ),
    ] = None,
    model_set: Annotated[
        str,
        Field(
            default="drug",
            description=(
                "ADMET model stack: 'drug' (pharmaceutical, default) or "
                "'agro' (agrochemical). 'pharma' is accepted as an alias "
                "for 'drug' but is never emitted."
            ),
        ),
    ] = "drug",
) -> str:
    """Predict ADMET properties for molecules directly (out-of-band, not RL).

    Use this to answer property questions ("what is the permeability/logP/
    solubility of these molecules?"). Returns per-molecule mean/std for each
    requested endpoint plus the applicability-domain pass flag. For RL scoring,
    use a TTP external objective with service='admet' and
    service_params={'endpoint': '<name>'} instead.
    """

    from reinvent4_mcp.tools.external import predict_admet as _impl

    return json.dumps(
        _impl(smiles, endpoints, model_set=model_set),
        indent=2,
        default=_json_default,
    )


# ---------------------------------------------------------------------------
# Tool: predict_retrodmta
# ---------------------------------------------------------------------------

@mcp.tool()
def predict_retrodmta(
    smiles: Annotated[
        list[str],
        Field(description="SMILES strings to predict RetroDMTA endpoints for"),
    ],
    target: Annotated[
        str,
        Field(
            description=(
                "Project target to predict for (required), e.g. 'FXA'. Call "
                "list_external_services to see the valid target names."
            ),
        ),
    ],
    window: Annotated[
        str,
        Field(default="latest", description="Temporal window index or 'latest'"),
    ] = "latest",
    method: Annotated[
        str,
        Field(
            default="ensemble",
            description="Model: ensemble | xgboost | chemprop | chemeleon",
        ),
    ] = "ensemble",
) -> str:
    """Predict RetroDMTA project endpoints for a target directly (out-of-band).

    ``target`` is required; without it the service cannot know which model to
    use. Returns per-molecule per-endpoint mean/std for that target. For RL
    scoring, use a TTP external objective with service='retrodmta' and
    service_params={'target': '<name>'} instead.
    """

    from reinvent4_mcp.tools.external import predict_retrodmta as _impl

    return json.dumps(
        _impl(smiles, target, window, method), indent=2, default=_json_default
    )


# ---------------------------------------------------------------------------
# Tool: profile_interactions
# ---------------------------------------------------------------------------

@mcp.tool()
def profile_interactions(
    smiles: Annotated[
        list[str] | None,
        Field(
            default=None,
            description=(
                "Ligand SMILES to profile. SMILES-only returns the medchem "
                "taxonomy (named functional-group / ring / scaffold fragments). "
                "Pair index-wise with structure_paths to also get interactions."
            ),
        ),
    ] = None,
    structure_paths: Annotated[
        list[str] | None,
        Field(
            default=None,
            description=(
                "Protein-ligand structure files (PDB/mmCIF), resolved under the "
                "service shared dir. Yields typed interactions (H-bond, ionic, "
                "hydrophobic, pi-stacking, ...) with residues/atoms/geometry. "
                "Index-aligned with smiles when both are given."
            ),
        ),
    ] = None,
) -> str:
    """Profile medchem fragments and/or protein-ligand interactions (out-of-band).

    Use this when the user wants to understand what a ligand IS (fragments, from
    SMILES) or what it's DOING in a pocket (typed interactions, from a structure /
    cofold). Not the RL loop: to reward interactions during optimization, add a
    TTP external objective with service='interaction_profiler' and
    service_params={'interaction_type': 'hbond'} (chained after a structure
    producer like Boltz via depends_on).
    """

    from reinvent4_mcp.tools.external import profile_interactions as _impl

    return json.dumps(
        _impl(smiles, structure_paths), indent=2, default=_json_default
    )


# ---------------------------------------------------------------------------
# Tool: ingest_fragment_activities
# ---------------------------------------------------------------------------

@mcp.tool()
def ingest_fragment_activities(
    path: Annotated[
        str,
        Field(
            description=(
                "Shared-volume path to a CSV of measured activities, one row "
                "per compound."
            ),
        ),
    ],
    smiles_col: Annotated[
        str,
        Field(default="std_smiles", description="Column holding the SMILES."),
    ] = "std_smiles",
    value_col: Annotated[
        str,
        Field(default="activity_value", description="Column holding the numeric value."),
    ] = "activity_value",
    unit_col: Annotated[
        str,
        Field(default="activity_unit", description="Column holding the unit, e.g. 'nM'."),
    ] = "activity_unit",
    type_col: Annotated[
        str,
        Field(default="activity_type", description="Column holding the type, e.g. 'IC50'."),
    ] = "activity_type",
    id_col: Annotated[
        str,
        Field(default="compound_id", description="Column holding the compound id."),
    ] = "compound_id",
) -> str:
    """Read measured activities from a CSV into the shape ``potency`` expects.

    Returns ``{n_loaded, n_no_value, activities}``, where ``activities`` is
    keyed by canonical SMILES and can be handed straight to
    ``profile_complex_interaction(potency=...)`` so ligand efficiency is
    computed against real numbers rather than left empty.
    """

    from reinvent4_mcp.tools.external import ingest_fragment_activities as _impl

    return json.dumps(
        _impl(path, smiles_col, value_col, unit_col, type_col, id_col),
        indent=2,
        default=_json_default,
    )


# ---------------------------------------------------------------------------
# Tool: score_molecules
# ---------------------------------------------------------------------------

@mcp.tool()
def score_molecules(
    smiles: Annotated[
        list[str],
        Field(description="List of SMILES strings to score"),
    ],
    components: Annotated[
        list[dict[str, Any]] | None,
        Field(
            default=None,
            description=(
                "Scoring components to evaluate (direct mode). Each dict must "
                "have 'type' (e.g. 'QED', 'MolecularWeight'). Optional keys: "
                "'name', 'weight' (float, default 1.0), 'transform' (dict with "
                "'type' and params), 'params' (dict of component-specific "
                "settings, use plain values -- per-endpoint wrapping is "
                "automatic, e.g. {'smiles': ['CCO'], 'radius': 3}). "
                "Omit when using ttp_id or ttp."
            ),
        ),
    ] = None,
    aggregation: Annotated[
        str,
        Field(
            default="geometric_mean",
            description=(
                "Aggregation method: 'geometric_mean' or 'arithmetic_mean'. "
                "Ignored in TTP mode (uses the TTP's aggregation setting)."
            ),
        ),
    ] = "geometric_mean",
    ttp_id: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "ID of a stored TTP to use for scoring. "
                "When external services (Boltz, PoseBusters) are healthy, "
                "all objectives are scored with pipeline artifact passing. "
                "Otherwise only native components are scored."
            ),
        ),
    ] = None,
    ttp: Annotated[
        dict[str, Any] | None,
        Field(
            default=None,
            description=(
                "Inline TTP definition (dict). Alternative to ttp_id. "
                "Use create_ttp to validate and persist TTPs."
            ),
        ),
    ] = None,
) -> str:
    """Score a batch of molecules using REINVENT4 scoring components.

    Two modes:
    1. Direct: pass 'components' list (e.g. [{"type": "QED", "weight": 0.5}])
    2. TTP-based: pass 'ttp_id' (stored) or 'ttp' (inline TTP dict).
       When external services are healthy, all objectives (including Boltz
       affinity and PoseBusters pose quality) are scored with pipeline
       artifact passing. Otherwise only native components are scored.

    Returns per-molecule component breakdowns (raw + transformed),
    summary statistics, and scoring_mode ('full' or 'native_only').

    In TTP mode every external objective runs through the same
    ExternalRESTScorer class, so read 'mapper_summary.external_components'
    (and each 'summary.per_component[name].source') to see which service,
    URL and selector -- e.g. admet endpoint, boltz affinity_mode -- produced
    a given score.

    Example (direct mode):
    [
      {"type": "QED", "weight": 0.5},
      {"type": "MolecularWeight", "weight": 0.3,
       "transform": {"type": "double_sigmoid", "high": 500, "low": 200,
                      "coef_div": 500, "coef_si": 20, "coef_se": 20}}
    ]
    """

    from reinvent4_mcp.tools.scoring import score_molecules as _impl

    ttp_dict = _ensure_dict(ttp) if ttp is not None else None
    result = _impl(smiles, components, aggregation, ttp_id=ttp_id, ttp=ttp_dict)
    return json.dumps(result, indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: create_ttp
# ---------------------------------------------------------------------------

@mcp.tool()
def create_ttp(
    ttp: Annotated[
        dict[str, Any],
        Field(
            description=(
                "Full TTP definition (JSON object). No extra fields allowed "
                "at any level -- unknown keys cause validation errors.\n"
                "\n"
                "Top-level fields:\n"
                "  id (str, required): lowercase slug, e.g. 'jak2-lead-opt'\n"
                "  name (str, required): human-readable name\n"
                "  scoring (object, required): see below\n"
                "  pre_scoring (object, optional): fast drug-likeness descriptors\n"
                "  target (object, optional): protein info for Boltz scoring\n"
                "  constraints (list, optional): structural filters\n"
                "  generation (object, optional): which algorithm builds the\n"
                "    molecules; omit for a plain de novo run\n"
                "\n"
                "IMPORTANT: 'name' must be unique across ALL pre_scoring\n"
                "components and scoring objectives combined.\n"
                "\n"
                "pre_scoring structure:\n"
                "  {components: [{name, component, weight?, transform?}, ...]}\n"
                "  'component' must be one of: QED, MolecularWeight, HBondDonors,\n"
                "  HBondAcceptors, SlogP, NumRotBond.\n"
                "\n"
                "scoring structure:\n"
                "  {aggregation?: 'geometric_mean'|'arithmetic_mean',\n"
                "   objectives: [{name, type, component?, weight?, transform?,\n"
                "                 params?, ...}]}\n"
                "  'type' MUST be exactly 'native' or 'external'.\n"
                "  Native objectives require 'component' (REINVENT4 class name).\n"
                "  Native objectives may include 'params' (dict) for component-\n"
                "  specific settings (e.g. reference SMILES for TanimotoSimilarity).\n"
                "  External objectives require 'service'; 'service_url' only for\n"
                "  a service not listed by list_external_services.\n"
                "\n"
                "constraints structure (list of objects):\n"
                "  [{type: 'custom_alerts', smarts: ['[pattern]', ...]}, ...]\n"
                "\n"
                "generation structure:\n"
                "  {mode: 'reinvent'|'libinvent'|'linkinvent'|'mol2mol',\n"
                "   model?, batch_size?, n_steps?, seed?, device?,\n"
                "   libinvent_input?: {scaffold, decoration_positions?},\n"
                "   linkinvent_input?: {fragments, min_linker_length?,\n"
                "                       max_linker_length?},\n"
                "   mol2mol_input?: {seed_smiles}}\n"
                "  Each attachment point needs a map number -- [*:1], not [*].\n"
                "  A libinvent or linkinvent profile is started with\n"
                "  run_generation_campaign; run_optimization cannot build it.\n"
                "\n"
                "transform structure (on any component/objective):\n"
                "  {type: 'sigmoid'|'reverse_sigmoid'|'double_sigmoid'|'step'|\n"
                "   'right_step'|'left_step'|'value_mapping'|'exponential_decay',\n"
                "   ...type-specific params (low, high, k, etc.)}\n"
            ),
        ),
    ],
) -> str:
    """Create and persist a new Therapeutic Target Profile (TTP).

    A TTP defines the full multi-property optimization objective:
    drug-likeness descriptors (pre-scoring), scoring objectives
    (native REINVENT4 components or external REST services), and
    structural constraints.

    Example TTP with QED and SA score objectives:

        {
          "id": "lead-opt-example",
          "name": "Lead Optimization (QED + SA)",
          "scoring": {
            "aggregation": "geometric_mean",
            "objectives": [
              {
                "name": "QED Score",
                "type": "native",
                "component": "QED",
                "weight": 1.0
              },
              {
                "name": "SA Score",
                "type": "native",
                "component": "SAScore",
                "weight": 0.8,
                "transform": {
                  "type": "reverse_sigmoid",
                  "low": 1.0, "high": 4.0, "k": 0.4
                }
              }
            ]
          }
        }

    Note: pre_scoring components and scoring objectives share the same
    name namespace -- every 'name' must be unique across both sections.

    Returns the created TTP with its version number, or an error if
    validation fails or the ID already exists.
    """

    from reinvent4_mcp.tools.ttp import create_ttp as _impl

    return json.dumps(_impl(_ensure_dict(ttp)), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: update_ttp
# ---------------------------------------------------------------------------

@mcp.tool()
def update_ttp(
    ttp_id: Annotated[
        str,
        Field(description="ID of the TTP to update"),
    ],
    updates: Annotated[
        dict[str, Any],
        Field(
            description=(
                "Partial update to apply. Uses the same schema as "
                "create_ttp -- provide only the fields you want to change. "
                "Nested dicts are merged recursively. Version is "
                "auto-incremented. The same validation rules apply: "
                "no extra fields, type must be 'native' or 'external', "
                "constraints must be a list, etc."
            ),
        ),
    ],
) -> str:
    """Update an existing TTP with partial changes.

    Merges the updates into the current TTP, validates the result,
    and auto-increments the version. Both the updated current file
    and a versioned snapshot are saved.

    Example -- change an objective's weight and add a constraint:

        ttp_id: "jak2-lead-opt"
        updates: {
          "scoring": {
            "objectives": [
              {"name": "SA Score", "type": "native",
               "component": "SAScore", "weight": 0.5}
            ]
          },
          "constraints": [
            {"type": "custom_alerts", "smarts": ["[#7]=[#7]"]}
          ]
        }

    Tip: call get_ttp first to see the current structure.
    """

    from reinvent4_mcp.tools.ttp import update_ttp as _impl

    return json.dumps(_impl(ttp_id, _ensure_dict(updates)), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: get_ttp
# ---------------------------------------------------------------------------

@mcp.tool()
def get_ttp(
    ttp_id: Annotated[
        str,
        Field(description="ID of the TTP to retrieve"),
    ],
    version: Annotated[
        int | None,
        Field(
            default=None,
            description=(
                "Specific version to retrieve. "
                "Omit to get the latest version."
            ),
        ),
    ] = None,
) -> str:
    """Retrieve a stored TTP by ID, optionally at a specific version.

    Returns the full TTP definition including all objectives, weights,
    transforms, target info, and constraints. Use list_ttps to see
    available TTP IDs.
    """

    from reinvent4_mcp.tools.ttp import get_ttp as _impl

    return json.dumps(_impl(ttp_id, version), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: list_ttps
# ---------------------------------------------------------------------------

@mcp.tool()
def list_ttps() -> str:
    """List all stored TTPs with summary metadata.

    Returns ID, name, and current version for each TTP.
    Use get_ttp to retrieve the full definition.
    """

    from reinvent4_mcp.tools.ttp import list_ttps as _impl

    return json.dumps(_impl(), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: delete_ttp
# ---------------------------------------------------------------------------

@mcp.tool()
def delete_ttp(
    ttp_id: Annotated[
        str,
        Field(description="ID of the TTP to delete"),
    ],
) -> str:
    """Delete a stored TTP and all its version history.

    This is irreversible. The TTP and all versioned snapshots are removed.
    Prefer update_ttp for corrections and campaign retries so provenance is
    preserved. Delete only when abandoning the identifier.
    """

    from reinvent4_mcp.tools.ttp import delete_ttp as _impl

    return json.dumps(_impl(ttp_id), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: run_optimization
# ---------------------------------------------------------------------------

@mcp.tool()
def run_optimization(
    ttp_id: Annotated[
        str | None,
        Field(
            default=None,
            description="ID of a stored TTP to optimise against.",
        ),
    ] = None,
    ttp: Annotated[
        dict[str, Any] | None,
        Field(
            default=None,
            description="Inline TTP definition (alternative to ttp_id).",
        ),
    ] = None,
    model: Annotated[
        str,
        Field(
            default=".reinvent",
            description=(
                "Prior model: registry key (e.g. '.reinvent') or file path. "
                "Use list_available_models to see options."
            ),
        ),
    ] = ".reinvent",
    max_steps: Annotated[
        int,
        Field(
            default=300,
            description="Maximum RL steps for this campaign.",
        ),
    ] = 300,
    batch_size: Annotated[
        int | None,
        Field(
            default=None,
            description=(
                "Molecules generated per RL step. Overrides process config. "
                "Lower values reduce Boltz cost per step."
            ),
        ),
    ] = None,
    sigma: Annotated[
        int | None,
        Field(default=None, description="DAP sigma (reward sharpness). Overrides process config."),
    ] = None,
    learning_rate: Annotated[
        float | None,
        Field(default=None, description="Optimizer learning rate. Overrides process config."),
    ] = None,
    resume_from: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Checkpoint path to resume from. Use the checkpoint_path "
                "returned by stop_optimization or get_optimization_status."
            ),
        ),
    ] = None,
    termination: Annotated[
        dict[str, Any] | None,
        Field(
            default=None,
            description="Early termination criteria (e.g. convergence threshold).",
        ),
    ] = None,
    seed_smiles: Annotated[
        list[str] | None,
        Field(
            default=None,
            description=(
                "List of SMILES strings used as input seeds for Mol2Mol "
                "priors. Required when model is a Mol2Mol variant "
                "(e.g. '.m2m_medium'). Ignored for "
                "Reinvent priors."
            ),
        ),
    ] = None,
) -> str:
    """Launch an RL optimization campaign using a TTP.

    Composes a Scorer from the TTP's full objective set (pre-scoring
    descriptors + native components + external REST services like Boltz
    affinity and PoseBusters pose validation), generates a REINVENT4
    config, and starts a background RL process.

    Monitor progress with get_optimization_status. Stop with
    stop_optimization. Resume from checkpoint by passing the
    checkpoint_path to resume_from.

    Workflow:
    1. run_optimization(ttp_id="jak2-lead", max_steps=300)
    2. get_optimization_status(job_id)  -- poll periodically
    3. stop_optimization(job_id)  -- if intervention needed
    4. update_ttp("jak2-lead", {...})  -- adjust weights/objectives
    5. run_optimization(ttp_id="jak2-lead", resume_from="<checkpoint>")
    """

    from reinvent4_mcp.tools.optimization import run_optimization as _impl

    ttp_dict = _ensure_dict(ttp) if ttp is not None else None
    result = _impl(
        ttp_id=ttp_id,
        ttp=ttp_dict,
        model=model,
        max_steps=max_steps,
        batch_size=batch_size,
        sigma=sigma,
        learning_rate=learning_rate,
        resume_from=resume_from,
        termination=termination,
        seed_smiles=seed_smiles,
    )
    return json.dumps(result, indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: get_optimization_status
# ---------------------------------------------------------------------------

@mcp.tool()
def get_optimization_status(
    job_id: Annotated[
        str,
        Field(description="ID of the optimization job to query."),
    ],
) -> str:
    """Monitor a running or completed optimization job with convergence diagnostics.

    Returns the current state, step count, score trajectory,
    per-component statistics, and automated convergence diagnostics
    with actionable warnings and recommendations.

    Key fields for agent decision-making:
    - trajectory.current_step / max_steps: progress
    - trajectory.mean_total_score: overall performance
    - trajectory.latest_mean_score: recent performance (plateau detection)
    - trajectory.per_component: per-component summary stats (mean/max/min)
    - trajectory.validity_rate: model health
    - trajectory.score_trajectory: step-by-step total score trend
    - trajectory.per_component_trajectory: step-by-step trend per component

    Convergence diagnostics (trajectory.diagnostics):
    - convergence_status: 'early' | 'improving' | 'plateau' | 'diverging' | 'stable' | 'completed'
    - plateau_detected: whether the score has stopped improving
    - diverging: whether scores are declining
    - improvement_rate: score change per step (recent window)
    - trend: early_mean, mid_mean, recent_mean, overall_improvement
    - component_health: per-component NaN rates and status ('healthy' | 'degraded' | 'failing')
    - warnings: list of actionable warnings (low_validity, score_plateau, component_failure, etc.)
    - recommendations: list of suggested actions for the agent

    Poll this every few minutes during a run. Use diagnostics.convergence_status
    and diagnostics.warnings to decide when to stop and adjust the TTP.
    """

    from reinvent4_mcp.tools.results import get_optimization_status as _impl

    return json.dumps(_impl(job_id), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: stop_optimization
# ---------------------------------------------------------------------------

@mcp.tool()
def stop_optimization(
    job_id: Annotated[
        str,
        Field(description="ID of the optimization job to stop."),
    ],
) -> str:
    """Stop a running optimization job and save a checkpoint.

    Sends a graceful interrupt to the REINVENT4 process, which saves
    a checkpoint file. The checkpoint can be used with run_optimization's
    resume_from parameter to continue the run after TTP adjustments.

    Returns the checkpoint path and final status.
    """

    from reinvent4_mcp.tools.optimization import stop_optimization as _impl

    return json.dumps(_impl(job_id), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: run_transfer_learning
# ---------------------------------------------------------------------------

@mcp.tool()
def run_transfer_learning(
    model: Annotated[
        str,
        Field(
            default=".reinvent",
            description=(
                "Prior model to fine-tune: registry key (e.g. '.reinvent') "
                "or file path. Use list_available_models to see options."
            ),
        ),
    ] = ".reinvent",
    smiles: Annotated[
        list[str] | None,
        Field(
            default=None,
            description=(
                "Inline list of SMILES strings for training. "
                "Provide either this or smiles_file, not both."
            ),
        ),
    ] = None,
    smiles_file: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Path to a .smi file with training SMILES (one per line). "
                "Provide either this or smiles, not both."
            ),
        ),
    ] = None,
    num_epochs: Annotated[
        int,
        Field(default=50, description="Number of training epochs."),
    ] = 50,
    batch_size: Annotated[
        int,
        Field(default=128, description="Training batch size."),
    ] = 128,
    save_every_n_epochs: Annotated[
        int,
        Field(default=5, description="Save a checkpoint every N epochs."),
    ] = 5,
    sample_batch_size: Annotated[
        int,
        Field(
            default=2000,
            description="Molecules sampled for validation metrics.",
        ),
    ] = 2000,
    validation_smiles: Annotated[
        list[str] | None,
        Field(
            default=None,
            description="Inline list of SMILES for validation (optional).",
        ),
    ] = None,
    validation_smiles_file: Annotated[
        str | None,
        Field(
            default=None,
            description="Path to a validation .smi file (optional).",
        ),
    ] = None,
    standardize_smiles: Annotated[
        bool,
        Field(default=True, description="Standardize SMILES before training."),
    ] = True,
    randomize_smiles: Annotated[
        bool,
        Field(default=True, description="Randomize SMILES representations."),
    ] = True,
    output_model_file: Annotated[
        str,
        Field(
            default="tl_model.model",
            description="Base name for the output model file.",
        ),
    ] = "tl_model.model",
) -> str:
    """Fine-tune a generative model on a SMILES dataset via transfer learning.

    Transfer learning adapts a prior model to produce molecules similar
    to the provided SMILES set. The resulting model can then be used as
    the starting agent for run_optimization (RL).

    Workflow:
    1. run_transfer_learning(model=".reinvent", smiles=[...], num_epochs=50)
    2. get_optimization_status(job_id)  -- monitor TL progress
    3. Use the output model in run_optimization(model="<output_model_path>")

    Epoch-level checkpoints are saved as <output_model_file>.<epoch>.chkpt
    in the job working directory. Choose the best epoch based on
    validation NLL (lower is better).
    """

    from reinvent4_mcp.tools.optimization import run_transfer_learning as _impl

    result = _impl(
        model=model,
        smiles=smiles,
        smiles_file=smiles_file,
        num_epochs=num_epochs,
        batch_size=batch_size,
        save_every_n_epochs=save_every_n_epochs,
        sample_batch_size=sample_batch_size,
        validation_smiles=validation_smiles,
        validation_smiles_file=validation_smiles_file,
        standardize_smiles=standardize_smiles,
        randomize_smiles=randomize_smiles,
        output_model_file=output_model_file,
    )
    return json.dumps(result, indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: get_top_candidates
# ---------------------------------------------------------------------------

@mcp.tool()
def get_top_candidates(
    job_id: Annotated[
        str | None,
        Field(
            default=None,
            description="ID of an optimization job to extract candidates from.",
        ),
    ] = None,
    csv_path: Annotated[
        str | None,
        Field(
            default=None,
            description="Direct path to a REINVENT4 output CSV (alternative to job_id).",
        ),
    ] = None,
    top_n: Annotated[
        int,
        Field(default=50, description="Number of top candidates to return."),
    ] = 50,
    min_score: Annotated[
        float | None,
        Field(default=None, description="Minimum total score threshold."),
    ] = None,
) -> str:
    """Extract the best molecules from an optimization run.

    Reads the output CSV, filters for valid molecules above the
    minimum score, deduplicates by SMILES, and returns the top-N
    ranked by total score with per-component breakdowns.

    Use after a run completes or during a run to peek at intermediate
    results.
    """

    from reinvent4_mcp.tools.results import get_top_candidates as _impl

    return json.dumps(
        _impl(job_id=job_id, csv_path=csv_path, top_n=top_n, min_score=min_score),
        indent=2,
        default=_json_default,
    )


# ---------------------------------------------------------------------------
# Tool: read_results
# ---------------------------------------------------------------------------

@mcp.tool()
def read_results(
    csv_path: Annotated[
        str,
        Field(description="Path to a REINVENT4 output CSV file."),
    ],
) -> str:
    """Parse a REINVENT4 output CSV and return summary statistics.

    Provides n_rows, unique SMILES count, validity rate,
    total score statistics, and per-component averages.
    Useful for analysing results from any REINVENT4 run without
    a tracked job ID.
    """

    from reinvent4_mcp.tools.results import read_results as _impl

    return json.dumps(_impl(csv_path), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: get_job_alerts
# ---------------------------------------------------------------------------

@mcp.tool()
def get_job_alerts() -> str:
    """Check all running jobs for problems that may need intervention.

    Returns alerts for: job failures, scoring service errors,
    score plateaus, divergence, component failures, validity drops,
    and stalled progress. Each alert includes severity and a
    recommended action.

    Call this periodically during long optimization runs.
    """

    from reinvent4_mcp.jobs.manager import get_job_manager

    return json.dumps(get_job_manager().get_alerts(), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: list_jobs
# ---------------------------------------------------------------------------

@mcp.tool()
def list_jobs() -> str:
    """List all tracked optimization jobs with their current status.

    Returns job IDs, states (queued/running/completed/failed/stopped),
    TTP IDs, and creation times. Use get_optimization_status for
    detailed metrics on a specific job.
    """

    from reinvent4_mcp.jobs.manager import get_job_manager

    manager = get_job_manager()
    return json.dumps(manager.list_jobs(), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: list_shared_assets
# ---------------------------------------------------------------------------

@mcp.tool()
def list_shared_assets(prefix: str = "", limit: int = 10) -> str:
    """List usable files on the shared volume (structures, MSAs, cofold outputs).

    Use this to GROUND any path before profiling or cofolding -- never guess or
    invent a file path. Returns categorized, /shared-relative paths:
    - ``structures``: hand-placed protein/complex PDB/mmCIF files (feed to
      ``profile_interactions`` structure_paths).
    - ``msas``: precomputed alignments (.a3m/.fasta/...) usable as a TTP
      target ``msa_path``.
    - ``cofold_outputs``: structures produced by Boltz (under
      ``predictions/``), newest first.

    At most ``limit`` paths are named per category, because the volume holds
    one cofold output per predicted molecule and naming thousands of them
    overflows the context. ``counts`` is always the true total, and
    ``not_listed`` says how many were withheld. Pass ``prefix`` to reach one
    job's files (``predictions/out_035c34d9``) or raise ``limit``.

    Returns an empty (non-error) result when nothing is present.
    """

    from reinvent4_mcp.tools.external import list_shared_assets as _impl

    return json.dumps(
        _impl(prefix=prefix or None, limit=limit), indent=2, default=_json_default
    )


# ---------------------------------------------------------------------------
# Tool: read_shared_file
# ---------------------------------------------------------------------------

@mcp.tool()
def read_shared_file(
    path: Annotated[str, Field(description="Shared-volume relative path (e.g. 'mpro_fbdd/fragments.csv')")],
    max_bytes: Annotated[int, Field(default=131072, description="Maximum bytes to return (default 128 KB).")] = 131072,
) -> str:
    """Read a text file from the shared volume and return its contents.

    Use this to read CSV manifests, FASTA files, SMILES lists, and similar
    text assets staged under /shared/. Paths are resolved relative to the
    shared-volume root (/shared/) and are sandboxed to that directory.

    Returns the file contents as a string, or an error dict if the file
    cannot be read.
    """
    import os  # noqa: PLC0415

    shared_root = os.environ.get("IP_SHARED_DIR", "/shared")
    p = path.strip()
    if os.path.isabs(p):
        abs_path = p
    else:
        abs_path = os.path.join(shared_root, p)

    # Sandbox: must stay inside shared_root
    abs_path = os.path.realpath(abs_path)
    if not abs_path.startswith(os.path.realpath(shared_root)):
        return json.dumps({"error": f"Path escapes shared volume: {path}"})

    try:
        size = os.path.getsize(abs_path)
        with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
            content = fh.read(max_bytes)
        truncated = size > max_bytes
        return json.dumps({
            "path": path,
            "size_bytes": size,
            "truncated": truncated,
            "content": content,
        })
    except FileNotFoundError:
        return json.dumps({"error": f"File not found: {path}"})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Tool: parse_ligand_file
# ---------------------------------------------------------------------------

@mcp.tool()
def parse_ligand_file(
    path: Annotated[str, Field(description=(
        "Shared-volume path to an SDF, MOL, or MOL2 file. "
        "Example: 'structure-prep/abc123/prepared/ligand_LIG_A_1001.sdf'"
    ))],
    max_mols: Annotated[int, Field(default=10, description=(
        "Maximum number of molecules to return from a multi-molecule SDF. "
        "Set to 1 for single-ligand files."
    ))] = 10,
) -> str:
    """Parse a ligand file (SDF / MOL / MOL2) and return structured data.

    Extracts SMILES, molecular properties and any SDF property blocks
    without requiring the agent to manually parse the molfile format.

    Returns a list of molecules, each with:
      - smiles: canonical RDKit SMILES
      - inchikey: standard InChIKey
      - n_heavy_atoms: heavy atom count
      - mw: molecular weight
      - formula: molecular formula
      - n_confs: number of 3D conformers in file
      - properties: dict of any >PROPERTY< tags from SDF

    Use this instead of read_shared_file when you need SMILES or
    molecular properties from a prepared ligand SDF artifact.
    """
    import os  # noqa: PLC0415

    shared_root = os.environ.get("IP_SHARED_DIR", "/shared")
    p = path.strip()
    if not os.path.isabs(p):
        abs_path = os.path.join(shared_root, p)
    else:
        abs_path = p

    abs_path = os.path.realpath(abs_path)
    if not abs_path.startswith(os.path.realpath(shared_root)):
        return json.dumps({"error": f"Path escapes shared volume: {path}"})
    if not os.path.isfile(abs_path):
        return json.dumps({"error": f"File not found: {path}"})

    try:
        from rdkit import Chem  # noqa: PLC0415
        from rdkit.Chem import Descriptors, AllChem  # noqa: PLC0415

        ext = os.path.splitext(abs_path)[1].lower()

        if ext in (".sdf", ".mol"):
            supplier = Chem.SDMolSupplier(abs_path, removeHs=True, sanitize=True)
            mols_raw = [m for m in supplier if m is not None]
            prop_suppliers = list(supplier)  # to access SDF properties
        elif ext == ".mol2":
            mol = Chem.MolFromMol2File(abs_path, removeHs=True, sanitize=True)
            mols_raw = [mol] if mol is not None else []
            prop_suppliers = mols_raw
        else:
            return json.dumps({"error": f"Unsupported format: {ext}. Use .sdf, .mol, or .mol2"})

        results = []
        for i, mol in enumerate(mols_raw[:max_mols]):
            if mol is None:
                continue
            # Sanitize and clean stereo
            Chem.SanitizeMol(mol)
            AllChem.AssignStereochemistry(mol, cleanIt=True, force=True)
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            inchikey = Chem.MolToInchiKey(Chem.MolFromSmiles(smiles) or mol)
            mw = round(Descriptors.MolWt(mol), 2)
            formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
            n_ha = mol.GetNumHeavyAtoms()
            n_confs = mol.GetNumConformers()

            # SDF property tags
            props = {}
            if hasattr(mol, "GetPropsAsDict"):
                props = {k: v for k, v in mol.GetPropsAsDict().items()
                         if not k.startswith("_")}

            results.append({
                "mol_index": i,
                "smiles": smiles,
                "inchikey": inchikey,
                "n_heavy_atoms": n_ha,
                "mw": mw,
                "formula": formula,
                "n_confs": n_confs,
                "properties": props,
            })

        n_total = len(mols_raw)
        return json.dumps({
            "path": path,
            "format": ext.lstrip(".").upper(),
            "n_mols_in_file": n_total,
            "n_returned": len(results),
            "truncated": n_total > max_mols,
            "molecules": results,
        }, indent=2)

    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})

@mcp.tool()
def fetch_target_info(
    query: Annotated[
        str,
        Field(
            description=(
                "A UniProt accession (e.g. 'O60885' for BRD4) or free text "
                "such as a gene name plus organism (e.g. 'BRD4 human')."
            ),
        ),
    ],
) -> str:
    """Look up a protein target on UniProt (canonical sequence, PDBs, function).

    Use this whenever a target is named but no sequence is provided, instead of
    recalling a sequence from memory. Returns the primary accession, protein/
    gene names, organism, the canonical amino-acid sequence (drop straight into
    a TTP ``target.protein_sequence``), a short function summary, and candidate
    PDB IDs. On network failure it returns an error and does NOT fabricate a
    sequence.
    """

    from reinvent4_mcp.tools.uniprot import fetch_target_info as _impl

    return json.dumps(_impl(query), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: fetch_compound
# ---------------------------------------------------------------------------

@mcp.tool()
def fetch_compound(
    name: Annotated[
        str,
        Field(
            description=(
                "Compound name or synonym to resolve to a real SMILES via "
                "PubChem (e.g. 'JQ1', 'aspirin', 'acetazolamide', 'imatinib')."
            ),
        ),
    ],
) -> str:
    """Resolve a molecule NAME to a real SMILES from PubChem.

    Use this whenever a molecule is referred to by name instead of a SMILES.
    NEVER recall a SMILES from memory - hallucinated SMILES are frequently
    invalid (RDKit cannot parse or 3D-embed them) and break cofolding. Returns
    the isomeric ``smiles`` (use it directly for score_molecules /
    profile_interactions), plus CID, formula, IUPAC name and InChIKey. On
    failure it returns an error and does not fabricate a structure - ask the
    user to paste the SMILES.

    When a name matches several compounds - salts, stereoisomers, mixtures -
    the result carries ``matches``, ``other_cids`` and ``ambiguous``. The
    first is returned, so check it is the one meant and call
    ``fetch_compound_by_cid`` if it is not.
    """

    from reinvent4_mcp.tools.pubchem import fetch_compound as _impl

    return json.dumps(_impl(name), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: fetch_compound_by_cid
# ---------------------------------------------------------------------------

@mcp.tool()
def fetch_compound_by_cid(
    cid: Annotated[
        int,
        Field(description="PubChem CID, which names exactly one structure."),
    ],
) -> str:
    """Fetch one PubChem compound by CID.

    This is how to resolve the ambiguity ``fetch_compound`` reports: a name
    may match a dozen compounds, a CID matches one. Returns the same fields.
    """

    from reinvent4_mcp.tools.pubchem import fetch_compound_by_cid as _impl

    return json.dumps(_impl(cid), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: lookup_compound_by_structure
# ---------------------------------------------------------------------------

@mcp.tool()
def lookup_compound_by_structure(
    smiles: Annotated[
        str,
        Field(description="SMILES to look up in PubChem by structure."),
    ],
) -> str:
    """Find out whether PubChem already knows a structure.

    Matches on the structure rather than the string, so a generated SMILES
    finds the registered compound it is the same molecule as - which is how a
    design reported as new is caught being a known compound. For a verdict
    across PubChem *and* ChEMBL at once, use ``check_compound_is_known``.
    """

    from reinvent4_mcp.tools.pubchem import fetch_compound_by_smiles as _impl

    return json.dumps(_impl(smiles), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: run_ttp_pipeline
# ---------------------------------------------------------------------------

@mcp.tool()
def run_ttp_pipeline(
    ttp_id: Annotated[
        str,
        Field(description="ID of a TTP that has a 'pipeline' section."),
    ],
    dry_run: Annotated[
        bool,
        Field(
            default=False,
            description=(
                "If true, return the resolved execution plan without calling "
                "any tool. Use to preview what the pipeline would do."
            ),
        ),
    ] = False,
) -> str:
    """Execute the preparation pipeline attached to a TTP.

    A pipeline is an ordered list of tool calls stored on the TTP. Each step
    names an MCP tool, its params, and which earlier steps it depends on.
    ``run_ttp_pipeline`` walks the topological order, calls each tool, stores
    its result under the step's ``store_as`` key, and interpolates
    ``${key}`` / ``${key.field}`` references in later steps' params before
    calling them.

    When a step returns an error and its ``condition`` is ``if_no_error``
    (the default), dependent steps are skipped and the error is surfaced. Use
    ``condition: always`` for cleanup or logging steps that must run
    regardless.

    Typical use: a structure-based campaign whose TTP pipeline holds the
    preparation steps (fetch target → prepare structure → characterize site →
    derive scaffold → ...). Opening the same TTP in a fresh session replays
    those steps with the same params rather than re-deriving the sequence.

    Call ``get_ttp`` to see the pipeline before running it.
    Use ``dry_run=true`` to preview the execution order and resolved params.
    """

    import re as _re  # noqa: PLC0415
    from reinvent4_mcp.tools.ttp import get_ttp as _get_ttp  # noqa: PLC0415

    raw = _get_ttp(ttp_id)
    if "error" in raw:
        return json.dumps(raw, indent=2)

    ttp_data = raw.get("ttp", raw)
    pipeline = ttp_data.get("pipeline")
    if not pipeline:
        return json.dumps({
            "error": f"TTP '{ttp_id}' has no pipeline section.",
            "hint": "Add a 'pipeline' field when creating or updating the TTP.",
        }, indent=2)

    from reinvent4_mcp.ttp.schema import PipelineSection  # noqa: PLC0415

    pipe = PipelineSection(**pipeline)
    steps = pipe.topological_order()

    if dry_run:
        return json.dumps({
            "ttp_id": ttp_id,
            "dry_run": True,
            "steps": [
                {"id": s.id, "tool": s.tool, "depends_on": s.depends_on,
                 "store_as": s.store_as or s.id, "params": s.params,
                 "condition": s.condition}
                for s in steps
            ],
        }, indent=2)

    def _interpolate(value: Any, ctx: dict[str, Any]) -> Any:
        """Replace '${key}' or '${key.field}' in string values."""
        if isinstance(value, str):
            # Full-match replacement so a param can be a non-string result.
            full = _re.fullmatch(r"\$\{([^}]+)\}", value)
            if full:
                ref = full.group(1)
                parts = ref.split(".", 1)
                base = ctx.get(parts[0])
                if base is None:
                    return value
                if len(parts) == 2 and isinstance(base, dict):
                    return base.get(parts[1], value)
                return base
            # Inline replacement — always a string result.
            def _sub(m: Any) -> str:
                ref = m.group(1)
                parts = ref.split(".", 1)
                base = ctx.get(parts[0])
                if base is None:
                    return m.group(0)
                if len(parts) == 2 and isinstance(base, dict):
                    return str(base.get(parts[1], m.group(0)))
                return str(base)
            return _re.sub(r"\$\{([^}]+)\}", _sub, value)
        if isinstance(value, dict):
            return {k: _interpolate(v, ctx) for k, v in value.items()}
        if isinstance(value, list):
            return [_interpolate(v, ctx) for v in value]
        return value

    from reinvent4_mcp.registry import TOOL_IMPLS  # noqa: PLC0415

    context: dict[str, Any] = {}
    results: list[dict[str, Any]] = []

    for step in steps:
        store_key = step.store_as or step.id

        # Skip if any dependency errored and condition requires success.
        if step.condition == "if_no_error":
            dep_errors = [d for d in step.depends_on if isinstance(context.get(d), dict) and "error" in context.get(d, {})]
            if dep_errors:
                result = {"skipped": True, "reason": f"dependency error in: {dep_errors}"}
                context[store_key] = result
                results.append({"step": step.id, "store_as": store_key, "result": result})
                continue

        resolved_params = _interpolate(step.params, context)
        step_result: Any = {"error": f"Tool '{step.tool}' is not a tool of this MCP server."}

        # Both halves of the tool surface: the extension tools, which are
        # closures and so reachable only through the registry that recorded
        # them, and this module's own, which are module-level functions.
        # Checked in that order because the extensions are the larger half
        # and the half a pipeline is most likely to chain.
        fn = TOOL_IMPLS.get(step.tool) or globals().get(step.tool)
        if fn is not None and callable(fn):
            try:
                raw_result = fn(**resolved_params)
                # Tools return JSON strings; parse so context values are dicts.
                if isinstance(raw_result, str):
                    try:
                        step_result = json.loads(raw_result)
                    except json.JSONDecodeError:
                        step_result = {"text": raw_result}
                else:
                    step_result = raw_result
            except Exception as exc:  # noqa: BLE001
                step_result = {"error": str(exc)}

        context[store_key] = step_result
        results.append({"step": step.id, "tool": step.tool, "store_as": store_key, "result": step_result})

    errors = [r for r in results if isinstance(r.get("result"), dict) and "error" in r["result"] and not r["result"].get("skipped")]
    return json.dumps({
        "ttp_id": ttp_id,
        "steps_run": len(results),
        "errors": len(errors),
        "results": results,
    }, indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: start_campaign
# ---------------------------------------------------------------------------

@mcp.tool()
def start_campaign(
    goal: Annotated[
        str,
        Field(description="What the campaign is for, in a sentence"),
    ],
    base_ttp_id: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "TTP every task starts from, each running a variant of it "
                "with its own objective weights. Usually omit it: the agent "
                "running each task then writes the profile that measures "
                "what that task is for. Name one only when the campaign is "
                "deliberately a sweep of a single profile and every task "
                "must be scored identically."
            ),
        ),
    ] = None,
    objectives: Annotated[
        list[str] | None,
        Field(
            default=None,
            description=(
                "Objective names every result is reported on. Fixed at "
                "creation, so results stay comparable. Use the names as they "
                "appear in the base TTP. Defaults to that TTP's own "
                "objectives, and to none when there is no base TTP."
            ),
        ),
    ] = None,
    concurrency: Annotated[
        int,
        Field(default=1, ge=1, description="Supervisors to hold in flight"),
    ] = 1,
    hours: Annotated[
        float | None,
        Field(default=None, gt=0, description="Dispatch window; omit for no deadline"),
    ] = None,
    grace_hours: Annotated[
        float,
        Field(
            default=0.0, ge=0,
            description="Drain window after the dispatch window closes, during "
                        "which in-flight tasks finish but nothing new is queued",
        ),
    ] = 0.0,
) -> str:
    """Open a fleet campaign: a goal, a pool size, and a wall-clock budget.

    This only opens the ledger entry. It dispatches no work and starts no
    supervisors, so a campaign can be set up and inspected before it begins
    spending GPU. The conductor loop is what fills it.
    """

    from reinvent4_mcp.tools.campaign import start_campaign as _impl

    return json.dumps(
        _impl(goal, base_ttp_id, objectives, concurrency, hours, grace_hours),
        indent=2, default=_json_default,
    )


# ---------------------------------------------------------------------------
# Tool: get_campaign_state
# ---------------------------------------------------------------------------

@mcp.tool()
def get_campaign_state(
    campaign_id: Annotated[str, Field(description="Campaign to inspect")],
) -> str:
    """Everything about a campaign in one call.

    Task counts by state, the supervisors currently holding tasks with their
    weight vectors and lease remaining, how many results are in, and how long
    is left in the dispatch and drain windows.

    ``window.phase`` follows the clock rather than the stored state, so a
    campaign whose conductor missed its own deadline still reports that it
    should be draining.
    """

    from reinvent4_mcp.tools.campaign import get_campaign_state as _impl

    return json.dumps(_impl(campaign_id), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: list_campaigns
# ---------------------------------------------------------------------------

@mcp.tool()
def list_campaigns() -> str:
    """Every campaign in the ledger, newest first, with enough to triage it.

    Each carries its task counts by state, how many results landed, and
    ``last_activity_at`` with ``last_activity_seconds_ago`` -- so "what ran
    most recently and did it produce anything" is answerable from this one
    call. ``phase`` follows the clock rather than the stored state, so a
    campaign whose conductor died and missed its own deadline still shows
    that it should be draining.
    """

    from reinvent4_mcp.tools.campaign import list_campaigns as _impl

    return json.dumps(_impl(), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: get_campaign_files
# ---------------------------------------------------------------------------

@mcp.tool()
def get_campaign_files(
    campaign_id: Annotated[str, Field(description="Campaign to locate on disk")],
    limit: Annotated[
        int,
        Field(default=20, description="How many tasks to report, newest first"),
    ] = 20,
) -> str:
    """Where a campaign's output is on disk, task by task.

    The ledger says what each task achieved; this says which directory it
    wrote while achieving it, so a result can be opened rather than only read
    about. Report these paths to the user when they ask where the files are.

    Each task gives its job id, its directory, and the notable files in it
    with sizes. Cofolded structures live on the shared volume instead -- use
    ``list_shared_assets`` for those.

    Tasks come back newest first and carry no scores, so do not describe the
    first one as the best: ``get_supervisor_result`` says what one achieved,
    and ``get_top_candidates`` ranks molecules within one job.
    """

    from reinvent4_mcp.tools.campaign import get_campaign_files as _impl

    return json.dumps(_impl(campaign_id, limit), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: enqueue_task
# ---------------------------------------------------------------------------

@mcp.tool()
def enqueue_task(
    campaign_id: Annotated[
        str,
        Field(description="Campaign to add to; must still be dispatching"),
    ],
    intent: Annotated[
        str,
        Field(
            description=(
                "What to do, in prose, handed to the sub-agent verbatim. "
                "This is the task. Say what the work is for and let the "
                "sub-agent decide how -- it can look a target or compound up "
                "and pick its own tools. For a design run this is also where "
                "chemistry the weights cannot express belongs, such as which "
                "chemotype to hold."
            ),
        ),
    ],
    kind: Annotated[
        str,
        Field(
            default="design",
            description=(
                "design | research | analysis | retrosynthesis | question | lead. "
                "'design' is a reinforcement-learning run; 'lead' hands a "
                "whole workstream to a sub-agent that conducts its own "
                "sub-campaign on this pool and reports one synthesis; every "
                "other kind is refused the job starters, because nothing "
                "would watch the run or read its results."
            ),
        ),
    ] = "design",
    title: Annotated[
        str,
        Field(default="", description="A few words naming the task for the "
                                      "person watching the campaign"),
    ] = "",
    context: Annotated[
        str,
        Field(
            default="",
            description=(
                "What the sub-agent needs to know but could not find out: "
                "the series, what has been tried, what another task found. "
                "It cannot see the conversation or the other sub-agents."
            ),
        ),
    ] = "",
    deliverable: Annotated[
        str,
        Field(default="", description="What to come back with, when the shape "
                                      "of the answer matters"),
    ] = "",
    weight_overrides: Annotated[
        dict[str, float] | None,
        Field(
            default=None,
            description=(
                "Design tasks on a campaign with a base TTP: objective name "
                "to weight. Names must match the base TTP's objectives -- an "
                "unknown name fails the task rather than running silently "
                "with the wrong weights."
            ),
        ),
    ] = None,
    fidelity_tier: Annotated[
        str,
        Field(
            default="exploration",
            description="'exploration' uses the scoring cache; 'validation' "
                        "bypasses the read so a result is never a cached sample",
        ),
    ] = "exploration",
    max_steps: Annotated[
        int | None,
        Field(default=None, gt=0, description="RL steps for a design task"),
    ] = None,
    derived_from: Annotated[
        list[str] | None,
        Field(
            default=None,
            description=(
                "Ids of this campaign's tasks whose reports led to this one. "
                "The user sees them as the lines from one experiment to the "
                "next; leave empty only for tasks that build on nothing yet."
            ),
        ),
    ] = None,
    workstream: Annotated[
        str,
        Field(default="", description="Id of the plan's workstream this task works on"),
    ] = "",
    priority: Annotated[
        int,
        Field(default=0, ge=-10, le=10,
              description="Higher is claimed first; leave 0 unless it must jump the queue"),
    ] = 0,
    concurrency: Annotated[
        int | None,
        Field(default=None, ge=1, le=64,
              description="For a 'lead': how many of the pool's sub-agents its workstream may hold"),
    ] = None,
) -> str:
    """Give one task to one sub-agent.

    The sub-agent that claims it does the whole job alone and reports back;
    read the reports with ``get_task_reports``. Tasks queued together run at
    the same time and cannot see each other, so make them answer different
    questions rather than near-identical variants of one.

    A design task given an ``intent`` will not quietly fall back to a default
    de novo run if its agent fails to start one -- it is handed back to be
    retried, because a substituted run would report a result for an
    experiment nobody asked for.
    """

    from reinvent4_mcp.tools.campaign import enqueue_task as _impl

    return json.dumps(
        _impl(
            campaign_id,
            intent,
            kind=kind,
            title=title or None,
            context=context or None,
            deliverable=deliverable or None,
            weight_overrides=weight_overrides or None,
            fidelity_tier=fidelity_tier,
            max_steps=max_steps,
            derived_from=derived_from or None,
            workstream=workstream or None,
            priority=priority,
            concurrency=concurrency,
        ),
        indent=2, default=_json_default,
    )


@mcp.tool()
def enqueue_tasks(
    campaign_id: Annotated[str, Field(description="Campaign to add to; must still be dispatching")],
    tasks: Annotated[
        list[dict[str, Any]],
        Field(
            min_length=1, max_length=200,
            description=(
                "The tasks, each an object with the fields enqueue_task takes: "
                "intent (required), kind, title, context, deliverable, "
                "workstream, priority, derived_from, max_steps, "
                "weight_overrides, fidelity_tier, and concurrency for a lead."
            ),
        ),
    ],
) -> str:
    """Queue several tasks in one call, all of them or none.

    The way to keep the team supplied: one call fills every free sub-agent
    and the ready queue behind them, instead of one task per turn. Make the
    tasks answer different questions -- they run at the same time and
    cannot see each other.

    When the batch has design tasks, ``runs`` says how many of them can
    start a run now. The rest wait for one, so a comparison across them
    should be read once all their reports are in.
    """

    from reinvent4_mcp.tools.campaign import enqueue_tasks as _impl

    return json.dumps(_impl(campaign_id, tasks), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tools: work the conductor does itself
# ---------------------------------------------------------------------------

@mcp.tool()
def start_own_task(
    campaign_id: Annotated[str, Field(description="Campaign to record it under")],
    intent: Annotated[str, Field(description="What the experiment is, in prose")],
    kind: Annotated[
        str,
        Field(default="design", description="design, research, analysis, retrosynthesis or question"),
    ] = "design",
    title: Annotated[str, Field(default="", description="A few words naming it, for the trajectory")] = "",
    context: Annotated[str, Field(default="", description="What it rests on")] = "",
    deliverable: Annotated[str, Field(default="", description="What you mean to come back with")] = "",
    derived_from: Annotated[
        list[str],
        Field(default_factory=list, description="Task ids whose results led to this one"),
    ] = [],
    workstream: Annotated[str, Field(default="", description="Workstream of the plan this works on")] = "",
) -> str:
    """Record an experiment you are about to run yourself, and take it.

    ``enqueue_task`` gives work away; this keeps it. Call it before starting
    a run or a line of work of your own, so what you do is in the ledger
    beside what sub-agents do: one trajectory, one set of reports, one pool
    of molecules, whether or not there are sub-agents.

    Working alone this is how a campaign has anything in it. Close it with
    ``finish_own_task`` -- a task left open shows as still running.
    """

    from reinvent4_mcp.tools.campaign import start_own_task as _impl

    return json.dumps(
        _impl(
            campaign_id, intent, kind=kind, title=title or None,
            context=context or None, deliverable=deliverable or None,
            derived_from=derived_from or None, workstream=workstream or None,
        ),
        indent=2, default=_json_default,
    )


@mcp.tool()
def finish_own_task(
    task_id: Annotated[str, Field(description="From start_own_task")],
    report: Annotated[str, Field(description="What happened and what it means, in prose")],
    summary: Annotated[str, Field(default="", description="One line, for the trajectory")] = "",
    candidates: Annotated[
        list[dict[str, Any]],
        Field(default_factory=list, description="Molecules it put forward: {smiles, total_score, components} each"),
    ] = [],
    job_ids: Annotated[
        list[str],
        Field(default_factory=list, description="Every run the experiment started"),
    ] = [],
    objective_profile: Annotated[
        dict[str, float],
        Field(default_factory=dict, description="Objective name to the value reached"),
    ] = {},
) -> str:
    """Close an experiment you ran yourself, with what it found.

    What you post is what the campaign's reports, its pooled molecules and
    its objective progress are computed from, so the user can read the
    result in the Campaign tab without reading the conversation back.
    """

    from reinvent4_mcp.tools.campaign import finish_own_task as _impl

    return json.dumps(
        _impl(
            task_id, report, summary=summary or None, candidates=candidates or None,
            job_ids=job_ids or None, objective_profile=objective_profile or None,
        ),
        indent=2, default=_json_default,
    )


# ---------------------------------------------------------------------------
# Tool: get_task_reports
# ---------------------------------------------------------------------------

@mcp.tool()
def get_task_reports(
    campaign_id: Annotated[str, Field(description="Campaign whose reports to read")],
    task_id: Annotated[
        str,
        Field(default="", description="One task; omit for every task that has reported"),
    ] = "",
) -> str:
    """What the sub-agents reported back.

    Each report is a sub-agent's own account of its task: what it did, what
    it found, what it could not settle and what it would do next. This is the
    only thing that crosses from a sub-agent to you -- their tool calls and
    transcripts are their own -- so read these before deciding what to
    dispatch next.

    Molecules are not included. ``get_supervisor_result`` has the candidates
    and scores for one task.
    """

    from reinvent4_mcp.tools.campaign import get_task_reports as _impl

    return json.dumps(
        _impl(campaign_id, task_id or None), indent=2, default=_json_default
    )


# ---------------------------------------------------------------------------
# Tools: the plan
# ---------------------------------------------------------------------------

@mcp.tool()
def set_campaign_plan(
    campaign_id: Annotated[str, Field(description="Campaign to plan")],
    workstreams: Annotated[
        list[dict[str, Any]],
        Field(
            min_length=1, max_length=40,
            description=(
                "The workstreams, each {id, name, question}: id a short handle "
                "tasks name it by, question what it has to settle for the goal. "
                "Optional share (rough fraction of the team, 0-1) and status "
                "(open, settled, dropped). Send all of them: this replaces the plan."
            ),
        ),
    ],
) -> str:
    """Split the goal into the workstreams a drug-discovery team would.

    Target biology and structure, known actives and SAR, design, selectivity,
    ADMET, synthesis -- whatever this goal needs. Tasks name their workstream,
    so get_campaign_coverage can show where the team is and where nobody is.
    """

    from reinvent4_mcp.tools.campaign import set_campaign_plan as _impl

    return json.dumps(_impl(campaign_id, workstreams), indent=2, default=_json_default)


@mcp.tool()
def get_campaign_coverage(
    campaign_id: Annotated[str, Field(description="Campaign whose coverage to read")],
) -> str:
    """How the work spreads over the plan: per workstream, what is queued,
    running and done, its latest finding and its waiting proposals, and the
    open workstreams nobody is working on."""

    from reinvent4_mcp.tools.campaign import get_campaign_coverage as _impl

    return json.dumps(_impl(campaign_id), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tools: proposals
# ---------------------------------------------------------------------------

@mcp.tool()
def list_proposals(
    campaign_id: Annotated[str, Field(description="Campaign whose open proposals to read")],
) -> str:
    """The next tasks sub-agents proposed at the end of their reports, still open.

    A sub-agent sees only its own task, so these are candidates: rank them
    against the campaign, take the ones worth a sub-agent with
    ``take_proposals`` and drop the rest with ``drop_proposals``.
    """

    from reinvent4_mcp.tools.campaign import list_proposals as _impl

    return json.dumps(_impl(campaign_id), indent=2, default=_json_default)


@mcp.tool()
def take_proposals(
    campaign_id: Annotated[str, Field(description="Campaign the proposals belong to")],
    proposal_ids: Annotated[list[str], Field(description="Proposals to queue as tasks")],
    priority: Annotated[
        int,
        Field(default=0, ge=-10, le=10,
              description="Higher is claimed first; leave 0 unless these must jump the queue"),
    ] = 0,
) -> str:
    """Queue proposals as tasks, each derived from the task that proposed it."""

    from reinvent4_mcp.tools.campaign import take_proposals as _impl

    return json.dumps(_impl(campaign_id, proposal_ids, priority), indent=2, default=_json_default)


@mcp.tool()
def drop_proposals(
    campaign_id: Annotated[str, Field(description="Campaign the proposals belong to")],
    proposal_ids: Annotated[list[str], Field(description="Proposals to set aside")],
) -> str:
    """Set proposals aside, so they are not offered again."""

    from reinvent4_mcp.tools.campaign import drop_proposals as _impl

    return json.dumps(_impl(campaign_id, proposal_ids), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tools: the atlas, what the campaign knows
# ---------------------------------------------------------------------------

@mcp.tool()
def atlas_overview(
    campaign_id: Annotated[str, Field(description="Campaign whose atlas to read")],
) -> str:
    """The map of what this campaign knows, branch by branch.

    Per branch: how many findings stand, how many are disputed, the latest
    claim and the tags to search by. Branches from the plan appear even when
    empty, and an empty branch is the part of the goal nobody has settled
    anything about. A page, where the reports behind it are a chapter.
    """

    from reinvent4_mcp.tools.campaign import atlas_overview as _impl

    return json.dumps(_impl(campaign_id), indent=2, default=_json_default)


@mcp.tool()
def atlas_branch(
    campaign_id: Annotated[str, Field(description="Campaign to read")],
    branch: Annotated[str, Field(description="Workstream id, as atlas_overview lists it")],
    limit: Annotated[int, Field(default=50, ge=1, le=200, description="Most findings to return")] = 50,
) -> str:
    """Everything one branch of the atlas holds, newest first.

    Each finding comes with the notes hung off it, so a claim another agent
    disputed arrives disputed rather than looking settled. Superseded
    findings are left out: this is what the campaign currently believes.
    """

    from reinvent4_mcp.tools.campaign import atlas_branch as _impl

    return json.dumps(_impl(campaign_id, branch, limit), indent=2, default=_json_default)


@mcp.tool()
def atlas_search(
    campaign_id: Annotated[str, Field(description="Campaign to search")],
    query: Annotated[
        str,
        Field(default="", description="Words that must all appear; empty returns the latest"),
    ] = "",
    tag: Annotated[str, Field(default="", description="Exact tag to require")] = "",
    limit: Annotated[int, Field(default=25, ge=1, le=200, description="Most findings to return")] = 25,
) -> str:
    """Findings matching words or a tag, across every branch.

    Use this before starting work somebody may already have done: a search
    that returns nothing costs a second and is worth a sub-agent's hour.
    """

    from reinvent4_mcp.tools.campaign import atlas_search as _impl

    return json.dumps(
        _impl(campaign_id, query, tag or None, limit), indent=2, default=_json_default
    )


@mcp.tool()
def atlas_finding(
    campaign_id: Annotated[str, Field(description="Campaign this belongs to")],
    branch: Annotated[
        str,
        Field(description="Workstream id to file it under. Findings are read a "
                          "branch at a time, so one filed in the wrong branch "
                          "is one nobody will see."),
    ],
    claim: Annotated[
        str,
        Field(description="The finding in one sentence, stated so it could be "
                          "disagreed with: what is true, not what you did"),
    ],
    detail: Annotated[
        str, Field(default="", description="Numbers, conditions and caveats behind it"),
    ] = "",
    evidence: Annotated[
        list[str] | None,
        Field(default=None,
              description="Where it came from: DOIs, ChEMBL ids, job ids, PDB codes"),
    ] = None,
    confidence: Annotated[
        Literal["high", "medium", "low"],
        Field(default="medium", description="Say low rather than say nothing"),
    ] = "medium",
    tags: Annotated[
        list[str] | None,
        Field(default=None,
              description="Handles to find it by later: a gene, a scaffold, an assay"),
    ] = None,
    task_id: Annotated[str, Field(default="", description="The task you are working")] = "",
    supersedes: Annotated[
        str, Field(default="", description="Id of the finding this corrects"),
    ] = "",
) -> str:
    """Record something the campaign now knows.

    Record what is true, not what you did. One claim per call, with the
    evidence that would let the next agent check it. Nothing is ever
    deleted: a finding that turns out wrong is corrected by recording the
    right one with ``supersedes`` set to the old id.
    """

    from reinvent4_mcp.tools.campaign import atlas_finding as _impl

    return json.dumps(
        _impl(
            campaign_id, branch, claim, detail or None, evidence, confidence,
            tags, task_id or None, supersedes or None,
        ),
        indent=2, default=_json_default,
    )


@mcp.tool()
def atlas_note(
    campaign_id: Annotated[str, Field(description="Campaign the finding belongs to")],
    finding_id: Annotated[str, Field(description="Finding to annotate")],
    body: Annotated[str, Field(description="What you have to say, with your own evidence")],
    stance: Annotated[
        Literal["supports", "disputes", "extends"],
        Field(default="extends", description="What your note does to the finding"),
    ] = "extends",
    task_id: Annotated[str, Field(default="", description="The task you are working")] = "",
) -> str:
    """Hang a second opinion off a finding somebody else recorded.

    Use this rather than recording a near-duplicate: an independent
    confirmation is worth more attached to the claim it confirms, and a
    dispute is worth most of all -- it is how the next agent learns the
    number it was about to build on is contested.
    """

    from reinvent4_mcp.tools.campaign import atlas_note as _impl

    return json.dumps(
        _impl(campaign_id, finding_id, body, stance, task_id or None),
        indent=2, default=_json_default,
    )


# ---------------------------------------------------------------------------
# Tool: list_supervisors
# ---------------------------------------------------------------------------

@mcp.tool()
def list_supervisors(
    campaign_id: Annotated[
        str,
        Field(
            default="",
            description="Narrow to one campaign's eligible supervisors and "
                        "include its in-flight tasks; omit for the whole fleet",
        ),
    ] = "",
) -> str:
    """The pool: how many supervisors exist, and how many could take work now.

    ``idle`` is how many would answer a dispatch immediately -- not
    ``concurrency``, which is only the ceiling a campaign asked for, and not
    the Deployment's replica count, which includes pods still starting or
    already shutting down.

    A supervisor is listed because it reported in, so a crashed one drops out
    once its presence expires. ``last_seen_seconds_ago`` is what says whether
    to believe the state beside it. Dead presences and expired leases are both
    swept before answering.
    """

    from reinvent4_mcp.tools.campaign import list_supervisors as _impl

    return json.dumps(
        _impl(campaign_id or None), indent=2, default=_json_default
    )


# ---------------------------------------------------------------------------
# Tool: get_supervisor_result
# ---------------------------------------------------------------------------

@mcp.tool()
def get_supervisor_result(
    campaign_id: Annotated[str, Field(description="Campaign the task belongs to")],
    task_id: Annotated[str, Field(description="Task whose result to fetch")],
) -> str:
    """What one finished task achieved: its objective profile and molecules."""

    from reinvent4_mcp.tools.campaign import get_supervisor_result as _impl

    return json.dumps(_impl(campaign_id, task_id), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: set_concurrency
# ---------------------------------------------------------------------------

@mcp.tool()
def set_concurrency(
    campaign_id: Annotated[str, Field(description="Campaign to adjust")],
    concurrency: Annotated[
        int,
        Field(ge=0, description="Supervisors to hold in flight"),
    ],
) -> str:
    """Change how many supervisors the campaign holds in flight.

    Recorded in the ledger, so the number survives a conductor restart. The
    conductor loop scales the supervisor Deployment to match on its next pass;
    this call on its own does not move pods.
    """

    from reinvent4_mcp.tools.campaign import set_concurrency as _impl

    return json.dumps(_impl(campaign_id, concurrency), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: cancel_supervisor
# ---------------------------------------------------------------------------

@mcp.tool()
def cancel_supervisor(
    task_id: Annotated[str, Field(description="Task to take back")],
) -> str:
    """Take a task back from the supervisor running it.

    The holder finds out on its next heartbeat and abandons the work without
    posting a result. This does not stop its RL job -- use stop_optimization
    on the job id if the GPU is what you need back.
    """

    from reinvent4_mcp.tools.campaign import cancel_supervisor as _impl

    return json.dumps(_impl(task_id), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: get_pool_capacity
# ---------------------------------------------------------------------------

@mcp.tool()
def get_pool_capacity(
    campaign_id: Annotated[
        str,
        Field(
            default="",
            description="Count only this campaign's eligible supervisors; "
                        "omit for the whole fleet",
        ),
    ] = "",
) -> str:
    """Whether there is room for more work, and which limit is the binding one.

    Two things have to be true to dispatch: a supervisor must be free to
    claim, and Boltz must have queue room to score what it starts. Either
    number alone misleads -- an idle supervisor handed work while the scorer
    is saturated only moves the wait, and a quiet scorer is no use with
    nothing alive to dispatch to. A design task also needs a run, and
    ``runs`` is the run window: how many REINVENT runs may hold a slot now,
    which closes while the cluster is slow to start them.

    ``can_dispatch`` is yes/no/unknown, or "not design" when only the run
    window is full -- research and question tasks need no run and can still
    go. ``binding_limit`` names whichever one said no. A ``None``
    queue_pressure means the gateway did not answer, which is not the same
    as idle.

    ``ceiling`` carries what max_supervisors() reports, so one call answers
    both "is there room now" and "how large could this pool ever be".
    """

    from reinvent4_mcp.tools.campaign import get_pool_capacity as _impl

    return json.dumps(
        _impl(campaign_id or None), indent=2, default=_json_default
    )


# ---------------------------------------------------------------------------
# Tool: max_supervisors
# ---------------------------------------------------------------------------

@mcp.tool()
def max_supervisors() -> str:
    """The largest supervisor pool this compute could run.

    Ask this before offering somebody a concurrency to choose, rather than
    letting them pick a number the machine will not honour. Two limits cap
    it and the smaller wins: what the scheduler would start (free CPUs in
    the partition under Slurm, the namespace quota under Kubernetes) and what
    the job backend will accept, since supervisors past the runner's
    concurrency limit have their tasks refused and burn their retries.

    ``max_supervisors`` of null means neither would say. That is unknown, not
    unlimited: offer a conservative default and say the ceiling is unreadable
    rather than inventing one.
    """

    from reinvent4_mcp.tools.campaign import max_supervisors as _impl

    return json.dumps(_impl(), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: conduct_campaign
# ---------------------------------------------------------------------------

@mcp.tool()
def conduct_campaign(
    campaign_id: Annotated[str, Field(description="Campaign to conduct")],
    task_budget_hours: Annotated[
        float,
        Field(default=0.0, ge=0,
              description="Wall clock per task, capped by the hard stop; "
                          "0 leaves it to the campaign's own window"),
    ] = 0.0,
    max_steps: Annotated[
        int,
        Field(default=0, ge=0, description="RL steps per task; 0 for the default"),
    ] = 0,
    concurrency: Annotated[
        int,
        Field(default=0, ge=0,
              description="Sub-agents to hold in flight; 0 keeps what the "
                          "campaign was opened with. Check max_supervisors "
                          "first"),
    ] = 0,
    propose: Annotated[
        bool,
        Field(
            default=False,
            description=(
                "Let the loop fill idle slots with tasks of its own. Off by "
                "default: you dispatch with enqueue_task, because you are "
                "holding the conversation that knows what the work is for. "
                "Turn it on to leave a campaign running unattended."
            ),
        ),
    ] = False,
) -> str:
    """Keep the sub-agent pool at size for a campaign you are directing.

    Opening a campaign does not run it. This starts the loop that holds the
    pool at the campaign's concurrency, closes the queue when the dispatch
    window ends, and finalizes at the hard stop. It returns at once, so say
    that it has started and carry on; use conducting_status later.

    It dispatches nothing unless ``propose`` is set: give out the work with
    enqueue_task and read it back with get_task_reports.

    Idempotent: asking twice does not start a second loop.
    """

    from reinvent4_mcp.tools.campaign import conduct_campaign as _impl

    return json.dumps(
        _impl(
            campaign_id,
            task_budget_hours=task_budget_hours or None,
            max_steps=max_steps or None,
            concurrency=concurrency or None,
            propose=propose,
        ),
        indent=2, default=_json_default,
    )


# ---------------------------------------------------------------------------
# Tool: stop_conducting
# ---------------------------------------------------------------------------

@mcp.tool()
def stop_conducting(
    campaign_id: Annotated[str, Field(description="Campaign to stop dispatching to")],
) -> str:
    """Stop dispatching to a campaign, leaving the campaign itself open.

    In-flight tasks keep running and their results still land in the ledger;
    what stops is the proposing of new work and the resizing of the pool.
    Use close_campaign to end the campaign, finalize_campaign to close the
    books.
    """

    from reinvent4_mcp.tools.campaign import stop_conducting as _impl

    return json.dumps(_impl(campaign_id), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: conducting_status
# ---------------------------------------------------------------------------

@mcp.tool()
def conducting_status(
    campaign_id: Annotated[
        str,
        Field(default="", description="One campaign, or omit for all of them"),
    ] = "",
) -> str:
    """What is being conducted here, and what the last pass did.

    ``last_tick`` is the conductor's own report of that pass: the window
    phase, how many tasks are in flight and queued, how many it dispatched,
    and whether it held back. ``last_error`` is set when a pass failed --
    the loop keeps going, because a ledger that is briefly unreachable is not
    a campaign that is over.
    """

    from reinvent4_mcp.tools.campaign import conducting_status as _impl

    return json.dumps(
        _impl(campaign_id or None), indent=2, default=_json_default
    )


# ---------------------------------------------------------------------------
# Tool: close_campaign
# ---------------------------------------------------------------------------

@mcp.tool()
def close_campaign(
    campaign_id: Annotated[str, Field(description="Campaign to stop dispatching")],
) -> str:
    """Stop dispatching, let in-flight tasks finish.

    The drain window brought forward by hand. Queued tasks stay claimable, so
    work already decided on still runs.
    """

    from reinvent4_mcp.tools.campaign import close_campaign as _impl

    return json.dumps(_impl(campaign_id), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: finalize_campaign
# ---------------------------------------------------------------------------

@mcp.tool()
def finalize_campaign(
    campaign_id: Annotated[str, Field(description="Campaign to close the books on")],
) -> str:
    """Close the books: abandon anything outstanding and end the campaign.

    Tasks still queued or claimed are marked abandoned rather than left
    looking like work someone will do.
    """

    from reinvent4_mcp.tools.campaign import finalize_campaign as _impl

    return json.dumps(_impl(campaign_id), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dict(value: Any) -> dict[str, Any]:
    """Coerce a JSON string to a dict if needed.

    LLMs sometimes pass tool arguments as JSON strings instead of
    parsed dicts.  This handles that gracefully instead of failing
    with a Pydantic validation error.
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    raise TypeError(f"Expected a JSON object (dict), got {type(value).__name__}")


def _json_default(obj: Any) -> Any:
    """Handle non-standard types for JSON serialization."""
    import numpy as np

    if isinstance(obj, (np.floating, float)):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, set):
        return list(obj)
    return repr(obj)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )


def main() -> None:
    """Launch the Drug Discovery MCP server with stdio transport."""

    config = get_config()
    configure_logging(config.logging.level)

    logger.info(f"Starting Drug Discovery MCP server v{__version__}")

    try:
        ensure_reinvent4_importable()
        logger.info("REINVENT4 is importable")
    except RuntimeError as exc:
        logger.warning(f"REINVENT4 not found at startup: {exc}")

    # Load all extension tool modules (blocks 01-10).
    # Each module must export register_tools(mcp) -> None.
    # A failure in one extension is logged and isolated; others still load.
    from reinvent4_mcp.registry import load_extensions  # noqa: PLC0415
    load_extensions(mcp)

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
