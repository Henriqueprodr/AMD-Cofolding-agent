# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""TTP Pydantic models and validation.

Defines the Therapeutic Target Profile schema used throughout the MCP
server.  A TTP has two scoring stages (pre-scoring and scoring) that
are merged into a single composed Scorer at runtime.

Validation rules
-----------------
* TTP ``id`` must be a lowercase alphanumeric slug.
* Pre-scoring components must reference known fast RDKit descriptors.
* Scoring objectives must be either ``native`` (REINVENT4 built-in) or
  ``external`` (REST service).  External objectives require ``service``,
  and ``service_url`` unless the service is one configured here.
* ``depends_on`` references are checked against sibling objective names.
* Transform ``type`` is normalised and validated against REINVENT4's
  transform registry.
* Weights must be positive.
"""

from __future__ import annotations

import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_TRANSFORM_TYPES = frozenset(
    {
        "sigmoid",
        "reverse_sigmoid",
        "double_sigmoid",
        "right_step",
        "left_step",
        "step",
        "value_mapping",
        "exponential_decay",
    }
)

_TRANSFORM_TYPES_NORMALISED = {t.replace("_", ""): t for t in VALID_TRANSFORM_TYPES}

VALID_PRESCORING_COMPONENTS = frozenset(
    {
        "QED",
        "MolecularWeight",
        "HBondDonors",
        "HBondAcceptors",
        "SlogP",
        "NumRotBond",
    }
)

# First-class external services that have a config entry + canonical URL in the
# process config (see reinvent4_mcp.config.enabled_external_services). This is an
# advisory allow-list, NOT an enforced one: the TTP mapper deliberately supports
# arbitrary custom REST services (it falls back to the TTP-supplied service_url
# for anything not listed here), so an unknown service is permitted rather than
# rejected. Keep this in sync with config.ScoringConfig when adding a service.
KNOWN_EXTERNAL_SERVICES = frozenset(
    {
        "boltz",
        "posebusters",
        "admet",
        "retrodmta",
        "interaction_profiler",
        # AiZynthFinder. Its /score returns a [0,1] route-quality score per
        # molecule, so it is an RL objective like the others and not only the
        # job-based planner that plan_synthesis drives.
        "synthesis_planner",
    }
)

_VALID_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")

_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------


class TransformConfig(BaseModel):
    """Score transform mapping raw component values to [0, 1].

    The ``type`` field selects the transform function.  Additional fields
    are type-specific parameters forwarded to REINVENT4 at runtime.

    Available types and their parameters (all besides ``type`` are optional
    unless marked required)::

        sigmoid / reverse_sigmoid  ->  low (req), high (req), k
        double_sigmoid             ->  low (req), high (req), coef_div, coef_si, coef_se
        right_step / left_step / step  ->  low, high
        value_mapping              ->  mapping (req, dict)
        exponential_decay          ->  k (req)

    ``step`` uses ``low`` and ``high``, not ``value``; ``value`` is not a
    parameter of any REINVENT4 transform and is rejected on creation so it
    does not waste a GPU slot.
    """

    model_config = {"extra": "allow"}

    type: str = Field(description="Transform function name")

    @field_validator("type")
    @classmethod
    def _validate_type(cls, v: str) -> str:
        normalised = v.lower().replace("_", "").replace("-", "")
        if normalised not in _TRANSFORM_TYPES_NORMALISED:
            raise ValueError(
                f"Unknown transform type '{v}'. "
                f"Valid types: {', '.join(sorted(VALID_TRANSFORM_TYPES))}"
            )
        return v

    @model_validator(mode="after")
    def _numbers_written_as_text_are_still_numbers(self) -> "TransformConfig":
        """Coerce ``"500"`` to ``500.0`` in the type-specific parameters.

        Every parameter here except ``mapping`` is a number, and the field is
        ``extra: allow``, so nothing was checking. An agent writing a TTP
        quotes them about half the time -- JSON from a language model does
        that -- and the result reached REINVENT as a string, where the first
        ``high - low`` raised "unsupported operand type(s) for -: 'str' and
        'str'" five seconds into a GPU job, naming neither the field nor the
        transform. Observed on LUMI, twice, on consecutive attempts at the
        same task.

        Coerced rather than rejected: "500" has exactly one meaning here, and
        failing the TTP would turn a typo the agent cannot see into a task it
        cannot complete.
        """

        for name, value in list((self.__pydantic_extra__ or {}).items()):
            if not isinstance(value, str):
                continue
            try:
                number = float(value)
            except ValueError:
                continue
            self.__pydantic_extra__[name] = number
        return self

    @model_validator(mode="after")
    def _conventional_parameters_are_filled_in(self) -> "TransformConfig":
        """Supply the parameters REINVENT4 requires but conventionally fixes.

        The sigmoid family's ``k`` is required by REINVENT4 and built while
        constructing the ``Scorer``, so omitting it surfaced as
        ``Parameters.__init__() missing 1 required positional argument: 'k'``
        five seconds into a GPU job, naming neither the transform nor the
        component it belonged to. Observed on LUMI, where it failed a campaign
        task whose sibling had already produced molecules.

        Only parameters with a conventional value are filled. A sigmoid with
        no ``low`` is a transform nobody can guess, and saying so belongs to
        whoever checks the TTP is complete, not to a field coercion.
        """

        from reinvent4_mcp.transforms import transform_spec

        spec = transform_spec(self.type)
        if spec is None:
            return self
        extra = self.__pydantic_extra__
        if extra is None:
            return self
        for name, value in spec.defaults.items():
            if extra.get(name) is None:
                extra[name] = value

        # Unknown parameters pass silently through ``extra="allow"`` and
        # reach REINVENT4, which uses strict dataclasses and rejects them
        # with an unhelpful TypeError mid-job. Catching them here turns a
        # wasted GPU slot into an immediate, named validation error at TTP
        # creation time. Observed: ``transform.value = 0.7`` on a ``step``
        # transform (correct: ``low`` or ``high``).
        known = set(spec.parameters) | {"type"}
        unknown = sorted(k for k in extra if k not in known and extra[k] is not None)
        if unknown:
            takes = ", ".join(spec.parameters) or "(none besides type)"
            raise ValueError(
                f"transform '{self.type}' got unknown parameter(s): "
                f"{', '.join(unknown)}. It takes: {takes}."
            )
        return self


# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------


class StructureTemplateConfig(BaseModel):
    """A protein structure used to condition Boltz ligand placement."""

    model_config = {"extra": "forbid"}

    path: str = Field(description="Shared-volume path to a PDB or mmCIF template")
    chain_id: Optional[str] = Field(
        default=None,
        description="Protein chain to use from the template",
    )
    force: bool = Field(
        default=False,
        description="Apply the template as a forced rather than soft constraint",
    )
    threshold: Optional[float] = Field(
        default=None,
        gt=0,
        description="Maximum backbone deviation in Angstrom when force is true",
    )

    @model_validator(mode="after")
    def _forced_template_has_threshold(self) -> "StructureTemplateConfig":
        if self.force and self.threshold is None:
            raise ValueError("A forced structure template requires a threshold")
        return self


class TargetConfig(BaseModel):
    """Protein target information for structure-based scoring.

    ``protein_sequence`` is required for Boltz cofolding.  Other fields
    supply additional structural context.
    """

    model_config = {"extra": "forbid"}

    protein_sequence: str = Field(
        description="Full amino-acid sequence of the target protein",
    )
    structure_path: Optional[str] = Field(
        default=None,
        description="Shared-volume path of the structure selected for this target",
    )
    structure_chain: Optional[str] = Field(
        default=None,
        description="Protein chain selected from structure_path",
    )
    templates: Optional[list[StructureTemplateConfig]] = Field(
        default=None,
        description="Structures used to condition Boltz ligand placement",
    )
    pdb_id: Optional[str] = Field(
        default=None,
        description="PDB identifier (e.g. '3KRR')",
    )
    msa_path: Optional[str] = Field(
        default=None,
        description="Path to pre-computed MSA file (.a3m)",
    )
    protein_pdb_path: Optional[str] = Field(
        default=None,
        description="Path to protein structure PDB file",
    )
    # E3: binding-site constraints
    binding_site_residues: Optional[list[list]] = Field(
        default=None,
        description=(
            "Pocket constraint contacts: list of [chain_id, author_resseq] pairs "
            "using PDB author numbering. The framework translates them to Boltz "
            "sequence positions via the residue_numbering.json artifact."
        ),
    )
    binding_site_max_distance_A: float = Field(
        default=6.0,
        ge=4.0,
        le=20.0,
        description="max_distance for the pocket constraint (Angstrom).",
    )
    binding_site_force: bool = Field(
        default=False,
        description="Apply the pocket constraint as a hard potential (E4).",
    )
    contact_pairs: Optional[list[dict]] = Field(
        default=None,
        description=(
            "Per-atom contact constraints. Each entry: "
            "{token1: ['A', resseq], token2: ['B', atom_name], "
            "max_distance: 3.5, force: true}."
        ),
    )
    covalent_bond: Optional[dict] = Field(
        default=None,
        description=(
            "Bond constraint for a covalent warhead (L1). "
            "{atom1: ['A', 145, 'SG'], atom2: ['B', 1, 'C7']}. "
            "Affinity prediction is suppressed automatically (L2)."
        ),
    )

    @field_validator("protein_sequence")
    @classmethod
    def _validate_sequence(cls, v: str) -> str:
        v = v.strip().upper()
        if len(v) < 10:
            raise ValueError("Protein sequence must be at least 10 residues")
        invalid = set(v) - _VALID_AMINO_ACIDS
        if invalid:
            raise ValueError(
                f"Invalid amino-acid characters: {', '.join(sorted(invalid))}. "
                f"Sequence must only contain standard amino acids."
            )
        return v


# ---------------------------------------------------------------------------
# Pre-scoring
# ---------------------------------------------------------------------------


class PreScoringComponent(BaseModel):
    """A single pre-scoring component (fast RDKit descriptor).

    Pre-scoring components compute drug-likeness descriptors for every
    molecule and contribute to the final aggregated score.
    """

    model_config = {"extra": "forbid"}

    name: str = Field(description="Display name for this component")
    type: Literal["native"] = Field(
        default="native",
        description="Component type (always 'native' for pre-scoring)",
    )
    component: str = Field(
        description=(
            "REINVENT4 component class name "
            "(e.g. 'QED', 'MolecularWeight', 'HBondDonors')"
        ),
    )
    weight: float = Field(
        default=1.0,
        gt=0,
        description="Relative weight in score aggregation",
    )
    transform: Optional[TransformConfig] = Field(
        default=None,
        description="Score transform to map raw values to [0, 1]",
    )

    @field_validator("component")
    @classmethod
    def _validate_component(cls, v: str) -> str:
        if v not in VALID_PRESCORING_COMPONENTS:
            raise ValueError(
                f"Unknown pre-scoring component '{v}'. "
                f"Valid: {', '.join(sorted(VALID_PRESCORING_COMPONENTS))}"
            )
        return v


class PreScoringSection(BaseModel):
    """Pre-scoring section of a TTP.

    Fast drug-likeness descriptors (QED, Lipinski properties) computed
    for all molecules.  These contribute to the final aggregated score
    alongside the full scoring objectives.
    """

    model_config = {"extra": "forbid"}

    comment: Optional[str] = Field(
        default=None,
        description="Human-readable description",
    )
    components: list[PreScoringComponent] = Field(
        default_factory=list,
        description="Pre-scoring components to evaluate",
    )

    @field_validator("components")
    @classmethod
    def _validate_unique_names(
        cls, v: list[PreScoringComponent]
    ) -> list[PreScoringComponent]:
        names = [c.name for c in v]
        seen: set[str] = set()
        dupes: set[str] = set()
        for n in names:
            if n in seen:
                dupes.add(n)
            seen.add(n)
        if dupes:
            raise ValueError(
                f"Duplicate pre-scoring component names: {', '.join(sorted(dupes))}"
            )
        return v


# ---------------------------------------------------------------------------
# Scoring objectives
# ---------------------------------------------------------------------------


class ScoringObjective(BaseModel):
    """A single scoring objective (native REINVENT4 or external REST).

    Native objectives use built-in REINVENT4 components.
    External objectives call REST services (Boltz, PoseBusters).
    """

    model_config = {"extra": "forbid"}

    name: str = Field(description="Display name for this objective")
    type: Literal["native", "external"] = Field(
        description="'native' for REINVENT4 components, 'external' for REST services",
    )

    # --- native fields ---
    component: Optional[str] = Field(
        default=None,
        description="REINVENT4 component class name (required when type='native')",
    )

    # --- external fields ---
    service: Optional[str] = Field(
        default=None,
        description=(
            "External service identifier: 'boltz', 'posebusters', 'admet', "
            "'retrodmta', or 'interaction_profiler'"
        ),
    )
    service_url: Optional[str] = Field(
        default=None,
        description=(
            "REST endpoint URL for the external service. Use the Docker "
            "Compose service name and the '/score' path: "
            "'http://boltz:8080/score' for boltz, "
            "'http://posebusters:8080/score' for posebusters, "
            "'http://admet-predictor:8084/score' for admet, "
            "'http://retrodmta-predictor:8085/score' for retrodmta. "
            "Note the service name is 'boltz' (not 'boltz2'). Optional for "
            "the services configured here, which always use the configured URL."
        ),
    )
    service_params: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Service-specific parameters forwarded in the request body. "
            "admet: {'model_set': 'drug'|'agro', 'endpoint': 'logp'} "
            "(one objective per endpoint). "
            "retrodmta: {'target': 'FXA', 'endpoint': 'FXA_pKi', 'window': 'latest', 'method': 'ensemble'}. "
            "interaction_profiler: {'interaction_type': 'hbond'} (counts that interaction "
            "type; needs a structure, so chain after a producer like boltz via depends_on)."
        ),
    )
    depends_on: Optional[str] = Field(
        default=None,
        description=(
            "Name of another objective whose output this objective consumes "
            "(pipeline dependency, e.g. PoseBusters depends on Boltz structures)"
        ),
    )

    # --- common fields ---
    weight: float = Field(
        default=1.0,
        gt=0,
        description="Relative weight in score aggregation",
    )
    transform: Optional[TransformConfig] = Field(
        default=None,
        description="Score transform to map raw values to [0, 1]",
    )
    params: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Component-specific parameters forwarded to REINVENT4. "
            "Use plain values -- per-endpoint wrapping is automatic. "
            "Example for TanimotoSimilarity: "
            "{'smiles': ['CCO'], 'radius': 3, 'use_counts': true, "
            "'use_features': false}"
        ),
    )
    comment: Optional[str] = Field(
        default=None,
        description="Human-readable note",
    )

    @model_validator(mode="after")
    def _validate_type_specific_fields(self) -> "ScoringObjective":
        if self.type == "native":
            if not self.component:
                raise ValueError(
                    f"Native objective '{self.name}' requires a 'component' field"
                )
        elif self.type == "external":
            if not self.service:
                raise ValueError(
                    f"External objective '{self.name}' requires a 'service' field"
                )
            if not self.service_url:
                self.service_url = _configured_url(self.service)
            if not self.service_url:
                raise ValueError(
                    f"External objective '{self.name}' requires a 'service_url' field: "
                    f"'{self.service}' is not a service configured here"
                )
        return self


def _configured_url(service: str) -> Optional[str]:
    """Where this deployment scores ``service``, when it is one it runs.

    The mapper uses this URL whatever the TTP says, so asking for it only
    gives a TTP one more way to fail validation.
    """
    from reinvent4_mcp.config import get_config

    return get_config().enabled_external_services.get(service)


class ScoringSection(BaseModel):
    """Scoring section of a TTP.

    Full scoring objectives (affinity, pose quality, SA, etc.) that
    evaluate every molecule.  Both native and external objectives are
    aggregated together with the pre-scoring components.
    """

    model_config = {"extra": "forbid"}

    comment: Optional[str] = Field(
        default=None,
        description="Human-readable description",
    )
    objectives: list[ScoringObjective] = Field(
        description="Scoring objectives to evaluate",
    )
    aggregation: Literal["geometric_mean", "arithmetic_mean"] = Field(
        default="geometric_mean",
        description="Score aggregation method across all components",
    )

    @field_validator("objectives")
    @classmethod
    def _validate_unique_names(
        cls, v: list[ScoringObjective]
    ) -> list[ScoringObjective]:
        names = [o.name for o in v]
        seen: set[str] = set()
        dupes: set[str] = set()
        for n in names:
            if n in seen:
                dupes.add(n)
            seen.add(n)
        if dupes:
            raise ValueError(
                f"Duplicate scoring objective names: {', '.join(sorted(dupes))}"
            )
        return v

    @model_validator(mode="after")
    def _validate_depends_on(self) -> "ScoringSection":
        objective_names = {o.name for o in self.objectives}
        dep_of: dict[str, str] = {}
        for obj in self.objectives:
            if obj.depends_on and obj.depends_on not in objective_names:
                raise ValueError(
                    f"Objective '{obj.name}' has depends_on='{obj.depends_on}' "
                    f"which is not a defined objective. "
                    f"Available: {', '.join(sorted(objective_names))}"
                )
            if obj.depends_on == obj.name:
                raise ValueError(
                    f"Objective '{obj.name}' cannot depend on itself"
                )
            if obj.depends_on:
                dep_of[obj.name] = obj.depends_on

        # Detect cycles in the depends_on graph (e.g. A->B->A): without a cycle
        # check the pipeline has no valid starting point. Each objective has at
        # most one dependency, so follow each chain and flag any revisit.
        for start in dep_of:
            seen: set[str] = set()
            node: str | None = start
            while node in dep_of:
                if node in seen:
                    chain = " -> ".join([*seen, node])
                    raise ValueError(
                        f"Circular depends_on detected involving objective "
                        f"'{start}' (cycle through: {chain})"
                    )
                seen.add(node)
                node = dep_of[node]
        return self


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


class ConstraintConfig(BaseModel):
    """Structural constraint (e.g. toxic SMARTS filter).

    Constraints define structural patterns that molecules must avoid.
    Currently supports ``custom_alerts`` (SMARTS-based substructure
    filters).  Additional constraint-specific fields are accepted via
    ``extra="allow"``.
    """

    model_config = {"extra": "allow"}

    type: str = Field(
        description="Constraint type (e.g. 'custom_alerts')",
    )
    smarts: Optional[list[str]] = Field(
        default=None,
        description="SMARTS patterns for substructure matching",
    )

    @model_validator(mode="after")
    def _validate_smarts_for_alerts(self) -> "ConstraintConfig":
        if self.type == "custom_alerts" and not self.smarts:
            raise ValueError(
                "Constraint type 'custom_alerts' requires a non-empty 'smarts' list"
            )
        return self


# ---------------------------------------------------------------------------
# Top-level TTP
# ---------------------------------------------------------------------------


class TTP(BaseModel):
    """Therapeutic Target Profile -- full multi-property optimisation spec.

    A TTP defines everything needed for a drug design campaign:

    * **target** -- protein sequence, PDB, MSA for structure-based scoring
    * **pre_scoring** -- fast drug-likeness descriptors (QED, Lipinski)
    * **scoring** -- full objectives (Boltz affinity, PoseBusters, SA, …)
    * **constraints** -- structural filters (toxic SMARTS, etc.)

    The ``id`` uniquely identifies the TTP in persistent storage.
    ``version`` is auto-incremented by the TTP store on each update.
    """

    model_config = {"extra": "forbid"}

    id: str = Field(
        description=(
            "Unique identifier -- lowercase alphanumeric with hyphens "
            "or underscores (e.g. 'jak2-oral-lead')"
        ),
    )
    name: str = Field(
        description="Human-readable name for this TTP",
    )
    version: int = Field(
        default=1,
        ge=1,
        description="Version number (auto-incremented on update)",
    )
    target: Optional[TargetConfig] = Field(
        default=None,
        description=(
            "Protein target information. Required when scoring "
            "objectives include Boltz-based services."
        ),
    )
    pre_scoring: Optional[PreScoringSection] = Field(
        default=None,
        description="Fast drug-likeness descriptors (QED, Lipinski)",
    )
    scoring: ScoringSection = Field(
        description="Full scoring objectives (native + external)",
    )
    constraints: Optional[list[ConstraintConfig]] = Field(
        default=None,
        description="Structural constraints (SMARTS filters, etc.)",
    )
    generation: Optional[GenerationSection] = Field(
        default=None,
        description=(
            "Generation campaign configuration (mode, model, batch size, etc.). "
            "None = use existing run_optimization defaults (backward compat)."
        ),
    )
    pipeline: Optional["PipelineSection"] = Field(
        default=None,
        description=(
            "Optional preparation pipeline. Steps are tool calls the agent runs "
            "in dependency order before starting the RL loop. Turning a TTP into "
            "a full workflow spec: the same TTP in a fresh session runs the same "
            "preparation, with params locked and each step's result stored."
        ),
    )

    @field_validator("id")
    @classmethod
    def _validate_id(cls, v: str) -> str:
        if not _ID_PATTERN.match(v):
            raise ValueError(
                f"Invalid TTP id '{v}'. "
                f"Must be lowercase alphanumeric with hyphens or underscores, "
                f"starting with a letter or digit."
            )
        return v

    @model_validator(mode="after")
    def _validate_target_for_boltz(self) -> "TTP":
        """Require target when Boltz objectives are defined."""
        boltz_objectives = [
            o.name
            for o in self.scoring.objectives
            if o.type == "external" and o.service == "boltz"
        ]
        if boltz_objectives and self.target is None:
            raise ValueError(
                f"Objectives {boltz_objectives} use Boltz scoring which requires "
                f"a 'target' section with at least 'protein_sequence'."
            )
        return self

    @model_validator(mode="after")
    def _validate_boltz_affinity_transform_direction(self) -> "TTP":
        """Warn when pred_value affinity mode is combined with an upward transform.

        affinity_pred_value is log10(IC50/µM) — MORE NEGATIVE = tighter = BETTER.
        An upward sigmoid (low < high, e.g. sigmoid(low=0.3, high=0.8)) rewards
        POSITIVE values (weak binders) instead of negative ones, which is wrong.

        Correct transforms for pred_value mode:
          reverse_sigmoid(low=-2.0, high=0.0)  → score=1.0 at −2 (10 nM), 0.0 at +2 (100 µM)
          sigmoid(low=-2.0, high=0.0)           → score=0.0 at −2, 1.0 at 0  ← also WRONG

        Safe default: use affinity_mode='probability_binary' with sigmoid(low=0.3, high=0.8).
        """
        import logging as _logging  # noqa: PLC0415
        _log = _logging.getLogger("reinvent")

        for obj in self.scoring.objectives:
            sp = obj.service_params or {}
            # service_params may be a dict or Pydantic model
            if hasattr(sp, "get"):
                affinity_mode = sp.get("affinity_mode", "probability_binary")
            else:
                affinity_mode = getattr(sp, "affinity_mode", "probability_binary")
            if affinity_mode != "pred_value":
                continue
            transform = obj.transform
            if transform is None:
                continue
            # transform is a TransformConfig Pydantic model — use attribute access
            t_type = getattr(transform, "type", "") or ""
            t_low = getattr(transform, "low", None)
            t_high = getattr(transform, "high", None)
            # Upward sigmoid (low < high) with pred_value rewards positive (weak) values
            if t_type == "sigmoid" and t_low is not None and t_high is not None:
                if float(t_low) < float(t_high) and float(t_high) >= 0:
                    _log.warning(
                        "⚠️  TTP objective '%s' uses affinity_mode='pred_value' with "
                        "sigmoid(low=%s, high=%s). affinity_pred_value is log10(IC50/µM): "
                        "MORE NEGATIVE = tighter = BETTER. This sigmoid rewards POSITIVE "
                        "(weak-binding) values. Use reverse_sigmoid(low=-2.0, high=0.0) "
                        "to reward tight binders, or switch to affinity_mode='probability_binary' "
                        "with sigmoid(low=0.3, high=0.8).",
                        obj.name, t_low, t_high,
                    )
        return self

    @model_validator(mode="after")
    def _validate_no_name_collisions(self) -> "TTP":
        """Ensure component/objective names are unique across sections."""
        all_names: list[str] = []
        if self.pre_scoring:
            all_names.extend(c.name for c in self.pre_scoring.components)
        all_names.extend(o.name for o in self.scoring.objectives)
        seen: set[str] = set()
        dupes: set[str] = set()
        for n in all_names:
            if n in seen:
                dupes.add(n)
            seen.add(n)
        if dupes:
            raise ValueError(
                f"Component/objective names must be unique across pre_scoring "
                f"and scoring sections. Duplicates: {', '.join(sorted(dupes))}"
            )
        return self

    # -- convenience properties -------------------------------------------

    @property
    def has_external_objectives(self) -> bool:
        return any(o.type == "external" for o in self.scoring.objectives)

    @property
    def native_objective_names(self) -> list[str]:
        names: list[str] = []
        if self.pre_scoring:
            names.extend(c.name for c in self.pre_scoring.components)
        names.extend(
            o.name for o in self.scoring.objectives if o.type == "native"
        )
        return names

    @property
    def external_objective_names(self) -> list[str]:
        return [
            o.name
            for o in self.scoring.objectives
            if o.type == "external"
        ]


# ===========================================================================
# Generation contract — block 08 additions
# ---------------------------------------------------------------------------
# All models below are ADDITIVE. The TTP class above gains one new optional
# field (``generation``) which defaults to ``None`` so every existing TTP
# continues to load and validate without changes.
# ===========================================================================

from enum import Enum  # noqa: E402  (after existing imports)


# ---------------------------------------------------------------------------
# SMILES helper validators (module-level so generation.py can import them)
# ---------------------------------------------------------------------------

def _validate_linkinvent_fragment(smiles: str, frag_idx: int) -> list[str]:
    """Return a list of error strings (empty = valid) for one LinkInvent fragment.

    Checks:
    - RDKit can parse the SMILES.
    - Exactly one attachment point atom (atomic num 0) is present.
    - That atom carries a map number (e.g. [*:1], not bare [*]).
    """
    errors: list[str] = []
    try:
        from rdkit import Chem  # noqa: PLC0415
    except ImportError:
        # RDKit unavailable — the structural checks below need it, so skip them
        # rather than report a false "cannot parse" error (matches
        # _validate_seed_smiles, which treats a missing RDKit as "cannot check").
        return errors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [f"Fragment {frag_idx}: cannot parse SMILES {smiles!r}"]

    dummy_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummy_atoms) != 1:
        errors.append(
            f"Fragment {frag_idx}: must have exactly 1 attachment point [*:n], "
            f"found {len(dummy_atoms)}"
        )
    map_nums = [a.GetAtomMapNum() for a in dummy_atoms]
    if any(m == 0 for m in map_nums):
        errors.append(
            f"Fragment {frag_idx}: attachment atom must carry a map number, "
            "e.g. [*:1] not [*]"
        )
    return errors


def _validate_libinvent_scaffold(smiles: str) -> list[str]:
    """Return a list of error strings (empty = valid) for a LibInvent scaffold.

    Checks:
    - RDKit can parse the SMILES.
    - At least one attachment point atom (atomic num 0) is present.
    - All attachment atoms carry map numbers.
    - Map numbers are unique (no duplicates).
    """
    try:
        from rdkit import Chem  # noqa: PLC0415
    except ImportError:
        # RDKit unavailable — skip structural checks rather than report a false
        # parse error (consistent with _validate_seed_smiles / linkinvent).
        return []

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [f"Scaffold: cannot parse SMILES {smiles!r}"]

    dummy_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if not dummy_atoms:
        return [
            "Scaffold: no attachment points [*:n] found — LibInvent needs "
            "at least one R-group position"
        ]

    map_nums = [a.GetAtomMapNum() for a in dummy_atoms]
    errors: list[str] = []
    if any(m == 0 for m in map_nums):
        errors.append(
            "Scaffold: all attachment points must carry map numbers, "
            "e.g. [*:1], [*:2] — not bare [*]"
        )
    if len(set(map_nums)) != len(map_nums):
        errors.append(
            f"Scaffold: duplicate attachment-point map numbers {map_nums}"
        )
    return errors


# ---------------------------------------------------------------------------
# Generation models
# ---------------------------------------------------------------------------

class GenerationMode(str, Enum):
    """Supported REINVENT generation algorithms."""
    reinvent   = "reinvent"
    libinvent  = "libinvent"
    linkinvent = "linkinvent"
    mol2mol    = "mol2mol"


class LinkInventInput(BaseModel):
    """Fragment pair (and optional linker constraints) for LinkInvent.

    Each fragment SMILES must contain exactly one attachment-point atom
    ([*:n]) with a unique map number across the pair.  The SMILES are
    validated by RDKit at model-construction time.
    """

    model_config = {"extra": "forbid"}

    fragments: list[str] = Field(
        description=(
            "Exactly 2+ fragment SMILES, each with one [*:n] attachment point "
            "and distinct map numbers, e.g. ['[*:1]CC', '[*:2]OC']."
        ),
    )
    allowed_atom_types: Optional[list[str]] = Field(
        default=None,
        description="Atom symbols allowed in the generated linker, e.g. ['C','N','O','S'].",
    )
    max_linker_length: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum number of heavy atoms in the linker.",
    )
    min_linker_length: Optional[int] = Field(
        default=None,
        ge=1,
        description="Minimum number of heavy atoms in the linker.",
    )

    @field_validator("fragments")
    @classmethod
    def _validate_fragments(cls, v: list[str]) -> list[str]:
        if len(v) < 2:
            raise ValueError(
                "linkinvent requires at least 2 fragment SMILES "
                "(one per side of the linker)."
            )
        errors: list[str] = []
        map_nums_seen: set[int] = set()

        for i, smi in enumerate(v):
            frag_errors = _validate_linkinvent_fragment(smi, i)
            errors.extend(frag_errors)

            # Cross-fragment map-number uniqueness
            try:
                from rdkit import Chem  # noqa: PLC0415
                mol = Chem.MolFromSmiles(smi)
            except ImportError:
                mol = None
            if mol is not None:
                for atom in mol.GetAtoms():
                    if atom.GetAtomicNum() == 0 and atom.GetAtomMapNum():
                        mn = atom.GetAtomMapNum()
                        if mn in map_nums_seen:
                            errors.append(
                                f"Fragment {i}: map number :{mn} is already used "
                                "in another fragment — each must be unique"
                            )
                        map_nums_seen.add(mn)

        if errors:
            raise ValueError("\n".join(errors))
        return v


class LibInventInput(BaseModel):
    """Scaffold (and optional R-group constraints) for LibInvent.

    The scaffold SMILES must contain at least one attachment-point atom
    ([*:n]) with a unique map number per position.  Validated by RDKit
    at model-construction time.
    """

    model_config = {"extra": "forbid"}

    scaffold: str = Field(
        description=(
            "Scaffold SMILES with [*:n] attachment points at every R-group "
            "position, e.g. 'c1ccc([*:1])cc1[*:2]'."
        ),
    )
    decoration_positions: Optional[list[int]] = Field(
        default=None,
        description=(
            "Map numbers to decorate (subset of attachment points in scaffold). "
            "None = decorate all positions."
        ),
    )

    @field_validator("scaffold")
    @classmethod
    def _validate_scaffold(cls, v: str) -> str:
        errors = _validate_libinvent_scaffold(v)
        if errors:
            raise ValueError("\n".join(errors))
        return v


class Mol2MolInput(BaseModel):
    """Seed molecules for Mol2Mol lead optimization."""

    model_config = {"extra": "forbid"}

    seed_smiles: list[str] = Field(
        min_length=1,
        max_length=4096,
        description=(
            "One or more valid seed SMILES that Mol2Mol conditions generation on."
        ),
    )

    @field_validator("seed_smiles")
    @classmethod
    def _validate_seed_smiles(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        invalid: list[str] = []
        try:
            from rdkit import Chem  # noqa: PLC0415
        except ImportError:
            Chem = None  # type: ignore[assignment]

        for value in values:
            smiles = (value or "").strip()
            if not smiles or (Chem is not None and Chem.MolFromSmiles(smiles) is None):
                invalid.append(value)
            else:
                cleaned.append(smiles)

        if invalid:
            raise ValueError(f"Invalid Mol2Mol seed SMILES: {invalid}")
        return cleaned


class GenerationSection(BaseModel):
    """Explicit campaign generation contract.

    Attached to a TTP to specify which REINVENT algorithm to use and
    how to configure the generation job.  All fields have safe defaults
    so a minimal ``GenerationSection(mode='reinvent')`` is valid.
    """

    model_config = {"extra": "forbid"}

    mode: GenerationMode = Field(
        default=GenerationMode.reinvent,
        description=(
            "Generation algorithm: 'reinvent' | 'libinvent' | 'linkinvent' | "
            "'mol2mol'."
        ),
    )
    model: str = Field(
        default=".reinvent",
        description=(
            "Prior/model registry key such as '.reinvent', '.libinvent', "
            "'.linkinvent', or '.m2m_medium'; alternatively a host-specific "
            "model path. Resolved via the model manifest."
        ),
    )
    batch_size: int = Field(
        default=128, ge=1, le=4096,
        description="Number of molecules generated per step.",
    )
    n_steps: int = Field(
        default=100, ge=1,
        description="Total number of RL steps.",
    )
    checkpoint_every_n: int = Field(
        default=50, ge=1,
        description="Save a checkpoint every N steps.",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility.  None = non-deterministic.",
    )
    device: str = Field(
        default="cuda",
        description="Compute device: 'cuda' or 'cpu'.",
    )

    # Mode-specific inputs — validated against mode before job launch
    linkinvent_input: Optional[LinkInventInput] = Field(
        default=None,
        description="Required for mode='linkinvent'.",
    )
    libinvent_input: Optional[LibInventInput] = Field(
        default=None,
        description="Required for mode='libinvent'.",
    )
    mol2mol_input: Optional[Mol2MolInput] = Field(
        default=None,
        description="Required for mode='mol2mol'.",
    )

    # Design-context artifact references
    target_artifact_ids: list[str] = Field(
        default_factory=list,
        description="Campaign-level target record artifact IDs.",
    )
    prepared_structure_artifact_ids: list[str] = Field(
        default_factory=list,
        description="Prepared structure artifact IDs (from structure preparator).",
    )
    site_artifact_ids: list[str] = Field(
        default_factory=list,
        description="Binding-site or interface artifact IDs.",
    )

    @model_validator(mode="after")
    def _check_mode_inputs(self) -> "GenerationSection":
        if self.mode == GenerationMode.linkinvent:
            if self.linkinvent_input is None:
                raise ValueError(
                    "linkinvent mode requires linkinvent_input "
                    "(fragment SMILES with attachment points)."
                )
        if self.mode == GenerationMode.libinvent:
            if self.libinvent_input is None:
                raise ValueError(
                    "libinvent mode requires libinvent_input "
                    "(scaffold SMILES with [*:n] attachment points)."
                )
        if self.mode == GenerationMode.mol2mol:
            if self.mol2mol_input is None:
                raise ValueError(
                    "mol2mol mode requires mol2mol_input "
                    "(one or more seed SMILES)."
                )
        return self


# ===========================================================================
# Pipeline contract — chaining extension
# ---------------------------------------------------------------------------
# A TTP that holds a PipelineSection is a full reproducible workflow spec.
# The agent runs the steps in dependency order: each step calls one MCP tool,
# the result is stored in the campaign context under ``store_as``, and later
# steps read from the context via ``${step_id}`` interpolation in params.
#
# Scoring objectives still use depends_on; the pipeline is for the outer
# orchestration that *prepares* the campaign before the RL loop starts.
# ===========================================================================


class PipelineStep(BaseModel):
    """One step in a TTP pipeline.

    A step calls a single MCP tool with its params. The result is stored in
    the campaign context under ``store_as`` and can be referenced by later
    steps as ``${store_as.field}`` in their params values.

    Example:
        A structure-based campaign needs six preparation steps before the RL
        loop can start. Without a pipeline contract the agent re-derives this
        sequence from the TTP contents every session, without a record and
        without a way to restart from a checkpoint. With the pipeline the
        sequence is committed to the TTP, params are locked, and a failed step
        names exactly what failed.
    """

    model_config = {"extra": "forbid"}

    id: str = Field(
        description=(
            "Step identifier — unique within this pipeline, used for dependency "
            "references and context key defaults. Slug: lowercase, hyphens, digits."
        ),
    )
    tool: str = Field(
        description=(
            "MCP tool name to call, exactly as it appears in the tool registry "
            "(e.g. 'prepare_structure', 'derive_scaffold_from_series', "
            "'fetch_all_activities', 'plan_synthesis')."
        ),
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Parameters forwarded to the tool. String values may reference the "
            "stored output of a previous step as '${step_id}' (whole result) or "
            "'${step_id.field}' (one field). The agent resolves references "
            "before the call."
        ),
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description=(
            "IDs of steps that must complete before this one runs. The agent "
            "topologically sorts the pipeline and runs independent steps first. "
            "An empty list means this step has no prerequisites."
        ),
    )
    store_as: Optional[str] = Field(
        default=None,
        description=(
            "Key under which the tool result is stored in the campaign context. "
            "Defaults to the step id. Later steps reference this as '${store_as}'."
        ),
    )
    condition: Literal["if_no_error", "always"] = Field(
        default="if_no_error",
        description=(
            "'if_no_error' (default): skip this step if any dependency returned "
            "an error. 'always': run unconditionally (use for cleanup or logging "
            "steps that must run even after a failure)."
        ),
    )
    comment: Optional[str] = Field(
        default=None,
        description="Human-readable note — why this step is here, what to watch for.",
    )

    @field_validator("id")
    @classmethod
    def _validate_id(cls, v: str) -> str:
        if not _ID_PATTERN.match(v):
            raise ValueError(
                f"Pipeline step id '{v}' must be a lowercase slug "
                "(letters, digits, hyphens, underscores)."
            )
        return v


class PipelineSection(BaseModel):
    """Ordered preparation workflow attached to a TTP.

    The agent executes the steps in dependency order before starting the RL
    loop. This turns a TTP into a complete reproducible spec: the same TTP
    opened in a fresh session runs the same preparation rather than asking
    the agent to re-derive it.

    Every step name in ``depends_on`` must reference another step in this
    pipeline, and there must be no cycles. Independent steps run first;
    their order among themselves is not specified and the agent may run them
    in any order (including in parallel if the tool supports it).
    """

    model_config = {"extra": "forbid"}

    comment: Optional[str] = Field(
        default=None,
        description="Human-readable description of this workflow.",
    )
    steps: list[PipelineStep] = Field(
        min_length=1,
        description="The steps of the pipeline, in any declared order.",
    )

    @field_validator("steps")
    @classmethod
    def _unique_step_ids(cls, v: list[PipelineStep]) -> list[PipelineStep]:
        seen: set[str] = set()
        dupes: set[str] = set()
        for s in v:
            (dupes if s.id in seen else seen).add(s.id)
        if dupes:
            raise ValueError(
                f"Duplicate pipeline step ids: {', '.join(sorted(dupes))}"
            )
        return v

    @model_validator(mode="after")
    def _validate_depends_on(self) -> "PipelineSection":
        ids = {s.id for s in self.steps}
        for step in self.steps:
            unknown = [d for d in step.depends_on if d not in ids]
            if unknown:
                raise ValueError(
                    f"Step '{step.id}' depends_on unknown ids: "
                    f"{', '.join(unknown)}. Known: {', '.join(sorted(ids))}"
                )
            if step.id in step.depends_on:
                raise ValueError(f"Step '{step.id}' cannot depend on itself.")
        # Cycle detection: each step has multiple possible predecessors, so
        # traverse the full reachability set rather than a single chain.
        for start in self.steps:
            visited: set[str] = set()
            queue = list(start.depends_on)
            while queue:
                node = queue.pop()
                if node == start.id:
                    raise ValueError(
                        f"Circular dependency detected involving step '{start.id}'."
                    )
                if node in visited:
                    continue
                visited.add(node)
                parent = next((s for s in self.steps if s.id == node), None)
                if parent:
                    queue.extend(parent.depends_on)
        return self

    def topological_order(self) -> list[PipelineStep]:
        """Steps in a valid execution order (dependencies before dependents).

        Uses Kahn's algorithm. Independent steps are returned in declaration
        order relative to each other.
        """
        from collections import deque  # noqa: PLC0415

        in_degree: dict[str, int] = {s.id: 0 for s in self.steps}
        dependents: dict[str, list[str]] = {s.id: [] for s in self.steps}
        for step in self.steps:
            for dep in step.depends_on:
                in_degree[step.id] += 1
                dependents[dep].append(step.id)

        by_id = {s.id: s for s in self.steps}
        queue: deque[str] = deque(
            s.id for s in self.steps if in_degree[s.id] == 0
        )
        result: list[PipelineStep] = []
        while queue:
            sid = queue.popleft()
            result.append(by_id[sid])
            for dep_id in dependents[sid]:
                in_degree[dep_id] -= 1
                if in_degree[dep_id] == 0:
                    queue.append(dep_id)
        return result


