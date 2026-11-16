# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Optimization tools: run_optimization, stop_optimization.

Implements the MCP tools for launching and controlling REINVENT4
agent-monitored RL optimization campaigns.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _is_mol2mol_model(model: str) -> bool:
    """Return True if the resolved *model* is a Mol2Mol prior.

    Detection is based on the checkpoint's declared ``model_type`` metadata so
    that priors are classified by what they actually are rather than by their
    registry key or file path.  If the checkpoint cannot be read (e.g. REINVENT4
    is unavailable, the file is missing, or the metadata is absent), this falls
    back to a substring match on the model key/path.
    """

    from reinvent4_mcp.jobs.toml_generator import resolve_model_path

    try:
        import torch

        ckpt = torch.load(
            resolve_model_path(model), map_location="cpu", weights_only=False
        )
        mtype = str(ckpt.get("model_type", "")).lower()
        if mtype:
            return "mol2mol" in mtype
    except Exception:
        logger.debug(
            "Could not read checkpoint metadata for model '%s'; "
            "falling back to key/path substring match",
            model,
            exc_info=True,
        )

    key = model.lower()
    return "m2m" in key or "mol2mol" in key


def run_optimization(
    *,
    ttp_id: Optional[str] = None,
    ttp: Optional[dict[str, Any]] = None,
    model: str = ".reinvent",
    max_steps: Optional[int] = None,
    batch_size: Optional[int] = None,
    sigma: Optional[int] = None,
    learning_rate: Optional[float] = None,
    resume_from: Optional[str] = None,
    termination: Optional[dict[str, Any]] = None,
    seed_smiles: Optional[list[str]] = None,
    fidelity_tier: str = "exploration",
) -> dict[str, Any]:
    """Launch an RL optimization campaign with full TTP scoring.

    Composes a Scorer from the TTP (native + external components),
    generates a REINVENT4 TOML config, and starts a background
    subprocess.  The agent can then monitor the run with
    ``get_optimization_status`` and intervene with
    ``stop_optimization``.

    Parameters
    ----------
    ttp_id:
        ID of a stored TTP.
    ttp:
        Inline TTP dict (alternative to *ttp_id*).
    model:
        Prior model registry key or file path.
    max_steps:
        Maximum RL steps.  None lets the system choose: 64 for
        exploration-tier Boltz-scored runs, 300 otherwise.
    batch_size:
        Molecules per step (overrides process config).
    sigma, learning_rate:
        DAP learning strategy overrides.
    resume_from:
        Checkpoint path to resume from.  Can be an absolute path
        or a path relative to the jobs working directory.
    termination:
        Early-termination criteria dict.
    seed_smiles:
        List of SMILES strings used as input seeds for Mol2Mol
        priors.  Required when *model* is a Mol2Mol variant;
        ignored for Reinvent priors.

    Returns
    -------
    dict
        Job summary with ``job_id``, ``working_dir``, and launch details.
    """

    from reinvent4_mcp.jobs.manager import get_job_manager
    from reinvent4_mcp.tools.ttp import resolve_ttp
    from reinvent4_mcp.ttp.mapper import get_mapper_summary, ttp_to_scorer_config
    from reinvent4_mcp.ttp.schema import TTP as TTPModel
    from reinvent4_mcp.ttp.schema import GenerationMode

    resolved = resolve_ttp(ttp_id=ttp_id, ttp_data=ttp)
    if isinstance(resolved, dict):
        return resolved

    assert isinstance(resolved, TTPModel)
    generation = resolved.generation
    if generation is not None:
        # The builder stores exact settings on the TTP because a natural-
        # language request is not a reliable transport for batch size.
        if model == ".reinvent":
            model = getattr(generation, "model", model)
        if max_steps is None:
            max_steps = getattr(generation, "n_steps", None)
        if batch_size is None:
            batch_size = getattr(generation, "batch_size", None)
        mol2mol_input = getattr(generation, "mol2mol_input", None)
        if seed_smiles is None and mol2mol_input is not None:
            seed_smiles = mol2mol_input.seed_smiles

    is_mol2mol = _is_mol2mol_model(model)
    if is_mol2mol and not seed_smiles:
        return {
            "error": (
                "Mol2Mol models require 'seed_smiles' -- a list of SMILES "
                "that the model conditions generation on."
            ),
        }
    if seed_smiles and not is_mol2mol:
        logger.warning(
            "seed_smiles provided but model '%s' is not Mol2Mol; "
            "seed_smiles will be ignored",
            model,
        )

    if is_mol2mol and seed_smiles:
        try:
            from rdkit import Chem
        except ImportError:
            logger.warning(
                "RDKit is unavailable; skipping seed_smiles validation. "
                "Invalid seeds may cause a downstream failure in mol2mol."
            )
        else:
            cleaned, bad, stripped = [], [], []
            for s in seed_smiles:
                s2 = (s or "").strip()
                mol = None if not s2 else Chem.MolFromSmiles(s2)
                if not s2 or mol is None:
                    bad.append(s)
                    continue
                # REINVENT's token vocabulary does not include isotope labels
                # such as [2H], [13C], [18F]. Strip them before writing the
                # seed file, and warn so the agent knows what changed.
                for atom in mol.GetAtoms():
                    atom.SetIsotope(0)
                canonical = Chem.MolToSmiles(mol)
                if canonical != Chem.MolToSmiles(Chem.MolFromSmiles(s2)):
                    stripped.append(s2)
                cleaned.append(canonical)
            if bad:
                return {
                    "error": (
                        "Invalid seed_smiles (empty or not parseable by "
                        f"RDKit): {bad}"
                    ),
                }
            if stripped:
                logger.warning(
                    "stripped isotope labels from seed_smiles (%d SMILES affected); "
                    "REINVENT's vocabulary does not include isotope tokens. "
                    "Originals: %s",
                    len(stripped), stripped,
                )
            seed_smiles = cleaned

    # This path builds de novo and Mol2Mol TOML only. A TTP asking for a
    # scaffold or linker run would otherwise launch the default prior and
    # report a healthy campaign that never decorated the scaffold, so name the
    # tool that can build it rather than quietly running something else.
    declared = resolved.generation.mode if resolved.generation else None
    if declared in (GenerationMode.libinvent, GenerationMode.linkinvent):
        return {
            "error": (
                f"TTP '{resolved.id}' asks for generation.mode="
                f"'{declared.value}', which run_optimization cannot build. "
                f"Call run_generation_campaign(ttp_id='{resolved.id}') instead."
            ),
        }

    try:
        scorer_config = ttp_to_scorer_config(resolved, fidelity_tier=fidelity_tier)
    except Exception as exc:
        return {"error": f"Failed to build scorer config from TTP: {exc}"}

    if not scorer_config.get("component"):
        return {"error": "TTP produced no scoring components."}

    mapper_summary = get_mapper_summary(resolved, scorer_config)

    # When max_steps is not explicitly set, cap Boltz-scored exploration runs
    # at 64 steps: exploration defaults make each step ~2 min, so 64 steps
    # fits in a session; 300 steps would run 10+ hours.
    effective_max_steps = max_steps
    if max_steps is None:
        uses_boltz = any(
            obj.service == "boltz"
            for obj in resolved.scoring.objectives
            if obj.type == "external"
        )
        effective_max_steps = 64 if (uses_boltz and fidelity_tier == "exploration") else 300

    manager = get_job_manager()
    result = manager.create_and_start(
        ttp_id=resolved.id,
        ttp_version=resolved.version,
        scorer_config=scorer_config,
        model=model,
        max_steps=effective_max_steps,
        batch_size=batch_size,
        sigma=sigma,
        learning_rate=learning_rate,
        resume_from=resume_from,
        termination=termination,
        seed_smiles=seed_smiles if is_mol2mol else None,
    )

    if "error" not in result:
        result["ttp_id"] = resolved.id
        result["ttp_version"] = resolved.version
        result["mapper_summary"] = mapper_summary

    return result


def run_transfer_learning(
    *,
    model: str = ".reinvent",
    smiles: Optional[list[str]] = None,
    smiles_file: Optional[str] = None,
    num_epochs: int = 50,
    batch_size: int = 128,
    save_every_n_epochs: int = 5,
    sample_batch_size: int = 2000,
    validation_smiles: Optional[list[str]] = None,
    validation_smiles_file: Optional[str] = None,
    standardize_smiles: bool = True,
    randomize_smiles: bool = True,
    output_model_file: str = "tl_model.model",
) -> dict[str, Any]:
    """Fine-tune a generative model on a SMILES dataset via transfer learning.

    Transfer learning adapts a prior model to generate molecules similar
    to a provided SMILES set.  The resulting model can then be used as
    the starting agent for ``run_optimization`` (RL).

    Either *smiles* (an inline list) or *smiles_file* (a path to a .smi
    file) must be provided.  Validation data is optional.

    Parameters
    ----------
    model:
        Prior model registry key or file path to fine-tune.
    smiles:
        Inline list of SMILES strings for training.
    smiles_file:
        Path to a .smi file with training SMILES (one per line).
    num_epochs:
        Number of training epochs.
    batch_size:
        Training batch size.
    save_every_n_epochs:
        Save an intermediate model checkpoint every N epochs.
    sample_batch_size:
        Number of molecules sampled for validation metrics.
    validation_smiles:
        Inline list of SMILES for validation.
    validation_smiles_file:
        Path to a validation .smi file.
    standardize_smiles:
        Standardize SMILES before training.
    randomize_smiles:
        Randomize SMILES representations during training.
    output_model_file:
        Base name for the output model file.

    Returns
    -------
    dict
        Job summary with ``job_id``, ``working_dir``, ``output_model_file``,
        and launch details.
    """

    from reinvent4_mcp.jobs.manager import get_job_manager

    manager = get_job_manager()
    return manager.create_and_start_tl(
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


def stop_optimization(job_id: str) -> dict[str, Any]:
    """Stop a running optimization job.

    With ``checkpoint_on_stop`` enabled in the process config, asks
    REINVENT4 to save a checkpoint and exit, and waits up to
    ``stop_grace_seconds`` for it: the job can only act on the request
    once it abandons the scoring call it is in, which takes as long as
    one RL step.  Returns the checkpoint path for use with
    ``resume_from`` in a subsequent ``run_optimization`` call, or says
    so in ``message`` when no checkpoint was written.

    Parameters
    ----------
    job_id:
        ID of the job to stop.

    Returns
    -------
    dict
        Stop confirmation with checkpoint path and final status.
    """

    from reinvent4_mcp.jobs.manager import get_job_manager

    manager = get_job_manager()
    return manager.stop_job(job_id)
