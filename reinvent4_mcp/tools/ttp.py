# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""TTP management tools: create, update, get, list, delete.

Provides the implementation behind the MCP ``create_ttp``,
``update_ttp``, ``get_ttp``, ``list_ttps``, and ``delete_ttp`` tools.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from reinvent4_mcp.config import get_config
from reinvent4_mcp.ttp.schema import TTP
from reinvent4_mcp.ttp.store import TTPStore

logger = logging.getLogger(__name__)

_store: TTPStore | None = None


def _get_store() -> TTPStore:
    global _store
    if _store is None:
        config = get_config()
        _store = TTPStore(config.ttp.storage_dir)
    return _store


def reset_store() -> None:
    """Discard the cached store (for tests)."""
    global _store
    _store = None


#: Native components built from a pattern, by the parameter that carries it
#: and whether REINVENT4 takes one per endpoint or a list of them. Mirrored
#: from the components' own ``Parameters``, where the outer list is the
#: endpoint dimension ``collect_params`` adds: ``GroupCount.smarts`` is
#: ``List[str]``, so one pattern, while ``CustomAlerts.smarts`` is
#: ``List[List[str]]``, so a list. Checked here because the registry that
#: would catch it is only importable where REINVENT4 itself is, which the
#: gateway is not, and a wrong shape otherwise surfaces as a GPU job that
#: dies seconds in naming neither the objective nor the parameter.
_PATTERN_PARAMS: dict[str, tuple[str, bool]] = {
    "groupcount": ("smarts", False),
    "matchingsubstructure": ("smarts", True),
    "customalerts": ("smarts", True),
    "reactionfilter": ("reaction_smarts", True),
    "tanimotosimilarity": ("smiles", True),
    "tanimotodistance": ("smiles", True),
    "mmp": ("reference_smiles", True),
}


def _endpoint_params_for_component(
    component: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Return one endpoint's native params with required safe defaults."""
    from reinvent4_mcp.ttp.mapper import _native_component_params_with_defaults

    endpoint = _native_component_params_with_defaults(component, params)
    normalized = component.lower().replace("-", "").replace("_", "")
    pattern = _PATTERN_PARAMS.get(normalized)
    if pattern is not None:
        name, many = pattern
        endpoint[name] = _patterns(component, name, endpoint.get(name), many)
    return endpoint


def _patterns(component: str, name: str, value: Any, many: bool) -> Any:
    """One endpoint's pattern parameter, in the shape REINVENT4 collects it in.

    A single pattern is accepted where a list is wanted and the other way
    round, since both are how an agent writes it; what cannot be guessed is
    a component given several patterns where only one is counted.
    """

    given = value
    if isinstance(given, str):
        given = [given]
    if isinstance(given, list) and len(given) == 1 and isinstance(given[0], list):
        given = given[0]
    if (
        not isinstance(given, list)
        or not given
        or not all(isinstance(one, str) and one.strip() for one in given)
    ):
        raise ValueError(
            f"{component}.params.{name} must be "
            + ("a non-empty list of patterns" if many else "one pattern")
        )
    if many:
        return given
    if len(given) > 1:
        raise ValueError(
            f"{component} takes one {name} pattern per objective; "
            f"give params.{name} a single pattern, and each further "
            "pattern an objective of its own"
        )
    return given[0]


def _validate_transforms(ttp: TTP) -> None:
    """Every transform with the parameters REINVENT4 needs to build it.

    The schema fills in the conventional ones; a bound has no conventional
    value, and without one REINVENT4 fails building the ``Scorer`` seconds
    into the job, naming neither the transform nor its objective.
    """
    from reinvent4_mcp.transforms import complete_transform

    owners = [*(ttp.pre_scoring.components if ttp.pre_scoring else []), *ttp.scoring.objectives]
    for owner in owners:
        if owner.transform is None:
            continue
        try:
            complete_transform(owner.transform.model_dump())
        except ValueError as exc:
            raise ValueError(f"Objective '{owner.name}': {exc}") from exc


def _validate_for_reinvent(ttp: TTP) -> None:
    """What a TTP has to get right for REINVENT4 to start a run from it."""
    _validate_transforms(ttp)
    _validate_native_component_params(ttp)


def _validate_native_component_params(ttp: TTP) -> None:
    """Validate native scorer parameters before a TTP reaches job launch."""
    native = [
        objective
        for objective in ttp.scoring.objectives
        if objective.type == "native"
    ]
    if not native:
        return

    # Shape validation for known components works even when the external
    # REINVENT checkout has not yet been restored (e.g. lightweight unit CI).
    prepared: list[tuple[str, dict[str, Any]]] = []
    for objective in native:
        component = objective.component or ""
        prepared.append(
            (component, _endpoint_params_for_component(component, objective.params or {}))
        )

    try:
        from reinvent4_mcp.utils import ensure_reinvent4_importable

        ensure_reinvent4_importable()
        from reinvent.scoring.importer import get_registry

        registry = get_registry()
    except (ImportError, ModuleNotFoundError, FileNotFoundError):
        logger.info(
            "REINVENT4 is unavailable; native parameter validation is limited "
            "to built-in structural checks."
        )
        return

    for component, endpoint in prepared:
        lookup = component.lower().replace("-", "").replace("_", "")
        if lookup not in registry:
            raise ValueError(
                f"Native scoring component {component!r} is unavailable in this runtime"
            )
        _, component_params = registry[lookup]
        if component_params is None:
            continue
        collected = {key: [value] for key, value in endpoint.items()}
        try:
            component_params(**collected)
        except Exception as exc:
            raise ValueError(f"Invalid params for {component}: {exc}") from exc


def create_ttp(ttp_data: dict[str, Any]) -> dict[str, Any]:
    """Validate and persist a new TTP.

    Returns a summary on success or an ``{"error": ...}`` dict on failure.
    """
    try:
        ttp = TTP.model_validate(ttp_data)
        _validate_for_reinvent(ttp)
    except Exception as exc:
        return {"error": f"Invalid TTP: {exc}"}

    try:
        created = _get_store().create(ttp)
    except FileExistsError as exc:
        return {
            "error": str(exc),
            "hint": f"Use update_ttp('{ttp.id}', updates) to preserve version history.",
        }

    return {
        "status": "created",
        "id": created.id,
        "name": created.name,
        "version": created.version,
        "ttp": created.model_dump(mode="json"),
    }


def update_ttp(ttp_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    """Apply a partial update to an existing TTP.

    Version is auto-incremented.  The returned dict contains the full
    updated TTP so the agent can inspect the result.
    """
    try:
        updated = _get_store().update(
            ttp_id,
            updates,
            validator=_validate_for_reinvent,
        )
    except FileNotFoundError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        return {"error": f"Invalid update: {exc}"}

    return {
        "status": "updated",
        "id": updated.id,
        "name": updated.name,
        "version": updated.version,
        "ttp": updated.model_dump(mode="json"),
    }


def get_ttp(
    ttp_id: str,
    version: Optional[int] = None,
) -> dict[str, Any]:
    """Retrieve a TTP by ID, optionally at a specific historical version."""
    try:
        ttp = _get_store().get(ttp_id, version=version)
    except FileNotFoundError as exc:
        return {"error": str(exc)}
    return ttp.model_dump(mode="json")


def list_ttps() -> list[dict[str, Any]]:
    """List all stored TTPs (latest versions only, summary metadata)."""
    return _get_store().list_all()


def delete_ttp(ttp_id: str) -> dict[str, Any]:
    """Delete a TTP and its version history.

    Prefer ``update_ttp`` for retries or corrections. Delete only when the
    identifier is being abandoned; deletion is irreversible.
    """
    try:
        _get_store().delete(ttp_id)
    except FileNotFoundError as exc:
        return {"error": str(exc)}
    return {"status": "deleted", "id": ttp_id}


def resolve_ttp(
    ttp_id: Optional[str] = None,
    ttp_data: Optional[dict[str, Any]] = None,
) -> TTP | dict[str, Any]:
    """Resolve a TTP from either an ID (stored) or inline data.

    Returns a validated ``TTP`` instance on success or an
    ``{"error": ...}`` dict on failure.
    """
    if ttp_id and ttp_data:
        return {"error": "Provide either ttp_id or ttp (inline), not both."}
    if ttp_id:
        try:
            return _get_store().get(ttp_id)
        except FileNotFoundError as exc:
            return {"error": str(exc)}
    if ttp_data:
        try:
            return TTP.model_validate(ttp_data)
        except Exception as exc:
            return {"error": f"Invalid inline TTP: {exc}"}
    return {"error": "Provide either ttp_id or ttp (inline TTP definition)."}
