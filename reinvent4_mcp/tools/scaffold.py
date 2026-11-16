# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deriving a scaffold from a series, and taking a series apart around one.

Everything upstream of this could already find a series -- ChEMBL returns the
actives for a target, a run returns its own output -- and everything
downstream could already use a scaffold: LibInvent decorates one, LinkInvent
joins two fragments, the sketcher draws one. Nothing in between turned a set
of molecules into the core they share, so the only route was for the agent to
read the SMILES and write the scaffold itself, which is the one thing it is
told never to do because invented SMILES are frequently unparseable.

Two operations, sharing their machinery:

``derive_scaffold``
    The core the series has in common, with its open positions marked as
    numbered attachment points -- the form LibInvent requires, where a bare
    ``[*]`` is refused.

``decompose_series``
    What sits at each of those positions, per molecule. This is the R-group
    table a medicinal chemist reads as SAR, and it falls out of the same
    substructure match.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: Seconds before the maximum-common-substructure search gives up. MCS is
#: worst-case exponential, and a series wide enough to be interesting is
#: exactly where it runs long.
_MCS_TIMEOUT = 20

#: A Murcko core shared by less than this fraction of the series is not
#: really "the" scaffold of it.
_DEFAULT_MIN_FRACTION = 0.5

#: Substituents listed per position before the rest are counted rather than
#: named, so a 5000-molecule series cannot fill the context window.
_MAX_R_GROUPS = 25

#: Heavy atoms below which a core is not worth calling a scaffold. Molecules
#: with nothing in common still have a single carbon in common, and MCS duly
#: returns it at full coverage; decorating that is generating from scratch
#: with extra steps.
_MIN_CORE_ATOMS = 6


def _rdkit():
    """RDKit, or None where it is not installed.

    Returned rather than imported at module scope so this module can be
    imported -- and its tools registered -- in an image that has no RDKit.
    A tool that then refuses with a clear reason is better than an extension
    that fails to load and takes its siblings' registration down with it.
    """

    try:
        from rdkit import Chem, RDLogger  # noqa: PLC0415

        RDLogger.DisableLog("rdApp.*")
        return Chem
    except ImportError:
        return None


def _parse(smiles: list[str]) -> tuple[list[Any], list[str]]:
    """The molecules that parse, and the strings that did not."""

    Chem = _rdkit()
    mols, bad = [], []
    for s in smiles:
        text = (s or "").strip()
        if not text:
            continue
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            bad.append(text)
        else:
            mol.SetProp("_input", text)
            mols.append(mol)
    return mols, bad


def _murcko_core(mols: list[Any]) -> tuple[Optional[Any], int]:
    """The most common Bemis-Murcko scaffold, and how many molecules have it."""

    Chem = _rdkit()
    from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: PLC0415

    counts: Counter[str] = Counter()
    for mol in mols:
        try:
            scaf = MurckoScaffold.GetScaffoldForMol(mol)
        except Exception:  # noqa: BLE001
            continue
        if scaf is not None and scaf.GetNumAtoms():
            counts[Chem.MolToSmiles(scaf)] += 1

    if not counts:
        return None, 0
    smiles, n = counts.most_common(1)[0]
    return Chem.MolFromSmiles(smiles), n


def _mcs_core(mols: list[Any], timeout: int) -> tuple[Optional[Any], int]:
    """The maximum common substructure, as a molecule rather than a pattern.

    The MCS itself is a SMARTS, which cannot be decorated or drawn. It is
    mapped back onto a real molecule here so what comes out is a structure.
    """

    Chem = _rdkit()
    from rdkit.Chem import rdFMCS  # noqa: PLC0415

    if len(mols) < 2:
        return None, 0

    result = rdFMCS.FindMCS(
        mols,
        timeout=timeout,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        bondCompare=rdFMCS.BondCompare.CompareAny,
    )
    if result.canceled and not result.smartsString:
        return None, 0
    patt = Chem.MolFromSmarts(result.smartsString)
    if patt is None or patt.GetNumAtoms() == 0:
        return None, 0

    # Lifted off a real molecule so the result is a sanitizable structure,
    # not a query. Tried in turn because a fragment cut out of one molecule
    # can be chemically invalid on its own.
    for mol in mols:
        match = mol.GetSubstructMatch(patt)
        if not match:
            continue
        core = _core_from_match(Chem, mol, match)
        if core is not None:
            n = sum(1 for m in mols if m.HasSubstructMatch(core))
            return core, n
    return None, 0


def _core_from_match(Chem: Any, mol: Any, match: tuple[int, ...]) -> Optional[Any]:
    """The matched atoms of *mol* as a molecule in their own right.

    An MCS may match a ring atom without the rest of its ring -- the SMARTS
    asks only that the atom *be* in a ring, not that the ring be shared. Cut
    out as-is that atom is written aromatic and unbonded, giving SMILES like
    ``cC(=O)Nc1ccccc1`` that nothing can read, so every MCS answer on such a
    series came back as "no common core". Partial rings are therefore dropped
    before the fragment is written.
    """

    keep = set(match)
    rings = mol.GetRingInfo().AtomRings()
    dropping = True
    while dropping:
        dropping = False
        for ring in rings:
            shared = keep.intersection(ring)
            if shared and len(shared) != len(ring):
                keep -= shared
                dropping = True
    if not keep:
        return None

    core = Chem.MolFromSmiles(
        Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(keep), canonical=True)
    )
    if core is None:
        return None
    # Dropping a ring can leave the rest in disconnected pieces, and a core
    # is one structure.
    pieces = Chem.GetMolFrags(core, asMols=True, sanitizeFrags=False)
    if len(pieces) > 1:
        core = max(pieces, key=lambda m: m.GetNumHeavyAtoms())
    return core


def _open_positions(core: Any, mols: list[Any]) -> dict[int, list[str]]:
    """Core atoms that carry something in at least one molecule.

    Keyed by atom index in *core*, valued by the substituent SMILES seen
    there. A position nothing ever substitutes is not an attachment point:
    marking it would tell LibInvent to decorate a place the series never
    does.
    """

    Chem = _rdkit()
    found: dict[int, list[str]] = defaultdict(list)

    for mol in mols:
        match = mol.GetSubstructMatch(core)
        if not match:
            continue
        in_core = set(match)
        for core_idx, mol_idx in enumerate(match):
            for nb in mol.GetAtomWithIdx(mol_idx).GetNeighbors():
                if nb.GetIdx() in in_core:
                    continue
                found[core_idx].append(_substituent(Chem, mol, nb.GetIdx(), in_core))
    return dict(found)


def _substituent(Chem: Any, mol: Any, start: int, in_core: set[int]) -> str:
    """The branch hanging off the core at *start*, as SMILES."""

    seen, stack = set(), [start]
    while stack:
        idx = stack.pop()
        if idx in seen or idx in in_core:
            continue
        seen.add(idx)
        for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
            if nb.GetIdx() not in in_core:
                stack.append(nb.GetIdx())
    try:
        return Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(seen), canonical=True)
    except Exception:  # noqa: BLE001
        return "?"


def _mark(core: Any, positions: list[int]) -> Optional[str]:
    """*core* with a numbered attachment point at each position."""

    Chem = _rdkit()
    rw = Chem.RWMol(core)
    for number, core_idx in enumerate(sorted(positions), start=1):
        dummy = Chem.Atom(0)
        dummy.SetAtomMapNum(number)
        new_idx = rw.AddAtom(dummy)
        rw.AddBond(core_idx, new_idx, Chem.BondType.SINGLE)
    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:  # noqa: BLE001
        return None
    return Chem.MolToSmiles(out)


def derive_scaffold(
    smiles: list[str],
    method: str = "auto",
    attachment_points: bool = True,
    min_fraction: float = _DEFAULT_MIN_FRACTION,
) -> dict[str, Any]:
    """The core a set of molecules shares, ready to decorate.

    Args:
        smiles: The series. Two or more molecules; one has no common core.
        method: ``murcko`` (ring systems and their linkers, fast and the
            usual answer), ``mcs`` (largest shared substructure, slower and
            keeps shared side chains), or ``auto`` -- Murcko first, falling
            back to MCS when the Murcko core is shared too narrowly.
        attachment_points: Mark the substituted positions as ``[*:n]``. Off
            gives the bare core.
        min_fraction: How much of the series a Murcko core must cover before
            ``auto`` accepts it.

    Returns:
        ``{scaffold, method, n_input, n_matched, coverage, attachment_points,
        r_group_counts, unparsed, note}``. ``scaffold`` goes straight into a
        LibInvent ``generation.libinvent.scaffold``.
    """

    if _rdkit() is None:
        return {"error": "RDKit is not available here, so a scaffold cannot be derived."}

    mols, bad = _parse(list(smiles or []))
    if len(mols) < 2:
        return {
            "error": (
                f"Need at least 2 parsable molecules to find a shared core; got {len(mols)}."
                + (f" Unparsable: {bad[:5]}." if bad else "")
            )
        }

    note: list[str] = []
    if method not in ("auto", "murcko", "mcs"):
        return {"error": f"Unknown method {method!r}; use 'auto', 'murcko' or 'mcs'."}

    core, n_matched, used = None, 0, method
    if method in ("auto", "murcko"):
        core, n_matched = _murcko_core(mols)
        used = "murcko"
        if method == "auto" and (core is None or n_matched / len(mols) < min_fraction):
            note.append(
                f"The most common Murcko scaffold covers only {n_matched}/{len(mols)}; "
                "fell back to the maximum common substructure."
            )
            core, n_matched, used = None, 0, "mcs"

    if core is None and used == "mcs":
        core, n_matched = _mcs_core(mols, _MCS_TIMEOUT)

    if core is None:
        return {
            "error": "No common core found.",
            "n_input": len(mols),
            "note": "The molecules may be too dissimilar to share a scaffold.",
        }

    Chem = _rdkit()
    bare = Chem.MolToSmiles(core)
    if core.GetNumHeavyAtoms() < _MIN_CORE_ATOMS:
        note.append(
            f"The core found has only {core.GetNumHeavyAtoms()} heavy atoms, which is "
            "too little to be a series scaffold -- these molecules have almost nothing "
            "in common. Decorating it would be de novo generation in disguise."
        )
    positions = _open_positions(core, mols)

    scaffold = bare
    if attachment_points:
        if not positions:
            note.append(
                "No substituted position found: every molecule is the core itself, "
                "so there is nothing for LibInvent to decorate."
            )
        else:
            marked = _mark(core, list(positions))
            if marked is None:
                note.append("Attachment points could not be added; returning the bare core.")
            else:
                scaffold = marked

    r_counts = {
        f"R{n}": len(set(positions[idx]))
        for n, idx in enumerate(sorted(positions), start=1)
    }

    return {
        "scaffold": scaffold,
        "bare_scaffold": bare,
        "method": used,
        "n_input": len(mols),
        "n_matched": n_matched,
        "coverage": round(n_matched / len(mols), 3),
        "attachment_points": len(positions) if attachment_points else 0,
        "r_group_counts": r_counts,
        "unparsed": bad,
        "note": " ".join(note) or None,
    }


def decompose_series(
    smiles: list[str],
    scaffold: Optional[str] = None,
) -> dict[str, Any]:
    """What each molecule hangs off each position of a shared core.

    The R-group table. Read down a column to see what a series has explored
    at one position, which is the comparison that SAR is read from and the
    one that is otherwise done by eye.

    Args:
        smiles: The series.
        scaffold: The core to decompose around, with or without attachment
            points. Derived with ``derive_scaffold`` when omitted.

    Returns:
        ``{scaffold, positions, rows, unmatched, unparsed}``. ``positions``
        names each R position with the distinct substituents seen there and
        how often; ``rows`` is one entry per molecule.
    """

    Chem = _rdkit()
    if Chem is None:
        return {"error": "RDKit is not available here, so a series cannot be decomposed."}

    mols, bad = _parse(list(smiles or []))
    if not mols:
        return {"error": "No parsable molecules given."}

    if scaffold:
        core = Chem.MolFromSmiles(scaffold.strip()) or Chem.MolFromSmarts(scaffold.strip())
        if core is None:
            return {"error": f"Cannot parse the scaffold {scaffold!r}."}
        # Attachment points mark where a generator may add, which is not part
        # of the substructure to match. Dropped here so a scaffold can come
        # straight back from derive_scaffold without being edited first.
        core = _strip_dummies(Chem, core)
    else:
        derived = derive_scaffold(smiles, attachment_points=False)
        if "error" in derived:
            return derived
        core = Chem.MolFromSmiles(derived["bare_scaffold"])

    positions = _open_positions(core, mols)
    order = sorted(positions)
    labels = {idx: f"R{n}" for n, idx in enumerate(order, start=1)}

    rows, unmatched = [], []
    for mol in mols:
        match = mol.GetSubstructMatch(core)
        source = mol.GetProp("_input") if mol.HasProp("_input") else Chem.MolToSmiles(mol)
        if not match:
            unmatched.append(source)
            continue
        in_core = set(match)
        groups: dict[str, list[str]] = defaultdict(list)
        for core_idx, mol_idx in enumerate(match):
            if core_idx not in labels:
                continue
            for nb in mol.GetAtomWithIdx(mol_idx).GetNeighbors():
                if nb.GetIdx() not in in_core:
                    groups[labels[core_idx]].append(_substituent(Chem, mol, nb.GetIdx(), in_core))
        rows.append({
            "smiles": source,
            "r_groups": {k: (v[0] if len(v) == 1 else v) for k, v in groups.items()} or {"R0": "H"},
        })

    summary = {}
    for idx in order:
        counts = Counter(positions[idx])
        summary[labels[idx]] = {
            "distinct": len(counts),
            "most_common": [
                {"group": g, "n": n} for g, n in counts.most_common(_MAX_R_GROUPS)
            ],
        }

    return {
        "scaffold": Chem.MolToSmiles(core),
        "positions": summary,
        "rows": rows,
        "unmatched": unmatched,
        "unparsed": bad,
    }


def _strip_dummies(Chem: Any, mol: Any) -> Any:
    """*mol* without its attachment-point atoms, for substructure matching."""

    rw = Chem.RWMol(mol)
    for idx in sorted(
        (a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0), reverse=True
    ):
        rw.RemoveAtom(idx)
    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:  # noqa: BLE001
        return mol
    return out


def register_tools(mcp: Any) -> None:
    """Register the scaffold tools."""

    @mcp.tool()
    def derive_scaffold_from_series(
        smiles: list[str],
        method: str = "auto",
        attachment_points: bool = True,
        min_fraction: float = _DEFAULT_MIN_FRACTION,
    ) -> dict[str, Any]:
        """Find the core a set of molecules shares, ready for LibInvent.

        Use this instead of reading a series and writing the scaffold by
        hand. The result's ``scaffold`` carries numbered ``[*:n]`` attachment
        points at the positions the series actually substitutes, which is the
        form ``generation.libinvent.scaffold`` requires -- a bare ``[*]`` is
        refused there.

        Typical chain: ``fetch_all_activities`` for a target, the actives'
        SMILES into this, the scaffold into ``create_ttp``, then
        ``run_generation_campaign``.

        Args:
            smiles: The series; two or more molecules.
            method: ``murcko``, ``mcs``, or ``auto`` (Murcko, falling back to
                MCS when the Murcko core is shared too narrowly).
            attachment_points: Mark substituted positions as ``[*:n]``.
            min_fraction: Coverage ``auto`` requires before accepting Murcko.

        Returns:
            ``{scaffold, method, coverage, attachment_points, r_group_counts,
            …}``. Check ``coverage``: a low one means the set is not one
            series.
        """

        return derive_scaffold(
            smiles=smiles,
            method=method,
            attachment_points=attachment_points,
            min_fraction=min_fraction,
        )

    @mcp.tool()
    def decompose_series_r_groups(
        smiles: list[str],
        scaffold: Optional[str] = None,
    ) -> dict[str, Any]:
        """Tabulate what each molecule carries at each position of a core.

        The R-group table. Use it to answer what a series has explored at a
        position and what it has not, and to compare molecules that differ at
        one point -- the reading that is otherwise done by eye.

        Args:
            smiles: The series.
            scaffold: The core, with or without attachment points. Derived
                automatically when omitted.

        Returns:
            ``{scaffold, positions, rows, unmatched}``. ``positions`` gives
            the distinct substituents per R position with their counts;
            ``rows`` is one per molecule.
        """

        return decompose_series(smiles=smiles, scaffold=scaffold)
