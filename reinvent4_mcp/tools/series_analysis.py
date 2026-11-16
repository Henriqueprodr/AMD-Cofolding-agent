# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Series-level reasoning: matched molecular pairs and diversity clustering.

Both tools operate on data the system already holds — the actives from
ChEMBL, the winners from a completed run — without requiring anything new.
Until now SAR had to be eyeballed and "give me ten diverse winners" got an
impressionistic answer. These are the two operations that make both concrete.

``compute_matched_molecular_pairs``
    Every pair of molecules in a set that differ by exactly one structural
    change, matched to the activity difference that change produced. Reading
    down the table is the structural-delta-to-activity-delta relationship
    that medicinal chemistry calls SAR.

``cluster_molecules``
    Groups molecules by structural similarity so "the ten most diverse
    winners" is not a synonym for "ten the agent picks". Butina clustering
    (the standard in MedChem) groups by a cutoff; k-means on fingerprints
    gives a fixed count.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: Fingerprint radius for all distance calculations.
_RADIUS = 2
#: Fingerprint bit length.
_NBITS = 2048
#: Maximum pairwise comparisons before the MMP search is capped, to prevent
#: a 50 000-compound run from blocking the event loop for a minute.
_MAX_PAIRS = 100_000
#: Maximum MMP results returned; a table that fills the context is useless.
_MAX_MMP_RESULTS = 500
#: Maximum clusters returned.
_MAX_CLUSTERS = 200


def _rdkit():
    try:
        from rdkit import Chem, RDLogger  # noqa: PLC0415
        RDLogger.DisableLog("rdApp.*")
        return Chem
    except ImportError:
        return None


def _fingerprint(mol: Any) -> Any:
    """Morgan fingerprint as a bit vector."""
    from rdkit.Chem import AllChem  # noqa: PLC0415
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=_RADIUS, nBits=_NBITS)


def _parse_mols(smiles: list[str]) -> tuple[list[Any], list[str]]:
    """The molecules that parse, with their input SMILES stored."""
    Chem = _rdkit()
    mols, bad = [], []
    for s in smiles:
        s = (s or "").strip()
        if not s:
            continue
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            bad.append(s)
        else:
            mol.SetProp("_smi", s)
            mols.append(mol)
    return mols, bad


def _mmp_transform(mol_a: Any, mol_b: Any) -> Optional[str]:
    """The ``before>>after`` change between two molecules, or None.

    Each molecule is cut once at every acyclic single bond; the pair is a
    matched pair where both share a cut that leaves the same core and a
    different environment. One cut only, so a molecule differing in two
    places is deliberately not reported -- the point of a matched pair is
    that one change is responsible for the activity difference.

    An approximation of mmpdb, which indexes a whole database rather than
    comparing a set pairwise, and is what to reach for on a corpus.
    """

    try:
        from rdkit.Chem import rdMMPA  # noqa: PLC0415
    except ImportError:
        return None

    frags_a = rdMMPA.FragmentMol(mol_a, maxCuts=1, resultsAsMols=False)
    frags_b = rdMMPA.FragmentMol(mol_b, maxCuts=1, resultsAsMols=False)
    if not frags_a or not frags_b:
        return None

    # frags are (core, context) pairs as SMILES; find a matching core.
    cores_a = {c: ctx for c, ctx in frags_a}
    for core_b, ctx_b in frags_b:
        if core_b in cores_a:
            ctx_a = cores_a[core_b]
            if ctx_a != ctx_b:
                return f"{ctx_a}>>{ctx_b}"
    return None


def compute_matched_molecular_pairs(
    smiles: list[str],
    activities: Optional[dict[str, float]] = None,
    max_results: int = _MAX_MMP_RESULTS,
) -> dict[str, Any]:
    """Pairs of molecules that differ by exactly one structural change.

    The result is the R-group SAR table's complement: where
    ``decompose_series_r_groups`` shows what sits at a position, this shows
    which single-atom or single-fragment swap produced how much of an
    activity change.

    Args:
        smiles: The series to compare pairwise.
        activities: Optional pIC50 (or similar) keyed by SMILES. When given,
            each pair carries the delta. Leave out if only the structural
            relationship matters.
        max_results: Pairs to return; the most different by activity first.

    Returns:
        ``{pairs, n_total_pairs, n_mmp_pairs, unparsed}``. Each entry in
        ``pairs`` has ``smiles_a``, ``smiles_b``, ``transform``, and
        optionally ``delta_activity``.
    """

    Chem = _rdkit()
    if Chem is None:
        return {"error": "RDKit is not available; matched molecular pairs need RDKit."}

    mols, bad = _parse_mols(list(smiles or []))
    if len(mols) < 2:
        return {
            "error": (
                f"Need at least 2 parsable molecules; got {len(mols)}."
                # Without the offending input the caller cannot tell a short
                # series from a series it passed as one joined string.
                + (f" Unparsable: {bad[:5]}." if bad else "")
                + (
                    " Pass one SMILES per list entry."
                    if len(smiles or []) < 2
                    else ""
                )
            )
        }

    # Cap pairwise comparisons before we enumerate them all.
    n_pairs = len(mols) * (len(mols) - 1) // 2
    if n_pairs > _MAX_PAIRS:
        return {
            "error": (
                f"The series has {len(mols)} molecules ({n_pairs:,} pairs). "
                f"That exceeds the {_MAX_PAIRS:,} pair cap. "
                "Filter to actives only or split into sub-series."
            )
        }

    act = activities or {}
    pairs: list[dict[str, Any]] = []

    for i in range(len(mols)):
        for j in range(i + 1, len(mols)):
            transform = _mmp_transform(mols[i], mols[j])
            if transform is None:
                continue
            smi_a = mols[i].GetProp("_smi")
            smi_b = mols[j].GetProp("_smi")
            entry: dict[str, Any] = {
                "smiles_a": smi_a,
                "smiles_b": smi_b,
                "transform": transform,
            }
            if smi_a in act and smi_b in act:
                entry["activity_a"] = round(act[smi_a], 3)
                entry["activity_b"] = round(act[smi_b], 3)
                entry["delta_activity"] = round(act[smi_b] - act[smi_a], 3)
            pairs.append(entry)

    # Any pair carrying a delta is worth sorting for: activities often cover
    # part of a series, and the pairs that have one are exactly the ones the
    # caller wants at the top.
    if any("delta_activity" in p for p in pairs):
        pairs.sort(key=lambda p: abs(p.get("delta_activity", 0)), reverse=True)

    return {
        "pairs": pairs[:max_results],
        "n_total_pairs": n_pairs,
        "n_mmp_pairs": len(pairs),
        "unparsed": bad,
        "note": (
            f"Showing {min(len(pairs), max_results)} of {len(pairs)} MMP pairs."
            if len(pairs) > max_results
            else None
        ),
    }


def cluster_molecules(
    smiles: list[str],
    method: str = "butina",
    n_clusters: Optional[int] = None,
    cutoff: float = 0.4,
    pick_representative: bool = True,
) -> dict[str, Any]:
    """Group molecules by structural similarity.

    Args:
        smiles: The molecules to cluster.
        method: ``butina`` (distance cutoff, the standard in MedChem) or
            ``kmedoids`` (fixed number of clusters).
        n_clusters: Number of clusters for ``kmedoids``. Ignored for
            ``butina``.
        cutoff: Tanimoto distance cutoff for ``butina``. 0.4 is loose
            (diverse), 0.2 is tight (similar compounds one cluster).
        pick_representative: Return the centroid molecule per cluster.

    Returns:
        ``{clusters, n_clusters, method, representatives, unparsed}``.
        ``representatives`` is the centroid SMILES per cluster.
    """

    Chem = _rdkit()
    if Chem is None:
        return {"error": "RDKit is not available; clustering needs RDKit."}

    mols, bad = _parse_mols(list(smiles or []))
    if not mols:
        return {"error": "No parsable molecules given."}

    if method not in ("butina", "kmedoids"):
        return {"error": f"Unknown method {method!r}; use 'butina' or 'kmedoids'."}

    fps = [_fingerprint(m) for m in mols]
    smi_list = [m.GetProp("_smi") for m in mols]

    if method == "butina":
        clusters_raw = _butina(fps, cutoff)
    else:
        k = n_clusters or min(10, len(mols))
        clusters_raw = _kmedoids(fps, k)

    if len(clusters_raw) > _MAX_CLUSTERS:
        clusters_raw = clusters_raw[:_MAX_CLUSTERS]

    clusters = []
    reps: list[str] = []
    for indices in clusters_raw:
        members = [smi_list[i] for i in indices]
        clusters.append({"size": len(members), "members": members})
        if pick_representative:
            reps.append(members[0])  # Butina and kmedoids both put the centroid first.

    return {
        "clusters": clusters,
        "n_clusters": len(clusters),
        "method": method,
        "representatives": reps if pick_representative else [],
        "unparsed": bad,
    }


def _butina(fps: list[Any], cutoff: float) -> list[list[int]]:
    """Butina clustering (Taylor-Butina)."""
    from rdkit.ML.Cluster import Butina  # noqa: PLC0415
    from rdkit import DataStructs  # noqa: PLC0415

    n = len(fps)
    # Lower-triangle distance matrix.
    dists = []
    for i in range(1, n):
        ds = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend(1 - s for s in ds)

    cs = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    return [list(c) for c in cs]


def _kmedoids(fps: list[Any], k: int) -> list[list[int]]:
    """k-medoids on Tanimoto distance — no numpy required."""
    from rdkit import DataStructs  # noqa: PLC0415
    import random  # noqa: PLC0415

    n = len(fps)
    k = min(k, n)
    random.seed(42)

    # Pairwise Tanimoto similarities as a symmetric matrix. The diagonal has
    # to be set: assignment picks the most similar medoid, so a medoid left
    # at self-similarity 0 is the worst match for its own cluster and gets
    # assigned to somebody else's, landing in two clusters at once.
    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        sim[i][i] = 1.0
        for j in range(i + 1, n):
            s = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            sim[i][j] = sim[j][i] = s

    # Seed by maximin: each new medoid is the molecule furthest from every
    # medoid already chosen, so the starting points span the set instead of
    # landing together in its densest region. The candidate's similarity to
    # its *nearest* medoid is what has to be small -- being far from one
    # medoid means nothing if it sits on top of another.
    medoids = [random.randrange(n)]
    for _ in range(k - 1):
        best, best_sim = -1, 2.0
        for cand in range(n):
            if cand in medoids:
                continue
            nearest = max(sim[cand][m] for m in medoids)
            if nearest < best_sim:
                best, best_sim = cand, nearest
        if best < 0:
            break
        medoids.append(best)
    k = len(medoids)

    for _ in range(50):  # iterations
        # Assign each molecule to its nearest medoid.
        assignment = [max(range(k), key=lambda m: sim[i][medoids[m]]) for i in range(n)]
        # Update medoids.
        new_medoids = []
        changed = False
        for ci in range(k):
            members = [i for i, a in enumerate(assignment) if a == ci]
            if not members:
                new_medoids.append(medoids[ci])
                continue
            new_m = max(members, key=lambda i: sum(sim[i][j] for j in members))
            if new_m != medoids[ci]:
                changed = True
            new_medoids.append(new_m)
        medoids = new_medoids
        if not changed:
            break

    # Build cluster lists with centroid first.
    assignment = [max(range(k), key=lambda m: sim[i][medoids[m]]) for i in range(n)]
    clusters: dict[int, list[int]] = {m: [m] for m in medoids}
    for i, ci in enumerate(assignment):
        m = medoids[ci]
        if i != m:
            clusters.setdefault(m, [m]).append(i)
    return [clusters[m] for m in medoids if m in clusters]


def register_tools(mcp: Any) -> None:
    """Register the series-analysis tools."""

    @mcp.tool()
    def find_matched_molecular_pairs(
        smiles: list[str],
        activities: Optional[dict[str, float]] = None,
        max_results: int = _MAX_MMP_RESULTS,
    ) -> dict[str, Any]:
        """Find every pair that differs by exactly one structural change.

        The SAR reading that is otherwise done by eye. Each row names the
        two molecules, the structural transformation (context>>change), and
        the activity delta when ``activities`` is given. Sort by
        ``|delta_activity|`` to find the most consequential changes.

        Typical chain: ``fetch_all_activities`` for the target, pass the
        active SMILES and their pIC50 as ``activities``, read the table.

        Args:
            smiles: The series to compare pairwise (up to ~450 molecules;
                above that the pair count exceeds the cap).
            activities: pIC50 (or similar) keyed by SMILES. Leave out if
                only the structural relationship matters.
            max_results: Pairs to return; most different by activity first.

        Returns:
            ``{pairs, n_total_pairs, n_mmp_pairs, unparsed}``.
        """

        return compute_matched_molecular_pairs(
            smiles=smiles, activities=activities, max_results=max_results
        )

    @mcp.tool()
    def cluster_molecules_by_similarity(
        smiles: list[str],
        method: str = "butina",
        n_clusters: Optional[int] = None,
        cutoff: float = 0.4,
        pick_representative: bool = True,
    ) -> dict[str, Any]:
        """Group molecules by structural similarity.

        Use this to answer "give me ten diverse winners" or "how spread out
        is this run's output" with actual clusters rather than an
        impressionistic pick. The ``representatives`` list is the centroid
        of each cluster, the most typical molecule in it.

        Args:
            smiles: The molecules to cluster.
            method: ``butina`` (cutoff-based, standard MedChem) or
                ``kmedoids`` (fixed number of groups).
            n_clusters: Number of groups for ``kmedoids``.
            cutoff: Tanimoto distance cutoff for ``butina``. 0.4 is loose
                (diverse clusters), 0.2 is tight (similar compounds share
                a cluster). Increase when too many clusters appear.
            pick_representative: Return one centroid per cluster.

        Returns:
            ``{clusters, n_clusters, method, representatives}``.
        """

        return cluster_molecules(
            smiles=smiles,
            method=method,
            n_clusters=n_clusters,
            cutoff=cutoff,
            pick_representative=pick_representative,
        )
