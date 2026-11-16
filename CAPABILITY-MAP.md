# Capability map, by discipline

What the five people on a small-molecule project would actually ask of this
system, whether it can answer, and — where it cannot — whether the gap is a
missing tool or a missing chain.

Written by walking the registered tool surface rather than the documentation:
**111 tools**, 58 declared in `reinvent4_mcp/server.py` and 53 across the
fourteen extension modules in `reinvent4_mcp/registry.py`. Where this says
something is missing, it means no tool and no service provides it, not that it
is undocumented.

Companion notes: [`CURATION-IN-TTP.md`](CURATION-IN-TTP.md) for the
provenance gap, [`TESTING.md`](TESTING.md) for what is covered,
[`README.md`](README.md) for the catalogue itself.

### How to read the tables

| Mark | Meaning |
|---|---|
| **Direct** | One tool answers it. |
| **Chain** | Possible today by composing tools; the chain is given. |
| **Partial** | Possible with a caveat that changes what the answer is worth. |
| **Missing** | No tool and no service. What would be needed is named. |

Two things apply everywhere and are not repeated in each row. Several tools
are asynchronous — `prepare_structure`, the synthesis planner, every run —
and return a handle to poll, so any chain containing one spans turns. And a
chain can now be declared rather than driven turn by turn: a TTP carries an
optional `pipeline` of steps, each naming a tool, its params and what it
depends on, which `run_ttp_pipeline` executes in topological order. Every one
of the 111 tools is reachable from a step.

---

## 1. Medicinal chemist

The best-served discipline. Design, property prediction, synthesis and
novelty are all first-class; the gaps are in *reasoning over a series* rather
than acting on a molecule.

| Question | | How |
|---|---|---|
| What has been tried against this target? | **Chain** | `search_chembl_targets` → `get_chembl_target` → `fetch_all_activities`; add `get_chembl_mechanisms` and `get_chembl_documents` for precedent |
| What are this compound's properties? | **Direct** | `score_molecules` with components (MW, logP, TPSA, QED, SA, Tanimoto to a reference…) — it is a general calculator for any component the REINVENT build exposes, listed by `list_scoring_components` |
| What is its ADMET risk? | **Direct** | `predict_admet` — 17 drug endpoints (logP, logD, both solubilities, Caco-2, two efflux ratios, four CYPs, hERG, two half-lives, PPB), with applicability-domain flags |
| How potent on my project target? | **Partial** | `predict_retrodmta`, but only for targets in the deployed manifest; an unlisted target fails the call |
| Make analogues of my lead | **Direct** | `run_optimization` with a Mol2Mol prior |
| Decorate this scaffold / link these fragments | **Chain** | `create_ttp` with a `generation` section → `run_generation_campaign`. Note `run_optimization` builds de novo and Mol2Mol only and now refuses these by name |
| Is this novel, or already known? | **Direct** | `check_compound_is_known` — exact structure in PubChem *and* ChEMBL by InChIKey |
| Is it patented? | **Direct** | `search_patents_by_structure` (SureChEMBL by SMILES, exact or similarity) and `search_patents_by_target`. Composition-of-matter claims, not the registry novelty `check_compound_is_known` answers. An unreachable database returns an error saying patent status is *unknown*, never clean |
| Can it be made, and from what? | **Direct** | `plan_synthesis`, `check_building_block_availability` |
| **What is the common scaffold of these actives?** | **Direct** | `derive_scaffold_from_series` — Murcko, falling back to the MCS when no two members share one, returned with numbered `[*:n]` points at the positions the series really varies, so it is LibInvent input as it stands. A core too small to be a scaffold is reported as such rather than returned at full coverage |
| **What did adding this group do, across the series?** | **Direct** | `find_matched_molecular_pairs` reports every pair differing by one change with its transform and, given activities, the delta; `decompose_series_r_groups` gives the R-group table at each position |
| Ligand efficiency / LLE | **Chain** | Potency from `fetch_all_activities` or `ingest_fragment_activities`, properties from `score_molecules`; the prompt defines LE and LLE and the agent computes them |
| Why is this compound better than that one? | **Chain** | `profile_interactions` on both + a score comparison; sound when both have structures, interpretive otherwise |
| Selectivity against related targets | **Chain** | `search_chembl_targets` then `fetch_all_activities` per target — one call each, no panel tool, and nothing reconciles differing assays |
| Does it trip a structural alert (PAINS…)? | **Partial** | Only if the REINVENT build ships an alerts component; `list_scoring_components` is the authority |
| Build a QSAR model on my own data | **Missing** | `run_transfer_learning` re-trains a *generative prior*, not a predictor. No fit/predict surface |
| What should we make next? | **Direct** | `get_top_candidates`, with the campaign tools for the wider loop |

## 2. Biologist

Target biology is well covered where ChEMBL and UniProt cover it, and absent
beyond them. There is no systems-biology layer at all.

| Question | | How |
|---|---|---|
| What is this protein and what does it do? | **Direct** | `fetch_target_info`, `fetch_uniprot_features` |
| Its domains and families? | **Direct** | `fetch_interpro_annotations` |
| Which diseases is it implicated in? | **Direct** | `get_chembl_indications` |
| How do known drugs act on it? | **Direct** | `get_chembl_mechanisms` |
| What assays produced these numbers? | **Direct** | `search_chembl_assays` — essential before comparing activities, and easy to skip |
| What should I read? | **Chain** | `get_chembl_documents`, `rcsb_get_pubmed` for the structures' citations |
| Which isoform does this structure cover? | **Chain** | `rcsb_get_entries` → `fetch_uniprot_features`, compared by hand |
| Where is it expressed? | **Direct** | `fetch_target_expression` — Human Protein Atlas, bulk tissue RNA or single-cell, so a target expressed in the wrong tissue is visible before any chemistry |
| What pathway is it in, what does it interact with? | **Direct** | `fetch_pathway_annotations` (Reactome via OpenTargets) |
| Species differences / orthologs? | **Direct** | `fetch_target_orthology` maps the accession across species, which is what reasoning about an animal model needs |
| Is the binding site conserved across the family? | **Missing** | MSAs exist only as Boltz input (`msa_path`); nothing computes conservation. Orthology answers the species question, not the family one |
| Knockout or genetic-validation evidence? | **Direct** | `fetch_target_genetic_evidence` — GWAS, pheWAS and rare-variant evidence from OpenTargets. Human genetics is the evidence that most predicts a target surviving the clinic, and nothing else here carries it |

## 3. Structural biologist

Also well served — this is the deepest part of the stack, with real
preparation rather than a naive strip-and-go.

| Question | | How |
|---|---|---|
| What structures exist, and how many? | **Direct** | `rcsb_search`, `rcsb_count` |
| Resolution, method, ligands, citations? | **Direct** | `rcsb_get_entries`, `rcsb_get_chem_comps`, `rcsb_get_pubmed` |
| Fetch them | **Direct** | `rcsb_fetch_files` |
| What is wrong with this file before I use it? | **Direct** | `validate_structure` — inventory plus a preview that *surfaces ambiguity* (chain roles, ligand vs additive, altlocs) instead of guessing |
| Clean it properly | **Chain** | `prepare_structure` with a profile → poll `get_preparation_status`; assembly, altlocs, missing atoms, protonation at pH, ligand force field |
| Get the ligand out | **Direct** | `extract_ligand_sdf` |
| Translate author numbering | **Direct** | `resolve_author_residue` — the quiet source of off-by-one errors in constraints |
| What is the pocket like? | **Direct** | `characterize_site_profile` (fpocket; now reported in `/health`, so an image without it can no longer look healthy while returning nothing) |
| What interactions does this complex make? | **Direct** | `profile_interactions`, `profile_complex_interaction` |
| What is conserved across *all* complexes? | **Direct** | `profile_complex_interaction(mode="hotspot_ensemble")` — hundreds of structures, 30-minute budget, returning per-residue occupancy, fragment families and compatible pairs |
| Predict a complex | **Direct** | `predict_structure_conditioned`, or a Boltz objective inside a TTP |
| Is this pose physically sensible? | **Direct** | `validate_pose` (PoseBusters); the profiler also reports ligand strain |
| Superpose and compare conformations | **Partial** | The UI superposes; the agent has no tool, so cross-structure geometry must be inferred from profiles |
| Conserved or bridging waters? | **Partial** | Preparation can *keep* pocket waters; nothing analyses them |
| Apo vs holo, cryptic pockets? | **Missing** | Would need pocket detection across an ensemble and a comparison step |
| Does the ligand fit the density? | **Missing** | No density or validation-report access |

## 4. Computational chemist

The discipline with the largest single hole. Generation, scoring and
structure prediction are strong; classical structure-based modelling is
absent.

| Task | | How |
|---|---|---|
| **Dock a library into the site** | **Missing** | No docking tool or service anywhere, although the prompt tells the agent to prepare `receptor_noh` *for GNINA* and the preparation emits exactly the right artifacts. See §6.2 |
| Cofold with the target | **Direct** | Boltz, via a TTP objective or `predict_structure_conditioned` |
| Force a ligand into the known site | **Direct** | `target.binding_site_residues` + `binding_site_force` in the TTP; the mapper emits `pocket_constraints` with `force`, and it reaches scaffold and linker runs too |
| Covalent design | **Partial** | `target.covalent_bond` constrains the bond and suppresses affinity prediction; there is no covalent docking or warhead reactivity model |
| Score an arbitrary library | **Direct** | `score_molecules`, components or a whole TTP |
| Build a multi-objective function | **Direct** | `create_ttp`, with `list_scoring_components` and `list_transforms` for what is available; unknown component names are now refused where the TOML is written, with the closest match suggested |
| Run RL generation and watch it | **Chain** | `run_optimization` → `get_optimization_status` → `read_results` / `get_top_candidates` |
| Bias a prior toward a focused set | **Direct** | `run_transfer_learning` |
| Rescore with physics (MM-GBSA, FEP) | **Missing** | No free-energy or MD capability of any kind |
| Generate conformers | **Partial** | Embedding happens inside preparation and cofolding; no standalone tool |
| Cluster output, measure diversity | **Direct** | `cluster_molecules_by_similarity` — Butina on a Tanimoto cutoff, or k-medoids for a fixed count, returning the centroid per cluster, so "ten diverse winners" is a named list. The UI's UMAP and chemical-space views are the same question answered for a human |
| Pharmacophore or shape search | **Missing** | No ROCS-like tool |
| Is the prediction in domain? | **Direct** | ADMET carries applicability-domain and response-domain flags, and RetroDMTA's inter-model spread is documented as the substitute |
| Reproduce last quarter's run | **Partial** | The TTP and manifest capture the run; the structure's provenance is not captured — see [`CURATION-IN-TTP.md`](CURATION-IN-TTP.md) |

## 5. Analytical chemist

The least served, and largely by design: the system has no instrument data
and no spectral models. Worth stating explicitly so nobody assumes otherwise.

| Question | | How |
|---|---|---|
| LogD for method development | **Direct** | `predict_admet` endpoint `logd` |
| Solubility for formulation | **Direct** | Both `aqueous_solubility` and `kinetic_solubility`, with the distinction (and the DMSO ceiling) documented |
| Confirm I have the right compound | **Partial** | `lookup_compound_by_structure` and `check_compound_is_known` confirm identity against databases — not against a measurement |
| Predicted pKa | **Missing** | Protonation is handled inside preparation and profiling (`pdb2pqr`/`propka` for protein, `dimorphite_dl` for ligands) but no pKa is ever returned |
| Predict an NMR or MS spectrum | **Missing** | No spectral prediction |
| Likely fragmentation or degradants | **Missing** | No fragmentation or forced-degradation model |
| Chromatographic retention | **Missing** | No retention model |
| Load my measured data and compare | **Partial** | `ingest_fragment_activities` takes a CSV but is scoped to fragment potency; there is no general assay-ingest path, so measurements mostly cannot re-enter the system |

---

## 6. The gaps that matter, in order

### 6.1 Docking — *the largest capability hole*

There is no docking anywhere, yet `prepare_structure` emits a GNINA-ready
receptor and the prompt instructs the agent to produce it. The preparation
half of the integration is built and the consumer never arrived. Cofolding
partly covers the need and is in some ways better, but it is far more
expensive per molecule, which makes "triage ten thousand compounds" currently
unanswerable.

### 6.2 Binding-site conservation across a family

`fetch_target_orthology` answers the species question — is the mouse target
the same target — but not the family one. MSAs exist only as Boltz input
(`msa_path`) and nothing computes conservation from them, so "is this pocket
the selectivity handle or the thing every family member shares" has no
implementation.

### 6.3 A predictor on your own data

`run_transfer_learning` re-trains a *generative prior*, not a predictor.
There is no fit/predict surface, so a project's own assay data cannot become
a model the RL loop scores against; it can only be read back through
`fetch_all_activities` or `ingest_fragment_activities`.

### 6.4 Preparation provenance

See [`CURATION-IN-TTP.md`](CURATION-IN-TTP.md). A TTP names the structure a
profile scored against but not the decisions that produced it, so a
structure-based profile is not reproducible from the TTP alone.

---

## 7. Chaining: the cross-cutting constraint

A chain can be executed two ways. The agent can call one tool, read the
result and decide the next call — flexible, and why much of the table says
Chain rather than Missing. Or the chain can be *declared*: a TTP carries an
optional `pipeline`, an ordered list of steps each naming a tool, its params,
what it depends on and where to store its result, which `run_ttp_pipeline`
executes in topological order. `${step.field}` interpolation passes one
step's output into the next. All 111 tools are reachable from a step, a
Reviewer may inspect a plan with `dry_run=true`, and executing it is Driver
work. Three limits remain:

**Intermediates live in context.** Shared-volume paths and job ids are held
in the transcript. A long chain is therefore bounded by the context window
and lost on compaction, and a resumed session cannot pick up a half-finished
preparation.

**Delegation loses its result.** A Driver may run the preparation tools —
none are withheld — but `create_ttp` and `update_ttp` are denied it, so work
it does has nowhere to be stored and dies with the task.

**Async steps split turns.** Preparation, synthesis planning and every run
return a handle. A chain crossing two of them is at least three turns, each
an opportunity to lose the thread.

The genuine counterweight is the **atlas**: `atlas_finding` files a claim
under a branch, `atlas_note` supports, disputes or extends it, and
`atlas_overview` / `atlas_branch` / `atlas_search` read it back. It is the one
place a conclusion outlives the agent that reached it, and an empty branch is
visibly the part of the goal nobody has settled. For "do a full analysis", it
is the difference between a long answer and an accumulating one.

## 8. What to build first

1. **A docking service** (§6.1) — the receptor side is already built and the
   prompt already tells the agent to produce a GNINA-ready receptor. The one
   remaining hole that blocks a whole way of working rather than a question.
2. **Preparation provenance** (§6.4, [`CURATION-IN-TTP.md`](CURATION-IN-TTP.md))
   — makes structure-based profiles reproducible, and is a prerequisite for
   trusting anything above at scale.
3. **Binding-site conservation** (§6.2) — the MSAs are already fetched for
   Boltz; nothing reads them for conservation.
4. **A predictor surface** (§6.3) — would let a project's own assay data
   become an RL objective instead of only a reference.
