# Curation in a TTP

A design note on making a structure-based profile say where its structure came
from, rather than only where the file currently sits.

Nothing here is implemented. It is written down because the gap is easy to
look at and conclude nothing is wrong: every individual tool works, and the
workflow succeeds whenever a person drives it attentively.

Related: [`README.md`](README.md) for the tool catalogue,
[`TESTING.md`](TESTING.md) for what is covered, `reinvent4_mcp/ttp/schema.py`
for the profile itself.

---

## 1. What a structure-based campaign takes today

A TTP that scores against a protein structure needs three things in its
`target` section: a `protein_sequence`, usually a `protein_pdb_path`, and
often `binding_site_residues`. None of them can be written by hand. Getting
them is a chain of seven calls:

| # | Call | Produces |
|---|---|---|
| 1 | `rcsb_fetch_files(pdb_id)` | a raw file on the shared volume |
| 2 | `validate_structure(path)` | inventory + `profile_preview` |
| 3 | *(judgement)* | a `PreparationProfile` resolving what step 2 flagged |
| 4 | `prepare_structure(path, profile)` | a `job_id` |
| 5 | `get_preparation_status(job_id)` | artifacts: `complex`, `receptor`, `sequences_fasta`, … |
| 6 | `characterize_site` / `profile_complex_interaction` | pocket residues |
| 7 | `create_ttp({target: {…}})` | the profile, with 5 and 6 pasted into it |

Steps 2–3 are the reason this cannot be one call: `validate_structure` exists
to surface ambiguities — which chain is the target, which HETATM groups are
ligands rather than buffer, which altloc to take — and a default answer to
those is a guess that silently changes the science.

## 2. The gap

`TargetConfig` stores the *outcome* of that chain and never the chain itself.
`protein_pdb_path` is a path on a shared volume. `protein_sequence` is a
string. `binding_site_residues` is a list of residue numbers. The one field
that could record provenance, `pdb_id`, is inert: nothing reads it, nothing
resolves it, and it can disagree with `protein_pdb_path` without complaint.

Four consequences, in the order they bite:

**A profile is not reproducible.** Re-running a TTP two months later reuses a
path. If the shared volume has been cleaned, the run fails with a missing
file. If it has *not* been cleaned, the run succeeds against a structure
nobody can now describe — prepared at which pH, with which waters kept, from
which assembly? The report says `3PXY_prepared.pdb`.

**The same curation is redone per profile.** Two TTPs against one target are
two manual passes through steps 1–6, and nothing notices when they diverge.
Comparing their results then compares two preparations as much as two
objectives.

**An agent holding only the TTP cannot re-derive the structure.** This is what
makes the whole chain a conversation rather than a specification: the
knowledge of how the input was made lives in the transcript, not in the
artifact. A supervisor handed the TTP to run cannot repair a missing file,
because it was never told what the file was.

**Delegated curation cannot be kept.** A Driver *may* run steps 1–6: none of
the preparation tools are withheld from it. What it may not do is
`create_ttp` or `update_ttp`, both denied. So a Driver can prepare a
structure and then has nowhere to put it — the result stays in its own
transcript and dies with the task. In practice the conducting agent must
therefore do all of steps 1–6 itself, and that work is invisible to the
campaign's plan.

`depends_on` does not help here. It orders *scoring objectives* within a
profile — it is validated for cycles, topologically sorted, and consumed by
the scoring plugins. It cannot sequence tools, and giving it that meaning
would overload a field whose current meaning is well defined.

## 3. What it would take

The shape that fits the existing schema is a `preparation` section that
declares provenance and policy, resolved once before a run and never read
during one:

```jsonc
{
  "target": { "protein_sequence": "…" },        // still required, still authoritative
  "preparation": {
    "source":  { "pdb_id": "3PXY", "assembly": 1 },
    "profile": { "version": "1.0", "stage3_curation": { "keep_waters": "pocket" } },
    "site":    { "from": "ligand", "ligand_id": "JQ1" },
    "resolved": {                                // written by the resolver, not by hand
      "prepared_at": "2026-11-14T22:10:00Z",
      "artifacts": { "receptor": "prep/3PXY/abc123/receptor.pdb" },
      "profile_digest": "sha256:…"
    }
  }
}
```

The three parts matter for different reasons. `source` and `profile` make the
preparation a declaration, so two profiles naming the same pair provably share
an input. `resolved` is a cache with a digest: a run checks whether the
artifacts exist and whether the digest still matches the declared profile, and
re-prepares only when one of those fails. `site` says how the pocket was
chosen rather than only which residues came out, which is what lets a pocket
be recomputed when the preparation changes under it.

Three things this must *not* do:

- **Not run during a campaign.** Resolution belongs before the first task is
  queued. A run that discovers mid-flight that it must prepare a structure has
  a GPU allocation waiting on a CPU job.
- **Not make `target` derived.** `protein_sequence` stays required and
  authoritative. A profile with no `preparation` section keeps working exactly
  as it does now, and an operator can still override any field by hand.
- **Not answer step 3 automatically.** The resolver applies a profile it is
  given. If `validate_structure` flags an ambiguity the declared profile does
  not address, it should refuse and say which, not pick a default.

## 4. Alternatives considered

**Leave it to the agent, improve the prompt.** Cheapest, and the current
answer. It does make the chain more reliably executed, but it cannot make a
stored TTP reproducible, because the TTP still does not hold the information.
Worth doing anyway, and independent of this note.

**A separate "prepared target" record the TTP points at by id.** Cleaner
normalisation and natural sharing between profiles. It is a second store to
build, back up, garbage-collect and migrate; the embedded form reuses the TTP
store that already exists. Worth revisiting if a third thing ever needs to
reference a preparation.

**A general pipeline/DAG field in the TTP.** Tempting, because it would
subsume this case and others. It turns the TTP from a declaration of *what is
wanted* into a program, which is the thing the TTP exists not to be — and the
conductor/supervisor split already provides execution. Rejected.

## 5. Where to start

Smallest first step with standalone value: make `pdb_id` meaningful rather
than decorative. Validate that it agrees with `protein_pdb_path` when both are
given, and record it in the run's provenance. That is a contained change, it
makes existing profiles more honest, and it is a prerequisite for anything in
section 3.
