// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type {
  AtlasBranch,
  CampaignAtlas,
  CampaignDetail,
  Finding,
  FleetEvent,
} from "../types/api";

/** Ledger events that change what the campaign knows. */
const LEARNING = new Set(["finding_recorded", "finding_noted"]);

/**
 * How much the campaign has learnt so far: the atlas is read again when this
 * moves. The campaign and its history are polled already, so the atlas costs
 * a read when something was recorded and none when nothing was.
 *
 * The findings and notes in the history, and the tasks that have reported:
 * the history the panel holds is its first page, which a busy campaign
 * outgrows, while a sub-agent records what it settled before it reports.
 */
export function learnt(
  events: readonly FleetEvent[] | undefined,
  detail: {
    results: readonly unknown[];
    tasks: readonly Pick<CampaignDetail["tasks"][number], "finished_at">[];
  },
): number {
  let count = detail.results.length;
  for (const task of detail.tasks) if (task.finished_at) count += 1;
  for (const event of events ?? []) if (LEARNING.has(event.kind)) count += 1;
  return count;
}

export interface BranchView extends AtlasBranch {
  /** Its current findings, newest first. */
  items: Finding[];
}

/**
 * The branches as the view lists them: those with something disputed first,
 * since they are what somebody has to settle, then the rest that hold
 * findings, newest news first, and last the ones nothing is known about.
 * Within a group the plan's order is kept, so a branch does not jump about
 * between two reads.
 */
export function branchViews(
  atlas: Pick<CampaignAtlas, "branches" | "items">,
): BranchView[] {
  const byBranch = new Map<string, Finding[]>();
  for (const finding of atlas.items)
    byBranch.set(finding.branch, [
      ...(byBranch.get(finding.branch) ?? []),
      finding,
    ]);
  const views = atlas.branches.map((branch) => ({
    ...branch,
    items: byBranch.get(branch.branch) ?? [],
  }));
  const rank = (view: BranchView) =>
    view.disputed > 0 ? 0 : view.findings > 0 ? 1 : 2;
  return views
    .map((view, i) => ({ view, i }))
    .sort((a, b) => rank(a.view) - rank(b.view) || a.i - b.i)
    .map(({ view }) => view);
}

/** How a finding's notes lean: supports, disputes and extends counted. */
export function stances(finding: Pick<Finding, "notes">): {
  supports: number;
  disputes: number;
  extends: number;
} {
  const counts = { supports: 0, disputes: 0, extends: 0 };
  for (const note of finding.notes) if (note.stance) counts[note.stance] += 1;
  return counts;
}

/** A job id as the gateway serves jobs, bare or written "job_…" / "job …". */
const JOB = /^(?:job[_\s:]*)?([0-9a-f]{6,32})$/i;

/** The run a piece of evidence names, so it can open; null for a DOI, a ChEMBL id, a PDB code. */
export function evidenceRun(
  item: string,
  known: ReadonlySet<string>,
): string | null {
  const id = JOB.exec(item.trim())?.[1]?.toLowerCase();
  // A run of hex digits could be anything; only a run the gateway has is a run.
  return id && known.has(id) ? id : null;
}

/** Where a piece of evidence can be read, outside this app. */
export interface EvidenceSource {
  /** What kind of record it is, for the tooltip. */
  kind: string;
  url: string;
}

/**
 * Evidence that names a public record, resolved to where it can be read.
 *
 * A finding's evidence is its whole justification, and an id the reader
 * cannot follow is an assertion rather than a citation: "CHEMBL3137841"
 * meant retyping it into a search box. Only unambiguous, well-formed ids
 * are linked -- a bare number or a word is left as text rather than sent
 * somewhere it probably does not exist.
 */
const SOURCES: { kind: string; pattern: RegExp; url: (id: string) => string }[] = [
  {
    kind: "PDB entry",
    // A digit then three alphanumerics, of which not all three may be
    // digits: without that a year such as "2024" reads as an entry and the
    // reader is sent to a structure that does not exist.
    pattern: /^(?:pdb[:\s]*)?([1-9](?![0-9]{3}$)[a-z0-9]{3})$/i,
    url: (id) => `https://www.rcsb.org/structure/${id.toUpperCase()}`,
  },
  {
    kind: "ChEMBL record",
    pattern: /^(chembl\d+)$/i,
    url: (id) => `https://www.ebi.ac.uk/chembl/explore/compound/${id.toUpperCase()}`,
  },
  {
    kind: "UniProt entry",
    pattern: /^(?:uniprot[:\s]*)?([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$/i,
    url: (id) => `https://www.uniprot.org/uniprotkb/${id.toUpperCase()}`,
  },
  {
    kind: "DOI",
    pattern: /^(?:doi[:\s]*)?(10\.\d{4,9}\/\S+)$/i,
    url: (id) => `https://doi.org/${id}`,
  },
  {
    kind: "PubMed article",
    pattern: /^(?:pmid[:\s]*)(\d{1,8})$/i,
    url: (id) => `https://pubmed.ncbi.nlm.nih.gov/${id}/`,
  },
  {
    kind: "PubChem compound",
    pattern: /^(?:cid[:\s]*)(\d{1,9})$/i,
    url: (id) => `https://pubchem.ncbi.nlm.nih.gov/compound/${id}`,
  },
  {
    kind: "gene",
    pattern: /^(?:hgnc[:\s]*)([A-Z][A-Z0-9-]{1,9})$/,
    url: (id) => `https://www.genenames.org/tools/search/#!/?query=${encodeURIComponent(id)}`,
  },
];

export function evidenceSource(item: string): EvidenceSource | null {
  const text = item.trim();
  // An address the agent wrote out is already the citation.
  if (/^https?:\/\/\S+$/i.test(text)) return { kind: "source", url: text };
  for (const source of SOURCES) {
    const id = source.pattern.exec(text)?.[1];
    // A PDB id and a UniProt accession can both be four or six characters;
    // the first pattern that matches in this order wins, which puts the
    // stricter shapes first.
    if (id) return { kind: source.kind, url: source.url(id) };
  }
  return null;
}
