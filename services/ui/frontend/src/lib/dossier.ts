// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * What a conversation learned about a target beyond its UniProt entry.
 *
 * A campaign asks about a target in pieces: the genetic evidence tying it to
 * a disease, where it is expressed, the pathways it sits in, the species it
 * has an ortholog in, who has patented around it, what its pocket looks like
 * and which constructs were crystallised with which mutations. Each of those
 * answers arrives as its own tool result, is read once in the chat and is
 * then scrolled away. Here they are kept per target, so the Target tab is a
 * dossier that fills in as the work goes rather than a UniProt card.
 *
 * Facts are keyed by whichever identity the call used -- the accession, the
 * Ensembl gene, the gene symbol or the structure path -- and `dossierFor`
 * gathers every key one target answers to.
 */

import type { Candidate } from "./candidates";
import { parseResult, type PocketResidue, type TargetFacts, type ToolEvent } from "./ttp";

export interface DiseaseLink {
  disease: string;
  /** OpenTargets' overall association score, 0-1. */
  score: number | null;
  /** The part of it carried by human genetics, where OpenTargets reports one. */
  geneticScore: number | null;
}

export interface ExpressionRow {
  tissue: string;
  cellType: string | null;
  /** "High", "Medium", "Low" or "Not detected". */
  level: string;
  reliability: string | null;
}

export interface Pathway {
  /** "Reactome", "KEGG" or "GO". */
  source: string;
  name: string;
}

export interface Ortholog {
  accession: string | null;
  organism: string;
  gene: string | null;
  identityPct: number | null;
}

export interface PatentHit {
  id: string;
  title: string | null;
}

/** One named sub-region of a pocket, as characterize_site_profile reports it. */
export interface Subpocket {
  name: string;
  residues: string[];
}

/** What characterize_site_profile said about one structure's site. */
export interface SiteProfile {
  /** The structure it was run on, as the call named it. */
  structure: string | null;
  /** orthosteric | allosteric | ppi | unknown. */
  siteType: string | null;
  volume: number | null;
  druggability: number | null;
  /** polar | hydrophobic | mixed. */
  character: string | null;
  subpockets: Subpocket[];
  /** Residues the profiler classified: catalytic, gatekeeper, anchor. */
  roles: { residue: string; role: string }[];
  activeSite: string[];
}

/** A crystallised construct that is not the wild-type protein. */
export interface Mutant {
  pdbId: string;
  /** As the depositor wrote it: "T790M, L858R". */
  mutations: string;
  /** What it was expressed in, where the entry says. */
  expressedIn: string | null;
}

export interface TargetDossier {
  diseases: DiseaseLink[];
  expression: ExpressionRow[];
  pathways: Pathway[];
  orthologs: Ortholog[];
  patents: PatentHit[];
  /** Patents the search found, which can be more than it returned. */
  patentTotal: number | null;
  sites: SiteProfile[];
  mutants: Mutant[];
  /** Tools the entries came from, in the order first seen. */
  sources: string[];
}

export const emptyDossier = (): TargetDossier => ({
  diseases: [],
  expression: [],
  pathways: [],
  orthologs: [],
  patents: [],
  patentTotal: null,
  sites: [],
  mutants: [],
  sources: [],
});

export const hasDossier = (d: TargetDossier) =>
  d.diseases.length > 0 ||
  d.expression.length > 0 ||
  d.pathways.length > 0 ||
  d.orthologs.length > 0 ||
  d.patents.length > 0 ||
  d.sites.length > 0 ||
  d.mutants.length > 0;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);
const str = (value: unknown): string | null => (typeof value === "string" && value.trim() ? value.trim() : null);
const num = (value: unknown): number | null => (typeof value === "number" && Number.isFinite(value) ? value : null);
const rows = (value: unknown): Record<string, unknown>[] => (Array.isArray(value) ? value.filter(isRecord) : []);
const list = (value: unknown): string[] => (Array.isArray(value) ? value.flatMap((v) => (str(v) ? [str(v) as string] : [])) : []);

/** An identity a dossier can be filed under; null for anything unusable as one. */
const identity = (value: unknown): string | null => str(value)?.toUpperCase() ?? null;

/**
 * Every identity a target answers to: its accession, its gene, its Ensembl
 * gene and the complexes its pocket was profiled on. A dossier filed under
 * any of them is this target's.
 */
export function targetKeys(facts: TargetFacts): string[] {
  const keys = [facts.uniprot, facts.gene, facts.name, facts.proteinPdbPath, ...facts.complexes, ...facts.pdbIds];
  return [...new Set(keys.flatMap((k) => (identity(k) ? [identity(k) as string] : [])))];
}

type Dossiers = Map<string, TargetDossier>;

function under(dossiers: Dossiers, key: string | null, tool: string): TargetDossier | null {
  const id = identity(key);
  if (!id) return null;
  let dossier = dossiers.get(id);
  if (!dossier) {
    dossier = emptyDossier();
    dossiers.set(id, dossier);
  }
  if (!dossier.sources.includes(tool)) dossier.sources.push(tool);
  return dossier;
}

/**
 * The dossier facts of every target the conversation asked about, filed
 * under each identity the asking used.
 *
 * A call and its result are read as a pair: the result of a genetic-evidence
 * or expression lookup does not say which target it is about in a form that
 * matches a UniProt card, but the call that asked does.
 */
export function extractDossiers(events: ToolEvent[]): Dossiers {
  const dossiers: Dossiers = new Map();
  /** The arguments of the last call of each tool, for the result that follows. */
  const pending = new Map<string, Record<string, unknown>>();

  for (const event of events) {
    if (event.type === "tool_call") {
      pending.set(event.name, event.arguments);
      continue;
    }
    if (event.isError) continue;
    const args = pending.get(event.name) ?? {};
    const payload = parseResult(event.result);
    if (!isRecord(payload) || typeof payload.error === "string") continue;

    switch (event.name) {
      case "fetch_target_genetic_evidence":
        addEvidence(dossiers, event.name, args, payload);
        break;
      case "fetch_target_expression":
        addExpression(dossiers, event.name, args, payload);
        break;
      case "fetch_pathway_annotations":
        addPathways(dossiers, event.name, args, payload);
        break;
      case "fetch_target_orthology":
        addOrthologs(dossiers, event.name, args, payload);
        break;
      case "search_patents_by_target":
        addPatents(dossiers, event.name, args, payload);
        break;
      case "characterize_site_profile":
        addSite(dossiers, event.name, args, payload);
        break;
      case "rcsb_get_entries":
        addEntries(dossiers, event.name, payload);
        break;
      default:
        break;
    }
  }
  return dossiers;
}

function addEvidence(dossiers: Dossiers, tool: string, args: Record<string, unknown>, payload: Record<string, unknown>) {
  const diseases = rows(payload.top_diseases).flatMap((row) => {
    const disease = str(row.disease) ?? str(row.name) ?? str(row.disease_name);
    return disease ? [{ disease, score: num(row.score), geneticScore: num(row.genetic_score) }] : [];
  });
  if (diseases.length === 0) return;
  // Filed under the Ensembl gene the call used and the symbol the answer
  // carries: a UniProt card has the second and not the first.
  for (const key of [args.ensembl_id, payload.ensembl_id, payload.target_symbol]) {
    const dossier = under(dossiers, key as string, tool);
    if (dossier) dossier.diseases = merge(dossier.diseases, diseases, (d) => d.disease);
  }
}

function addExpression(dossiers: Dossiers, tool: string, args: Record<string, unknown>, payload: Record<string, unknown>) {
  const expression = rows(payload.tissue_rows).flatMap((row) => {
    const tissue = str(row.tissue);
    const level = str(row.level);
    return tissue && level ? [{ tissue, cellType: str(row.cell_type), level, reliability: str(row.reliability) }] : [];
  });
  if (expression.length === 0) return;
  const dossier = under(dossiers, (args.uniprot_accession ?? payload.uniprot_accession) as string, tool);
  if (dossier) dossier.expression = merge(dossier.expression, expression, (e) => `${e.tissue}/${e.cellType ?? ""}`);
}

function addPathways(dossiers: Dossiers, tool: string, args: Record<string, unknown>, payload: Record<string, unknown>) {
  const named = (value: unknown, source: string): Pathway[] =>
    [...list(value), ...rows(value).flatMap((row) => (str(row.name) ? [str(row.name) as string] : []))].map((name) => ({ source, name }));
  const pathways = [
    ...named(payload.reactome_pathways, "Reactome"),
    ...named(payload.kegg_pathways, "KEGG"),
    ...named(payload.go_biological_process, "GO"),
  ];
  if (pathways.length === 0) return;
  const dossier = under(dossiers, (args.uniprot_accession ?? payload.uniprot_accession) as string, tool);
  if (dossier) dossier.pathways = merge(dossier.pathways, pathways, (p) => `${p.source}:${p.name}`);
}

function addOrthologs(dossiers: Dossiers, tool: string, args: Record<string, unknown>, payload: Record<string, unknown>) {
  const orthologs = rows(payload.orthologs).flatMap((row) => {
    const organism = str(row.organism);
    return organism ? [{ accession: str(row.accession), organism, gene: str(row.gene), identityPct: num(row.identity_pct) }] : [];
  });
  if (orthologs.length === 0) return;
  const dossier = under(dossiers, (args.uniprot_accession ?? payload.uniprot_accession) as string, tool);
  if (dossier) dossier.orthologs = merge(dossier.orthologs, orthologs, (o) => o.organism);
}

function addPatents(dossiers: Dossiers, tool: string, args: Record<string, unknown>, payload: Record<string, unknown>) {
  const patents = rows(payload.patents).flatMap((row) => {
    const id = str(row.id) ?? str(row.publication_number) ?? str(row.patent_id);
    return id ? [{ id, title: str(row.title) }] : [];
  });
  if (patents.length === 0) return;
  // The query is free text: "BRD4 bromodomain" files under BRD4 as well, so
  // a card that knows only the gene still finds it.
  const query = str(args.query);
  for (const key of [query, ...(query ? query.split(/\s+/) : [])]) {
    const dossier = under(dossiers, key, tool);
    if (!dossier) continue;
    dossier.patents = merge(dossier.patents, patents, (p) => p.id);
    dossier.patentTotal = num(payload.total) ?? dossier.patentTotal;
  }
}

function addSite(dossiers: Dossiers, tool: string, args: Record<string, unknown>, payload: Record<string, unknown>) {
  const profile = isRecord(payload.site_profile) ? payload.site_profile : payload;
  const structure = str(args.structure);
  const site: SiteProfile = {
    structure,
    siteType: str(profile.site_type),
    volume: num(profile.pocket_volume_A3),
    druggability: num(profile.pocket_druggability),
    character: str(profile.pocket_character),
    subpockets: rows(profile.subpockets).flatMap((row) => {
      const name = str(row.name) ?? str(row.id) ?? str(row.subpocket_id);
      return name ? [{ name, residues: list(row.residues) }] : [];
    }),
    roles: rows(profile.residue_roles).flatMap((row) => {
      const residue = str(row.residue) ?? str(row.label);
      const role = str(row.role);
      return residue && role ? [{ residue, role }] : [];
    }),
    activeSite: list(profile.active_site_residues),
  };
  if (!site.siteType && site.volume === null && site.subpockets.length === 0 && site.roles.length === 0) return;
  // The accession where the call gave one, and the structure either way:
  // a profile of an apo receptor names no protein, but the target whose
  // pocket was profiled on that file is the one it belongs to.
  for (const key of [args.uniprot_accession as string, structure, structure?.split("/").pop() ?? null]) {
    const dossier = under(dossiers, key, tool);
    if (dossier) dossier.sites = merge(dossier.sites, [site], (s) => s.structure ?? "");
  }
}

function addEntries(dossiers: Dossiers, tool: string, payload: Record<string, unknown>) {
  for (const entryRow of rows(payload.entries)) {
    const pdbId = str(entryRow.entry_id) ?? str(entryRow.id) ?? str(entryRow.pdb_id);
    for (const entity of rows(entryRow.polymer_entities)) {
      const mutations = str(entity.mutations);
      if (!mutations || !pdbId) continue;
      const mutant: Mutant = { pdbId: pdbId.toUpperCase(), mutations, expressedIn: str(entity.expression_system) };
      for (const key of [pdbId, ...list(entity.uniprot_ids), ...list(entity.gene_names)]) {
        const dossier = under(dossiers, key, tool);
        if (dossier) dossier.mutants = merge(dossier.mutants, [mutant], (m) => `${m.pdbId}:${m.mutations}`);
      }
    }
  }
}

/** *have* plus the entries of *add* it does not already hold, newest last. */
function merge<T>(have: T[], add: T[], key: (item: T) => string): T[] {
  const seen = new Set(have.map(key));
  return [...have, ...add.filter((item) => !seen.has(key(item)))];
}

/** Everything known about *facts*, gathered from every identity it answers to. */
export function dossierFor(dossiers: Dossiers, facts: TargetFacts): TargetDossier {
  const out = emptyDossier();
  for (const key of targetKeys(facts)) {
    const found = dossiers.get(key);
    if (!found) continue;
    out.diseases = merge(out.diseases, found.diseases, (d) => d.disease);
    out.expression = merge(out.expression, found.expression, (e) => `${e.tissue}/${e.cellType ?? ""}`);
    out.pathways = merge(out.pathways, found.pathways, (p) => `${p.source}:${p.name}`);
    out.orthologs = merge(out.orthologs, found.orthologs, (o) => o.organism);
    out.patents = merge(out.patents, found.patents, (p) => p.id);
    out.patentTotal = found.patentTotal ?? out.patentTotal;
    out.sites = merge(out.sites, found.sites, (s) => s.structure ?? "");
    out.mutants = merge(out.mutants, found.mutants, (m) => `${m.pdbId}:${m.mutations}`);
    out.sources = merge(out.sources, found.sources, (s) => s);
  }
  return out;
}

/** Residue labels any site profile classified, by role. */
export function residueRoles(dossier: TargetDossier): { residue: string; role: string }[] {
  return dossier.sites.flatMap((site) => site.roles);
}

/**
 * Residues a covalent warhead could be aimed at: the cysteines, and the
 * lysines, serines and threonines that have been used for reversible
 * covalent chemistry, among those the pocket or a site profile named.
 *
 * A handle is only worth reporting where the ligands actually reach it, so
 * the pocket contacts come first and carry how often they were touched.
 */
const COVALENT_RESIDUES: Record<string, string> = { CYS: "cysteine", LYS: "lysine", SER: "serine", THR: "threonine", TYR: "tyrosine" };

export interface CovalentHandle {
  label: string;
  residue: string;
  /** Contacts across the profiled complexes; 0 for one only a site profile named. */
  contacts: number;
}

export function covalentHandles(facts: TargetFacts, dossier: TargetDossier): CovalentHandle[] {
  const handles = new Map<string, CovalentHandle>();
  for (const residue of facts.pocket) {
    const kind = COVALENT_RESIDUES[residue.resname.toUpperCase()];
    if (kind) handles.set(residue.label, { label: residue.label, residue: kind, contacts: residue.contacts });
  }
  for (const site of dossier.sites) {
    for (const label of [...site.activeSite, ...site.subpockets.flatMap((p) => p.residues), ...site.roles.map((r) => r.residue)]) {
      const kind = COVALENT_RESIDUES[label.slice(0, 3).toUpperCase()];
      if (kind && !handles.has(label)) handles.set(label, { label, residue: kind, contacts: 0 });
    }
  }
  return [...handles.values()].sort((a, b) => b.contacts - a.contacts || a.label.localeCompare(b.label));
}

/**
 * The best molecule this target has: the highest-scoring candidate folded
 * against one of the complexes its pocket was profiled on.
 *
 * Where the conversation only ever worked on one target, every candidate is
 * that target's, so the best one is reported without a complex to tie it.
 */
export function bestForTarget(facts: TargetFacts, candidates: readonly Candidate[], onlyTarget: boolean): Candidate | null {
  const complexes = new Set(facts.complexes);
  const mine = candidates.filter((c) => (c.structurePath && complexes.has(c.structurePath)) || (onlyTarget && c.score !== undefined));
  let best: Candidate | null = null;
  for (const candidate of mine) {
    if (candidate.score === undefined) continue;
    if (!best || best.score === undefined || candidate.score > best.score) best = candidate;
  }
  return best;
}

/** The pocket residues a label names, for focusing the viewer on a subpocket. */
export function residuesNamed(facts: TargetFacts, labels: readonly string[]): PocketResidue[] {
  const wanted = new Set(labels.map((l) => l.toUpperCase()));
  return facts.pocket.filter((residue) => wanted.has(residue.label.toUpperCase()));
}
