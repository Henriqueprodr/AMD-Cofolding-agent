// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";

import type { Candidate } from "./candidates";
import { bestForTarget, covalentHandles, dossierFor, extractDossiers, hasDossier } from "./dossier";
import { emptyTargetFacts, type TargetFacts, type ToolEvent } from "./ttp";

const call = (name: string, args: Record<string, unknown>): ToolEvent => ({ type: "tool_call", name, arguments: args });
const result = (name: string, payload: unknown): ToolEvent => ({ type: "tool_result", name, result: payload });

const target = (over: Partial<TargetFacts> = {}): TargetFacts => ({ ...emptyTargetFacts(), ...over });

const candidate = (over: Partial<Candidate>): Candidate => ({
  smiles: "CCO",
  origin: "run_optimization",
  origins: ["run_optimization"],
  components: {},
  gate: { state: "unknown", reasons: [] } as unknown as Candidate["gate"],
  ...over,
});

describe("extractDossiers", () => {
  it("files genetic evidence under the Ensembl gene asked for and the symbol answered with", () => {
    const events = [
      call("fetch_target_genetic_evidence", { ensembl_id: "ENSG00000146648" }),
      result("fetch_target_genetic_evidence", {
        target_symbol: "EGFR",
        top_diseases: [{ disease: "lung carcinoma", score: 0.82, genetic_score: 0.41 }],
      }),
    ];
    const dossiers = extractDossiers(events);
    // A UniProt card knows the gene symbol, not the Ensembl id the call used.
    expect(dossiers.get("ENSG00000146648")?.diseases[0].disease).toBe("lung carcinoma");
    expect(dossierFor(dossiers, target({ gene: "EGFR" })).diseases).toHaveLength(1);
  });

  it("reaches the target of an expression or pathway lookup through the accession the call used", () => {
    const events = [
      call("fetch_target_expression", { uniprot_accession: "P00533" }),
      result("fetch_target_expression", { tissue_rows: [{ tissue: "lung", cell_type: "pneumocytes", level: "High", reliability: "Enhanced" }] }),
      call("fetch_pathway_annotations", { uniprot_accession: "P00533" }),
      result("fetch_pathway_annotations", { reactome_pathways: ["Signaling by EGFR"], kegg_pathways: [{ name: "ErbB signaling" }] }),
    ];
    const dossier = dossierFor(extractDossiers(events), target({ uniprot: "P00533" }));
    expect(dossier.expression).toEqual([{ tissue: "lung", cellType: "pneumocytes", level: "High", reliability: "Enhanced" }]);
    expect(dossier.pathways).toEqual([
      { source: "Reactome", name: "Signaling by EGFR" },
      { source: "KEGG", name: "ErbB signaling" },
    ]);
  });

  it("ties a site profile of an apo receptor to the target whose pocket was profiled on it", () => {
    // The call names no protein, only a file. The complex the contacts came
    // from is the link, and it is what the Target tab already holds.
    const events = [
      call("characterize_site_profile", { structure: "/shared/prep/egfr/receptor.pdb" }),
      result("characterize_site_profile", {
        site_profile: {
          site_type: "orthosteric",
          pocket_volume_A3: 568.2,
          pocket_druggability: 0.845,
          pocket_character: "hydrophobic",
          subpockets: [{ name: "back pocket", residues: ["THR790", "MET793"] }],
          residue_roles: [{ residue: "THR790", role: "gatekeeper" }],
        },
      }),
    ];
    const dossier = dossierFor(extractDossiers(events), target({ complexes: ["/shared/prep/egfr/receptor.pdb"] }));
    expect(dossier.sites[0].siteType).toBe("orthosteric");
    expect(dossier.sites[0].subpockets[0].residues).toEqual(["THR790", "MET793"]);
  });

  it("keeps the mutants a deposited entry was crystallised with", () => {
    const events = [
      call("rcsb_get_entries", { entry_ids: ["2ITX"] }),
      result("rcsb_get_entries", {
        entries: [
          {
            entry_id: "2itx",
            polymer_entities: [{ mutations: "T790M", expression_system: "Spodoptera frugiperda", uniprot_ids: ["P00533"], gene_names: ["EGFR"] }],
          },
        ],
      }),
    ];
    const dossier = dossierFor(extractDossiers(events), target({ uniprot: "P00533" }));
    expect(dossier.mutants).toEqual([{ pdbId: "2ITX", mutations: "T790M", expressedIn: "Spodoptera frugiperda" }]);
  });

  it("says nothing about a lookup that failed", () => {
    const events = [
      call("fetch_target_expression", { uniprot_accession: "P00533" }),
      result("fetch_target_expression", { error: "Human Protein Atlas could not be reached" }),
    ];
    expect(hasDossier(dossierFor(extractDossiers(events), target({ uniprot: "P00533" })))).toBe(false);
  });

  it("gathers one target's facts from every identity they were filed under", () => {
    const events = [
      call("fetch_target_genetic_evidence", { ensembl_id: "ENSG00000146648" }),
      result("fetch_target_genetic_evidence", { target_symbol: "EGFR", top_diseases: [{ disease: "lung carcinoma", score: 0.8 }] }),
      call("search_patents_by_target", { query: "EGFR inhibitor" }),
      result("search_patents_by_target", { patents: [{ id: "WO2005012914", title: "Quinazolines" }], total: 412 }),
      call("fetch_target_expression", { uniprot_accession: "P00533" }),
      result("fetch_target_expression", { tissue_rows: [{ tissue: "lung", level: "High" }] }),
    ];
    const dossier = dossierFor(extractDossiers(events), target({ uniprot: "P00533", gene: "EGFR" }));
    expect(dossier.diseases).toHaveLength(1);
    expect(dossier.patents).toHaveLength(1);
    expect(dossier.patentTotal).toBe(412);
    expect(dossier.expression).toHaveLength(1);
  });
});

describe("covalentHandles", () => {
  it("ranks the nucleophilic residues the ligands reach above those only named", () => {
    const facts = target({
      pocket: [
        { chain: "A", resname: "CYS", resnum: 797, label: "CYS797", contacts: 4, types: ["hydrophobic"] },
        { chain: "A", resname: "LEU", resnum: 718, label: "LEU718", contacts: 9, types: ["hydrophobic"] },
      ],
    });
    const events = [
      call("characterize_site_profile", { structure: "r.pdb", uniprot_accession: "P00533" }),
      result("characterize_site_profile", { site_profile: { site_type: "orthosteric", active_site_residues: ["LYS745"] } }),
    ];
    const handles = covalentHandles(facts, dossierFor(extractDossiers(events), target({ uniprot: "P00533" })));
    expect(handles.map((h) => h.label)).toEqual(["CYS797", "LYS745"]);
    expect(handles[0].residue).toBe("cysteine");
  });
});

describe("bestForTarget", () => {
  it("takes the best molecule folded against one of this target's complexes", () => {
    const facts = target({ complexes: ["a.cif"] });
    const best = bestForTarget(
      facts,
      [candidate({ smiles: "C1", score: 0.4, structurePath: "a.cif" }), candidate({ smiles: "C2", score: 0.9, structurePath: "b.cif" })],
      false,
    );
    expect(best?.smiles).toBe("C1");
  });

  it("takes the best of all where the conversation only worked on one target", () => {
    // Nothing was profiled, so no molecule names a complex; there is still
    // only one target they could be for.
    const best = bestForTarget(target({ gene: "EGFR" }), [candidate({ smiles: "C1", score: 0.4 }), candidate({ smiles: "C2", score: 0.9 })], true);
    expect(best?.smiles).toBe("C2");
  });
});
