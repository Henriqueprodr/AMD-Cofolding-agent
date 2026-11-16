// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import type { AtlasBranch, Finding, FleetEvent } from "../types/api";
import { branchViews, evidenceRun, evidenceSource, learnt, stances } from "./atlas";

const event = (kind: string): FleetEvent => ({
  event_id: 1,
  task_id: null,
  kind,
  payload: {},
  created_at: "",
});
const branch = (name: string, findings: number, disputed = 0): AtlasBranch => ({
  branch: name,
  name,
  in_plan: true,
  findings,
  disputed,
  latest: null,
  tags: [],
});
const finding = (
  id: string,
  on: string,
  notes: Finding["notes"] = [],
): Finding => ({
  finding_id: id,
  campaign_id: "c1",
  branch: on,
  claim: id,
  detail: null,
  evidence: [],
  confidence: "medium",
  tags: [],
  author: "sup-1",
  task_id: null,
  created_at: "",
  supersedes: null,
  superseded_by: null,
  notes,
});
const note = (stance: Finding["notes"][number]["stance"]) => ({
  note_id: 1,
  finding_id: "f",
  body: "",
  stance,
  author: "sup-2",
  task_id: null,
  created_at: "",
});

describe("learnt", () => {
  it("moves with every finding, note and report, and not with anything else", () => {
    const quiet = { results: [], tasks: [{ finished_at: null }] };
    expect(
      learnt([event("task_claimed"), event("task_started_job")], quiet),
    ).toBe(0);
    expect(
      learnt([event("finding_recorded"), event("finding_noted")], quiet),
    ).toBe(2);
  });

  it("still moves once the history the panel holds has stopped growing", () => {
    // The panel reads the first page of the history; a busy campaign outgrows it, but its tasks keep reporting.
    const page = [event("finding_recorded")];
    expect(
      learnt(page, {
        results: [{}],
        tasks: [{ finished_at: "t" }, { finished_at: null }],
      }),
    ).toBe(3);
  });
});

describe("branchViews", () => {
  it("puts what needs settling first and what nobody knows last, else the plan's order", () => {
    const views = branchViews({
      branches: [
        branch("empty", 0),
        branch("potency", 2),
        branch("admet", 1, 1),
        branch("pk", 1),
      ],
      items: [
        finding("f1", "potency"),
        finding("f2", "admet"),
        finding("f3", "potency"),
        finding("f4", "pk"),
      ],
    });
    expect(views.map((view) => view.branch)).toEqual([
      "admet",
      "potency",
      "pk",
      "empty",
    ]);
    expect(views[1].items.map((item) => item.finding_id)).toEqual(["f1", "f3"]);
    expect(views[3].items).toEqual([]);
  });
});

describe("stances", () => {
  it("counts how a finding's notes lean, leaving out notes without a stance", () => {
    expect(
      stances({
        notes: [
          note("disputes"),
          note("supports"),
          note("disputes"),
          note(null),
        ],
      }),
    ).toEqual({ supports: 1, disputes: 2, extends: 0 });
  });
});

describe("evidenceRun", () => {
  const runs = new Set(["a7ad1527"]);

  it("opens a run the gateway has, however the agent wrote its id", () => {
    expect(evidenceRun("a7ad1527", runs)).toBe("a7ad1527");
    expect(evidenceRun("job_a7ad1527", runs)).toBe("a7ad1527");
    expect(evidenceRun(" Job A7AD1527 ", runs)).toBe("a7ad1527");
  });

  it("leaves anything else as text, hex digits included", () => {
    // A run of hex digits could be anything; only a run the gateway has is a run.
    expect(evidenceRun("deadbeef", runs)).toBeNull();
    expect(evidenceRun("CHEMBL1957266", runs)).toBeNull();
    expect(evidenceRun("10.1021/jm901137j", runs)).toBeNull();
  });
});

describe("evidenceSource", () => {
  const url = (item: string) => evidenceSource(item)?.url ?? null;

  it("sends a public id to the record it names", () => {
    expect(url("1IQK")).toBe("https://www.rcsb.org/structure/1IQK");
    expect(url("CHEMBL1957266")).toBe("https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL1957266");
    expect(url("O60885")).toBe("https://www.uniprot.org/uniprotkb/O60885");
    expect(url("10.1021/jm901137j")).toBe("https://doi.org/10.1021/jm901137j");
    expect(url("PMID:20871596")).toBe("https://pubmed.ncbi.nlm.nih.gov/20871596/");
  });

  it("reads the id however the agent prefixed it", () => {
    expect(url("pdb 1IQK")).toBe("https://www.rcsb.org/structure/1IQK");
    expect(url("uniprot:O60885")).toBe("https://www.uniprot.org/uniprotkb/O60885");
    expect(url("doi:10.1021/jm901137j")).toBe("https://doi.org/10.1021/jm901137j");
    expect(url("chembl1957266")).toBe("https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL1957266");
  });

  it("passes an address the agent already wrote out straight through", () => {
    expect(url("https://www.rcsb.org/structure/3MXF")).toBe("https://www.rcsb.org/structure/3MXF");
  });

  it("leaves anything it cannot place as plain text", () => {
    // Sending these somewhere would invent a citation rather than follow one.
    expect(evidenceSource("the 2024 review")).toBeNull();
    expect(evidenceSource("2024")).toBeNull();
    expect(evidenceSource("deadbeef")).toBeNull();
    expect(evidenceSource("")).toBeNull();
  });

  it("names the kind of record, for the tooltip", () => {
    expect(evidenceSource("1IQK")?.kind).toBe("PDB entry");
    expect(evidenceSource("CHEMBL25")?.kind).toBe("ChEMBL record");
  });
});
