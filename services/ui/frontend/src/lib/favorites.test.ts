// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { beforeEach, describe, expect, it } from "vitest";

import type { StructurePanel } from "../types/api";
import { basketGroups, clearKept, isKept, keptOf, toggleKept, type KeptStructure } from "./favorites";
import type { ShortlistEntry } from "./shortlist";

const panel = (over: Partial<StructurePanel> = {}): StructurePanel => ({
  kind: "structure",
  tool: "boltz_cofold",
  name: "complex.cif",
  path: "predictions/run-1/complex.cif",
  url: "/files/complex.cif",
  format: "cif",
  ...over,
});

const kept = (over: Partial<KeptStructure>): KeptStructure => ({ path: "a.cif", at: 1, ...over });
const starred = (over: Partial<ShortlistEntry>): ShortlistEntry => ({ smiles: "CCO", at: 1, ...over });

describe("keptOf", () => {
  it("remembers how a structure was made, so the basket can group by it", () => {
    expect(keptOf(panel()).category).toBe("predicted");
    expect(keptOf(panel({ path: "rcsb/3mxf.cif", tool: "rcsb" })).category).toBe("experimental");
    expect(keptOf(panel({ path: "prep/egfr/receptor.pdb", format: "pdb" })).category).toBe("prepared");
  });

  it("leaves the category off a path that does not say", () => {
    // Guessing wrong would file it under a heading it does not belong to.
    expect(keptOf(panel({ path: "uploads/thing.cif", tool: "upload" })).category).toBeUndefined();
  });
});

describe("the kept structures", () => {
  beforeEach(() => clearKept("s1"));

  it("keeps a structure and drops it again", () => {
    expect(toggleKept("s1", keptOf(panel()))).toBe(true);
    expect(isKept("s1", "predictions/run-1/complex.cif")).toBe(true);
    expect(toggleKept("s1", keptOf(panel()))).toBe(false);
    expect(isKept("s1", "predictions/run-1/complex.cif")).toBe(false);
  });

  it("is per conversation", () => {
    toggleKept("s1", keptOf(panel()));
    expect(isKept("s2", "predictions/run-1/complex.cif")).toBe(false);
  });
});

describe("basketGroups", () => {
  it("puts the structures under how they were made, in the order the Structures tab browses", () => {
    const groups = basketGroups(
      [kept({ path: "a.cif", category: "predicted" }), kept({ path: "b.cif", category: "experimental" }), kept({ path: "c.pdb", category: "prepared" })],
      [],
    );
    expect(groups.map((g) => g.label)).toEqual(["RCSB PDB", "Prepared", "Predicted"]);
  });

  it("puts the molecules under the run they were starred in, after the structures", () => {
    const groups = basketGroups([kept({ path: "a.cif", category: "predicted" })], [starred({ smiles: "CCO", source: "job-7" }), starred({ smiles: "CCN" })]);
    expect(groups.map((g) => g.label)).toEqual(["Predicted", "Molecules", "Molecules from job-7"]);
  });

  it("shows the newest of a group first", () => {
    const groups = basketGroups([kept({ path: "old.cif", category: "predicted", at: 1 }), kept({ path: "new.cif", category: "predicted", at: 2 })], []);
    expect(groups[0].items.map((i) => i.key)).toEqual(["new.cif", "old.cif"]);
  });

  it("files a structure kept before the category was recorded rather than dropping it", () => {
    expect(basketGroups([kept({ path: "a.cif" })], [])).toHaveLength(1);
  });
});
