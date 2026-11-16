// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import type { StructurePanel } from "../types/api";
import { BASKET_COLOURS, basketColour, basketEntries, browseGroups, toggleShown } from "./basket";

const panel = (path: string): StructurePanel => ({ kind: "structure", tool: "boltz", name: path.split("/").pop() ?? path, path, url: `/api/artifacts/${path}`, format: "cif" });
const title = (p: StructurePanel) => p.name;

describe("basketEntries", () => {
  it("lists the structures, then the target's entries not opened yet", () => {
    const entries = basketEntries([panel("jobs/a/mol_0_model_0.cif"), panel("uploads/s/1M17.cif")], ["1m17", "4HJO", "4HJO"], title);
    expect(entries.map((entry) => entry.key)).toEqual(["jobs/a/mol_0_model_0.cif", "uploads/s/1M17.cif", "pdb:4HJO"]);
    expect(entries[2]).toMatchObject({ kind: "pdb", id: "4HJO", label: "PDB 4HJO" });
  });

  it("offers a handful of the target's entries, not all of them", () => {
    const ids = Array.from({ length: 20 }, (_, i) => `${i % 10}ABC`.slice(0, 4));
    expect(basketEntries([], ids, title).length).toBeLessThanOrEqual(8);
  });
});

describe("browseGroups", () => {
  const made = (path: string, tool = "boltz"): StructurePanel => ({ ...panel(path), tool });

  it("groups by where a structure came from, deposited entries first", () => {
    const entries = basketEntries(
      [made("predictions/run-1/complex.cif"), made("prep/egfr/receptor.pdb"), made("rcsb/3mxf.cif", "rcsb")],
      ["4HJO"],
      title,
    );
    const groups = browseGroups(entries, new Set());
    expect(groups.map((g) => g.label)).toEqual(["RCSB PDB", "Prepared", "Predicted"]);
    // A target entry not opened yet is a deposited one too.
    expect(groups[0].entries.map((e) => e.key)).toEqual(["rcsb/3mxf.cif", "pdb:4HJO"]);
  });

  it("leaves out a group with nothing in it", () => {
    expect(browseGroups(basketEntries([made("predictions/a/c.cif")], [], title), new Set()).map((g) => g.key)).toEqual(["predicted"]);
  });

  it("keeps a structure whose path says nothing rather than filing it under a guess", () => {
    const groups = browseGroups(basketEntries([made("uploads/thing.cif", "upload")], [], title), new Set());
    expect(groups.map((g) => g.label)).toEqual(["Other"]);
  });

  it("shows a kept structure under what it is as well as under Basket", () => {
    const entries = basketEntries([made("predictions/run-1/complex.cif")], [], title);
    const groups = browseGroups(entries, new Set(["predictions/run-1/complex.cif"]));
    expect(groups.map((g) => g.label)).toEqual(["Predicted", "Basket"]);
    expect(groups[1].entries).toHaveLength(1);
  });
});

describe("toggleShown", () => {
  it("adds at the end and takes out, but keeps the last one", () => {
    expect(toggleShown(["a"], "b")).toEqual(["a", "b"]);
    expect(toggleShown(["a", "b"], "a")).toEqual(["b"]);
    expect(toggleShown(["a"], "a")).toEqual(["a"]);
  });
});

describe("basketColour", () => {
  it("colours by place among those shown", () => {
    expect(basketColour(["a", "b"], "b")).toBe(BASKET_COLOURS[1]);
    expect(basketColour(["a", "b"], "c")).toBeNull();
  });
});
