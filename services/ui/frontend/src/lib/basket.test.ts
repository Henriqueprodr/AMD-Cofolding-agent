// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import type { StructurePanel } from "../types/api";
import { BASKET_COLOURS, basketColour, basketEntries, toggleShown } from "./basket";

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
