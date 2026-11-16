// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { StructurePanel } from "../types/api";
import { structureKind } from "./poses";

/**
 * The structure basket: what the viewer can show of the conversation, and
 * which of it is shown at once, each structure in its own colour.
 *
 * The first one shown is loaded as usual; the others are superposed on it by
 * their proteins. Molecules are not in it: one only has a place beside a
 * protein once it has been cofolded or docked, and then it is a structure.
 */

/** Apart on both a dark and a light canvas, and from the element colours of the atoms that are not carbon. */
export const BASKET_COLOURS = ["#4e9af1", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#edc948", "#76b7b2", "#ff9da7"] as const;

export type BasketEntry = { key: string; label: string } & ({ kind: "structure"; panel: StructurePanel } | { kind: "pdb"; id: string });

/** Target entries offered before the list stops being a list. */
const PDB_SHOWN = 8;

const stem = (path: string) => (path.split("/").pop() ?? path).replace(/\.(cif|mmcif|pdb|ent)$/i, "").toUpperCase();

/**
 * The conversation's structures, then the PDB entries of its target that it
 * has not opened yet, which are fetched when picked.
 */
export function basketEntries(structures: readonly StructurePanel[], pdbIds: readonly string[], title: (panel: StructurePanel) => string): BasketEntry[] {
  const seen = new Set<string>();
  const out: BasketEntry[] = [];
  for (const panel of structures) {
    if (seen.has(panel.path)) continue;
    seen.add(panel.path);
    out.push({ kind: "structure", key: panel.path, label: title(panel), panel });
  }
  const opened = new Set(structures.map((panel) => stem(panel.path)));
  for (const id of pdbIds.slice(0, PDB_SHOWN)) {
    const code = id.toUpperCase();
    if (opened.has(code)) continue;
    opened.add(code);
    out.push({ kind: "pdb", key: `pdb:${code}`, label: `PDB ${code}`, id: code });
  }
  return out;
}

/**
 * The basket as a flat strip was unreadable past a handful of entries: a
 * conversation that retrieves a dozen PDB entries, prepares each and folds
 * against them puts fifty chips in one scrolling row, all alike. Browsing
 * is by where a structure came from instead.
 */
export type BrowseCategory = "experimental" | "prepared" | "predicted" | "other" | "kept";

export interface BrowseGroup {
  key: BrowseCategory;
  label: string;
  hint: string;
  entries: BasketEntry[];
}

const BROWSE: Record<BrowseCategory, { label: string; hint: string }> = {
  experimental: { label: "RCSB PDB", hint: "Deposited entries, retrieved or offered by the target" },
  prepared: { label: "Prepared", hint: "Receptors and complexes preparation wrote" },
  predicted: { label: "Predicted", hint: "Cofolded and docked complexes" },
  other: { label: "Other", hint: "Structures whose path does not say where they came from" },
  kept: { label: "Basket", hint: "What you kept, from here and from the chat" },
};

/** Kept last: it repeats entries the groups above already hold, which is the point of it. */
const BROWSE_ORDER: BrowseCategory[] = ["experimental", "prepared", "predicted", "other", "kept"];

/**
 * The entries grouped for browsing, empty groups left out.
 *
 * A kept structure appears both under what it is and under Basket, since
 * the Basket is a shortcut to it rather than another place it lives.
 */
export function browseGroups(entries: readonly BasketEntry[], kept: ReadonlySet<string>): BrowseGroup[] {
  const groups = new Map<BrowseCategory, BasketEntry[]>();
  const into = (key: BrowseCategory, entry: BasketEntry) => {
    const list = groups.get(key) ?? [];
    list.push(entry);
    groups.set(key, list);
  };
  for (const entry of entries) {
    if (entry.kind === "pdb") {
      into("experimental", entry);
      continue;
    }
    into(structureKind(entry.panel.path, Boolean(entry.panel.metrics), entry.panel.tool) ?? "other", entry);
    if (kept.has(entry.panel.path)) into("kept", entry);
  }
  return BROWSE_ORDER.flatMap((key) => {
    const list = groups.get(key);
    return list?.length ? [{ key, label: BROWSE[key].label, hint: BROWSE[key].hint, entries: list }] : [];
  });
}

/**
 * Shown after ticking *key*: added at the end, or taken out. The last one
 * shown stays, since the viewer always shows something.
 */
export function toggleShown(shown: readonly string[], key: string): string[] {
  if (!shown.includes(key)) return [...shown, key];
  return shown.length > 1 ? shown.filter((one) => one !== key) : [...shown];
}

/** The colour of a shown structure, by its place among those shown; null for one not shown. */
export function basketColour(shown: readonly string[], key: string): string | null {
  const at = shown.indexOf(key);
  return at < 0 ? null : BASKET_COLOURS[at % BASKET_COLOURS.length];
}
