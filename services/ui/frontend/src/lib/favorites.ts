// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * Structures the user kept, per conversation, in the browser beside the
 * molecule shortlist.
 *
 * A conversation produces far more structures than are worth coming back to:
 * every retrieved entry, every prepared receptor, every pose of every run.
 * Keeping one puts it in the Basket, which is the two shortlists -- kept
 * structures and starred molecules -- read as one place.
 *
 * An entry holds what was known where it was kept, since the panel it came
 * from scrolls away: the file is the identity, the rest is for reading.
 */

import { useSyncExternalStore } from "react";
import type { StructurePanel } from "../types/api";
import type { ShortlistEntry } from "./shortlist";
import { structureKind, type StructureKind } from "./poses";

export interface KeptStructure {
  /** The file, which is what identifies it. */
  path: string;
  label?: string;
  /** How it was made, so the Basket can group by it. */
  category?: StructureKind;
  /** Where it was kept from: a run id, a tool, "structures". */
  source?: string;
  /** The target it is of, where the panel knew one. */
  target?: string;
  /** The ligand it holds, for a complex. */
  smiles?: string;
  at: number;
}

const KEY = "cofold.basket:";
const listeners = new Set<() => void>();
const cache = new Map<string, KeptStructure[]>();
const EMPTY: KeptStructure[] = [];

const notify = () => listeners.forEach((listener) => listener());

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function read(session: string): KeptStructure[] {
  const cached = cache.get(session);
  if (cached) return cached;
  let list: KeptStructure[] = EMPTY;
  try {
    const raw = localStorage.getItem(KEY + session);
    const parsed: unknown = raw ? JSON.parse(raw) : null;
    if (Array.isArray(parsed)) list = parsed.filter((e): e is KeptStructure => !!e && typeof e.path === "string");
  } catch {
    list = EMPTY;
  }
  cache.set(session, list);
  return list;
}

function write(session: string, list: KeptStructure[]) {
  cache.set(session, list);
  try {
    if (list.length) localStorage.setItem(KEY + session, JSON.stringify(list));
    else localStorage.removeItem(KEY + session);
  } catch {
    // Storage full or blocked: the basket stays for this page only.
  }
  notify();
}

export function isKept(session: string, path: string): boolean {
  return read(session).some((e) => e.path === path);
}

/** Keep a structure, or drop it when it is already kept. Returns whether it is kept now. */
export function toggleKept(session: string, entry: Omit<KeptStructure, "at">): boolean {
  const list = read(session);
  if (list.some((e) => e.path === entry.path)) {
    write(
      session,
      list.filter((e) => e.path !== entry.path),
    );
    return false;
  }
  write(session, [...list, { ...entry, at: Date.now() }]);
  return true;
}

export function dropKept(session: string, path: string): void {
  write(
    session,
    read(session).filter((e) => e.path !== path),
  );
}

export function clearKept(session: string): void {
  write(session, []);
}

export function useKept(session: string): KeptStructure[] {
  // Rendered on the server there is no storage to read and nothing kept.
  return useSyncExternalStore(
    subscribe,
    () => read(session),
    () => EMPTY,
  );
}

/** What a panel is worth remembering as, for the keep button beside it. */
export function keptOf(panel: StructurePanel, source?: string): Omit<KeptStructure, "at"> {
  const category = structureKind(panel.path, Boolean(panel.metrics), panel.tool);
  return {
    path: panel.path,
    ...(panel.label || panel.name ? { label: panel.label ?? panel.name } : {}),
    ...(category ? { category } : {}),
    ...(source ? { source } : {}),
    ...(panel.smiles ? { smiles: panel.smiles } : {}),
  };
}

// -- The basket --------------------------------------------------------------

/** One line of the basket: a kept structure or a starred molecule. */
export type BasketItem =
  | { kind: "structure"; key: string; label: string; entry: KeptStructure }
  | { kind: "molecule"; key: string; label: string; entry: ShortlistEntry };

/** A branch of the basket tree: structures by how they were made, molecules by where they came from. */
export interface BasketGroup {
  key: string;
  label: string;
  items: BasketItem[];
}

const CATEGORY_LABEL: Record<StructureKind, string> = {
  experimental: "RCSB PDB",
  prepared: "Prepared",
  predicted: "Predicted",
};

const fileName = (path: string) => path.split("/").pop() ?? path;

/**
 * The basket as two levels: structures grouped by how they were made and
 * molecules by where they were starred, each group's newest first.
 *
 * Grouping is by what the user would look for rather than by the store they
 * came from, so a kept receptor and a kept pose are apart even though both
 * are structures.
 */
export function basketGroups(structures: readonly KeptStructure[], molecules: readonly ShortlistEntry[]): BasketGroup[] {
  const groups = new Map<string, BasketGroup>();
  const into = (key: string, label: string, item: BasketItem) => {
    const group = groups.get(key) ?? { key, label, items: [] };
    group.items.push(item);
    groups.set(key, group);
  };

  for (const entry of [...structures].sort((a, b) => b.at - a.at)) {
    const category = entry.category ?? "predicted";
    into(`structure:${category}`, CATEGORY_LABEL[category], {
      kind: "structure",
      key: entry.path,
      label: entry.label ?? fileName(entry.path),
      entry,
    });
  }
  for (const entry of [...molecules].sort((a, b) => b.at - a.at)) {
    // A run id is the most useful heading a starred molecule has; without
    // one they sit together rather than each in a group of its own.
    const source = entry.source && entry.source !== "molecules" ? entry.source : null;
    into(`molecule:${source ?? ""}`, source ? `Molecules from ${source}` : "Molecules", {
      kind: "molecule",
      key: entry.smiles,
      label: entry.label ?? entry.smiles,
      entry,
    });
  }

  // Structures before molecules, and within each the order of CATEGORY_LABEL,
  // which is the order the Structures tab browses them in.
  const order = ["structure:experimental", "structure:prepared", "structure:predicted"];
  return [...groups.values()].sort((a, b) => {
    const at = order.indexOf(a.key);
    const bt = order.indexOf(b.key);
    if (at >= 0 || bt >= 0) return (at < 0 ? order.length : at) - (bt < 0 ? order.length : bt);
    return a.label.localeCompare(b.label);
  });
}

export const basketCount = (structures: readonly KeptStructure[], molecules: readonly ShortlistEntry[]) => structures.length + molecules.length;
