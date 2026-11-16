// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * Which of a tool's structure files are worth a card in the conversation.
 *
 * Preparation writes a directory, not a file: alongside the complex and the
 * receptor it emits the waters it kept, the ions, a hydrogen-free receptor for
 * docking, and one file per ligand instance including every sulfate and
 * glycerol the crystallographer used. Every one of those paths became a card,
 * so preparing four entries buried the answer under thirty boxes of which
 * three were worth opening.
 *
 * Nothing is deleted: the filter is only about the chat, and the Structures
 * tab still lists everything.
 */

/** Preparation by-products: a file nobody opens on its own. */
const BY_PRODUCT = /^(water|waters|ions|receptor_noh)\.(pdb|cif|mmcif)$/i;

/**
 * Chemical component ids that are crystallisation chemistry rather than a
 * ligand: metal ions, buffers, cryoprotectants, detergents and solvents.
 */
const ADDITIVES = new Set([
  // Waters and the commonest counter-ions.
  "HOH", "DOD", "NA", "K", "LI", "CS", "RB", "CL", "BR", "IOD", "F",
  // Metals, which are often structural but are not what a card is for.
  "MG", "CA", "ZN", "MN", "FE", "FE2", "CU", "CU1", "NI", "CO", "CD", "HG",
  "SR", "BA", "AU", "AG", "PT", "PB", "W", "MO", "V",
  // Buffers, salts and cryoprotectants.
  "SO4", "PO4", "NO3", "ACT", "ACY", "EDO", "GOL", "PEG", "PG4", "PGE", "1PE",
  "2PE", "P6G", "TRS", "MES", "EPE", "BIS", "IMD", "FMT", "CIT", "FLC", "TLA",
  "MLA", "MPD", "DMS", "DMF", "EOH", "IPA", "MOH", "URE", "GAI", "AZI", "CN",
  "SCN", "BME", "DTT", "TCE", "CAC", "BCT", "CO3", "NH4", "OXY", "PER",
  // Detergents and lipids used to crystallise membrane proteins.
  "LDA", "BOG", "OCT", "C8E", "LMT", "UNL", "UNX",
]);

/** The chemical component id of a `ligand_<CID>_<CHAIN>_<SEQ>` file, if the name is one. */
export function ligandCompId(name: string): string | null {
  const match = /^ligand_([A-Za-z0-9]{1,3})_[A-Za-z0-9]+_-?\d+\.(pdb|cif|mmcif|sdf|mol2?)$/i.exec(name);
  return match ? match[1].toUpperCase() : null;
}

/** Whether a structure file is crystallisation chemistry or a preparation by-product. */
export function isStructureNoise(path: string): boolean {
  const name = path.replace(/\\/g, "/").split("/").pop() ?? "";
  if (BY_PRODUCT.test(name)) return true;
  const comp = ligandCompId(name);
  return comp !== null && ADDITIVES.has(comp);
}

/**
 * The structures worth showing, and the rest. Both are kept: the overflow
 * list names everything, so a sulfate is still one click away.
 */
export function partitionStructures<T extends { path: string }>(
  structures: readonly T[],
): { shown: T[]; noise: T[] } {
  const shown: T[] = [];
  const noise: T[] = [];
  for (const structure of structures) (isStructureNoise(structure.path) ? noise : shown).push(structure);
  return { shown, noise };
}
