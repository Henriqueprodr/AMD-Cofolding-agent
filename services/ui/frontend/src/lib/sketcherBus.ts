// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * Lets any panel open the molecule sketcher: "Edit in sketcher" on a
 * molecule card, a SMILES chip in the composer, the composer's Sketch button.
 * The composer hosts the sketcher, since what the user draws goes into the
 * message.
 */
/**
 * What is being drawn. A scaffold and a linker fragment are molecules with a
 * hole in them: they are only valid with attachment points, and the generators
 * that take them want those points numbered. Drawn as a plain molecule they
 * look finished and are refused later, so the sketcher has to know which is
 * being asked for.
 */
export type SketchMode = "molecule" | "scaffold" | "fragment";

export interface SketchRequest {
  /** Start from this molecule; an empty canvas when unset. */
  smiles?: string;
  /** Replace this SMILES of the message with the result, instead of inserting it. */
  replace?: string;
  /** What the result has to be; a plain molecule when unset. */
  mode?: SketchMode;
  /** Receive the result here instead of putting it in the message. */
  onResult?: (smiles: string) => void;
}

type Listener = (request: SketchRequest) => void;

const listeners = new Set<Listener>();

export function openSketcher(smiles?: string, replace?: string): void {
  listeners.forEach((listener) => listener({ smiles, replace }));
}

/** Open the sketcher for a scaffold or a linker fragment, handing the result back. */
export function sketchFor(
  mode: SketchMode,
  smiles: string | undefined,
  onResult: (smiles: string) => void,
): void {
  listeners.forEach((listener) => listener({ smiles, mode, onResult }));
}

export function onSketchRequest(listener: Listener): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}
