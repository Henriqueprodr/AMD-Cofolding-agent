// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * Attachment points, for the two generators that decorate a structure rather
 * than invent a whole one.
 *
 * LibInvent takes a scaffold and fills its R-group positions; LinkInvent takes
 * two fragments and grows a linker between them. Both mark the open positions
 * with a dummy atom, and both require it to carry a map number -- `[*:1]`, not
 * a bare `[*]` (see _validate_libinvent_scaffold in reinvent4_mcp/ttp/
 * schema.py). A sketcher draws the position but not the number, so the number
 * has to be put on here; unnumbered, the profile is refused on submit with an
 * error about a structure the user drew correctly.
 */

/** A dummy atom as SMILES writes it: `*`, `[*]`, or `[*:1]`. */
const POINT = /\[\*(?::(\d+))?\]|\*/g;

/** How many attachment points a SMILES has. */
export function countAttachmentPoints(smiles: string): number {
  return [...smiles.matchAll(POINT)].length;
}

/** The map numbers already written, in the order they appear. */
export function mapNumbers(smiles: string): number[] {
  return [...smiles.matchAll(POINT)].flatMap((m) => (m[1] ? [Number(m[1])] : []));
}

/**
 * The same SMILES with every bare attachment point numbered.
 *
 * Numbers already written are kept, so a scaffold edited twice keeps its
 * positions; the rest are filled from the lowest number not in use.
 */
export function numberAttachmentPoints(smiles: string, from = 1): string {
  const used = new Set(mapNumbers(smiles));
  let next = from;
  const take = () => {
    while (used.has(next)) next += 1;
    used.add(next);
    return next;
  };
  return smiles.replace(POINT, (whole, num: string | undefined) => (num ? whole : `[*:${take()}]`));
}

/** Fragment SMILES from one field: separated by whitespace, commas or a pipe. */
export const fragmentList = (text: string) => text.split(/[\s,|]+/).filter(Boolean);

/** Why LibInvent would refuse this scaffold. */
export function scaffoldProblems(smiles: string): string[] {
  const text = smiles.trim();
  if (!text) return ["Draw or paste a scaffold with at least one attachment point."];

  const points = countAttachmentPoints(text);
  if (points === 0) {
    return ["A LibInvent scaffold needs at least one attachment point — the position to decorate."];
  }

  const numbers = mapNumbers(text);
  const problems: string[] = [];
  if (numbers.length !== points) problems.push("Every attachment point needs a number, like [*:1].");
  if (new Set(numbers).size !== numbers.length) problems.push("Two attachment points share a number.");
  return problems;
}

/** Why LinkInvent would refuse these fragments. */
export function fragmentProblems(fragments: readonly string[]): string[] {
  if (fragments.length < 2) return ["LinkInvent needs two fragments, one for each side of the linker."];

  const problems: string[] = [];
  fragments.forEach((smiles, index) => {
    const points = countAttachmentPoints(smiles);
    if (points !== 1) {
      problems.push(
        `Fragment ${index + 1} has ${points} attachment points; each fragment needs exactly one, where the linker joins.`,
      );
    }
  });

  const numbers = fragments.flatMap((s) => mapNumbers(s));
  const numbered = fragments.reduce((n, s) => n + countAttachmentPoints(s), 0);
  if (numbers.length !== numbered) problems.push("Every attachment point needs a number, like [*:1].");
  else if (new Set(numbers).size !== numbers.length) problems.push("The two fragments share an attachment-point number.");
  return problems;
}

/**
 * Two fragments numbered so their attachment points do not collide.
 *
 * Numbering each on its own would give both `[*:1]`, which LinkInvent refuses
 * because it cannot tell the two ends of the linker apart.
 */
export function numberFragments(fragments: readonly string[]): string[] {
  let next = 1;
  return fragments.map((smiles) => {
    const out = numberAttachmentPoints(smiles, next);
    next = Math.max(next, ...mapNumbers(out)) + 1;
    return out;
  });
}
