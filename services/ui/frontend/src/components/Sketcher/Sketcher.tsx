// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { lazy, Suspense, useState } from "react";
import { createPortal } from "react-dom";
import { Atom, Bond, Vec2, type Ketcher } from "ketcher-core";
import type { Theme } from "../../hooks/useTheme";
import { countAttachmentPoints, numberAttachmentPoints } from "../../lib/attachment";
import type { SketchMode } from "../../lib/sketcherBus";

// When a mode is pre-specified (scaffold / fragment) the toggle is hidden and
// the mode is locked to what the caller asked for. When opened from the
// generic Draw button (no mode), all three segments are shown so the user can
// switch to scaffold or fragment mode and place attachment points for
// LibInvent / LinkInvent without leaving the dialog.
const MODE_LABELS: Record<SketchMode, string> = {
  molecule: "Molecule",
  scaffold: "Scaffold",
  fragment: "Fragment",
};
const ALL_MODES: SketchMode[] = ["molecule", "scaffold", "fragment"];

// Ketcher and its WebAssembly chemistry engine are several megabytes: they
// load the first time the sketcher opens, never with the app.
const KetcherEditor = lazy(() => import("./KetcherEditor"));

interface Props {
  /** The molecule to start from; an empty canvas when unset. */
  smiles?: string;
  /** What is being drawn; a plain molecule when unset. */
  mode?: SketchMode;
  theme: Theme;
  onUse: (smiles: string) => void;
  onClose: () => void;
}

const TITLE: Record<SketchMode, string> = {
  molecule: "Sketch a molecule",
  scaffold: "Sketch a scaffold",
  fragment: "Sketch a linker fragment",
};

const HINT: Record<SketchMode, string> = {
  molecule: "Use puts the SMILES in your message.",
  scaffold: "Select an atom, then click Add * for every position to decorate. They are numbered for you.",
  fragment: "Select the atom where the linker joins, then click Add * once.",
};

/** A molecule editor in a dialog; "Use" hands back what was drawn, as SMILES. */
export function Sketcher({ smiles, mode = "molecule", theme, onUse, onClose }: Props) {
  const [ketcher, setKetcher] = useState<Ketcher | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [reading, setReading] = useState(false);
  // When a specific mode was requested by the caller the toggle stays hidden.
  // When opened from the generic Draw button (mode prop is undefined / "molecule"
  // by default), the user can freely switch to scaffold or fragment to place
  // attachment points for LibInvent / LinkInvent.
  const callerLocked = smiles !== undefined || mode !== "molecule";
  const [activeMode, setActiveMode] = useState<SketchMode>(mode);

  function switchMode(next: SketchMode) {
    setActiveMode(next);
    setKetcher(null);  // remount KetcherEditor with the new toolbar
    setProblem(null);
  }

  /** What the drawing has to be before it is worth handing back. */
  function wrong(drawn: string): string | null {
    const points = countAttachmentPoints(drawn);
    if (activeMode === "scaffold" && points === 0) {
      return "A scaffold needs at least one attachment point — the position to decorate.";
    }
    if (activeMode === "fragment" && points !== 1) {
      return `A linker fragment needs exactly one attachment point, where the linker joins; this has ${points}.`;
    }
    return null;
  }

  async function use() {
    if (!ketcher) return;
    setReading(true);
    setProblem(null);
    try {
      const drawn = (await ketcher.getSmiles()).trim();
      if (!drawn) {
        setProblem("Draw a molecule first.");
        return;
      }
      const bad = wrong(drawn);
      if (bad) {
        setProblem(bad);
        return;
      }
      // The sketcher draws the position but never its number, and both
      // generators refuse a bare [*], so number it on the way out.
      onUse(activeMode === "molecule" ? drawn : numberAttachmentPoints(drawn));
    } catch (reason) {
      setProblem(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setReading(false);
    }
  }

  function addAttachmentPoint() {
    if (!ketcher) return;
    const selected = ketcher.editor.selection()?.atoms ?? [];
    if (selected.length !== 1) {
      setProblem("Select exactly one atom, then click Add *.");
      return;
    }
    const next = ketcher.editor.struct().clone();
    const parent = next.atoms.get(selected[0]);
    if (!parent) {
      setProblem("The selected atom is no longer in the drawing.");
      return;
    }
    const distance = next.getAvgBondLength() || 1;
    const star = next.atoms.add(
      new Atom({
        label: "*",
        fragment: parent.fragment,
        pp: new Vec2(parent.pp.x + distance, parent.pp.y),
      }),
    );
    next.bonds.add(new Bond({ begin: selected[0], end: star, type: Bond.PATTERN.TYPE.SINGLE }));
    next.markFragments();
    ketcher.editor.struct(next, true);
    ketcher.editor.selection({ atoms: [star] });
    setProblem(null);
  }

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-2 sm:p-6" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Molecule sketcher"
        className="bg-raised border-subtle shadow-card flex h-[min(46rem,94vh)] w-[min(68rem,98vw)] flex-col overflow-hidden rounded-xl border"
      >
        <header className="border-subtle flex h-11 shrink-0 items-center gap-2 border-b px-4">
          <span aria-hidden="true" className="size-2 rounded-full" style={{ background: "var(--color-molecule)" }} />
          <h2 className="text-sm font-semibold">{TITLE[activeMode]}</h2>
          {/* Mode toggle — hidden when the caller locked the mode (e.g. TTP
              scaffold picker). Shown when opened from the generic Draw button
              so the user can switch to scaffold/fragment to place attachment
              points for LibInvent / LinkInvent. */}
          {!callerLocked && (
            <div className="border-subtle ml-2 flex overflow-hidden rounded-md border text-[11px]">
              {ALL_MODES.map((m) => (
                <button
                  key={m}
                  type="button"
                  onClick={() => switchMode(m)}
                  className={`cursor-pointer px-2.5 py-0.5 ${activeMode === m ? "bg-accent text-white" : "text-muted hover:text-body"}`}
                  style={activeMode === m ? { background: "var(--color-accent)" } : undefined}
                >
                  {MODE_LABELS[m]}
                </button>
              ))}
            </div>
          )}
          <button
            type="button"
            onClick={onClose}
            aria-label="Close the sketcher"
            className="text-muted hover:text-body ml-auto grid size-7 cursor-pointer place-items-center rounded-lg text-base"
          >
            ×
          </button>
        </header>

        <div className={`ketcher-host relative min-h-0 flex-1 ${theme === "dark" ? "ketcher-dark" : ""}`}>
          {/* key={activeMode} remounts KetcherEditor on mode switch so the
              toolbar updates immediately. The drawing is intentionally cleared
              because a molecule is not valid as a scaffold (and vice versa). */}
          <Suspense key={activeMode} fallback={<p className="text-muted grid h-full place-items-center text-xs">Loading the sketcher…</p>}>
            <KetcherEditor initial={smiles} mode={activeMode} onReady={setKetcher} onError={setProblem} />
          </Suspense>
        </div>

        <footer className="border-subtle flex shrink-0 items-center gap-2 border-t px-4 py-2.5">
          <span className="text-muted min-w-0 flex-1 truncate text-xs" role="status" style={problem ? { color: "var(--color-accent)" } : undefined}>
            {problem ?? (ketcher ? HINT[activeMode] : "")}
          </span>
          {activeMode !== "molecule" && (
            <button
              type="button"
              onClick={addAttachmentPoint}
              disabled={!ketcher}
              className="border-subtle text-body hover:bg-sunken h-8 cursor-pointer rounded-lg border px-3 text-xs font-semibold disabled:cursor-not-allowed disabled:opacity-40"
            >
              Add *
            </button>
          )}
          <button
            type="button"
            onClick={onClose}
            className="text-muted hover:text-body hover:bg-sunken h-8 cursor-pointer rounded-lg px-3 text-xs font-medium"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={() => void use()}
            disabled={!ketcher || reading}
            className="h-8 cursor-pointer rounded-lg px-4 text-xs font-semibold text-white hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
            style={{ background: "var(--color-accent)" }}
          >
            Use
          </button>
        </footer>
      </div>
    </div>,
    document.body,
  );
}
