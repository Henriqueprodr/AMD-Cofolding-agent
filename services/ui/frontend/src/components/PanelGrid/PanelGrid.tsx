// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useMemo, useState, type ReactNode } from "react";
import type { InteractionsPanel, MoleculePanel, Panel, PropertiesPanel, StructurePanel } from "../../types/api";
import type { Theme } from "../../hooks/useTheme";
import { mergeTurnPanels, structureTitle } from "../../lib/artifacts";
import { isStructureNoise, partitionStructures } from "../../lib/structureNoise";
import { useOverflowCount } from "../../hooks/useOverflowCount";
import { sendToViewer } from "../../lib/viewerBus";
import { StructureCard } from "../StructureCard/StructureCard";
import { MoleculeCard } from "../MoleculeCard/MoleculeCard";
import { InteractionsCard } from "../InteractionsCard/InteractionsCard";
import { PropertiesCard } from "../PropertiesCard/PropertiesCard";

interface Props {
  panels: Panel[];
  theme: Theme;
  openStructurePath: string | null;
  onOpenMolecule?: (smiles: string) => void;
  onShowProperties?: () => void;
}

/** What tool calls produced: complexes and their contacts, predictions, then molecules. Tables feed the candidates, not the chat. */
export function PanelGrid({ panels, theme, openStructurePath, onOpenMolecule, onShowProperties }: Props) {
  const merged = useMemo(() => mergeTurnPanels(panels), [panels]);
  const of = <K extends Panel["kind"]>(kind: K) => merged.filter((panel) => panel.kind === kind) as Extract<Panel, { kind: K }>[];
  const structures: StructurePanel[] = of("structure");
  const interactions: InteractionsPanel[] = of("interactions");
  const properties: PropertiesPanel[] = of("properties");
  const molecules: MoleculePanel[] = of("molecule");

  return (
    <div className="@container space-y-3">
      {structures.length > 0 && <StructureRow structures={structures} theme={theme} openStructurePath={openStructurePath} />}

      {interactions.map((panel, index) => (
        <InteractionsCard key={panel.structurePath ?? `${panel.smiles}-${index}`} panel={panel} />
      ))}

      {properties.map((panel, index) => (
        <PropertiesCard key={index} panel={panel} theme={theme} limit={5} onOpen={onShowProperties} onOpenMolecule={onOpenMolecule} />
      ))}

      {molecules.length > 0 && (
        <Folded label={`${molecules.length} molecule${molecules.length === 1 ? "" : "s"}`}>
          <div className="grid gap-3 @md:grid-cols-2 @3xl:grid-cols-3">
            {molecules.map((panel) => (
              <MoleculeCard key={panel.smiles} panel={panel} theme={theme} onOpen={onOpenMolecule} />
            ))}
          </div>
        </Folded>
      )}

    </div>
  );
}

/** Narrowest a card may be before the row starts hiding them behind "…". */
const CARD_WIDTH = 224;

/**
 * One row of complexes, however many a tool produced.
 *
 * A grid that wrapped turned preparing four entries into six rows of boxes,
 * most of them a water file or a sulfate. The row keeps to its line: the
 * crystallisation chemistry and the preparation by-products go straight into
 * "…", and so does anything else that does not fit, where the whole list is
 * one click away.
 */
function StructureRow({ structures, theme, openStructurePath }: { structures: StructurePanel[]; theme: Theme; openStructurePath: string | null }) {
  const { shown, noise } = useMemo(() => partitionStructures(structures), [structures]);
  const [ref, fit] = useOverflowCount(shown.length, CARD_WIDTH, 40);
  const visible = shown.slice(0, fit);
  const rest = [...shown.slice(fit), ...noise];

  return (
    <div ref={ref} className="flex min-w-0 items-stretch gap-2">
      {visible.map((panel) => (
        <div key={panel.path} className="min-w-0 flex-1">
          <StructureCard panel={panel} theme={theme} isOpen={openStructurePath === panel.path} compact />
        </div>
      ))}
      {rest.length > 0 && <MoreStructures structures={rest} openStructurePath={openStructurePath} />}
    </div>
  );
}

/** The structures the row did not show, by name, opening in the one viewer. */
function MoreStructures({ structures, openStructurePath }: { structures: StructurePanel[]; openStructurePath: string | null }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="relative shrink-0">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        aria-label={`${structures.length} more structure${structures.length === 1 ? "" : "s"}`}
        title={`${structures.length} more: waters, ions and anything that did not fit`}
        className="bg-raised border-subtle text-muted hover:text-body h-full cursor-pointer rounded-xl border px-2.5 text-xs touch:min-w-11"
      >
        …
      </button>
      {open && (
        <ul className="bg-raised border-subtle absolute right-0 z-20 mt-1 max-h-72 w-64 overflow-y-auto rounded-lg border py-1 shadow-lg">
          {structures.map((panel) => (
            <li key={panel.path}>
              <button
                type="button"
                onClick={() => {
                  setOpen(false);
                  sendToViewer({ path: panel.path, source: "chat" });
                }}
                className={`hover:bg-sunken flex w-full cursor-pointer items-baseline gap-2 px-3 py-1.5 text-left text-[11px] ${openStructurePath === panel.path ? "text-body font-semibold" : "text-muted"}`}
              >
                <span className="min-w-0 truncate">{structureTitle(panel)}</span>
                {isStructureNoise(panel.path) && <span className="text-faint ml-auto shrink-0 text-[10px]">by-product</span>}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/** Results that the workbench also lists, one click away instead of in the answer's way. */
function Folded({ label, children }: { label: string; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div>
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        className="text-muted hover:text-body flex cursor-pointer items-center gap-1.5 text-[11px] font-medium touch:min-h-11"
      >
        <span aria-hidden="true" className={`inline-block transition-transform ${open ? "rotate-90" : ""}`}>
          ›
        </span>
        {label}
      </button>
      {open && <div className="mt-2">{children}</div>}
    </div>
  );
}
