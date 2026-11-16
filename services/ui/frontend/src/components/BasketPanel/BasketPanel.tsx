// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useMemo, useState } from "react";
import type { Theme } from "../../hooks/useTheme";
import { basketGroups, dropKept, useKept, type BasketItem } from "../../lib/favorites";
import { removeStar, useShortlist } from "../../lib/shortlist";
import { openMolecule } from "../../lib/moleculePage";
import { sendToViewer } from "../../lib/viewerBus";
import { MoleculeThumb } from "../MoleculeTable/MoleculeTable";

interface Props {
  sessionId: string | null;
  theme: Theme;
}

/**
 * What the user kept of this conversation: the structures starred where they
 * appeared and the molecules starred in the runs, as one tree.
 *
 * The groups are what to look under rather than where it was stored -- the
 * retrieved entries, the prepared receptors, the predicted complexes, then
 * the molecules by the run they came from -- so the basket reads the same
 * way as the Structures tab browses.
 */
export function BasketPanel({ sessionId, theme }: Props) {
  const session = sessionId ?? "new";
  const structures = useKept(session);
  const molecules = useShortlist(session);
  const groups = useMemo(() => basketGroups(structures, molecules), [structures, molecules]);
  const [closed, setClosed] = useState<ReadonlySet<string>>(new Set());

  if (groups.length === 0) {
    return (
      <p className="text-faint flex-1 px-8 py-12 text-center text-xs leading-relaxed">
        Nothing kept yet. The star beside a structure or a molecule — in the chat, in the Structures tab, in a run&apos;s results or
        analysis — puts it here.
      </p>
    );
  }

  return (
    <div className="min-h-0 flex-1 space-y-2 overflow-y-auto p-3">
      {groups.map((group) => {
        const open = !closed.has(group.key);
        return (
          <section key={group.key} className="bg-raised border-subtle shadow-card overflow-hidden rounded-xl border">
            <h3>
              <button
                type="button"
                aria-expanded={open}
                onClick={() =>
                  setClosed((was) => {
                    const next = new Set(was);
                    if (open) next.add(group.key);
                    else next.delete(group.key);
                    return next;
                  })
                }
                className="hover:bg-sunken flex w-full cursor-pointer items-center gap-2 px-3 py-2 text-left text-xs font-semibold"
              >
                <span className="text-faint w-2 shrink-0 text-[10px]">{open ? "▾" : "▸"}</span>
                <span className="min-w-0 truncate">{group.label}</span>
                <span className="text-faint ml-auto shrink-0 text-[11px] font-normal">{group.items.length}</span>
              </button>
            </h3>
            {open && (
              <ul className="border-subtle border-t">
                {group.items.map((item) => (
                  <li key={item.key} className="border-subtle flex items-center gap-2 border-b px-3 py-1.5 last:border-b-0">
                    <Item item={item} session={session} theme={theme} />
                  </li>
                ))}
              </ul>
            )}
          </section>
        );
      })}
    </div>
  );
}

function Item({ item, session, theme }: { item: BasketItem; session: string; theme: Theme }) {
  if (item.kind === "structure") {
    return (
      <>
        <button
          type="button"
          onClick={() => sendToViewer({ path: item.entry.path, source: "basket" })}
          title={item.entry.path}
          className="min-w-0 flex-1 cursor-pointer truncate text-left text-[11px]"
        >
          {item.label}
          {item.entry.source && <span className="text-faint ml-1.5 font-mono text-[10px]">{item.entry.source}</span>}
        </button>
        <Drop onClick={() => dropKept(session, item.entry.path)} label={`Remove ${item.label} from the basket`} />
      </>
    );
  }
  return (
    <>
      <MoleculeThumb smiles={item.entry.smiles} theme={theme} width={56} height={36} />
      <button
        type="button"
        onClick={() => openMolecule(item.entry.smiles)}
        title={item.entry.smiles}
        className="min-w-0 flex-1 cursor-pointer truncate text-left font-mono text-[10px]"
      >
        {item.label}
      </button>
      {item.entry.values?.total_score !== undefined && (
        <span className="text-muted shrink-0 font-mono text-[10px]">{item.entry.values.total_score.toFixed(3)}</span>
      )}
      <Drop onClick={() => removeStar(session, item.entry.smiles)} label="Remove from the basket" />
    </>
  );
}

function Drop({ onClick, label }: { onClick: () => void; label: string }) {
  return (
    <button type="button" onClick={onClick} aria-label={label} className="text-faint hover:text-body shrink-0 cursor-pointer px-1 text-[13px] leading-none">
      ×
    </button>
  );
}
