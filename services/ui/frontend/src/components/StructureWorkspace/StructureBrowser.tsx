// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useMemo, useState } from "react";
import type { Stacked } from "../../lib/molstar";
import { basketColour, browseGroups, type BasketEntry, type BrowseCategory } from "../../lib/basket";
import { toggleKept, useKept, keptOf } from "../../lib/favorites";

interface Props {
  entries: readonly BasketEntry[];
  /** The structures on screen, the active one first; their order sets their colours. */
  shown: readonly string[];
  /** The ones superposed on the first, in the order of `shown` after it. */
  others: readonly string[];
  placed: Stacked[] | null;
  onPick: (entry: BasketEntry) => void;
  session: string;
}

/**
 * The conversation's structures, browsed by where they came from: the
 * deposited entries, what preparation wrote, what was cofolded or docked,
 * and what the user kept.
 *
 * One group is open at a time and scrolls on its own, so a conversation
 * with fifty structures is a short list under a heading rather than fifty
 * identical chips in one scrolling row. Picking one shows it with the
 * others, superposed and in its own colour, as the strip did.
 */
export function StructureBrowser({ entries, shown, others, placed, onPick, session }: Props) {
  const kept = useKept(session);
  const keptPaths = useMemo(() => new Set(kept.map((one) => one.path)), [kept]);
  const groups = useMemo(() => browseGroups(entries, keptPaths), [entries, keptPaths]);
  const [opened, setOpened] = useState<BrowseCategory | null>(null);
  // Whichever group holds what is on screen, until one is chosen: opening on
  // a group that does not hold the active structure hides where you are.
  const active = opened ?? groups.find((group) => group.entries.some((entry) => shown.includes(entry.key)))?.key ?? groups[0]?.key ?? null;
  const group = groups.find((one) => one.key === active) ?? null;

  if (groups.length === 0) return null;

  return (
    <div className="border-subtle shrink-0 border-b">
      <div role="tablist" aria-label="Structures" className="flex items-center gap-1 overflow-x-auto px-3 pt-1">
        {groups.map((one) => {
          const selected = one.key === active;
          const on = one.entries.filter((entry) => shown.includes(entry.key)).length;
          return (
            <button
              key={one.key}
              type="button"
              role="tab"
              aria-selected={selected}
              title={one.hint}
              onClick={() => setOpened(one.key)}
              className={`shrink-0 cursor-pointer rounded-t-md px-2 py-1 text-[11px] whitespace-nowrap ${selected ? "bg-sunken text-body" : "text-muted hover:text-body"}`}
            >
              {one.label}
              <span className="text-faint ml-1">{one.entries.length}</span>
              {on > 0 && <span aria-hidden="true" className="ml-1 inline-block size-1.5 rounded-full align-middle" style={{ background: "var(--color-accent)" }} />}
            </button>
          );
        })}
      </div>
      {group && (
        <ul className="bg-sunken max-h-32 overflow-y-auto px-2 py-1">
          {group.entries.map((entry) => {
            const colour = basketColour(shown, entry.key);
            const fit = placed?.[others.indexOf(entry.key)];
            const note = fit ? (fit.superposed ? (fit.caRmsd !== null ? `${fit.caRmsd.toFixed(1)} Å` : null) : "not superposed") : null;
            return (
              <li key={entry.key} className="flex items-center gap-1">
                <button
                  type="button"
                  aria-pressed={colour !== null}
                  onClick={() => onPick(entry)}
                  title={
                    fit && !fit.superposed
                      ? "Its protein shares too little with the first one's to superpose; it is drawn in its own frame."
                      : colour
                        ? "Shown; click to take it out"
                        : "Show it with the others"
                  }
                  className={`hover:bg-inset flex min-w-0 flex-1 cursor-pointer items-center gap-1.5 rounded px-1 py-0.5 text-left text-[11px] ${colour ? "text-body" : "text-muted"}`}
                >
                  <span aria-hidden="true" className="size-2.5 shrink-0 rounded-full border" style={{ background: colour ?? "transparent", borderColor: colour ?? "currentColor" }} />
                  <span className="truncate">{entry.label}</span>
                  {note && <span className="text-faint ml-auto shrink-0 text-[10px]">{note}</span>}
                </button>
                {entry.kind === "structure" && (
                  <button
                    type="button"
                    aria-pressed={keptPaths.has(entry.panel.path)}
                    aria-label={keptPaths.has(entry.panel.path) ? `Remove ${entry.label} from the basket` : `Keep ${entry.label} in the basket`}
                    title={keptPaths.has(entry.panel.path) ? "In the basket" : "Keep in the basket"}
                    onClick={() => toggleKept(session, keptOf(entry.panel, "structures"))}
                    className={`grid size-5 shrink-0 cursor-pointer place-items-center rounded ${keptPaths.has(entry.panel.path) ? "" : "text-faint hover:text-body"}`}
                    style={keptPaths.has(entry.panel.path) ? { color: "var(--color-warn)" } : undefined}
                  >
                    <svg width={11} height={11} viewBox="0 0 12 12" aria-hidden="true">
                      <path
                        d="M6 1.2l1.45 3 3.3.42-2.42 2.27.62 3.27L6 8.58 3.05 10.16l.62-3.27L1.25 4.62l3.3-.42z"
                        fill={keptPaths.has(entry.panel.path) ? "currentColor" : "none"}
                        stroke="currentColor"
                        strokeWidth={1}
                        strokeLinejoin="round"
                      />
                    </svg>
                  </button>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
