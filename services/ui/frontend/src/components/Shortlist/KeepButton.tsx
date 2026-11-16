// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { StructurePanel } from "../../types/api";
import { keptOf, toggleKept, useKept } from "../../lib/favorites";
import { useShortlistSession } from "../../lib/shortlist";

interface Props {
  panel: StructurePanel;
  /** Where it is kept from, kept with the entry. */
  source?: string;
  /** Defaults to the open conversation. */
  session?: string;
  className?: string;
}

/** Adds a structure to the conversation's basket, or takes it off. */
export function KeepButton({ panel, source, session, className = "" }: Props) {
  const fallback = useShortlistSession();
  const key = session ?? fallback;
  const on = useKept(key).some((e) => e.path === panel.path);
  return (
    <button
      type="button"
      aria-pressed={on}
      aria-label={on ? "Remove from the basket" : "Keep in the basket"}
      title={on ? "In the basket" : "Keep in the basket"}
      onClick={(event) => {
        event.stopPropagation();
        toggleKept(key, keptOf(panel, source));
      }}
      className={`grid size-5 shrink-0 cursor-pointer place-items-center rounded ${on ? "" : "text-faint hover:text-body"} ${className}`}
      style={on ? { color: "var(--color-warn)" } : undefined}
    >
      <svg width={12} height={12} viewBox="0 0 12 12" aria-hidden="true">
        <path
          d="M6 1.2l1.45 3 3.3.42-2.42 2.27.62 3.27L6 8.58 3.05 10.16l.62-3.27L1.25 4.62l3.3-.42z"
          fill={on ? "currentColor" : "none"}
          stroke="currentColor"
          strokeWidth={1}
          strokeLinejoin="round"
        />
      </svg>
    </button>
  );
}
