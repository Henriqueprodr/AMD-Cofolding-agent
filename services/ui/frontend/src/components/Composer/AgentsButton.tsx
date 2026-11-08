// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useState } from "react";
import { fetchCapacity } from "../../lib/api";

interface Props {
  /** Sub-agents this conversation may run at once; 1 is working alone. */
  count: number;
  onSet: (count: number) => void;
}

/**
 * What to offer when the deployment will not say what it could run. Four is
 * enough for the work to divide and few enough to be safe anywhere; the
 * alternative, picking a number off an unreadable scheduler, would be a
 * guess dressed as a measurement.
 */
const FALLBACK_CEILING = 4;

/**
 * The scheduler's answer is the whole partition, which is thousands and no
 * use as the end of a slider. Offer a working range and say what the real
 * ceiling is in the tooltip.
 */
const OFFERED = 8;

/**
 * How many sub-agents the conversation may run at once.
 *
 * Nothing starts here. Above one the agent is told it may spend them, and it
 * decides whether the work divides into experiments worth running side by
 * side — which is what the user asked for: a ceiling they set, not a fleet
 * they launch.
 */
export function AgentsButton({ count, onSet }: Props) {
  const [open, setOpen] = useState(false);
  const [ceiling, setCeiling] = useState<number | null>(null);
  const [basis, setBasis] = useState<string | null>(null);

  useEffect(() => {
    if (!open || ceiling !== null) return;
    let cancelled = false;
    fetchCapacity()
      .then((capacity) => {
        if (cancelled) return;
        setCeiling(capacity.max_supervisors ?? FALLBACK_CEILING);
        setBasis(capacity.max_supervisors === null ? "the deployment would not say what it could run" : (capacity.detail ?? null));
      })
      .catch(() => {
        if (!cancelled) {
          setCeiling(FALLBACK_CEILING);
          setBasis("the ceiling could not be read");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [open, ceiling]);

  const max = Math.max(2, Math.min(OFFERED, ceiling ?? FALLBACK_CEILING));
  const on = count > 1;

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-label="Sub-agents"
        title={on ? `Up to ${count} sub-agents at once` : "Let the agent run experiments side by side"}
        className={
          on
            ? "flex h-7 shrink-0 cursor-pointer items-center gap-1 rounded-full border px-2.5 text-[11px] touch:h-9"
            : "text-muted hover:text-body hover:bg-sunken grid size-7 shrink-0 cursor-pointer place-items-center rounded-lg touch:size-9"
        }
        style={on ? { borderColor: "var(--color-accent)" } : undefined}
      >
        <AgentsIcon />
        {on && <span className="font-medium tabular-nums">{count}</span>}
      </button>
      {open && (
        <div
          role="dialog"
          aria-label="Sub-agents"
          className="bg-raised border-subtle shadow-card absolute bottom-full left-0 z-20 mb-1 w-60 rounded-lg border p-3 text-xs"
          onMouseLeave={() => setOpen(false)}
        >
          <div className="mb-2 flex items-baseline justify-between">
            <span className="text-faint">Sub-agents at once</span>
            <span className="tabular-nums">{count}</span>
          </div>
          <input
            type="range"
            min={1}
            max={max}
            step={1}
            value={Math.min(count, max)}
            onChange={(event) => onSet(Number(event.target.value))}
            aria-label="Sub-agents at once"
            className="w-full cursor-pointer accent-[var(--color-accent)]"
          />
          <div className="text-faint mt-0.5 flex justify-between text-[10px] tabular-nums">
            <span>1</span>
            <span>{max}</span>
          </div>
          <p className="text-faint mt-2 leading-snug">
            {count > 1
              ? "The agent may run experiments side by side and coordinate them. It still decides whether the work divides."
              : "The agent works alone."}
          </p>
          {basis && <p className="text-faint mt-1 text-[10px] leading-snug">{basis}</p>}
        </div>
      )}
    </div>
  );
}

function AgentsIcon() {
  return (
    <svg viewBox="0 0 24 24" className="size-4" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <circle cx="12" cy="5" r="2.2" />
      <circle cx="5.5" cy="18" r="2.2" />
      <circle cx="18.5" cy="18" r="2.2" />
      <path d="M12 7.2v3.3M12 10.5H6.8a1.3 1.3 0 0 0-1.3 1.3v4M12 10.5h5.2a1.3 1.3 0 0 1 1.3 1.3v4" />
    </svg>
  );
}
