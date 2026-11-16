// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useState } from "react";
import { fetchCapacity } from "../../lib/api";

interface Props {
  /** Sub-agents this conversation may run at once; 1 is working alone. */
  count: number;
  onSet: (count: number) => void;
}

/** The fixed choices; "…" after them is any number the deployment can host. */
const STEPS = [2, 4, 6, 8] as const;

/** Turning sub-agents on without saying how many means this many. */
const DEFAULT_MAX = 2;

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
  // Null until read, and null again when the deployment will not say: a
  // number typed then stands, because clamping to a guessed ceiling would
  // refuse a team the deployment may well be able to host.
  const [ceiling, setCeiling] = useState<number | null>(null);
  const [typing, setTyping] = useState(false);
  const [typed, setTyped] = useState("");
  const [reduced, setReduced] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchCapacity()
      .then((capacity) => {
        if (!cancelled) setCeiling(capacity.max_supervisors);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, []);

  const on = count > 1;
  const custom = on && !STEPS.includes(count as (typeof STEPS)[number]);

  /** Take a typed number: at least 1, and never more than the deployment can host. */
  function apply() {
    const wanted = Math.floor(Number(typed));
    if (!Number.isFinite(wanted) || wanted < 1) return;
    const allowed = ceiling === null ? wanted : Math.min(wanted, Math.max(1, ceiling));
    setReduced(allowed < wanted ? allowed : null);
    setTyping(false);
    setTyped("");
    onSet(allowed);
  }

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-label="Sub-agents"
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
          className="bg-raised border-subtle shadow-card absolute bottom-full left-0 z-20 mb-1 rounded-lg border p-2.5 text-xs"
          // Not while a number is being typed: the pointer drifting off would throw it away.
          onMouseLeave={() => !typing && setOpen(false)}
        >
          <div className="mb-2 flex items-center justify-between gap-3">
            <span className="text-body font-medium">Sub-agents</span>
            <button
              type="button"
              role="switch"
              aria-checked={on}
              aria-label="Use sub-agents"
              onClick={() => onSet(on ? 1 : Math.min(DEFAULT_MAX, ceiling ?? DEFAULT_MAX))}
              className="relative h-4 w-7 shrink-0 cursor-pointer rounded-full transition-colors"
              style={{ background: on ? "var(--color-accent)" : "var(--text-faint)" }}
            >
              <span
                aria-hidden="true"
                className={`bg-raised absolute top-0.5 left-0.5 size-3 rounded-full shadow-sm transition-transform ${on ? "translate-x-3" : ""}`}
              />
            </button>
          </div>
          <div role="group" aria-label="At most" className="bg-sunken flex gap-0.5 rounded-md p-0.5">
            {STEPS.map((n) => (
              <Choice
                key={n}
                on={on && !custom && count === n}
                disabled={ceiling !== null && n > ceiling}
                onClick={() => { onSet(n); setReduced(null); }}
              >
                {n}
              </Choice>
            ))}
            {/* "…" expands to a type-a-number field; pressed state means a custom value is active. */}
            {typing ? (
              <input
                autoFocus
                type="number"
                min={2}
                max={ceiling ?? undefined}
                value={typed}
                onChange={(e) => setTyped(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") apply(); if (e.key === "Escape") setTyping(false); }}
                onBlur={apply}
                aria-label="Number of sub-agents"
                className="bg-raised text-body shadow-card w-12 rounded px-1.5 py-0.5 text-center text-[11px] tabular-nums outline-none"
              />
            ) : (
              <Choice on={custom} onClick={() => { setTyping(true); setTyped(custom ? String(count) : ""); }}>
                {custom ? count : "…"}
              </Choice>
            )}
          </div>
          {reduced !== null && (
            <p className="text-faint mt-1.5 text-[10px]">
              Reduced to {reduced} — deployment limit.
            </p>
          )}
        </div>
      )}
    </div>
  );
}

/** One choice in the row: how many at most. */
function Choice({
  on,
  disabled = false,
  title,
  onClick,
  children,
}: {
  on: boolean;
  disabled?: boolean;
  title?: string;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      aria-pressed={on}
      disabled={disabled}
      title={title}
      onClick={onClick}
      className={`min-w-7 cursor-pointer rounded px-1.5 py-0.5 text-center text-[11px] tabular-nums transition-colors disabled:cursor-default disabled:opacity-35 ${
        on ? "bg-raised text-body shadow-card" : "text-muted enabled:hover:text-body"
      }`}
    >
      {children}
    </button>
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
