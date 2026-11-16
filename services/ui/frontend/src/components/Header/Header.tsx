// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { ReactNode } from "react";
import type { ConnectionState } from "../../types/api";
import type { Theme } from "../../hooks/useTheme";
import type { AgentInfo } from "../../hooks/useSession";

interface Props {
  theme: Theme;
  onToggleTheme: () => void;
  connection: ConnectionState;
  agent: AgentInfo | null;
  gatewayError: string | null;
  title: string;
  onReconnect: () => void;
  railOpen: boolean;
  onToggleRail: () => void;
  workbenchOpen: boolean;
  onToggleWorkbench: () => void;
}

const CONNECTION_LABEL: Record<ConnectionState, string> = {
  connecting: "Connecting",
  open: "Connected",
  closed: "Reconnecting",
  error: "Disconnected",
};

/**
 * The head of the chat column: which conversation this is and whether the
 * agent can be reached. The product name and the model live
 * in the conversation rail, so this line stays about the work.
 */
export function Header({
  theme,
  onToggleTheme,
  connection,
  gatewayError,
  title,
  onReconnect,
  railOpen,
  onToggleRail,
  workbenchOpen,
  onToggleWorkbench,
}: Props) {
  const healthy = connection === "open";

  return (
    <header className="border-subtle flex h-14 shrink-0 items-center gap-3 border-b px-4">
      {!railOpen && (
        <IconButton label="Show conversations" onClick={onToggleRail}>
          <svg viewBox="0 0 24 24" className="size-4" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
            <rect x="3" y="4" width="18" height="16" rx="2" />
            <path d="M9 4v16" />
          </svg>
        </IconButton>
      )}

      <div className="min-w-0 flex-1">
        <h1 className="truncate text-sm font-semibold">
          {title || "New conversation"}
        </h1>
      </div>

      <div className="flex shrink-0 items-center gap-1 sm:gap-1.5">
        <button
          type="button"
          onClick={healthy ? undefined : onReconnect}
          disabled={healthy}
          aria-label={CONNECTION_LABEL[connection]}
          className="text-muted flex items-center gap-1.5 px-1.5 text-[11px] disabled:cursor-default touch:min-h-11 touch:min-w-11 touch:justify-center"
        >
          <span
            aria-hidden="true"
            className={`size-1.5 rounded-full ${connection === "connecting" || connection === "closed" ? "animate-pulse" : ""}`}
            style={{ background: healthy && !gatewayError ? "var(--color-ok)" : "var(--color-accent)" }}
          />
          {!(healthy && !gatewayError) && <span className="hidden sm:inline">{CONNECTION_LABEL[connection].toLowerCase()}</span>}
        </button>

        <button
          type="button"
          onClick={onToggleWorkbench}
          aria-pressed={workbenchOpen}
          aria-label={workbenchOpen ? "Hide the workbench" : "Show the workbench"}
          aria-keyshortcuts="Control+B"
          className={`cursor-pointer rounded-md px-2.5 py-1 text-xs touch:min-h-11 ${workbenchOpen ? "text-body bg-inset" : "text-muted hover:text-body hover:bg-inset"}`}
        >
          Workbench
        </button>

        <IconButton label={theme === "dark" ? "Switch to the light theme" : "Switch to the dark theme"} onClick={onToggleTheme}>
          {theme === "dark" ? <SunIcon /> : <MoonIcon />}
        </IconButton>
      </div>
    </header>
  );
}

function IconButton({ label, onClick, children }: { label: string; onClick: () => void; children: ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      className="text-muted hover:text-body hover:bg-inset grid size-8 shrink-0 cursor-pointer place-items-center rounded-lg touch:size-11"
    >
      {children}
    </button>
  );
}

function SunIcon() {
  return (
    <svg viewBox="0 0 24 24" className="size-4" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" strokeLinecap="round" />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg viewBox="0 0 24 24" className="size-4" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
