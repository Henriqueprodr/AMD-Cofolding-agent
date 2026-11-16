// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useMemo, useState } from "react";
import type { CampaignDetail } from "../../types/api";
import { agentPage, knownAgent, type AgentPage } from "../../lib/fleet";
import { useFocusedAgent, useFocusedTask } from "../../lib/workbenchBus";

/** The open sub-agent and what its page shows; both null on the conductor's. */
export function useAgentPage(details: readonly CampaignDetail[], labels: ReadonlyMap<string, string>): { agent: string | null; page: AgentPage | null } {
  const focused = useFocusedAgent();
  const picked = useFocusedTask()?.taskId ?? null;
  const agent = knownAgent(focused, labels) ? focused : null;
  const page = useMemo(() => (agent ? agentPage(details, labels, agent, picked) : null), [details, labels, agent, picked]);
  return { agent, page };
}

/**
 * The clock, re-read every *everyMs*.
 *
 * The graph places running tasks at "now" and the headings say how long ago
 * things happened, so both need a clock that moves; reading Date.now() while
 * rendering would make the output depend on when React happened to redraw.
 */
export function useNow(everyMs = 10_000): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), everyMs);
    return () => window.clearInterval(timer);
  }, [everyMs]);
  return now;
}

/** The tail of an id, which is the part that tells two of them apart. */
export function shortId(id: string): string {
  return id.length > 10 ? `…${id.slice(-8)}` : id;
}
