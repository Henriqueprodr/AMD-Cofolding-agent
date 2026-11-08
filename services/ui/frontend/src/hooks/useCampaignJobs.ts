// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useState } from "react";
import { fetchCampaign, fetchCampaigns } from "../lib/api";
import { campaignActive } from "../lib/fleet";

/** Refreshed this often while a campaign is running; sub-agents start runs as they go. */
const POLL_MS = 15_000;

const NONE: ReadonlySet<string> = new Set();

/**
 * The REINVENT runs started by this conversation's campaigns.
 *
 * A sub-agent runs in its own session, so the gateway records it as the
 * launcher and the conversation's own list of runs leaves these out. They
 * are still the conversation's work: without them, asking for a campaign
 * empties the Jobs tab of the very runs it set going.
 *
 * Empty, never an error, when there is no ledger: a Jobs tab that showed
 * nothing because the fleet is away would be worse than one that shows what
 * it can.
 */
export function useCampaignJobs(sessionId: string | null | undefined): ReadonlySet<string> {
  const [jobs, setJobs] = useState<ReadonlySet<string>>(NONE);
  const [live, setLive] = useState(false);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;
    void (async () => {
      try {
        const campaigns = await fetchCampaigns(sessionId);
        if (cancelled) return;
        const details = await Promise.all(campaigns.map((campaign) => fetchCampaign(campaign.campaign_id).catch(() => null)));
        if (cancelled) return;
        const ids = new Set<string>();
        for (const detail of details) for (const task of detail?.tasks ?? []) if (task.job_id) ids.add(task.job_id);
        setJobs(ids.size ? ids : NONE);
        setLive(campaigns.some(campaignActive));
      } catch {
        // The ledger is away or this deployment has none. Not a reason to
        // tell the user anything: the Jobs tab has its own runs regardless.
        if (!cancelled) {
          setJobs(NONE);
          setLive(false);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [sessionId, tick]);

  useEffect(() => {
    if (!live) return;
    const timer = window.setInterval(() => setTick((t) => t + 1), POLL_MS);
    return () => window.clearInterval(timer);
  }, [live]);

  return jobs;
}
