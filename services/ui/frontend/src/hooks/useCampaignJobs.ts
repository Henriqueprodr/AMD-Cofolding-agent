// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useMemo, useSyncExternalStore } from "react";
import type { CampaignDetail, FleetEvent } from "../types/api";
import { ApiError, fetchCampaign, fetchCampaignEvents, fetchCampaigns } from "../lib/api";
import { campaignActive, taskPhase } from "../lib/fleet";
import { usePanelShown } from "./panelShownContext";

/** Refreshed this often while a campaign is running; sub-agents start runs as they go. */
const POLL_MS = 15_000;

const NONE: ReadonlySet<string> = new Set();
const NO_CAMPAIGNS: readonly CampaignDetail[] = [];
const NO_EVENTS: ReadonlyMap<string, readonly FleetEvent[]> = new Map();

/** What is known about one conversation's campaigns. */
export interface CampaignsRead {
  details: readonly CampaignDetail[];
  /** The ledger's history per campaign, for the step-by-step view. */
  events: ReadonlyMap<string, readonly FleetEvent[]>;
  /** Why the ledger could not be read; the panel shows this rather than an empty fleet. */
  error: string | null;
  /** Nothing has been read for this conversation yet, so empty means unknown. */
  loading: boolean;
  /** Anything still able to change, so the caller knows the numbers move. */
  live: boolean;
}

const UNREAD: CampaignsRead = { details: NO_CAMPAIGNS, events: NO_EVENTS, error: null, loading: true, live: false };

/**
 * One reading of the ledger per conversation, shared by everything that asks.
 *
 * Module-level rather than per hook because the Campaign tab is left and
 * reopened constantly, and three components ask for the same campaigns at
 * once. Held per mount it meant a fetch of the list plus one per campaign
 * every time, and seconds of "Reading the campaigns…" over an answer that
 * had not changed. Kept here, reopening shows what was read last and the
 * refresh happens behind it.
 */
const read = new Map<string, CampaignsRead>();
const reading = new Map<string, Promise<void>>();
const watchers = new Set<() => void>();

function announce(): void {
  for (const watcher of watchers) watcher();
}

async function load(sessionId: string): Promise<void> {
  const pending = reading.get(sessionId);
  if (pending) return pending;

  const run = (async () => {
    try {
      const campaigns = await fetchCampaigns(sessionId);
      const loaded = await Promise.all(
        campaigns.map(async (campaign) => ({
          detail: await fetchCampaign(campaign.campaign_id).catch(() => null),
          // The history is what the step-by-step view reads; a campaign is
          // still worth showing without it.
          events: await fetchCampaignEvents(campaign.campaign_id)
            .then((read) => read.events)
            .catch(() => []),
        })),
      );
      const found = loaded.map(({ detail }) => detail).filter((detail): detail is CampaignDetail => detail !== null);
      const events = new Map<string, readonly FleetEvent[]>();
      for (const one of loaded) if (one.detail) events.set(one.detail.campaign.campaign_id, one.events);
      read.set(sessionId, {
        details: found.length ? found : NO_CAMPAIGNS,
        events: events.size ? events : NO_EVENTS,
        error: null,
        loading: false,
        // A closed campaign whose sub-agents still hold tasks is still moving.
        live: campaigns.some(campaignActive) || found.some((detail) => !detail.campaign.finalized_at && detail.tasks.some((task) => taskPhase(task) === "running")),
      });
    } catch (reason: unknown) {
      // The last good reading is kept: the ledger going away for a poll is
      // not the campaigns ceasing to exist, and blanking the tab on a blip
      // reads as the work having vanished.
      const previous = read.get(sessionId);
      read.set(sessionId, { ...(previous ?? UNREAD), error: message(reason), loading: false });
    } finally {
      reading.delete(sessionId);
      announce();
    }
  })();

  reading.set(sessionId, run);
  return run;
}

/**
 * Read the conversation's campaigns again now. For when the conductor has
 * just changed them: the poll only runs while a campaign is live, so the
 * first campaign of a conversation would otherwise not be seen at all.
 */
export function refreshCampaigns(sessionId: string): void {
  void load(sessionId);
}

/** A 503 is the ledger being away, which is not the same as an empty fleet. */
function message(reason: unknown): string {
  if (reason instanceof ApiError && reason.status === 503) {
    return `The campaign ledger cannot be reached, so what is running is unknown. ${reason.message}`;
  }
  if (reason instanceof ApiError && reason.status === 404) {
    // The route itself is absent, so this gateway is older than campaigns.
    // Left as the bare "Not Found" it reads as "you have no campaigns",
    // which sends the reader looking for a campaign that was never possible.
    return "This gateway does not serve campaigns. It is running a build from before the fleet existed, so no campaign can be started or read until it is updated.";
  }
  return reason instanceof Error ? reason.message : String(reason);
}

/** No conversation is not a conversation whose campaigns have yet to arrive. */
const NO_SESSION: CampaignsRead = { details: NO_CAMPAIGNS, events: NO_EVENTS, error: null, loading: false, live: false };

/**
 * This conversation's campaigns, each with its tasks and results.
 *
 * Never an error where there is no ledger: a tab that showed nothing because
 * the fleet is away would be worse than one that shows what it can.
 */
export function useCampaigns(sessionId: string | null | undefined): CampaignsRead {
  const id = sessionId ?? null;
  const shown = usePanelShown();
  const state = useSyncExternalStore(
    (notify) => {
      watchers.add(notify);
      return () => watchers.delete(notify);
    },
    () => (id === null ? NO_SESSION : (read.get(id) ?? UNREAD)),
  );

  useEffect(() => {
    if (id === null) return;
    void load(id);
  }, [id]);

  useEffect(() => {
    if (id === null || !state.live || !shown) return;
    const timer = window.setInterval(() => {
      if (!document.hidden) void load(id);
    }, POLL_MS);
    return () => window.clearInterval(timer);
  }, [id, state.live, shown]);

  return state;
}

/** This conversation's campaigns, for callers that have nothing to say about a ledger that is away. */
export function useCampaignDetails(sessionId: string | null | undefined): readonly CampaignDetail[] {
  return useCampaigns(sessionId).details;
}

/**
 * The REINVENT runs started by this conversation's campaigns.
 *
 * A sub-agent runs in its own session, so the gateway records it as the
 * launcher and the conversation's own list of runs leaves these out. They
 * are still the conversation's work: without them, asking for a campaign
 * empties the run list of the very runs it set going.
 */
export function useCampaignJobs(sessionId: string | null | undefined): ReadonlySet<string> {
  const details = useCampaignDetails(sessionId);
  return useMemo(() => {
    const ids = new Set<string>();
    for (const detail of details) {
      for (const task of detail.tasks) if (task.job_id) ids.add(task.job_id);
      // A task that started more than one run reports them all; only the
      // one it was harvested from is on the task row.
      for (const result of detail.results) for (const id of result.job_ids ?? []) ids.add(id);
    }
    return ids.size ? ids : NONE;
  }, [details]);
}
