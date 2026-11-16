// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useState } from "react";
import type { CampaignAtlas } from "../types/api";
import { fetchCampaignAtlas } from "../lib/api";
import { usePanelShown } from "./panelShownContext";

export interface AtlasRead {
  atlas: CampaignAtlas | null;
  error: string | null;
  /** Read again now: a failed read is otherwise retried only once something new is learnt. */
  retry: () => void;
}

/**
 * What one campaign knows, read again whenever `learnt` moves.
 *
 * No timer of its own: the campaign's history is polled already, and its
 * count of findings and notes (lib/atlas `learnt`) says when there is
 * something new to read. A panel that is hidden reads nothing and catches
 * up when it is shown.
 */
export function useCampaignAtlas(
  campaignId: string,
  learnt: number,
): AtlasRead {
  const shown = usePanelShown();
  const [read, setRead] = useState<{
    atlas: CampaignAtlas | null;
    error: string | null;
    key: string | null;
  }>({ atlas: null, error: null, key: null });
  const [attempt, setAttempt] = useState(0);
  const key = `${campaignId}:${learnt}:${attempt}`;

  useEffect(() => {
    if (!shown || read.key === key) return;
    let cancelled = false;
    fetchCampaignAtlas(campaignId)
      .then((atlas) => !cancelled && setRead({ atlas, error: null, key }))
      .catch((reason: unknown) => {
        // The last reading stays: the ledger away for a poll is not the campaign forgetting.
        if (!cancelled)
          setRead((previous) => ({
            ...previous,
            error: reason instanceof Error ? reason.message : String(reason),
            key,
          }));
      });
    return () => {
      cancelled = true;
    };
  }, [campaignId, key, shown, read.key]);

  // Another campaign's reading is not this one's.
  const mine = read.atlas?.campaign_id === campaignId ? read.atlas : null;
  return {
    atlas: mine,
    error: read.error,
    retry: () => setAttempt((n) => n + 1),
  };
}
