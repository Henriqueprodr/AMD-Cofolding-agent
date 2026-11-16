// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useCallback, useEffect, useMemo, useState, useSyncExternalStore } from "react";
import type { SessionRuns, SessionSummary } from "../types/api";
import { ApiError, deleteSession, listSessions, renameSession } from "../lib/api";

/** How often the list is refreshed, so turns started in other tabs show up. */
const POLL_MS = 15000;

interface SessionsState {
  sessions: SessionSummary[];
  error: string | null;
  refresh: () => void;
  rename: (id: string, title: string) => Promise<void>;
  remove: (id: string) => Promise<void>;
}

/** The last list read, for the parts of the page that only need one conversation's runs; null before the first. */
let latest: SessionSummary[] | null = null;
/** Whether a read has come back at all, so a failed one can be told from one still in flight. */
let probed = false;
let refreshLatest: (() => void) | null = null;
const listeners = new Set<() => void>();

function announce() {
  for (const listener of listeners) listener();
}

function publish(list: SessionSummary[]) {
  latest = list;
  probed = true;
  announce();
}

/** A read came back without a list. Says so, so readers can stop waiting on one. */
function publishFailure() {
  probed = true;
  announce();
}

const subscribe = (listener: () => void) => {
  listeners.add(listener);
  return () => listeners.delete(listener);
};

/** Read the list again now, e.g. when a run may have started. */
export const refreshSessionList = () => refreshLatest?.();

const useLatest = () => useSyncExternalStore(subscribe, () => latest);
const useProbed = () => useSyncExternalStore(subscribe, () => probed);

/** The REINVENT jobs of one conversation, as the last list read said; null for none. */
export function useSessionRuns(sessionId: string | null | undefined): SessionRuns | null {
  const list = useLatest();
  return useMemo(() => (sessionId ? (list?.find((s) => s.id === sessionId)?.runs ?? null) : null), [list, sessionId]);
}

/**
 * Whether the gateway says which conversation launched which run: null until
 * the list is read, false for a gateway too old to (no conversation has `runs`).
 */
export function useRunsKnown(): boolean | null {
  return runsKnown(useLatest(), useProbed());
}

/**
 * Whether the gateway says which conversation launched which run, given the
 * last list read and whether a read has come back at all.
 *
 * A read that came back empty-handed counts as "does not say", not as "not
 * read yet": null holds Summary, Results and Analysis on their loading state
 * for as long as the gateway is unreachable, which reads as three blank panels.
 */
export function runsKnown(list: SessionSummary[] | null, probed: boolean): boolean | null {
  if (list === null) return probed ? false : null;
  return list.length === 0 || list.some((s) => s.runs !== undefined);
}

/** Job id → the conversation that launched it, as the last list read said. */
export function useJobOwners(): ReadonlyMap<string, string> {
  const list = useLatest();
  return useMemo(() => {
    const owners = new Map<string, string>();
    for (const session of list ?? []) for (const job of session.runs?.jobs ?? []) if (job.launched) owners.set(job.id, session.id);
    return owners;
  }, [list]);
}

/** The saved conversations, newest first. */
export function useSessions(): SessionsState {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    listSessions()
      .then((list) => {
        setSessions(list);
        publish(list);
        setError(null);
      })
      .catch((reason: unknown) => {
        setError(reason instanceof Error ? reason.message : String(reason));
        publishFailure();
      });
  }, []);

  useEffect(() => {
    refreshLatest = refresh;
    return () => {
      if (refreshLatest === refresh) refreshLatest = null;
    };
  }, [refresh]);

  useEffect(() => {
    refresh();
    const timer = window.setInterval(() => {
      if (document.visibilityState === "visible") refresh();
    }, POLL_MS);
    const onVisible = () => {
      if (document.visibilityState === "visible") refresh();
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [refresh]);

  const rename = useCallback(
    async (id: string, title: string) => {
      const updated = await renameSession(id, title);
      setSessions((previous) => previous.map((s) => (s.id === id ? { ...s, ...updated } : s)));
    },
    [],
  );

  const remove = useCallback(async (id: string) => {
    // A purge from the data view, or another tab, may have deleted it already.
    await deleteSession(id).catch((reason: unknown) => {
      if (!(reason instanceof ApiError && reason.status === 404)) throw reason;
    });
    setSessions((previous) => previous.filter((s) => s.id !== id));
  }, []);

  return { sessions, error, refresh, rename, remove };
}
