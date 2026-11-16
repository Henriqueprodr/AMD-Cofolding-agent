// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useMemo, useSyncExternalStore } from "react";
import { callTool } from "../lib/api";
import {
  parseAlerts,
  parseJobs,
  toolError,
  type JobAlert,
  type JobSummary,
} from "../lib/jobs";
import { onRunsChanged } from "../lib/workbenchBus";

const NO_ALERTS: readonly JobAlert[] = [];

/** One reading of the jobs and the monitor's alerts. */
export interface JobsRead {
  /** Every job the gateway knows, newest first; null until the first reading. */
  jobs: JobSummary[] | null;
  /** The monitor's alerts, for the run they name. */
  alerts: readonly JobAlert[];
  /** Why the list could not be read; the last good reading is kept beside it. */
  error: string | null;
}

/**
 * One reading of the job list, shared by everything that shows a run.
 *
 * Module-level rather than per hook because the run bar, the Jobs tab, the
 * results, the analysis, the picker and four of the campaign views all want
 * the same list at once, and held per mount that was six timers and six
 * parses of the same answer. Here it is one of each, and a panel that is
 * reopened sees what was read last while the refresh happens behind it.
 */
let read: JobsRead = { jobs: null, alerts: NO_ALERTS, error: null };
let reading: Promise<void> | null = null;
const watchers = new Set<() => void>();

/**
 * What one answer makes of the last reading.
 *
 * A gateway that answers with an error is not a gateway with no jobs, so the
 * last good list is kept beside the error: blanking the run out from under
 * the reader on one bad poll reads as the run having vanished.
 */
export function nextRead(
  previous: JobsRead,
  list: unknown,
  alerted: unknown | null,
): JobsRead {
  const failure = toolError(list);
  return {
    jobs: failure ? previous.jobs : parseJobs(list),
    alerts: alerted === null ? previous.alerts : parseAlerts(alerted),
    error: failure,
  };
}

async function load(): Promise<void> {
  if (reading) return reading;
  const run = (async () => {
    try {
      const [list, alerted] = await Promise.all([
        callTool("list_jobs", {}),
        callTool("get_job_alerts", {}).catch(() => null),
      ]);
      read = nextRead(read, list.result, alerted ? alerted.result : null);
    } catch (reason: unknown) {
      read = {
        ...read,
        error: reason instanceof Error ? reason.message : String(reason),
      };
    } finally {
      reading = null;
      for (const watcher of watchers) watcher();
    }
  })();
  reading = run;
  return run;
}

/** Read the jobs again now, e.g. after a call that may have started one. */
export const refreshJobs = (): void => void load();

/*
 * One timer for all of them, at the shortest period anybody asked for: the
 * run bar wants a slow watch for runs of other conversations, an open panel
 * wants to keep up with the one it shows.
 */
const asked = new Map<object, number>();
let timer: number | null = null;
let period = 0;

/** The period the one timer runs at; 0 for no timer at all. */
export const shortest = (periods: Iterable<number>): number => {
  const all = [...periods];
  return all.length ? Math.min(...all) : 0;
};

function retime(): void {
  const next = shortest(asked.values());
  if (next === period) return;
  period = next;
  if (timer !== null) window.clearInterval(timer);
  timer = period
    ? window.setInterval(() => void (document.hidden || load()), period)
    : null;
}

/**
 * Ask for the list to be read again every `everyMs`, or null for not at all:
 * a view that only wants the news somebody else's timer already brings. The
 * run bar settles this from the list itself, so it is its own hook.
 */
export function useJobWatch(everyMs: number | null): void {
  const key = useMemo(() => ({}), []);
  useEffect(() => {
    if (everyMs === null) return;
    asked.set(key, everyMs);
    retime();
    return () => {
      asked.delete(key);
      retime();
    };
  }, [key, everyMs]);
}

/**
 * The jobs and the alerts, read on mount, whenever a run changes, when the
 * page is shown again, and every `everyMs` while the caller wants it.
 */
export function useJobList(everyMs: number | null = null): JobsRead {
  const state = useSyncExternalStore(
    (notify) => {
      watchers.add(notify);
      return () => watchers.delete(notify);
    },
    () => read,
  );

  useEffect(() => {
    void load();
    return onRunsChanged(() => void load());
  }, []);

  useEffect(() => {
    const onVisible = () => void (document.hidden || load());
    document.addEventListener("visibilitychange", onVisible);
    return () => document.removeEventListener("visibilitychange", onVisible);
  }, []);

  useJobWatch(everyMs);
  return state;
}
