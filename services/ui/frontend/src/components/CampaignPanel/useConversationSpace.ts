// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useMemo, useState } from "react";
import { jobFileUrl } from "../../lib/api";
import { mergeRuns, runMolecules, type ConversationSpace } from "../../lib/chemicalSpace";
import { CONDUCTOR, type PooledCandidate } from "../../lib/fleet";
import { isActive, resultsFile, type JobSummary } from "../../lib/jobs";
import { moleculesOf, TOTAL, type Molecules } from "../../lib/runAnalysis";
import { parseRunResults } from "../../lib/runResults";

/** A live run's molecules are read again after this long. */
const REFRESH_MS = 60_000;

type Read = Pick<Molecules, "n" | "smiles" | "values" | "lastStep" | "step">;

interface Entry {
  /** Null when the results could not be read; tried again later. */
  molecules: Read | null;
  /** Finished and read, so never read again. */
  final: boolean;
  at: number;
}

/**
 * Per run, for the whole page: a results file is megabytes, and coming back
 * to the tab, or to the conversation, must not download them all again.
 */
const cache = new Map<string, Entry>();
const reading = new Set<string>();

async function readRun(job: JobSummary): Promise<Read | null> {
  try {
    const response = await fetch(jobFileUrl(job.id, resultsFile(job)));
    if (!response.ok) return null;
    const molecules = moleculesOf(parseRunResults(await response.text()));
    // The map needs the molecules, their totals and their steps; the rest of
    // a run's table would only be held for nothing.
    const total = molecules.values.get(TOTAL);
    return { n: molecules.n, smiles: molecules.smiles, values: total ? new Map([[TOTAL, total]]) : new Map(), lastStep: molecules.lastStep, step: molecules.step };
  } catch {
    return null;
  }
}

/**
 * The molecules of every run in *runs*, pooled for the conversation's map,
 * each run's under the agent *agents* gives it (the conductor's by default).
 */
export function useConversationSpace(
  runs: readonly JobSummary[],
  agents: ReadonlyMap<string, string>,
): { space: ConversationSpace | null; loading: boolean } {
  const read = useRunReads(runs);

  const space = useMemo(() => {
    const started = (job: JobSummary) => Date.parse(job.startedAt || job.createdAt || "") || 0;
    const parts = [...runs]
      .sort((a, b) => started(a) - started(b))
      .flatMap((job) => {
        const molecules = read.get(job.id)?.molecules;
        return molecules ? [runMolecules(job.id, agents.get(job.id) ?? CONDUCTOR, molecules)] : [];
      });
    return parts.length ? mergeRuns(parts) : null;
  }, [runs, agents, read]);

  const loading = runs.some((job) => job.state !== "queued" && !read.has(job.id));
  return { space, loading };
}

/** Each run's results file, read once and held for the page. */
function useRunReads(runs: readonly JobSummary[]): ReadonlyMap<string, Entry> {
  const [read, setRead] = useState<ReadonlyMap<string, Entry>>(() => new Map(cache));

  useEffect(() => {
    for (const job of runs) {
      if (job.state === "queued" || reading.has(job.id)) continue;
      const seen = cache.get(job.id);
      if (seen && (seen.final || Date.now() - seen.at < REFRESH_MS)) continue;
      reading.add(job.id);
      // Not cancelled when the list changes under it, which it does every
      // few seconds: a large file over a slow link would never finish.
      void readRun(job).then((molecules) => {
        cache.set(job.id, { molecules, final: molecules !== null && !isActive(job.state), at: Date.now() });
        reading.delete(job.id);
        setRead(new Map(cache));
      });
    }
  }, [runs]);

  return read;
}

/**
 * The best molecules of every run of the conversation, in the shape the
 * pooled list takes.
 *
 * A campaign's molecules were only ever what a sub-agent posted back, so a
 * conversation working alone had an empty Molecules view however many runs
 * it had finished, and one with sub-agents left out everything the conductor
 * ran itself. Read from the same results files the map is drawn from, so it
 * costs nothing beyond what the Analysis view already fetched.
 */
export function useRunCandidates(
  runs: readonly JobSummary[],
  agents: ReadonlyMap<string, string>,
  perRun = 50,
): PooledCandidate[] {
  const read = useRunReads(runs);
  return useMemo(() => {
    const found: PooledCandidate[] = [];
    for (const job of runs) {
      const molecules = read.get(job.id)?.molecules;
      const totals = molecules?.values.get(TOTAL);
      if (!molecules || !totals) continue;
      const best = [...molecules.smiles.keys()]
        .filter((i) => Number.isFinite(totals[i]))
        .sort((a, b) => totals[b] - totals[a])
        .slice(0, perRun);
      for (const i of best) {
        found.push({
          smiles: molecules.smiles[i],
          score: totals[i],
          // Not a task's: this is the run itself, which the pooled list
          // labels by the agent that started it.
          taskId: job.id,
          owner: agents.get(job.id) ?? CONDUCTOR,
          jobId: job.id,
          candidate: { smiles: molecules.smiles[i], score: totals[i] },
        });
      }
    }
    return found;
  }, [runs, agents, read, perRun]);
}
