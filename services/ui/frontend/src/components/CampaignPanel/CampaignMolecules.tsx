// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useMemo, useState } from "react";
import type { Theme } from "../../hooks/useTheme";
import { useCampaigns } from "../../hooks/useCampaignJobs";
import { agentLabels, pooledCandidates, taskOwners, type PooledCandidate } from "../../lib/fleet";
import { loadJobStructures, openRunStructure, runStructurePanel, type JobStructure } from "../../lib/jobStructures";
import { short } from "../RunAnalysis/plotUtils";
import { Card, Placeholder } from "../RunAnalysis/shared";
import { MoleculeTable, type MoleculeRow } from "../MoleculeTable/MoleculeTable";
import { shortId, useAgentPage } from "./shared";

interface Props {
  sessionId: string | null;
  theme: Theme;
  onOpenMolecule?: (smiles: string) => void;
}

interface Row extends MoleculeRow {
  pooled: PooledCandidate;
}

const SHOWN = 25;

/**
 * The overview's Molecules: every molecule the sub-agents posted back, best
 * first, pooled across the conversation's campaigns; on a sub-agent's page,
 * only the ones of the task it is on, or of the task picked in the trajectory.
 *
 * Pooled because this is the output the chemist came for, not one list per
 * agent; which agent found a molecule rides along.
 */
export function CampaignMolecules({ sessionId, theme, onOpenMolecule }: Props) {
  const { details, loading } = useCampaigns(sessionId);
  const [all, setAll] = useState(false);
  const labels = useMemo(() => agentLabels(details), [details]);
  const { agent, page } = useAgentPage(details, labels);
  const { tasks, results } = useMemo(() => {
    const tasks = page ? page.held.map(({ task }) => task) : details.flatMap((detail) => detail.tasks);
    const ids = new Set(tasks.map((task) => task.task_id));
    return { tasks, results: details.flatMap((detail) => detail.results).filter((result) => ids.has(result.task_id)) };
  }, [details, page]);
  const finished = results.length;
  const pooled = useMemo(() => pooledCandidates(results, taskOwners(tasks)), [results, tasks]);
  const shown = useMemo(() => pooled.slice(0, all ? pooled.length : SHOWN), [pooled, all]);
  const complexes = useRunStructures(shown);

  const rows = useMemo(
    (): Row[] =>
      shown.map((pooled) => {
        const complex = pooled.jobId ? complexes.get(pooled.jobId)?.get(pooled.smiles) : undefined;
        return {
          pooled,
          smiles: pooled.smiles,
          complex: complex && `Show the Boltz complex from run ${pooled.jobId}${complex.step !== null ? `, step ${complex.step}` : ""} in the viewer`,
          cells: {
            score: { text: pooled.score === null ? "—" : short(pooled.score), score: pooled.score },
            from: {
              text: pooled.owner ? (labels.get(pooled.owner) ?? shortId(pooled.owner)) : shortId(pooled.taskId),
              sub: pooled.owner ? shortId(pooled.taskId) : undefined,
              title: `Task ${pooled.taskId}${pooled.owner ? `, run by ${pooled.owner}` : ""}`,
            },
          },
        };
      }),
    [shown, complexes, labels],
  );
  const openComplex = (row: Row) => {
    const { jobId, smiles } = row.pooled;
    const entry = jobId ? complexes.get(jobId)?.get(smiles) : undefined;
    const panel = entry && jobId && runStructurePanel(entry.path, smiles, jobId, entry.step, entry.metrics);
    if (panel) openRunStructure(sessionId ?? "new", panel);
  };

  return (
    <div className="min-h-0 flex-1 overflow-y-auto p-3">
      <Card
        title="Best molecules"
        actions={
          <span className="text-faint text-[11px]">
            {pooled.length} from {finished} finished {finished === 1 ? "task" : "tasks"}
          </span>
        }
      >
        {pooled.length === 0 ? (
          <Placeholder height={100}>
            {loading
              ? "Reading the campaigns…"
              : page && !page.held.length
                ? `${agent} is waiting for its next task. What it posted before is under its bubbles in the trajectory.`
                : "Nothing posted yet. Molecules appear as sub-agents finish their runs."}
          </Placeholder>
        ) : (
          <MoleculeTable<Row>
            columns={[
              { key: "score", label: "Score", strong: true, colour: "var(--color-ok)" },
              { key: "from", label: "Found by", text: true, muted: true },
            ]}
            rows={rows}
            theme={theme}
            onOpen={onOpenMolecule && ((row) => onOpenMolecule(row.smiles))}
            onOpenStructure={openComplex}
            maxHeight={520}
            footer={
              pooled.length > SHOWN && (
                <button type="button" onClick={() => setAll((v) => !v)} className="text-muted hover:text-body w-full cursor-pointer py-1.5 text-[11px]">
                  {all ? "Show the top 25" : `Show all ${pooled.length}`}
                </button>
              )
            }
          />
        )}
      </Card>
    </div>
  );
}

/**
 * The complexes of the runs behind *shown*, by run then SMILES. Asked for
 * once per run on screen, not per molecule: a result is posted when its run
 * is done, so what the run cofolded does not change after.
 */
function useRunStructures(shown: readonly PooledCandidate[]): ReadonlyMap<string, ReadonlyMap<string, JobStructure>> {
  const [loaded, setLoaded] = useState<ReadonlyMap<string, ReadonlyMap<string, JobStructure>>>(new Map());
  const runs = useMemo(() => [...new Set(shown.flatMap((one) => (one.jobId ? [one.jobId] : [])))].sort().join(" "), [shown]);
  useEffect(() => {
    let cancelled = false;
    for (const id of runs ? runs.split(" ") : []) {
      loadJobStructures(id)
        .then((found) => {
          if (!cancelled && found.size) setLoaded((was) => (was.get(id) === found ? was : new Map(was).set(id, found)));
        })
        // A run cleaned up since, or one that cofolded nothing: no button.
        .catch(() => undefined);
    }
    return () => {
      cancelled = true;
    };
  }, [runs]);
  return loaded;
}
