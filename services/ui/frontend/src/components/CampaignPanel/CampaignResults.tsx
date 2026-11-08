// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useMemo, useState } from "react";
import type { Theme } from "../../hooks/useTheme";
import type { CampaignDetail } from "../../types/api";
import { agentSummaries, objectiveProgress, pooledCandidates, taskOwners, type PooledCandidate } from "../../lib/fleet";
import { short } from "../RunAnalysis/plotUtils";
import { Card, Placeholder } from "../RunAnalysis/shared";
import { MoleculeTable, type MoleculeRow } from "../MoleculeTable/MoleculeTable";
import { shortId } from "./shared";

interface Props {
  detail: CampaignDetail;
  theme: Theme;
  onOpenMolecule?: (smiles: string) => void;
  /** Open one sub-agent's task in the drill-down. */
  onOpenTask: (taskId: string) => void;
}

interface Row extends MoleculeRow {
  pooled: PooledCandidate;
}

const SHOWN = 25;

/**
 * What the campaign has produced, in the order a chemist asks for it: the
 * molecules first, pooled across every sub-agent, then who found what, then
 * how the objectives are moving.
 *
 * The pooling is the point. Which agent ran which task is useful for reading
 * the search, but the output of the campaign is one list of molecules, not
 * one list per agent.
 */
export function CampaignResults({ detail, theme, onOpenMolecule, onOpenTask }: Props) {
  const [all, setAll] = useState(false);
  const owners = useMemo(() => taskOwners(detail.tasks), [detail.tasks]);
  const pooled = useMemo(() => pooledCandidates(detail.results, owners), [detail.results, owners]);
  const agents = useMemo(() => agentSummaries(detail), [detail]);
  const objectives = useMemo(
    () => objectiveProgress(detail.results, detail.campaign.objectives),
    [detail.results, detail.campaign.objectives],
  );

  const rows = useMemo(
    (): Row[] =>
      pooled.slice(0, all ? pooled.length : SHOWN).map((pooled) => ({
        pooled,
        smiles: pooled.smiles,
        cells: {
          score: { text: pooled.score === null ? "—" : short(pooled.score), score: pooled.score },
          from: {
            text: shortId(pooled.taskId),
            sub: pooled.owner ? shortId(pooled.owner) : undefined,
            title: `Task ${pooled.taskId}${pooled.owner ? `, run by ${pooled.owner}` : ""}`,
          },
        },
      })),
    [pooled, all],
  );

  return (
    <div className="flex flex-col gap-2.5 p-3">
      <Card
        title="Best molecules"
        actions={
          <span className="text-faint text-[11px]">
            {pooled.length} from {detail.results.length} finished {detail.results.length === 1 ? "task" : "tasks"}
          </span>
        }
      >
        {pooled.length === 0 ? (
          <Placeholder height={100}>Nothing posted yet. Molecules appear as sub-agents finish their runs.</Placeholder>
        ) : (
          <MoleculeTable
            columns={[
              { key: "score", label: "Score", strong: true, colour: "var(--color-ok)" },
              { key: "from", label: "Found by", text: true, muted: true },
            ]}
            rows={rows}
            theme={theme}
            onOpen={onOpenMolecule && ((row) => onOpenMolecule(row.smiles))}
            maxHeight={320}
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

      <Card title="Sub-agents">
        {agents.length === 0 ? (
          <Placeholder height={70}>No sub-agent has reported yet.</Placeholder>
        ) : (
          <ul className="flex flex-col gap-px">
            {agents.map((agent) => (
              <li key={agent.owner} className="flex items-baseline gap-2 py-1 text-[11.5px]">
                <span
                  className="mt-1 inline-block size-1.5 shrink-0 self-start rounded-full"
                  style={{ background: agent.live ? "var(--color-ok)" : "var(--text-faint)" }}
                  title={agent.live ? "reporting" : "no longer reporting"}
                />
                <span className="min-w-0 flex-1 truncate" title={agent.owner}>
                  {shortId(agent.owner)}
                  {agent.current && (
                    <button
                      type="button"
                      onClick={() => onOpenTask(agent.current!)}
                      className="text-faint hover:text-body ml-1.5 cursor-pointer underline-offset-2 hover:underline"
                    >
                      on {shortId(agent.current)}
                    </button>
                  )}
                </span>
                <span className="text-muted shrink-0 tabular-nums">
                  {agent.completed} done
                  {agent.failed > 0 && <span className="text-faint">, {agent.failed} failed</span>}
                </span>
                <span className="text-faint w-20 shrink-0 text-right tabular-nums" title="Molecules from its tasks, and its best score">
                  {agent.molecules} mol
                  {agent.best !== null && ` · ${short(agent.best)}`}
                </span>
              </li>
            ))}
          </ul>
        )}
      </Card>

      <Card
        title="Objectives"
        actions={
          detail.campaign.objectives.length === 0 && (
            <span className="text-faint text-[11px]" title="This campaign declares no fixed objectives; each task writes its own profile">
              as each task scored it
            </span>
          )
        }
      >
        {objectives.length === 0 ? (
          <Placeholder height={70}>No scores yet. Objectives appear as tasks post their results.</Placeholder>
        ) : (
          <ul className="flex flex-col gap-1.5">
            {objectives.map((objective) => (
              <li key={objective.name}>
                <div className="flex items-baseline gap-2 text-[11.5px]">
                  <span className="min-w-0 flex-1 truncate" title={objective.name}>
                    {objective.name}
                  </span>
                  {objective.delta !== null && (
                    <span className="text-faint shrink-0 tabular-nums" title="How much the newest best improved on the one before it">
                      +{short(objective.delta)}
                    </span>
                  )}
                  <button
                    type="button"
                    onClick={() => onOpenTask(objective.bestTask)}
                    title={`Best so far, by task ${objective.bestTask}`}
                    className="shrink-0 cursor-pointer tabular-nums underline-offset-2 hover:underline"
                  >
                    {short(objective.best)}
                  </button>
                </div>
                {/* Mean behind best, so a single lucky task is visibly not the
                    campaign; both on the same 0–1 scale the profiles use. */}
                <div className="bg-inset relative mt-1 h-1 overflow-hidden rounded-full">
                  <span className="absolute inset-y-0 left-0 rounded-full" style={{ width: `${bar(objective.mean)}%`, background: "var(--color-ok)", opacity: 0.3 }} />
                  <span className="absolute inset-y-0 left-0 w-px" style={{ left: `${bar(objective.best)}%`, background: "var(--color-ok)" }} />
                </div>
                <p className="text-faint mt-0.5 text-[10px]">
                  mean {short(objective.mean)} over {objective.tasks} {objective.tasks === 1 ? "task" : "tasks"}
                </p>
              </li>
            ))}
          </ul>
        )}
      </Card>
    </div>
  );
}

/** A profile score as a percentage of the bar, clamped: profiles are 0–1 but nothing enforces it. */
const bar = (value: number) => Math.max(0, Math.min(100, value * 100));
