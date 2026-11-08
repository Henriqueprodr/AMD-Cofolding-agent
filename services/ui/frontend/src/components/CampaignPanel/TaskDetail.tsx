// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useMemo, useState } from "react";
import type { Theme } from "../../hooks/useTheme";
import type { CampaignDetail, FleetResult } from "../../types/api";
import { callTool } from "../../lib/api";
import { relativeTime } from "../../lib/campaign";
import { taskPhase, ttpIdOf } from "../../lib/fleet";
import { asTtp, objectivesOf, parseResult, payloadError, sourceLabel, targetRange, targetSummary, type Ttp } from "../../lib/ttp";
import { short } from "../RunAnalysis/plotUtils";
import { Card, Placeholder } from "../RunAnalysis/shared";
import { MoleculeTable, type MoleculeRow } from "../MoleculeTable/MoleculeTable";
import { shortId, useNow } from "./shared";

interface Props {
  detail: CampaignDetail;
  taskId: string;
  theme: Theme;
  onBack: () => void;
  onOpenMolecule?: (smiles: string) => void;
  /** Show the task's REINVENT run in the Jobs tab. */
  onOpenJob: (jobId: string) => void;
}

const PHASE_WORD: Record<ReturnType<typeof taskPhase>, string> = {
  done: "finished",
  running: "running",
  failed: "failed",
  pending: "waiting for an agent",
};

/**
 * One sub-agent's task, in the same terms a single run is read in: what was
 * asked, the molecules it found, and the profile it was scored against.
 *
 * The TTP is shown whole rather than as a difference from the campaign's,
 * because on most campaigns there is nothing to difference against: the task
 * writes its own.
 */
export function TaskDetail({ detail, taskId, theme, onBack, onOpenMolecule, onOpenJob }: Props) {
  const now = useNow();
  const task = detail.tasks.find((t) => t.task_id === taskId) ?? null;
  const result = detail.results.find((r) => r.task_id === taskId) ?? null;
  const ttpId = task ? ttpIdOf(detail.campaign, task.task_id) : null;

  if (!task) {
    return (
      <div className="p-3">
        <BackLink onBack={onBack} />
        <Placeholder height={100}>That task is no longer in the campaign.</Placeholder>
      </div>
    );
  }

  const phase = taskPhase(task);
  const parents = task.spec.derived_from ?? [];

  return (
    <div className="flex flex-col gap-2.5 p-3">
      <BackLink onBack={onBack} />

      <Card
        title="Task"
        actions={
          <span className="text-faint text-[11px]">
            {PHASE_WORD[phase]}
            {task.attempts > 1 && ` · attempt ${task.attempts}`}
          </span>
        }
      >
        <p className="text-[12px] leading-relaxed">{task.spec.intent || <span className="text-faint">The conductor recorded no intent for this task.</span>}</p>
        {task.spec.rationale && <p className="text-muted mt-1.5 text-[11px] leading-relaxed">{task.spec.rationale}</p>}
        {task.spec.weight_overrides && Object.keys(task.spec.weight_overrides).length > 0 && (
          <p className="text-faint mt-1.5 text-[10.5px]">
            weights ·{" "}
            {Object.entries(task.spec.weight_overrides)
              .map(([name, weight]) => `${name} ${weight}`)
              .join(", ")}
          </p>
        )}
        <dl className="text-faint mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[10.5px]">
          <Fact label="task">{task.task_id}</Fact>
          {task.lease_owner && <Fact label="sub-agent">{task.lease_owner}</Fact>}
          {parents.length > 0 && <Fact label="came from">{parents.map(shortId).join(", ")}</Fact>}
          {task.job_id && (
            <Fact label="run">
              <button type="button" onClick={() => onOpenJob(task.job_id!)} className="cursor-pointer underline-offset-2 hover:underline">
                {task.job_id}
              </button>
            </Fact>
          )}
          {task.error && <Fact label="error">{task.error}</Fact>}
        </dl>
      </Card>

      <Molecules result={result} phase={phase} theme={theme} onOpenMolecule={onOpenMolecule} />

      {result && Object.keys(result.objective_profile).length > 0 && (
        <Card title="What it scored" actions={<span className="text-faint text-[11px]">posted {relativeTime(result.posted_at, now)}</span>}>
          <ul className="flex flex-col gap-1">
            {Object.entries(result.objective_profile)
              .sort(([a], [b]) => a.localeCompare(b))
              .map(([name, value]) => (
                <li key={name} className="flex items-baseline gap-2 text-[11.5px]">
                  <span className="min-w-0 flex-1 truncate">{name}</span>
                  <span className="tabular-nums">{short(value)}</span>
                </li>
              ))}
          </ul>
          {result.summary && <p className="text-muted mt-2 text-[11px] leading-relaxed">{result.summary}</p>}
        </Card>
      )}

      {ttpId && <TtpSummaryCard key={ttpId} ttpId={ttpId} authored={!detail.campaign.base_ttp_id} />}
    </div>
  );
}

function BackLink({ onBack }: { onBack: () => void }) {
  return (
    <button type="button" onClick={onBack} className="text-muted hover:text-body -mb-0.5 w-fit cursor-pointer text-[11px]">
      ‹ Back to the campaign
    </button>
  );
}

function Fact({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <>
      <dt>{label}</dt>
      <dd className="text-muted min-w-0 truncate font-mono">{children}</dd>
    </>
  );
}

interface Row extends MoleculeRow {
  smiles: string;
}

function Molecules({
  result,
  phase,
  theme,
  onOpenMolecule,
}: {
  result: FleetResult | null;
  phase: ReturnType<typeof taskPhase>;
  theme: Theme;
  onOpenMolecule?: (smiles: string) => void;
}) {
  const rows = useMemo((): Row[] => {
    const candidates = result?.candidates ?? [];
    return candidates
      .map((candidate) => {
        const score = typeof candidate.score === "number" && Number.isFinite(candidate.score) ? candidate.score : null;
        return { smiles: String(candidate.smiles ?? "").trim(), score };
      })
      .filter((row) => row.smiles)
      .sort((a, b) => (b.score ?? -Infinity) - (a.score ?? -Infinity))
      .map(({ smiles, score }) => ({ smiles, cells: { score: { text: score === null ? "—" : short(score), score } } }));
  }, [result]);

  return (
    <Card title="Molecules" actions={<span className="text-faint text-[11px]">{rows.length || ""}</span>}>
      {rows.length === 0 ? (
        <Placeholder height={90}>
          {phase === "running"
            ? "The run is going; molecules appear when the sub-agent posts its result."
            : phase === "failed"
              ? "The task failed before it posted anything."
              : "Nothing posted for this task."}
        </Placeholder>
      ) : (
        <MoleculeTable
          columns={[{ key: "score", label: "Score", strong: true, colour: "var(--color-ok)" }]}
          rows={rows}
          theme={theme}
          onOpen={onOpenMolecule && ((row) => onOpenMolecule(row.smiles))}
          maxHeight={280}
        />
      )}
    </Card>
  );
}

/**
 * The profile the task was scored against, whole.
 *
 * Read through the tool rather than the ledger: the ledger stores what a
 * task reported, and the TTP is a separate thing the sub-agent wrote.
 */
function TtpSummaryCard({ ttpId, authored }: { ttpId: string; authored: boolean }) {
  const [ttp, setTtp] = useState<Ttp | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Mounted keyed on ttpId, so another task's profile starts from nothing
  // rather than being cleared here.
  useEffect(() => {
    let cancelled = false;
    callTool("get_ttp", { ttp_id: ttpId })
      .then(({ result }) => {
        if (cancelled) return;
        const parsed = parseResult(result);
        const found = asTtp(parsed);
        if (found) setTtp(found);
        else setError(payloadError(result) ?? "That profile could not be read.");
      })
      .catch((reason: unknown) => {
        if (!cancelled) setError(reason instanceof Error ? reason.message : String(reason));
      });
    return () => {
      cancelled = true;
    };
  }, [ttpId]);

  const objectives = useMemo(() => (ttp ? [...objectivesOf(ttp, "pre_scoring"), ...objectivesOf(ttp, "scoring")] : []), [ttp]);

  return (
    <Card title="Profile" actions={<span className="text-faint truncate text-[11px]">{ttpId}</span>}>
      {error ? (
        <p className="text-faint text-[11px] leading-relaxed">
          {authored
            ? // A sub-agent that failed once sometimes writes its retry under
              // another id; the run does not count, but the task is still here.
              `No profile under ${ttpId}. The sub-agent writes its own, and this one either has not yet or wrote it under another id.`
            : error}
        </p>
      ) : !ttp ? (
        <Placeholder height={60}>Reading the profile…</Placeholder>
      ) : (
        <>
          {targetSummary(ttp) && <p className="text-faint mb-1.5 text-[10.5px]">{targetSummary(ttp)}</p>}
          <ul className="flex flex-col gap-1">
            {objectives.map((objective) => (
              <li key={objective.name} className="flex items-baseline gap-2 text-[11.5px]">
                <span className="min-w-0 flex-1 truncate" title={`${sourceLabel(objective)} · ${objective.component ?? ""}`}>
                  {objective.name}
                </span>
                {targetRange(objective.transform) && <span className="text-faint shrink-0 text-[10.5px] tabular-nums">{targetRange(objective.transform)}</span>}
                <span className="text-muted w-10 shrink-0 text-right tabular-nums" title="weight">
                  ×{+objective.weight.toPrecision(3)}
                </span>
              </li>
            ))}
          </ul>
        </>
      )}
    </Card>
  );
}
