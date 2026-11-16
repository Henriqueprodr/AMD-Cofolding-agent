// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { memo, useEffect, useMemo, useRef, useState } from "react";
import type { Artifacts } from "../../lib/artifacts";
import { callTool, jobFileUrl } from "../../lib/api";
import {
  errorSummary,
  isActive,
  jobTitle,
  parseStatus,
  toolError,
  type JobState,
  type JobStatus,
  type JobSummary,
} from "../../lib/jobs";
import {
  LOG_TAIL_BYTES,
  countDown,
  ownJobIds,
  parseStepTimes,
  runEventText,
  runProgress,
  runsToShow,
  RunWatch,
  secondsPerStep,
  shortDuration,
  type RunEvent,
} from "../../lib/runProgress";
import { notifyRunsChanged, onRunsChanged, openRun } from "../../lib/workbenchBus";
import { cofoldShown, inFlightCofold, runUsesBoltz, type CofoldStatus } from "../../lib/cofold";
import { agentLabels, campaignRuns } from "../../lib/fleet";
import { useCampaignDetails } from "../../hooks/useCampaignJobs";
import { useJobList, useJobWatch } from "../../hooks/useJobList";
import { refreshSessionList, useSessionRuns } from "../../hooks/useSessions";
import { Bar, CofoldRow, Eta } from "./CofoldRow";
import { ScoreTrend } from "./ScoreTrend";
import { useCofoldStatus } from "./useCofoldStatus";

interface Props {
  sessionId: string | null;
  /** The conversation's tool calls: the runs it started are followed first. */
  steps: Artifacts["steps"];
  /** When the agent turn in progress started; null when none is. Its open calls may be cofolding. */
  turnAt?: number | null;
  /** A short in-app message when a run ends or raises an alert, in any conversation. */
  onNotice?: (message: string) => void;
}

/** get_optimization_status reads the whole results CSV, so not more often than this. */
const POLL_MS = 20_000;
/**
 * list_jobs while a run is going on, for runs of other conversations. Not
 * while the page is hidden: the list is read again as soon as it is shown,
 * and RunWatch then reports whatever ended meanwhile.
 */
const WATCH_MS = 60_000;
/** How long a finished run stays in the bar. */
const DONE_MS = 8_000;
/** How long a run's finished cofold batch stays while the run prepares the next. */
const HOLD_MS = 180_000;
const ASKED_KEY = "cofold.notify.asked";
/** Before the first reading of the job list: nothing to show and nothing to report. */
const NO_JOBS: JobSummary[] = [];

const watch = new RunWatch();

/** Asks once ever, the first time a run is seen; the browser remembers the answer. */
function askNotifyOnce() {
  try {
    if (typeof Notification === "undefined" || Notification.permission !== "default" || localStorage.getItem(ASKED_KEY)) return;
    localStorage.setItem(ASKED_KEY, "1");
    void Notification.requestPermission().catch(() => undefined);
  } catch {
    // Blocked storage or no Notification API: in-app messages only.
  }
}

function announce(events: RunEvent[], jobs: readonly JobSummary[], onNotice?: (message: string) => void) {
  for (const event of events) {
    const job = jobs.find((j) => j.id === event.jobId);
    const text = runEventText(event, job ? jobTitle(job) : event.jobId);
    onNotice?.(event.kind === "completed" ? text.title : `${text.title}: ${text.body}`);
    if (document.hidden && typeof Notification !== "undefined" && Notification.permission === "granted") {
      try {
        const note = new Notification(text.title, { body: text.body, tag: event.key });
        note.onclick = () => {
          window.focus();
          openRun(event.jobId, event.kind === "completed" || event.kind === "stopped" ? "results" : "jobs");
          note.close();
        };
      } catch {
        // Some browsers only allow notifications from a service worker.
      }
    }
  }
}

interface Live {
  status: JobStatus;
  secondsPerStep: number | null;
  logStep: number | null;
  /** When the status was read, to count the estimate down from. */
  at: number;
}

const errorText = (error: unknown) => (error instanceof Error ? error.message : String(error));

async function readLogTail(jobId: string): Promise<string> {
  try {
    const response = await fetch(jobFileUrl(jobId, "reinvent.log"), { headers: { Range: `bytes=-${LOG_TAIL_BYTES}` } });
    return response.ok ? await response.text() : "";
  } catch {
    return "";
  }
}

async function readLive(jobId: string): Promise<Live | null> {
  const [{ result }, log] = await Promise.all([callTool("get_optimization_status", { job_id: jobId }), readLogTail(jobId)]);
  const status = parseStatus(result);
  if (!status) return null;
  const marks = parseStepTimes(log);
  return { status, secondsPerStep: secondsPerStep(marks), logStep: marks.length ? marks[marks.length - 1].step : null, at: Date.now() };
}

/**
 * A slim bar above the composer while a REINVENT run of this conversation
 * goes on, with its progress and an estimate of the time left, and under it
 * what Boltz cofolds for the conversation. With several going at once it
 * shows the first, and a tab on its top edge opens all of them in a box over
 * the conversation, one line each. A click opens the run in the
 * workbench; Stop asks first, then calls stop_optimization. A finished run
 * stays a few seconds with its outcome.
 */
export const RunBar = memo(function RunBar({ sessionId, steps, turnAt = null, onNotice }: Props) {
  const runs = useSessionRuns(sessionId);
  const campaigns = useCampaignDetails(sessionId);
  const delegated = useMemo(() => campaignRuns(campaigns, agentLabels(campaigns)), [campaigns]);
  const conversation = useMemo(() => ownJobIds(steps, runs, delegated.keys()), [steps, runs, delegated]);
  const { jobs: read, alerts } = useJobList();
  const jobs = read ?? NO_JOBS;
  const [live, setLive] = useState<Record<string, Live>>({});
  const [pollTick, setPollTick] = useState(0);
  const [finished, setFinished] = useState<{ job: JobSummary | JobStatus; at: number } | null>(null);
  const [now, setNow] = useState(() => Date.now());
  const [expanded, setExpanded] = useState(false);
  /** The run whose Stop is waiting to be confirmed. */
  const [confirm, setConfirm] = useState<string | null>(null);
  const [stopping, setStopping] = useState<ReadonlySet<string>>(() => new Set());
  const [stopError, setStopError] = useState<{ jobId: string; text: string } | null>(null);

  const stateOf = (job: JobSummary): JobState => live[job.id]?.status.state ?? job.state;
  const shown = runsToShow(
    jobs.map((job) => ({ ...job, state: stateOf(job) })),
    conversation,
  );
  const shownKey = shown.map((job) => job.id).join(",");

  // Watched only while a run is going on; one that starts announces itself (onRunsChanged).
  const anyRunning = jobs.some((job) => isActive(job.state));
  useJobWatch(anyRunning ? WATCH_MS : null);

  useEffect(() => announce(watch.jobs(jobs), jobs, onNotice), [jobs, onNotice]);
  useEffect(() => {
    if (anyRunning) announce(watch.alertsOf(alerts), jobs, onNotice);
  }, [anyRunning, jobs, alerts, onNotice]);

  useEffect(() => onRunsChanged(refreshSessionList), []);

  useEffect(() => {
    if (!shownKey) return;
    let cancelled = false;
    const ids = shownKey.split(",");
    Promise.all(ids.map((id) => readLive(id).catch(() => null))).then((results) => {
      if (cancelled) return;
      const next: Record<string, Live> = {};
      results.forEach((entry, i) => {
        if (entry) next[ids[i]] = entry;
      });
      setLive((previous) => ({ ...previous, ...next }));
      const statuses = results.filter((entry): entry is Live => entry !== null).map((entry) => entry.status);
      announce(watch.jobs(statuses), statuses, onNotice);
      const done = results.find((entry) => entry && entry.status.state !== "running" && entry.status.state !== "queued");
      if (done) setFinished({ job: done.status, at: Date.now() });
    });
    return () => {
      cancelled = true;
    };
  }, [shownKey, pollTick, onNotice]);

  const active = shown.length > 0;
  useEffect(() => {
    if (active) askNotifyOnce();
  }, [active]);
  useEffect(() => {
    if (!active) return;
    const timer = window.setInterval(() => {
      if (!document.hidden) setPollTick((t) => t + 1);
    }, POLL_MS);
    return () => window.clearInterval(timer);
  }, [active]);

  const ticking = active || finished !== null;
  useEffect(() => {
    if (!ticking) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [ticking]);

  useEffect(() => {
    if (!finished) return;
    const timer = window.setTimeout(() => setFinished(null), Math.max(0, finished.at + DONE_MS - Date.now()));
    return () => window.clearTimeout(timer);
  }, [finished]);

  const cofoldCalls = useMemo(() => inFlightCofold(steps, turnAt), [steps, turnAt]);
  const designing = shown.filter((job) => job.type === "optimization" && job.state === "running");
  const boltzUse = designing.map((job) => runUsesBoltz(steps, job.ttpId, live[job.id]?.status.components));
  const boltzRun = boltzUse.includes(true) ? true : boltzUse.includes(null) ? null : false;
  const cofold = useCofoldStatus(cofoldCalls, boltzRun, designing.length > 0);
  const ownActive = cofold.status?.available === true && cofold.status.active;
  const [held, setHeld] = useState<{ session: string | null; status: CofoldStatus; at: number } | null>(null);
  if (ownActive && cofold.status && (held?.at !== cofold.at || held.session !== sessionId)) setHeld({ session: sessionId, status: cofold.status, at: cofold.at });
  const holding =
    !ownActive && held !== null && held.session === sessionId && designing.length > 0 && boltzRun !== false && cofold.at - held.at < HOLD_MS;
  const cofoldRow = cofoldShown(cofold.status, cofoldCalls) ? (
    <CofoldRow status={cofold.status} at={cofold.at} calls={cofoldCalls} memory={cofold.memory} />
  ) : holding ? (
    <CofoldRow status={held.status} at={held.at} calls={[]} finished />
  ) : null;

  async function stop(job: JobSummary) {
    setConfirm(null);
    setStopping((previous) => new Set(previous).add(job.id));
    setStopError(null);
    try {
      const { result } = await callTool("stop_optimization", { job_id: job.id });
      // The stopped job comes back whole, with whatever its process last wrote as its error.
      const stopped = parseStatus(result)?.state === "stopped";
      const failure = stopped ? null : toolError(result);
      if (failure) setStopError({ jobId: job.id, text: errorSummary(failure) });
    } catch (reason) {
      setStopError({ jobId: job.id, text: errorText(reason) });
    } finally {
      setStopping((previous) => {
        const next = new Set(previous);
        next.delete(job.id);
        return next;
      });
      setPollTick((t) => t + 1);
      notifyRunsChanged();
    }
  }

  if (shown.length) {
    const open = expanded && shown.length > 1;
    const toggle = (next: boolean) => {
      setExpanded(next);
      // A Stop asked of a line that is folded away would be answered unseen.
      setConfirm(null);
    };
    const line = (job: JobSummary, divided: boolean) => (
      <Line key={job.id} divided={divided}>
        <RunLine
          job={job}
          entry={live[job.id]}
          now={now}
          title={delegated.get(job.id)?.title || jobTitle(job)}
          stopping={stopping.has(job.id)}
          error={stopError?.jobId === job.id ? stopError.text : null}
          confirming={confirm === job.id}
          onConfirm={(asking) => setConfirm(asking ? job.id : null)}
          onStop={() => void stop(job)}
        />
      </Line>
    );
    return (
      <Shell extra={cofoldRow} more={shown.length > 1 ? { count: shown.length, open, toggle } : null} list={open ? shown.map((job, i) => line(job, i > 0)) : null}>
        {line(shown[0], false)}
      </Shell>
    );
  }

  if (finished) {
    const job = finished.job;
    const status = job as Partial<JobStatus>;
    const state = job.state;
    const progress = runProgress({ state, currentStep: status.currentStep ?? null, maxSteps: job.maxSteps, startedAt: job.startedAt, finishedAt: job.finishedAt, now });
    const color = state === "completed" ? "var(--color-ok)" : state === "failed" ? "var(--color-accent)" : "var(--color-warn)";
    const label = state === "completed" ? "Finished" : state === "failed" ? "Failed" : "Stopped";
    return (
      <Shell extra={cofoldRow}>
        <Line>
          <button type="button" onClick={() => openRun(job.id, "jobs")} className="col-span-8 grid min-w-0 cursor-pointer grid-cols-subgrid items-center text-left">
            <span aria-hidden="true" className="size-1.5 rounded-full" style={{ background: color }} />
            <span className="font-medium" style={{ color }}>
              {label}
            </span>
            <span className="text-faint min-w-0 truncate">{delegated.get(job.id)?.title || jobTitle(job)}</span>
            <Bar fraction={progress.fraction} colour={color} />
            <span className="text-muted text-right font-mono tabular-nums">
              {progress.done}/{progress.max ?? "—"}
            </span>
            <span />
            <span className="text-faint text-right tabular-nums">
              {progress.elapsedMs !== null ? shortDuration(progress.elapsedMs) : ""}
            </span>
            <span />
          </button>
          <Actions>
            <BarButton tone="primary" onClick={() => openRun(job.id, state === "failed" ? "jobs" : "results")}>
              Job
            </BarButton>
            <BarButton onClick={() => setFinished(null)}>×</BarButton>
          </Actions>
        </Line>
      </Shell>
    );
  }

  return cofoldRow ? (
    <Shell>
      <Line>{cofoldRow}</Line>
    </Shell>
  ) : null;
});

interface RunLineProps {
  job: JobSummary;
  entry: Live | undefined;
  now: number;
  /** The task behind a campaign's run, else the run's own title. */
  title: string;
  stopping: boolean;
  error: string | null;
  confirming: boolean;
  onConfirm: (asking: boolean) => void;
  onStop: () => void;
}

/** One running job: what it is, how far it has got, and its buttons. */
function RunLine({ job, entry, now, title, stopping, error, confirming, onConfirm, onStop }: RunLineProps) {
  const input = {
    state: job.state,
    currentStep: entry?.status.currentStep ?? null,
    logStep: entry?.logStep ?? null,
    maxSteps: entry?.status.maxSteps ?? job.maxSteps,
    startedAt: job.startedAt,
    secondsPerStep: entry?.secondsPerStep ?? null,
  };
  const progress = runProgress({ ...input, now });
  const estimate = entry ? runProgress({ ...input, now: entry.at }) : null;
  const remaining = estimate ? countDown(estimate.remainingMs, now - (entry?.at ?? now)) : null;

  if (confirming) {
    return (
      <>
        <span className="col-span-8 min-w-0 leading-tight whitespace-normal">
          Stop <b className="font-semibold">{title}</b> at step {progress.done}? REINVENT saves a checkpoint, so it can be resumed.
        </span>
        <Actions>
          <BarButton onClick={() => onConfirm(false)}>Cancel</BarButton>
          <BarButton tone="bad" onClick={onStop}>
            Stop run
          </BarButton>
        </Actions>
      </>
    );
  }

  return (
    <>
      <button type="button" onClick={() => openRun(job.id, "jobs")} className="col-span-8 grid min-w-0 cursor-pointer grid-cols-subgrid items-center text-left">
        <MoleculeStepProgress job={job} />
        <span className="font-medium">{job.state === "queued" ? "Starting" : stopping ? "Stopping" : ""}</span>
        <span className="text-faint min-w-0 truncate">{error ? <span style={{ color: "var(--color-accent)" }}>{error}</span> : title}</span>
        <Bar fraction={progress.fraction ?? 0} />
        <span className="text-muted text-right font-mono tabular-nums">
          {progress.done}/{progress.max ?? "—"}
        </span>
        <span>
          <ScoreTrend status={entry?.status} width={56} className="hidden @lg:block" />
        </span>
        <span />
        <span className="text-faint tabular-nums">{job.state === "queued" ? "" : <Eta remainingMs={remaining} />}</span>
      </button>
      <Actions>
        <BarButton tone="primary" onClick={() => openRun(job.id, "jobs")}>
          Job
        </BarButton>
        {job.type === "optimization" && job.state === "running" && (
          <BarButton tone="bad" disabled={stopping} onClick={() => onConfirm(true)}>
            <StopIcon /> Stop
          </BarButton>
        )}
      </Actions>
    </>
  );
}

/** Compact progress for the Boltz batch currently blocking this RL step. */
function MoleculeStepProgress({ job }: { job: JobSummary }) {
  const done = job.moleculesCompleted;
  const total = job.moleculesExpected;
  if (done == null || total == null || total <= 0) {
    return <span aria-hidden="true" className="size-1.5 animate-pulse rounded-full" style={{ background: "var(--color-accent)" }} />;
  }
  const fraction = Math.max(0, Math.min(1, done / total));
  const label = `${done}/${total} cofolded in this step`;
  return (
    <span
      role="img"
      aria-label={label}
      title={label}
      className="grid size-5 shrink-0 place-items-center rounded-full"
      style={{ background: `conic-gradient(var(--color-accent) ${fraction * 360}deg, var(--border-subtle) 0deg)` }}
    >
      <span className="bg-sunken grid size-3.5 place-items-center rounded-full text-[7px] font-semibold tabular-nums">{done}</span>
    </span>
  );
}

const GRID = "grid grid-cols-[auto_auto_minmax(0,1fr)_auto_auto_auto_auto_auto_auto] gap-x-1.5 @lg:gap-x-2 text-[11px] whitespace-nowrap";

/**
 * The bar; *extra* is a line under the runs, e.g. what Boltz is cofolding.
 * With more than one run, a tab on its top edge opens *list*, every run, in
 * a box over the conversation that closes on a click anywhere else.
 */
function Shell({
  children,
  extra,
  more = null,
  list = null,
}: {
  children: React.ReactNode;
  extra?: React.ReactNode;
  more?: { count: number; open: boolean; toggle: (open: boolean) => void } | null;
  list?: React.ReactNode;
}) {
  const box = useRef<HTMLDivElement>(null);
  const close = more?.toggle;
  useEffect(() => {
    if (!list || !close) return;
    const outside = (event: PointerEvent) => {
      if (!box.current?.contains(event.target as Node)) close(false);
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape") close(false);
    };
    document.addEventListener("pointerdown", outside);
    document.addEventListener("keydown", escape);
    return () => {
      document.removeEventListener("pointerdown", outside);
      document.removeEventListener("keydown", escape);
    };
  }, [list, close]);

  return (
    <div className="@container mx-auto w-full max-w-3xl px-3 sm:px-4" role="status" aria-live="polite">
      <div ref={box} className="relative">
        {list && (
          <div
            aria-label="Every run going on"
            className={`bg-raised border-subtle shadow-card absolute inset-x-0 bottom-full z-20 mb-3 max-h-64 overflow-y-auto rounded-lg border ${GRID}`}
          >
            {list}
          </div>
        )}
        {more && (
          <button
            type="button"
            onClick={() => more.toggle(!more.open)}
            aria-expanded={more.open}
            aria-label={more.open ? "Hide the list of runs" : `Show all ${more.count} runs`}
            title={`${more.count} runs`}
            className="bg-sunken border-subtle text-muted hover:text-body absolute -top-2 left-1/2 z-10 -translate-x-1/2 cursor-pointer rounded-md border px-1.5 text-[10px] leading-3 tracking-widest"
          >
            …
          </button>
        )}
        <div className={`bg-sunken border-subtle rounded-t-lg border border-b-0 ${GRID}`}>
          {children}
          {extra && <Line divided>{extra}</Line>}
        </div>
      </div>
    </div>
  );
}

/** Every line shares the bar's columns: dot, label, detail, progress, count, extra, elapsed, time left, actions. */
function Line({ children, divided = false }: { children: React.ReactNode; divided?: boolean }) {
  return <div className={`col-span-full grid min-h-7 grid-cols-subgrid items-center py-0.5 pr-1 pl-2.5${divided ? " border-subtle border-t" : ""}`}>{children}</div>;
}

function Actions({ children }: { children: React.ReactNode }) {
  return <span className="col-start-9 -my-0.5 flex items-center justify-end gap-0.5">{children}</span>;
}

function BarButton({
  children,
  onClick,
  disabled,
  tone,
  expanded,
}: {
  children: React.ReactNode;
  onClick: () => void;
  disabled?: boolean;
  tone?: "bad" | "primary";
  /** For a button that shows and hides more of the bar. */
  expanded?: boolean;
}) {
  const look =
    tone === "bad"
      ? "text-accent hover:bg-accent-soft"
      : tone === "primary"
        ? "bg-raised border-subtle text-body hover:bg-inset border shadow-sm"
        : "text-muted hover:text-body hover:bg-inset";
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-expanded={expanded}
      className={`inline-flex shrink-0 cursor-pointer items-center gap-1 rounded-md px-2 py-0.5 font-medium disabled:cursor-default disabled:opacity-40 ${look}`}
    >
      {children}
    </button>
  );
}

function StopIcon() {
  return (
    <svg width={8} height={8} viewBox="0 0 10 10" aria-hidden="true">
      <rect x={1} y={1} width={8} height={8} rx={1.5} fill="currentColor" />
    </svg>
  );
}
