// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * The contract with the agent gateway.
 *
 * Server side: reinvent4_agent/web/server.py (endpoints) and
 * reinvent4_agent/web/events.py (event shapes). Keep the two in step.
 */

// ---------------------------------------------------------------------------
// panels — the four kinds of data one prompt can produce
// ---------------------------------------------------------------------------

/** A protein-ligand complex written by Boltz, shown in the 3D viewer. */
export interface StructurePanel {
  kind: "structure";
  tool: string;
  /** File name, for the card heading. */
  name: string;
  /** Path relative to the shared volume; also the panel identity. */
  path: string;
  /** Where to fetch the file from. */
  url: string;
  format: "cif" | "pdb";
  /** The ligand in this complex, when the tool result says which it is. */
  smiles?: string;
  label?: string | null;
  metrics?: StructureMetrics;
}

/** Boltz and PoseBusters values of one predicted complex. */
export interface StructureMetrics {
  confidence?: number;
  ligandIptm?: number;
  iptm?: number;
  plddt?: number;
  /** log10(IC50 / 1 µM); lower binds tighter. */
  affinity?: number;
  affinity1?: number;
  affinity2?: number;
  binderProbability?: number;
  posebustersPassed?: number;
  posebustersTotal?: number;
  failedChecks?: string[];
}

/** One predicted endpoint: an ensemble mean, its spread and its unit. */
export interface PropertyValue {
  mean: number;
  std?: number;
  unit?: string;
  condition?: string;
}

/** ADMET or RetroDMTA predictions, endpoints by molecule. */
export interface PropertiesPanel {
  kind: "properties";
  tool: string;
  service: string | null;
  target: string | null;
  endpoints: string[];
  molecules: {
    smiles: string | null;
    standardized: string | null;
    label: string | null;
    ok: boolean;
    error: string | null;
    /** Inside the models' applicability domain. */
    adPass: boolean | null;
    values: Record<string, PropertyValue>;
  }[];
}

/** One protein-ligand contact from the interaction profiler. */
export interface Contact {
  type: string;
  subtype: string | null;
  chain: string | null;
  resname: string;
  resnum: number | null;
  /** "ASN140". */
  label: string;
  receptorAtom: string | null;
  ligandAtom: string | null;
  /** 0-based index into the ligand's atoms, -1 or null when not attributable. */
  ligandAtomIndex: number | null;
  distance: number | null;
  fragment: string | null;
  confidence: unknown;
}

/** The contacts of one complex. */
export interface InteractionsPanel {
  kind: "interactions";
  tool: string;
  name: string | null;
  /** The structure these contacts were measured on, as a structure panel path. */
  structurePath: string | null;
  smiles: string | null;
  error: string | null;
  counts: Record<string, number>;
  contacts: Contact[];
  totalContacts: number;
}

/** A ligand as SMILES, drawn as a 2D graph. */
export interface MoleculePanel {
  kind: "molecule";
  tool: string;
  smiles: string;
  label: string | null;
}

/** Predictions and measurements per molecule. */
export interface TablePanel {
  kind: "table";
  tool: string;
  columns: string[];
  rows: Record<string, string | number | boolean | null>[];
  totalRows: number;
  truncated: boolean;
}

export type Panel = StructurePanel | MoleculePanel | TablePanel | PropertiesPanel | InteractionsPanel;

// ---------------------------------------------------------------------------
// events — what the gateway pushes down the WebSocket
// ---------------------------------------------------------------------------

/**
 * Fields the gateway adds to every event it records in a session. Events
 * that concern one browser only (ready, session, notice) have neither.
 */
export interface Recorded {
  seq: number;
  /** ISO 8601, UTC. */
  ts: string;
}

/** Sent first on every connection; not part of the conversation. */
export interface ReadyEvent {
  type: "ready";
  model: string;
  backend: string;
  reasoningEffort: string | null;
  tools: string[];
}

/** Which conversation this socket is attached to. Sent after ready. */
export interface SessionEvent extends SessionSummary {
  type: "session";
  /** Events up to this number are replayed right after this one. */
  lastSeq: number;
}

/** A message for this browser only, such as a refused second question. */
export interface NoticeEvent {
  type: "notice";
  message: string;
}

/**
 * The autonomy budget changed. Like a notice it is not recorded, so a
 * browser that reconnects reads the deadline from the session summary.
 */
export interface AutoEvent {
  type: "auto";
  /** ISO 8601, UTC; null when the agent no longer works on its own. */
  until: string | null;
}

export interface UserMessageEvent extends Recorded {
  type: "user_message";
  content: string;
  effort?: Effort;
  /** Sent while a turn ran: that turn reads it after its current step. */
  interjected?: boolean;
}

export interface ThinkingEvent extends Recorded {
  type: "thinking";
  content: string;
}

export interface ToolCallEvent extends Recorded {
  type: "tool_call";
  id: string;
  name: string;
  arguments: Record<string, unknown>;
}

export interface ToolResultEvent extends Recorded {
  type: "tool_result";
  id: string | null;
  name: string;
  result: string;
  truncated: boolean;
  isError: boolean;
  panels: Panel[];
  elapsedMs: number | null;
}

/**
 * How far a running tool has got, for the ones that say.
 *
 * Carries the call_id of the tool_call it belongs to, so the step already
 * on screen grows a bar rather than a second row appearing. Tools that
 * report nothing send none of these at all.
 */
export interface ToolProgressEvent extends Recorded {
  type: "tool_progress";
  id: string | null;
  name: string;
  done: number;
  /** Null when the tool knows it is working but not how much of. */
  total: number | null;
  /** What it is on right now, such as "Fetching 1IQK". */
  message: string | null;
}

export interface AssistantMessageEvent extends Recorded {
  type: "assistant_message";
  content: string;
  inputTokens: number;
  outputTokens: number;
}

export interface StatusEvent extends Recorded {
  type: "status";
  state: "busy" | "idle";
  /** Set when the agent opened this turn by itself, under its autonomy budget. */
  auto?: boolean;
}

export interface CancelledEvent extends Recorded {
  type: "cancelled";
}

export interface ErrorEvent extends Recorded {
  type: "error";
  message: string;
}

export interface MonitorAlertEvent extends Recorded {
  type: "monitor_alert";
  alert: Record<string, unknown>;
}

export interface MonitorResponseEvent extends Recorded {
  type: "monitor_response";
  content: string;
}

/** One thing an experiment showed, as the agent wrote it down. */
export interface CampaignFinding {
  /** ISO 8601, UTC. */
  ts: string;
  text: string;
}

/**
 * The log the agent keeps while it works on a goal: what it is after, what
 * each experiment showed and what it does next. Also GET
 * /api/sessions/{id}/campaign; an empty log has every field null or empty.
 */
export interface Campaign {
  goal: string | null;
  /** Oldest first. */
  findings: CampaignFinding[];
  next: string | null;
  /** ISO 8601, UTC; null while the log is empty. */
  updated: string | null;
}

/** The campaign log changed. Each event carries the whole log, so the last one is the current state. */
export interface CampaignEvent extends Recorded {
  type: "campaign";
  campaign: Campaign;
}

// ---------------------------------------------------------------------------
// the fleet — campaigns run by sub-agents, GET /api/fleet/*
//
// A different thing from the `Campaign` above, which is the log one agent
// keeps of its own goal. A FleetCampaign is a piece of work the conductor
// split into tasks and handed to supervisors; the names collide because both
// are called a campaign by the people who use them.
// ---------------------------------------------------------------------------

/** A campaign the conductor opened: one conversation's worth of work. */
export interface FleetCampaign {
  campaign_id: string;
  goal: string;
  /** The TTP every task varies; null when each task writes its own, the usual way round. */
  base_ttp_id: string | null;
  /** Declared objectives; empty when the tasks bring their own. */
  objectives: string[];
  /** Sub-agents the conductor wants working at once. */
  concurrency: number;
  state: string;
  created_at: string;
  dispatch_until: string | null;
  hard_stop_at: string | null;
  finalized_at: string | null;
  session_id: string | null;
}

/** What the conductor asked for, in its own words: the sub-agent's brief. */
export interface TaskSpec {
  /** Handed to the sub-agent verbatim; this is the task. */
  intent?: string;
  /** What sort of work it is: design, research, analysis, retrosynthesis, question. */
  kind?: string;
  /** A few words for the person watching, so a task is recognisable without the brief. */
  title?: string;
  /** What the sub-agent could not have found out for itself. */
  context?: string;
  /** What to come back with, when the shape of the answer matters. */
  deliverable?: string;
  /** Tasks whose results led the conductor here: the lines between the bubbles. */
  derived_from?: string[];
  /** Only on campaigns with a declared objective list. */
  weight_overrides?: Record<string, number>;
  /** One line for the person watching. */
  rationale?: string;
}

/** One experiment, from proposed to finished. */
export interface FleetTask {
  task_id: string;
  campaign_id: string;
  spec: TaskSpec;
  state: string;
  attempts: number;
  created_at: string;
  /** The supervisor holding it; null when nobody has claimed it, and again once it is done. */
  lease_owner: string | null;
  /** The supervisor that last claimed it, kept after it finished; added by the gateway from the claims. */
  ran_by?: string | null;
  lease_expires_at: string | null;
  claimed_at: string | null;
  finished_at: string | null;
  error: string | null;
  /** The REINVENT run it started, once it has one. */
  job_id: string | null;
}

/** One molecule a task put forward. */
export interface FleetCandidate {
  smiles: string;
  score?: number;
  [key: string]: unknown;
}

/** What a finished task reported. */
export interface FleetResult {
  task_id: string;
  campaign_id: string;
  job_id: string | null;
  /** Score per objective of the task's best molecule; empty for work that scored nothing. */
  objective_profile: Record<string, number>;
  candidates: FleetCandidate[];
  summary: string | null;
  /** The sub-agent's own account of the task, in prose. The only thing that crosses back. */
  report?: string | null;
  /** Every run it started, not only the one it was harvested from. */
  job_ids?: string[];
  posted_at: string;
}

/** One sub-agent, as it last reported itself. */
export interface FleetSupervisor {
  owner: string;
  state: string;
  campaign_id: string | null;
  task_id: string | null;
  tasks_completed: number;
  tasks_failed: number;
  detail: string | null;
  first_seen_at: string;
  last_seen_at: string;
  /** Age of the report: what says whether to believe the state beside it. */
  last_seen_seconds_ago: number | null;
}

/**
 * One thing that happened in a campaign, as the ledger recorded it.
 *
 * The ledger's own history, which is the only account of the work that is
 * not an agent's word for it: who claimed what, when a run started, what
 * came back.
 */
export interface FleetEvent {
  event_id: number;
  task_id: string | null;
  /** campaign_created, task_enqueued, task_claimed, task_started_job, task_done, task_failed… */
  kind: string;
  payload: Record<string, unknown>;
  created_at: string;
}

/** One line of a sub-agent's transcript, as its agent wrote it. */
export interface SubagentEvent {
  /** Local time of the supervisor's node, without a zone. */
  ts?: string;
  /** user_message, thinking, tool_call, tool_result, assistant_message, max_rounds, reset… */
  event: string;
  content?: string;
  name?: string;
  arguments?: Record<string, unknown>;
  elapsed_s?: number;
  is_error?: boolean;
  error?: string;
  /** The opening of what a tool returned; older transcripts kept only its length. */
  excerpt?: string;
  content_length?: number;
  round?: number;
  /** A message from the person watching, handed over mid-turn. */
  interjected?: boolean;
}

/** A message left for a sub-agent, POST /api/fleet/tasks/{id}/messages. */
export interface QueuedMessage {
  ts: string;
  text: string;
}

/** GET /api/fleet/tasks/{id}/conversation. */
export interface TaskConversation {
  task_id: string;
  /** False when the sub-agent wrote no transcript under this task's name. */
  found: boolean;
  events: SubagentEvent[];
  truncated: boolean;
  /** Messages left for it that it has not read yet; absent from older gateways. */
  queued?: QueuedMessage[];
}

/** GET /api/fleet/campaigns/{id}/events, oldest first. */
export interface CampaignEvents {
  events: FleetEvent[];
  last_event_id: number;
}

export type FindingConfidence = "high" | "medium" | "low";

/** A second opinion on a finding; it never overwrites the finding itself. */
export interface FindingNote {
  note_id: number;
  finding_id: string;
  body: string;
  stance: "supports" | "disputes" | "extends" | null;
  author: string;
  task_id: string | null;
  created_at: string;
}

/** Something the campaign now knows, under one branch of its plan. */
export interface Finding {
  finding_id: string;
  campaign_id: string;
  branch: string;
  claim: string;
  detail: string | null;
  /** Where it came from: a DOI, a ChEMBL id, a job id, a PDB code. */
  evidence: string[];
  confidence: FindingConfidence;
  tags: string[];
  author: string;
  task_id: string | null;
  created_at: string;
  supersedes: string | null;
  superseded_by: string | null;
  notes: FindingNote[];
}

/** One branch of the atlas: a workstream of the plan, or one an agent filed under. */
export interface AtlasBranch {
  branch: string;
  name: string;
  in_plan: boolean;
  findings: number;
  disputed: number;
  latest: { finding_id: string; claim: string; confidence: FindingConfidence; created_at: string } | null;
  tags: string[];
}

/** GET /api/fleet/campaigns/{id}/atlas: the branches and their current findings, newest first. */
export interface CampaignAtlas {
  campaign_id: string;
  goal: string;
  /** Current findings over the whole campaign. */
  findings: number;
  branches: AtlasBranch[];
  empty_branches: string[];
  items: Finding[];
  /** More findings than one read hands out; the newest are the ones shown. */
  truncated: boolean;
}

/** GET /api/fleet/campaigns/{id}: read together because separately they tear. */
export interface CampaignDetail {
  campaign: FleetCampaign;
  tasks: FleetTask[];
  results: FleetResult[];
  supervisors: FleetSupervisor[];
  /** Null when the ledger could not say; absent from an older gateway. */
  metrics?: CampaignMetrics | null;
}

/** How busy a campaign keeps its team, over it and its leads' campaigns. */
export interface CampaignMetrics {
  concurrency: number;
  running: number;
  queued: number;
  free_slots: number;
  backlog_target: number;
  /** Share of the slots' time spent holding a task; null before any claim. */
  utilisation: number | null;
  busy_seconds: number;
  handoffs: number;
  handoff_seconds: { median: number | null; max: number | null };
}

/**
 * GET /api/fleet/capacity. A null *max_supervisors* is unknown, not
 * unlimited, and a control offering a choice under it has to say so.
 */
export interface FleetCapacity {
  max_supervisors: number | null;
  binding_limit: string | null;
  detail?: string;
}

/** Events that belong to the conversation and are replayed on reconnect. */
export type ConversationEvent =
  | UserMessageEvent
  | ThinkingEvent
  | ToolCallEvent
  | ToolResultEvent
  | ToolProgressEvent
  | AssistantMessageEvent
  | StatusEvent
  | CancelledEvent
  | ErrorEvent
  | MonitorAlertEvent
  | MonitorResponseEvent
  | CampaignEvent;

/**
 * The sub-agent allowance changed. Not recorded, like the autonomy budget:
 * a browser that reconnects reads the count from the session summary.
 */
export interface AgentsEvent {
  type: "agents";
  count: number;
}

export type AgentEvent = ReadyEvent | SessionEvent | NoticeEvent | AutoEvent | AgentsEvent | ConversationEvent;

/** How hard the model reasons on a turn; the gateway sets it on the session's backend. */
export type Effort = "low" | "medium" | "high";

/**
 * What the browser sends. An *auto* of more than 0 minutes starts or extends
 * the autonomy budget, 0 ends it; the gateway caps it.
 */
export type ClientMessage =
  | { type: "user_message"; content: string; effort?: Effort }
  | { type: "cancel" }
  | { type: "auto"; minutes: number }
  /** How many sub-agents the conversation may run at once; 1 is working alone. */
  | { type: "agents"; count: number };

// ---------------------------------------------------------------------------
// models — which model answers, and switching to the other one
// ---------------------------------------------------------------------------

/** One served model. A model without *efforts* does not reason in levels. */
export interface ModelOption {
  key: string;
  id: string;
  label: string;
  efforts: Effort[];
  defaultEffort: Effort | null;
}

/**
 * GET /api/models, and the answer to a POST to it. Only one model runs at a
 * time, so a switch restarts the server: the POST answers 202 with state
 * "loading" and the browser polls until it is ready or failed.
 */
export interface ModelsState {
  /** The key of the model now serving. */
  current: string;
  state: "ready" | "loading" | "failed";
  /** False where there is no cluster to restart; the model is then fixed. */
  switchable: boolean;
  models: ModelOption[];
}

// ---------------------------------------------------------------------------
// sessions — saved conversations
// ---------------------------------------------------------------------------

export interface SessionSummary {
  id: string;
  title: string;
  /** ISO 8601, UTC. */
  created: string;
  updated: string;
  turns: number;
  busy: boolean;
  /** End of the autonomy budget, ISO 8601 UTC; null when there is none. */
  auto_until?: string | null;
  /** Sub-agents this conversation may run at once; 1, the default, is alone. */
  agents_allowed?: number;
  inputTokens?: number;
  outputTokens?: number;
  /** The REINVENT jobs of the conversation; absent when it has none. */
  runs?: SessionRuns;
}

export interface SessionRuns {
  /** Launched here and still running or queued. */
  running: number;
  /** Launched here, including runs continued from those. */
  total: number;
  /** `launched` false: only referenced here, e.g. a status asked for. */
  jobs: { id: string; state: string | null; launched: boolean }[];
}

// ---------------------------------------------------------------------------
// transcript — what the UI keeps and renders
// ---------------------------------------------------------------------------

/**
 * Blocks are appended in the order the agent produced them, so the transcript
 * reads as a record of what was done when. That ordering is the whole point of
 * showing tool activity inline rather than in a separate report.
 */
export type TurnBlock =
  | { kind: "thinking"; id: string; content: string }
  | {
      kind: "tool";
      id: string;
      name: string;
      arguments: Record<string, unknown>;
      /** Epoch ms when the call started. */
      at: number;
      result?: ToolResultEvent;
      /** How far it has got, while it is still running and if it says. */
      progress?: { done: number; total: number | null; message: string | null };
      /** The same call sent this many times in a row with the same answer, shown once. */
      repeats?: number;
    }
  | { kind: "text"; id: string; content: string }
  | { kind: "error"; id: string; message: string }
  | { kind: "cancelled"; id: string }
  | { kind: "alert"; id: string; alert: Record<string, unknown> }
  /** A message the user sent while this turn ran. */
  | { kind: "user"; id: string; content: string };

export interface UserTurn {
  role: "user";
  id: string;
  content: string;
  at: number;
}

export interface AgentTurn {
  role: "agent";
  id: string;
  /** Epoch ms when the turn started. */
  at: number;
  /** Epoch ms when it ended; unset while running. */
  endedAt?: number;
  blocks: TurnBlock[];
  /** The agent started this turn by itself, under its autonomy budget. */
  auto?: boolean;
  /** Set once the turn finishes; drives the "working" indicator. */
  done: boolean;
  tokens?: { input: number; output: number };
}

export type Turn = UserTurn | AgentTurn;

export type ConnectionState = "connecting" | "open" | "closed" | "error";

export interface SessionInfo {
  backend: string;
  model: string;
  endpoint: string;
  tools: string[];
  ready: boolean;
}
