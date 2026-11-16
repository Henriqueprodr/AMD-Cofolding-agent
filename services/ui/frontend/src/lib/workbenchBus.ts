// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useSyncExternalStore } from "react";
import type { DockView } from "./dock";

/**
 * Lets anything in the page bring up a workbench tab: an "Open candidates"
 * link under an answer, "Open jobs" on a monitor alert. The app listens and
 * switches tabs.
 *
 * The same set the dock knows, so a tab added there can be asked for here
 * without the two lists drifting apart.
 */
export type WorkbenchTab = DockView;

type Listener = (tab: WorkbenchTab) => void;

const listeners = new Set<Listener>();

export function openWorkbench(tab: WorkbenchTab): void {
  listeners.forEach((listener) => listener(tab));
}

export function onWorkbenchRequest(listener: Listener): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/*
 * The REINVENT run picked for the workbench (see scopedJob), null for the
 * conversation's own. Shared by Summary, Results, Analysis and the run bar, so
 * a run picked in one is the run shown in the others; the app clears it when
 * another conversation opens.
 */
let focused: string | null = null;
const runListeners = new Set<() => void>();
const changeListeners = new Set<() => void>();

export function focusRun(jobId: string | null): void {
  if (jobId === focused) return;
  focused = jobId;
  runListeners.forEach((listener) => listener());
}

export const focusedRun = (): string | null => focused;

export function useFocusedRun(): string | null {
  return useSyncExternalStore(
    (listener) => {
      runListeners.add(listener);
      return () => runListeners.delete(listener);
    },
    focusedRun,
  );
}

/** Show one run in the Jobs or Results tab. */
export function openRun(jobId: string, tab: "jobs" | "results" = "jobs"): void {
  focusRun(jobId);
  openWorkbench(tab);
}

/*
 * The campaign task the Campaign tab is reading, null for none. Shared the
 * same way as the focused run, because two things point at it: the bubble
 * plot above and the campaign's own lists below, and a task opened from one
 * has to be the task the other shows as open.
 */
let task: FocusedTask | null = null;
const taskListeners = new Set<() => void>();

export interface FocusedTask {
  campaignId: string;
  /** Null for the campaign as a whole. */
  taskId: string | null;
}

export function focusTask(next: FocusedTask | null): void {
  if (next?.campaignId === task?.campaignId && next?.taskId === task?.taskId) return;
  task = next;
  taskListeners.forEach((listener) => listener());
}

export const focusedTask = (): FocusedTask | null => task;

export function useFocusedTask(): FocusedTask | null {
  return useSyncExternalStore((listener) => {
    taskListeners.add(listener);
    return () => taskListeners.delete(listener);
  }, focusedTask);
}

/** Show one sub-agent's task, with its brief and its report. */
export function openTask(next: FocusedTask): void {
  // A task is read on the overview, which an open sub-agent would cover.
  focusAgent(null);
  focusTask(next);
  openWorkbench("campaign");
}

/** Show one task as its sub-agent saw it: the conversation, molecules and analysis of that task alone. */
export function openStoredTask(label: string, next: FocusedTask): void {
  focusAgent(label);
  focusTask(next);
  openWorkbench("campaign");
}

/*
 * The sub-agent whose side of the work is open, by its S-label, null for
 * none. Shared like the task above: the plot, the history and the campaign's
 * own lists all open one, and all of them have to agree on which.
 */
let agent: string | null = null;
const agentListeners = new Set<() => void>();

export function focusAgent(next: string | null): void {
  if (next === agent) return;
  agent = next;
  agentListeners.forEach((listener) => listener());
}

export const focusedAgent = (): string | null => agent;

export function useFocusedAgent(): string | null {
  return useSyncExternalStore((listener) => {
    agentListeners.add(listener);
    return () => agentListeners.delete(listener);
  }, focusedAgent);
}

/*
 * The sub-agent whose conversation the chat column shows instead of the
 * conductor's, null for the conductor's. Apart from the focused agent above:
 * reading one in the chat does not have to move the workbench.
 */
let chatting: string | null = null;
const chatListeners = new Set<() => void>();

export function openAgentChat(label: string | null): void {
  if (label === chatting) return;
  chatting = label;
  chatListeners.forEach((listener) => listener());
}

export const agentChat = (): string | null => chatting;

export function useAgentChat(): string | null {
  return useSyncExternalStore((listener) => {
    chatListeners.add(listener);
    return () => chatListeners.delete(listener);
  }, agentChat);
}

/** Say that a run was started or stopped here, so views of the job list reload. */
export function notifyRunsChanged(): void {
  changeListeners.forEach((listener) => listener());
}

export function onRunsChanged(listener: () => void): () => void {
  changeListeners.add(listener);
  return () => {
    changeListeners.delete(listener);
  };
}
