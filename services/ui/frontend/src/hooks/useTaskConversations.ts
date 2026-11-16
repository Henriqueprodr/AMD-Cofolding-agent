// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useMemo, useState } from "react";
import type { TaskConversation } from "../types/api";
import { fetchTaskConversation } from "../lib/api";
import { stuckOn } from "../lib/subagentChat";
import { usePanelShown } from "./panelShownContext";

/** A running sub-agent's transcript is read again this often; it writes a line per step. */
const LIVE_MS = 5_000;

/** Per task, for the page: switching between agents must not read them all again. */
const read = new Map<string, TaskConversation>();
const listeners = new Set<() => void>();

/** Read one task's conversation now, e.g. after leaving it a message, rather than at the next poll. */
export async function reloadTaskConversation(taskId: string): Promise<void> {
  try {
    read.set(taskId, await fetchTaskConversation(taskId));
  } catch {
    return;
  }
  listeners.forEach((listener) => listener());
}

/**
 * The transcripts of *taskIds*, as far as they have been read. *live* are
 * the tasks still being worked on, read again while they are; a finished
 * one is read once.
 */
export function useTaskConversations(taskIds: readonly string[], live: ReadonlySet<string>): ReadonlyMap<string, TaskConversation> {
  const [, setVersion] = useState(0);
  const shown = usePanelShown();
  const key = taskIds.join(",");
  const liveKey = [...live].sort().join(",");

  useEffect(() => {
    const bump = () => setVersion((v) => v + 1);
    listeners.add(bump);
    return () => {
      listeners.delete(bump);
    };
  }, []);

  useEffect(() => {
    if (!key) return;
    let cancelled = false;
    const ids = key.split(",");
    const running = new Set(liveKey ? liveKey.split(",") : []);
    const load = (only: (id: string) => boolean) =>
      Promise.all(
        ids.filter(only).map((id) =>
          fetchTaskConversation(id)
            .then((conversation) => read.set(id, conversation))
            .catch(() => undefined),
        ),
      ).then(() => {
        if (!cancelled) setVersion((v) => v + 1);
      });

    void load((id) => !read.has(id) || running.has(id));
    const timer =
      running.size && shown
        ? window.setInterval(() => {
            if (!document.hidden) void load((id) => running.has(id));
          }, LIVE_MS)
        : 0;
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [key, liveKey, shown]);

  return new Map(taskIds.flatMap((id) => (read.has(id) ? [[id, read.get(id)!] as const] : [])));
}

/** Of *running* tasks, those whose agent keeps resending a call that fails the same way, with that call's name. */
export function useStuckTasks(running: readonly string[]): ReadonlyMap<string, string> {
  const key = running.join(",");
  const live = useMemo(() => new Set(key ? key.split(",") : []), [key]);
  const conversations = useTaskConversations(running, live);
  const stuck = new Map<string, string>();
  for (const [id, conversation] of conversations) {
    const name = conversation.found ? stuckOn(conversation.events) : null;
    if (name) stuck.set(id, name);
  }
  return stuck;
}
