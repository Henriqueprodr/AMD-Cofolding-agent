// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useState } from "react";

/**
 * The clock, re-read every *everyMs*.
 *
 * The graph places running tasks at "now" and the headings say how long ago
 * things happened, so both need a clock that moves; reading Date.now() while
 * rendering would make the output depend on when React happened to redraw.
 */
export function useNow(everyMs = 10_000): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), everyMs);
    return () => window.clearInterval(timer);
  }, [everyMs]);
  return now;
}

/** The tail of an id, which is the part that tells two of them apart. */
export function shortId(id: string): string {
  return id.length > 10 ? `…${id.slice(-8)}` : id;
}
