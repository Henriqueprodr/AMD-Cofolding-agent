// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useLayoutEffect, useRef, useState } from "react";

/**
 * How many items of at least `itemWidth` pixels fit across the container,
 * keeping `reserveWidth` free for a "…" overflow button when some do not.
 *
 * Returns `[containerRef, visibleCount]`. Width is taken from the container
 * alone rather than from the items, so the items the count hides can be left
 * out of the DOM entirely -- hiding them with `visibility` instead would keep
 * their width and push the "…" button away from the last visible item.
 *
 * Re-measures when the container resizes (the dock being dragged, the window)
 * and when `itemCount` changes.
 */
export function useOverflowCount(itemCount: number, itemWidth: number, reserveWidth = 36): [React.RefObject<HTMLDivElement | null>, number] {
  const ref = useRef<HTMLDivElement>(null);
  const [count, setCount] = useState(itemCount);

  useLayoutEffect(() => {
    const container = ref.current;
    if (!container) return;

    function measure() {
      if (!container) return;
      const room = container.clientWidth;
      if (!room) return;
      const all = Math.floor(room / itemWidth);
      // Only pay for the "…" button when it is actually needed.
      const fit = all >= itemCount ? itemCount : Math.floor((room - reserveWidth) / itemWidth);
      setCount(Math.max(1, Math.min(fit, itemCount)));
    }

    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(container);
    return () => observer.disconnect();
  }, [itemCount, itemWidth, reserveWidth]);

  return [ref, count];
}
