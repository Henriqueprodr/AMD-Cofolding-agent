// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { ReactNode } from "react";
import { PanelShownContext, usePanelShown } from "./panelShownContext";

/**
 * Marks whether what it wraps is on screen, so a hidden panel stops polling.
 *
 * The dock nests its panels, so this nests too: a view is on screen only if
 * every panel holding it is as well.
 */
export function PanelShown({
  shown,
  children,
}: {
  shown: boolean;
  children: ReactNode;
}) {
  const parent = usePanelShown();
  return (
    <PanelShownContext.Provider value={parent && shown}>
      {children}
    </PanelShownContext.Provider>
  );
}
