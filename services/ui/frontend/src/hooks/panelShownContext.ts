// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { createContext, useContext } from "react";

/**
 * Whether the panel reading this is the one actually on screen.
 *
 * The dock hides a panel rather than unmounting it, so coming back to a tab
 * finds it as it was left rather than fetching and laying it out again. Its
 * timers were kept the same way, which nobody asked for: after opening the
 * runs, the results, the analysis and the molecules once, a conversation
 * polled from all four for as long as it stayed open and threw all but one
 * of the answers away. `document.hidden` does not help -- the window is in
 * front, it is the panel that is not.
 *
 * The dock nests its panels, so the provider multiplies: a view is on screen
 * only if every panel holding it is too.
 */
export const PanelShownContext = createContext(true);

export const usePanelShown = () => useContext(PanelShownContext);
