// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { PanelShown } from "./PanelShown";
import { usePanelShown } from "./panelShownContext";

function Reads() {
  return <i>{usePanelShown() ? "polling" : "paused"}</i>;
}

const read = (node: React.ReactElement) =>
  renderToStaticMarkup(node).replace(/<\/?i>/g, "");

describe("usePanelShown", () => {
  it("polls where nothing says the panel is hidden", () => {
    expect(read(<Reads />)).toBe("polling");
  });

  it("pauses a hidden panel", () => {
    expect(
      read(
        <PanelShown shown={false}>
          <Reads />
        </PanelShown>,
      ),
    ).toBe("paused");
  });

  it("polls a shown panel", () => {
    expect(
      read(
        <PanelShown shown>
          <Reads />
        </PanelShown>,
      ),
    ).toBe("polling");
  });

  it("pauses the open view of a tab nobody is looking at", () => {
    // The Campaign tab keeps its views mounted and so does the dock, so the
    // run analysis left open inside a tab that was switched away from is
    // shown by its own reckoning and on screen by nobody else's.
    expect(
      read(
        <PanelShown shown={false}>
          <PanelShown shown>
            <Reads />
          </PanelShown>
        </PanelShown>,
      ),
    ).toBe("paused");
  });
});
