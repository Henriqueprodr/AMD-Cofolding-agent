// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { Header } from "./Header";

const noop = () => undefined;
const render = () =>
  renderToStaticMarkup(
    <Header
      theme="light"
      onToggleTheme={noop}
      connection="open"
      agent={null}
      gatewayError={null}
      title="EGFR binders"
      onReconnect={noop}
      railOpen
      onToggleRail={noop}
      workbenchOpen
      onToggleWorkbench={noop}
    />,
  );

describe("Header", () => {
  it("names the conversation and nothing else about it", () => {
    const html = render();
    expect(html).toContain("EGFR binders");
    expect(html).not.toContain("No target yet");
    expect(html).not.toContain("working…");
    expect(html).not.toContain("Actions");
  });
});
