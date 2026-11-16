// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { CampaignDetail, FleetEvent } from "../../types/api";
import type { CampaignView } from "../../lib/campaign";

// The branches read the ledger through a hook; this card's job is the layout
// around them, so they stand in as a marker.
vi.mock("./WhatWeKnow", () => ({
  CampaignBranches: ({ campaignId }: { campaignId: string }) => <div data-branches={campaignId} />,
}));

const { CampaignMemory } = await import("./CampaignMemory");

function detail(campaignId: string, goal: string): CampaignDetail {
  return {
    campaign: {
      campaign_id: campaignId,
      goal,
      base_ttp_id: null,
      objectives: [],
      concurrency: 2,
      state: "dispatching",
      created_at: "2026-11-16T12:00:00+00:00",
      dispatch_until: null,
      hard_stop_at: null,
      finalized_at: null,
      session_id: "s-1",
    },
    tasks: [],
    results: [],
    supervisors: [],
  };
}

const MEMORY: CampaignView = {
  goal: "Optimize JQ1 for BRD4 BD1 potency",
  next: "Wait for the SAR report",
  findings: [{ ts: "2026-11-16T12:30:00+00:00", text: "Asn140 anchors the acetyl-lysine mimic" }],
  updated: "2026-11-16T12:30:00+00:00",
};

const REPORTS = [
  {
    taskId: "t1",
    campaignId: "c1",
    title: "BRD4 Structural Landscape",
    excerpt: "Five co-crystal structures cover BD1.",
    postedAt: "2026-11-16T12:40:00+00:00",
  },
];

const render = (props: Partial<Parameters<typeof CampaignMemory>[0]> = {}) =>
  renderToStaticMarkup(
    <CampaignMemory
      details={[detail("c1", "Optimize JQ1 for BRD4 BD1 potency")]}
      events={new Map<string, readonly FleetEvent[]>()}
      memory={MEMORY}
      reports={REPORTS}
      labels={new Map()}
      runs={new Set()}
      now={Date.parse("2026-11-16T12:45:00+00:00")}
      onOpenRun={() => undefined}
      onOpenReport={() => undefined}
      {...props}
    />,
  );

describe("CampaignMemory", () => {
  it("puts the goal, the log and the reports under one Memory heading", () => {
    const html = render();

    expect(html).toContain("Memory");
    // The three cards it replaces must not come back as headings of their own.
    expect(html).not.toContain("What we know");
    expect(html).toContain("Optimize JQ1 for BRD4 BD1 potency");
    expect(html).toContain("Wait for the SAR report");
    expect(html).toContain("Asn140 anchors the acetyl-lysine mimic");
    expect(html).toContain("BRD4 Structural Landscape");
    expect(html).toContain("1 report");
  });

  it("keeps two campaigns in the same card, named, rather than duplicating itself", () => {
    const html = render({
      details: [detail("c1", "Optimize JQ1"), detail("c2", "Explore BD1 SAR")],
    });

    expect(html.match(/>Memory</g) ?? []).toHaveLength(1);
    expect(html).toContain('data-branches="c1"');
    expect(html).toContain('data-branches="c2"');
    // Each campaign is labelled once the conversation has more than one.
    expect(html).toContain("Explore BD1 SAR");
  });

  it("does not name a lone campaign, which would repeat the goal above it", () => {
    const html = render({ details: [detail("c1", "Optimize JQ1 for BRD4 BD1 potency")] });

    expect(html.match(/Optimize JQ1 for BRD4 BD1 potency/g) ?? []).toHaveLength(1);
  });

  it("shows nothing at all when the conversation has no campaign and no log", () => {
    expect(render({ details: [], memory: null, reports: [] })).toBe("");
  });

  it("still fills in for a single agent, which has a log but no sub-agent reports", () => {
    const html = render({ details: [], reports: [] });

    expect(html).toContain("Optimize JQ1 for BRD4 BD1 potency");
    expect(html).toContain("Asn140 anchors the acetyl-lysine mimic");
  });
});
