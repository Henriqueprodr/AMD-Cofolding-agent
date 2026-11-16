// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import {
  countAttachmentPoints,
  fragmentList,
  fragmentProblems,
  mapNumbers,
  numberAttachmentPoints,
  numberFragments,
  scaffoldProblems,
} from "./attachment";

describe("counting attachment points", () => {
  it("counts every way SMILES writes a dummy atom", () => {
    expect(countAttachmentPoints("*c1ccccc1")).toBe(1);
    expect(countAttachmentPoints("[*]c1ccccc1")).toBe(1);
    expect(countAttachmentPoints("[*:1]c1ccccc1[*:2]")).toBe(2);
  });

  it("finds none in an ordinary molecule", () => {
    expect(countAttachmentPoints("CCO")).toBe(0);
  });

  it("reads back the numbers that are written", () => {
    expect(mapNumbers("[*:3]c1ccccc1[*:7]")).toEqual([3, 7]);
    expect(mapNumbers("[*]c1ccccc1")).toEqual([]);
  });
});

describe("numbering what the sketcher drew", () => {
  it("numbers bare points, which is the whole reason this exists", () => {
    // Ketcher draws the position but never the number, and LibInvent refuses
    // a bare [*].
    expect(numberAttachmentPoints("[*]c1ccccc1[*]")).toBe("[*:1]c1ccccc1[*:2]");
  });

  it("numbers the bare asterisk form too", () => {
    expect(numberAttachmentPoints("*c1ccccc1")).toBe("[*:1]c1ccccc1");
  });

  it("leaves numbers the user already chose alone", () => {
    expect(numberAttachmentPoints("[*:5]c1ccccc1")).toBe("[*:5]c1ccccc1");
  });

  it("fills around the numbers in use rather than colliding with them", () => {
    expect(numberAttachmentPoints("[*:1]c1ccccc1[*]")).toBe("[*:1]c1ccccc1[*:2]");
    expect(numberAttachmentPoints("[*:2]c1ccccc1[*]")).toBe("[*:2]c1ccccc1[*:1]");
  });

  it("leaves a molecule with no attachment points untouched", () => {
    expect(numberAttachmentPoints("CCO")).toBe("CCO");
  });
});

describe("a LibInvent scaffold", () => {
  it("is accepted when every point is numbered", () => {
    expect(scaffoldProblems("[*:1]c1ccccc1[*:2]")).toEqual([]);
  });

  it("must have somewhere to decorate", () => {
    expect(scaffoldProblems("c1ccccc1")[0]).toMatch(/at least one attachment point/);
  });

  it("is not silently accepted unnumbered", () => {
    expect(scaffoldProblems("[*]c1ccccc1")[0]).toMatch(/needs a number/);
  });

  it("rejects two points sharing a number", () => {
    expect(scaffoldProblems("[*:1]c1ccccc1[*:1]")).toContain("Two attachment points share a number.");
  });

  it("asks for something rather than nothing", () => {
    expect(scaffoldProblems("   ")).toHaveLength(1);
  });
});

describe("LinkInvent fragments", () => {
  it("accepts two fragments with one numbered point each", () => {
    expect(fragmentProblems(["[*:1]CC", "[*:2]OC"])).toEqual([]);
  });

  it("needs both sides of the linker", () => {
    expect(fragmentProblems(["[*:1]CC"])[0]).toMatch(/two fragments/);
  });

  it("refuses a fragment with two open positions", () => {
    // Which end would the linker use?
    expect(fragmentProblems(["[*:1]C[*:2]", "[*:3]OC"])[0]).toMatch(/exactly one/);
  });

  it("refuses a fragment with none", () => {
    expect(fragmentProblems(["CC", "[*:2]OC"])[0]).toMatch(/exactly one/);
  });

  it("refuses the same number on both ends", () => {
    expect(fragmentProblems(["[*:1]CC", "[*:1]OC"])).toContain(
      "The two fragments share an attachment-point number.",
    );
  });

  it("numbers a drawn pair so the two ends stay apart", () => {
    // Numbering each on its own would make both [*:1].
    expect(numberFragments(["[*]CC", "[*]OC"])).toEqual(["[*:1]CC", "[*:2]OC"]);
  });

  it("keeps numbering a pair that is already partly numbered", () => {
    expect(numberFragments(["[*:4]CC", "[*]OC"])).toEqual(["[*:4]CC", "[*:5]OC"]);
  });
});

describe("reading fragments from one field", () => {
  it("splits on the separators someone would actually type", () => {
    expect(fragmentList("[*:1]CC | [*:2]OC")).toEqual(["[*:1]CC", "[*:2]OC"]);
    expect(fragmentList("[*:1]CC\n[*:2]OC")).toEqual(["[*:1]CC", "[*:2]OC"]);
    expect(fragmentList("[*:1]CC, [*:2]OC")).toEqual(["[*:1]CC", "[*:2]OC"]);
  });

  it("ignores empty space", () => {
    expect(fragmentList("   ")).toEqual([]);
  });
});
