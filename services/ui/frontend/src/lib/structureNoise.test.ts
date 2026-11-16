// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import { isStructureNoise, ligandCompId, partitionStructures } from "./structureNoise";

const PREPARED = "prepared/1m17";

describe("isStructureNoise", () => {
  it("skips the waters and ions preparation writes beside the complex", () => {
    expect(isStructureNoise(`${PREPARED}/water.pdb`)).toBe(true);
    expect(isStructureNoise(`${PREPARED}/ions.pdb`)).toBe(true);
  });

  it("skips the hydrogen-free receptor, which is the receptor again for docking", () => {
    expect(isStructureNoise(`${PREPARED}/receptor_noh.pdb`)).toBe(true);
  });

  it("keeps what the chemist asked for", () => {
    for (const name of ["complex.pdb", "receptor.pdb", "binder.pdb", "pocket.pdb", "1M17.cif"]) {
      expect(isStructureNoise(`${PREPARED}/${name}`), name).toBe(false);
    }
  });

  it("skips a ligand file that is crystallisation chemistry", () => {
    // One prepared entry wrote a file per retained ligand instance, sulfates
    // and glycerols included, and each one became a card in the answer.
    expect(isStructureNoise(`${PREPARED}/ligand_SO4_A_401.pdb`)).toBe(true);
    expect(isStructureNoise(`${PREPARED}/ligand_GOL_B_502.pdb`)).toBe(true);
    expect(isStructureNoise(`${PREPARED}/ligand_ZN_A_301.pdb`)).toBe(true);
  });

  it("keeps a ligand file that is the ligand", () => {
    expect(isStructureNoise(`${PREPARED}/ligand_AQ4_A_999.sdf`)).toBe(false);
    expect(isStructureNoise(`${PREPARED}/ligand_MI1_A_1.pdb`)).toBe(false);
  });

  it("reads the component id out of the file name, and only that shape", () => {
    expect(ligandCompId("ligand_AQ4_A_999.sdf")).toBe("AQ4");
    expect(ligandCompId("receptor.pdb")).toBeNull();
  });
});

describe("partitionStructures", () => {
  it("keeps both sides, in the order they came", () => {
    const given = [
      { path: `${PREPARED}/complex.pdb` },
      { path: `${PREPARED}/water.pdb` },
      { path: `${PREPARED}/ligand_AQ4_A_999.sdf` },
      { path: `${PREPARED}/ligand_SO4_A_401.pdb` },
    ];

    const { shown, noise } = partitionStructures(given);

    expect(shown.map((s) => s.path)).toEqual([given[0].path, given[2].path]);
    // Nothing is dropped: the overflow list names them, so a sulfate is still
    // one click away.
    expect(noise.map((s) => s.path)).toEqual([given[1].path, given[3].path]);
  });
});
