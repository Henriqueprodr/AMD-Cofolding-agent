// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { ComponentProps } from "react";
import { Editor } from "ketcher-react";
import { StandaloneStructServiceProvider } from "ketcher-standalone";
import type { Ketcher, StructServiceProvider } from "ketcher-core";
import type { SketchMode } from "../../lib/sketcherBus";
import "ketcher-react/dist/index.css";
import "./sketcher.css";

// One chemistry engine (Indigo in WebAssembly) for every opening of the sketcher.
const structServiceProvider = new StandaloneStructServiceProvider() as unknown as StructServiceProvider;

type Buttons = NonNullable<ComponentProps<typeof Editor>["buttons"]>;

/** Buttons that lead away from drawing one small molecule. */
const HIDDEN: (keyof Buttons)[] = [
  "help",
  "about",
  "miew",
  "settings",
  "fullscreen",
  "sgroup",
  "reaction-plus",
  "arrows",
  "reaction-mapping-tools",
  "shape",
  "text",
  "create-monomer",
];

// The R-group group holds the attachment point tool, which is the only way to
// mark a position for LibInvent or LinkInvent. It stays hidden when drawing an
// ordinary molecule, where an attachment point is a mistake rather than the
// point.
const hiddenFor = (mode: SketchMode): Buttons =>
  Object.fromEntries(
    [...HIDDEN, ...(mode === "molecule" ? (["rgroup"] as (keyof Buttons)[]) : [])].map((name) => [
      name,
      { hidden: true },
    ]),
  );

interface Props {
  initial?: string;
  /** What is being drawn; decides whether attachment points can be placed. */
  mode?: SketchMode;
  onReady: (ketcher: Ketcher) => void;
  onError: (message: string) => void;
}

export default function KetcherEditor({ initial, mode = "molecule", onReady, onError }: Props) {
  return (
    <Editor
      staticResourcesUrl=""
      structServiceProvider={structServiceProvider}
      errorHandler={onError}
      disableMacromoleculesEditor
      buttons={hiddenFor(mode)}
      onInit={(ketcher: Ketcher) => {
        const start = initial ? ketcher.setMolecule(initial) : Promise.resolve();
        start.then(
          () => onReady(ketcher),
          () => {
            onError(`The sketcher could not read ${initial}.`);
            onReady(ketcher);
          },
        );
      }}
    />
  );
}
