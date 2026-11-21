// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// The only 3Dmol-aware file. Everything else talks to this small surface,
// so swapping renderers later is a one-file change.

(function (global) {
  "use strict";

  let viewer = null;
  let current = null; // { format, data, meta }
  let spinning = false;

  // Solvent and counter-ions: present in most crystal structures, almost
  // never what the user means by "the ligand".
  const SOLVENT = ["HOH", "WAT", "NA", "CL", "SO4", "ZN", "MG"];

  function init(element) {
    viewer = global.$3Dmol.createViewer(element, {
      backgroundColor: getComputedStyle(document.body)
        .getPropertyValue("--bg")
        .trim() || "#11141a",
    });
    return viewer;
  }

  /** Load a structure. `payload` is {format, data, meta} from the server. */
  function load(payload) {
    if (!viewer) return;
    current = payload;
    spinning = false;
    viewer.spin(false);
    viewer.removeAllModels();
    viewer.removeAllSurfaces();
    viewer.addModel(payload.data, payload.format);
    applyRepresentation(defaultReps(payload.meta || {}));
    frame();
    viewer.render();
  }

  /** zoomTo() alone crops a small molecule at the panel edges. */
  function frame() {
    viewer.zoomTo();
    viewer.zoom(0.85);
  }

  /** Which toggles make sense for what was just loaded. */
  function defaultReps(meta) {
    return {
      protein: meta.has_protein !== false && !!meta.has_protein,
      ligand: true,
      water: false,
    };
  }

  function applyRepresentation(reps) {
    if (!viewer || !current) return;
    const meta = current.meta || {};
    viewer.setStyle({}, {}); // clear

    // A very large structure is unusable as cartoon+sticks; say so rather
    // than hanging the tab.
    const heavy = (meta.n_atoms || 0) > 50000;

    if (reps.protein && meta.has_protein) {
      viewer.setStyle({ hetflag: false }, { cartoon: { color: "spectrum" } });
    }
    if (reps.ligand) {
      // Prefer the resnames the server identified: selecting every hetero
      // atom would draw the crystallographic waters as ligand spheres.
      let sel;
      if (meta.ligand_resnames && meta.ligand_resnames.length) {
        sel = { resn: meta.ligand_resnames };
      } else if (meta.has_protein) {
        sel = { hetflag: true, not: { resn: SOLVENT } };
      } else {
        sel = {};
      }
      if (!heavy) {
        viewer.setStyle(sel, {
          stick: { radius: 0.17, colorscheme: "greenCarbon" },
          sphere: { radius: 0.32, colorscheme: "greenCarbon" },
        });
      } else {
        viewer.setStyle(sel, { stick: { radius: 0.17 } });
      }
    }
    if (reps.water) {
      viewer.addStyle({ resn: SOLVENT }, { sphere: { radius: 0.3 } });
    }
    viewer.render();
  }

  function reset() {
    if (!viewer) return;
    viewer.spin(false);
    spinning = false;
    if (current) applyRepresentation(defaultReps(current.meta || {}));
    frame();
    viewer.render();
  }

  function toggleSpin() {
    if (!viewer) return spinning;
    spinning = !spinning;
    viewer.spin(spinning ? "y" : false);
    return spinning;
  }

  function clear() {
    if (!viewer) return;
    current = null;
    viewer.removeAllModels();
    viewer.render();
  }

  function currentPayload() {
    return current;
  }

  global.MolViewer = {
    init,
    load,
    applyRepresentation,
    defaultReps,
    reset,
    toggleSpin,
    clear,
    currentPayload,
  };
})(window);
