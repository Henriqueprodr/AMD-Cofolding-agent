// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// WebSocket client + transcript rendering.
//
// XSS rule for this file: the ONLY innerHTML assignment is the DOMPurify
// output in renderMarkdown(). Everything else -- tool arguments, SMILES,
// file paths, web-search titles -- goes in through textContent.

(function () {
  "use strict";

  const md = window.markdownit({ html: false, linkify: true, breaks: false });

  const el = (id) => document.getElementById(id);
  const transcript = el("transcript");
  const statusbar = el("statusbar");
  const input = el("input");

  let socket = null;
  let session = null;
  let busy = false;
  let currentTurn = null;
  const toolCards = new Map(); // call_id -> <details>

  // ---------------------------------------------------------------- utils

  function scrolledToBottom() {
    return (
      transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight <
      120
    );
  }

  function append(node) {
    const stick = scrolledToBottom();
    transcript.appendChild(node);
    if (stick) transcript.scrollTop = transcript.scrollHeight;
    return node;
  }

  function div(className, text) {
    const node = document.createElement("div");
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function renderMarkdown(text) {
    const node = document.createElement("div");
    node.innerHTML = window.DOMPurify.sanitize(md.render(text || ""));
    node.querySelectorAll("a").forEach((a) => {
      a.setAttribute("rel", "noopener noreferrer");
      a.setAttribute("target", "_blank");
    });
    return node;
  }

  function summarizeArgs(args) {
    if (!args || typeof args !== "object") return "";
    return Object.entries(args)
      .map(([k, v]) => {
        let shown;
        if (Array.isArray(v)) shown = `[${v.length} item${v.length === 1 ? "" : "s"}]`;
        else if (v === null || v === undefined) shown = "null";
        else if (typeof v === "object") shown = "{…}";
        else {
          shown = String(v);
          if (shown.length > 40) shown = shown.slice(0, 37) + "…";
        }
        return `${k}=${shown}`;
      })
      .join(", ");
  }

  function scoreClass(value) {
    if (typeof value !== "number") return "";
    if (value >= 0.7) return "score-good";
    if (value >= 0.4) return "score-ok";
    return "score-bad";
  }

  function fmt(value) {
    if (value === null || value === undefined) return "—";
    if (typeof value !== "number") return String(value);
    return Math.abs(value) >= 100 ? value.toFixed(2) : value.toFixed(3);
  }

  // ------------------------------------------------------------ rendering

  function addUser(text) {
    append(div("msg user", text));
  }

  function addAssistant(agent, text) {
    const wrap = div("msg assistant");
    wrap.dataset.agent = agent || "driver";
    if (agent && agent !== "driver") {
      wrap.appendChild(
        div("agent-badge", agent === "reviewer" ? "Reviewer (read-only)" : agent)
      );
    }
    wrap.appendChild(renderMarkdown(text));
    append(wrap);
  }

  function addStatus(text, level) {
    append(div("status " + (level || ""), text));
  }

  function addThinking(text) {
    append(div("thinking", text));
  }

  function addToolCall(msg) {
    const card = document.createElement("details");
    card.className = "tool";
    const summary = document.createElement("summary");
    const name = document.createElement("span");
    name.className = "tool-name";
    name.textContent = "> " + msg.name;
    summary.appendChild(name);
    summary.appendChild(document.createTextNode("(" + summarizeArgs(msg.arguments) + ")"));
    const pending = div("tool-summary", " running…");
    summary.appendChild(pending);
    card.appendChild(summary);

    const args = document.createElement("pre");
    args.textContent = JSON.stringify(msg.arguments, null, 2);
    card.appendChild(args);

    toolCards.set(msg.call_id, card);
    append(card);
  }

  function addToolResult(msg) {
    const card = toolCards.get(msg.call_id);
    if (!card) return;
    card.dataset.error = String(!!msg.is_error);
    const pending = card.querySelector("summary .tool-summary");
    if (pending) pending.textContent = " " + msg.summary;

    if (msg.inline) {
      const body = document.createElement("pre");
      body.textContent = msg.inline;
      card.appendChild(body);
    } else if (msg.payload_ref) {
      const link = document.createElement("button");
      link.type = "button";
      link.textContent = `full result (${msg.bytes} bytes)`;
      link.addEventListener("click", async () => {
        const res = await fetch("/api/result/" + encodeURIComponent(msg.payload_ref));
        const body = document.createElement("pre");
        body.textContent = res.ok ? await res.text() : "could not load result";
        card.appendChild(body);
        link.remove();
      });
      card.appendChild(link);
    }
  }

  function addBuiltinActivity(msg) {
    const card = document.createElement("details");
    card.className = "tool builtin";
    card.dataset.error = String(!!msg.is_error);

    const summary = document.createElement("summary");
    const name = document.createElement("span");
    name.className = "tool-name";
    name.textContent = "~ " + msg.name;
    summary.appendChild(name);
    const tag = document.createElement("span");
    tag.className = "web-tag";
    tag.textContent = "web";
    summary.appendChild(tag);
    summary.appendChild(
      document.createTextNode(" (" + summarizeArgs(msg.arguments) + ")")
    );
    card.appendChild(summary);

    const body = document.createElement("pre");
    body.textContent = msg.result || "(no content returned)";
    card.appendChild(body);
    append(card);
  }

  function addScoreTable(msg) {
    const wrap = div("score-wrap");
    const table = document.createElement("table");
    table.className = "score-table";

    const thead = document.createElement("thead");
    const headRow = document.createElement("tr");
    ["SMILES", "Total", ...msg.components, ""].forEach((label, i) => {
      const th = document.createElement("th");
      th.textContent = label;
      if (i < msg.components.length + 2) {
        th.addEventListener("click", () => sortBy(table, i));
      }
      headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    msg.molecules.forEach((mol) => {
      const row = document.createElement("tr");

      const smiles = document.createElement("td");
      smiles.textContent = mol.smiles || "";
      smiles.title = mol.smiles || "";
      smiles.addEventListener("click", () =>
        navigator.clipboard?.writeText(mol.smiles || "")
      );
      row.appendChild(smiles);

      const total = document.createElement("td");
      total.textContent = fmt(mol.total_score);
      total.className = scoreClass(mol.total_score);
      row.appendChild(total);

      msg.components.forEach((component) => {
        const cell = document.createElement("td");
        const entry = (mol.components || {})[component] || {};
        cell.textContent = fmt(entry.transformed);
        cell.className = scoreClass(entry.transformed);
        if (entry.raw !== undefined && entry.raw !== null) {
          const raw = document.createElement("span");
          raw.className = "raw";
          raw.textContent = " (" + fmt(entry.raw) + ")";
          cell.appendChild(raw);
        }
        row.appendChild(cell);
      });

      const action = document.createElement("td");
      const button = document.createElement("button");
      button.type = "button";
      button.className = "btn-3d";
      button.textContent = "3D";
      button.addEventListener("click", () =>
        loadStructure(
          { source: "smiles", smiles: mol.smiles, label: mol.smiles },
          button
        )
      );
      action.appendChild(button);
      row.appendChild(action);

      tbody.appendChild(row);
    });
    table.appendChild(tbody);

    const caption = document.createElement("caption");
    caption.className = "score-caption";
    const s = msg.summary || {};
    let text = `${s.n_valid ?? "?"}/${s.n_molecules ?? msg.molecules.length} valid`;
    if (typeof s.mean_total_score === "number")
      text += ` · mean ${fmt(s.mean_total_score)}`;
    if (typeof s.best_total_score === "number")
      text += ` · best ${fmt(s.best_total_score)}`;
    if (msg.truncated_at) text += ` · showing first ${msg.truncated_at} rows`;
    caption.textContent = text;
    table.appendChild(caption);

    wrap.appendChild(table);
    append(wrap);

    // Structure paths are deduplicated upstream and are NOT row-aligned,
    // so they get their own strip rather than per-row buttons.
    if (msg.structures && msg.structures.length) {
      const strip = div("structure-strip");
      strip.appendChild(div("muted", "structures produced this run:"));
      msg.structures.forEach((path) => {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = path.split("/").pop();
        button.title = path;
        button.addEventListener("click", () =>
          loadStructure({ source: "shared", path: path, label: path }, button)
        );
        strip.appendChild(button);
      });
      append(strip);
    }
  }

  function sortBy(table, columnIndex) {
    const tbody = table.querySelector("tbody");
    const rows = Array.from(tbody.querySelectorAll("tr"));
    const asc = table.dataset.sortCol === String(columnIndex)
      ? table.dataset.sortAsc !== "true"
      : false;
    rows.sort((a, b) => {
      const x = a.children[columnIndex]?.textContent || "";
      const y = b.children[columnIndex]?.textContent || "";
      const nx = parseFloat(x);
      const ny = parseFloat(y);
      if (!isNaN(nx) && !isNaN(ny)) return asc ? nx - ny : ny - nx;
      return asc ? x.localeCompare(y) : y.localeCompare(x);
    });
    rows.forEach((row) => tbody.appendChild(row));
    table.dataset.sortCol = String(columnIndex);
    table.dataset.sortAsc = String(asc);
  }

  // -------------------------------------------------------------- viewer

  const provenance = el("viewer-provenance");

  async function loadStructure(request, trigger) {
    const label = trigger ? trigger.textContent : "";
    if (trigger) trigger.disabled = true;
    provenance.textContent = "loading…";
    try {
      const res = await fetch("/api/structure", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        provenance.textContent = "could not load: " + (detail.detail || res.status);
        return;
      }
      const payload = await res.json();
      document.querySelector("main").classList.remove("viewer-hidden");
      window.MolViewer.load(payload);
      syncRepCheckboxes(payload.meta || {});
      provenance.textContent = describeProvenance(payload.meta || {}, request);
    } catch (err) {
      provenance.textContent = "could not load: " + err;
    } finally {
      if (trigger) {
        trigger.disabled = false;
        trigger.textContent = label;
      }
    }
  }

  // A viewer that does not say where its coordinates came from is
  // misleading here: some are computed, some are measured.
  function describeProvenance(meta, request) {
    if (meta.source === "rdkit") {
      return `${request.label || meta.canonical_smiles || "molecule"} · computed conformer (${meta.method})`;
    }
    if (meta.source === "rcsb") {
      return `${meta.pdb_id} · experimental structure (RCSB PDB)`;
    }
    if (meta.source === "shared") {
      return `${meta.path} · from this run's shared volume`;
    }
    return "loaded";
  }

  function syncRepCheckboxes(meta) {
    const reps = window.MolViewer.defaultReps(meta);
    el("rep-protein").checked = reps.protein;
    el("rep-protein").disabled = !meta.has_protein;
    el("rep-ligand").checked = reps.ligand;
    el("rep-water").checked = reps.water;
  }

  function currentReps() {
    return {
      protein: el("rep-protein").checked,
      ligand: el("rep-ligand").checked,
      water: el("rep-water").checked,
    };
  }

  ["rep-protein", "rep-ligand", "rep-water"].forEach((id) =>
    el(id).addEventListener("change", () =>
      window.MolViewer.applyRepresentation(currentReps())
    )
  );

  el("viewer-reset").addEventListener("click", () => window.MolViewer.reset());
  el("viewer-spin").addEventListener("click", () => window.MolViewer.toggleSpin());
  el("viewer-toggle").addEventListener("click", () =>
    document.querySelector("main").classList.toggle("viewer-hidden")
  );
  el("pdb-load").addEventListener("click", () => {
    const id = el("pdb-input").value.trim();
    if (id) loadStructure({ source: "pdb_id", pdb_id: id, label: id });
  });
  el("pdb-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter") el("pdb-load").click();
  });
  el("viewer-download").addEventListener("click", () => {
    const payload = window.MolViewer.currentPayload();
    if (!payload) return;
    const blob = new Blob([payload.data], { type: "text/plain" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "structure." + payload.format;
    link.click();
    URL.revokeObjectURL(link.href);
  });

  // ---------------------------------------------------------- connection

  function setBusy(value) {
    busy = value;
    el("send").disabled = value;
    el("stop").hidden = !value;
  }

  function handle(msg) {
    switch (msg.type) {
      case "session":
        session = msg;
        el("session-meta").textContent =
          `${msg.backend} · ${msg.model} · ${msg.tools.length} tools` +
          (msg.multi_agent ? " · driver+reviewer" : "") +
          (msg.web_tools ? " · web" : "");
        break;
      case "replay_start":
        transcript.replaceChildren();
        addStatus(`— replaying ${msg.count} earlier events —`);
        break;
      case "replay_end":
        addStatus("— live —");
        break;
      case "user_echo":
        currentTurn = msg.turn_id;
        addUser(msg.text);
        break;
      case "assistant_message":
        if (msg.text) addAssistant(msg.agent, msg.text);
        break;
      case "thinking":
        addThinking(msg.text);
        break;
      case "tool_call":
        addToolCall(msg);
        break;
      case "tool_result":
        addToolResult(msg);
        break;
      case "score_table":
        addScoreTable(msg);
        break;
      case "builtin_activity":
        addBuiltinActivity(msg);
        break;
      case "status":
        addStatus(msg.text, msg.level);
        break;
      case "error":
        addStatus(msg.text, "error");
        break;
      case "busy":
        setBusy(msg.busy);
        statusbar.textContent = msg.busy ? "working…" : "";
        break;
      case "turn_end":
        if (msg.tokens) {
          el("tokens").textContent =
            `${msg.tokens.total} tokens (${msg.tokens.input} in / ${msg.tokens.output} out)`;
        }
        statusbar.textContent = `turn finished in ${(msg.elapsed_ms / 1000).toFixed(1)}s`;
        break;
      case "session_reset":
        transcript.replaceChildren();
        toolCards.clear();
        break;
      case "monitor_alert":
        addStatus("job alert: " + JSON.stringify(msg.alert), "warn");
        break;
      case "monitor_response":
        addAssistant(msg.agent, msg.text);
        break;
      case "coordinator_turn":
        addStatus(`${msg.agent} turn ${msg.turn}`);
        break;
      default:
        break;
    }
  }

  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    socket = new WebSocket(`${proto}://${location.host}/ws`);
    socket.addEventListener("message", (event) => handle(JSON.parse(event.data)));
    socket.addEventListener("close", () => {
      addStatus("connection lost — retrying in 2s", "warn");
      setBusy(false);
      setTimeout(connect, 2000);
    });
  }

  function send(text) {
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    socket.send(JSON.stringify({ type: "user_message", text: text }));
  }

  el("composer").addEventListener("submit", (event) => {
    event.preventDefault();
    const text = input.value.trim();
    if (!text || busy) return;
    send(text);
    input.value = "";
  });

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      el("composer").requestSubmit();
    }
  });

  el("stop").addEventListener("click", () => {
    socket?.send(JSON.stringify({ type: "stop", turn_id: currentTurn }));
  });

  window.MolViewer.init(el("viewer"));
  connect();
})();
