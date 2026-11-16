#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A fake agent gateway for developing the UI without a GPU.

Speaks the same contract as ``reinvent4_agent.web.server`` — the same
WebSocket events, the same artifact endpoint — but the answers are canned and
the "structures" are small PDB files generated here.  That makes the whole
front end, including the 3D viewer and the tables, testable on a laptop.

Run it::

    pip install fastapi uvicorn websockets
    python services/ui/mock/mock_gateway.py

then start the UI with ``npm run dev`` in services/ui/frontend.  The Vite
proxy already points at this port.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

logger = logging.getLogger("mock-gateway")

MODEL = "Qwen/Qwen3-32B (mock)"
TOOLS = [
    "list_scoring_components",
    "list_available_models",
    "score_molecules",
    "run_optimization",
    "get_top_candidates",
    "predict_admet",
    "profile_interactions",
]

#: Ligands used by the canned answers.  Real SMILES, so the 2D drawer has
#: something meaningful to render.
LIGANDS = [
    ("JQ1-like", "CC1=C(C)c2c(s1)-c1nnc(C)n1[C@@H](CC(=O)OC(C)(C)C)c1ccc(Cl)cc1-2"),
    ("Fragment A", "O=C(Nc1ccc(F)cc1)c1cccnc1"),
    ("Fragment B", "CN1CCN(c2ccc(C(=O)Nc3ccccc3)cc2)CC1"),
    ("Fragment C", "OC(=O)c1ccc(-c2ccccc2)cc1"),
    ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("Caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
]

SHARED = Path(tempfile.gettempdir()) / "cofolding-mock-shared"


def _ago(minutes: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat(timespec="seconds")


#: One conversation, so the Campaign tab has something to be about: its
#: campaigns are read by session id, and a browser with none asks for
#: nothing.
MOCK_SESSION_ID = "s-20261109-090000-a1b2c3"

#: Runs started in the conversation itself rather than by a sub-agent. The
#: Campaign tab draws these as squares beside the sub-agents' circles, so the
#: mock needs both to be worth looking at.
MOCK_OWN_JOBS = [
    {"job_id": "80999001", "ttp_id": "ttp-jq1-baseline", "ttp_version": 2, "job_type": "optimization",
     "state": "completed", "created_at": _ago(140), "started_at": _ago(139), "finished_at": _ago(118),
     "max_steps": 50, "has_checkpoint": True},
    {"job_id": "80999002", "ttp_id": "ttp-jq1-polarity", "ttp_version": 1, "job_type": "optimization",
     "state": "failed", "created_at": _ago(115), "started_at": _ago(114), "finished_at": _ago(112),
     "max_steps": 50, "has_checkpoint": False, "error": "the ADMET scorer timed out"},
]


#: A campaign with no base TTP, which is the usual way round: each task
#: writes its own profile from the conductor's intent.
MOCK_CAMPAIGN = {
    "campaign_id": "camp-mock-1",
    "goal": "Improve on the JQ1 series against BRD4 without losing selectivity",
    "base_ttp_id": None,
    "objectives": [],
    "concurrency": 3,
    "state": "open",
    "created_at": _ago(95),
    "dispatch_until": None,
    "hard_stop_at": None,
    "finalized_at": None,
    "session_id": MOCK_SESSION_ID,
}


def _task(task_id, state, minutes, intent, *, kind="design", title=None, context=None,
          deliverable=None, owner=None, parents=(), job=None, error=None):
    spec = {"kind": kind, "intent": intent}
    if title:
        spec["title"] = title
    if context:
        spec["context"] = context
    if deliverable:
        spec["deliverable"] = deliverable
    if parents:
        spec["derived_from"] = list(parents)
    return {
        "task_id": task_id,
        "campaign_id": MOCK_CAMPAIGN["campaign_id"],
        "spec": spec,
        "state": state,
        "attempts": 1,
        "created_at": _ago(minutes + 2),
        "lease_owner": owner,
        "lease_expires_at": None,
        "claimed_at": None if owner is None else _ago(minutes),
        "finished_at": None,
        "error": error,
        "job_id": job,
    }


#: Tasks of several kinds, because a campaign is a team being given work and
#: only some of it is a design run.
MOCK_TASKS = [
    _task("t-aaa1", "done", 90, "Hold the JQ1 triazolodiazepine core and vary the pendant aryl",
          title="Vary the pendant aryl", owner="sup-1", job="81000001"),
    _task("t-aaa2", "done", 86, "Scaffold-hop away from the diazepine; keep the acetyl-lysine mimic",
          title="Scaffold hop", owner="sup-2", job="81000002"),
    _task("t-aaa7", "done", 84, "What is known about BRD4 BD1 selectivity over BD2, and which chemotypes achieve it?",
          kind="research", title="BD1 over BD2 selectivity",
          context="We are working the JQ1 series and have no selectivity data of our own.",
          deliverable="The chemotypes that achieve it, with references.", owner="sup-3"),
    _task("t-aaa3", "failed", 60, "Push logP below 3 on the best of the first round",
          title="Drop the logP", owner="sup-1", parents=("t-aaa1",), error="the run stopped at step 4"),
    _task("t-aaa4", "done", 48, "Merge what held affinity in the first task with the polarity of the second",
          title="Merge the two series", owner="sup-3", parents=("t-aaa1", "t-aaa2"), job="81000004"),
    _task("t-aaa8", "done", 40, "Plan a route to the merged series' best compound and check the building blocks are buyable",
          kind="retrosynthesis", title="Route to the merged lead", owner="sup-1", parents=("t-aaa4",)),
    _task("t-aaa5", "claimed", 12, "Grow into the ZA channel from the merged series",
          title="Grow into the ZA channel", owner="sup-2", parents=("t-aaa4",), job="81000005"),
    _task("t-aaa6", "queued", 2, "Re-score the merged series with a pose-quality gate",
          kind="analysis", title="Re-score with a pose gate", parents=("t-aaa4",)),
]

MOCK_RESULTS = [
    {
        "task_id": "t-aaa1",
        "campaign_id": MOCK_CAMPAIGN["campaign_id"],
        "job_id": "81000001",
        "objective_profile": {"Boltz affinity": 0.71, "QED": 0.58, "MW": 0.62},
        "candidates": [
            {"smiles": LIGANDS[0][1], "score": 0.74},
            {"smiles": LIGANDS[1][1], "score": 0.61},
        ],
        "summary": "The core tolerates a para-chloro; affinity held and QED slipped.",
        "report": (
            "## Report\n\n"
            "Ran ttp-t-aaa1 for 50 steps against BRD4 BD1 and kept the triazolodiazepine "
            "core fixed, varying only the pendant aryl.\n\n"
            "The para-chloro analogue is the best of the round: Boltz affinity 0.71 with the "
            "core intact. QED slipped to 0.58, which is the halogen and the extra ring, not "
            "a scoring artefact.\n\n"
            "**What I could not settle.** Nothing in the run distinguishes BD1 from BD2 — "
            "the scoring profile has no selectivity term, so a selective and a pan-BET "
            "compound score the same here."
        ),
        "job_ids": ["81000001"],
        "posted_at": _ago(80),
    },
    {
        "task_id": "t-aaa2",
        "campaign_id": MOCK_CAMPAIGN["campaign_id"],
        "job_id": "81000002",
        "objective_profile": {"Boltz affinity": 0.52, "QED": 0.77, "MW": 0.81},
        "candidates": [{"smiles": LIGANDS[2][1], "score": 0.66}, {"smiles": LIGANDS[3][1], "score": 0.49}],
        "summary": "Cleaner properties, a clear step down in affinity.",
        "report": (
            "## Report\n\n"
            "Hopped to a 3,5-dimethylisoxazole head group, which keeps the acetyl-lysine "
            "mimic and drops the diazepine.\n\n"
            "Properties are cleaner across the board — QED 0.77, MW well inside range — and "
            "affinity is a clear step down at 0.52. On this evidence the isoxazole is worth "
            "keeping as the polar half of a merge rather than as a series of its own."
        ),
        "job_ids": ["81000002"],
        "posted_at": _ago(74),
    },
    {
        "task_id": "t-aaa7",
        "campaign_id": MOCK_CAMPAIGN["campaign_id"],
        "job_id": None,
        # Research starts nothing and scores nothing: the report is the whole
        # of what it produced, which is the case the tab has to read well.
        "objective_profile": {},
        "candidates": [],
        "summary": "BD1 selectivity is achievable; the published handles are the ZA channel and the gatekeeper.",
        "report": (
            "## Report\n\n"
            "BD1-selective BET inhibition is established but narrow. Three chemotypes carry "
            "most of the published selectivity:\n\n"
            "- **Dihydropteridinones** — up to 30-fold BD1 over BD2, from a substituent "
            "reaching the ZA channel.\n"
            "- **Tetrahydroquinolines** — around 10-fold, and the selectivity tracks the "
            "gatekeeper contact rather than the acetyl-lysine mimic.\n"
            "- **Isoxazole-azepines** — roughly equipotent; the selectivity reported for "
            "these is within assay error.\n\n"
            "The common handle is the ZA channel: BD1 is wider there. Our JQ1 series makes "
            "no contact in it at all.\n\n"
            "**What I could not settle.** I found no public structure of a BD1-selective "
            "compound bound to BD2, so the selectivity is inferred from potency ratios "
            "rather than seen."
        ),
        "job_ids": [],
        "posted_at": _ago(70),
    },
    {
        "task_id": "t-aaa4",
        "campaign_id": MOCK_CAMPAIGN["campaign_id"],
        "job_id": "81000004",
        "objective_profile": {"Boltz affinity": 0.79, "QED": 0.69, "MW": 0.7},
        "candidates": [{"smiles": LIGANDS[0][1], "score": 0.83}, {"smiles": LIGANDS[4][1], "score": 0.55}],
        "summary": "The merge is the best of the campaign so far on both counts.",
        "report": (
            "## Report\n\n"
            "Merged the chlorophenyl of the first task onto the isoxazole head of the "
            "second, which is the combination the two reports pointed at.\n\n"
            "Best compound of the campaign so far: Boltz affinity 0.79 at QED 0.69, so the "
            "merge recovers most of the affinity the hop cost without giving the properties "
            "back. Both halves earn their place — removing either loses more than it saves.\n\n"
            "**Next.** Nothing here reaches the ZA channel, which the selectivity work says "
            "is where BD1 selectivity comes from. That is the obvious next variation."
        ),
        "job_ids": ["81000004"],
        "posted_at": _ago(35),
    },
    {
        "task_id": "t-aaa8",
        "campaign_id": MOCK_CAMPAIGN["campaign_id"],
        "job_id": None,
        "objective_profile": {},
        "candidates": [],
        "summary": "Three routes under six steps; one is blocked on a building block nobody sells.",
        "report": (
            "## Report\n\n"
            "Three routes to the merged lead, all under six steps:\n\n"
            "1. **Suzuki last** (4 steps) — both partners are catalogue items, £180 for a "
            "gram of the boronate. This is the one to run.\n"
            "2. **Amide last** (5 steps) — cheaper starting materials, but the "
            "isoxazole acid is made in two steps nobody has published a yield for.\n"
            "3. **Convergent, SNAr first** (4 steps) — blocked: the fluoroarene is not "
            "sold by any supplier I can see, and making it is three steps on its own.\n\n"
            "**What I could not settle.** Route 2's yields are a guess. If route 1's "
            "boronate goes out of stock, somebody should check route 2 properly before "
            "committing to it."
        ),
        "job_ids": [],
        "posted_at": _ago(30),
    },
]

MOCK_SUPERVISORS = [
    {
        "owner": "sup-1",
        "state": "idle",
        "campaign_id": MOCK_CAMPAIGN["campaign_id"],
        "task_id": None,
        "tasks_completed": 2,
        "tasks_failed": 1,
        "detail": None,
        "first_seen_at": _ago(95),
        "last_seen_at": _ago(1),
        "last_seen_seconds_ago": 7,
    },
    {
        "owner": "sup-2",
        "state": "working",
        "campaign_id": MOCK_CAMPAIGN["campaign_id"],
        "task_id": "t-aaa5",
        "tasks_completed": 1,
        "tasks_failed": 0,
        "detail": "step 18 of 50",
        "first_seen_at": _ago(92),
        "last_seen_at": _ago(1),
        "last_seen_seconds_ago": 4,
    },
    {
        "owner": "sup-3",
        "state": "gone",
        "campaign_id": MOCK_CAMPAIGN["campaign_id"],
        "task_id": None,
        "tasks_completed": 2,
        "tasks_failed": 0,
        "detail": "finished its task and exited",
        "first_seen_at": _ago(70),
        "last_seen_at": _ago(30),
        "last_seen_seconds_ago": 1800,
    },
]


def _finding(finding_id: str, branch: str, claim: str, minutes: int, notes: list | None = None, **fields) -> dict:
    return {
        "finding_id": finding_id, "campaign_id": MOCK_CAMPAIGN["campaign_id"], "branch": branch,
        "claim": claim, "detail": fields.get("detail"), "evidence": fields.get("evidence", []),
        "confidence": fields.get("confidence", "medium"), "tags": fields.get("tags", []),
        "author": fields.get("author", "sup-1"), "task_id": fields.get("task_id"),
        "created_at": _ago(minutes), "supersedes": None, "superseded_by": None, "notes": notes or [],
    }


# What the campaign knows: one branch settled, one argued over, one the plan
# named that nobody has filed anything under yet.
MOCK_FINDINGS = [
    _finding("f-3", "potency", "A para-chloro on the phenyl holds Boltz affinity at 0.71", 20,
             confidence="high", tags=["sar", "chloro"], task_id="t-aaa1",
             evidence=["81000001", "CHEMBL1957266"],
             detail="Seen in 14 of the top 20 of run 81000001; the unsubstituted core scored 0.64."),
    _finding("f-2", "admet", "The morpholine tail costs QED but not affinity", 45, author="sup-2", tags=["qed"],
             notes=[{"note_id": 1, "finding_id": "f-2", "body": "Run 81000003 lost affinity with it, 0.71 to 0.58.",
                     "stance": "disputes", "author": "sup-3", "task_id": "t-aaa3", "created_at": _ago(30)}]),
    _finding("f-1", "potency", "The triazolo core is needed; swapping it drops affinity below 0.5", 80,
             confidence="high", tags=["sar", "core"], task_id="t-aaa1"),
]

MOCK_ATLAS = {
    "campaign_id": MOCK_CAMPAIGN["campaign_id"],
    "goal": MOCK_CAMPAIGN["goal"],
    "findings": len(MOCK_FINDINGS),
    "branches": [
        {"branch": "potency", "name": "Potency", "in_plan": True, "findings": 2, "disputed": 0,
         "latest": {"finding_id": "f-3", "claim": MOCK_FINDINGS[0]["claim"], "confidence": "high", "created_at": _ago(20)},
         "tags": ["chloro", "core", "sar"]},
        {"branch": "admet", "name": "ADMET", "in_plan": True, "findings": 1, "disputed": 1,
         "latest": {"finding_id": "f-2", "claim": MOCK_FINDINGS[1]["claim"], "confidence": "medium", "created_at": _ago(45)},
         "tags": ["qed"]},
        {"branch": "selectivity", "name": "Selectivity over BRD2/3", "in_plan": True, "findings": 0, "disputed": 0,
         "latest": None, "tags": []},
    ],
    "empty_branches": ["selectivity"],
    "items": MOCK_FINDINGS,
    "truncated": False,
}


def write_sample_structures() -> list[str]:
    """Write small PDB files so the 3D viewer has something real to load."""
    out = SHARED / "boltz_run"
    out.mkdir(parents=True, exist_ok=True)

    paths = []
    for index in range(2):
        path = out / f"complex_{index + 1}.pdb"
        path.write_text(_sample_pdb(seed=index))
        paths.append(str(path))
    return paths


def _sample_pdb(seed: int) -> str:
    """Build a short helix with a small ligand next to it.

    Real coordinates matter here: the viewer has to compute a cartoon and a
    surface, and it cannot do that from placeholder atoms.
    """
    rng = random.Random(seed)
    lines = ["HEADER    MOCK COMPLEX", "REMARK    Generated by mock_gateway.py"]

    serial = 1
    # An alpha helix: 1.5 A rise and 100 degrees of turn per residue.
    import math

    for residue in range(1, 19):
        angle = math.radians(100 * residue)
        z = 1.5 * residue
        for name, radius, offset in (("N", 1.5, -0.4), ("CA", 2.3, 0.0), ("C", 2.0, 0.4), ("O", 2.4, 0.6)):
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            lines.append(
                f"ATOM  {serial:5d}  {name:<3s} ALA A{residue:4d}    "
                f"{x:8.3f}{y:8.3f}{z + offset:8.3f}  1.00 20.00           {name[0]}"
            )
            serial += 1

    # A ligand parked beside the helix, as a HETATM residue.
    for index in range(8):
        x = 6.0 + rng.uniform(-1.2, 1.2)
        y = rng.uniform(-1.2, 1.2)
        z = 12.0 + index * 0.9
        lines.append(
            f"HETATM{serial:5d}  C{index + 1:<2d} LIG B{1:4d}    " f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 25.00           C"
        )
        serial += 1

    lines.append("END")
    return "\n".join(lines) + "\n"


def make_app() -> FastAPI:
    app = FastAPI(title="Mock cofolding agent gateway")
    structures = write_sample_structures()
    logger.info("Sample structures written to %s", SHARED)

    @app.get("/api/health")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok", "mcp": "ready", "tools": len(TOOLS)})

    @app.get("/api/session")
    async def session() -> dict:
        return {
            "backend": "openai",
            "model": MODEL,
            "endpoint": "mock",
            "tools": TOOLS,
            "ready": True,
        }

    # -- one conversation -----------------------------------------------------

    def _summary() -> dict:
        return {
            "id": MOCK_SESSION_ID,
            "title": "BRD4 — improving on JQ1",
            "created": _ago(150),
            "updated": _ago(1),
            "turns": 6,
            "busy": False,
            "agents_allowed": 3,
            "runs": {
                "running": 0,
                "total": len(MOCK_OWN_JOBS),
                "jobs": [{"id": job["job_id"], "state": job["state"], "launched": True} for job in MOCK_OWN_JOBS],
            },
        }

    @app.get("/api/sessions")
    async def sessions() -> list[dict]:
        return [_summary()]

    @app.get("/api/sessions/{session_id}/events")
    async def session_events(session_id: str) -> dict:
        return {"session": _summary(), "events": []}

    @app.post("/api/tools/{name}")
    async def tool(name: str) -> dict:
        """Only the reads the workbench makes on its own, without the agent."""
        if name == "list_jobs":
            return {"name": name, "result": MOCK_OWN_JOBS, "panels": []}
        if name == "get_job_alerts":
            return {"name": name, "result": [], "panels": []}
        raise HTTPException(status_code=403, detail=f"{name} is not a direct tool in the mock")

    # -- the fleet ------------------------------------------------------------
    # A campaign part way through, so the Campaign tab has a shape to be
    # developed against: tasks at different stages, lineage between them, and
    # results from the ones that finished.

    @app.get("/api/fleet/capacity")
    async def fleet_capacity() -> dict:
        return {
            "max_supervisors": 6,
            "binding_limit": "jobs",
            "detail": "the job backend is the binding limit at 6 (mock)",
        }

    @app.get("/api/fleet/campaigns")
    async def fleet_campaigns(session: str | None = None) -> list[dict]:
        return [MOCK_CAMPAIGN]

    @app.get("/api/fleet/campaigns/{campaign_id}")
    async def fleet_campaign(campaign_id: str) -> dict:
        if campaign_id != MOCK_CAMPAIGN["campaign_id"]:
            raise HTTPException(status_code=404, detail=f"No campaign {campaign_id}")
        return {
            "campaign": MOCK_CAMPAIGN,
            "tasks": MOCK_TASKS,
            "results": MOCK_RESULTS,
            "supervisors": MOCK_SUPERVISORS,
        }

    @app.get("/api/fleet/campaigns/{campaign_id}/atlas")
    async def fleet_atlas(campaign_id: str) -> dict:
        if campaign_id != MOCK_CAMPAIGN["campaign_id"]:
            raise HTTPException(status_code=404, detail=f"No campaign {campaign_id}")
        return MOCK_ATLAS

    @app.get("/api/artifacts/{path:path}")
    async def artifact(path: str) -> FileResponse:
        candidate = (SHARED / path).resolve()
        if not candidate.is_relative_to(SHARED.resolve()) or not candidate.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found")
        return FileResponse(candidate, media_type="chemical/x-pdb")

    @app.websocket("/api/chat")
    async def chat(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_json({"type": "ready", "model": MODEL, "backend": "openai", "tools": TOOLS})
        # Which conversation this socket is on. Without it the browser has no
        # session id, and everything read per conversation — the campaigns
        # above among them — asks for nothing.
        await websocket.send_json({"type": "session", "lastSeq": 0, **_summary()})

        try:
            while True:
                raw = await websocket.receive_text()
                message = json.loads(raw)
                if message.get("type") == "auto":
                    minutes = message.get("minutes") or 0
                    until = datetime.now(timezone.utc) + timedelta(minutes=minutes) if minutes else None
                    await websocket.send_json({"type": "auto", "until": until.isoformat() if until else None})
                    continue
                if message.get("type") != "user_message":
                    continue
                await _reply(websocket, message.get("content", ""), structures)
        except WebSocketDisconnect:
            logger.info("Mock client disconnected")

    return app


async def _reply(websocket: WebSocket, prompt: str, structures: list[str]) -> None:
    """Play back a plausible turn, with pauses so streaming is visible."""
    lowered = prompt.lower()
    await websocket.send_json({"type": "status", "state": "busy"})
    await asyncio.sleep(0.4)

    await websocket.send_json(
        {
            "type": "thinking",
            "content": (
                "The user asked about a design campaign. I should check which "
                "scoring components exist before proposing a run, then score "
                "the candidates and report the affinities."
            ),
        }
    )
    await asyncio.sleep(0.5)

    if "available" in lowered or "component" in lowered or "what" in lowered:
        await _tool(
            websocket,
            "list_scoring_components",
            {},
            {"components": ["QED", "MolecularWeight", "Boltz2Affinity", "ADMET", "SlogP"]},
        )
        await _answer(
            websocket,
            "There are five scoring components available:\n\n"
            "- **QED** — drug likeness\n"
            "- **MolecularWeight** and **SlogP** — physicochemical filters\n"
            "- **Boltz2Affinity** — cofolded binding affinity\n"
            "- **ADMET** — the toxicity and pharmacokinetics models\n\n"
            "Tell me a target and I will start a campaign.",
        )
        return

    if "admet" in lowered or "toxicity" in lowered:
        rows = [
            {
                "smiles": smiles,
                "name": name,
                "logP": round(random.uniform(1.0, 4.5), 3),
                "solubility": round(random.uniform(-5.5, -1.5), 3),
                "hERG_risk": round(random.uniform(0.02, 0.6), 3),
            }
            for name, smiles in LIGANDS[:4]
        ]
        await _tool(websocket, "predict_admet", {"smiles": [s for _, s in LIGANDS[:4]]}, {"results": rows})
        await _answer(
            websocket,
            "ADMET predictions are in the table above. **Fragment B** has the "
            "highest hERG risk, so I would drop it. The other three stay inside "
            "a normal lead-like range.",
        )
        return

    # Default: a full design + cofold + score turn.
    await _tool(
        websocket,
        "run_optimization",
        {"target": "BRD4", "n_steps": 50, "scoring": "Boltz2Affinity+QED"},
        {"job_id": "job-mock-001", "state": "completed", "molecules_generated": 412},
    )
    await asyncio.sleep(0.4)

    selected = LIGANDS[:4]
    await _tool(
        websocket,
        "score_molecules",
        {"smiles": [s for _, s in selected], "protein_sequence": "MSAESGPGTRLRNLPVMGDGLETSQMSTTQAQAQ…"},
        {
            "scores": [0.87, 0.64, 0.52, 0.41],
            "smiles": [s for _, s in selected],
            "metadata": {
                "affinity_pred_value": [-9.4, -8.1, -7.2, -6.6],
                "qed": [0.71, 0.66, 0.58, 0.49],
                "name": [n for n, _ in selected],
            },
            "artifacts": {"structures": structures},
        },
    )

    await _answer(
        websocket,
        "I ran a 50-step REINVENT campaign against **BRD4** and cofolded the "
        "four best candidates with Boltz.\n\n"
        "The strongest is the JQ1-like scaffold at a predicted affinity of "
        "**-9.4 kcal/mol** with a QED of 0.71 — a good balance of potency and "
        "drug likeness. Its complex is in the viewer; the ligand sits in the "
        "acetyl-lysine pocket.\n\n"
        "Next I would check the ADMET profile of the top two. Ask me to do that.",
    )


async def _tool(websocket: WebSocket, name: str, arguments: dict, result: dict) -> None:
    """Send a tool call, wait, then its result — as the real gateway does."""
    call_id = f"call-{random.randint(1000, 9999)}"
    await websocket.send_json({"type": "tool_call", "id": call_id, "name": name, "arguments": arguments})
    await asyncio.sleep(random.uniform(0.6, 1.4))

    raw = json.dumps(result, indent=2)
    # The mock imports the real classifier when it is importable, so the UI is
    # exercised against the same panel logic that runs in the real gateway.
    panels = _panels(name, raw)
    await websocket.send_json(
        {
            "type": "tool_result",
            "id": call_id,
            "name": name,
            "result": raw,
            "truncated": False,
            "isError": False,
            "panels": panels,
        }
    )


def _panels(name: str, raw: str) -> list[dict]:
    try:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from reinvent4_agent.web import artifacts

        artifacts.SHARED_ROOT = SHARED
        return artifacts.extract_panels(name, raw)
    except Exception:
        logger.warning("Real classifier unavailable; sending no panels", exc_info=True)
        return []


async def _answer(websocket: WebSocket, text: str) -> None:
    await asyncio.sleep(0.5)
    await websocket.send_json(
        {
            "type": "assistant_message",
            "content": text,
            "inputTokens": random.randint(2000, 6000),
            "outputTokens": random.randint(300, 900),
        }
    )
    await websocket.send_json({"type": "status", "state": "idle"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock agent gateway for UI development.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s: %(message)s")

    import uvicorn

    uvicorn.run(make_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
