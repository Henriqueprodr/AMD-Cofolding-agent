# AGENTS.md

Operational playbook for executing the multi-agent fleet and running drug
discovery tests against the **current Kubernetes deployment**. Every command
here has been run against the live cluster.

For the topology and why it is shaped this way, see
[`k8s/README.md`](k8s/README.md). For a narrative handover, see
[`FLEET-HANDOVER.md`](FLEET-HANDOVER.md). For the test catalogue, see
[`TESTING.md`](TESTING.md).

---

## 1. Setup

```bash
export PROJECT=demo-lux
cd /path/to/AMD-Cofolding-agent
```

Cluster is `k8s.staging.silogen.ai` (RKE2 v1.34.1), namespace `demo-lux`,
fleet pinned to node `useocpm2m-silogen-007`.

Container names differ per Deployment and are required on `exec`:

| Deployment | `--container` |
|---|---|
| `dd-agentic` | `dd-agent` |
| `dd-agentic-conductor` | `dd-conductor` |
| `dd-agentic-supervisor` | `dd-supervisor` |

### The auth quirk — read this before debugging a failure

The kubeconfig uses an OIDC exec plugin. Two behaviours will waste your time:

- Every command prints `memcache.go ... couldn't get current server API group
  list`. **Harmless.** Filter with `2>&1 | grep -v memcache.go`.
- Commands intermittently fail with `error: You must be logged in to the
  server (Unauthorized)`. **Retry — it succeeds within a few attempts.** Do
  not interpret this as a broken cluster or a lost grant. Running
  `kubectl oidc-login get-token` by hand fails with `unauthorized_client` and
  is *not* a valid test of access; use `kubectl get nodes`.

Wrap anything important:

```bash
kretry() {
  for i in 1 2 3 4 5; do
    out=$("$@" 2>&1)
    if ! echo "$out" | grep -q "Unauthorized"; then echo "$out"; return 0; fi
    sleep 5
  done
  echo "$out"; return 1
}
```

---

## 2. Preflight

Run this before any campaign. It is the difference between a failed run and a
run that completes full of `NaN`.

```bash
# every Deployment ready?
kubectl -n "${PROJECT}" get deploy

# the scoring chain: expect healthy_replicas 4, upstream_reachable true
kubectl -n "${PROJECT}" exec deployment/dd-agentic -c dd-agent -- python -c \
  "import urllib.request;print(urllib.request.urlopen('http://dd-agentic-boltz-gateway:8091/health').read().decode())"

# the RL runner: expect active_jobs 0, max_concurrent_jobs 6
kubectl -n "${PROJECT}" exec deployment/dd-agentic -c dd-agent -- python -c \
  "import urllib.request;print(urllib.request.urlopen('http://dd-agentic-reinvent-runner:8086/health').read().decode())"

# can we dispatch, and if not which limit says no?
kubectl -n "${PROJECT}" exec deployment/dd-agentic-conductor -c dd-conductor -- python -c \
  "from reinvent4_mcp.tools.campaign import get_pool_capacity; print(get_pool_capacity())"
```

`get_pool_capacity` is the single most useful preflight call. A healthy idle
fleet returns:

```
{'can_dispatch': 'yes', 'binding_limit': None,
 'supervisors': {'idle': 6, 'busy': 0, 'alive': 6},
 'scorer': {'queue_pressure': 0.0, 'verdict': 'room', ...}}
```

It reports **both** constraints — idle supervisors *and* Boltz queue room —
and names whichever is binding. Either number alone misleads: an idle
supervisor handed work while Boltz is saturated only moves the wait.

Also confirm no campaign is already live, because the conductor refuses to
attach when several are:

```bash
bash k8s/scripts/campaign-status.sh --namespace "${PROJECT}"
```

---

## 3. Capabilities in the current deployment

### Scorers and tool services

| Service | Deployed | Endpoint | Provides |
|---|:---:|---|---|
| `boltz` ×4 behind gateway | yes | `dd-agentic-boltz-gateway:8091/score` | Cofolding, binding affinity, structure conditioning |
| `posebusters` | yes | `dd-agentic-posebusters:8082/score` | Pose validity / `PoseQuality` |
| `interaction-profiler` | yes | `dd-agentic-interaction-profiler:8083/score` | Interface metrics, `HBonds`, fragments |
| `admet` ×3 | yes | `dd-agentic-admet:8084/score` | `Caco2`, `Solubility`, `logD`, 17 drug + 25 agro endpoints |
| `retrodmta` | yes | `dd-agentic-retrodmta:8085/score` | Legacy + Mpro/CHIKV/EV-A71/FatA target models |
| `reinvent-runner` | yes | `dd-agentic-reinvent-runner:8086` | RL jobs, native scoring, priors |
| `campaign-store` | yes | `dd-agentic-campaign-store:8087` | The fleet ledger |
| `structure-prep` | **no** | `dd-agentic-structure-prep:8089` | Receptor prep, ligand extraction |
| `synthesis-planner` | **no** | `dd-agentic-synthesis-planner:8088` | AiZynthFinder routes, stock checks |
| `annotation` | **no** | `dd-agentic-annotation:8090` | Cached UniProt/InterPro proxy |

**The last three have manifests but are not running here.** A TTP that
references one will score `NaN` for every molecule rather than failing — the
scoring component turns a connection error into `NaN`, which is right for a
flaky service mid-campaign and catastrophic for a whole run. You get a
completed job full of `NaN` and a geometric mean built on them.

Native components (`QED`, `MolecularWeight`, `SAScore`, `logP`, HBD/HBA,
rotatable bonds) need no service — they run inside the runner.

### LLM tiers

| Tier | Model | Tuned for | Used by |
|---|---|---|---|
| `dd-agentic-aim-conductor` | `openai/gpt-oss-120b` | latency | conductor |
| `dd-agentic-aim-supervisor` | `Qwen/Qwen3-32B-FP8` | throughput | supervisors, interactive agent |

### Base TTPs

| TTP | v | Objectives | All services deployed? |
|---|:-:|---|:-:|
| `brd4-bd1-lead-opt` | 3 | `MolecularWeight`, `logD`, `BoltzAffinity` | yes |
| `brd4-bd1-allscore` | 2 | `BoltzAffinity`, `PoseQuality`, `HBonds`, `Caco2`, `Solubility`, `logD`, `QED`, `MolecularWeight`, `SAScore` | yes |
| `brd4-bd1-series-a` | 2 | `Boltz_Affinity`, `PoseBusters`, `QED`, `MW`, `LogP`, `HBD`, `HBA`, `RotBonds`, `SA_Score` | yes |
| `brd4-bd1-series-b` | 2 | series-a plus `Caco2_permeability`, `Aqueous_solubility` | yes |
| `brd4_bd1_test` | — | smoke-test profile | yes |

Objective names are **not** consistent across TTPs (`BoltzAffinity` vs
`Boltz_Affinity`, `SAScore` vs `SA_Score`). A weight override naming an
objective the base TTP does not have fails the task when a supervisor picks it
up. Always read the objectives first:

```bash
kubectl -n "${PROJECT}" exec deployment/dd-agentic -c dd-agent -- python -c "
import json; d=json.load(open('/root/.reinvent4_mcp/ttps/brd4-bd1-lead-opt.json'))
print([o['name'] for o in d['scoring']['objectives']])"
```

List all base TTPs (stripping per-task variants and version files):

```bash
kubectl -n "${PROJECT}" exec deployment/dd-agentic -c dd-agent -- sh -c \
  "ls -1 /root/.reinvent4_mcp/ttps | sed 's/_v[0-9]*\.json$//; s/\.json$//' | sed -E 's/-[0-9a-f]{8}$//' | sort -u"
```

---

## 4. Running a fleet campaign

### Step 1 — size the pool

Supervisors hold **no GPU**, so scaling is CPU-only. The ceiling is **6**,
set by the runner's `max_concurrent_jobs`; past that, surplus supervisors have
tasks refused with `Maximum concurrent jobs (6) already active` and burn their
retries doing it.

```bash
kubectl -n "${PROJECT}" scale deployment/dd-agentic-supervisor --replicas=4
kubectl -n "${PROJECT}" rollout status deployment/dd-agentic-supervisor
```

### Step 2 — open the campaign and conduct it

```bash
kubectl -n "${PROJECT}" exec -it deployment/dd-agentic-conductor -c dd-conductor -- \
  python -m reinvent4_agent.conductor --start \
    --goal "improve BRD4 BD1 affinity without losing drug-likeness" \
    --base-ttp brd4-bd1-lead-opt \
    --concurrency 4 --hours 4 --grace-hours 1 \
    --task-budget-hours 0.5 --max-steps 60
```

**This blocks.** It is the conductor loop, not a submit command. Leave it in
its own terminal and monitor from a second shell.

| Flag | Meaning |
|---|---|
| `--start` | open a new campaign instead of attaching |
| `--goal` | what the campaign is for, carried in the ledger |
| `--base-ttp` | the TTP every task varies the weights of |
| `--objectives` | names each result is reported on (default: all the TTP declares) |
| `--concurrency` | supervisors to hold in flight |
| `--hours` / `--grace-hours` | dispatch window / drain window |
| `--task-budget-hours` | per-task wall clock, capped by the hard stop |
| `--max-steps` | RL steps per task |
| `--tick-seconds` | how often the loop reconsiders |
| `--no-scale` | do not touch the supervisor replica count |

To attach to an already-open campaign, drop `--start` and its flags. With no
campaign id it conducts the only live one and **refuses to guess when several
are live** — pass the id positionally in that case.

### Step 3 (alternative) — open without conducting

Opening a campaign dispatches nothing and spends no GPU, so it can be set up
and inspected first:

```bash
kubectl -n "${PROJECT}" exec -i deployment/dd-agentic-conductor -c dd-conductor -- python - <<'PY'
from reinvent4_mcp.tools.campaign import start_campaign
print(start_campaign(
    goal="smoke test: affinity vs size",
    base_ttp_id="brd4-bd1-lead-opt",
    objectives=["BoltzAffinity", "MolecularWeight", "logD"],
    concurrency=2, hours=1.0, grace_hours=0.5,
))
PY
```

### Step 4 — steer by hand

All thirteen campaign tools are importable from
`reinvent4_mcp.tools.campaign` inside the conductor or agent pod. This is the
scripting path; the same tools are what a conductor agent calls
conversationally.

```bash
kubectl -n "${PROJECT}" exec -i deployment/dd-agentic-conductor -c dd-conductor -- python - <<'PY'
from reinvent4_mcp.tools.campaign import enqueue_task, set_concurrency, cancel_supervisor

CID = "<campaign-id>"

# one weight vector by hand, for a trade-off the proposer has not reached
print(enqueue_task(
    CID,
    weight_overrides={"BoltzAffinity": 1.4, "MolecularWeight": 0.6},
    fidelity_tier="exploration",   # or "validation" to bypass the cache read
    max_steps=3,
))

# change how many supervisors the campaign holds in flight
print(set_concurrency(CID, 6))

# take a task back from the supervisor running it
# print(cancel_supervisor("<task-id>"))
PY
```

`fidelity_tier` is worth understanding: `exploration` reads and writes the
scoring cache; `validation` bypasses the read, so a reported result is never a
cached sample.

**Do not pass `intent` on a first test.** A task carrying an `intent` turns off
the de novo fallback — if the agent finishes without starting a run, the task
fails and is retried rather than being substituted. Two tasks in the last
`brd4-bd1-allscore` campaign died exactly this way with
`the agent did not start a run ... (verdict: single-agent)`.

### The two numbers that must agree

| Number | Where | What it does |
|---|---|---|
| `concurrency` | on the campaign, in the ledger | what the conductor *targets* — tops the queue up until `queued + claimed == N` |
| `replicas` | the `dd-agentic-supervisor` Deployment | how many supervisors exist to claim that work |

The conductor reconciles the second from the first via a `deployments/scale`
grant scoped by name to the supervisor Deployment. If that ever returns `403`,
set both by hand. If they disagree the campaign runs at the smaller of the
two, with the difference sitting in `queued`.

---

## 5. Monitoring

```bash
bash k8s/scripts/campaign-status.sh --namespace "${PROJECT}"             # once
bash k8s/scripts/campaign-status.sh --namespace "${PROJECT}" --watch 30  # loop
bash k8s/scripts/campaign-status.sh --namespace "${PROJECT}" --campaign <id>
```

This is the primary tool. It prints pool capacity *and* ledger activity,
joins each in-flight task to its RL job and current step, and prints the
reason for every failed task straight from the ledger — most failures never
reach a pod log at all.

**Capacity and activity are different questions.** `kubectl get deploy` says
how many supervisors exist; the ledger says how many hold a task. An idle
supervisor is a running pod with nothing claimed and looks identical to a busy
one in `kubectl get pods`. A healthy campaign has the two numbers equal:

- **`claimed` below the pod count** — supervisors idle. Queue empty, or the
  conductor is holding dispatch back because the Boltz queue is deep (it logs
  `holding dispatch:`).
- **`queued` above zero with every pod busy** — the campaign wants more
  concurrency than the pool has pods.

Supporting views:

```bash
# all supervisors at once
kubectl -n "${PROJECT}" logs --selector app.kubernetes.io/name=dd-agentic-supervisor \
  --prefix --tail=20 --follow

# the conductor's reasoning
kubectl -n "${PROJECT}" logs deployment/dd-agentic-conductor --tail=50 --follow

# per-supervisor state, lease remaining, lifetime task counts
kubectl -n "${PROJECT}" exec deployment/dd-agentic-conductor -c dd-conductor -- python -c \
  "from reinvent4_mcp.tools.campaign import list_supervisors; print(list_supervisors())"

# gateway replica view and queue depth
kubectl -n "${PROJECT}" port-forward deployment/dd-agentic-boltz-proxy 8404:8404 &
curl -s "http://localhost:8404/stats;csv" | cut -d, -f1,2,18
```

A newly scaled Boltz replica turns `UP` in the gateway about **30 s after the
pod is Ready** (10 s DNS hold plus `rise 2 inter 10s`). Checking immediately
shows the old count and looks like a failure.

---

## 6. Harvesting results

```bash
kubectl -n "${PROJECT}" exec -i deployment/dd-agentic-conductor -c dd-conductor -- python - <<'PY'
from reinvent4_mcp.tools.campaign import (
    get_campaign_state, get_supervisor_result,
    get_campaign_files, list_campaigns,
)

CID = "<campaign-id>"
print(get_campaign_state(CID))     # counts, who holds what, time left
print(get_campaign_files(CID))     # where each task's output is on disk
# print(get_supervisor_result(CID, "<task-id>"))   # one task: profile + molecules
PY
```

A profile carries only the objectives its scorers actually reported. A result
with molecules, an empty profile and "no per-objective scores were readable" in
its summary means a broken scorer, not a bad run.

The objective profile is the best *single molecule's* scores — not the best
value seen per objective, which would mix molecules and describe a compound
nobody made.

---

## 7. Winding down

```bash
kubectl -n "${PROJECT}" exec -i deployment/dd-agentic-conductor -c dd-conductor -- python - <<'PY'
from reinvent4_mcp.tools.campaign import close_campaign, finalize_campaign
CID = "<campaign-id>"
print(close_campaign(CID))      # stop dispatching, let in-flight finish
print(finalize_campaign(CID))   # close the books, abandon anything outstanding
PY
```

The conductor does this itself at the hard stop. Doing it by hand, two things
matter:

- **Order.** Close the ledger first so a supervisor cannot post a result
  against a finished campaign, and only then stop jobs.
- **Abandoning a task does not stop its RL job.** The job lives in the runner
  and holds a GPU until told to stop; the first live wind-down left two
  running for this reason. Always check afterwards:

```bash
kubectl -n "${PROJECT}" exec deployment/dd-agentic -c dd-agent -- python -c \
  "import urllib.request;print(urllib.request.urlopen('http://dd-agentic-reinvent-runner:8086/health').read().decode())"
```

`active_jobs` must be `0`. Then release the pool:

```bash
kubectl -n "${PROJECT}" scale deployment/dd-agentic-supervisor --replicas=0
```

Whole fleet up or down, in dependency order with readiness waits between
tiers:

```bash
bash k8s/scripts/fleet-up.sh --namespace "${PROJECT}" --supervisors 4
bash k8s/scripts/fleet-up.sh --namespace "${PROJECT}" --down
```

`--down` takes the runner with it, and RL jobs live in the runner's pod —
anything still running is **killed, not checkpointed**. AIM is left alone
either way; it is reconciled by the AIMServices.

---

## 8. Single-agent work (no fleet)

The interactive Driver + Reviewer agent, for exploratory drug discovery rather
than a parallel sweep:

```bash
kubectl -n "${PROJECT}" exec --stdin --tty deployment/dd-agentic --container dd-agent -- dd-agent
```

In-session commands: `/help`, `/review [job_id]` (summon the Reviewer — multi-
agent mode only), `/reset`, `/exit`. This pod runs with
`REINVENT4_AGENT_MULTI_AGENT=true`, so the Reviewer is active and the
background campaign monitor runs every `--monitor-interval` seconds.
`--no-multi-agent` runs the Driver alone and roughly halves token spend.

Supervisors and the conductor are deliberately Driver-only.

Scripted scoring smoke test through the same pod (pipes two fixed prompts and
requires the tool-rendered score tables, not just the model's prose):

```bash
bash k8s/scripts/uat-scoring.sh --namespace "${PROJECT}"
```

---

## 9. Verification harnesses

These prove properties that are easy to assume and easy to get wrong. Run
them after any change to placement, storage, or the scoring chain.

```bash
# Boltz writes structures to /shared in pod A; PoseBusters and the profiler
# re-open those absolute paths from pods B and C. Proves the cross-pod handoff.
./k8s/scripts/verify-artifact-handoff.py --namespace "${PROJECT}"

# Concurrent batches land on idle replicas rather than stacking on a busy one,
# and surplus requests queue rather than fail.
./k8s/scripts/verify-boltz-placement.py --namespace "${PROJECT}"

# Scorer latency and cache hit rate, harvested from external_scoring.jsonl.
# Prior campaigns have already measured what the fleet is sized on.
./k8s/scripts/analyze-scoring-latency.py

# Time one Boltz /score call with and without the optimized kernels.
./k8s/scripts/ab-boltz-kernels.py --namespace "${PROJECT}"

# Point an AIM tier at another catalog model (--tier is required).
./k8s/scripts/switch-aim-model.sh --tier supervisor --model <name>
```

Local test suite — no GPU, no cluster, REINVENT4 and the services mocked:

```bash
pip install pytest numpy pydantic PyYAML pandas fastapi httpx requests rdkit gemmi
python -m pytest tests/ -v

python -m pytest tests/test_k8s_manifests.py -v   # the manifest invariants
python -m pytest tests/test_conductor.py tests/test_supervisor.py tests/test_campaign_store.py -v
```

There is no test CI — only CodeQL. Run the suite yourself.

---

## 10. Hard limits

| Limit | Value | Set by |
|---|---|---|
| Supervisors | **6** | runner's `max_concurrent_jobs` |
| Boltz replicas | 4 (5th possible) | 1 free GPU on node `007` |
| Free GPUs, node `007` | **1** | 8 total, 7 in use |
| Free GPUs, node `017` | **0** | both AIM tiers plus other namespaces |
| Task lease | 3600 s | outlives a whole RL step |
| Task attempts | 2 | `TASK_MAX_ATTEMPTS` |
| Gateway upstream timeout | 2100 s | one scoring call |

Boltz is the measured bottleneck, not agent count — replicas were observed at
100% while a third job waited.

---

## 11. Rules that must not be broken

Each is asserted by a test, and each has been broken at least once in a way
that produced *results* rather than an error.

**Never `kubectl apply` a manifest directly.** `k8s/apply.sh --node` stamps
`nodeSelector[kubernetes.io/hostname]` onto every Deployment. Without it a pod
lands on a node that cannot attach the `ReadWriteOnce` claim and fails with a
`Multi-Attach` error that never resolves. It bites the supervisor pool first,
because it is the only workload that scales past one replica.

**No GPU Deployment may surge during a rollout.** The surge pod asks for
another `amd.com/gpu` and sits `Pending`. The runner uses `Recreate`; Boltz
uses a rolling update pinned to `maxSurge: 0`.

**No `amd.com/gpu` on the agent, supervisor, or conductor.** REINVENT4 lives
in `reinvent-runner`; that split is what lets agent count and RL capacity
scale independently.

**One campaign-store replica, one conductor.** The atomic claim relies on
SQLite's write lock with `BEGIN IMMEDIATE` — two store replicas would hand the
same weight vector to two supervisors. Two conductors would each top the pool
up to `N` and run the campaign at `2N`, which is the one overcommit the ledger
cannot catch because each dispatch is individually legitimate.

**The supervisor pool ships at `replicas: 0`.** Applying the manifests must
never start agents against an empty queue.

**Supervisors and the conductor hardcode `REINVENT4_AGENT_MULTI_AGENT: false`**
rather than reading the ConfigMap, whose `true` belongs to the interactive
agent. Pointing them at it gives every supervisor a Reviewer that produces no
molecules and doubles token spend.

**The conductor's writable TTP mount and `REINVENT4_AGENT_DENY_TOOLS:
"update_ttp,delete_ttp"` are one decision.** Writable because opening a
campaign on a new target starts with creating a base TTP; the deny-list is
what prevents rewriting a base TTP that tasks are already measured against.
Relax either half alone and the guarantee is gone.

**The Driver cannot end its own run.** `stop_optimization` is withheld from a
supervisor's Driver. The first live run had it poll 13 times in six minutes,
decide the job had stalled, and stop it inside its first step — nothing to
checkpoint, no candidates.

**Boltz is reached through the gateway, never the replicas.** `leastconn`,
`maxconn 1` per replica, `http-reuse never`, and `BOLTZ_MAX_CONCURRENCY=1` to
match. More than one replica is only allowed when the ConfigMap dials the
gateway.

**Images are node-local.** `*:local` tags are imported into one node's kubelet
containerd store; nothing is pushed to a registry. Building on an arm64 Mac
produces arm64 images that import cleanly and crash-loop with
`exec format error`. Build on the GPU node.

**Comments explain why, not what.** Match the surrounding code: a comment
restating the next line is noise; one recording why a number is `6` and not
the node's CPU count is the point. Update [`TESTING.md`](TESTING.md) when you
add a test.
