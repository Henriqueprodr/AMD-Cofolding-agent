# Fleet Handover

How to run and monitor the multi-agent supervisor fleet on the internal
staging cluster. Written for someone picking this up without having deployed
it — it covers the live state, the daily operations, and the failure modes
that have actually happened.

Deeper references: [`AGENTS.md`](AGENTS.md) for the command-by-command
playbook, [`k8s/README.md`](k8s/README.md) for the topology and why it is
shaped this way, [`README.md`](README.md) for the agent and its tool
catalogue, [`TESTING.md`](TESTING.md) for what is covered by tests.

---

## 1. State as of 6 November 2026

Everything below is deployed and healthy. The ledger was finalized on this
date, so there are **no live campaigns** — the pool is idle and ready.

### Cluster

`k8s.staging.silogen.ai`, RKE2 v1.34.1, five Ready nodes. Namespace is
**`demo-lux`**.

| Node | CPU | Memory | MI300X | GPUs free |
|---|---:|---:|---:|---:|
| `useocpm2m-silogen-007` | 224 | 2.0 TiB | 8 | **1** |
| `useocpm2m-silogen-017` | 112 | 2.0 TiB | 8 | **0** |
| `useocpm2m-silogen-int-004` | 16 | 57 GiB | 0 | — |
| `useocpm2m-silogen-int-005` | 16 | 63 GiB | 0 | — |
| `useocpm2m-silogen-int-020` | 16 | 94 GiB | 0 | — |

The fleet is pinned to **`007`**, where it holds five GPUs: four Boltz
replicas plus the REINVENT runner. The other three GPUs on that node belong to
`aimsb-cofolding-agent-test` and `llm-test`, which are not ours.

Both AIM tiers run on **`017`**, one GPU each. The remaining six GPUs there
belong to other namespaces, including a four-GPU job in `demo`. There is no
room to move AIM, and nothing enforces its placement — see §7.

### What is running

| Deployment | Replicas | GPU |
|---|---:|---:|
| `dd-agentic` (interactive agent) | 1 | 0 |
| `dd-agentic-conductor` | 1 | 0 |
| `dd-agentic-supervisor` | **6** | 0 |
| `dd-agentic-campaign-store` | 1 | 0 |
| `dd-agentic-boltz` | 4 | 4 |
| `dd-agentic-reinvent-runner` | 1 | 1 |
| `dd-agentic-boltz-gateway` / `-proxy` | 1 each | 0 |
| `dd-agentic-admet` | 3 | 0 |
| `dd-agentic-posebusters` | 1 | 0 |
| `dd-agentic-interaction-profiler` | 1 | 0 |
| `dd-agentic-retrodmta` | 1 | 0 |

Two AIM tiers, both `Running`:

| Tier | Model | Tuned for | Dialled by |
|---|---|---|---|
| `dd-agentic-aim-conductor` | `openai/gpt-oss-120b` | latency | the conductor |
| `dd-agentic-aim-supervisor` | `Qwen/Qwen3-32B-FP8` | throughput | supervisors, interactive agent |

They are split because a conductor takes one turn at a time with a person
waiting, while N supervisors grind through tool calls with nobody watching any
single response. Serving both from one latency-tuned endpoint does not work at
concurrency.

**Not deployed in this namespace:** `structure-prep`, `annotation`, and
`synthesis-planner`. Manifests exist for all three. A TTP that references one
of them will score `NaN` rather than fail — see §7.

---

## 2. Access

The kubeconfig at `~/.kube/config` uses an OIDC exec plugin
(`kubectl oidc-login` against `kc.silogen.ai`). It works from a cached token.
Two quirks worth knowing so you do not chase them:

- Every command prints `memcache.go ... couldn't get current server API group
  list` warnings. They are noise from discovery caching; ignore them.
- Occasional genuine `Unauthorized` failures resolve on an immediate retry.
  Running `kubectl oidc-login get-token` by hand fails with
  `unauthorized_client` — do not use that to test your access. Use
  `kubectl get nodes`.

Set this in any shell you work from:

```bash
export PROJECT=demo-lux
```

Confirm your grant:

```bash
kubectl auth can-i --namespace demo-lux create deployments
kubectl auth can-i --namespace demo-lux scale deployments
```

---

## 3. The mental model

A **campaign** is one goal explored in parallel. The conductor opens it, and
the ledger holds a queue of **tasks** — each one a weight vector over the
campaign's objectives. Each supervisor claims a task on a lease, applies the
weights to a variant of the base TTP, runs its own RL job on the shared
runner, posts the achieved objective profile, and claims the next one. What
the campaign produces is those results, one per task.

```
conductor ──opens/steers──> campaign-store (ledger, SQLite, 1 replica)
                                  ▲ claim / heartbeat / result
                                  │
                          supervisor ×N  (CPU only, no GPU)
                                  │
                                  ├──> reinvent-runner  (RL jobs, 1 GPU)
                                  └──> boltz-gateway ──> proxy ──> boltz ×4
```

Three things follow from this that are easy to get wrong:

- **Supervisors hold no GPU.** Scaling the pool is a CPU-only operation. GPU
  pressure comes entirely from Boltz and the runner.
- **A supervisor is the agent**, same image, different entrypoint
  (`python -m reinvent4_agent.supervisor`). There is no second codebase.
- **Leases, not assignments.** A supervisor that dies mid-task gives the task
  back when its lease expires (default 3600 s), without anyone noticing the
  crash. `TASK_MAX_ATTEMPTS: 2` stops a task that fails for its own reasons
  from cycling through the whole pool.

---

## 4. Running a campaign

### Open one

```bash
kubectl --namespace "${PROJECT}" exec -it deployment/dd-agentic-conductor -- \
  python -m reinvent4_agent.conductor --start \
    --goal "improve BRD4 BD1 affinity without losing drug-likeness" \
    --base-ttp brd4-bd1-lead-opt \
    --concurrency 4 --hours 4 --grace-hours 1 \
    --task-budget-hours 0.5 --max-steps 60
```

This blocks — it is the conductor loop, not a submit command. Run it in a
terminal you can leave open, or detach it, and monitor from a second shell.

To attach to a campaign that is already open, drop `--start` and its flags. It
conducts the only live campaign, and **refuses to guess when several are
live** — name the campaign id in that case.

Useful flags: `--tick-seconds` changes how often the loop reconsiders;
`--no-scale` stops it touching the supervisor Deployment's replica count.

### Base TTPs available

| TTP | v | Objectives |
|---|:-:|---|
| `brd4-bd1-lead-opt` | 3 | `MolecularWeight`, `logD`, `BoltzAffinity` |
| `brd4-bd1-allscore` | 2 | `BoltzAffinity`, `PoseQuality`, `HBonds`, `Caco2`, `Solubility`, `logD`, `QED`, `MolecularWeight`, `SAScore` |
| `brd4-bd1-series-a` | 2 | `Boltz_Affinity`, `PoseBusters`, `QED`, `MW`, `LogP`, `HBD`, `HBA`, `RotBonds`, `SA_Score` |
| `brd4-bd1-series-b` | 2 | series-a plus `Caco2_permeability`, `Aqueous_solubility` |
| `brd4_bd1_test` | — | smoke-test profile |

`brd4-bd1-series-a-v2` also exists as a variant.

**All of these reference only services that are deployed** — boltz,
posebusters, interaction-profiler and admet, plus native components that run
inside the runner. So the `NaN` risk in §7 does not apply to them as written;
it applies to any new TTP that reaches for structure-prep, annotation or the
synthesis planner.

Note that objective names are **not** consistent across TTPs
(`BoltzAffinity` vs `Boltz_Affinity`, `SAScore` vs `SA_Score`). A weight
override naming an objective the base TTP does not have fails the task when a
supervisor picks it up, so read the objectives before steering by hand.

For a first run use **`brd4-bd1-lead-opt`**: three objectives, one external
scorer plus ADMET, and the shortest path to a first set of results.
`brd4-bd1-allscore` is the full nine-objective profile, and its last two
campaigns failed every task — see §7 for why.

### Set the concurrency

There are **two numbers and they have to agree**:

| Number | Where | What it does |
|---|---|---|
| `concurrency` | on the campaign, in the ledger | what the conductor *targets* — it tops the queue up until `queued + claimed == N` |
| `replicas` | the `dd-agentic-supervisor` Deployment | how many supervisors exist to claim that work |

Normally you set only the first and the conductor reconciles the second on its
next tick, through a `deployments/scale` grant scoped by name to the
supervisor Deployment alone. If that reconciliation ever returns `403`, set
both by hand:

```bash
kubectl --namespace "${PROJECT}" scale deployment/dd-agentic-supervisor --replicas=4
```

If they disagree the campaign still runs, just at the smaller of the two, with
the difference sitting in `queued`.

**The ceiling here is 6.** The runner's `max_concurrent_jobs` is 6, sized
against four Boltz replicas folding one batch at a time. Past that, surplus
supervisors have their tasks refused with `Maximum concurrent jobs (6) already
active` and burn their retries doing it.

### Bring the fleet up from scratch

```bash
k8s/scripts/fleet-up.sh --namespace "${PROJECT}" --supervisors 4
```

Deployments have no dependency graph, so this script is it: it scales each
tier in order and waits on `kubectl rollout status` between them, gated on
each service's own `/health`. Order matters — a supervisor that starts before
the runner is ready claims a task, cannot reach the runner, and burns one of
its two attempts.

---

## 5. Monitoring

### The one command to run

```bash
k8s/scripts/campaign-status.sh --namespace "${PROJECT}"            # once
k8s/scripts/campaign-status.sh --namespace "${PROJECT}" --watch 30 # every 30s
```

It prints both halves, because **capacity and activity are different
questions**. `kubectl get deploy` says how many supervisors *exist*; the
ledger says how many are actually holding a task. An idle supervisor is a
running pod with nothing claimed, and in `kubectl get pods` it looks exactly
like a busy one.

It also joins each in-flight task to its RL job and current step, and prints
the reason for every failed task straight from the ledger — most failures
never reach a pod log at all.

A healthy campaign has those two numbers equal. The two ways they diverge:

- **`claimed` below the pod count** — supervisors are idle. Either the queue
  is empty, or the conductor is holding dispatch back because the Boltz queue
  is deep. It logs `holding dispatch:` when it does.
- **`queued` above zero with every pod busy** — the campaign wants more
  concurrency than the pool has pods. That is the two numbers disagreeing.

### Everything else

```bash
# all supervisors at once
kubectl --namespace "${PROJECT}" logs --selector app.kubernetes.io/name=dd-agentic-supervisor \
  --prefix --tail=20 --follow

# the scoring chain
kubectl --namespace "${PROJECT}" exec deployment/dd-agentic -c dd-agent -- python -c \
  "import urllib.request;print(urllib.request.urlopen('http://dd-agentic-boltz-gateway:8091/health').read().decode())"

# the runner's job slots
kubectl --namespace "${PROJECT}" exec deployment/dd-agentic -c dd-agent -- python -c \
  "import urllib.request;print(urllib.request.urlopen('http://dd-agentic-reinvent-runner:8086/health').read().decode())"
```

The gateway should report `healthy_replicas: 4` and `upstream_reachable: true`.
The runner reports `active_jobs` against `max_concurrent_jobs`.

### Reading a result

A profile carries only the objectives its scorers actually reported. **A
profile missing an objective, or an empty one with "no per-objective scores
were readable" in the summary, means a broken scorer, not a bad run.** This is
the single most important thing to know when reading a campaign that looks
like it produced nothing.

---

## 6. Winding down

```bash
# stop dispatching, let in-flight tasks finish
close_campaign    # MCP tool, or POST /campaigns/{id}/close

# close the books and abandon anything outstanding
finalize_campaign # MCP tool, or POST /campaigns/{id}/finalize
```

The conductor does this itself at the hard stop. If you do it by hand, know
that **abandoning a task in the ledger does not stop its RL job**. The job
lives in the runner and holds a GPU until something tells it to stop; the
first live wind-down left two running for exactly this reason. Check
`active_jobs` on the runner afterwards and stop anything left over.

Order matters: close the ledger first so a supervisor cannot post a result
against a finished campaign, and only then stop the jobs.

To take the fleet down entirely:

```bash
k8s/scripts/fleet-up.sh --namespace "${PROJECT}" --down
```

`--down` takes the runner with it, and RL jobs live in the runner's pod —
anything still running is **killed, not checkpointed**. Stop the jobs first if
the work matters. AIM is left alone either way; it is reconciled by the
AIMServices, so scaling their Deployments would simply be undone.

---

## 7. Known failure modes

**Every result is missing the same objective.** A scorer is down or not
deployed. The scoring component turns a connection error into `NaN` for every
molecule — correct for a flaky service mid-campaign, but with one down for a
whole run you get a completed job full of `NaN` and a geometric mean built on
them. A failure that produces results rather than an error. This is why
`fleet-up.sh` starts every CPU scorer regardless of which objectives the
campaign uses. Remember that structure-prep, annotation and synthesis-planner
are **not** deployed here.

**`the agent did not start a run ... (verdict: single-agent)`.** A task
carrying an `intent` turns off the de novo fallback: normally a supervisor
whose agent finishes without starting a run starts one itself, but an intent
may have asked for a different generative prior, and a substituted run would
post into the same front while measuring something else. So the task fails
instead. Two tasks in the last `brd4-bd1-allscore` campaign died this way. For
a first test, enqueue without an intent.

**A pod stuck in `ContainerCreating` with `Multi-Attach error`.** The pod
landed on a node that cannot attach `dd-agentic-data`, because the claim is
`ReadWriteOnce` and another node holds it. It will never start — this is not a
transient pull. The Deployment is missing its node pin, which means a manifest
was applied with `kubectl apply` instead of `k8s/apply.sh --node`. Patch the
`nodeSelector` back on; the recipe is in
[`k8s/README.md`](k8s/README.md#a-pod-stuck-in-containercreating-with-a-multi-attach-error).

**An AIM tier drifts onto the Boltz node.** `AIMService` has no placement
field, and a `nodeSelector` patched onto the predictor Deployment is stripped
by the operator within about 20 seconds. What places AIM is the scheduler,
which means capacity at the moment the AIMService is created. This has already
happened once: raising the conductor's context recreated its predictor when
`007` had exactly one GPU free, and it landed there. The only symptom is Boltz
quietly having one fewer replica to scale into. Check with:

```bash
kubectl --namespace "${PROJECT}" get pods \
  --selector app.kubernetes.io/component=inference \
  --output custom-columns='NAME:.metadata.name,NODE:.spec.nodeName'
```

Making it durable needs cluster-admin — either a Kyverno mutation injecting
the `nodeSelector`, or a taint on `007` that the pinned Deployments tolerate
and AIM does not. Worth asking for.

**Scaling Boltz looks like it did nothing.** A new replica turns `UP` in the
gateway about 30 s *after* the pod is Ready — 10 s DNS hold plus
`rise 2 inter 10s`. Checking immediately shows the old count.

---

## 8. Capacity, in one place

- **6 supervisors** is the ceiling, set by the runner's `max_concurrent_jobs`.
- **1 free GPU** on `007`. Enough for a fifth Boltz replica if folding turns
  out to be the bottleneck, and nothing beyond that.
- **0 free GPUs** on `017`. AIM cannot be resized or recreated safely without
  first freeing capacity, and recreating it risks §7.
- **Boltz is the measured bottleneck**, not agent count. Both replicas were
  observed at 100% while a third job waited, which is why the GPU budget
  favours Boltz over more AIM tiers.
