<!--
Copyright Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Run the agent and its UI on LUMI

One LUMI-G node runs everything: two vLLM tiers, the agent gateway, REINVENT,
Boltz and the other scoring services. Your laptop serves the UI and reaches
the node through an SSH tunnel.

Both tiers serve Gemma 4 31B, on two GCDs each:

| GCDs | |
|------|--|
| 0-1 | the chat tier, :8001 — what a person talks to, and what the conductor plans with |
| 2-3 | the supervisor tier, :8002 — many turns, nobody waiting on any one of them |
| 4 | REINVENT |
| 5-7 | Boltz |

They are separate instances of the same model rather than one shared one
because a supervisor pool generating flat out would otherwise sit in front of
the person's next turn. Each is tensor parallel over two GCDs: the weights are
59 GB and a single 64 GB GCD leaves almost nothing for the KV cache.

Supervisors are not part of this job. The conductor submits them as separate
CPU jobs and cancels them as the campaign's queue grows and drains — they
think with the Gemma tier over HTTP and drive REINVENT through the runner, so
a GPU of their own would only sit idle.

## You need

- A LUMI project with GPU hours (`standard-g`) and about 100 GB free on its
  scratch.
- SSH access, with an alias `lumi` in `~/.ssh/config`:

  ```
  Host lumi
      HostName lumi.csc.fi
      User <your LUMI username>
      IdentityFile ~/.ssh/<your key>
  ```

- On the laptop: git, and Node 20.19+ or 22.12+.

## 1. Set up once (on LUMI)

Clone onto scratch, not `$HOME`; the home quota is too small.

```bash
cd /scratch/<project>/$USER
git clone https://github.com/silogen/AMD-Cofolding-agent.git
cd AMD-Cofolding-agent
```

Restore the gitignored dependencies (REINVENT4, Boltz, PoseBusters and the
model weights archive) as described in
[Setup, step 1](../../README.md#1-restore-the-gitignored-dependencies).

Then build the images and download the model in one CPU job (about an hour):

```bash
mkdir -p logs && HF_TOKEN=<token> sbatch --account=<project> services/lumi/build.sbatch
```

`logs/cofolding-build-<job>.out` lists each target as OK or FAILED. Gemma is a
gated repository, hence `HF_TOKEN`; if someone on the project has already
pulled it, point `HF_HOME` at their cache instead and drop the `gemma` target.

The building-block catalog is not part of the build: it is 1.3 GB and lives
outside the images. Without it the synthesis planner still answers, but every
building block comes back `unknown` rather than purchasable. Stage it once:

```bash
mkdir -p data/aizynth-stock && curl -L -o data/aizynth-stock/zinc_stock.hdf5 \
  https://zenodo.org/api/records/11430881/files/zinc_stock.hdf5/content
```

## 2. Start a session (on LUMI)

```bash
sbatch --account=<project> services/lumi/session.sbatch
```

The job runs for 4 hours (`--time=` to change it). It is ready when
`logs/cofolding-session-<job>.out` says `READY`, after 10 to 20 minutes.

## 3. Open the UI (on the laptop)

Clone the same repository on the laptop, then from its root:

```powershell
.\services\lumi\start-ui.ps1        # Windows
```

```bash
bash services/lumi/start-ui.sh      # macOS, Linux
```

The script waits for `READY`, opens the tunnel, builds the UI and serves it
on <http://localhost:5173>. Behind a TLS-inspecting proxy, run `npm ci` once in
`services/ui/frontend` with `NODE_OPTIONS=--use-system-ca` first.

## Day to day

| To | Run on LUMI |
|----|-------------|
| Stop | `scancel <job>` |
| Restart only the gateway, after a code change | `srun --overlap --jobid=<job> pkill -f '[d]d-web --host'` |
| Run a campaign | `srun --overlap --jobid=<job> bash services/lumi/conduct.sh <campaign_id>` |
| See the supervisor pool | `squeue -u $USER --name dd-supervisor` |
| Chat with gpt-oss-120b instead | `sbatch --export=ALL,CHAT_MODEL=gptoss ...` (build target `gptoss` first) |
| Share model weights across a project | set `HF_HOME` to a project directory for both jobs |

A gateway restart ends the chat turn and any REINVENT run in progress, so
restart when nothing is running; queued runs start again afterwards.
Conversations, jobs and results are kept in `data/`; the logs of each
session are in `logs/session-<job>/`.

## Security

- The gateway listens on the node's network so the tunnel can reach it, which
  means every LUMI user can connect to it. It refuses any request without the
  access token in `~/.dd-web-token`, which `session.sbatch` creates readable by
  you only. The laptop's proxy adds the token; it never reaches the browser.
  Do not share it.
- The chat tier listens on `127.0.0.1` only; its clients are all inside the
  allocation.
- The supervisor tier (`:8002`) listens on the node, because the pool jobs that
  use it run on other nodes. Every LUMI user can reach it while the job runs.
- The scoring services (ports 8082 to 8091) have no access control, so other
  LUMI users can reach them while the job runs.
