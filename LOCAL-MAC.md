# Running locally (Mac/laptop, no GPU, no API key)

Upstream this project targets a Slurm cluster with AMD ROCm GPUs: the
images are built on `rocm/pytorch`, and Boltz2 cofolding, PoseBusters
rescoring and the RL runs all assume real GPUs. None of that exists on a
laptop. This fork adds a second, smaller way to run it:

| Piece | Where it runs | Notes |
|---|---|---|
| REINVENT4 + `dd-mcp` | container (`dd-agent:local`, CPU, linux/arm64) | native scoring, TTPs, UniProt/PubChem lookups |
| `dd-agent` CLI | host virtualenv | so it can use the `claude` CLI you are logged in to |
| Boltz2, PoseBusters, ADMET, RetroDMTA, Interaction Profiler | **not started** | GPU and/or licensed model weights |

The agent reaches the MCP server over stdio through `docker exec`, so it
behaves exactly as it does on the cluster, minus the external services.
TTP scoring falls back to native-only components when they are missing.

## 1. Prerequisites

- Docker (OrbStack is fine) running
- A Claude subscription, with `claude` installed and logged in
- Python >= 3.10 on the host

## 2. Run it

```bash
bash services/local/run-local.sh          # terminal chat
bash services/local/run-local.sh --web    # browser UI + 3D viewer
```

The first run builds the image (REINVENT4 + CPU torch + priors, ~4 GB,
10-20 min) and creates `.venv-agent/`. Later runs start in seconds.

```bash
# one-shot instead of a chat session
echo "Score aspirin CC(=O)OC1=CC=CC=C1C(=O)O for drug-likeness" \
  | bash services/local/run-local.sh

# Driver + Reviewer (uses roughly twice the tokens)
bash services/local/run-local.sh --multi-agent
```

Stop the background MCP container with `docker rm -f dd-mcp-local`.

While editing the MCP server, `DD_MCP_DEV_MOUNT=1` bind-mounts the source
read-only over the image's installed copy, so changes take effect on the next
`docker exec` instead of needing an image rebuild.

## 2b. The browser UI

`--web` serves `http://127.0.0.1:8765`: the same agent, with tool calls and
score tables rendered as HTML and a 3D viewer beside the chat. The terminal
CLI is unchanged and both can run at once (each gets its own MCP session).

The viewer has three sources of geometry, and always says which one it used:

| Source | Where it comes from | Works without a GPU |
|---|---|---|
| `3D` button on a score row | RDKit conformer (ETKDGv3 + MMFF), computed in the container | yes |
| PDB ID box | the experimental entry from RCSB | yes |
| "structures produced this run" | a file on `/shared`, i.e. a Boltz cofold output | needs a GPU run first |

That distinction matters: a computed conformer shows a molecule's shape and
says nothing about how it binds. Only the third source is a real pose, and on
this Mac nothing produces one — drop a complex PDB into `data/shared/` to
exercise that path.

The server binds loopback only and has **no authentication**; anything beyond
`127.0.0.1` needs `--allow-remote` and is on you. Requests are additionally
checked against an `Origin` allowlist, because binding to loopback does not
stop another page in your browser from posting to it.

## 2c. What it looks things up with

Chemistry facts come from PubChem, ChEMBL and UniProt; Claude's web search is
the "what's new" layer only, and `run_python` does the arithmetic. See
[Grounding and source precedence](README.md#grounding-and-source-precedence)
in the main README — the short version is that the agent looks a SMILES up
every time rather than recalling one, and names its source next to every
number.

## 3. The `claude-code` backend (no API key)

`--backend claude-code` runs turns through the Claude Code harness
(`claude-agent-sdk`), which drives the `claude` binary you are already
signed in to — the same way T3 Code and other agent front-ends use an
existing Claude login. Your credentials stay with the CLI: this code
never reads `~/.claude`, the keychain, or any token file.

```bash
dd-agent --backend claude-code              # host, inherits your login
dd-agent --backend claude-code --model opus # CLI model aliases
```

Headless or in a container there is no interactive login to inherit, so
mint a token once and pass it in:

```bash
claude setup-token
export CLAUDE_CODE_OAUTH_TOKEN="<token it prints>"
```

Implementation: [`reinvent4_agent/backends/claude_code_backend.py`](reinvent4_agent/backends/claude_code_backend.py).
`dd-agent` keeps its own agentic loop — the backend hands the SDK an
in-process MCP server whose tool handlers park each call, surface it to
`dd-agent` as a normal `ToolCall`, and wait for the result. Tool
scoping, the Rich display, session logs and the Driver/Reviewer handoff
all work unchanged.

## 4. What does not work here

- **Binding poses.** Nothing on this machine predicts how a ligand sits in a
  pocket — that is Boltz2, and it needs a GPU. The viewer's RDKit conformers
  are molecular shape only.
- `run_optimization` / RL: the prior is in the image, but CPU-only
  REINVENT training is impractically slow.
- Anything needing Boltz2, PoseBusters, ADMET, RetroDMTA or the
  Interaction Profiler: the first two need GPUs, the next two need the
  licensed weights archive. They are disabled in
  `reinvent4_mcp_config.local.yaml`.
- `docker compose up`: the compose stack is ROCm/linux-amd64 and stays
  the cluster path.
