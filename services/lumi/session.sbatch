#!/bin/bash
#SBATCH --job-name=cofolding-session
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# One LUMI-G node running the multi-agent stack for the browser UI:
#   GCD 0-1  vLLM serving Gemma 4 31B on 127.0.0.1:8001 -- the tier a person
#            talks to, and the tier the conductor plans campaigns with
#   GCD 2-3  vLLM serving Gemma 4 31B on 127.0.0.1:8002 -- the supervisor tier,
#            many turns with nobody waiting on any one of them
#   GCD 4    REINVENT (dd-mcp inside dd-web)
#   GCD 5-7  Boltz
#   :8000    dd-web, the gateway the browser UI talks to (token required)
#   :8082-91 the scoring and tool services, on the same ports as the cluster
#
# Two instances of the same model rather than one shared one: a supervisor pool
# generating flat out would otherwise sit in front of the person's next turn,
# and the cluster separates the tiers for the same reason. Each is tensor
# parallel over two GCDs because the weights are 59 GB and a single 64 GB GCD
# would leave almost nothing for the KV cache.
#
# CHAT_MODEL=gptoss serves gpt-oss-120b on the chat tier instead. It works on
# these GPUs -- see services/lumi/vllm_serve.sh for the MoE backend that makes
# MXFP4 run on gfx90a -- but it is slower here than Gemma and is not the default.
#
# Supervisors are not started here. They are separate CPU jobs the conductor
# submits and cancels as the campaign's queue grows and drains -- that is what
# reinvent4_agent.pool's Slurm scaler does, and supervisor.sbatch is what it
# submits. They need no GPU of their own: a supervisor drives REINVENT through
# the runner and thinks with Gemma over HTTP.
#
# Submit from the repository root (see services/lumi/README.md):
#   mkdir -p logs && sbatch --account=<project> services/lumi/session.sbatch

set -uo pipefail
REPO=${REPO:-$SLURM_SUBMIT_DIR}
[[ -f $REPO/services/singularity/run.sh ]] || { echo "Submit from the repository root (got $REPO)"; exit 1; }
JOB=$SLURM_JOB_ID
NODE=$(hostname)
export DATA_DIR=${DATA_DIR:-$REPO/data}
LOGS=$REPO/logs/session-$JOB
mkdir -p "$LOGS"

# --- The two model tiers ----------------------------------------------------
# Named by role rather than by model so swapping a checkpoint is one variable.
CHAT_MODEL=${CHAT_MODEL:-gemma}           # what people and the conductor use
SUPERVISOR_MODEL=${SUPERVISOR_MODEL:-gemma}
CHAT_PORT=8001
SUPERVISOR_PORT=8002
CHAT_GPUS=${CHAT_GPUS:-0,1}
SUPERVISOR_GPUS=${SUPERVISOR_GPUS:-2,3}

case "$CHAT_MODEL" in
gptoss) CHAT_MODEL_ID=openai/gpt-oss-120b ;;
gemma)  CHAT_MODEL_ID=google/gemma-4-31B-it ;;
qwen)   CHAT_MODEL_ID=Qwen/Qwen3.8-27B ;;
*)      echo "CHAT_MODEL must be gptoss, gemma or qwen"; exit 1 ;;
esac
SUPERVISOR_MODEL_ID=google/gemma-4-31B-it

export REINVENT4_AGENT_BACKEND=openai
export REINVENT4_AGENT_API_KEY=local-vllm
export REINVENT4_AGENT_BASE_URL=http://127.0.0.1:$CHAT_PORT/v1
export REINVENT4_AGENT_MODEL=$CHAT_MODEL_ID
export REINVENT4_AGENT_LOG_TIMESTAMPS=1
# Boltz keeps 5-7; REINVENT gets 4. Boltz runs one worker until this branch
# carries the worker-pool modules, so 6 and 7 are headroom rather than load.
export REINVENT_GPU=4 BOLTZ_GPU=5 BOLTZ_GPUS=${BOLTZ_GPUS:-5} DD_WEB_PORT=8000
# On LUMI the GPUs are visible without --rocm, which would mount the host's older ROCm.
export APPTAINER_GPU_FLAG=""
# Run the agent from this checkout, so a code change needs a gateway restart, not a rebuild.
export DEV_SOURCE=1

# The laptop tunnel reaches dd-web through a login node, so every LUMI user
# could too: dd-web listens on the node but refuses requests without the token.
export DD_WEB_HOST=0.0.0.0
TOKEN_FILE=$HOME/.dd-web-token
[[ -s $TOKEN_FILE ]] || (umask 077; openssl rand -hex 24 > "$TOKEN_FILE")
REINVENT4_WEB_TOKEN=$(cat "$TOKEN_FILE")
export REINVENT4_WEB_TOKEN
AUTH=(-H "X-DD-Token: $REINVENT4_WEB_TOKEN")

echo "Node $NODE  job $JOB  start $(date +%T)"
echo "chat=$CHAT_MODEL_ID on GCD $CHAT_GPUS   supervisors=$SUPERVISOR_MODEL_ID on GCD $SUPERVISOR_GPUS"

# --- Service discovery ------------------------------------------------------
# Slurm has no DNS, so the URLs the tools read cannot be service names. The
# registry is how a supervisor on some other node finds these; the conductor
# and the supervisor both apply it before reading any URL, and on Kubernetes
# applying it is a no-op because the environment is already complete.
REGISTRY=$DATA_DIR/service-registry.json
python3 - "$REGISTRY" "$NODE" <<'PY'
import json, pathlib, sys
registry, node = pathlib.Path(sys.argv[1]), sys.argv[2]
ports = {
    "posebusters": 8082, "interaction_profiler": 8083,
    "campaign_store": 8087, "synthesis": 8088,
    "structure_prep": 8089, "annotation": 8090, "boltz": 8091,
}
registry.parent.mkdir(parents=True, exist_ok=True)
part = registry.with_suffix(".part")
part.write_text(json.dumps({k: f"http://{node}:{p}" for k, p in ports.items()}, indent=2))
part.replace(registry)
print(f"registry at {registry} -> {node}")
PY

# What a supervisor job needs to know to join this session. Written as a file
# rather than passed through --export so a supervisor submitted an hour later
# still picks up the right node.
SESSION_ENV=$DATA_DIR/session.env
cat > "$SESSION_ENV" <<EOF
# Written by session.sbatch for job $JOB on $NODE. Sourced by supervisor.sbatch.
export SESSION_NODE=$NODE
export SESSION_JOB=$JOB
export REPO=$REPO
export DATA_DIR=$DATA_DIR
export DD_SERVICE_REGISTRY=$REGISTRY
export SUPERVISOR_BASE_URL=http://$NODE:$SUPERVISOR_PORT/v1
export SUPERVISOR_MODEL_ID=$SUPERVISOR_MODEL_ID
export CHAT_BASE_URL=http://$NODE:$CHAT_PORT/v1
export CHAT_MODEL_ID=$CHAT_MODEL_ID
EOF

# The conductor runs inside this allocation and submits supervisors from it.
export DD_SERVICE_REGISTRY=$REGISTRY
export SUPERVISOR_JOB_NAME=${SUPERVISOR_JOB_NAME:-dd-supervisor}
export SUPERVISOR_SBATCH=$REPO/services/lumi/supervisor.sbatch
export SLURM_ACCOUNT=${SLURM_JOB_ACCOUNT:-}
export CAMPAIGN_STORE_URL=http://127.0.0.1:8087

# --- The two vLLM tiers -----------------------------------------------------
# Serving both at once rather than swapping: the conductor's turn and its
# supervisors' turns overlap by design, and a swap would serialise them.
serve() {  # model gpus port host logfile
    VLLM_MODEL=$1 VLLM_GPUS=$2 VLLM_PORT=$3 VLLM_HOST=$4 \
        bash "$REPO/services/lumi/vllm_serve.sh" > "$5" 2>&1 &
    echo $!
}
# The chat tier only ever has clients inside this allocation -- dd-web and the
# conductor -- so it stays on the loopback. The supervisor tier cannot: its
# clients are pool jobs on other nodes, which means it is reachable by every
# LUMI user while the session runs. See the security note in the README.
CHAT_PID=$(serve "$CHAT_MODEL" "$CHAT_GPUS" "$CHAT_PORT" 127.0.0.1 "$LOGS/vllm-chat.log")
SUP_PID=$(serve "$SUPERVISOR_MODEL" "$SUPERVISOR_GPUS" "$SUPERVISOR_PORT" 0.0.0.0 "$LOGS/vllm-supervisor.log")

bash "$REPO/services/singularity/run.sh" --vendor amd --web > "$LOGS/stack.log" 2>&1 &
STACK=$!

trap 'kill $STACK $CHAT_PID $SUP_PID 2>/dev/null; pkill -f "vllm serve" 2>/dev/null; wait' EXIT TERM

wait_for() {  # name url pid
    for _ in $(seq 1 240); do
        curl -sf "${AUTH[@]}" "$2" >/dev/null && { echo "$1 up at $(date +%T)"; return 0; }
        kill -0 "$3" 2>/dev/null || { echo "$1 exited early, see $LOGS"; return 1; }
        sleep 10
    done
    echo "$1 not up after 40 min, see $LOGS"; return 1
}
wait_for "vLLM chat ($CHAT_MODEL)" "http://127.0.0.1:$CHAT_PORT/health" "$CHAT_PID" || exit 1
wait_for "vLLM supervisor ($SUPERVISOR_MODEL)" "http://127.0.0.1:$SUPERVISOR_PORT/health" "$SUP_PID" || exit 1
wait_for dd-web http://127.0.0.1:8000/api/health "$STACK" || exit 1

curl -s -o /dev/null -w "without token: HTTP %{http_code}\n" "http://$NODE:8000/api/session"
echo "READY node=$NODE"
echo "  start a campaign with:"
echo "    srun --overlap --jobid=$JOB bash $REPO/services/lumi/conduct.sh <campaign_id>"
wait $STACK
