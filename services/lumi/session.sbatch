#!/bin/bash
#SBATCH --job-name=cofolding-session
# Overridable with `sbatch --account=...`, but not omittable: without one
# Slurm falls to the default association, whose submit limit is zero, and
# refuses the job with AssocMaxSubmitJobLimit -- which reads like a quota
# that has run out rather than an account that was never named.
#SBATCH --account=project_462001170
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
# How the conductor, which cannot run sbatch from inside its container, asks
# for supervisors. Per-session so a stale request cannot resize a new pool.
POOL_DIR=$DATA_DIR/pool-$JOB
mkdir -p "$POOL_DIR"

# --- The two model tiers ----------------------------------------------------
# Named by role rather than by model so swapping a checkpoint is one variable.
CHAT_MODEL=${CHAT_MODEL:-gemma}           # what people and the conductor use
SUPERVISOR_MODEL=${SUPERVISOR_MODEL:-gemma}
CHAT_PORT=8001
SUPERVISOR_PORT=8002
RUNNER_PORT=8086
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

# The MCP config is the other half of discovery: the registry carries the URLs
# the agent's tools dial, this carries the ones the scoring components dial
# from inside an RL job. Generated from the checkout's template so there is one
# place to edit, with the two things only this node knows substituted in.
# Where REINVENT runs execute: "slurm" submits each as its own small-g job
# with a Boltz of its own (run.sbatch), "local" keeps them on this node's GCD 4
# behind the one shared Boltz. The run limit follows, since on this node it
# is what one GCD holds and on Slurm it is what the campaign may queue.
RUN_BACKEND=${RUN_BACKEND:-slurm}
case "$RUN_BACKEND" in
slurm) MAX_CONCURRENT_JOBS=${MAX_CONCURRENT_JOBS:-16} ;;
local) MAX_CONCURRENT_JOBS=${MAX_CONCURRENT_JOBS:-4} ;;
*)     echo "RUN_BACKEND must be slurm or local"; exit 1 ;;
esac
RUN_RELAY_DIR=$DATA_DIR/run-relay-$JOB
RUN_JOB_NAME=${RUN_JOB_NAME:-dd-run}

MCP_CONFIG=$DATA_DIR/reinvent4_mcp_config.session.yaml
sed -e "s|http://localhost:|http://$NODE:|g" \
    -e "/^runner:/,\$ s|enabled: false|enabled: true|" \
    -e "s|max_concurrent_jobs: 2|max_concurrent_jobs: $MAX_CONCURRENT_JOBS|" \
    "$REPO/services/singularity/reinvent4_mcp_config.singularity.yaml" > "$MCP_CONFIG.part"
grep -q "http://$NODE:8086" "$MCP_CONFIG.part" || { echo "config generation missed the runner URL"; exit 1; }
mv "$MCP_CONFIG.part" "$MCP_CONFIG"
export MCP_CONFIG_FILE=$MCP_CONFIG
echo "MCP config at $MCP_CONFIG -> $NODE"

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
export MCP_CONFIG_FILE=$MCP_CONFIG
export SUPERVISOR_POOL_CONTROL=$POOL_DIR
export SUPERVISOR_BASE_URL=http://$NODE:$SUPERVISOR_PORT/v1
export SUPERVISOR_MODEL_ID=$SUPERVISOR_MODEL_ID
# Loopback, unlike every other URL in this file. The others are read by
# supervisors, which run on other nodes; this one is read by the conductor,
# which runs inside this allocation -- and the chat tier binds the loopback
# precisely because nothing outside should reach it. Addressed by node name
# it resolves to the node's network IP and the connection is refused.
export CHAT_BASE_URL=http://127.0.0.1:$CHAT_PORT/v1
export CHAT_MODEL_ID=$CHAT_MODEL_ID
export RUNNER_BASE_URL=http://$NODE:$RUNNER_PORT
export CAMPAIGN_STORE_URL=http://$NODE:8087
EOF

# The conductor runs inside this allocation and asks the relay below to submit
# supervisors; it cannot submit them itself, see pool_relay.
export DD_SERVICE_REGISTRY=$REGISTRY
export SUPERVISOR_JOB_NAME=${SUPERVISOR_JOB_NAME:-dd-supervisor}
export SUPERVISOR_SBATCH=$REPO/services/lumi/supervisor.sbatch
export SLURM_ACCOUNT=${SLURM_JOB_ACCOUNT:-}
export CAMPAIGN_STORE_URL=http://127.0.0.1:8087
# What a supervisor costs and where it runs; must match supervisor.sbatch,
# since this is what the pool ceiling is computed from.
export SUPERVISOR_PARTITION=${SUPERVISOR_PARTITION:-small}
export SUPERVISOR_CPUS=${SUPERVISOR_CPUS:-4}
export SUPERVISOR_POOL_CONTROL=$POOL_DIR
# Empty, not unset: there is no boltz-gateway in this stack, and the two mean
# different things to the conductor's admission control.
export BOLTZ_GATEWAY_STATS_URL=

# --- The supervisor pool relay ----------------------------------------------
# Slurm's clients do not work inside the container: sbatch needs the host's
# munge socket and user database, and bind-mounting them gets as far as
# "Invalid user: $USER" before failing. So the conductor writes the number it
# wants to pool.want and this loop, on the host and inside the allocation,
# makes it true -- the submitting and cancelling that SlurmPoolScaler would
# have done, in the one place that can actually do it.
pool_relay() {
    local want have rows ids waiting running surplus take id n
    while true; do
        want=$(cat "$POOL_DIR/pool.want" 2>/dev/null | tr -dc '0-9')
        want=${want:-0}
        # --sort V is submission order, so the tail of each list below is the
        # youngest work and therefore the cheapest to give back.
        rows=$(squeue --noheader --user "$USER" --name "$SUPERVISOR_JOB_NAME" \
                      --states PENDING,CONFIGURING,RUNNING,COMPLETING,SUSPENDED \
                      --format '%i %T' --sort V 2>/dev/null)
        ids=$(echo "$rows" | awk 'NF {print $1}')
        have=$(echo "$ids" | grep -c '[0-9]' || true)
        if (( want > have )); then
            for ((n = have; n < want; n++)); do
                id=$(sbatch --parsable --job-name "$SUPERVISOR_JOB_NAME" \
                            ${SLURM_ACCOUNT:+--account "$SLURM_ACCOUNT"} \
                            "$SUPERVISOR_SBATCH" 2>&1) \
                    && echo "relay: submitted supervisor $id" \
                    || echo "relay: sbatch failed: $id"
            done
        elif (( want < have )); then
            # A supervisor that is working is asked to stop, not cancelled.
            # It finishes its task and exits on SIGTERM, which takes far
            # longer than the ~30s scancel allows before killing it, so
            # cancelling one destroys the RL run it is holding. --signal
            # delivers the signal and leaves the allocation alone, so the job
            # ends when the work does. Only jobs that never started, which
            # hold nothing, are cancelled outright.
            surplus=$((have - want))
            waiting=$(echo "$rows" | awk '$2 == "PENDING" || $2 == "CONFIGURING" {print $1}')
            running=$(echo "$rows" | awk 'NF && $2 != "PENDING" && $2 != "CONFIGURING" {print $1}')
            take=$(echo "$waiting" | grep -c '[0-9]' || true)
            (( take > surplus )) && take=$surplus
            if (( take > 0 )); then
                # shellcheck disable=SC2086
                scancel $(echo "$waiting" | tail -n "$take") \
                    && echo "relay: cancelled $take supervisor(s) that had not started"
            fi
            if (( surplus > take )); then
                # shellcheck disable=SC2086
                scancel --signal=TERM --full $(echo "$running" | tail -n $((surplus - take))) \
                    && echo "relay: asked $((surplus - take)) supervisor(s) to finish and exit"
            fi
        fi
        # The ceiling, measured here for the same reason the submitting is:
        # sinfo does not work inside the container either. %C is
        # "allocated/idle/other/total" CPUs across the partition.
        cpus=$(sinfo --noheader --partition "$SUPERVISOR_PARTITION" --format %C 2>/dev/null \
               | head -1 | cut -d/ -f2)
        cpus=${cpus:-0}
        printf '{"replicas": %d, "want": %d, "jobs": [%s], "at": %d, "max_supervisors": %d, "capacity_basis": "idle CPUs in partition %s", "idle_cpus": %d, "cpus_each": %d, "partition": "%s"}\n' \
            "$have" "$want" \
            "$(echo "$ids" | grep '[0-9]' | sed 's/.*/"&"/' | paste -sd, -)" \
            "$(date +%s)" \
            "$(( have + cpus / SUPERVISOR_CPUS ))" "$SUPERVISOR_PARTITION" \
            "$cpus" "$SUPERVISOR_CPUS" "$SUPERVISOR_PARTITION" \
            > "$POOL_DIR/pool.state.part"
        mv "$POOL_DIR/pool.state.part" "$POOL_DIR/pool.state"
        sleep "${POOL_RELAY_SECONDS:-20}"
    done
}
pool_relay >> "$LOGS/pool-relay.log" 2>&1 &
RELAY_PID=$!

# --- The run relay ----------------------------------------------------------
# The runner is in a container too, so it hands each run over as a file:
# submit/<run> to start one, stop/<run> to checkpoint and stop it. This loop
# submits run.sbatch, follows the Slurm job, and writes what it sees to the
# run's slurm.json, which is all the runner reads.
run_state() {  # run_dir slurm_id state exit_code error
    python3 - "$@" <<'PY'
import json, pathlib, sys, time
run_dir, sid, state, code, error = sys.argv[1:6]
record = {"slurm_id": sid or None, "state": state, "at": int(time.time()),
          "exit_code": int(code) if code.lstrip("-").isdigit() else None}
if error:
    record["error"] = error
path = pathlib.Path(run_dir) / "slurm.json"
part = path.with_suffix(".part")
part.write_text(json.dumps(record))
part.replace(path)
PY
}
run_relay() {
    local f id dir sid left state code
    mkdir -p "$RUN_RELAY_DIR"/{submit,stop,active}

    # Runs an earlier session handed over are followed too. Their relay died
    # with that session, so their slurm.json still says what it last saw --
    # RUNNING, say -- and the runner, which adopts them on that word, would
    # hold their slots forever. Every one of them has ended by now (no run
    # outlives its session), so the poll below only has to ask sacct how.
    python3 - "$DATA_DIR/jobs" "$RUN_RELAY_DIR/active" <<'PY'
import json, pathlib, sys
jobs, active = map(pathlib.Path, sys.argv[1:3])
unfinished = {"PENDING", "CONFIGURING", "REQUEUED", "RESIZING", "SUSPENDED", "RUNNING", "COMPLETING"}
for path in jobs.glob("job_*/slurm.json"):
    try:
        seen = json.loads(path.read_text())
    except (OSError, ValueError):
        continue
    sid, run = str(seen.get("slurm_id") or ""), path.parent.name[len("job_"):]
    if seen.get("state") in unfinished and sid.isdigit() and not (active / run).exists():
        (active / run).write_text(sid + "\n")
        print(f"relay: following run {run} (Slurm {sid}) from an earlier session")
PY

    # Build a sid->run-id reverse map so the batched squeue/sacct responses
    # can be matched back to runs. Declared outside the loop so bash does not
    # re-declare it on every iteration (harmless, but noisy with set -x).
    declare -A SID_TO_RUN

    while true; do
        # --- submit new runs -------------------------------------------------
        left=$(squeue --noheader --jobs "$JOB" --format %L 2>/dev/null)
        [[ $left =~ ^[0-9:-]+$ ]] || left=""
        for f in "$RUN_RELAY_DIR"/submit/*; do
            [[ -e $f ]] || break
            id=$(basename "$f"); dir=$DATA_DIR/jobs/job_$id
            # Nothing a run scores against outlives this session, so neither
            # may the run. The gpus-per-node and remaining args are in
            # run.sbatch's own #SBATCH lines; the relay only overrides time.
            # A run that uses Boltz needs 2 GCDs (one for REINVENT, one for
            # its private Boltz); a purely CPU-scored run only needs 1.
            if grep -q 'service = "boltz"' "$dir/config.toml" 2>/dev/null; then
                run_gpus=2
            else
                run_gpus=1
            fi
            sid=$(sbatch --parsable --job-name "$RUN_JOB_NAME" \
                         ${SLURM_ACCOUNT:+--account "$SLURM_ACCOUNT"} \
                         ${left:+--time "$left"} \
                         --gpus-per-node=$run_gpus \
                         --output "$dir/slurm-%j.out" \
                         --export "ALL,RUN_ID=$id,SESSION_ENV=$SESSION_ENV" \
                         "$REPO/services/lumi/run.sbatch" 2>&1)
            if [[ $sid =~ ^[0-9]+$ ]]; then
                run_state "$dir" "$sid" PENDING "" ""
                echo "$sid" > "$RUN_RELAY_DIR/active/$id"
                SID_TO_RUN[$sid]=$id
                echo "relay: run $id is Slurm job $sid"
            else
                run_state "$dir" "" SUBMIT_FAILED "" "sbatch refused the run: $sid"
                echo "relay: sbatch failed for run $id: $sid"
            fi
            rm -f "$f"
        done

        # --- forward stop requests -------------------------------------------
        for f in "$RUN_RELAY_DIR"/stop/*; do
            [[ -e $f ]] || break
            id=$(basename "$f"); sid=$(cat "$RUN_RELAY_DIR/active/$id" 2>/dev/null)
            if [[ -n $sid ]]; then
                # One squeue for the state so we know whether to hard-cancel or
                # checkpoint-stop; this is rare so a single call is fine.
                state=$(squeue --noheader --jobs "$sid" --format %T 2>/dev/null)
                if [[ $state == PENDING || $state == CONFIGURING ]]; then
                    scancel "$sid" && echo "relay: cancelled run $id before it started"
                else
                    scancel --signal=USR1 --batch "$sid" && echo "relay: asked run $id to checkpoint and stop"
                fi
            fi
            rm -f "$f"
        done

        # --- poll active runs in one batched call ----------------------------
        # Rebuilding the map each tick covers runs adopted after a restart.
        SID_TO_RUN=()
        for f in "$RUN_RELAY_DIR"/active/*; do
            [[ -e $f ]] || break
            sid=$(cat "$f"); SID_TO_RUN[$sid]=$(basename "$f")
        done

        if [[ ${#SID_TO_RUN[@]} -gt 0 ]]; then
            ids_csv=$(IFS=,; echo "${!SID_TO_RUN[*]}")

            # One squeue for all active jobs; outputs "<sid> <state>" lines.
            # Emptied every pass: declaring it again keeps last pass's entries,
            # and a run that left the queue would read as still in it forever.
            declare -A LIVE_STATE
            LIVE_STATE=()
            while read -r s st; do
                LIVE_STATE[$s]=$st
            done < <(squeue --noheader --jobs "$ids_csv" --format "%i %T" 2>/dev/null)

            # Jobs not in squeue output have left the queue; collect them for
            # one batched sacct call.
            finished_ids=()
            for sid in "${!SID_TO_RUN[@]}"; do
                id=${SID_TO_RUN[$sid]}; dir=$DATA_DIR/jobs/job_$id
                if [[ -n ${LIVE_STATE[$sid]:-} ]]; then
                    run_state "$dir" "$sid" "${LIVE_STATE[$sid]}" "" ""
                else
                    finished_ids+=("$sid")
                fi
            done

            # One sacct call for all finished jobs.
            if [[ ${#finished_ids[@]} -gt 0 ]]; then
                fin_csv=$(IFS=,; echo "${finished_ids[*]}")
                while IFS='|' read -r sid_col st_raw code_raw; do
                    sid=${sid_col%%[^0-9]*}; [[ -n $sid ]] || continue
                    id=${SID_TO_RUN[$sid]:-}; [[ -n $id ]] || continue
                    dir=$DATA_DIR/jobs/job_$id
                    state=${st_raw%% *}; code=${code_raw%%:*}
                    # An squeue that missed a live run must not end it here.
                    case $state in PENDING|CONFIGURING|REQUEUED|RESIZING|SUSPENDED|RUNNING|COMPLETING) continue ;; esac
                    run_state "$dir" "$sid" "$state" "$code" ""
                    echo "relay: run $id (Slurm $sid) ended $state, exit $code"
                    rm -f "$RUN_RELAY_DIR/active/$id"
                done < <(sacct --noheader --parsable2 --allocations \
                              --jobs "$fin_csv" --format JobID,State,ExitCode 2>/dev/null)
            fi
        fi

        sleep "${RUN_RELAY_SECONDS:-10}"
    done
}
RUN_RELAY_PID=
if [[ $RUN_BACKEND == slurm ]]; then
    run_relay >> "$LOGS/run-relay.log" 2>&1 &
    RUN_RELAY_PID=$!
fi

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

# --- The RL runner ----------------------------------------------------------
# One process owns the GPU REINVENT runs on, and every agent -- the one behind
# the UI and every supervisor in the pool -- starts jobs through it. Not an
# indirection for its own sake: supervisors are CPU jobs with no GPU to run on,
# and a shared job table is what keeps the pool from oversubscribing this GCD.
AGENT_SIF=$REPO/services/singularity/dd-agent-amd.sif
SITE=/opt/venv/lib/python3.12/site-packages
mkdir -p "$DATA_DIR"/{shared,jobs,logs,tb_logs,ttps,agent_logs} "$RUN_RELAY_DIR"
RELAY_FLAGS=()
[[ $RUN_BACKEND == slurm ]] && RELAY_FLAGS=(--env REINVENT_RUN_RELAY=/run-relay --bind "$RUN_RELAY_DIR:/run-relay")
PRIORS_DIR=${PRIORS_DIR:-$REPO/priors}
RUNNER_PRIOR_BIND=()
[[ -d $PRIORS_DIR ]] && RUNNER_PRIOR_BIND=(--bind "$PRIORS_DIR:/app/REINVENT4/priors")
singularity exec \
    "${RELAY_FLAGS[@]}" \
    "${RUNNER_PRIOR_BIND[@]}" \
    --env ROCR_VISIBLE_DEVICES=$REINVENT_GPU \
    --env REINVENT4_ROOT=/app/REINVENT4 \
    --env REINVENT4_MCP_CONFIG=/app/reinvent4_mcp_config.singularity.yaml \
    --env PYTHONPATH=$SITE/reinvent4_mcp/_plugins \
    --env REINVENT_RUNNER_SELF=1 \
    --env PORT=$RUNNER_PORT \
    --bind "$DATA_DIR/shared:/shared" \
    --bind "$MCP_CONFIG:/app/reinvent4_mcp_config.singularity.yaml" \
    --bind "$DATA_DIR/ttps:/data/reinvent4_mcp/ttps" \
    --bind "$DATA_DIR/jobs:/data/reinvent4_mcp/jobs" \
    --bind "$DATA_DIR/logs:/data/reinvent4_mcp/logs" \
    --bind "$DATA_DIR/tb_logs:/data/reinvent4_mcp/tb_logs" \
    --bind "$REPO/reinvent4_agent:$SITE/reinvent4_agent" \
    --bind "$REPO/reinvent4_mcp:$SITE/reinvent4_mcp" \
    --bind "$REPO/services/reinvent-runner:/app/runner" \
    --pwd /app/runner \
    "$AGENT_SIF" uvicorn app:app --host 0.0.0.0 --port $RUNNER_PORT --workers 1 \
    > "$LOGS/runner.log" 2>&1 &
RUNNER_PID=$!

bash "$REPO/services/singularity/run.sh" --vendor amd --web > "$LOGS/stack.log" 2>&1 &
STACK=$!

# Supervisors outlive this script unless they are cancelled: they are separate
# jobs, and the session they joined is gone the moment this exits.
cleanup() {
    kill "$RELAY_PID" $RUN_RELAY_PID 2>/dev/null
    scancel --name "$SUPERVISOR_JOB_NAME" --user "$USER" 2>/dev/null
    scancel --name "$RUN_JOB_NAME" --user "$USER" 2>/dev/null
    kill "$STACK" "$RUNNER_PID" "$CHAT_PID" "$SUP_PID" 2>/dev/null
    pkill -f "vllm serve" 2>/dev/null
    wait
}
trap cleanup EXIT TERM

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
wait_for reinvent-runner "http://127.0.0.1:$RUNNER_PORT/health" "$RUNNER_PID" || exit 1
wait_for dd-web http://127.0.0.1:8000/api/health "$STACK" || exit 1

curl -s -o /dev/null -w "without token: HTTP %{http_code}\n" "http://$NODE:8000/api/session"
echo "READY node=$NODE"
echo "  start a campaign with:"
echo "    srun --overlap --jobid=$JOB bash $REPO/services/lumi/conduct.sh <campaign_id>"
wait $STACK
