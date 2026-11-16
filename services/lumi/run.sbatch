#!/bin/bash
#SBATCH --job-name=dd-run
#SBATCH --account=project_462001170
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=14
#SBATCH --mem=120G
#SBATCH --time=04:00:00
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# One REINVENT run as its own job. Not submitted by hand: the session's run
# relay submits it for each run the runner hands over, with RUN_ID, the
# session's remaining time as --time, and --output in the run's directory.
#
# Two GCDs: REINVENT on the first and a Boltz of the run's own on the second.
# Boltz is what a run waits on, and one shared instance on the session node
# serialised every run of the campaign behind one GPU; a run that brings its
# own scales with the partition instead. The other scorers are CPU services
# and stay on the session node, reached by its hostname as before.

set -uo pipefail
RUN_ID=${RUN_ID:?the run relay passes RUN_ID}
SESSION_ENV=${SESSION_ENV:-${DATA_DIR:-$SLURM_SUBMIT_DIR/data}/session.env}
[[ -f $SESSION_ENV ]] || { echo "no session to join: $SESSION_ENV does not exist" >&2; exit 1; }
# shellcheck disable=SC1090
source "$SESSION_ENV"

JOB_DIR=$DATA_DIR/jobs/job_$RUN_ID
[[ -x $JOB_DIR/worker.sh ]] || { echo "run $RUN_ID has no worker.sh in $JOB_DIR" >&2; exit 1; }
SIF=$REPO/services/singularity/dd-agent-amd.sif
BOLTZ_SIF=$REPO/services/singularity/boltz-amd.sif
SITE=/opt/venv/lib/python3.12/site-packages
PRIORS_DIR=${PRIORS_DIR:-$REPO/priors}
PRIOR_BIND=()
[[ -d $PRIORS_DIR ]] && PRIOR_BIND=(--bind "$PRIORS_DIR:/app/REINVENT4/priors")

IFS=, read -r -a GCDS <<< "${ROCR_VISIBLE_DEVICES:-0}"
REINVENT_GCD=${GCDS[0]}
# If only 1 GCD was allocated (no-Boltz run), skip Boltz entirely.
USE_BOLTZ=0
[[ ${#GCDS[@]} -ge 2 ]] && USE_BOLTZ=1
# Join all allocated GCDs for the Boltz container so the wrapper can shard
# the batch across GCDs (one boltz predict subprocess per GCD, each pinned
# via ROCR_VISIBLE_DEVICES inside the wrapper).  REINVENT shares GCD 0 — it
# is idle while Boltz is scoring, and Boltz is done before REINVENT backprops.
printf -v BOLTZ_GCDS '%s,' "${GCDS[@]}"; BOLTZ_GCDS=${BOLTZ_GCDS%,}
if (( USE_BOLTZ )); then
    echo "run $RUN_ID: Slurm job $SLURM_JOB_ID on $(hostname), REINVENT on GCD $REINVENT_GCD, Boltz on GCDs $BOLTZ_GCDS"
else
    echo "run $RUN_ID: Slurm job $SLURM_JOB_ID on $(hostname), REINVENT on GCD $REINVENT_GCD, no Boltz"
fi

# A stop can come before REINVENT starts (e.g. while Boltz is loading).
# Bash's default for USR1 is to die, which Slurm records as a failure;
# noted here instead, it ends the job cleanly at the next check.
STOP=0
trap 'STOP=1' USR1
stopped_early() { echo "stopped before REINVENT started, at $(date +%T)"; exit 0; }

BOLTZ_PORT=
BOLTZ=
if (( USE_BOLTZ )); then
    # Two runs can share a node, so the port is whatever is free.
    BOLTZ_PORT=$(python3 -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')
    mkdir -p "$DATA_DIR/boltz_cache"
    # A child of this script, not a `singularity instance`: started from a batch
    # script an instance is shut down about ten seconds later, while the same
    # command from a job step stays up.
    singularity run \
        --env PORT=$BOLTZ_PORT \
        --env ROCR_VISIBLE_DEVICES=$BOLTZ_GCDS \
        --bind "$DATA_DIR/shared:/shared" \
        --bind "$DATA_DIR/boltz_cache:/cache" \
        --bind "$REPO/services/boltz2:/app/wrapper" \
        "$BOLTZ_SIF" > "$JOB_DIR/boltz.log" 2>&1 &
    BOLTZ=$!
    trap 'kill "$BOLTZ" 2>/dev/null' EXIT

    for i in $(seq 1 180); do
        (( STOP )) && stopped_early
        kill -0 "$BOLTZ" 2>/dev/null || { echo "Boltz exited before it was up; see $JOB_DIR/boltz.log" >&2; exit 1; }
        curl -sf "http://127.0.0.1:$BOLTZ_PORT/health" >/dev/null && break
        (( i == 180 )) && { echo "Boltz did not come up on :$BOLTZ_PORT in 15 min" >&2; exit 1; }
        sleep 5
    done
    echo "Boltz up on :$BOLTZ_PORT at $(date +%T)"
    (( STOP )) && stopped_early
fi

# The same image and binds as the runner, so worker.sh -- the command the
# runner would have run itself -- finds every path it names.
# SCORER_URL_OVERRIDES is only set when this run has its own Boltz; without it
# the scorer keeps the session-node URL, which is what a no-Boltz run expects.
BOLTZ_OVERRIDE=()
(( USE_BOLTZ )) && BOLTZ_OVERRIDE=(--env "SCORER_URL_OVERRIDES=boltz=http://127.0.0.1:$BOLTZ_PORT")
singularity exec \
    "${BOLTZ_OVERRIDE[@]}" \
    "${PRIOR_BIND[@]}" \
    --env ROCR_VISIBLE_DEVICES=$REINVENT_GCD \
    --env REINVENT4_MCP_CONFIG=/app/reinvent4_mcp_config.singularity.yaml \
    --bind "$DATA_DIR/shared:/shared" \
    --bind "$MCP_CONFIG_FILE:/app/reinvent4_mcp_config.singularity.yaml" \
    --bind "$DATA_DIR/ttps:/data/reinvent4_mcp/ttps" \
    --bind "$DATA_DIR/jobs:/data/reinvent4_mcp/jobs" \
    --bind "$DATA_DIR/logs:/data/reinvent4_mcp/logs" \
    --bind "$DATA_DIR/tb_logs:/data/reinvent4_mcp/tb_logs" \
    --bind "$REPO/reinvent4_agent:$SITE/reinvent4_agent" \
    --bind "$REPO/reinvent4_mcp:$SITE/reinvent4_mcp" \
    "$SIF" bash "/data/reinvent4_mcp/jobs/job_$RUN_ID/worker.sh" &
WORKER=$!

# The relay stops a run with scancel --signal=USR1 --batch, which reaches this
# script and not the worker; REINVENT turns USR1 into a checkpointing stop,
# so it is handed on rather than acted on here.
trap 'echo "stop requested at $(date +%T)"; kill -USR1 "$WORKER" 2>/dev/null' USR1
# If a stop arrived while Boltz was loading, forward it now that REINVENT is up.
(( STOP )) && kill -USR1 "$WORKER" 2>/dev/null
rc=0
while true; do
    wait "$WORKER"
    rc=$?
    kill -0 "$WORKER" 2>/dev/null || break
done
echo "run $RUN_ID exited $rc at $(date +%T)"
exit $rc
