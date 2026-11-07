#!/bin/bash
#SBATCH --job-name=dd-supervisor
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/supervisor-%j.out
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# One supervisor: claims a task from the ledger, runs it, reports the outcome,
# claims the next. This is a pool member, not something to submit by hand --
# the conductor sizes the pool and reinvent4_agent.pool's Slurm scaler submits
# and cancels these as the campaign's queue grows and drains.
#
# A CPU partition on purpose. A supervisor thinks with Gemma over HTTP and
# drives REINVENT through the runner, so the only GPU it would occupy is one
# it would spend idle -- and a CPU job starts sooner, which is the whole point
# of scaling the pool in the first place.
#
# Everything about which session to join comes from session.env, written by
# session.sbatch, rather than from the submitting environment: a supervisor
# submitted an hour into a campaign still has to find the right node.

set -uo pipefail
SESSION_ENV=${SESSION_ENV:-${DATA_DIR:-$SLURM_SUBMIT_DIR/data}/session.env}
if [[ ! -f $SESSION_ENV ]]; then
    echo "no session to join: $SESSION_ENV does not exist" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$SESSION_ENV"

if ! curl -sf --max-time 10 "${SUPERVISOR_BASE_URL%/v1}/health" >/dev/null; then
    echo "the supervisor model at $SUPERVISOR_BASE_URL is not answering;" \
         "session $SESSION_JOB on $SESSION_NODE is probably gone" >&2
    exit 1
fi

export REINVENT4_AGENT_BACKEND=openai
export REINVENT4_AGENT_BASE_URL=$SUPERVISOR_BASE_URL
export REINVENT4_AGENT_MODEL=$SUPERVISOR_MODEL_ID
export REINVENT4_AGENT_API_KEY=local-vllm
export REINVENT4_AGENT_LOG_DIR=/logs
# Driver only, no Reviewer: the same choice the cluster's supervisors make.
# A pool of ten supervisors each running a Reviewer would double the load on
# the Gemma tier to second-guess work the conductor already scores.
export REINVENT4_AGENT_MULTI_AGENT=false
export CAMPAIGN_STORE_URL=${CAMPAIGN_STORE_URL:-http://$SESSION_NODE:8087}
# The rest of the service URLs come from here, applied by supervisor.main().
export DD_SERVICE_REGISTRY

SIF=$REPO/services/singularity/dd-agent-amd.sif
[[ -f $SIF ]] || { echo "no agent image at $SIF (build it with build.sbatch)" >&2; exit 1; }

mkdir -p "$DATA_DIR/agent_logs" "$DATA_DIR/shared"
echo "supervisor $SLURM_JOB_ID on $(hostname) joining session $SESSION_JOB on $SESSION_NODE"

exec singularity exec \
    --bind "$DATA_DIR/shared:/shared" \
    --bind "$DATA_DIR/agent_logs:/logs" \
    --bind "$DATA_DIR/ttps:/data/reinvent4_mcp/ttps" \
    --bind "$DATA_DIR/jobs:/data/reinvent4_mcp/jobs" \
    --bind "$REPO/reinvent4_agent:/app/reinvent4_agent_src" \
    --env PYTHONPATH=/app/reinvent4_agent_src/.. \
    "$SIF" python -m reinvent4_agent.supervisor
