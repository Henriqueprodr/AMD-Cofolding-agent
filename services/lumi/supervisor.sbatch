#!/bin/bash
#SBATCH --job-name=dd-supervisor
# The pool scaler passes --account itself, from the session job's own, and a
# flag beats this. It is here for the other caller: a person submitting one
# supervisor by hand, who would otherwise be told AssocMaxSubmitJobLimit.
#SBATCH --account=project_462001170
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

# Passed with --env rather than exported: the image's %environment declares
# defaults for these three, and a declared variable beats an inherited one, so
# exporting them here would leave the supervisor talking to api.anthropic.com.
# Driver only, no Reviewer: the same choice the cluster's supervisors make.
# A pool of ten supervisors each running a Reviewer would double the load on
# the Gemma tier to second-guess work the conductor already scores.
AGENT_ENV=(
    --env REINVENT4_AGENT_BACKEND=openai
    --env "REINVENT4_AGENT_BASE_URL=$SUPERVISOR_BASE_URL"
    --env "REINVENT4_AGENT_MODEL=$SUPERVISOR_MODEL_ID"
    --env REINVENT4_AGENT_API_KEY=local-vllm
    --env REINVENT4_AGENT_LOG_DIR=/logs
    --env REINVENT4_AGENT_MULTI_AGENT=false
    --env REINVENT4_AGENT_NO_VERIFY_SSL=false
    --env "CAMPAIGN_STORE_URL=${CAMPAIGN_STORE_URL:-http://$SESSION_NODE:8087}"
    # The rest of the service URLs come from here, applied by supervisor.main().
    --env DD_SERVICE_REGISTRY=/tmp/service-registry.json
)

SIF=$REPO/services/singularity/dd-agent-amd.sif
[[ -f $SIF ]] || { echo "no agent image at $SIF (build it with build.sbatch)" >&2; exit 1; }

mkdir -p "$DATA_DIR/agent_logs" "$DATA_DIR/shared"
echo "supervisor $SLURM_JOB_ID on $(hostname) joining session $SESSION_JOB on $SESSION_NODE"

# Bound over the copies installed in the image, the same way run.sh does it
# with DEV_SOURCE=1, so a pool job started now runs the code in the checkout.
SITE=/opt/venv/lib/python3.12/site-packages
exec singularity exec \
    --bind "$DATA_DIR/shared:/shared" \
    --bind "$DATA_DIR/agent_logs:/logs" \
    --bind "$DATA_DIR/ttps:/data/reinvent4_mcp/ttps" \
    --bind "$DATA_DIR/jobs:/data/reinvent4_mcp/jobs" \
    --bind "$REPO/reinvent4_agent:$SITE/reinvent4_agent" \
    --bind "$REPO/reinvent4_mcp:$SITE/reinvent4_mcp" \
    --bind "$MCP_CONFIG_FILE:/app/reinvent4_mcp_config.singularity.yaml" \
    --bind "$DD_SERVICE_REGISTRY:/tmp/service-registry.json" \
    --env REINVENT4_MCP_CONFIG=/app/reinvent4_mcp_config.singularity.yaml \
    --env PYTHONPATH=$SITE/reinvent4_mcp/_plugins \
    "${AGENT_ENV[@]}" \
    "$SIF" python -m reinvent4_agent.supervisor
