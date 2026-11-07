#!/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Run the conductor for one campaign inside a running session.
#
#   srun --overlap --jobid=<session job> bash services/lumi/conduct.sh <campaign_id>
#
# The conductor plans with the chat tier -- the same gpt-oss-120b a person
# talks to, because a person is waiting on its turns too -- and sizes the
# supervisor pool as the queue grows and drains. On Slurm that means
# submitting and cancelling supervisor.sbatch jobs; reinvent4_agent.pool picks
# that scaler on its own, by noticing there is no ServiceAccount token here
# and that squeue and sbatch are.
#
# Run inside the allocation rather than from a login node so the conductor
# shares the session's network and can reach the services on 127.0.0.1.
set -uo pipefail

[[ $# -gt 0 ]] || { echo "usage: conduct.sh <campaign_id> | --start --goal ... " >&2; exit 2; }

SESSION_ENV=${SESSION_ENV:-${DATA_DIR:-$PWD/data}/session.env}
[[ -f $SESSION_ENV ]] || { echo "no session to join: $SESSION_ENV" >&2; exit 1; }
# shellcheck disable=SC1090
source "$SESSION_ENV"

# Passed with --env rather than exported: the image's %environment declares
# defaults for the agent variables, and a declared variable beats an inherited
# one, so exporting them would leave the conductor talking to api.anthropic.com.
AGENT_ENV=(
    --env REINVENT4_AGENT_BACKEND=openai
    --env "REINVENT4_AGENT_BASE_URL=$CHAT_BASE_URL"
    --env "REINVENT4_AGENT_MODEL=$CHAT_MODEL_ID"
    --env REINVENT4_AGENT_API_KEY=local-vllm
    --env REINVENT4_AGENT_LOG_DIR=/logs
    # A conductor steers a fleet; it does not need a second opinion on its own turn.
    --env REINVENT4_AGENT_MULTI_AGENT=false
    # Creating a base TTP is how a campaign on a new target begins; editing or
    # deleting one would rewrite what its own campaigns are measured against.
    --env REINVENT4_AGENT_DENY_TOOLS=update_ttp,delete_ttp
    --env REINVENT4_AGENT_NO_VERIFY_SSL=false
    --env "CAMPAIGN_STORE_URL=${CAMPAIGN_STORE_URL:-http://$SESSION_NODE:8087}"
    --env DD_SERVICE_REGISTRY=/tmp/service-registry.json
    # No boltz-gateway in this stack: one Boltz, no proxy to ask how deep its
    # queue is. Empty rather than unset, so the conductor skips admission
    # control instead of falling back to the cluster's hostname.
    --env BOLTZ_GATEWAY_STATS_URL=
)
export SUPERVISOR_JOB_NAME=${SUPERVISOR_JOB_NAME:-dd-supervisor}
export SUPERVISOR_SBATCH=$REPO/services/lumi/supervisor.sbatch

SIF=$REPO/services/singularity/dd-agent-amd.sif
[[ -f $SIF ]] || { echo "no agent image at $SIF (build it with build.sbatch)" >&2; exit 1; }

# Not bound in: sbatch and friends. They are on the host but unusable from a
# container -- munge and the user database do not cross the boundary -- so the
# conductor states the pool size it wants in here and session.sbatch's relay
# submits for it. RelayPoolScaler picks this up from the directory alone.
export SUPERVISOR_POOL_CONTROL
[[ -d ${SUPERVISOR_POOL_CONTROL:-} ]] || {
    echo "no pool relay at '${SUPERVISOR_POOL_CONTROL:-}': the conductor will not be able to scale" >&2
}

SITE=/opt/venv/lib/python3.12/site-packages
exec singularity exec \
    --bind "$DATA_DIR/shared:/shared" \
    --bind "$DATA_DIR/agent_logs:/logs" \
    --bind "$DATA_DIR/ttps:/data/reinvent4_mcp/ttps" \
    --bind "$DATA_DIR/jobs:/data/reinvent4_mcp/jobs" \
    --bind "$SUPERVISOR_POOL_CONTROL" \
    --bind "$REPO/reinvent4_agent:$SITE/reinvent4_agent" \
    --bind "$REPO/reinvent4_mcp:$SITE/reinvent4_mcp" \
    --bind "$MCP_CONFIG_FILE:/app/reinvent4_mcp_config.singularity.yaml" \
    --bind "$DD_SERVICE_REGISTRY:/tmp/service-registry.json" \
    --env REINVENT4_MCP_CONFIG=/app/reinvent4_mcp_config.singularity.yaml \
    --env PYTHONPATH=$SITE/reinvent4_mcp/_plugins \
    "${AGENT_ENV[@]}" \
    "$SIF" python -m reinvent4_agent.conductor "$@"
