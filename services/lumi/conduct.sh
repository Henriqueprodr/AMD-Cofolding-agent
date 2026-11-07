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

CAMPAIGN=${1:-}
[[ -n $CAMPAIGN ]] || { echo "usage: conduct.sh <campaign_id>" >&2; exit 2; }

SESSION_ENV=${SESSION_ENV:-${DATA_DIR:-$PWD/data}/session.env}
[[ -f $SESSION_ENV ]] || { echo "no session to join: $SESSION_ENV" >&2; exit 1; }
# shellcheck disable=SC1090
source "$SESSION_ENV"

export REINVENT4_AGENT_BACKEND=openai
export REINVENT4_AGENT_BASE_URL=$CHAT_BASE_URL
export REINVENT4_AGENT_MODEL=$CHAT_MODEL_ID
export REINVENT4_AGENT_API_KEY=local-vllm
export REINVENT4_AGENT_LOG_DIR=/logs
# A conductor steers a fleet; it does not need a second opinion on its own turn.
export REINVENT4_AGENT_MULTI_AGENT=false
# Creating a base TTP is how a campaign on a new target begins; editing or
# deleting one would rewrite the thing its own campaigns are measured against.
export REINVENT4_AGENT_DENY_TOOLS=update_ttp,delete_ttp
export CAMPAIGN_STORE_URL=${CAMPAIGN_STORE_URL:-http://$SESSION_NODE:8087}
export DD_SERVICE_REGISTRY
export SUPERVISOR_JOB_NAME=${SUPERVISOR_JOB_NAME:-dd-supervisor}
export SUPERVISOR_SBATCH=$REPO/services/lumi/supervisor.sbatch

SIF=$REPO/services/singularity/dd-agent-amd.sif
[[ -f $SIF ]] || { echo "no agent image at $SIF (build it with build.sbatch)" >&2; exit 1; }

# sbatch and squeue live on the host, not in the image, so the scaler needs
# them bound in along with the Slurm config they read.
SLURM_BINDS=()
for p in /usr/bin/sbatch /usr/bin/squeue /usr/bin/scancel /etc/slurm /var/run/munge /usr/lib64/slurm; do
    [[ -e $p ]] && SLURM_BINDS+=(--bind "$p")
done

exec singularity exec \
    --bind "$DATA_DIR/shared:/shared" \
    --bind "$DATA_DIR/agent_logs:/logs" \
    --bind "$DATA_DIR/ttps:/data/reinvent4_mcp/ttps" \
    --bind "$DATA_DIR/jobs:/data/reinvent4_mcp/jobs" \
    --bind "$REPO/reinvent4_agent:/app/reinvent4_agent_src" \
    --env PYTHONPATH=/app/reinvent4_agent_src/.. \
    "${SLURM_BINDS[@]}" \
    "$SIF" python -m reinvent4_agent.conductor "$CAMPAIGN"
