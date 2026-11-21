#!/usr/bin/env bash
# Run the Drug Discovery agent on a laptop (macOS/Linux + Docker or
# OrbStack), with no GPU, no Slurm and no LLM API key.
#
#   REINVENT4 + dd-mcp  ->  container  (heavy scientific stack, CPU)
#   dd-agent            ->  host venv  (so it can use the `claude` CLI
#                                       you are already logged in to)
#
# The agent talks to the MCP server over stdio through `docker exec`,
# exactly as it would to a local subprocess.
#
#   bash services/local/run-local.sh              # terminal chat
#   bash services/local/run-local.sh --web        # browser UI + 3D viewer
#   bash services/local/run-local.sh --multi-agent
#   echo "score aspirin" | bash services/local/run-local.sh
set -euo pipefail

WEB=0
ARGS=()
for arg in "$@"; do
    case "$arg" in
        --web) WEB=1 ;;
        *) ARGS+=("$arg") ;;
    esac
done
set -- ${ARGS[@]+"${ARGS[@]}"}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

IMAGE="${DD_IMAGE:-dd-agent:local}"
CONTAINER="${DD_MCP_CONTAINER:-dd-mcp-local}"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
VENV="${DD_AGENT_VENV:-$ROOT/.venv-agent}"

say() { printf '\033[2m[run-local]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[31m[run-local] %s\033[0m\n' "$*" >&2; exit 1; }

command -v docker >/dev/null || die "docker not found (start OrbStack?)"
docker info >/dev/null 2>&1 || die "docker daemon not reachable (start OrbStack?)"

# --- image ------------------------------------------------------------
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    say "building $IMAGE (first time: this takes a while)"
    docker build -f "$ROOT/services/dd-agent/Dockerfile.local-cpu" \
        -t "$IMAGE" "$ROOT"
fi

# --- MCP container ----------------------------------------------------
mkdir -p "$DATA_DIR"/{shared,ttps,jobs,logs,tb_logs,agent_logs}
if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    say "starting $CONTAINER"
    # DD_MCP_DEV_MOUNT=1 shadows the image's installed reinvent4_mcp with the
    # source tree, so MCP-server edits take effect on the next `docker exec`
    # instead of needing an image rebuild. Read-only: the container writes
    # only under /shared.
    dev_mount=()
    if [ "${DD_MCP_DEV_MOUNT:-0}" = "1" ]; then
        say "dev mount: $ROOT -> /src (MCP source shadows the image)"
        dev_mount=(-v "$ROOT:/src:ro" -e "PYTHONPATH=/src")
    fi
    docker run -d --name "$CONTAINER" \
        "${dev_mount[@]}" \
        -v "$DATA_DIR/shared:/shared" \
        -v "$DATA_DIR/ttps:/root/.reinvent4_mcp/ttps" \
        -v "$DATA_DIR/jobs:/data/reinvent4_mcp/jobs" \
        -v "$DATA_DIR/logs:/data/reinvent4_mcp/logs" \
        -v "$DATA_DIR/tb_logs:/data/reinvent4_mcp/tb_logs" \
        -v "$DATA_DIR/agent_logs:/logs" \
        --entrypoint sleep "$IMAGE" infinity >/dev/null
fi

# --- host-side agent --------------------------------------------------
need_install=0
[ -x "$VENV/bin/dd-agent" ] || need_install=1
# An existing venv from before --web existed has no dd-web / FastAPI.
[ "$WEB" = "1" ] && [ ! -x "$VENV/bin/dd-web" ] && need_install=1

if [ "$need_install" = "1" ]; then
    say "creating agent venv at $VENV"
    # A pyenv shim can exist on PATH without the version being enabled,
    # so test each candidate by running it.
    PY=""
    for cand in python3.13 python3.12 python3.11 python3.10 python3; do
        if "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' \
             >/dev/null 2>&1; then
            PY="$cand"; break
        fi
    done
    [ -n "$PY" ] || die "no Python >= 3.10 found on PATH"
    say "using $("$PY" -V 2>&1)"
    [ -d "$VENV" ] || "$PY" -m venv "$VENV"
    "$VENV/bin/pip" install -q --upgrade pip
    # Editable: this is a working checkout, so agent edits should take
    # effect without a reinstall. The [web] extra is two small wheels, so
    # install it either way and `--web` always works.
    "$VENV/bin/pip" install -q -e "$ROOT/reinvent4_agent[web]" claude-agent-sdk
fi

# `claude-code` drives the Claude CLI this machine is logged in to.  In a
# container (or CI) there is no interactive login, so pass a token from
# `claude setup-token` as CLAUDE_CODE_OAUTH_TOKEN instead.
export REINVENT4_AGENT_BACKEND="${REINVENT4_AGENT_BACKEND:-claude-code}"
export REINVENT4_AGENT_MULTI_AGENT="${REINVENT4_AGENT_MULTI_AGENT:-false}"
export REINVENT4_MCP_COMMAND="$HERE/dd-mcp-docker.sh"
export REINVENT4_AGENT_LOG_DIR="${REINVENT4_AGENT_LOG_DIR:-$DATA_DIR/agent_logs}"
export DD_MCP_CONTAINER="$CONTAINER"

if [ "$WEB" = "1" ]; then
    exec "$VENV/bin/dd-web" "$@"
fi
exec "$VENV/bin/dd-agent" "$@"
