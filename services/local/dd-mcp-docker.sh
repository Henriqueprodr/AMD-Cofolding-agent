#!/usr/bin/env bash
# stdio shim: dd-agent spawns this, and it hands the agent the dd-mcp
# server running inside the local container.  Nothing may be written to
# stdout except the MCP protocol itself.
set -euo pipefail
CONTAINER="${DD_MCP_CONTAINER:-dd-mcp-local}"
exec docker exec -i "$CONTAINER" dd-mcp "$@"
