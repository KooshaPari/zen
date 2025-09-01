#!/usr/bin/env bash
set -euo pipefail

API_BASE="${API_BASE:-http://localhost:8080}"
AUTH_HEADER=()
if [[ -n "${TOKEN:-}" ]]; then AUTH_HEADER=("-H" "Authorization: Bearer $TOKEN"); fi

echo "[API] Health check (/healthz)" >&2
curl -sS "$API_BASE/healthz" | jq . || true

echo "[MCP] List tools" >&2
curl -sS -X POST "$API_BASE/mcp" \
  -H 'Content-Type: application/json' \
  "${AUTH_HEADER[@]}" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | jq . || true

echo "[MCP] Call version tool" >&2
curl -sS -X POST "$API_BASE/mcp" \
  -H 'Content-Type: application/json' \
  "${AUTH_HEADER[@]}" \
  -d '{
    "jsonrpc":"2.0",
    "id":2,
    "method":"tools/call",
    "params":{"name":"version","arguments":{}}
  }' | jq . || true

echo "[MCP] Done" >&2
