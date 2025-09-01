#!/usr/bin/env bash
set -euo pipefail

API_BASE="${API_BASE:-http://localhost:8080}"

echo "[API] Health check" >&2
curl -sS "$API_BASE/health" | jq . || true

echo "[API] Create single LLM task" >&2
CREATE=$(curl -sS -X POST "$API_BASE/tasks" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "anthropic/claude-3.5-haiku",
    "message": "Return the word: hello",
    "task_description": "hello-smoke"
  }')
echo "$CREATE" | jq . || true

TASK_ID=$(echo "$CREATE" | jq -r .task_id 2>/dev/null || true)
if [[ -n "${TASK_ID:-}" && "$TASK_ID" != "null" ]]; then
  echo "[API] Poll task results for $TASK_ID" >&2
  curl -sS "$API_BASE/tasks/$TASK_ID/results" | jq . || true
fi

echo "[API] Batch (parallel) with two items" >&2
curl -sS -X POST "$API_BASE/llm/batch" \
  -H 'Content-Type: application/json' \
  -d '{
    "batch_mode": "parallel",
    "batch_items": [
      {"model": "anthropic/claude-3.5-haiku", "message": "Batch A"},
      {"model": "anthropic/claude-3.5-haiku", "message": "Batch B"}
    ]
  }' | jq . || true

echo "[API] Done" >&2
