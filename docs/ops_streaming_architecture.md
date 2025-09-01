# Zen MCP Server: Streamable HTTP + Redis + NATS Ops Guide
Status: Implemented — features are env‑gated and default to in‑memory when disabled.

This document describes the operational model for the streamable HTTP server with Redis persistence and NATS JetStream events. All features are env-gated and default to in-memory/in-proc behavior.

## Components
- HTTP API (aiohttp): streamable endpoints (SSE/WebSocket)
- Redis: source of truth for tasks/batches, locks, indexes, message streams, TTL retention
- NATS JetStream: lifecycle events, fan-out subscribers, optional A2A

## Environment flags
- Core
  - ZEN_RUN_MODE=local|server
  - SERVER_HOST, SERVER_PORT
  - MAX_CONCURRENT_TASKS, TASK_QUEUE_MAX
  - AGENT_TASK_RETENTION_SEC (default 3600)
- Storage
  - ZEN_STORAGE=memory|redis
  - REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_TLS=0|1
- Events
  - ZEN_EVENT_BUS=inline|nats
  - NATS_URL, NATS_JETSTREAM=1, NATS_USER, NATS_PASS, NATS_TLS=0|1
- Streams
  - TASK_MESSAGES_MAXLEN (default 1000)

## Redis keys and streams
- task:<id> (JSON)
  - {schema_version, agent, status, timestamps, metrics?, last_error?, final?}
  - TTL = AGENT_TASK_RETENTION_SEC
- task:<id>:messages (Redis Stream)
  - fields: ts (ms), event, data (JSON)
  - XTRIM MAXLEN ~ TASK_MESSAGES_MAXLEN
- inbox:status:<status> (ZSET by updated_at)
- tasks:by_created_at (ZSET)
- batch:<id> (JSON {schema_version, description, created_at})
- batch:<id>:tasks (Set of task_ids)

## NATS subjects (lifecycle)
- tasks.created
- tasks.updated
- tasks.completed
- tasks.failed
- tasks.timeout
- tasks.cancelled

Payload (compact JSON): {event, task_id, status?, timestamps?, deltas?{metrics,action}}

## HTTP API
- POST /tasks → {task_id} (202)
- GET /tasks/{id} → snapshot
- GET /tasks/{id}/results → final report or 202
- GET /tasks/{id}/stream → SSE snapshots via StreamingManager
- GET /tasks/{id}/events → SSE real-time (StreamingManager)
- GET /tasks/{id}/messages/stream → SSE replay+live from Redis Streams (fallback to /events)
- POST /tasks/{id}/cancel → 202
- POST /batches → {batch_id, task_ids}
- GET /batches/{id}/summary → aggregate
- GET /metrics → JSON metrics
- GET /health, /readyz → probes

## Backpressure and concurrency
- MAX_CONCURRENT_TASKS via semaphore; enqueue up to TASK_QUEUE_MAX; else 429 Retry-After
- Idempotency via Idempotency-Key header; idem:<hash> → task_id (TTL)

## Operations
- TTL sweep: task keys adhere to TTL; Streams trimmed by MAXLEN
- Health: /health checks process; /readyz baseline; can be extended to check Redis/NATS
- Metrics: /metrics includes queue depth, connection counts, storage/event modes

## Example NATS consumer
Install nats-py and run the durable consumer:

```bash
pip install nats-py
python examples/nats_durable_consumer.py --subject tasks.completed --durable zen_1
```

## Notes
- All features are best-effort; when disabled by env or missing brokers, the system falls back to in-proc/in-memory behavior without breaking the server.
- Schema versioning is applied to all persisted JSON blobs.
