# Agent-to-Agent Communications

This guide summarizes the subjects, HTTP endpoints, and helper APIs for agent messaging with blocking and non-blocking flows.

## Subjects
- Direct agent channels
  - `a2a.agent.<agent_id>.in` — inbound to agent
  - `a2a.agent.<agent_id>.out` — responses/notifications from agent
- Task channels
  - `a2a.task.<task_id>.rpc` — request/response RPC
  - `a2a.task.<task_id>.events` — task lifecycle/progress updates
- Broadcast
  - `a2a.events` — global intent-style events
- Discovery
  - `discovery.advertise` — advertisements to registries

## HTTP Endpoints
- `POST /a2a/message` — forward a structured A2A message; returns response if available
- `POST /a2a/advertise` — register an AgentCard
- `POST /a2a/discover` — discover agents by capability/org
- Dev RPC helpers (always enabled)
  - `POST /a2a/test/rpc-responder/start` {"task_id"}
  - `POST /a2a/test/rpc-responder/stop` {"task_id"}
  - `POST /a2a/test/rpc` {"task_id","method","params","timeout"}

## Blocking vs Non-Blocking
- Blocking RPC: `rpc_task(task_id, method, params, timeout_seconds)`
  - Publishes an RPC request to `a2a.task.<task_id>.rpc` with a correlation id; awaits a response or times out
- Blocking Chat: `chat_send(to_agent, text, metadata, timeout_seconds)`
  - Publishes a `CHAT_REQUEST` to `a2a.agent.<to>.in`; awaits a `CHAT_RESPONSE`
- Non-Blocking: `broadcast_event(intent, payload)` and one-way direct messaging via `a2a.agent.<id>.in/out`

## Code Examples

Python (blocking chat)

```python
from utils.a2a_protocol import get_a2a_manager

mgr = get_a2a_manager()
await mgr._ensure_nats()
res = await mgr.chat_send("agent-beta", "hello", {"topic":"greeting"}, timeout_seconds=5)
print(res)  # {"ok": True, "echo": "hello", "metadata": {"topic":"greeting"}}
```

Python (RPC)

```python
ack = await mgr.rpc_task("T123", "ping", {"x":1}, timeout_seconds=5)
```

Curl (start responder and invoke RPC)

```bash
curl -sS -X POST http://127.0.0.1:8080/a2a/test/rpc-responder/start -H 'Content-Type: application/json' \
  --data '{"task_id":"T123"}'

curl -sS -X POST http://127.0.0.1:8080/a2a/test/rpc -H 'Content-Type: application/json' \
  --data '{"task_id":"T123","method":"ping","params":{"x":1},"timeout":5}'
```

## Observability
- `/metrics` includes publish counters:
  - `publish_js_attempts`, `publish_js_fallbacks`, `publish_core_success`
- `/readyz` adds:
  - `js_streams_ok` boolean
  - `js_warnings` list (e.g., `js_publish_fallbacks_detected`)

## Notes
- JetStream is preferred; communicator automatically falls back to core NATS on publish errors.
- Streams are sized for single-node usage; increase limits if you need longer retention.

