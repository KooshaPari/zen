# Messaging API — Quick curl guide (MVP)

Deprecated (Legacy REST): These examples use the legacy `server_http.py` endpoints. The active server is Streamable HTTP MCP via `/mcp`. For current usage, integrate messaging through tools or consume global SSE at `/events/live`.

Prereqs
- Server running: http://127.0.0.1:8080
- Optional: HTTP_API_KEY set; if so, add header `Authorization: Bearer $HTTP_API_KEY`
- Optional: Redis enabled for persistence (`ZEN_STORAGE=redis`), NATS for events (`ZEN_EVENT_BUS=nats`)

## Channels

Create a channel
```bash
curl -sS POST http://127.0.0.1:8080/channels \
  -H 'Content-Type: application/json' \
  -d '{"project_id":"proj1","name":"dev","visibility":"project","created_by":"agentA"}' | jq
```

List channels for a project
```bash
curl -sS 'http://127.0.0.1:8080/channels?project_id=proj1&limit=50' | jq
```

Post channel message (mentions create unread for recipients)
```bash
curl -sS POST http://127.0.0.1:8080/messages/channel \
  -H 'Content-Type: application/json' \
  -d '{"channel_id":"ch:proj1:dev","from":"agentA","body":"hello team","mentions":["agentB"]}' | jq
```

Get channel history
```bash
curl -sS http://127.0.0.1:8080/messages/channel/ch:proj1:dev/history | jq
```

## Direct messages

Send a DM
```bash
curl -sS POST http://127.0.0.1:8080/messages/dm \
  -H 'Content-Type: application/json' \
  -d '{"a":"agentA","b":"agentB","from":"agentA","body":"ping"}' | jq
```

Get DM history
```bash
curl -sS http://127.0.0.1:8080/messages/dm/agentA/agentB/history | jq
```

## Unread + mark-read

Get unread summary + items
```bash
curl -sS 'http://127.0.0.1:8080/inbox/messages?agent_id=agentB&limit=25' | jq
```

Mark a message as read
```bash
curl -sS POST http://127.0.0.1:8080/inbox/mark_read \
  -H 'Content-Type: application/json' \
  -d '{"agent_id":"agentB","message_id":"<copy-from-unread-items-id>"}' | jq
```

## Threads

Reply to a thread
```bash
curl -sS POST http://127.0.0.1:8080/threads/reply \
  -H 'Content-Type: application/json' \
  -d '{"channel_id":"ch:proj1:dev","root_message_id":"<message-id>","from":"agentB","body":"on it"}' | jq
```

Get thread history
```bash
curl -sS http://127.0.0.1:8080/threads/<root_message_id>/history | jq
```

## SSE stream

Open an SSE stream for messaging events (messaging_posted, messaging_read, channels.created)
```bash
curl -N http://127.0.0.1:8080/messages/stream
```

## Prompt injection (optional)

Enable prompt injection of top unread into LLM tasks
```bash
export ZEN_MSG_INJECTION=1
curl -sS POST http://127.0.0.1:8080/tasks \
  -H 'Content-Type: application/json' \
  -d '{"type":"llm","agent_id":"agentB","message":"Summarize the new messages"}' | jq
```

Notes
- Feature flags: ZEN_STORAGE=redis, ZEN_EVENT_BUS=nats, ZEN_MSG_INJECTION=1
- Redis keys under inbox:agent:<id>:unread (ZSET), messages:* namespaces
- In-memory fallback works without Redis but does not persist across runs
