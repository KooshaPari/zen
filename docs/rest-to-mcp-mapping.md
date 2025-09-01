# REST → MCP JSON‑RPC Migration Cheatsheet

This guide maps common legacy REST endpoints to the new Streamable HTTP MCP interface.

Health
- Legacy: `GET /health`
- MCP HTTP: `GET /healthz`

List Tools
- Legacy: n/a
- MCP HTTP: `POST /mcp` with body `{ "jsonrpc": "2.0", "id": 1, "method": "tools/list" }`

Run an LLM task (single)
- Legacy: `POST /tasks` (agent_type=llm, model, message)
- MCP HTTP: `POST /mcp` `tools/call` using either:
  - `name: "chat"` with `{ "prompt": "...", "model": "auto" }`, or
  - `name: "deploy"` with `{ "prompt": "...", "agent_type": "llm", "model": "auto", "execution_mode": "sync" }`

Run a workflow task (e.g., analyze)
- Legacy: `POST /tasks` (workflow=analyze, message, workflow_params)
- MCP HTTP: `POST /mcp` `tools/call` with `name: "analyze"` and arguments like `{ "prompt": "...", "model": "auto", "focus": "performance" }`

Batch LLM tasks
- Legacy: `POST /llm/batch` with `batch_mode`, `batch_items` array
- MCP HTTP: `POST /mcp` `tools/call` with `name: "deploy"` and arguments:
  `{ "batch_items": [ {"model":"auto","prompt":"A"}, {"model":"auto","prompt":"B"} ], "batch_mode": "parallel" }`

Streaming updates (SSE)
- Legacy per-task: `GET /tasks/{id}/events` or `/tasks/{id}/stream`
- MCP HTTP:
  - Per-task SSE: `GET /stream/{task_id}`
  - Global SSE: `GET /events/live`

Cancel a task
- Legacy: `POST /tasks/{id}/cancel`
- MCP HTTP: Not currently exposed as a generic RPC; design jobs to be idempotent and short‑lived, or rely on streaming completion and restart patterns.

Task results
- Legacy: `GET /tasks/{id}/results`
- MCP HTTP: `tools/call` returns a result immediately for sync mode; for streaming, follow SSE until completion messages, then the initial `tools/call` response includes the content or continuation as appropriate.

Router decision
- Legacy: `POST /router/decide`
- MCP HTTP: Use `model: "auto"` in tool arguments; the model router chooses and enforces limits internally.

Messages API (channels/DM/inbox)
- Legacy REST endpoints like `/messages/channel`, `/messages/dm`, `/inbox/messages` existed in the HTTP server; they can still be exposed by the MCP HTTP server as HTTP endpoints when enabled.
- MCP focus is on tool execution; for messaging orchestration, use SSE `/events/live` and the project graph facilities if those routes are enabled.

Artifacts/Projects
- Legacy endpoints (`/projects/*`, `/artifacts/*`) were part of the minimal REST server; use tools or custom adapters over MCP if needed.

Notes
- Prefer `name: "deploy"` for a single, unified surface that supports LLM, workflows, batch, streaming, and continuation.
- `model: "auto"` lets the router pick an allowed/optimal model per tool category.
- For multi‑turn, use continuation metadata the server includes or maintain application‑level session context.

