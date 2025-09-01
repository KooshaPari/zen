**QA Workflows**
- **Goal**: Provide consistent smoke/regression checks across CLI, API, Web, MCP (stdio/http), Desktop, and Mobile surfaces.
- **Style**: Curl/bash-first where possible; stdio via lightweight proc scripts; browser via Playwright guidance.

**CLI**
- Command: `./zen-mcp-server` runs the MCP stdio server.
- Smoke: use `scripts/qa/mcp_stdio_ping.py` to initialize, list tools, and exit.
- Notes: set `LOG_LEVEL=INFO` to reduce noise; ensure `.env` or API keys present for model-backed tools.

**API**
- Server: `python -m server_http --host 0.0.0.0 --port 8080`.
- Health: `curl -s http://localhost:8080/health`.
- LLM task (single):
  - `curl -s -X POST http://localhost:8080/tasks -H 'Content-Type: application/json' -d '{"model":"anthropic/claude-3.5-haiku","message":"Say hello","task_description":"hello"}'`.
- LLM batch (parallel):
  - `curl -s -X POST http://localhost:8080/llm/batch -H 'Content-Type: application/json' -d '{"batch_mode":"parallel","batch_items":[{"model":"anthropic/claude-3.5-haiku","message":"Task A"},{"model":"anthropic/claude-3.5-haiku","message":"Task B"}]}'`.
- SSE (live events): `curl -N http://localhost:8080/tasks/<TASK_ID>/events`.
- See `scripts/qa/curl_api_smoke.sh` for a runnable sequence.

**Web (Playwright)**
- Purpose: Frontend smoke against the HTTP surface (basic health + JSON payload assertions).
- Suggested setup:
  - `npm init -y && npm i -D @playwright/test`
  - `npx playwright install --with-deps`
  - Example spec name: `tests/web/health.spec.ts`:
    - GET `/health` returns 200 and JSON body with status.
    - Optionally, drive a page against a local UI if applicable.
- Run: `npx playwright test`.
- CI: add a matrix job gating on `server_http` up.

**MCP (HTTP)**
- Server: `python -m server_mcp_http` (FastAPI) — see `/docs` and `/redoc` when running.
- Client: `python clients/mcp_http_client.py` or `ts-node clients/mcp_http_client.ts`.
- Smoke:
  - List tools: run the client and confirm non-empty tool list.
  - Get a prompt: `prompt <name>` in client REPL.
  - Call a trivial tool (e.g., demo `multiply` if exposed by your server build).

**MCP (stdio)**
- Server: `./zen-mcp-server` (wraps `server.py`).
- Smoke: `python scripts/qa/mcp_stdio_ping.py` performs `initialize` and `tools/list` using JSON-RPC over stdio framing.
- Extend: add a `tools/call` for a read-only tool to validate roundtrips.

**Desktop (via Desktop Automation MCP)**
- Intent: Validate the Desktop MCP is reachable and enumerates tools. Attach as an external MCP alongside Zen.
- Guidance:
  - Start your Desktop MCP process (outside this repo).
  - From the Zen client/orchestrator, ensure the Desktop MCP is registered.
  - Smoke via MCP HTTP (or stdio) list and call a no-op/read-only tool (e.g., selector dump) to confirm.
- Notes: Keep automations read-only in smoke (focus on discovery and a safe query).

**Mobile (via mobile-next MCP)**
- Intent: Validate mobile MCP attach is correct and tool usage guidance is followed.
- Guidance:
  - Start mobile-next MCP with the proper transport and auth.
  - Attach to the same orchestration environment as Desktop/Zen.
  - Smoke: list tools; call a safe info tool (e.g., device info) to confirm connectivity.
- Usage notes to inform agents:
  - Tools often require explicit permissions; surface any denial clearly.
  - Prefer idempotent calls in smoke; fence destructive actions behind flags.

**Test Matrix (what to assert)**
- CLI/stdio: `initialize` ok, non-empty `tools/list`.
- API: `/health` ok, single task ok, batch parallel ok, SSE streams events.
- Web: `/health` spec green.
- MCP HTTP: client can list prompts and get one; at least one tool call ok.
- Desktop/Mobile MCP: list tools ok; one safe tool call ok.

**Agent Guidance: Parallelization**
- Side note to agents (embedded in system prompts): When decomposing independent sub-tasks, prefer a single `agent_batch` call with `coordination_strategy: 'parallel'` (2–10 tasks) over serial tool calls.

**How to run locally**
- Start stdio MCP: `./zen-mcp-server`.
- Start HTTP API: `python -m server_http --host 0.0.0.0 --port 8080`.
- Start MCP HTTP: `python -m server_mcp_http`.
- Run curls: `bash scripts/qa/curl_api_smoke.sh`.
- Stdio smoke: `python scripts/qa/mcp_stdio_ping.py`.
