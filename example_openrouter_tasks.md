# Streamable MCP HTTP Examples

Use `POST /mcp` for JSON‑RPC calls. Common operations:
- `tools/list` — enumerate available tools
- `tools/call` — execute a tool by name with arguments

## 1) List Tools
```bash
curl -sS -X POST http://localhost:8080/mcp \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | jq .
```

## 2) Call a Workflow Tool (analyze)
```bash
curl -sS -X POST http://localhost:8080/mcp \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc":"2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
      "name": "analyze",
      "arguments": {
        "prompt": "def slow_function():\n    time.sleep(10)\n    return \"done\"",
        "model": "auto",
        "focus": "performance"
      }
    }
  }' | jq .
```

## 3) Unified Execution (Deploy Tool)
The `deploy` tool unifies LLM, workflows, batch, and streaming.
```bash
curl -sS -X POST http://localhost:8080/mcp \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "id": 3,
    "method": "tools/call",
    "params": {
      "name": "deploy",
      "arguments": {
        "prompt": "Summarize the design of utils/model_router.py",
        "agent_type": "llm",
        "execution_mode": "sync",
        "model": "auto",
        "thinking_mode": "medium"
      }
    }
  }' | jq .
```

## Streaming
- Per-task stream: `GET /stream/{task_id}`
- Global stream: `GET /events/live`

The server includes `task_id` in streaming responses when applicable.

## Provider Configuration (env)
```bash
export GEMINI_API_KEY="your-gemini-key"
export OPENAI_API_KEY="your-openai-key"
export XAI_API_KEY="your-grok-key"
export OPENROUTER_API_KEY="your-openrouter-key"
export CUSTOM_API_URL="http://localhost:11434"  # e.g., Ollama
```
